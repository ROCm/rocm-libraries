// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlan.hpp"
#include "plans/SdpaFwdArgsBuilder.hpp"
#include "plans/SdpaFwdLaunchParams.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <unordered_map>

namespace asm_sdpa_engine
{

SdpaFwdPlan::SdpaFwdPlan(CachedModule kernel, SdpaFwdParams params)
    : _kernel(std::move(kernel))
    , _params(std::move(params))
{
}

size_t SdpaFwdPlan::getWorkspaceSize(const Handle& /*handle*/) const
{
    // Forward-only kernel requires no workspace (uses 64KB LDS internally)
    return 0;
}

void SdpaFwdPlan::execute(const Handle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* /*workspace*/) const
{
    // Build UID->ptr map from device buffers
    std::unordered_map<int64_t, void*> uidToPtrMap;
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    // Build the packed kernel-argument struct (pure; no device calls).
    // Non-const: launchKernel takes a void* to the argument buffer.
    fmha_fwd_v3_args args = buildFwdKernelArgs(_params, uidToPtrMap);

    // Attention scale — resolved at execute for runtime pass-by-value support.
    args.scalar
        = static_cast<float>(hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
            _params.attnScale, deviceBuffers, numDeviceBuffers)));
    const auto launchParams = computeFwdLaunchParams(_params);

    if(!launchKernel("fwd",
                     _kernel->function(),
                     &args,
                     sizeof(args),
                     launchParams.gridDimX,
                     launchParams.gridDimY,
                     launchParams.gridDimZ,
                     launchParams.blockDimX,
                     handle.getStream()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlan::execute: hipModuleLaunchKernel failed for SDPA forward");
    }
}

} // namespace asm_sdpa_engine
