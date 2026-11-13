// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaFwdPlan.hpp"
#include "device/LaunchDeviceOrdinal.hpp"
#include "device/ScopedDevice.hpp"
#include "plans/SdpaFwdArgsBuilder.hpp"
#include "plans/SdpaFwdLaunchParams.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <unordered_map>

namespace asm_sdpa_engine
{

SdpaFwdPlan::SdpaFwdPlan(std::unique_ptr<SdpaFwdKernels> kernels, SdpaFwdParams params)
    : _kernels(std::move(kernels))
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
    // Throws BAD_PARAM before the launch for a scale the kernel mishandles.
    args.scalar = plan_utils::resolveSupportedAttnScale(
        _params.attnScale, deviceBuffers, numDeviceBuffers, "SdpaFwdPlan::execute");
    const auto launchParams = computeFwdLaunchParams(_params);

    // Resolved from the handle this call was given, not the one the plan was built under:
    // a plan may be executed under any handle, and its module must be the one loaded on
    // the device the launch runs on.
    const int deviceOrdinal = hip_kernel_provider::device::launchDeviceOrdinal(handle.getStream());
    const CachedModule kernel = _kernels->forOrdinal(deviceOrdinal);

    // A module has to be launched from the device it was loaded on.
    const hip_kernel_provider::device::ScopedDevice binding(deviceOrdinal);
    if(!binding.bound())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "SdpaFwdPlan::execute: cannot make device "
                                                           + std::to_string(deviceOrdinal)
                                                           + " current to launch SDPA forward");
    }

    if(!launchKernel("fwd",
                     kernel->function(),
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
