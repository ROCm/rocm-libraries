// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmPlan.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <utility>

namespace ck_dsl_provider {

ConvImplicitGemmPlan::ConvImplicitGemmPlan(std::shared_ptr<HipModule> module, std::int64_t xUid,
                                           std::int64_t wUid, std::int64_t yUid)
    : _module(std::move(module)), _xUid(xUid), _wUid(wUid), _yUid(yUid) {
    if (_module == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "ConvImplicitGemmPlan: refusing to construct with null HipModule");
    }
}

std::size_t ConvImplicitGemmPlan::getWorkspaceSize(const ::CkDslHandle& /*handle*/) const {
    // The implicit-GEMM kernel allocates its LDS via smem_alloc with
    // statically-sized regions; no external workspace is needed for
    // M1. If a future variant adds a workspace requirement (e.g. for
    // the cshuffle epilogue's larger LDS budget) it surfaces here.
    return 0;
}

void ConvImplicitGemmPlan::execute(const ::CkDslHandle& /*handle*/,
                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                   std::uint32_t numDeviceBuffers, void* /*workspace*/) const {
    // Defensive: surface a clear message if the runtime hands us a
    // bad buffer array. Distinguishes a real misuse from the "stub"
    // path below.
    if (deviceBuffers == nullptr && numDeviceBuffers > 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvImplicitGemmPlan::execute: deviceBuffers is null but numDeviceBuffers > 0");
    }
    // (void)-discard to keep the unused-parameter warning quiet
    // without dropping the validation above. Once I-8 wires the
    // launch path these arguments are read for real.
    (void)numDeviceBuffers;

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED,
        "ConvImplicitGemmPlan::execute is a stub (M1 step I-7). The launch path -- "
        "uid -> device-buffer lookup, LaunchAbi packing against argSchema, "
        "HipModule::launch with grid/block from the artifact -- lands in step I-8.");
}

}  // namespace ck_dsl_provider
