// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmPlan.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../runtime/LaunchAbi.hpp"

namespace ck_dsl_provider {

namespace {

/// Linear scan over the device-buffer array. Matches miopen-provider's
/// findDeviceBuffer pattern; the array is typically <10 entries so an
/// O(n) lookup is the right shape. Throws with the missing uid in the
/// message so a graph-vs-buffer mismatch surfaces with concrete
/// context.
const hipdnnPluginDeviceBuffer_t& findDeviceBuffer(std::int64_t uid,
                                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                   std::uint32_t numDeviceBuffers,
                                                   const char* role) {
    for (std::uint32_t i = 0; i < numDeviceBuffers; ++i) {
        if (deviceBuffers[i].uid == uid) {
            return deviceBuffers[i];
        }
    }
    std::ostringstream oss;
    oss << "ConvImplicitGemmPlan::execute: no device buffer for " << role << " (uid=" << uid
        << "); searched " << numDeviceBuffers << " entries";
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, oss.str());
}

}  // namespace

ConvImplicitGemmPlan::ConvImplicitGemmPlan(std::shared_ptr<HipModule> module, std::int64_t xUid,
                                           std::int64_t wUid, std::int64_t yUid,
                                           std::int32_t xBytes, std::int32_t wBytes,
                                           std::int32_t yBytes)
    : _module(std::move(module)),
      _xUid(xUid),
      _wUid(wUid),
      _yUid(yUid),
      _xBytes(xBytes),
      _wBytes(wBytes),
      _yBytes(yBytes) {
    if (_module == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "ConvImplicitGemmPlan: refusing to construct with null HipModule");
    }
    if (_xBytes <= 0 || _wBytes <= 0 || _yBytes <= 0) {
        std::ostringstream oss;
        oss << "ConvImplicitGemmPlan: tensor byte sizes must be positive; got xBytes=" << _xBytes
            << " wBytes=" << _wBytes << " yBytes=" << _yBytes;
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       oss.str());
    }
}

std::size_t ConvImplicitGemmPlan::getWorkspaceSize(const ::CkDslHandle& /*handle*/) const {
    // The implicit-GEMM kernel allocates its scratch in static LDS;
    // no external workspace is needed for M1. If a future variant
    // needs an external buffer (e.g. a global-memory scratchpad for
    // multi-block reductions) it surfaces here.
    return 0;
}

void ConvImplicitGemmPlan::execute(const ::CkDslHandle& handle,
                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                   std::uint32_t numDeviceBuffers, void* /*workspace*/) const {
    if (deviceBuffers == nullptr && numDeviceBuffers > 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvImplicitGemmPlan::execute: deviceBuffers is null but numDeviceBuffers > 0");
    }

    const auto& xBuf = findDeviceBuffer(_xUid, deviceBuffers, numDeviceBuffers, "X");
    const auto& wBuf = findDeviceBuffer(_wUid, deviceBuffers, numDeviceBuffers, "W");
    const auto& yBuf = findDeviceBuffer(_yUid, deviceBuffers, numDeviceBuffers, "Y");

    // Kernel signature is (A: ptr, B: ptr, D: ptr, A_bytes: i32,
    // B_bytes: i32, D_bytes: i32). The bytes are buffer-rsrc bounds
    // used by the DSL kernel for free OOB clamping (see
    // ck_dsl/instances/conv_implicit_gemm.py: a_rsrc =
    // b.buffer_rsrc(A, A_bytes)). Order matches the module's
    // argSchema exactly.
    std::vector<ArgValue> values = {
        ArgValue::pointer(xBuf.ptr), ArgValue::pointer(wBuf.ptr), ArgValue::pointer(yBuf.ptr),
        ArgValue::i32(_xBytes),      ArgValue::i32(_wBytes),      ArgValue::i32(_yBytes),
    };
    std::vector<std::byte> packed = LaunchAbi::pack(_module->argSchema(), values);

    HIPDNN_PLUGIN_LOG_INFO("ConvImplicitGemmPlan::execute launching '"
                           << _module->kernelName() << "' grid=(" << _module->grid().x << ","
                           << _module->grid().y << "," << _module->grid().z << ") block=("
                           << _module->block().x << "," << _module->block().y << ","
                           << _module->block().z << ") xBytes=" << _xBytes << " wBytes=" << _wBytes
                           << " yBytes=" << _yBytes);

    _module->launch(packed, _module->grid(), _module->block(), _module->ldsBytes(),
                    handle.getStream());
}

}  // namespace ck_dsl_provider
