// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmPlan.hpp"

#include <array>
#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <string>
#include <utility>

#include "../../runtime/KernelArtifact.hpp"

namespace ck_dsl_provider {

namespace {

/// Linear scan over the device-buffer array. The array is typically
/// <10 entries so an O(n) lookup is the right shape. Throws with the
/// missing uid in the message so a graph-vs-buffer mismatch surfaces
/// with concrete context.
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

    // Validate the kernel's argument schema is exactly the shape we
    // pre-pack against. If a future kernel variant grows or rearranges
    // its parameter list, the plan must refuse to launch rather than
    // patch pointers into wrong-sized slots. The schema arrives via
    // KernelArtifact at JIT time, so this is fixed for the lifetime of
    // each cached HipModule.
    const auto& schema = _module->argSchema();
    if (schema.size() != 6 || schema[0].kind != ArgSchema::Kind::Pointer ||
        schema[1].kind != ArgSchema::Kind::Pointer || schema[2].kind != ArgSchema::Kind::Pointer ||
        schema[3].kind != ArgSchema::Kind::I32 || schema[4].kind != ArgSchema::Kind::I32 ||
        schema[5].kind != ArgSchema::Kind::I32) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "ConvImplicitGemmPlan: kernel argSchema does not match the expected "
            "(Pointer, Pointer, Pointer, I32, I32, I32) layout for the conv kernel signature");
    }

    // Pre-write the three i32 byte counts at their natural-alignment
    // offsets. Pointer slots stay zero; execute() patches them in.
    std::memcpy(_argTemplate.data() + kXBytesOffset, &_xBytes, sizeof(_xBytes));
    std::memcpy(_argTemplate.data() + kWBytesOffset, &_wBytes, sizeof(_wBytes));
    std::memcpy(_argTemplate.data() + kYBytesOffset, &_yBytes, sizeof(_yBytes));

    // One-shot launch-shape log: useful operator diagnostic at plan
    // construction without polluting the per-call hot path. Mirrors the
    // grid/block info the per-launch log used to emit.
    HIPDNN_PLUGIN_LOG_INFO("ConvImplicitGemmPlan: built plan for kernel '"
                           << _module->kernelName() << "' grid=(" << _module->grid().x << ","
                           << _module->grid().y << "," << _module->grid().z << ") block=("
                           << _module->block().x << "," << _module->block().y << ","
                           << _module->block().z << ") xBytes=" << _xBytes << " wBytes=" << _wBytes
                           << " yBytes=" << _yBytes);
}

std::size_t ConvImplicitGemmPlan::getWorkspaceSize(const ::CkDslHandle& /*handle*/) const {
    // The implicit-GEMM kernel allocates its scratch in static LDS;
    // no external workspace is needed. If a future variant needs an
    // external buffer (e.g. a global-memory scratchpad for multi-block
    // reductions) it surfaces here.
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

    // Stack-resident scratch copy of the template; no heap allocation
    // on this path. Per-call work is three memcpys (pointers in), one
    // template-copy assignment, and one HIP launch call.
    std::array<std::byte, kArgBufferSize> args = _argTemplate;
    std::memcpy(args.data() + kXPtrOffset, &xBuf.ptr, sizeof(xBuf.ptr));
    std::memcpy(args.data() + kWPtrOffset, &wBuf.ptr, sizeof(wBuf.ptr));
    std::memcpy(args.data() + kYPtrOffset, &yBuf.ptr, sizeof(yBuf.ptr));

    _module->launch(args.data(), args.size(), _module->grid(), _module->block(),
                    _module->ldsBytes(), handle.getStream());
}

}  // namespace ck_dsl_provider
