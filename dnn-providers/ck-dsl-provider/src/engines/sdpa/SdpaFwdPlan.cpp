// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaFwdPlan.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../runtime/KernelArtifact.hpp"
#include "../../runtime/LaunchAbi.hpp"

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
    oss << "SdpaFwdPlan::execute: no device buffer for " << role << " (uid=" << uid
        << "); searched " << numDeviceBuffers << " entries";
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, oss.str());
}

}  // namespace

SdpaFwdPlan::SdpaFwdPlan(std::shared_ptr<HipModule> module, std::int64_t qUid, std::int64_t kUid,
                         std::int64_t vUid, std::int64_t oUid, float scaleLog2,
                         std::int32_t seqlenQ, std::int32_t seqlenK, std::int32_t strideQToken,
                         std::int32_t strideQHead, std::int32_t strideKToken,
                         std::int32_t strideKHead, std::int32_t strideVToken,
                         std::int32_t strideVHead, std::int32_t strideOToken,
                         std::int32_t strideOHead, bool hasStats, std::int64_t statsUid)
    : _module(std::move(module)),
      _qUid(qUid),
      _kUid(kUid),
      _vUid(vUid),
      _oUid(oUid),
      _scaleLog2(scaleLog2),
      _seqlenQ(seqlenQ),
      _seqlenK(seqlenK),
      _strideQToken(strideQToken),
      _strideQHead(strideQHead),
      _strideKToken(strideKToken),
      _strideKHead(strideKHead),
      _strideVToken(strideVToken),
      _strideVHead(strideVHead),
      _strideOToken(strideOToken),
      _strideOHead(strideOHead),
      _hasStats(hasStats),
      _statsUid(statsUid) {
    if (_module == nullptr) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SdpaFwdPlan: refusing to construct with null HipModule");
    }

    // Validate the kernel's argument schema matches the ABI execute()
    // packs against: the 15-slot base layout (Pointer x4, F32, I32 x10),
    // plus -- when opt-in stats are enabled -- a 16th Pointer slot
    // (LSE_out). If a future kernel variant grows or rearranges its
    // parameter list, the plan must refuse to launch rather than pack
    // values into wrong-typed slots. The schema arrives via KernelArtifact
    // at JIT time, so this is fixed for the lifetime of each cached
    // HipModule. The stats flag and the schema width must agree: a 16-slot
    // schema iff stats are enabled (otherwise the cache key and the loaded
    // module have drifted).
    const auto& schema = _module->argSchema();
    auto rejectSlot = [&](const std::string& detail) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlan: kernel argSchema does not match the expected "
            "(Pointer x4, F32, I32 x10[, Pointer]) layout for the FMHA-forward kernel signature; " +
                detail);
    };
    const std::size_t expectedSlots = _hasStats ? 16u : 15u;
    if (schema.size() != expectedSlots) {
        std::ostringstream oss;
        oss << "got " << schema.size() << " slots, expected " << expectedSlots << " (stats "
            << (_hasStats ? "enabled" : "disabled") << ")";
        rejectSlot(oss.str());
    }
    for (std::size_t i = 0; i < 4; ++i) {
        if (schema[i].kind != ArgSchema::Kind::Pointer) {
            std::ostringstream oss;
            oss << "slot " << i << " must be Pointer";
            rejectSlot(oss.str());
        }
    }
    if (schema[4].kind != ArgSchema::Kind::F32) {
        rejectSlot("slot 4 must be F32 (scale_log2)");
    }
    for (std::size_t i = 5; i < 15; ++i) {
        if (schema[i].kind != ArgSchema::Kind::I32) {
            std::ostringstream oss;
            oss << "slot " << i << " must be I32";
            rejectSlot(oss.str());
        }
    }
    if (_hasStats && schema[15].kind != ArgSchema::Kind::Pointer) {
        rejectSlot("slot 15 must be Pointer (LSE_out)");
    }

    // One-shot launch-shape log: useful operator diagnostic at plan
    // construction without polluting the per-call hot path.
    HIPDNN_PLUGIN_LOG_INFO("SdpaFwdPlan: built plan for kernel '"
                           << _module->kernelName() << "' grid=(" << _module->grid().x << ","
                           << _module->grid().y << "," << _module->grid().z << ") block=("
                           << _module->block().x << "," << _module->block().y << ","
                           << _module->block().z << ") seqlen_q=" << _seqlenQ
                           << " seqlen_k=" << _seqlenK);
}

std::size_t SdpaFwdPlan::getWorkspaceSize(const ::CkDslHandle& /*handle*/) const {
    // The FMHA-forward kernel allocates its scratch in static LDS; no
    // external workspace is needed. If a future variant needs an
    // external buffer (e.g. a global-memory scratchpad for split-K
    // reductions) it surfaces here.
    return 0;
}

void SdpaFwdPlan::execute(const ::CkDslHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          std::uint32_t numDeviceBuffers, void* /*workspace*/) const {
    if (deviceBuffers == nullptr && numDeviceBuffers > 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaFwdPlan::execute: deviceBuffers is null but numDeviceBuffers > 0");
    }

    const auto& qBuf = findDeviceBuffer(_qUid, deviceBuffers, numDeviceBuffers, "Q");
    const auto& kBuf = findDeviceBuffer(_kUid, deviceBuffers, numDeviceBuffers, "K");
    const auto& vBuf = findDeviceBuffer(_vUid, deviceBuffers, numDeviceBuffers, "V");
    const auto& oBuf = findDeviceBuffer(_oUid, deviceBuffers, numDeviceBuffers, "O");

    // ABI order: Q, K, V, O, scale_log2, seqlen_q, seqlen_k, then the
    // eight token/head strides for Q/K/V/O. Matches the kernel signature
    // and the Python _sdpa_fwd_arg_schema order exactly.
    std::vector<ArgValue> values = {
        ArgValue::pointer(qBuf.ptr),  ArgValue::pointer(kBuf.ptr),  ArgValue::pointer(vBuf.ptr),
        ArgValue::pointer(oBuf.ptr),  ArgValue::f32(_scaleLog2),    ArgValue::i32(_seqlenQ),
        ArgValue::i32(_seqlenK),      ArgValue::i32(_strideQToken), ArgValue::i32(_strideQHead),
        ArgValue::i32(_strideKToken), ArgValue::i32(_strideKHead),  ArgValue::i32(_strideVToken),
        ArgValue::i32(_strideVHead),  ArgValue::i32(_strideOToken), ArgValue::i32(_strideOHead),
    };

    // Opt-in stats (LSE) output: resolve the head-major [B, Hq, Sq] f32
    // buffer by uid and append it as the 16th arg (slot index 15), after
    // the 15 base args -- matching the schema's conditional LSE_out slot.
    if (_hasStats) {
        const auto& lseBuf =
            findDeviceBuffer(_statsUid, deviceBuffers, numDeviceBuffers, "LSE_out");
        values.push_back(ArgValue::pointer(lseBuf.ptr));
    }

    std::vector<std::byte> packed = LaunchAbi::pack(_module->argSchema(), values);

    _module->launch(packed.data(), packed.size(), _module->grid(), _module->block(),
                    _module->ldsBytes(), handle.getStream());
}

}  // namespace ck_dsl_provider
