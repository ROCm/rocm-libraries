// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaFwdPlan.hpp"

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

    // Validate the kernel's argument schema matches the unified
    // paged/varlen tiled-2D ABI: a fixed 18-slot layout
    // (Pointer x10, F32 x5, I32 x3). If a future kernel variant grows or
    // rearranges its parameter list, the plan must refuse to launch
    // rather than pack values into wrong-typed slots. The schema arrives
    // via KernelArtifact at JIT time, so this is fixed for the lifetime of
    // each cached HipModule. The unified kernel emits no LSE, so
    // ``hasStats`` must be false here.
    const auto& schema = _module->argSchema();
    auto rejectSlot = [&](const std::string& detail) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaFwdPlan: kernel argSchema does not match the expected "
            "(Pointer x10, F32 x5, I32 x3) layout for the unified paged/varlen "
            "FMHA-forward kernel signature; " +
                detail);
    };
    if (_hasStats) {
        rejectSlot("opt-in stats (LSE) are not supported on the unified paged path");
    }
    constexpr std::size_t kUnifiedSlots = 18u;
    if (schema.size() != kUnifiedSlots) {
        std::ostringstream oss;
        oss << "got " << schema.size() << " slots, expected " << kUnifiedSlots;
        rejectSlot(oss.str());
    }
    for (std::size_t i = 0; i < 10; ++i) {
        if (schema[i].kind != ArgSchema::Kind::Pointer) {
            std::ostringstream oss;
            oss << "slot " << i << " must be Pointer";
            rejectSlot(oss.str());
        }
    }
    for (std::size_t i = 10; i < 15; ++i) {
        if (schema[i].kind != ArgSchema::Kind::F32) {
            std::ostringstream oss;
            oss << "slot " << i << " must be F32";
            rejectSlot(oss.str());
        }
    }
    for (std::size_t i = 15; i < 18; ++i) {
        if (schema[i].kind != ArgSchema::Kind::I32) {
            std::ostringstream oss;
            oss << "slot " << i << " must be I32";
            rejectSlot(oss.str());
        }
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

void SdpaFwdPlan::execute(const ::CkDslHandle& /*handle*/,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          std::uint32_t numDeviceBuffers, void* /*workspace*/) const {
    if (deviceBuffers == nullptr && numDeviceBuffers > 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaFwdPlan::execute: deviceBuffers is null but numDeviceBuffers > 0");
    }

    // Resolve the Q/K/V/O buffers so a graph-vs-buffer UID mismatch still
    // surfaces here with concrete context (the same validation the dense
    // path performed). The resolved pointers are not yet bound: the
    // unified 18-arg launch is deferred to Phase 4.
    static_cast<void>(findDeviceBuffer(_qUid, deviceBuffers, numDeviceBuffers, "Q"));
    static_cast<void>(findDeviceBuffer(_kUid, deviceBuffers, numDeviceBuffers, "K"));
    static_cast<void>(findDeviceBuffer(_vUid, deviceBuffers, numDeviceBuffers, "V"));
    static_cast<void>(findDeviceBuffer(_oUid, deviceBuffers, numDeviceBuffers, "O"));

    // PHASE-4 TODO (gfx950 GPU launch): bind the unified 18-slot ABI and
    // launch. The remaining work is:
    //   1. Compute the host marshalling arrays via ``SdpaMarshalling``
    //      (block_tables / cu_seqlens_q / seqused_k + block_table_stride)
    //      from the problem shape carried on the plan.
    //   2. Upload those three i32 arrays to device memory (RAII-owned for
    //      the launch's lifetime) and resolve the sink / alibi / qq_bias
    //      pointers (0 when the feature is off).
    //   3. Pack the 18 args in schema order -- pointers [output, query,
    //      key_cache, value_cache, sink, block_tables, seq_lens,
    //      alibi_slopes, qq_bias, query_start_len], f32 [scale, k_scale,
    //      v_scale, out_scale, softcap], i32 [num_seqs,
    //      block_table_stride, qq_bias_stride_0] -- and launch via
    //      ``HipModule::launch`` on the handle's stream.
    // The host marshalling LOGIC (step 1) is implemented + unit-tested in
    // Phase 2; the device binding (steps 2-3) is Phase 4. The GPU-gated
    // SDPA tests skip on non-gfx950 hosts, so host CI never reaches here.
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "SdpaFwdPlan::execute: the unified paged/varlen 18-arg launch is not yet "
        "wired (Phase 4, gfx950). Host marshalling logic is implemented and "
        "unit-tested; device buffer binding + kernel launch remain.");
}

}  // namespace ck_dsl_provider
