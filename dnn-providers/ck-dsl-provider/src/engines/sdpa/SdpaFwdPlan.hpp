// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstddef>
#include <cstdint>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <memory>

#include "../../CkDslHandle.hpp"
#include "../../runtime/HipModule.hpp"

namespace ck_dsl_provider {

/// IPlan for one compiled FMHA-forward kernel.
///
/// The plan owns a shared reference to the ``HipModule`` produced by
/// ``JitCache``; the same module may back multiple plans built for the
/// same signature. All launch metadata (grid, block, ldsBytes,
/// argSchema) lives on the module. The plan additionally remembers the
/// four tensor UIDs (resolved to device pointers at execute time) and
/// the launch-time scalars the kernel consumes: the attention scale (in
/// log2 space), the two sequence lengths, and the eight token/head
/// strides for Q/K/V/O.
///
/// **Schema-driven packing.** The plan packs its arguments per call via
/// ``LaunchAbi::pack`` against the module's argSchema. The unified
/// paged/varlen tiled-2D kernel exposes a fixed 18-slot ABI, validated in
/// the ctor:
///   slots 0..9   -> Pointer (output, query, key_cache, value_cache,
///                   sink, block_tables, seq_lens, alibi_slopes, qq_bias,
///                   query_start_len)
///   slots 10..14 -> F32     (scale, k_scale, v_scale, out_scale, softcap)
///   slots 15..17 -> I32     (num_seqs, block_table_stride,
///                   qq_bias_stride_0)
///
/// **Opt-in stats (LSE).** The unified paged kernel emits NO LSE output
/// (the adapter gate declines any stats request), so ``hasStats`` is
/// always false on this path; the 18-slot ABI has no LSE_out slot. The
/// ctor parameter is retained for source compatibility with the plan
/// builder's call site.
///
/// **Phase-4 GPU launch (TODO).** ``execute`` cannot yet bind the three
/// host-marshalled integer arrays (block_tables / cu_seqlens_q /
/// seqused_k -- see ``SdpaMarshalling``) to device memory: that device
/// allocation + the 18-arg launch land in Phase 4 (gfx950). The
/// host-side marshalling LOGIC is implemented and unit-tested now; the
/// device binding is the remaining piece. ``execute`` therefore throws a
/// clearly-marked "not yet wired" error so a premature call fails loudly
/// rather than launching with unbound buffers. The GPU-gated SDPA tests
/// skip on non-gfx950 hosts, so this does not affect host CI.
class SdpaFwdPlan : public hipdnn_plugin_sdk::IPlan<::CkDslHandle> {
   public:
    SdpaFwdPlan(std::shared_ptr<HipModule> module, std::int64_t qUid, std::int64_t kUid,
                std::int64_t vUid, std::int64_t oUid, float scaleLog2, std::int32_t seqlenQ,
                std::int32_t seqlenK, std::int32_t strideQToken, std::int32_t strideQHead,
                std::int32_t strideKToken, std::int32_t strideKHead, std::int32_t strideVToken,
                std::int32_t strideVHead, std::int32_t strideOToken, std::int32_t strideOHead,
                bool hasStats = false, std::int64_t statsUid = -1);

    std::size_t getWorkspaceSize(const ::CkDslHandle& handle) const override;

    /// Resolve Q/K/V/O device pointers from ``deviceBuffers`` by uid (so a
    /// graph-vs-buffer mismatch surfaces here) and -- in Phase 4 -- bind +
    /// launch the unified 18-slot ABI. The host marshalling of
    /// block_tables / cu_seqlens_q / seqused_k is implemented in
    /// ``SdpaMarshalling`` and unit-tested now; the device-side upload of
    /// those arrays and the kernel launch are the remaining Phase-4 work,
    /// so this currently throws a clearly-marked "not yet wired" error
    /// rather than launching with unbound buffers.
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` if a UID is
    /// missing from ``deviceBuffers`` or (until Phase 4 lands the launch)
    /// to report the unwired unified ABI. Workspace is unused (this kernel
    /// allocates its scratch in static LDS).
    void execute(const ::CkDslHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 std::uint32_t numDeviceBuffers, void* workspace = nullptr) const override;

    /// Test-only: expose the loaded module so the plan-builder test can
    /// confirm the plan actually wraps a HipModule with the expected
    /// kernel name. Production callers use ``execute``.
    const HipModule& moduleForTesting() const {
        return *_module;
    }
    std::int64_t qUidForTesting() const {
        return _qUid;
    }
    std::int64_t oUidForTesting() const {
        return _oUid;
    }
    float scaleLog2ForTesting() const {
        return _scaleLog2;
    }

   private:
    std::shared_ptr<HipModule> _module;
    std::int64_t _qUid;
    std::int64_t _kUid;
    std::int64_t _vUid;
    std::int64_t _oUid;
    float _scaleLog2;
    std::int32_t _seqlenQ;
    std::int32_t _seqlenK;
    std::int32_t _strideQToken;
    std::int32_t _strideQHead;
    std::int32_t _strideKToken;
    std::int32_t _strideKHead;
    std::int32_t _strideVToken;
    std::int32_t _strideVHead;
    std::int32_t _strideOToken;
    std::int32_t _strideOHead;
    bool _hasStats;
    std::int64_t _statsUid;
};

}  // namespace ck_dsl_provider
