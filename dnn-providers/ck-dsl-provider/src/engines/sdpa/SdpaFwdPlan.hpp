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
/// **GPU launch.** ``execute`` resolves Q/K/V/O (+ optional sink) to
/// device pointers, marshals the host integer arrays the kernel reads
/// (cu_seqlens_q / seqused_k, plus block_tables for the non-paged paths
/// -- see ``SdpaMarshalling``), uploads them into the caller-provided
/// workspace, binds the unified 18-slot ABI, and launches on the handle's
/// stream. All four marshalling paths are wired:
///   - **dense** (!paged && !varlen): the degenerate one-block-per-position
///     layout; block_tables/cu_seqlens_q/seqused_k all come from
///     ``marshalDenseDegenerate`` and are uploaded into the workspace.
///   - **real-paged** (paged && !varlen): cu_seqlens_q/seqused_k come from
///     ``marshalDenseDegenerate`` (uniform lengths); the block_tables slot
///     binds the graph's Page_table_K device buffer DIRECTLY (no upload).
///   - **varlen** (!paged && varlen): per-sequence lengths are read from
///     the graph's seq_len_q/seq_len_kv device buffers via a small D2H
///     copy, then ``marshalVarlen`` builds all three arrays (uploaded).
///   - **paged + varlen**: the varlen seq-lens drive cu_seqlens_q/seqused_k
///     (uploaded) and the graph's block table binds directly (no upload).
/// Workspace bytes come from ``getWorkspaceSize`` (the plan never
/// allocates device memory internally).
class SdpaFwdPlan : public hipdnn_plugin_sdk::IPlan<::CkDslHandle> {
   public:
    /// Construction params beyond the tensor UIDs + launch scalars:
    ///   ``batch``     -- num_seqs (the dense-degenerate sequence count).
    ///   ``blockSize`` -- paged KV block size in tokens (one of 16/32/64);
    ///                    sizes the degenerate block table + the
    ///                    ``block_table_stride`` arg.
    ///   ``isPaged`` / ``isVarlen`` -- marshalling-path selectors; all four
    ///                    combinations are wired (see the class comment).
    ///   ``useSinks``  -- bind the sink_ptr from ``sinkUid`` (else null).
    ///   ``sinkUid``   -- device-buffer uid for the sink tensor (the SDPA
    ///                    sink_token uid; -1 when sinks are off).
    ///   ``pageTableUid`` -- device-buffer uid for the graph's Page_table_K
    ///                    tensor, bound directly to the block_tables slot on
    ///                    the paged paths (-1 when not paged).
    ///   ``seqLenQUid`` / ``seqLenKvUid`` -- device-buffer uids for the
    ///                    graph's seq_len_q / seq_len_kv tensors; read via a
    ///                    small D2H copy on the varlen paths (-1 otherwise).
    SdpaFwdPlan(std::shared_ptr<HipModule> module, std::int64_t qUid, std::int64_t kUid,
                std::int64_t vUid, std::int64_t oUid, float scaleLog2, std::int32_t seqlenQ,
                std::int32_t seqlenK, std::int32_t strideQToken, std::int32_t strideQHead,
                std::int32_t strideKToken, std::int32_t strideKHead, std::int32_t strideVToken,
                std::int32_t strideVHead, std::int32_t strideOToken, std::int32_t strideOHead,
                std::int32_t batch, std::int32_t blockSize, bool isPaged, bool isVarlen,
                bool useSinks, std::int64_t pageTableUid = -1, std::int64_t seqLenQUid = -1,
                std::int64_t seqLenKvUid = -1, std::int64_t sinkUid = -1, bool hasStats = false,
                std::int64_t statsUid = -1);

    /// Bytes the caller must allocate and pass as ``workspace`` to
    /// ``execute``: enough device memory to hold the three host-marshalled
    /// i32 arrays (block_tables / cu_seqlens_q / seqused_k), each rounded
    /// up to a 256-byte sub-region. Returns 0 only defensively when
    /// ``blockSize <= 0``.
    std::size_t getWorkspaceSize(const ::CkDslHandle& handle) const override;

    /// Resolve Q/K/V/O (+ optional sink) device pointers from
    /// ``deviceBuffers`` by uid (so a graph-vs-buffer mismatch surfaces
    /// here), marshal the host integer arrays for the selected path,
    /// upload the workspace-bound ones, bind the unified 18-slot ABI, and
    /// launch on the handle's stream. The path is chosen from
    /// ``isPaged`` / ``isVarlen``: the varlen paths first D2H-copy the
    /// graph's per-sequence lengths; the paged paths bind the graph's
    /// block table directly (see the class comment for the four cases).
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` if a UID is
    /// missing, if ``workspace`` is null (the caller must provide
    /// ``getWorkspaceSize`` bytes -- the plan never allocates device
    /// memory itself), or if a HIP copy/launch fails.
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
    std::int32_t _batch;
    std::int32_t _blockSize;
    bool _isPaged;
    bool _isVarlen;
    bool _useSinks;
    std::int64_t _pageTableUid;
    std::int64_t _seqLenQUid;
    std::int64_t _seqLenKvUid;
    std::int64_t _sinkUid;
    bool _hasStats;
    std::int64_t _statsUid;
};

}  // namespace ck_dsl_provider
