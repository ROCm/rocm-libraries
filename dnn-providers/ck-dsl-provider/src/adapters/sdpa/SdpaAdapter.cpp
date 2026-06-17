// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaAdapter.hpp"

#include <cmath>
#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <limits>
#include <sstream>
#include <string>

namespace ck_dsl_provider {

namespace {

using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using TensorMap = SdpaAdapter::TensorMap;

[[noreturn]] void throwBadParam(const std::string& msg) {
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "SdpaAdapter: " + msg);
}

const TensorAttributes& lookupTensor(const TensorMap& tensorMap, std::int64_t uid,
                                     const char* role) {
    auto it = tensorMap.find(uid);
    if (it == tensorMap.end() || it->second == nullptr) {
        std::ostringstream oss;
        oss << "tensor map missing entry for " << role << " uid=" << uid;
        throwBadParam(oss.str());
    }
    return *it->second;
}

std::int32_t narrowToI32(std::int64_t value, const char* fieldName) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        std::ostringstream oss;
        oss << "field '" << fieldName << "' value " << value << " does not fit in int32_t";
        throwBadParam(oss.str());
    }
    return static_cast<std::int32_t>(value);
}

void checkDtypeHalfOrBf16(const TensorAttributes& t, const char* role) {
    // The unified paged/varlen kernel emits both FP16 and BF16 I/O.
    // Reject anything else at the adapter boundary so the capability gate
    // (tryBuildSpec) can decline cleanly and the engine selection layer
    // can fall through to other engines.
    if (t.data_type() != DataType::HALF && t.data_type() != DataType::BFLOAT16) {
        std::ostringstream oss;
        oss << role << " data_type must be HALF (FP16) or BFLOAT16; got "
            << static_cast<int>(t.data_type());
        throwBadParam(oss.str());
    }
}

// Map the (already-validated half-or-bf16) Q dtype to the kernel's dtype
// string. Q drives codegen; K/V/O are required to match it (checked by
// the caller).
const char* dtypeString(const TensorAttributes& t) {
    return t.data_type() == DataType::BFLOAT16 ? "bf16" : "f16";
}

void check4dDims(const TensorAttributes& t, const char* role) {
    if (t.dims() == nullptr || t.dims()->size() != 4) {
        std::ostringstream oss;
        oss << role << " dims must be 4-D ([B, H, S, D]); got size "
            << (t.dims() == nullptr ? 0u : t.dims()->size());
        throwBadParam(oss.str());
    }
}

std::int32_t getDim(const TensorAttributes& t, std::uint32_t idx, const char* role,
                    const char* fieldName) {
    // Caller has already validated dims is 4-D.
    auto raw = t.dims()->Get(idx);
    return narrowToI32(raw, (std::string(role) + "." + fieldName).c_str());
}

std::int32_t getStride(const TensorAttributes& t, std::uint32_t idx, const char* role,
                       const char* fieldName) {
    if (t.strides() == nullptr || t.strides()->size() != 4) {
        std::ostringstream oss;
        oss << role << " strides must be 4-D ([B, H, S, D]); got size "
            << (t.strides() == nullptr ? 0u : t.strides()->size());
        throwBadParam(oss.str());
    }
    auto raw = t.strides()->Get(idx);
    return narrowToI32(raw, (std::string(role) + "." + fieldName).c_str());
}

// Enforce the two memory-layout invariants the FMHA-forward kernel
// hard-assumes (BSHD-compatible layout). The kernel has no batch stride:
// it folds the batch offset as ``batch_idx * seqlen * stride_token`` and
// adds the head-dim index as a raw element offset with contiguous vector
// loads, so:
//   1. the head-dim (last axis) must be unit-stride for every tensor;
//   2. for batch>1 the batch stride (strides[0]) must equal
//      ``seqlen * sequence-dim stride`` (= S * strides[2]).
// For batch==1 the batch term is multiplied by 0, so the batch-stride
// check is skipped (a B==1 tensor of any compatible head/seq strides is
// fine).
//
// The comparison is done in int64_t using the RAW (un-narrowed) strides
// and dims so a large-but-valid stride does not false-trip on the
// i32-narrowing bound: ``S * strides[2]`` is computed and compared as
// int64_t.
void checkBshdLayout(const TensorAttributes& t, std::int32_t B, std::int32_t S, const char* role) {
    if (t.strides() == nullptr || t.strides()->size() != 4) {
        std::ostringstream oss;
        oss << role << " strides must be 4-D ([B, H, S, D]); got size "
            << (t.strides() == nullptr ? 0u : t.strides()->size());
        throwBadParam(oss.str());
    }
    const std::int64_t strideBatch = t.strides()->Get(0);
    const std::int64_t strideToken = t.strides()->Get(2);
    const std::int64_t strideHead = t.strides()->Get(3);

    if (strideHead != 1) {
        std::ostringstream oss;
        oss << role << " head-dim (last axis) must be unit-stride (contiguous); got stride "
            << strideHead;
        throwBadParam(oss.str());
    }

    if (B > 1) {
        const std::int64_t expectedBatch = static_cast<std::int64_t>(S) * strideToken;
        if (strideBatch != expectedBatch) {
            std::ostringstream oss;
            oss << role << " batch stride (" << strideBatch << ") must equal seqlen (" << S
                << ") * sequence stride (" << strideToken << ") = " << expectedBatch
                << "; the FMHA kernel requires a BSHD-compatible layout (heads interleaved "
                   "within each sequence position) for batch>1";
            throwBadParam(oss.str());
        }
    }
}

// Derive the paged-KV block size (tokens per physical block) for a real
// paged graph. hipDNN's SDPA surface has NO dedicated block-size field:
// paging is expressed only by the Page_table_K/V tensors plus the
// optional ``max_seq_len_kv`` scalar. The vLLM-style page table is shaped
// [num_seqs, max_blocks_per_seq], so the block size is recovered as
//   ceil(max_seq_len_kv / max_blocks_per_seq)
// then snapped up to the nearest kernel-supported value in {16, 32, 64}.
//
// When the page table is not rank-2, or ``max_seq_len_kv`` is absent, the
// pre-launch block size is genuinely not determinable from the graph
// alone (the true value lives in the host-side KV-cache allocation). In
// that case this falls back to a documented default of 16 (the smallest
// supported block) so the gate still yields a compile-time, kernel-valid
// block size; Phase 4 real-paged launch correctness validates the actual
// cache layout against this choice.
//
// The result is always one of {16, 32, 64}; a derived value above 64 is
// declined by the caller (a single physical block cannot exceed the
// kernel's max block size).
std::int32_t derivePagedBlockSize(const TensorAttributes& pageTableK,
                                  const SdpaAdapter::SdpaAttributes& sdpaAttr) {
    constexpr std::int32_t kDefaultBlockSize = 16;

    const bool haveBlocksPerSeq = pageTableK.dims() != nullptr && pageTableK.dims()->size() == 2;
    if (!haveBlocksPerSeq || !sdpaAttr.max_seq_len_kv().has_value()) {
        return kDefaultBlockSize;
    }

    const std::int64_t maxBlocksPerSeq = pageTableK.dims()->Get(1);
    const std::int64_t maxSeqLenKv = sdpaAttr.max_seq_len_kv().value();
    if (maxBlocksPerSeq <= 0 || maxSeqLenKv <= 0) {
        return kDefaultBlockSize;
    }

    // ceil(maxSeqLenKv / maxBlocksPerSeq).
    const std::int64_t raw = (maxSeqLenKv + maxBlocksPerSeq - 1) / maxBlocksPerSeq;

    // Snap up to the nearest supported block size; anything above 64 is
    // returned as-is so the caller can decline it with a clear reason.
    if (raw <= 16) {
        return 16;
    }
    if (raw <= 32) {
        return 32;
    }
    if (raw <= 64) {
        return 64;
    }
    return narrowToI32(raw, "derived block_size");
}

}  // namespace

SdpaSpec SdpaAdapter::buildSpec(const SdpaAttributes& sdpaAttr, const TensorMap& tensorMap) {
    const auto& Q = lookupTensor(tensorMap, sdpaAttr.q_tensor_uid(), "Q");
    const auto& K = lookupTensor(tensorMap, sdpaAttr.k_tensor_uid(), "K");
    const auto& V = lookupTensor(tensorMap, sdpaAttr.v_tensor_uid(), "V");
    const auto& O = lookupTensor(tensorMap, sdpaAttr.o_tensor_uid(), "O");

    check4dDims(Q, "Q");
    check4dDims(K, "K");
    check4dDims(V, "V");
    check4dDims(O, "O");

    checkDtypeHalfOrBf16(Q, "Q");
    checkDtypeHalfOrBf16(K, "K");
    checkDtypeHalfOrBf16(V, "V");
    checkDtypeHalfOrBf16(O, "O");

    // Q drives codegen; all four I/O tensors must share its dtype so the
    // single kernel binary is consistent.
    if (K.data_type() != Q.data_type() || V.data_type() != Q.data_type() ||
        O.data_type() != Q.data_type()) {
        std::ostringstream oss;
        oss << "Q/K/V/O must share one data_type; got Q=" << static_cast<int>(Q.data_type())
            << " K=" << static_cast<int>(K.data_type()) << " V=" << static_cast<int>(V.data_type())
            << " O=" << static_cast<int>(O.data_type());
        throwBadParam(oss.str());
    }

    // Tensor dim convention (rank-4): [B, H, S, D].
    //   Q.dims = [B, Hq,  Sq,  D ]
    //   K.dims = [B, Hkv, Skv, D ]
    //   V.dims = [B, Hkv, Skv, Dv]
    //   O.dims = [B, Hq,  Sq,  Dv]
    auto B = getDim(Q, 0, "Q", "B");
    auto Hq = getDim(Q, 1, "Q", "Hq");
    auto Sq = getDim(Q, 2, "Q", "Sq");
    auto D = getDim(Q, 3, "Q", "D");

    auto Bk = getDim(K, 0, "K", "B");
    auto Hkv = getDim(K, 1, "K", "Hkv");
    auto Skv = getDim(K, 2, "K", "Skv");
    auto Dk = getDim(K, 3, "K", "D");

    auto Bv = getDim(V, 0, "V", "B");
    auto Hkv_v = getDim(V, 1, "V", "Hkv");
    auto Skv_v = getDim(V, 2, "V", "Skv");
    auto Dv = getDim(V, 3, "V", "Dv");

    auto Bo = getDim(O, 0, "O", "B");
    auto Hq_o = getDim(O, 1, "O", "Hq");
    auto Sq_o = getDim(O, 2, "O", "Sq");
    auto Dv_o = getDim(O, 3, "O", "Dv");

    // Batch must agree across all four tensors.
    if (Bk != B || Bv != B || Bo != B) {
        std::ostringstream oss;
        oss << "batch dimension must match across Q/K/V/O; got Q.B=" << B << " K.B=" << Bk
            << " V.B=" << Bv << " O.B=" << Bo;
        throwBadParam(oss.str());
    }

    // Single head_size kernel: Dqk == Dv == D.
    if (Dk != D) {
        std::ostringstream oss;
        oss << "K head_size (" << Dk << ") must equal Q head_size (" << D << ")";
        throwBadParam(oss.str());
    }
    if (Dv != D) {
        std::ostringstream oss;
        oss << "V head_size Dv (" << Dv << ") must equal Q head_size D (" << D
            << "); a single head_size is supported";
        throwBadParam(oss.str());
    }
    if (Dv_o != D) {
        std::ostringstream oss;
        oss << "O head_size (" << Dv_o << ") must equal Q head_size D (" << D << ")";
        throwBadParam(oss.str());
    }

    // K and V share the kv sequence length.
    if (Skv_v != Skv) {
        std::ostringstream oss;
        oss << "V seqlen_k (" << Skv_v << ") must equal K seqlen_k (" << Skv << ")";
        throwBadParam(oss.str());
    }

    // O mirrors Q's query head + sequence layout.
    if (Hq_o != Hq) {
        std::ostringstream oss;
        oss << "O num_query_heads (" << Hq_o << ") must equal Q num_query_heads (" << Hq << ")";
        throwBadParam(oss.str());
    }
    if (Sq_o != Sq) {
        std::ostringstream oss;
        oss << "O seqlen_q (" << Sq_o << ") must equal Q seqlen_q (" << Sq << ")";
        throwBadParam(oss.str());
    }

    // K and V share the kv head count.
    if (Hkv_v != Hkv) {
        std::ostringstream oss;
        oss << "V num_kv_heads (" << Hkv_v << ") must equal K num_kv_heads (" << Hkv << ")";
        throwBadParam(oss.str());
    }

    // GQA: the query heads must partition evenly across the kv heads.
    if (Hkv <= 0 || Hq % Hkv != 0) {
        std::ostringstream oss;
        oss << "num_query_heads (" << Hq << ") must be a positive multiple of num_kv_heads (" << Hkv
            << ") for grouped-query attention";
        throwBadParam(oss.str());
    }

    // head_size must be one of the unified paged kernel's supported
    // values {64, 128, 256}. This is a deliberate POC narrowing relative
    // to the prior dense adapter's wider {32, 64, 128, 192, 256}: the
    // paged tiled-2D kernel only handles {64, 128, 256} (all % 32). Sizes
    // 32 and 192 are declined.
    if (D != 64 && D != 128 && D != 256) {
        std::ostringstream oss;
        oss << "head_size (" << D
            << ") must be one of {64, 128, 256} (the unified paged kernel does not support "
               "32 or 192)";
        throwBadParam(oss.str());
    }

    // All dims must be positive. This must precede the ``% 16`` seqlen
    // checks below, which a zero seqlen would otherwise pass
    // (``0 % 16 == 0``).
    if (B <= 0 || Hq <= 0 || Hkv <= 0 || Sq <= 0 || Skv <= 0 || D <= 0) {
        std::ostringstream oss;
        oss << "Q/K/V/O dims must all be positive; got B/Hq/Hkv/Sq/Skv/D = " << B << "/" << Hq
            << "/" << Hkv << "/" << Sq << "/" << Skv << "/" << D;
        throwBadParam(oss.str());
    }

    // Sequence lengths must be a multiple of the tile (16).
    if (Sq % 16 != 0) {
        std::ostringstream oss;
        oss << "seqlen_q (" << Sq << ") must be a multiple of 16";
        throwBadParam(oss.str());
    }
    if (Skv % 16 != 0) {
        std::ostringstream oss;
        oss << "seqlen_k (" << Skv << ") must be a multiple of 16";
        throwBadParam(oss.str());
    }

    // ---- Capability gate: the broad-but-safe variant matrix ----------
    //
    // The unified paged/varlen tiled-2D kernel applies causal masking
    // UNCONDITIONALLY and supports only top-left alignment, an optional
    // left (causal) window, GQA, sinks, varlen, and real paged KV. Every
    // throwBadParam below is a CLEAN capability decline: SdpaFwdPlanBuilder
    // wraps buildSpec in tryBuildSpec, which converts the throw into
    // isApplicable=false plus the reason string -- there is no separate
    // return-bool path. NO hybrid dense routing: a variant the single
    // kernel cannot do is declined, never silently downgraded.

    // -- Masking --
    //
    // Alignment is expressed two ways on the surface: the deprecated
    // ``causal_mask`` / ``causal_mask_bottom_right`` bools and the newer
    // ``diagonal_alignment`` + ``left_bound`` / ``right_bound`` band. The
    // gate honours both.
    if (sdpaAttr.alibi_mask()) {
        throwBadParam("ALiBi mask not supported");
    }
    if (sdpaAttr.padding_mask()) {
        throwBadParam("padding mask not supported");
    }
    if (sdpaAttr.attn_mask_tensor_uid().has_value()) {
        throwBadParam("additive attn_mask tensor (bias) not supported");
    }

    // Bottom-right alignment (either spelling) is not supported -- the
    // kernel only does top-left causal.
    const bool bottomRight =
        sdpaAttr.causal_mask_bottom_right() ||
        sdpaAttr.diagonal_alignment() ==
            hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    if (bottomRight) {
        throwBadParam(
            "bottom-right causal mask / diagonal_alignment=BOTTOM_RIGHT not supported "
            "(top-left causal only)");
    }

    // A right-side window is not supported: the kernel models only a left
    // (look-back) causal window. left_bound > 0 selects a sliding window;
    // left_bound == 0 (or unset) means full causal context.
    const std::int64_t rightBound =
        sdpaAttr.right_bound().has_value() ? sdpaAttr.right_bound().value() : 0;
    if (rightBound != 0) {
        throwBadParam(
            "right_bound != 0 not supported; only a left (causal look-back) window is modelled");
    }
    const std::int64_t leftBound =
        sdpaAttr.left_bound().has_value() ? sdpaAttr.left_bound().value() : 0;
    if (leftBound < 0) {
        throwBadParam("left_bound must be non-negative");
    }

    // The kernel applies causal masking unconditionally, so a non-causal
    // (bidirectional / full-context) request cannot be honoured. Causal is
    // signalled by the deprecated ``causal_mask`` bool OR a positive left
    // window (a windowed band is inherently causal here). Anything else --
    // including the prior "no mask" case -- is declined.
    const bool causalRequested = sdpaAttr.causal_mask() || leftBound > 0;
    if (!causalRequested) {
        throwBadParam(
            "non-causal/bidirectional attention not supported; the unified paged kernel applies "
            "causal masking unconditionally (set causal_mask or a left_bound window)");
    }
    const std::string maskMode = "causal";
    const std::int32_t slidingWindow = narrowToI32(leftBound, "left_bound");

    // -- Scale: only the scalar attn_scale is supported (handled at
    //    extraction); a per-element scale tensor is declined.
    if (sdpaAttr.scale_tensor_uid().has_value()) {
        throwBadParam("per-element scale tensor not supported (scalar attn_scale only)");
    }

    // -- LSE / softmax-stats output. The dense path CAN emit LSE
    //    (generate_stats, ABI slot 16) but the unified paged kernel emits
    //    NONE. Declining it here is a deliberate REGRESSION vs the dense
    //    path (further work needed); it is never silently dropped.
    const bool wantStats =
        (sdpaAttr.generate_stats().has_value() && sdpaAttr.generate_stats().value()) ||
        sdpaAttr.stats_tensor_uid().has_value();
    if (wantStats) {
        throwBadParam(
            "LSE/stats output (generate_stats) not supported on the unified paged SDPA path "
            "(regression vs dense; further work needed)");
    }
    if (sdpaAttr.max_tensor_uid().has_value() || sdpaAttr.sum_exp_tensor_uid().has_value()) {
        throwBadParam("max/sum_exp outputs not supported");
    }

    // -- Dropout (tensors or probability) is not supported.
    if (sdpaAttr.seed_tensor_uid().has_value() || sdpaAttr.offset_tensor_uid().has_value() ||
        sdpaAttr.dropout_mask_tensor_uid().has_value() ||
        sdpaAttr.dropout_scale_tensor_uid().has_value() ||
        sdpaAttr.dropout_probability().has_value()) {
        throwBadParam("dropout not supported");
    }

    // -- Block mask is not supported.
    if (sdpaAttr.block_mask_tensor_uid().has_value()) {
        throwBadParam("block mask not supported");
    }

    // -- FP8 quantization scales/descales are deferred (the kernel uses
    //    scalar k/v/out scales; hipDNN expresses descales as tensors).
    if (sdpaAttr.descale_q_tensor_uid().has_value() ||
        sdpaAttr.descale_k_tensor_uid().has_value() ||
        sdpaAttr.descale_v_tensor_uid().has_value() ||
        sdpaAttr.descale_s_tensor_uid().has_value() || sdpaAttr.scale_s_tensor_uid().has_value() ||
        sdpaAttr.scale_o_tensor_uid().has_value()) {
        throwBadParam("FP8 descale/scale tensors not supported (deferred)");
    }
    if (sdpaAttr.amax_s_tensor_uid().has_value() || sdpaAttr.amax_o_tensor_uid().has_value()) {
        throwBadParam("amax_s/amax_o outputs not supported");
    }
    if (sdpaAttr.rng_dump_tensor_uid().has_value()) {
        throwBadParam("rng_dump not supported");
    }

    // -- Variable-length sequences (cu_seqlens). Accepted: presence of the
    //    seqlen tensors selects the varlen marshalling path. Both must be
    //    present together so the kernel has consistent cu_seqlens for Q and
    //    KV.
    const bool haveSeqLenQ = sdpaAttr.seq_len_q_tensor_uid().has_value();
    const bool haveSeqLenKv = sdpaAttr.seq_len_kv_tensor_uid().has_value();
    if (haveSeqLenQ != haveSeqLenKv) {
        throwBadParam(
            "variable-length sequences require both seq_len_q and seq_len_kv tensors; got only "
            "one");
    }
    const bool isVarlen = haveSeqLenQ && haveSeqLenKv;

    // -- Paged KV. Accepted when BOTH page tables are present and
    //    consistent (the kernel takes a single block table). Only one
    //    present, or a K/V table mismatch, is declined. block_size is
    //    derived from the page-table layout + max_seq_len_kv and must land
    //    in {16, 32, 64}.
    const bool havePageK = sdpaAttr.page_table_k_tensor_uid().has_value();
    const bool havePageV = sdpaAttr.page_table_v_tensor_uid().has_value();
    if (havePageK != havePageV) {
        throwBadParam(
            "mismatched page_table_k/page_table_v: both must be present for paged KV "
            "(the kernel takes a single block table)");
    }
    bool isPaged = false;
    // Dense (non-paged) graphs carry no page table, so no block size is
    // derivable from the graph. The unified tiled-2D kernel still runs them in
    // single-block mode over contiguous KV (no reformat); default to its
    // smallest supported block (16) so the applicability gate (which requires
    // block_size in {16,32,64}) accepts dense SDPA. This mirrors the gtest
    // harness, which constructs dense specs with block_size=16. The paged
    // branch below overrides this with the page-table-derived value.
    std::int32_t blockSize = 16;
    if (havePageK && havePageV) {
        const auto& pageTableK =
            lookupTensor(tensorMap, sdpaAttr.page_table_k_tensor_uid().value(), "page_table_k");
        const auto& pageTableV =
            lookupTensor(tensorMap, sdpaAttr.page_table_v_tensor_uid().value(), "page_table_v");

        // The single-table kernel requires K and V page tables to describe
        // the same block layout (same dims). A divergent layout is a
        // single-table-vs-two-table mismatch and is declined.
        const bool dimsMatch = pageTableK.dims() != nullptr && pageTableV.dims() != nullptr &&
                               pageTableK.dims()->size() == pageTableV.dims()->size();
        bool extentsMatch = dimsMatch;
        if (dimsMatch) {
            for (std::uint32_t i = 0; i < pageTableK.dims()->size(); ++i) {
                if (pageTableK.dims()->Get(i) != pageTableV.dims()->Get(i)) {
                    extentsMatch = false;
                    break;
                }
            }
        }
        if (!extentsMatch) {
            throwBadParam(
                "mismatched page_table_k/page_table_v: the single-table kernel requires identical "
                "K and V block-table layouts");
        }

        isPaged = true;
        blockSize = derivePagedBlockSize(pageTableK, sdpaAttr);
        if (blockSize != 16 && blockSize != 32 && blockSize != 64) {
            std::ostringstream oss;
            oss << "derived paged block_size (" << blockSize << ") must be one of {16, 32, 64}";
            throwBadParam(oss.str());
        }
    }

    // -- Sinks. Accepted: presence of the Sink_token tensor enables the
    //    kernel's attention-sink path.
    const bool useSinks = sdpaAttr.sink_token_tensor_uid().has_value();

    SdpaSpec spec{};
    spec.problem.B = B;
    spec.problem.Hq = Hq;
    spec.problem.Hkv = Hkv;
    spec.problem.Sq = Sq;
    spec.problem.Skv = Skv;
    spec.problem.D = D;

    // Enforce the kernel's BSHD-compatible layout contract before
    // recording the launch-time strides: head-dim unit-stride for all
    // four tensors, and (for batch>1) batch stride == seqlen * sequence
    // stride. Q/O use seqlen_q; K/V use seqlen_k.
    checkBshdLayout(Q, B, Sq, "Q");
    checkBshdLayout(K, B, Skv, "K");
    checkBshdLayout(V, B, Skv, "V");
    checkBshdLayout(O, B, Sq, "O");

    // Strides for the kernel ABI: token = sequence-dim stride
    // (strides[2]); head = head-dim stride (strides[1]).
    spec.problem.stride_q_token = getStride(Q, 2, "Q", "stride_q_token");
    spec.problem.stride_q_head = getStride(Q, 1, "Q", "stride_q_head");
    spec.problem.stride_k_token = getStride(K, 2, "K", "stride_k_token");
    spec.problem.stride_k_head = getStride(K, 1, "K", "stride_k_head");
    spec.problem.stride_v_token = getStride(V, 2, "V", "stride_v_token");
    spec.problem.stride_v_head = getStride(V, 1, "V", "stride_v_head");
    spec.problem.stride_o_token = getStride(O, 2, "O", "stride_o_token");
    spec.problem.stride_o_head = getStride(O, 1, "O", "stride_o_head");

    // Attention scale: explicit value when set, otherwise the standard
    // 1/sqrt(head_size). The kernel consumes the scale in log2 space
    // (it computes exp2 in the softmax), so fold log2(e) in here. The
    // constant is spelled out locally to avoid the POSIX-only M_LOG2E
    // macro.
    constexpr float kLog2E = 1.44269504088896340736f;
    float attn_scale = sdpaAttr.attn_scale_value().has_value()
                           ? sdpaAttr.attn_scale_value().value()
                           : (1.0f / std::sqrt(static_cast<float>(D)));
    spec.problem.scale_log2 = attn_scale * kLog2E;

    // Dtype string drives codegen ("f16" | "bf16"); Q is the source of
    // truth (K/V/O were enforced to match above).
    spec.dtype = dtypeString(Q);
    spec.mask_mode = maskMode;

    // The unified paged kernel emits no LSE; the gate has already declined
    // any stats request, so this path never produces stats.
    spec.generate_stats = false;

    // Unified paged/varlen problem lanes. ``is_paged`` is true for a real
    // paged graph (the dense-degenerate one-block-per-sequence layout is
    // synthesized later during marshalling, Task 2c); ``block_size`` is
    // the derived paged block size (0 when not a real paged graph).
    spec.is_paged = isPaged;
    spec.block_size = blockSize;
    spec.is_varlen = isVarlen;
    spec.sliding_window = slidingWindow;
    spec.use_sinks = useSinks;

    // Mirror the problem-driven variant lanes onto the perf knobs so the
    // analytic policy and the kernel-key mapping see them consistently
    // (the scorer-driven selection in Task 2b overwrites the perf axes but
    // these problem-driven lanes stay).
    spec.knobs.use_sinks = useSinks;
    spec.knobs.sliding_window = slidingWindow;

    return spec;
}

}  // namespace ck_dsl_provider
