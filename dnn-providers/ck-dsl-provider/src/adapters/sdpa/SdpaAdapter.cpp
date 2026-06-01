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

void checkDtypeHalf(const TensorAttributes& t, const char* role) {
    // The DSL FMHA-forward kernel currently only emits FP16 I/O. Reject
    // anything else at the adapter boundary so applicability + the
    // engine selection layer can fall through cleanly to other engines.
    if (t.data_type() != DataType::HALF) {
        std::ostringstream oss;
        oss << role << " data_type must be HALF (FP16); got " << static_cast<int>(t.data_type());
        throwBadParam(oss.str());
    }
}

void checkDtypeFloat(const TensorAttributes& t, const char* role) {
    // The optional forward stats (LSE) output is natural-log values
    // stored in f32; the kernel writes FLOAT for this tensor.
    if (t.data_type() != DataType::FLOAT) {
        std::ostringstream oss;
        oss << role << " data_type must be FLOAT (f32); got " << static_cast<int>(t.data_type());
        throwBadParam(oss.str());
    }
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

// Validate the optional forward stats (LSE) output tensor. The kernel
// writes natural-log LSE as a flat contiguous head-major [B, Hq, Sq]
// f32 buffer, so:
//   * dtype must be FLOAT (f32);
//   * dims must be rank-3 [B, Hq, Sq] (canonical) or rank-4
//     [B, Hq, Sq, 1] (some frontends carry a trailing unit axis);
//   * the B/Hq/Sq prefix must match the Q-derived extents;
//   * the [B, Hq, Sq] dims must be contiguous head-major: strides
//     {Hq*Sq, Sq, 1} (innermost Sq unit-stride, head stride == Sq,
//     batch stride == Hq*Sq).
void checkStatsTensor(const TensorAttributes& stats, std::int32_t B, std::int32_t Hq,
                      std::int32_t Sq) {
    checkDtypeFloat(stats, "stats");

    // Rank-3 [B, Hq, Sq] is canonical; rank-4 [B, Hq, Sq, 1] is accepted
    // too. Validate the B/Hq/Sq prefix against Q either way.
    if (stats.dims() == nullptr || (stats.dims()->size() != 3 && stats.dims()->size() != 4)) {
        std::ostringstream oss;
        oss << "stats dims must be rank-3 ([B, Hq, Sq]) or rank-4 ([B, Hq, Sq, 1]); got size "
            << (stats.dims() == nullptr ? 0u : stats.dims()->size());
        throwBadParam(oss.str());
    }
    const std::uint32_t statsRank = stats.dims()->size();
    if (statsRank == 4) {
        auto statsLast = getDim(stats, 3, "stats", "trailing");
        if (statsLast != 1) {
            std::ostringstream oss;
            oss << "stats rank-4 trailing dim (" << statsLast << ") must be 1 ([B, Hq, Sq, 1])";
            throwBadParam(oss.str());
        }
    }

    auto Bs = getDim(stats, 0, "stats", "B");
    auto Hq_s = getDim(stats, 1, "stats", "Hq");
    auto Sq_s = getDim(stats, 2, "stats", "Sq");
    if (Bs != B || Hq_s != Hq || Sq_s != Sq) {
        std::ostringstream oss;
        oss << "stats shape [B, Hq, Sq] must match Q; expected {" << B << ", " << Hq << ", " << Sq
            << "}, got {" << Bs << ", " << Hq_s << ", " << Sq_s << "}";
        throwBadParam(oss.str());
    }

    // Contiguous head-major: strides == {Hq*Sq, Sq, 1}. The kernel writes
    // stats as a flat contiguous [B, Hq, Sq] buffer, so a non-contiguous
    // layout is rejected.
    if (stats.strides() == nullptr || stats.strides()->size() != statsRank) {
        std::ostringstream oss;
        oss << "stats strides must match its rank (" << statsRank << "); got size "
            << (stats.strides() == nullptr ? 0u : stats.strides()->size());
        throwBadParam(oss.str());
    }
    const std::int64_t statsStrideBatch = stats.strides()->Get(0);
    const std::int64_t statsStrideHead = stats.strides()->Get(1);
    const std::int64_t statsStrideSeq = stats.strides()->Get(2);
    const std::int64_t expectedStatsStrideBatch = static_cast<std::int64_t>(Hq) * Sq;
    if (statsStrideSeq != 1 || statsStrideHead != Sq ||
        statsStrideBatch != expectedStatsStrideBatch) {
        std::ostringstream oss;
        oss << "stats must be contiguous head-major [B, Hq, Sq]; expected strides {" << Hq * Sq
            << ", " << Sq << ", 1}, got {" << statsStrideBatch << ", " << statsStrideHead << ", "
            << statsStrideSeq << "}";
        throwBadParam(oss.str());
    }
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

    checkDtypeHalf(Q, "Q");
    checkDtypeHalf(K, "K");
    checkDtypeHalf(V, "V");
    checkDtypeHalf(O, "O");

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

    // head_size must be one of the kernel-supported values (matches the
    // DSL's validate_common_spec).
    if (D != 32 && D != 64 && D != 128 && D != 192 && D != 256) {
        std::ostringstream oss;
        oss << "head_size (" << D << ") must be one of {32, 64, 128, 192, 256}";
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

    // Mask support: top-left causal or no mask only.
    if (sdpaAttr.alibi_mask()) {
        throwBadParam("ALiBi mask is not supported");
    }
    if (sdpaAttr.padding_mask()) {
        throwBadParam("padding mask is not supported");
    }
    if (sdpaAttr.causal_mask_bottom_right()) {
        throwBadParam("bottom-right causal mask is not supported (top-left causal only)");
    }
    if (sdpaAttr.left_bound().has_value() || sdpaAttr.right_bound().has_value()) {
        throwBadParam("sliding-window attention (left_bound/right_bound) is not supported");
    }
    std::string maskMode = sdpaAttr.causal_mask() ? "causal" : "none";

    // Reject every advanced feature the M1 forward kernel does not model.
    if (sdpaAttr.attn_mask_tensor_uid().has_value()) {
        throwBadParam("additive attn_mask tensor not supported");
    }
    if (sdpaAttr.scale_tensor_uid().has_value()) {
        throwBadParam("per-element scale tensor not supported");
    }
    // Opt-in forward stats (LSE) output. The request is signalled either
    // by ``generate_stats() == true`` or by the presence of a
    // ``stats_tensor_uid``; either way the output tensor must be provided
    // so the kernel has somewhere to write. The single combined LSE
    // output is supported (head-major [B, Hq, Sq] f32 contiguous); the
    // separate max / sum_exp outputs are not (rejected below). The stats
    // tensor itself is validated after the Q-derived B/Hq/Sq extents are
    // known.
    const bool wantStats =
        (sdpaAttr.generate_stats().has_value() && sdpaAttr.generate_stats().value()) ||
        sdpaAttr.stats_tensor_uid().has_value();
    if (wantStats && !sdpaAttr.stats_tensor_uid().has_value()) {
        throwBadParam(
            "generate_stats requested but no stats_tensor_uid provided; the LSE output tensor "
            "must be supplied");
    }
    if (sdpaAttr.seq_len_q_tensor_uid().has_value() ||
        sdpaAttr.seq_len_kv_tensor_uid().has_value()) {
        throwBadParam("variable-length sequences not supported");
    }
    if (sdpaAttr.seed_tensor_uid().has_value() || sdpaAttr.offset_tensor_uid().has_value() ||
        sdpaAttr.dropout_mask_tensor_uid().has_value() ||
        sdpaAttr.dropout_scale_tensor_uid().has_value()) {
        throwBadParam("dropout not supported");
    }
    if (sdpaAttr.page_table_k_tensor_uid().has_value() ||
        sdpaAttr.page_table_v_tensor_uid().has_value()) {
        throwBadParam("paged KV not supported");
    }
    if (sdpaAttr.block_mask_tensor_uid().has_value()) {
        throwBadParam("block mask not supported");
    }
    if (sdpaAttr.sink_token_tensor_uid().has_value()) {
        throwBadParam("sink tokens not supported");
    }
    if (sdpaAttr.max_tensor_uid().has_value() || sdpaAttr.sum_exp_tensor_uid().has_value()) {
        throwBadParam("max/sum_exp outputs not supported in M1 forward");
    }
    // FP8 quantization scales/descales -- FP16-only kernel cannot consume them.
    if (sdpaAttr.descale_q_tensor_uid().has_value() ||
        sdpaAttr.descale_k_tensor_uid().has_value() ||
        sdpaAttr.descale_v_tensor_uid().has_value() ||
        sdpaAttr.descale_s_tensor_uid().has_value() || sdpaAttr.scale_s_tensor_uid().has_value() ||
        sdpaAttr.scale_o_tensor_uid().has_value()) {
        throwBadParam("FP8 descale/scale tensors not supported (FP16 only in M1 forward)");
    }
    if (sdpaAttr.amax_s_tensor_uid().has_value() || sdpaAttr.amax_o_tensor_uid().has_value()) {
        throwBadParam("amax_s/amax_o outputs not supported in M1 forward");
    }
    if (sdpaAttr.rng_dump_tensor_uid().has_value()) {
        throwBadParam("rng_dump not supported");
    }
    // Dropout requested via probability (the dropout *tensors* are
    // already rejected above).
    if (sdpaAttr.dropout_probability().has_value()) {
        throwBadParam("dropout not supported");
    }

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

    spec.dtype = "f16";
    spec.mask_mode = maskMode;

    // Opt-in forward stats (LSE) output. When requested, validate the
    // supplied stats tensor (dtype FLOAT, head-major [B, Hq, Sq]
    // contiguous) here so isApplicable declines a malformed stats request
    // rather than failing at launch. The stats UID itself is not stored
    // on the spec -- the plan builder reads it straight off the FB node;
    // the spec only records that stats are enabled (codegen-relevant).
    if (wantStats) {
        const auto& stats = lookupTensor(tensorMap, sdpaAttr.stats_tensor_uid().value(), "stats");
        checkStatsTensor(stats, B, Hq, Sq);
    }
    spec.generate_stats = wantStats;

    return spec;
}

}  // namespace ck_dsl_provider
