// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaBwdAdapter.hpp"

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
using TensorMap = SdpaBwdAdapter::TensorMap;

[[noreturn]] void throwBadParam(const std::string& msg) {
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "SdpaBwdAdapter: " + msg);
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
    // Q/K/V/O/dO are FP16 I/O for the bwd kernel; reject anything else at
    // the adapter boundary so applicability + engine selection can fall
    // through cleanly to other engines.
    if (t.data_type() != DataType::HALF) {
        std::ostringstream oss;
        oss << role << " data_type must be HALF (FP16); got " << static_cast<int>(t.data_type());
        throwBadParam(oss.str());
    }
}

void checkDtypeFloat(const TensorAttributes& t, const char* role) {
    // The gradient outputs (dQ/dK/dV) are f32 accumulators and the LSE
    // stats are natural-log values stored in f32; the bwd kernel emits
    // and consumes FLOAT for these.
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

// Enforce the two memory-layout invariants the FMHA kernels hard-assume
// (BSHD-compatible layout). The kernels have no batch stride: they fold
// the batch offset as ``batch_idx * seqlen * stride_token`` and add the
// head-dim index as a raw element offset with contiguous vector loads,
// so:
//   1. the head-dim (last axis) must be unit-stride for every tensor;
//   2. for batch>1 the batch stride (strides[0]) must equal
//      ``seqlen * sequence-dim stride`` (= S * strides[2]).
// For batch==1 the batch term is multiplied by 0, so the batch-stride
// check is skipped.
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

}  // namespace

SdpaBwdSpec SdpaBwdAdapter::buildSpec(const SdpaBackwardAttributes& sdpaAttr,
                                      const TensorMap& tensorMap) {
    const auto& Q = lookupTensor(tensorMap, sdpaAttr.q_tensor_uid(), "Q");
    const auto& K = lookupTensor(tensorMap, sdpaAttr.k_tensor_uid(), "K");
    const auto& V = lookupTensor(tensorMap, sdpaAttr.v_tensor_uid(), "V");
    const auto& O = lookupTensor(tensorMap, sdpaAttr.o_tensor_uid(), "O");
    const auto& dO = lookupTensor(tensorMap, sdpaAttr.do_tensor_uid(), "dO");
    const auto& stats = lookupTensor(tensorMap, sdpaAttr.stats_tensor_uid(), "stats");
    const auto& dQ = lookupTensor(tensorMap, sdpaAttr.dq_tensor_uid(), "dQ");
    const auto& dK = lookupTensor(tensorMap, sdpaAttr.dk_tensor_uid(), "dK");
    const auto& dV = lookupTensor(tensorMap, sdpaAttr.dv_tensor_uid(), "dV");

    check4dDims(Q, "Q");
    check4dDims(K, "K");
    check4dDims(V, "V");
    check4dDims(O, "O");
    check4dDims(dO, "dO");
    check4dDims(dQ, "dQ");
    check4dDims(dK, "dK");
    check4dDims(dV, "dV");

    // I/O tensors are FP16; gradient accumulators and the LSE stats are
    // f32.
    checkDtypeHalf(Q, "Q");
    checkDtypeHalf(K, "K");
    checkDtypeHalf(V, "V");
    checkDtypeHalf(O, "O");
    checkDtypeHalf(dO, "dO");
    checkDtypeFloat(dQ, "dQ");
    checkDtypeFloat(dK, "dK");
    checkDtypeFloat(dV, "dV");
    checkDtypeFloat(stats, "stats");

    // Tensor dim convention (rank-4): [B, H, S, D].
    //   Q.dims  = [B, Hq,  Sq,  D]
    //   K.dims  = [B, Hkv, Skv, D]
    //   V.dims  = [B, Hkv, Skv, D]
    //   O.dims  = [B, Hq,  Sq,  D]
    //   dO.dims = [B, Hq,  Sq,  D]
    //   dQ.dims = [B, Hq,  Sq,  D]
    //   dK.dims = [B, Hkv, Skv, D]
    //   dV.dims = [B, Hkv, Skv, D]
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

    auto Bdo = getDim(dO, 0, "dO", "B");
    auto Hq_do = getDim(dO, 1, "dO", "Hq");
    auto Sq_do = getDim(dO, 2, "dO", "Sq");
    auto D_do = getDim(dO, 3, "dO", "D");

    auto Bdq = getDim(dQ, 0, "dQ", "B");
    auto Hq_dq = getDim(dQ, 1, "dQ", "Hq");
    auto Sq_dq = getDim(dQ, 2, "dQ", "Sq");
    auto D_dq = getDim(dQ, 3, "dQ", "D");

    auto Bdk = getDim(dK, 0, "dK", "B");
    auto Hkv_dk = getDim(dK, 1, "dK", "Hkv");
    auto Skv_dk = getDim(dK, 2, "dK", "Skv");
    auto D_dk = getDim(dK, 3, "dK", "D");

    auto Bdv = getDim(dV, 0, "dV", "B");
    auto Hkv_dv = getDim(dV, 1, "dV", "Hkv");
    auto Skv_dv = getDim(dV, 2, "dV", "Skv");
    auto D_dv = getDim(dV, 3, "dV", "D");

    // stats: rank-3 [B, Hq, Sq] is the canonical shape, but a rank-4
    // [B, Hq, Sq, 1] is accepted too (some frontends carry a trailing
    // unit axis). Validate the B/Hq/Sq prefix against Q either way.
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

    // Batch must agree across all tensors.
    if (Bk != B || Bv != B || Bo != B || Bdo != B || Bdq != B || Bdk != B || Bdv != B || Bs != B) {
        std::ostringstream oss;
        oss << "batch dimension must match across all tensors; got Q.B=" << B << " K.B=" << Bk
            << " V.B=" << Bv << " O.B=" << Bo << " dO.B=" << Bdo << " dQ.B=" << Bdq
            << " dK.B=" << Bdk << " dV.B=" << Bdv << " stats.B=" << Bs;
        throwBadParam(oss.str());
    }

    // Single head_size kernel: Dqk == Dv == D across every tensor.
    if (Dk != D || Dv != D || Dv_o != D || D_do != D || D_dq != D || D_dk != D || D_dv != D) {
        std::ostringstream oss;
        oss << "head_size must match across all tensors; got Q.D=" << D << " K.D=" << Dk
            << " V.D=" << Dv << " O.D=" << Dv_o << " dO.D=" << D_do << " dQ.D=" << D_dq
            << " dK.D=" << D_dk << " dV.D=" << D_dv;
        throwBadParam(oss.str());
    }

    // K/V/dK/dV share the kv sequence length.
    if (Skv_v != Skv || Skv_dk != Skv || Skv_dv != Skv) {
        std::ostringstream oss;
        oss << "seqlen_k must match across K/V/dK/dV; got K.Skv=" << Skv << " V.Skv=" << Skv_v
            << " dK.Skv=" << Skv_dk << " dV.Skv=" << Skv_dv;
        throwBadParam(oss.str());
    }

    // O/dO/dQ/stats mirror Q's query head + sequence layout.
    if (Hq_o != Hq || Hq_do != Hq || Hq_dq != Hq || Hq_s != Hq) {
        std::ostringstream oss;
        oss << "num_query_heads must match across Q/O/dO/dQ/stats; got Q.Hq=" << Hq
            << " O.Hq=" << Hq_o << " dO.Hq=" << Hq_do << " dQ.Hq=" << Hq_dq << " stats.Hq=" << Hq_s;
        throwBadParam(oss.str());
    }
    if (Sq_o != Sq || Sq_do != Sq || Sq_dq != Sq || Sq_s != Sq) {
        std::ostringstream oss;
        oss << "seqlen_q must match across Q/O/dO/dQ/stats; got Q.Sq=" << Sq << " O.Sq=" << Sq_o
            << " dO.Sq=" << Sq_do << " dQ.Sq=" << Sq_dq << " stats.Sq=" << Sq_s;
        throwBadParam(oss.str());
    }

    // K/V/dK/dV share the kv head count.
    if (Hkv_v != Hkv || Hkv_dk != Hkv || Hkv_dv != Hkv) {
        std::ostringstream oss;
        oss << "num_kv_heads must match across K/V/dK/dV; got K.Hkv=" << Hkv << " V.Hkv=" << Hkv_v
            << " dK.Hkv=" << Hkv_dk << " dV.Hkv=" << Hkv_dv;
        throwBadParam(oss.str());
    }

    // GQA: the query heads must partition evenly across the kv heads.
    if (Hkv <= 0 || Hq % Hkv != 0) {
        std::ostringstream oss;
        oss << "num_query_heads (" << Hq << ") must be a positive multiple of num_kv_heads (" << Hkv
            << ") for grouped-query attention";
        throwBadParam(oss.str());
    }

    // head_size must be one of the kernel-supported values AND a multiple
    // of 64: the bwd kernel needs head_size >= WARP_SIZE (64), so the
    // forward path's head_size==32 case is rejected here.
    if ((D != 64 && D != 128 && D != 192 && D != 256) || D % 64 != 0) {
        std::ostringstream oss;
        oss << "head_size (" << D
            << ") must be one of {64, 128, 192, 256} (a multiple of 64); the bwd kernel "
               "requires head_size >= WARP_SIZE";
        throwBadParam(oss.str());
    }

    // All dims must be positive. This must precede the ``% 16`` seqlen
    // checks below, which a zero seqlen would otherwise pass
    // (``0 % 16 == 0``).
    if (B <= 0 || Hq <= 0 || Hkv <= 0 || Sq <= 0 || Skv <= 0 || D <= 0) {
        std::ostringstream oss;
        oss << "dims must all be positive; got B/Hq/Hkv/Sq/Skv/D = " << B << "/" << Hq << "/" << Hkv
            << "/" << Sq << "/" << Skv << "/" << D;
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

    // Reject every advanced feature the bwd kernel does not model.
    if (sdpaAttr.attn_mask_tensor_uid().has_value()) {
        throwBadParam("additive attn_mask tensor not supported");
    }
    if (sdpaAttr.scale_tensor_uid().has_value()) {
        throwBadParam("per-element scale tensor not supported");
    }
    if (sdpaAttr.seq_len_q_tensor_uid().has_value() ||
        sdpaAttr.seq_len_kv_tensor_uid().has_value()) {
        throwBadParam("variable-length sequences not supported");
    }
    if (sdpaAttr.seed_tensor_uid().has_value() || sdpaAttr.offset_tensor_uid().has_value() ||
        sdpaAttr.dropout_mask_tensor_uid().has_value() ||
        sdpaAttr.dropout_scale_tensor_uid().has_value() ||
        sdpaAttr.dropout_scale_inv_tensor_uid().has_value()) {
        throwBadParam("dropout not supported");
    }
    if (sdpaAttr.dbias_tensor_uid().has_value()) {
        throwBadParam("dbias output not supported");
    }
    // Dropout requested via probability (the dropout *tensors* are
    // already rejected above).
    if (sdpaAttr.dropout_probability().has_value()) {
        throwBadParam("dropout not supported");
    }

    SdpaBwdSpec spec{};
    spec.problem.B = B;
    spec.problem.Hq = Hq;
    spec.problem.Hkv = Hkv;
    spec.problem.Sq = Sq;
    spec.problem.Skv = Skv;
    spec.problem.D = D;

    // Enforce the kernels' BSHD-compatible layout contract before
    // recording the launch-time strides: head-dim unit-stride for the
    // FP16 I/O tensors and the f32 gradients, and (for batch>1) batch
    // stride == seqlen * sequence stride. Q/O/dO/dQ use seqlen_q; K/V/
    // dK/dV use seqlen_k.
    checkBshdLayout(Q, B, Sq, "Q");
    checkBshdLayout(K, B, Skv, "K");
    checkBshdLayout(V, B, Skv, "V");
    checkBshdLayout(dO, B, Sq, "dO");
    checkBshdLayout(dQ, B, Sq, "dQ");
    checkBshdLayout(dK, B, Skv, "dK");
    checkBshdLayout(dV, B, Skv, "dV");

    // Input strides for the kernel ABI: token = sequence-dim stride
    // (strides[2]); head = head-dim stride (strides[1]).
    spec.problem.stride_q_token = getStride(Q, 2, "Q", "stride_q_token");
    spec.problem.stride_q_head = getStride(Q, 1, "Q", "stride_q_head");
    spec.problem.stride_k_token = getStride(K, 2, "K", "stride_k_token");
    spec.problem.stride_k_head = getStride(K, 1, "K", "stride_k_head");
    spec.problem.stride_v_token = getStride(V, 2, "V", "stride_v_token");
    spec.problem.stride_v_head = getStride(V, 1, "V", "stride_v_head");
    spec.problem.stride_do_token = getStride(dO, 2, "dO", "stride_do_token");
    spec.problem.stride_do_head = getStride(dO, 1, "dO", "stride_do_head");

    // Gradient head strides must match the matching input head stride:
    // the kernel reuses the input head stride when writing each gradient.
    const std::int32_t stride_dq_head = getStride(dQ, 1, "dQ", "stride_dq_head");
    const std::int32_t stride_dk_head = getStride(dK, 1, "dK", "stride_dk_head");
    const std::int32_t stride_dv_head = getStride(dV, 1, "dV", "stride_dv_head");
    if (stride_dq_head != spec.problem.stride_q_head) {
        std::ostringstream oss;
        oss << "dQ head stride (" << stride_dq_head << ") must equal Q head stride ("
            << spec.problem.stride_q_head << "); the kernel reuses the input head stride for dQ";
        throwBadParam(oss.str());
    }
    if (stride_dk_head != spec.problem.stride_k_head) {
        std::ostringstream oss;
        oss << "dK head stride (" << stride_dk_head << ") must equal K head stride ("
            << spec.problem.stride_k_head << "); the kernel reuses the input head stride for dK";
        throwBadParam(oss.str());
    }
    if (stride_dv_head != spec.problem.stride_v_head) {
        std::ostringstream oss;
        oss << "dV head stride (" << stride_dv_head << ") must equal V head stride ("
            << spec.problem.stride_v_head << "); the kernel reuses the input head stride for dV";
        throwBadParam(oss.str());
    }

    // Gradient token strides for the kernel ABI.
    spec.problem.stride_dq_token = getStride(dQ, 2, "dQ", "stride_dq_token");
    spec.problem.stride_dk_token = getStride(dK, 2, "dK", "stride_dk_token");
    spec.problem.stride_dv_token = getStride(dV, 2, "dV", "stride_dv_token");

    // stats must be contiguous head-major: strides == {Hq*Sq, Sq, 1}
    // (innermost Sq unit-stride, head stride == Sq, batch stride ==
    // Hq*Sq). The LSE-prep kernel reads stats as a flat contiguous
    // [B, Hq, Sq] buffer, so a non-contiguous layout is rejected.
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

    // Attention scale: explicit value when set, otherwise the standard
    // 1/sqrt(head_size). The kernel consumes the scale in log2 space for
    // the softmax (scale_log2 = attn_scale * log2(e)) and the raw value
    // (scale_inv) elsewhere. The log2(e) constant is spelled out locally
    // to avoid the POSIX-only M_LOG2E macro.
    constexpr float kLog2E = 1.44269504088896340736f;
    float attn_scale = sdpaAttr.attn_scale_value().has_value()
                           ? sdpaAttr.attn_scale_value().value()
                           : (1.0f / std::sqrt(static_cast<float>(D)));
    spec.problem.scale_log2 = attn_scale * kLog2E;
    spec.problem.scale_inv = attn_scale;

    spec.dtype = "f16";
    spec.mask_mode = maskMode;
    return spec;
}

}  // namespace ck_dsl_provider
