// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslAttnParamParser.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cmath>
#include <stdexcept>

namespace ck_dsl_plugin {
namespace CkDslAttnParamParser {

namespace {
namespace fb = hipdnn_flatbuffers_sdk::data_objects;

std::string mapDataType(fb::DataType dt) {
    switch (dt) {
        case fb::DataType::HALF:
            return "fp16";
        case fb::DataType::BFLOAT16:
            return "bf16";
        default:
            return "";
    }
}
const fb::TensorAttributes* lookupTensor(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph, int64_t uid) {
    const auto& m = graph.getTensorMap();
    auto it = m.find(uid);
    return it != m.end() ? it->second : nullptr;
}
// CK mask_enum: 0=no,1=top_left,2=bottom_right,3=window_generic
int mapMask(const fb::SdpaAttributes* a) {
    if (a->left_bound().has_value() || a->right_bound().has_value()) return 3;
    if (a->causal_mask_bottom_right()) return 2;
    if (a->causal_mask()) return 1;
    return 0;
}
int mapBias(const fb::SdpaAttributes* a) {
    if (a->alibi_mask()) return 2;
    if (a->attn_mask_tensor_uid().value_or(0) != 0) return 1;
    return 0;
}

}  // namespace

bool isPhysicalBhsdLayout(int64_t strideH, int64_t strideS) {
    return strideH > strideS;
}

bool isSdpaGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
    if (graph.nodeCount() != 1) return false;
    return graph.getNode(0).attributes_type() == fb::NodeAttributes::SdpaAttributes;
}

ParsedAttnParams parseSdpaGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
    const auto* attr = graph.getNode(0).attributes_as_SdpaAttributes();
    if (attr == nullptr) throw std::runtime_error("CkDslAttn: not an SdpaAttributes node");

    ParsedAttnParams p;
    p.q_uid = attr->q_tensor_uid();
    p.k_uid = attr->k_tensor_uid();
    p.v_uid = attr->v_tensor_uid();
    p.o_uid = attr->o_tensor_uid();
    p.bias_uid = attr->attn_mask_tensor_uid().value_or(0);
    if (attr->stats_tensor_uid().value_or(0) != 0) {
        p.lse_uid = attr->stats_tensor_uid().value_or(0);
        p.has_lse = true;
    }
    if (attr->generate_stats()) p.has_lse = true;

    const auto* q = lookupTensor(graph, p.q_uid);
    const auto* k = lookupTensor(graph, p.k_uid);
    const auto* v = lookupTensor(graph, p.v_uid);
    if (!q || !k || !v) throw std::runtime_error("CkDslAttn: missing Q/K/V");

    p.dtype = mapDataType(q->data_type());
    if (p.dtype.empty()) throw std::runtime_error("CkDslAttn: unsupported dtype");

    const auto* qd = q->dims();
    const auto* kd = k->dims();
    const auto* vd = v->dims();
    // SDPA Q/K/V are rank-4 [B,H,S,D]. Validate the dims vectors exist and carry
    // all four extents before indexing (a malformed/unsupported graph could omit
    // dims or carry a lower rank). flatbuffers::Vector::Get only guards with
    // FLATBUFFERS_ASSERT, which is compiled out under NDEBUG, so an unchecked
    // Get() on a null/short vector is an out-of-bounds read in release builds.
    // Throw a clear error instead; isApplicable() catches it and declines.
    // Mirrors the rank checks in CkDslParamParser::parseGemmGraph.
    if (qd == nullptr || kd == nullptr || vd == nullptr || qd->size() < 4 ||
        kd->size() < 4 || vd->size() < 4)
        throw std::runtime_error("CkDslAttn: expected rank-4 [B,H,S,D] Q/K/V dims");
    // hipDNN SDPA tensors are *logical* BHSD: dims are [B, H, S, D] (matching the
    // asm_sdpa engine and the benchmark graph naming bN_hN_sN_dN). Read the
    // logical extents accordingly.
    p.batch = qd->Get(0);
    p.nhead_q = qd->Get(1);
    p.seqlen_q = qd->Get(2);
    p.hdim_q = qd->Get(3);
    p.nhead_k = kd->Get(1);
    p.seqlen_k = kd->Get(2);
    p.hdim_v = vd->Get(3);

    // Applicability is driven by the *physical* memory layout, read from Q's
    // strides (aligned to the logical dims [B,H,S,D]). The ck_dsl kernel is
    // BSHD-native (S-major-over-H: the S axis must be stored outside H). So:
    //   stride(H) > stride(S)  -> physical is BHSD-contiguous (H-major-over-S)
    //                             -> decline (is_bhsd=true); a transpose into the
    //                                kernel's BSHD/paged layout is a follow-on.
    //   stride(S) > stride(H)  -> physical is already the kernel's native BSHD
    //                             memory -> run (is_bhsd=false).
    // If strides are absent in the serialized graph, assume the kernel-native
    // BSHD layout and let validation catch a true mismatch. This preserves prior
    // run-all behavior rather than declining outright.
    const auto* qs = q->strides();
    if (qs != nullptr && qs->size() >= 3) {
        const int64_t stride_h = qs->Get(1);  // stride of the H axis (dim 1)
        const int64_t stride_s = qs->Get(2);  // stride of the S axis (dim 2)
        // Only classify when both strides are well-formed (positive). Zero /
        // broadcast / negative strides are not a real physical layout and would
        // make the stride-order comparison meaningless, so treat them like
        // absent strides and fall back to the conservative kernel-native default
        // (not BHSD). Mirrors detectBLayout's `stride <= 0 -> Unknown` guard.
        p.is_bhsd =
            (stride_h > 0 && stride_s > 0) && isPhysicalBhsdLayout(stride_h, stride_s);
    } else {
        p.is_bhsd = false;
    }

    p.mask_type = mapMask(attr);
    p.bias_type = mapBias(attr);
    p.scale = attr->attn_scale_value().value_or(0.0f);
    if (p.scale == 0.0f && p.hdim_q > 0) p.scale = 1.0f / std::sqrt(static_cast<float>(p.hdim_q));
    return p;
}

ck_dsl::Problem buildProblem(const ParsedAttnParams& p, const std::string& arch) {
    ck_dsl::Problem prob;
    prob.op = "attention";
    prob.dtype = p.dtype;
    prob.layout = p.is_bhsd ? "BHSD" : "BSHD";
    prob.arch = arch;
    prob.M = p.seqlen_q;
    prob.N = p.seqlen_k;
    prob.K = p.hdim_q;
    // Attention dims for the FMHA ML feature extractor.
    prob.batch = p.batch;
    prob.nhead_q = p.nhead_q;
    prob.nhead_k = p.nhead_k;
    prob.seqlen_q = p.seqlen_q;
    prob.seqlen_k = p.seqlen_k;
    prob.hdim_q = p.hdim_q;
    prob.hdim_v = p.hdim_v;
    prob.mask_type = p.mask_type;
    return prob;
}

}  // namespace CkDslAttnParamParser
}  // namespace ck_dsl_plugin
