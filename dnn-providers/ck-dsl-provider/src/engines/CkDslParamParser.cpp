// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslParamParser.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <stdexcept>

namespace ck_dsl_plugin {
namespace CkDslParamParser {

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
}  // namespace

BLayout detectBLayout(long K, long N, int64_t stride_outer, int64_t stride_inner) {
    // B logical dims are [K, N]: stride_outer is the stride of the K axis,
    // stride_inner is the stride of the N axis.
    //
    //   Row-major [K,N]: the N axis is contiguous -> stride_inner == 1 and
    //                    stride_outer == N. (standard NN B)
    //   RCR (B is [N,K]): the K axis is contiguous -> stride_outer == 1 and
    //                     stride_inner == K. (shipped ck_dsl ABI; B^T view)
    //
    // 1xN / Kx1 degenerate cases (N==1 or K==1) are ambiguous because both
    // candidate strides collapse to 1; report Unknown so the caller decides.
    if (stride_outer <= 0 || stride_inner <= 0) return BLayout::Unknown;
    if (N <= 1 || K <= 1) return BLayout::Unknown;

    const bool rowmajor_kn = (stride_inner == 1 && stride_outer == N);
    const bool rcr_nk = (stride_outer == 1 && stride_inner == K);
    if (rcr_nk && !rowmajor_kn) return BLayout::RCR_NK;
    if (rowmajor_kn && !rcr_nk) return BLayout::RowMajor_KN;
    return BLayout::Unknown;
}

const char* bLayoutName(BLayout l) {
    switch (l) {
        case BLayout::RowMajor_KN:
            return "RRR";
        case BLayout::RCR_NK:
            return "RCR";
        default:
            return "unknown";
    }
}

bool isSupportedBLayout(BLayout l) {
    return l == BLayout::RCR_NK;
}

bool isGemmGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
    if (graph.nodeCount() != 1) return false;
    return graph.getNode(0).attributes_type() == fb::NodeAttributes::MatmulAttributes;
}

ParsedGemmParams parseGemmGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
    const auto& node = graph.getNode(0);
    const auto* attr = node.attributes_as_MatmulAttributes();
    if (attr == nullptr) throw std::runtime_error("CkDslParamParser: not a MatmulAttributes node");

    ParsedGemmParams p;
    p.a_uid = attr->a_tensor_uid();
    p.b_uid = attr->b_tensor_uid();
    p.c_uid = attr->c_tensor_uid();

    const auto* a = lookupTensor(graph, p.a_uid);
    const auto* b = lookupTensor(graph, p.b_uid);
    const auto* c = lookupTensor(graph, p.c_uid);
    if (a == nullptr || b == nullptr || c == nullptr)
        throw std::runtime_error("CkDslParamParser: missing A/B/C tensors");

    p.dtype = mapDataType(a->data_type());
    if (p.dtype.empty()) throw std::runtime_error("CkDslParamParser: unsupported dtype");

    // Matmul C[M,N] = A[M,K] x B[K,N]. Take the last two dims (rank-2 GEMM;
    // higher-rank batched matmul is a follow-on family).
    const auto* ad = a->dims();
    const auto* cd = c->dims();
    if (ad == nullptr || cd == nullptr || ad->size() < 2 || cd->size() < 2)
        throw std::runtime_error("CkDslParamParser: expected rank>=2 A and C");
    p.M = ad->Get(ad->size() - 2);
    p.K = ad->Get(ad->size() - 1);
    p.N = cd->Get(cd->size() - 1);

    // Detect the physical layout of B from its declared strides instead of
    // silently assuming the shipped RCR (B stored [N,K]) ABI. B's logical dims
    // are [K,N]; we read the last-two-dim strides (outer=K axis, inner=N axis).
    const auto* bd = b->dims();
    const auto* bs = b->strides();
    if (bd == nullptr || bd->size() < 2)
        throw std::runtime_error("CkDslParamParser: expected rank>=2 B");
    // K-consistency: B's [K,N] declared dims must agree with A's K and C's N.
    const long bK = bd->Get(bd->size() - 2);
    const long bN = bd->Get(bd->size() - 1);
    if (bK != p.K || bN != p.N)
        throw std::runtime_error("CkDslParamParser: B dims [K,N] disagree with A/C");

    if (bs == nullptr || bs->size() < 2) {
        // No strides declared: fall back to the shipped RCR ABI (the kernel's
        // native B[N,K] layout) rather than rejecting, preserving prior behavior.
        p.b_layout = BLayout::RCR_NK;
    } else {
        const int64_t s_outer = bs->Get(bs->size() - 2);  // stride of K axis
        const int64_t s_inner = bs->Get(bs->size() - 1);  // stride of N axis
        p.b_layout = detectBLayout(p.K, p.N, s_outer, s_inner);
    }
    return p;
}

ck_dsl::Problem buildProblem(const ParsedGemmParams& p, const std::string& arch) {
    ck_dsl::Problem prob;
    prob.op = "gemm";
    prob.dtype = p.dtype;
    // The shipped ck_dsl GEMM is RCR (B stored [N,K]). The dispatcher catalog
    // only holds RCR candidates, so we tag the problem RCR for both the
    // detected-RCR case and the strides-absent fallback. Other detected layouts
    // are rejected upstream (CkDslGemmPlanBuilder) before reaching select().
    prob.layout = "RCR";  // ck_dsl shipped GEMM convention; see CkDslGemmPlan note
    prob.arch = arch;
    prob.M = p.M;
    prob.N = p.N;
    prob.K = p.K;
    return prob;
}

}  // namespace CkDslParamParser
}  // namespace ck_dsl_plugin
