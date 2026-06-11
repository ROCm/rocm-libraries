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
    return p;
}

ck_dsl::Problem buildProblem(const ParsedGemmParams& p, const std::string& arch) {
    ck_dsl::Problem prob;
    prob.op = "gemm";
    prob.dtype = p.dtype;
    prob.layout = "RCR";  // ck_dsl shipped GEMM convention; see CkDslGemmPlan note
    prob.arch = arch;
    prob.M = p.M;
    prob.N = p.N;
    prob.K = p.K;
    return prob;
}

}  // namespace CkDslParamParser
}  // namespace ck_dsl_plugin
