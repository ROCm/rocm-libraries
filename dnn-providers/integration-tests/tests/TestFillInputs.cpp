// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp4Tensor.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp6Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input-init/FillInputs.hpp"
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests;

namespace hipdnn_integration_tests
{

// Lets EXPECT_EQ on recipes print both sides on failure.
static void PrintTo(const FillRecipe& f, std::ostream* os)
{
    if(f.kind == FillRecipe::Kind::FIXED)
    {
        *os << "FIXED(" << f.value << ")";
        return;
    }
    if(f.kind == FillRecipe::Kind::FREE)
    {
        *os << "FREE[" << f.lo << ", " << f.hi << "]";
        return;
    }
    *os << InputFillRecipes::kindToString(f.kind) << "(lo=" << f.lo << ", hi=" << f.hi
        << ", scale=" << f.scale << ", special=" << static_cast<int>(f.special)
        << ", fraction=" << f.fraction << ")";
}

} // namespace hipdnn_integration_tests

namespace
{

const std::vector<int64_t> kDims = {2, 3};
const std::vector<int64_t> kStrides = {3, 1};

InputTensorMap makeTensors(const std::vector<int64_t>& uids)
{
    InputTensorMap map;
    for(const int64_t uid : uids)
    {
        map[uid] = std::make_unique<hipdnn_data_sdk::utilities::Tensor<float>>(kDims, kStrides);
        map[uid]->fillTensorWithValue(0.f);
    }
    return map;
}

struct GraphResult
{
    flatbuffers::FlatBufferBuilder builder;
    const Graph* graph = nullptr;

    const Node& node(uint32_t i) const
    {
        return *graph->nodes()->Get(i);
    }

    std::vector<int64_t> leafInputUids(const std::set<int64_t>& outputUids) const
    {
        std::vector<int64_t> uids;
        for(const auto* t : *graph->tensors())
        {
            if(!t->virtual_() && outputUids.count(t->uid()) == 0)
            {
                uids.push_back(t->uid());
            }
        }
        return uids;
    }
};

// ── Conv fwd (single node) ──────────────────────────────────────────────────

GraphResult buildConvFwdGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "w", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "y", DataType::FLOAT, &kStrides, &kDims));

    auto conv = CreateConvolutionFwdAttributesDirect(b, 1, 2, 3);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "conv", DataType::FLOAT, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── Conv + bias (2-node fused) ──────────────────────────────────────────────
// conv.y (uid 10) is virtual; bias (uid 4) is leaf

GraphResult buildConvBiasGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "w", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 10, "conv_y", DataType::FLOAT, &kStrides, &kDims, true));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 4, "bias", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 5, "out", DataType::FLOAT, &kStrides, &kDims));

    auto conv = CreateConvolutionFwdAttributesDirect(b, 1, 2, 10);
    auto add = CreatePointwiseAttributes(b,
                                         PointwiseMode::ADD,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         10,
                                         4,
                                         flatbuffers::nullopt,
                                         5);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "conv", DataType::FLOAT, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));
    nodes.push_back(CreateNodeDirect(
        b, "bias_add", DataType::FLOAT, NodeAttributes::PointwiseAttributes, add.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── Conv + bias + relu (3-node fused) ───────────────────────────────────────
// conv.y (uid 10) virtual, bias_add.out (uid 11) virtual, relu.in_0=uid 11, relu.out_0=uid 6

GraphResult buildConvBiasReluGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "w", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 10, "conv_y", DataType::FLOAT, &kStrides, &kDims, true));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 4, "bias", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 11, "bias_out", DataType::FLOAT, &kStrides, &kDims, true));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 6, "out", DataType::FLOAT, &kStrides, &kDims));

    auto conv = CreateConvolutionFwdAttributesDirect(b, 1, 2, 10);
    auto add = CreatePointwiseAttributes(b,
                                         PointwiseMode::ADD,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         10,
                                         4,
                                         flatbuffers::nullopt,
                                         11);
    auto relu = CreatePointwiseAttributes(b,
                                          PointwiseMode::RELU_FWD,
                                          flatbuffers::nullopt,
                                          flatbuffers::nullopt,
                                          flatbuffers::nullopt,
                                          flatbuffers::nullopt,
                                          11,
                                          flatbuffers::nullopt,
                                          flatbuffers::nullopt,
                                          6);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "conv", DataType::FLOAT, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));
    nodes.push_back(CreateNodeDirect(
        b, "bias_add", DataType::FLOAT, NodeAttributes::PointwiseAttributes, add.Union()));
    nodes.push_back(CreateNodeDirect(
        b, "relu", DataType::FLOAT, NodeAttributes::PointwiseAttributes, relu.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── Batchnorm training with runtime PBV scalars ─────────────────────────────
// uids: x=1, y=2, scale=3, bias=4, epsilon=5, prev_mean=8, prev_variance=9, momentum=10
GraphResult buildBatchnormTrainingRuntimePbvGraph()
{
    GraphResult result;
    result.builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        kStrides,
        kDims,
        /*withMeanVariance=*/false,
        /*overrideShapeEnabled=*/false,
        /*runtimeEpsilon=*/true,
        /*withRunningStatsAndMomentum=*/true,
        /*runtimeMomentum=*/true);
    result.graph = GetGraph(result.builder.GetBufferPointer());
    return result;
}

InputTensorMap makeTensorsFromGraph(const GraphResult& gr, const std::vector<int64_t>& uids)
{
    std::unordered_map<int64_t, const TensorAttributes*> attributes;
    for(const auto* tensor : *gr.graph->tensors())
    {
        attributes.emplace(tensor->uid(), tensor);
    }

    InputTensorMap inputs;
    for(const int64_t uid : uids)
    {
        inputs.emplace(uid,
                       hipdnn_test_sdk::detail::createTensorFromAttribute(*attributes.at(uid)));
    }
    return inputs;
}

float scalarValue(const InputTensorMap& inputs, int64_t uid)
{
    return *static_cast<const float*>(inputs.at(uid)->rawHostData());
}

// ── SDPA forward (minimal inputs) ───────────────────────────────────────────

GraphResult buildSdpaFwdGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "q", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "k", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "v", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 4, "o", DataType::FLOAT, &kStrides, &kDims));

    auto sdpa = CreateSdpaAttributes(b, 1, 2, 3, 4);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "sdpa_fwd", DataType::FLOAT, NodeAttributes::SdpaAttributes, sdpa.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── SDPA forward with optional seq_len_q ────────────────────────────────────

GraphResult buildSdpaFwdWithSeqLenQGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "q", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "k", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "v", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 4, "o", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 5, "seq_len_q", DataType::FLOAT, &kStrides, &kDims));

    auto sdpa = CreateSdpaAttributes(b,
                                     1,
                                     2,
                                     3,
                                     4,
                                     flatbuffers::nullopt, // attn_mask
                                     flatbuffers::nullopt, // scale
                                     5); // seq_len_q

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "sdpa_fwd", DataType::FLOAT, NodeAttributes::SdpaAttributes, sdpa.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── SDPA backward standalone ────────────────────────────────────────────────
// O and stats are leaf inputs (not virtual) — filled as free(0, 1)

GraphResult buildSdpaBwdStandaloneGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "q", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "k", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "v", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 4, "o", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 5, "do", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 6, "stats", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 7, "dq", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 8, "dk", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 9, "dv", DataType::FLOAT, &kStrides, &kDims));

    auto bwd = CreateSdpaBackwardAttributes(b, 1, 2, 3, 4, 5, 6, 7, 8, 9);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "sdpa_bwd", DataType::FLOAT, NodeAttributes::SdpaBackwardAttributes, bwd.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── SDPA fwd+bwd fused ─────────────────────────────────────────────────────
// O (uid 10) and stats (uid 11) are virtual inter-node tensors.
// Leaf inputs: Q(1), K(2), V(3) from fwd + dO(5) from bwd.
// Outputs: dQ(7), dK(8), dV(9).

GraphResult buildSdpaFwdBwdFusedGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "q", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "k", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "v", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 10, "o", DataType::FLOAT, &kStrides, &kDims, true));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 11, "stats", DataType::FLOAT, &kStrides, &kDims, true));
    tensors.push_back(CreateTensorAttributesDirect(b, 5, "do", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 7, "dq", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 8, "dk", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 9, "dv", DataType::FLOAT, &kStrides, &kDims));

    auto fwd = CreateSdpaAttributes(b,
                                    1,
                                    2,
                                    3,
                                    10,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    11); // stats_tensor_uid

    auto bwd = CreateSdpaBackwardAttributes(b, 1, 2, 3, 10, 5, 11, 7, 8, 9);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "sdpa_fwd", DataType::FLOAT, NodeAttributes::SdpaAttributes, fwd.Union()));
    nodes.push_back(CreateNodeDirect(
        b, "sdpa_bwd", DataType::FLOAT, NodeAttributes::SdpaBackwardAttributes, bwd.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── MoE grouped matmul (single node, NONE mode) ─────────────────────────────
// Leaf inputs: token(1), weight(2), first_token_offset(3). Output: output(4).
// first_token_offset is declared INT32 to match the real graph; the fill runs
// through ITensor, so the host buffer type makeTensors() picks is immaterial.

GraphResult buildMoeGroupedMatmulGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(b, 1, "token", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "weight", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(
        b, 3, "first_token_offset", DataType::INT32, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 4, "output", DataType::FLOAT, &kStrides, &kDims));

    auto moe = CreateMoeGroupedMatmulAttributes(
        b, 1, 2, 3, flatbuffers::nullopt, flatbuffers::nullopt, 4);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(b,
                                     "moe_grouped_matmul",
                                     DataType::FLOAT,
                                     NodeAttributes::MoeGroupedMatmulAttributes,
                                     moe.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── MoE grouped matmul backward (single node) ───────────────────────────────
// Leaf inputs: doutput(1), token(2), first_token_offset(3). Output: dweight(4).
// first_token_offset is declared INT32 to match the real graph; the fill runs
// through ITensor, so the host buffer type makeTensors() picks is immaterial.

GraphResult buildMoeGroupedMatmulBwdGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(b, 1, "doutput", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "token", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(
        b, 3, "first_token_offset", DataType::INT32, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 4, "dweight", DataType::FLOAT, &kStrides, &kDims));

    auto moeBwd = CreateMoeGroupedMatmulBwdAttributes(b, 1, 2, 3, 4);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(b,
                                     "moe_grouped_matmul_bwd",
                                     DataType::FLOAT,
                                     NodeAttributes::MoeGroupedMatmulBwdAttributes,
                                     moeBwd.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── Matmul with a configurable A data type (single node) ────────────────────
// Leaf inputs: a(1), b(2). Output: c(3). Matmul registers no per-op fills, so
// a's recipe comes from its data type alone.

GraphResult buildMatmulGraph(DataType aType,
                             const std::vector<int64_t>& aDims = kDims,
                             const std::vector<int64_t>& aStrides = kStrides)
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "a", aType, &aStrides, &aDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "b", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "c", DataType::FLOAT, &kStrides, &kDims));

    auto matmul = CreateMatmulAttributes(b, 1, 2, 3);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "matmul", DataType::FLOAT, NodeAttributes::MatmulAttributes, matmul.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── Block-scale dequantize + matmul (MX matmul, 2-node fused) ───────────────
// Leaf inputs: x(1), scale(2), b(3). dequant.y (uid 10) is virtual. Output: c(4).

GraphResult buildBlockScaleDequantizeMatmulGraph(DataType xType)
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", xType, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "scale", DataType::FP8_E8M0, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 10, "dequant_y", DataType::FLOAT, &kStrides, &kDims, true));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "b", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 4, "c", DataType::FLOAT, &kStrides, &kDims));

    auto dequant = CreateBlockScaleDequantizeAttributesDirect(b, 1, 2, 10);
    auto matmul = CreateMatmulAttributes(b, 10, 3, 4);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(b,
                                     "dequant",
                                     DataType::FLOAT,
                                     NodeAttributes::BlockScaleDequantizeAttributes,
                                     dequant.Union()));
    nodes.push_back(CreateNodeDirect(
        b, "matmul", DataType::FLOAT, NodeAttributes::MatmulAttributes, matmul.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// Fills the MX matmul graph and returns the recipes in effect. Pre-set
// `recipes` entries act as test overrides.
InputFillRecipes fillBlockScaleDequantizeMatmul(DataType xType, InputFillRecipes recipes = {})
{
    const auto gr = buildBlockScaleDequantizeMatmulGraph(xType);
    const auto leafUids = gr.leafInputUids({4});
    auto inputs = makeTensorsFromGraph(gr, leafUids);

    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr);
    EXPECT_TRUE(result.filled) << result.reason;
    return recipes;
}

// Distinct |value|s a default fill puts into a typed (unpacked) A operand.
template <typename T>
std::set<float> filledMagnitudes(DataType aType)
{
    // 64K draws keep even the narrowest rounding bin (E3M2's zero, ~0.1% of the
    // range) populated many times over.
    const std::vector<int64_t> dims = {256, 256};
    const std::vector<int64_t> strides = {256, 1};
    const auto gr = buildMatmulGraph(aType, dims, strides);
    const std::vector<int64_t> ownedUids = {1};
    auto inputs = makeTensorsFromGraph(gr, ownedUids);
    InputFillRecipes recipes;

    const auto result = fillInputs(*gr.graph, inputs, ownedUids, recipes, nullptr);
    EXPECT_TRUE(result.filled) << result.reason;

    auto& tensor = *inputs.at(1);
    const auto* values = static_cast<const T*>(tensor.rawHostData());
    std::set<float> magnitudes;
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        magnitudes.insert(std::fabs(static_cast<float>(values[i])));
    }
    return magnitudes;
}

FillResult runFill(const GraphResult& gr, const std::set<int64_t>& outputUids)
{
    const auto leafUids = gr.leafInputUids(outputUids);
    auto inputs = makeTensors(leafUids);
    InputFillRecipes recipes;
    return fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr);
}

// uids: x=1 (float), w=2 (half), bias=4 (float). x and w are sized from the device
// threshold, so they are generated on the device; bias is not.
constexpr int64_t kLargeCols = 128;
constexpr int64_t kLargeRows
    = static_cast<int64_t>(DeviceInputFiller::minElements() / kLargeCols) * 2;

GraphResult buildMixedSizeConvBiasGraph()
{
    GraphResult r;
    auto& b = r.builder;

    const std::vector<int64_t> largeDims = {kLargeRows, kLargeCols};
    const std::vector<int64_t> largeStrides = {kLargeCols, 1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &largeStrides, &largeDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "w", DataType::HALF, &largeStrides, &largeDims));
    tensors.push_back(CreateTensorAttributesDirect(
        b, 10, "conv_y", DataType::FLOAT, &largeStrides, &largeDims, true));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 4, "bias", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 5, "out", DataType::FLOAT, &largeStrides, &largeDims));

    auto conv = CreateConvolutionFwdAttributesDirect(b, 1, 2, 10);
    auto add = CreatePointwiseAttributes(b,
                                         PointwiseMode::ADD,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         flatbuffers::nullopt,
                                         10,
                                         4,
                                         flatbuffers::nullopt,
                                         5);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "conv", DataType::FLOAT, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));
    nodes.push_back(CreateNodeDirect(
        b, "bias_add", DataType::FLOAT, NodeAttributes::PointwiseAttributes, add.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

template <class T>
hipdnn_data_sdk::utilities::TensorBase<T>& typedTensor(InputTensorMap& inputs, int64_t uid)
{
    return dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>&>(*inputs.at(uid));
}

// ── Domain fills: nonzero, zeros, large, subnormal, special ──────────────────

namespace types = hipdnn_data_sdk::types;

template <typename T>
DataType dataTypeOf()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return DataType::DOUBLE;
    }
    else if constexpr(std::is_same_v<T, types::half>)
    {
        return DataType::HALF;
    }
    else if constexpr(std::is_same_v<T, types::bfloat16>)
    {
        return DataType::BFLOAT16;
    }
    else if constexpr(std::is_same_v<T, types::fp8_e4m3>)
    {
        return DataType::FP8_E4M3;
    }
    else if constexpr(std::is_same_v<T, types::fp8_e5m2>)
    {
        return DataType::FP8_E5M2;
    }
    else if constexpr(std::is_same_v<T, types::fp8_e4m3_fnuz>)
    {
        return DataType::FP8_E4M3_FNUZ;
    }
    else if constexpr(std::is_same_v<T, types::fp8_e5m2_fnuz>)
    {
        return DataType::FP8_E5M2_FNUZ;
    }
    else if constexpr(std::is_same_v<T, types::fp8_e8m0>)
    {
        return DataType::FP8_E8M0;
    }
    else if constexpr(std::is_same_v<T, types::fp4_e2m1>)
    {
        return DataType::FP4_E2M1;
    }
    else if constexpr(std::is_same_v<T, types::fp6_e2m3>)
    {
        return DataType::FP6_E2M3;
    }
    else if constexpr(std::is_same_v<T, types::fp6_e3m2>)
    {
        return DataType::FP6_E3M2;
    }
    else if constexpr(std::is_same_v<T, std::int8_t>)
    {
        return DataType::INT8;
    }
    else if constexpr(std::is_same_v<T, std::uint8_t>)
    {
        return DataType::UINT8;
    }
    else if constexpr(std::is_same_v<T, std::int32_t>)
    {
        return DataType::INT32;
    }
    else if constexpr(std::is_same_v<T, std::int64_t>)
    {
        return DataType::INT64;
    }
    else
    {
        static_assert(std::is_same_v<T, bool>, "no DataType for this element type");
        return DataType::BOOLEAN;
    }
}

template <typename T>
double asDouble(T v)
{
    if constexpr(std::is_same_v<T, double> || std::is_integral_v<T>)
    {
        return static_cast<double>(v);
    }
    else
    {
        return static_cast<double>(static_cast<float>(v));
    }
}

template <typename T>
struct SingleFill
{
    FillResult result;
    std::vector<T> values;
};

// Fills one rows x cols input of element type T through fillInputs() with `recipe` and
// per-tensor `seed`, on the host. The tensor is built as a Tensor<T> directly, since the tensor
// factory has no entry for every data type (fp8 fnuz). It starts at 1.0 so an element a fill
// never writes shows up as a 1.0 where no kind produces one. `values` holds the elements in
// row-major order when the fill was accepted.
template <typename T>
SingleFill<T> fillSingle(const FillRecipe& recipe,
                         unsigned int seed = 7,
                         int64_t rows = 64,
                         int64_t cols = 64)
{
    const std::vector<int64_t> dims = {rows, cols};
    const std::vector<int64_t> strides = {cols, 1};
    const auto gr = buildMatmulGraph(dataTypeOf<T>(), dims, strides);
    const std::vector<int64_t> ownedUids = {1};
    InputTensorMap inputs;
    inputs[1] = std::make_shared<hipdnn_data_sdk::utilities::Tensor<T>>(dims, strides);
    inputs.at(1)->fillTensorWithValue(1.0f);

    InputFillRecipes recipes;
    recipes.set(1, recipe).setSeed(1, seed);

    SingleFill<T> out;
    out.result = fillInputs(*gr.graph, inputs, ownedUids, recipes, nullptr);
    if(out.result.filled)
    {
        const auto* data = static_cast<const T*>(inputs.at(1)->rawHostData());
        out.values.assign(data, data + inputs.at(1)->elementCount());
    }
    return out;
}

template <typename T>
bool sameBytes(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// Whether `v` is subnormal in its own type, decided from the encoding where the type is a
// plain IEEE one rather than from the limits the fill itself consults.
template <typename T>
bool isSubnormalValue(T v)
{
    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        return std::fpclassify(v) == FP_SUBNORMAL;
    }
    else if constexpr(std::is_same_v<T, types::half>)
    {
        return (v.data & 0x7C00U) == 0U && (v.data & 0x03FFU) != 0U;
    }
    else if constexpr(std::is_same_v<T, types::bfloat16>)
    {
        return (v.data & 0x7F80U) == 0U && (v.data & 0x007FU) != 0U;
    }
    else
    {
        const double magnitude = std::fabs(asDouble(v));
        return magnitude > 0.0 && magnitude < asDouble(std::numeric_limits<T>::min());
    }
}

// Asserts the fill of `recipe` for T is refused, naming uid 1 and `inReason`.
template <typename T>
void expectUnsupported(const FillRecipe& recipe, const std::string& inReason)
{
    const auto filled = fillSingle<T>(recipe);
    EXPECT_FALSE(filled.result.filled);
    EXPECT_EQ(filled.result.deviceFilled, 0u);
    EXPECT_NE(filled.result.reason.find("uid 1"), std::string::npos) << filled.result.reason;
    EXPECT_NE(filled.result.reason.find(inReason), std::string::npos) << filled.result.reason;
}

// Fills a T input with the subnormal kind and checks the distinct magnitudes drawn are
// exactly `expected`: the type's whole subnormal set, and nothing outside it.
template <typename T>
void expectSubnormalMagnitudes(const std::set<double>& expected)
{
    SCOPED_TRACE(EnumNameDataType(dataTypeOf<T>()));
    const auto filled = fillSingle<T>(FillRecipe::subnormal());
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;
    std::set<double> magnitudes;
    for(const T v : filled.values)
    {
        magnitudes.insert(std::fabs(asDouble(v)));
    }
    EXPECT_EQ(magnitudes, expected);
}

// Fills a bit-packed tensor and an unpacked one of the same shape and strides with the same
// recipe and seed, and checks every packed code equals the unpacked element at that slot.
// The packed codes are decoded here, LSB first, independently of the fill's packing.
template <typename T, typename PackedT>
void expectPackedMatchesUnpacked(const FillRecipe& recipe, std::size_t bitsPerElement)
{
    SCOPED_TRACE(EnumNameDataType(dataTypeOf<T>()));
    const std::vector<int64_t> dims = {4, 8};
    const std::vector<std::vector<int64_t>> layouts = {{8, 1}, {1, 4}};
    for(const auto& strides : layouts)
    {
        std::string layoutName = "strides ";
        layoutName += std::to_string(strides[0]);
        layoutName += ',';
        layoutName += std::to_string(strides[1]);
        SCOPED_TRACE(layoutName);
        const auto gr = buildMatmulGraph(dataTypeOf<T>(), dims, strides);
        const std::vector<int64_t> ownedUids = {1};

        auto unpacked = makeTensorsFromGraph(gr, ownedUids);
        InputTensorMap packed;
        packed[1] = std::make_shared<PackedT>(dims, strides);

        InputFillRecipes unpackedRecipes;
        unpackedRecipes.set(1, recipe).setSeed(1, 11);
        InputFillRecipes packedRecipes;
        packedRecipes.set(1, recipe).setSeed(1, 11);
        ASSERT_TRUE(fillInputs(*gr.graph, unpacked, ownedUids, unpackedRecipes, nullptr).filled);
        ASSERT_TRUE(fillInputs(*gr.graph, packed, ownedUids, packedRecipes, nullptr).filled);

        const auto* values = static_cast<const T*>(unpacked.at(1)->rawHostData());
        const auto* bytes = static_cast<const std::uint8_t*>(packed.at(1)->rawHostData());
        const unsigned mask = (1U << bitsPerElement) - 1U;
        for(std::size_t slot = 0; slot < 32; ++slot)
        {
            unsigned code = 0;
            for(std::size_t bit = 0; bit < bitsPerElement; ++bit)
            {
                const std::size_t position = slot * bitsPerElement + bit;
                code |= ((static_cast<unsigned>(bytes[position / 8]) >> (position % 8)) & 1U)
                        << bit;
            }
            EXPECT_EQ(code, static_cast<unsigned>(values[slot].data) & mask) << "slot " << slot;
        }
    }
}

} // namespace

// ── Test cases ──────────────────────────────────────────────────────────────

TEST(TestFillInputs, SingleConvFwd)
{
    const auto gr = buildConvFwdGraph();
    const auto result = runFill(gr, {3});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestFillInputs, ConvPlusBiasFused)
{
    const auto gr = buildConvBiasGraph();
    const auto result = runFill(gr, {5});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestFillInputs, ConvPlusBiasPlusReluFused)
{
    const auto gr = buildConvBiasReluGraph();
    const auto result = runFill(gr, {6});

    EXPECT_TRUE(result.filled) << result.reason;
}

// Unit seam: verify the fill policy itself. Epsilon is the fixed-value
// path; momentum is the seeded random path and must reproduce across runs.
TEST(TestFillInputs, RuntimePbvScalarsUseFixedAndDeterministicRandomFills)
{
    const auto graph = buildBatchnormTrainingRuntimePbvGraph();
    const std::vector<int64_t> leafUids = {1, 3, 4, 5, 8, 9, 10};

    auto firstInputs = makeTensorsFromGraph(graph, leafUids);
    InputFillRecipes firstRecipes;
    const auto firstResult = fillInputs(*graph.graph, firstInputs, leafUids, firstRecipes, nullptr);
    ASSERT_TRUE(firstResult.filled) << firstResult.reason;

    EXPECT_FLOAT_EQ(scalarValue(firstInputs, 5), 1e-5f);
    const float firstMomentum = scalarValue(firstInputs, 10);
    EXPECT_GE(firstMomentum, 0.0f);
    EXPECT_LE(firstMomentum, 1.0f);

    auto secondInputs = makeTensorsFromGraph(graph, leafUids);
    InputFillRecipes secondRecipes;
    const auto secondResult
        = fillInputs(*graph.graph, secondInputs, leafUids, secondRecipes, nullptr);
    ASSERT_TRUE(secondResult.filled) << secondResult.reason;
    EXPECT_FLOAT_EQ(scalarValue(secondInputs, 10), firstMomentum);
}

TEST(TestFillInputs, SdpaFwdMinimalInputs)
{
    const auto gr = buildSdpaFwdGraph();
    const auto result = runFill(gr, {4});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestFillInputs, SdpaFwdWithSeqLenQFills)
{
    const auto gr = buildSdpaFwdWithSeqLenQGraph();
    const auto result = runFill(gr, {4});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestFillInputs, SdpaBwdStandaloneFills)
{
    const auto gr = buildSdpaBwdStandaloneGraph();
    const auto result = runFill(gr, {7, 8, 9});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestFillInputs, SdpaFwdBwdFusedSucceeds)
{
    const auto gr = buildSdpaFwdBwdFusedGraph();
    const auto result = runFill(gr, {7, 8, 9});

    EXPECT_TRUE(result.filled) << result.reason;
}

// An op missing from applyDefaultFills() makes fillInputs() refuse the whole
// graph, which the harness turns into a GTEST_SKIP. runFill() cannot catch
// that: it discards fillInputs()'s own return value and only inspects the
// recipe table afterward, so this checks fillInputs()'s result directly.
TEST(TestFillInputs, MoeGroupedMatmulFillsAllInputs)
{
    const auto gr = buildMoeGroupedMatmulGraph();
    const auto leafUids = gr.leafInputUids({4});
    auto inputs = makeTensors(leafUids);
    InputFillRecipes recipes;

    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr);

    EXPECT_TRUE(result.filled) << result.reason;
}

// An op missing from applyDefaultFills() makes fillInputs() refuse the whole
// graph, which the harness turns into a GTEST_SKIP. For MoE backward that skip
// is invisible today -- the GPU test stops at engine support first, since no
// provider implements the op -- so it would only surface once a provider lands.
// runFill() cannot catch it: an unregistered op leaves the recipe table
// untouched, and the verdict lives in fillInputs()'s own result.
TEST(TestFillInputs, MoeGroupedMatmulBwdFillsAllInputs)
{
    const auto gr = buildMoeGroupedMatmulBwdGraph();
    const auto leafUids = gr.leafInputUids({4});
    auto inputs = makeTensors(leafUids);
    InputFillRecipes recipes;

    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr);

    EXPECT_TRUE(result.filled) << result.reason;
}

// A device filler only changes how large tensors are generated. Small ones must take
// the host path -- no HIP call, so this also runs where there is no GPU -- and come out
// exactly as a fill without a filler makes them, or the filler would change every small
// tensor's values.
TEST(TestFillInputs, DeviceFillerLeavesSmallTensorsOnTheHostPath)
{
    const auto graph = buildBatchnormTrainingRuntimePbvGraph();
    const std::vector<int64_t> leafUids = {1, 3, 4, 5, 8, 9, 10};

    auto hostInputs = makeTensorsFromGraph(graph, leafUids);
    InputFillRecipes hostRecipes;
    const auto hostResult = fillInputs(*graph.graph, hostInputs, leafUids, hostRecipes, nullptr);
    ASSERT_TRUE(hostResult.filled);
    EXPECT_EQ(hostResult.deviceFilled, 0u);

    auto deviceInputs = makeTensorsFromGraph(graph, leafUids);
    InputFillRecipes deviceRecipes;
    DeviceInputFiller device;
    const auto deviceResult
        = fillInputs(*graph.graph, deviceInputs, leafUids, deviceRecipes, &device);
    ASSERT_TRUE(deviceResult.filled);
    EXPECT_EQ(deviceResult.deviceFilled, 0u);

    for(const int64_t uid : leafUids)
    {
        auto& expected = *hostInputs.at(uid);
        auto& actual = *deviceInputs.at(uid);
        ASSERT_EQ(expected.elementSpace(), actual.elementSpace()) << "uid " << uid;
        EXPECT_EQ(std::memcmp(expected.rawHostData(),
                              actual.rawHostData(),
                              expected.elementSpace() * expected.elementSize()),
                  0)
            << "uid " << uid;
    }
}

// Large tensors are generated on the device and are done by the time fillInputs()
// returns. Dropping `fillPending`, the final wait or the type dispatch each leaves a
// tensor unfilled or still being written, which only a run on a device can see.
TEST(TestFillInputs, DeviceFillerFillsLargeTensorsOnTheDeviceWithinRange)
{
    SKIP_IF_NO_DEVICES();
    if(!DeviceInputFiller::isSupported())
    {
        GTEST_SKIP() << "rocRAND not available. Skipping test.";
    }

    const auto graph = buildMixedSizeConvBiasGraph();
    const std::vector<int64_t> leafUids = {1, 2, 4};

    auto inputs = makeTensorsFromGraph(graph, leafUids);
    InputFillRecipes recipes;
    DeviceInputFiller device;
    const auto result = fillInputs(*graph.graph, inputs, leafUids, recipes, &device);
    ASSERT_TRUE(result.filled) << result.reason;
    EXPECT_EQ(result.deviceFilled, 2u);

    auto& x = typedTensor<float>(inputs, 1);
    auto& w = typedTensor<hipdnn_data_sdk::types::half>(inputs, 2);
    auto& bias = typedTensor<float>(inputs, 4);

    using hipdnn_data_sdk::utilities::MemoryLocation;
    EXPECT_EQ(x.memory().location(), MemoryLocation::DEVICE);
    EXPECT_EQ(w.memory().location(), MemoryLocation::DEVICE);
    EXPECT_NE(bias.memory().location(), MemoryLocation::DEVICE);

    // The non-const access migrates the device-written data to the host.
    const float* xData = x.memory().hostData();
    for(size_t i = 0; i < x.elementSpace(); ++i)
    {
        ASSERT_GE(xData[i], -1.0f) << "x[" << i << "]";
        ASSERT_LE(xData[i], 1.0f) << "x[" << i << "]";
    }

    const auto* wData = w.memory().hostData();
    std::set<float> distinct;
    size_t zeros = 0;
    for(size_t i = 0; i < w.elementSpace(); ++i)
    {
        const auto value = static_cast<float>(wData[i]);
        ASSERT_GE(value, -1.0f) << "w[" << i << "]";
        ASSERT_LE(value, 1.0f) << "w[" << i << "]";
        distinct.insert(value);
        zeros += value == 0.0f ? 1 : 0;
    }

    // The host fill draws a float and rounds it to half, which gives thousands of
    // distinct values over this many elements and essentially never an exact zero.
    // A 16-bit uniform scaled to [-1, 1] gives about 2000 distinct values and a zero
    // about every 2700 elements, which is a different distribution of inputs.
    EXPECT_GT(distinct.size(), 4000u);
    EXPECT_EQ(zeros, 0u);
}

// The threshold is inclusive: a tensor of exactly minElements() goes to the device and one
// element fewer stays on the host, so neither side drifts off the measured crossover.
TEST(TestFillInputs, DeviceFillerThresholdIsInclusiveOfMinElements)
{
    SKIP_IF_NO_DEVICES();
    if(!DeviceInputFiller::isSupported())
    {
        GTEST_SKIP() << "rocRAND not available. Skipping test.";
    }

    const auto elements = static_cast<int64_t>(DeviceInputFiller::minElements());
    const std::vector<int64_t> belowDims = {elements - 1};
    const std::vector<int64_t> atDims = {elements};
    const std::vector<int64_t> strides = {1};
    hipdnn_data_sdk::utilities::Tensor<float> below(belowDims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> at(atDims, strides);

    DeviceInputFiller device;
    const auto recipe = FillRecipe::free(-1.0f, 1.0f);
    EXPECT_FALSE(device.tryFill(below, recipe, 1));
    EXPECT_TRUE(device.tryFill(at, recipe, 1));
    device.waitForFills();
}

// ── Data-type defaults ──────────────────────────────────────────────────────

// A data type added to the schema without a fill decision would otherwise fall
// back to [-1, 1] silently, which is how FP4 ended up reaching 3 of its 8
// magnitudes.
TEST(TestFillInputs, DefaultFillCoversEveryDataType)
{
    for(const auto dataType : EnumValuesDataType())
    {
        if(dataType == DataType::UNSET)
        {
            EXPECT_THROW(defaultFillFor(dataType), std::invalid_argument);
            continue;
        }
        EXPECT_NO_THROW(defaultFillFor(dataType)) << EnumNameDataType(dataType);
    }
}

// The default must reach every representable magnitude, including 0 and the
// largest finite value: 8 for E2M1, 32 for each FP6 format.
TEST(TestFillInputs, NarrowFloatDefaultsReachFullRange)
{
    namespace types = hipdnn_data_sdk::types;

    const auto fp4 = filledMagnitudes<types::fp4_e2m1>(DataType::FP4_E2M1);
    EXPECT_EQ(fp4.size(), 8u);
    EXPECT_FLOAT_EQ(*fp4.rbegin(), 6.0f);

    const auto e2m3 = filledMagnitudes<types::fp6_e2m3>(DataType::FP6_E2M3);
    EXPECT_EQ(e2m3.size(), 32u);
    EXPECT_FLOAT_EQ(*e2m3.rbegin(), 7.5f);

    const auto e3m2 = filledMagnitudes<types::fp6_e3m2>(DataType::FP6_E3M2);
    EXPECT_EQ(e3m2.size(), 32u);
    EXPECT_FLOAT_EQ(*e3m2.rbegin(), 28.0f);
}

// The dtype default reaches inputs of ops that register no fills of their own,
// and is recorded so meta.inputs names the range.
TEST(TestFillInputs, NarrowDtypeOnGenericOpGetsDtypeDefault)
{
    const auto gr = buildMatmulGraph(DataType::FP4_E2M1);
    const auto leafUids = gr.leafInputUids({3});
    auto inputs = makeTensorsFromGraph(gr, leafUids);
    InputFillRecipes recipes;

    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr);

    ASSERT_TRUE(result.filled) << result.reason;
    EXPECT_EQ(recipes.fill(1), FillRecipe::free(-6.0f, 6.0f));
    EXPECT_EQ(recipes.fills().count(2), 0u);
}

// Float graphs keep the generic fill and record nothing, so their recaptured
// meta.inputs stay byte-identical.
TEST(TestFillInputs, FloatGraphRecordsNoDtypeDefaults)
{
    const auto gr = buildConvFwdGraph();
    const auto leafUids = gr.leafInputUids({3});
    auto inputs = makeTensors(leafUids);
    InputFillRecipes recipes;

    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr);

    ASSERT_TRUE(result.filled) << result.reason;
    EXPECT_TRUE(recipes.fills().empty());
}

// ── Block-scale dequantize (MX matmul) ──────────────────────────────────────

// Scale is 2^-floor(log2(amax)) * [0.5, 2] against the operand's range, so the
// dequantized FP4/FP6 block peaks at 3 to 3.75. FP8 keeps today's [-1, 1] /
// [0.5, 2], and its generic operand range is not recorded.
TEST(TestFillInputs, BlockScaleDequantizeScaleNormalized)
{
    struct Case
    {
        DataType xType;
        FillRecipe x;
        FillRecipe scale;
    };
    const std::vector<Case> cases = {
        {DataType::FP4_E2M1, FillRecipe::free(-6.0f, 6.0f), FillRecipe::free(0.125f, 0.5f)},
        {DataType::FP6_E2M3, FillRecipe::free(-7.5f, 7.5f), FillRecipe::free(0.125f, 0.5f)},
        {DataType::FP6_E3M2, FillRecipe::free(-28.0f, 28.0f), FillRecipe::free(0.03125f, 0.125f)},
        {DataType::FP8_E4M3, FillRecipe{}, FillRecipe::free(0.5f, 2.0f)},
        {DataType::FP8_E5M2, FillRecipe{}, FillRecipe::free(0.5f, 2.0f)},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(EnumNameDataType(c.xType));
        const auto recipes = fillBlockScaleDequantizeMatmul(c.xType);
        EXPECT_EQ(recipes.fill(1), c.x);
        EXPECT_EQ(recipes.fills().count(1), c.x != FillRecipe{} ? 1u : 0u);
        EXPECT_EQ(recipes.fill(2), c.scale);
        EXPECT_EQ(recipes.fills().count(3), 0u);
    }
}

// A test override of the operand range renormalizes the scale default.
TEST(TestFillInputs, OverrideOfOperandRenormalizesScale)
{
    InputFillRecipes overrides;
    overrides.set(1, FillRecipe::free(-2.0f, 2.0f));

    const auto recipes = fillBlockScaleDequantizeMatmul(DataType::FP4_E2M1, overrides);

    EXPECT_EQ(recipes.fill(1), FillRecipe::free(-2.0f, 2.0f));
    EXPECT_EQ(recipes.fill(2), FillRecipe::free(0.25f, 1.0f));
}

// Recaptured bundles carry the operand and scale ranges in metadata.inputs.
TEST(TestFillInputs, BlockScaleDequantizeRangesRecordedInMetadata)
{
    const auto inputs = fillBlockScaleDequantizeMatmul(DataType::FP4_E2M1).toJson();

    ASSERT_TRUE(inputs.contains("1"));
    EXPECT_EQ(inputs.at("1").at("kind").get<std::string>(), "free");
    EXPECT_FLOAT_EQ(inputs.at("1").at("lo").get<float>(), -6.0f);
    EXPECT_FLOAT_EQ(inputs.at("1").at("hi").get<float>(), 6.0f);

    ASSERT_TRUE(inputs.contains("2"));
    EXPECT_FLOAT_EQ(inputs.at("2").at("lo").get<float>(), 0.125f);
    EXPECT_FLOAT_EQ(inputs.at("2").at("hi").get<float>(), 0.5f);
}

// ── Domain fills ────────────────────────────────────────────────────────────

template <typename T>
class TestFillInputsDomain : public ::testing::Test
{
};

using DomainTypes = ::testing::Types<float,
                                     double,
                                     hipdnn_data_sdk::types::half,
                                     hipdnn_data_sdk::types::bfloat16,
                                     hipdnn_data_sdk::types::fp8_e4m3,
                                     hipdnn_data_sdk::types::fp8_e5m2>;
TYPED_TEST_SUITE(TestFillInputsDomain, DomainTypes, );

// A seed that is ignored, or a generator reading state it never set, gives the same bytes
// for every seed or different bytes for the same one.
TYPED_TEST(TestFillInputsDomain, EveryKindIsDeterministicPerSeed)
{
    using T = TypeParam;
    const std::vector<FillRecipe> recipes
        = {FillRecipe::nonzero(-1.0f, 1.0f),
           FillRecipe::large(64.0f),
           FillRecipe::subnormal(),
           FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 0.25f)};

    for(const auto& recipe : recipes)
    {
        SCOPED_TRACE(InputFillRecipes::kindToString(recipe.kind));
        const auto first = fillSingle<T>(recipe, 7);
        const auto again = fillSingle<T>(recipe, 7);
        const auto other = fillSingle<T>(recipe, 8);
        ASSERT_TRUE(first.result.filled) << first.result.reason;
        ASSERT_TRUE(again.result.filled) << again.result.reason;
        ASSERT_TRUE(other.result.filled) << other.result.reason;

        EXPECT_TRUE(sameBytes(first.values, again.values));
        EXPECT_FALSE(sameBytes(first.values, other.values));
    }
}

// An element that rounds to zero in a narrow type (a half below 2^-25, an fp8 below half its
// smallest subnormal) is a zero, so a nonzero fill that only tested the float draw would
// leave zeros in the tensor. The range is 1.5 smallest subnormals wide, so a third of the
// draws round to zero and the redraw has to catch them.
TYPED_TEST(TestFillInputsDomain, NonzeroNeverYieldsZeroAfterRoundingToTheDataType)
{
    using T = TypeParam;
    const double smallest = asDouble(std::numeric_limits<T>::denorm_min());
    const float reach
        = std::max(static_cast<float>(1.5 * smallest), std::numeric_limits<float>::denorm_min());

    const auto filled = fillSingle<T>(FillRecipe::nonzero(-reach, reach));
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;

    std::size_t atSmallest = 0;
    for(const T v : filled.values)
    {
        const double d = asDouble(v);
        ASSERT_NE(d, 0.0);
        atSmallest += std::fabs(d) == smallest ? 1U : 0U;
    }
    // Where the range is a normal float, rounding really did reach the smallest subnormal, so
    // the redraw path ran.
    if(smallest > 1e-30)
    {
        EXPECT_GT(atSmallest, 0u);
    }
}

// A range whose every value rounds to zero in the type cannot be made nonzero; drawing
// forever, or returning zeros, hides it.
TYPED_TEST(TestFillInputsDomain, NonzeroOverARangeThatRoundsToZeroIsUnsupported)
{
    using T = TypeParam;
    const double smallest = asDouble(std::numeric_limits<T>::denorm_min());
    const auto reach = static_cast<float>(0.25 * smallest);

    expectUnsupported<T>(FillRecipe::nonzero(-reach, reach), "\"nonzero\"");
    expectUnsupported<T>(FillRecipe::nonzero(0.0f, 0.0f), "\"nonzero\"");
}

// A subnormal produced by rounding a tiny float flushes to zero or rounds up to the smallest
// normal in a narrow type; every element must be a real subnormal of its own type.
TYPED_TEST(TestFillInputsDomain, SubnormalFillsEveryElementBelowTheSmallestNormal)
{
    using T = TypeParam;
    const auto filled = fillSingle<T>(FillRecipe::subnormal());
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;

    std::set<double> distinct;
    std::size_t negatives = 0;
    for(const T v : filled.values)
    {
        ASSERT_TRUE(isSubnormalValue(v)) << asDouble(v);
        distinct.insert(asDouble(v));
        negatives += asDouble(v) < 0.0 ? 1U : 0U;
    }
    // Both signs, and many subnormal codes where the type has many.
    EXPECT_GT(negatives, 0u);
    EXPECT_LT(negatives, filled.values.size());
    EXPECT_GT(distinct.size(), std::numeric_limits<T>::digits > 6 ? 100u : 4u);
}

// A large fill beyond what the type can hold must clamp to its largest finite value, not
// overflow to infinity or NaN; and the magnitudes must sit in the upper half [top/2, top].
TYPED_TEST(TestFillInputsDomain, LargeStaysWithinTheScaleAndTheTypesMaximum)
{
    using T = TypeParam;
    const double maxFinite = asDouble(std::numeric_limits<T>::max());
    // 1/128 covers the rounding of a bound that is not itself representable.
    constexpr double slack = 1.0 / 128.0;

    for(const float scale : {64.0f, 1e30f})
    {
        SCOPED_TRACE(scale);
        const double top = std::min(static_cast<double>(scale), maxFinite);
        const auto filled = fillSingle<T>(FillRecipe::large(scale));
        ASSERT_TRUE(filled.result.filled) << filled.result.reason;

        std::size_t negatives = 0;
        for(const T v : filled.values)
        {
            const double d = asDouble(v);
            ASSERT_TRUE(std::isfinite(d)) << d;
            EXPECT_LE(std::fabs(d), maxFinite);
            EXPECT_LE(std::fabs(d), top * (1.0 + slack));
            EXPECT_GE(std::fabs(d), 0.5 * top * (1.0 - slack));
            negatives += d < 0.0 ? 1U : 0U;
        }
        EXPECT_GT(negatives, 0u);
        EXPECT_LT(negatives, filled.values.size());
    }
}

// A fraction of 0.25 of 4096 elements is exactly 1024 non-finite, at positions fixed by the
// seed and spread over the tensor; the rest are the free draw inside [lo, hi]. A per-element
// coin flip, or a planted value leaking into the free share, breaks the count.
TYPED_TEST(TestFillInputsDomain, SpecialPlantsExactlyTheRequestedShareAtSeededPositions)
{
    using T = TypeParam;
    const auto recipe = FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 0.25f);

    const auto filled = fillSingle<T>(recipe, 7);
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;
    std::set<std::size_t> positions;
    for(std::size_t i = 0; i < filled.values.size(); ++i)
    {
        const double d = asDouble(filled.values[i]);
        if(std::isnan(d))
        {
            positions.insert(i);
        }
        else
        {
            ASSERT_TRUE(std::isfinite(d)) << "index " << i;
            EXPECT_GE(d, -1.0) << "index " << i;
            EXPECT_LE(d, 1.0) << "index " << i;
        }
    }
    EXPECT_EQ(positions.size(), 1024u);

    // Spread: neither half of the tensor holds nearly all of them.
    const auto inFirstHalf = static_cast<std::size_t>(
        std::count_if(positions.begin(), positions.end(), [&](std::size_t p) { return p < 2048; }));
    EXPECT_GT(inFirstHalf, 400u);
    EXPECT_LT(inFirstHalf, 624u);

    // The positions are a function of the seed.
    const auto again = fillSingle<T>(recipe, 7);
    const auto other = fillSingle<T>(recipe, 8);
    ASSERT_TRUE(again.result.filled);
    ASSERT_TRUE(other.result.filled);
    const auto nanPositions = [](const std::vector<T>& values) {
        std::set<std::size_t> result;
        for(std::size_t i = 0; i < values.size(); ++i)
        {
            if(std::isnan(asDouble(values[i])))
            {
                result.insert(i);
            }
        }
        return result;
    };
    EXPECT_EQ(nanPositions(again.values), positions);
    EXPECT_NE(nanPositions(other.values), positions);
}

// An all-zero fill that skipped elements, or wrote something near zero, would leave a
// normalization's variance test unexercised; every element, including strided ones, is zero.
TYPED_TEST(TestFillInputsDomain, ZerosFillsEveryElementWithZero)
{
    using T = TypeParam;
    const auto filled = fillSingle<T>(FillRecipe::zeros());
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;
    for(const T v : filled.values)
    {
        ASSERT_EQ(asDouble(v), 0.0);
    }
}

// The sign of an infinity is the recipe's: +inf must not come out as -inf or NaN.
TYPED_TEST(TestFillInputsDomain, SpecialInfinitiesKeepTheirSign)
{
    using T = TypeParam;
    if constexpr(std::numeric_limits<T>::has_infinity)
    {
        for(const auto special :
            {FillRecipe::SpecialValue::POS_INF, FillRecipe::SpecialValue::NEG_INF})
        {
            const bool positive = special == FillRecipe::SpecialValue::POS_INF;
            const auto filled = fillSingle<T>(FillRecipe::nonFinite(special, 0.5f));
            ASSERT_TRUE(filled.result.filled) << filled.result.reason;
            std::size_t planted = 0;
            for(const T v : filled.values)
            {
                const double d = asDouble(v);
                if(std::isinf(d))
                {
                    EXPECT_EQ(d > 0.0, positive);
                    ++planted;
                }
                else
                {
                    EXPECT_TRUE(std::isfinite(d));
                }
            }
            EXPECT_EQ(planted, filled.values.size() / 2);
        }
    }
    else
    {
        GTEST_SKIP() << "this data type has no infinity";
    }
}

// ── Domain fills: data types the kinds cannot or can only partly use ──────────────────────

// A fraction of 1 is every element, 0 is none, and a positive fraction too small to round to
// one element still plants one: a "special" case must not silently be a plain FREE one.
TEST(TestFillInputs, SpecialFractionBoundsAndTheMinimumOfOne)
{
    const auto nan = FillRecipe::SpecialValue::QUIET_NAN;
    const auto countNan = [](const SingleFill<float>& filled) {
        return std::count_if(
            filled.values.begin(), filled.values.end(), [](float v) { return std::isnan(v); });
    };

    const auto all = fillSingle<float>(FillRecipe::nonFinite(nan, 1.0f), 7, 4, 4);
    const auto none = fillSingle<float>(FillRecipe::nonFinite(nan, 0.0f), 7, 4, 4);
    const auto tiny = fillSingle<float>(FillRecipe::nonFinite(nan, 0.001f), 7, 4, 4);
    ASSERT_TRUE(all.result.filled && none.result.filled && tiny.result.filled);
    EXPECT_EQ(countNan(all), 16);
    EXPECT_EQ(countNan(none), 0);
    EXPECT_EQ(countNan(tiny), 1);
}

// The free share of a special fill honors [lo, hi]; using the default range instead would put
// values outside it.
TEST(TestFillInputs, SpecialFreeShareFollowsTheRecipeRange)
{
    const auto filled = fillSingle<float>(
        FillRecipe::nonFinite(FillRecipe::SpecialValue::POS_INF, 0.25f, 10.0f, 20.0f));
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;
    for(const float v : filled.values)
    {
        if(!std::isinf(v))
        {
            ASSERT_GE(v, 10.0f);
            ASSERT_LE(v, 20.0f);
        }
    }
}

// A value the type cannot hold must be refused, not stored as something else: e4m3 has one
// NaN and no infinity, the sub-byte floats have neither, e8m0 has a NaN only, and an integer
// has no non-finite value at all.
TEST(TestFillInputs, SpecialOnATypeWithoutTheValueIsUnsupported)
{
    const auto nan = FillRecipe::SpecialValue::QUIET_NAN;
    const auto inf = FillRecipe::SpecialValue::POS_INF;
    const auto negInf = FillRecipe::SpecialValue::NEG_INF;

    expectUnsupported<types::fp8_e4m3>(FillRecipe::nonFinite(inf, 0.25f), "on fp8_e4m3");
    expectUnsupported<types::fp8_e4m3>(FillRecipe::nonFinite(negInf, 0.25f), "on fp8_e4m3");
    expectUnsupported<types::fp8_e4m3_fnuz>(FillRecipe::nonFinite(inf, 0.25f), "on fp8_e4m3_fnuz");
    expectUnsupported<types::fp8_e5m2_fnuz>(FillRecipe::nonFinite(inf, 0.25f), "on fp8_e5m2_fnuz");
    expectUnsupported<types::fp8_e8m0>(FillRecipe::nonFinite(inf, 0.25f), "on fp8_e8m0");
    expectUnsupported<types::fp4_e2m1>(FillRecipe::nonFinite(nan, 0.25f), "on fp4_e2m1");
    expectUnsupported<types::fp4_e2m1>(FillRecipe::nonFinite(inf, 0.25f), "on fp4_e2m1");
    expectUnsupported<types::fp6_e2m3>(FillRecipe::nonFinite(nan, 0.25f), "on fp6_e2m3");
    expectUnsupported<types::fp6_e3m2>(FillRecipe::nonFinite(inf, 0.25f), "on fp6_e3m2");
    expectUnsupported<std::int32_t>(FillRecipe::nonFinite(nan, 0.25f), "on int32");
}

// e4m3's single NaN encoding is the value planted, and the types that do have infinity take
// it: the refusals above must not have spread to the supported combinations.
TEST(TestFillInputs, SpecialOnTypesThatHoldTheValueIsAccepted)
{
    const auto nan = FillRecipe::SpecialValue::QUIET_NAN;

    const auto e4m3 = fillSingle<types::fp8_e4m3>(FillRecipe::nonFinite(nan, 0.5f));
    ASSERT_TRUE(e4m3.result.filled) << e4m3.result.reason;
    EXPECT_EQ(std::count_if(e4m3.values.begin(),
                            e4m3.values.end(),
                            [](types::fp8_e4m3 v) { return std::isnan(asDouble(v)); }),
              2048);

    const auto e5m2 = fillSingle<types::fp8_e5m2>(
        FillRecipe::nonFinite(FillRecipe::SpecialValue::NEG_INF, 0.5f));
    ASSERT_TRUE(e5m2.result.filled) << e5m2.result.reason;
    EXPECT_EQ(std::count_if(
                  e5m2.values.begin(),
                  e5m2.values.end(),
                  [](types::fp8_e5m2 v) { return asDouble(v) < 0.0 && std::isinf(asDouble(v)); }),
              2048);

    const auto e8m0 = fillSingle<types::fp8_e8m0>(FillRecipe::nonFinite(nan, 0.5f, 0.5f, 2.0f));
    ASSERT_TRUE(e8m0.result.filled) << e8m0.result.reason;
    EXPECT_EQ(std::count_if(e8m0.values.begin(),
                            e8m0.values.end(),
                            [](types::fp8_e8m0 v) { return std::isnan(asDouble(v)); }),
              2048);
}

// Converting an infinity from float saturates it to the largest finite value in the fp8 types, so
// a planted non-finite value must be written as the type's own encoding. Checked by raw bits
// (OCP and IEEE encodings), with every element planted: half NaN 0x7E00 and +-inf 0x7C00 / 0xFC00,
// bf16 0x7FC0 and 0x7F80 / 0xFF80, e4m3 NaN 0x7F, e5m2 NaN 0x7F and +-inf 0x7C / 0xFC, the fnuz
// formats' NaN 0x80, e8m0 NaN 0xFF.
template <typename T>
static void expectEveryElementHasBits(const FillRecipe& recipe, unsigned int bits)
{
    SCOPED_TRACE(EnumNameDataType(dataTypeOf<T>()));
    const auto filled = fillSingle<T>(recipe, 7, 4, 4);
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;
    ASSERT_EQ(filled.values.size(), 16u);
    for(const T v : filled.values)
    {
        ASSERT_EQ(static_cast<unsigned int>(v.data), bits);
    }
}

TEST(TestFillInputs, SpecialValuesAreWrittenAsTheTypesOwnEncoding)
{
    const auto nan = FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 1.0f);
    const auto inf = FillRecipe::nonFinite(FillRecipe::SpecialValue::POS_INF, 1.0f);
    const auto negInf = FillRecipe::nonFinite(FillRecipe::SpecialValue::NEG_INF, 1.0f);

    expectEveryElementHasBits<types::half>(nan, 0x7E00U);
    expectEveryElementHasBits<types::half>(inf, 0x7C00U);
    expectEveryElementHasBits<types::half>(negInf, 0xFC00U);
    expectEveryElementHasBits<types::bfloat16>(nan, 0x7FC0U);
    expectEveryElementHasBits<types::bfloat16>(inf, 0x7F80U);
    expectEveryElementHasBits<types::bfloat16>(negInf, 0xFF80U);
    expectEveryElementHasBits<types::fp8_e4m3>(nan, 0x7FU);
    expectEveryElementHasBits<types::fp8_e5m2>(nan, 0x7FU);
    expectEveryElementHasBits<types::fp8_e5m2>(inf, 0x7CU);
    expectEveryElementHasBits<types::fp8_e5m2>(negInf, 0xFCU);
    expectEveryElementHasBits<types::fp8_e4m3_fnuz>(nan, 0x80U);
    expectEveryElementHasBits<types::fp8_e5m2_fnuz>(nan, 0x80U);
    expectEveryElementHasBits<types::fp8_e8m0>(nan, 0xFFU);
}

// Integers and e8m0 have no subnormal values; "every element is subnormal" has no answer there,
// so the fill is refused rather than writing zeros or the smallest value.
TEST(TestFillInputs, SubnormalOnATypeWithoutSubnormalsIsUnsupported)
{
    expectUnsupported<std::int32_t>(FillRecipe::subnormal(), "\"subnormal\" on int32");
    expectUnsupported<std::int8_t>(FillRecipe::subnormal(), "\"subnormal\" on int8");
    expectUnsupported<std::uint8_t>(FillRecipe::subnormal(), "\"subnormal\" on uint8");
    expectUnsupported<std::int64_t>(FillRecipe::subnormal(), "\"subnormal\" on int64");
    expectUnsupported<types::fp8_e8m0>(FillRecipe::subnormal(), "\"subnormal\" on fp8_e8m0");
}

// The narrow floats have only a handful of subnormals; the fill must draw those and no value
// outside them (not zero, not the smallest normal), and reach each one. Expected sets are
// k * 2^-n from the formats' definitions: fp4 e2m1 0.5; fp6 e2m3 k/8, k = 1..7; fp6 e3m2 k/16,
// k = 1..3; fp8 e4m3 k * 2^-9, k = 1..7; fp8 e5m2 k * 2^-16, k = 1..3.
TEST(TestFillInputs, SubnormalOnNarrowFormatsDrawsExactlyTheirSubnormalValues)
{
    expectSubnormalMagnitudes<types::fp4_e2m1>({0.5});
    expectSubnormalMagnitudes<types::fp6_e2m3>({0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875});
    expectSubnormalMagnitudes<types::fp6_e3m2>({0.0625, 0.125, 0.1875});
    expectSubnormalMagnitudes<types::fp8_e4m3>({std::ldexp(1.0, -9),
                                                std::ldexp(2.0, -9),
                                                std::ldexp(3.0, -9),
                                                std::ldexp(4.0, -9),
                                                std::ldexp(5.0, -9),
                                                std::ldexp(6.0, -9),
                                                std::ldexp(7.0, -9)});
    expectSubnormalMagnitudes<types::fp8_e5m2>(
        {std::ldexp(1.0, -16), std::ldexp(2.0, -16), std::ldexp(3.0, -16)});
}

// e8m0 holds only powers of two, so it has no zero to fill with.
TEST(TestFillInputs, ZerosOnATypeWithoutZeroIsUnsupported)
{
    expectUnsupported<types::fp8_e8m0>(FillRecipe::zeros(), "\"zeros\" on fp8_e8m0");
}

// Zeros and the fixed kind work on every element type, including those with no typed generator
// (bool), through ITensor; the other kinds there are refused, not silently FREE.
TEST(TestFillInputs, ElementTypesWithoutAGeneratorTakeOnlyZeros)
{
    const auto zeros = fillSingle<bool>(FillRecipe::zeros());
    ASSERT_TRUE(zeros.result.filled) << zeros.result.reason;
    EXPECT_EQ(std::count(zeros.values.begin(), zeros.values.end(), true), 0);

    expectUnsupported<bool>(FillRecipe::nonzero(-1.0f, 1.0f), "no host generator");
    expectUnsupported<bool>(FillRecipe::large(10.0f), "no host generator");
}

// An integer nonzero fill draws from the integers of [lo, hi] without zero: [-1, 1] is
// {-1, 1}, both reached. Truncating a float draw would give mostly zero.
TEST(TestFillInputs, NonzeroOnIntegersSkipsZero)
{
    const auto filled = fillSingle<std::int8_t>(FillRecipe::nonzero(-1.0f, 1.0f));
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;
    const std::set<std::int8_t> seen(filled.values.begin(), filled.values.end());
    EXPECT_EQ(seen, (std::set<std::int8_t>{-1, 1}));

    const auto positive = fillSingle<std::uint8_t>(FillRecipe::nonzero(-3.0f, 3.0f));
    ASSERT_TRUE(positive.result.filled) << positive.result.reason;
    const std::set<std::uint8_t> positiveSeen(positive.values.begin(), positive.values.end());
    EXPECT_EQ(positiveSeen, (std::set<std::uint8_t>{1, 2, 3}));

    expectUnsupported<std::int32_t>(FillRecipe::nonzero(0.0f, 0.0f), "no nonzero integer");
    expectUnsupported<std::int32_t>(FillRecipe::nonzero(0.2f, 0.8f), "no nonzero integer");
}

// Fills a T input with `recipe` and checks every element is in [lowest, highest], nonzero, and
// that more than one value was drawn.
template <typename T>
void expectNonzeroValuesIn(const FillRecipe& recipe, std::int64_t lowest, std::int64_t highest)
{
    SCOPED_TRACE(EnumNameDataType(dataTypeOf<T>()));
    const auto filled = fillSingle<T>(recipe);
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;
    std::set<std::int64_t> seen;
    for(const T v : filled.values)
    {
        const auto value = static_cast<std::int64_t>(v);
        ASSERT_GE(value, lowest);
        ASSERT_LE(value, highest);
        ASSERT_NE(value, 0);
        seen.insert(value);
    }
    EXPECT_GT(seen.size(), 1u);
}

// The integers of [lo, hi] are judged against the type before it clamps: [200, 300] on int8
// used to clamp both ends to 127 and fill everything with 127, a value outside the range. A
// range the type cannot reach, an empty one, and one holding only zero are refused, for every
// integer type.
TEST(TestFillInputs, NonzeroOnIntegersRefusesRangesTheTypeCannotReach)
{
    const std::string why = "no nonzero integer";
    expectUnsupported<std::int8_t>(FillRecipe::nonzero(200.0f, 300.0f), why);
    expectUnsupported<std::int8_t>(FillRecipe::nonzero(-300.0f, -200.0f), why);
    expectUnsupported<std::uint8_t>(FillRecipe::nonzero(256.0f, 300.0f), why);
    expectUnsupported<std::uint8_t>(FillRecipe::nonzero(-300.0f, -5.0f), why);
    expectUnsupported<std::int32_t>(FillRecipe::nonzero(1e10f, 2e10f), why);
    expectUnsupported<std::int32_t>(FillRecipe::nonzero(-2e10f, -1e10f), why);
    expectUnsupported<std::int64_t>(FillRecipe::nonzero(1e30f, 2e30f), why);
    expectUnsupported<std::int64_t>(FillRecipe::nonzero(-2e30f, -1e30f), why);

    expectUnsupported<std::int8_t>(FillRecipe::nonzero(0.0f, 0.0f), why);
    expectUnsupported<std::int8_t>(FillRecipe::nonzero(-0.5f, 0.5f), why);
    expectUnsupported<std::uint8_t>(FillRecipe::nonzero(-0.5f, 0.5f), why);
    expectUnsupported<std::int32_t>(FillRecipe::nonzero(-0.5f, 0.5f), why);
    expectUnsupported<std::int64_t>(FillRecipe::nonzero(0.0f, 0.0f), why);
    expectUnsupported<std::int64_t>(FillRecipe::nonzero(-0.5f, 0.5f), why);
}

// A range straddling the type's edge is cut to the part the type holds, never to the edge alone.
TEST(TestFillInputs, NonzeroOnIntegersDrawsFromTheRangeIntersectedWithTheType)
{
    expectNonzeroValuesIn<std::int8_t>(FillRecipe::nonzero(100.0f, 300.0f), 100, 127);
    expectNonzeroValuesIn<std::int8_t>(FillRecipe::nonzero(-300.0f, -100.0f), -128, -100);
    expectNonzeroValuesIn<std::uint8_t>(FillRecipe::nonzero(200.0f, 400.0f), 200, 255);
    expectNonzeroValuesIn<std::int32_t>(FillRecipe::nonzero(2e9f, 3e9f),
                                        static_cast<std::int64_t>(2e9f),
                                        std::numeric_limits<std::int32_t>::max());
    expectNonzeroValuesIn<std::int64_t>(FillRecipe::nonzero(5e18f, 1e19f),
                                        static_cast<std::int64_t>(5e18f),
                                        std::numeric_limits<std::int64_t>::max());
}

// Ordinary ranges are untouched by the intersection and never yield zero, with both signs
// reached when the range has both.
TEST(TestFillInputs, NonzeroOnIntegersOrdinaryRangesNeverYieldZero)
{
    expectNonzeroValuesIn<std::int8_t>(FillRecipe::nonzero(-5.0f, 5.0f), -5, 5);
    expectNonzeroValuesIn<std::int32_t>(FillRecipe::nonzero(-1000.0f, 7.0f), -1000, 7);
    expectNonzeroValuesIn<std::int64_t>(FillRecipe::nonzero(-3.0f, 2.0f), -3, 2);
    expectNonzeroValuesIn<std::uint8_t>(FillRecipe::nonzero(0.0f, 9.0f), 1, 9);
    expectNonzeroValuesIn<std::int32_t>(FillRecipe::nonzero(10.0f, 20.0f), 10, 20);
}

// An integer large fill is in [ceil(top / 2), top] with top at most the type's maximum: an
// int8 scale of 1000 is [64, 127], a uint8 scale of 300 is [128, 255], and a scale below 1 has
// no integer.
TEST(TestFillInputs, LargeOnIntegersClampsToTheTypeMaximum)
{
    const auto wide = fillSingle<std::int8_t>(FillRecipe::large(1000.0f));
    ASSERT_TRUE(wide.result.filled) << wide.result.reason;
    for(const std::int8_t v : wide.values)
    {
        ASSERT_GE(std::abs(static_cast<int>(v)), 64);
        ASSERT_LE(std::abs(static_cast<int>(v)), 127);
    }

    const auto unsignedWide = fillSingle<std::uint8_t>(FillRecipe::large(300.0f));
    ASSERT_TRUE(unsignedWide.result.filled) << unsignedWide.result.reason;
    for(const std::uint8_t v : unsignedWide.values)
    {
        ASSERT_GE(v, 128);
    }

    expectUnsupported<std::int32_t>(FillRecipe::large(0.5f), "scale >= 1");
}

// "At most the type's maximum" with a literal: half's is 65504, e4m3's 448. A scale beyond
// them is clamped, so every element is finite, and the draw reaches the upper end.
TEST(TestFillInputs, LargeBeyondTheTypeMaximumClampsToIt)
{
    const auto narrow = fillSingle<types::half>(FillRecipe::large(1e30f));
    ASSERT_TRUE(narrow.result.filled) << narrow.result.reason;
    double widest = 0.0;
    for(const auto v : narrow.values)
    {
        const double d = std::fabs(asDouble(v));
        ASSERT_TRUE(std::isfinite(d));
        ASSERT_GE(d, 32752.0);
        ASSERT_LE(d, 65504.0);
        widest = std::max(widest, d);
    }
    EXPECT_GT(widest, 60000.0);

    const auto e4m3 = fillSingle<types::fp8_e4m3>(FillRecipe::large(1e30f));
    ASSERT_TRUE(e4m3.result.filled) << e4m3.result.reason;
    for(const auto v : e4m3.values)
    {
        const double d = std::fabs(asDouble(v));
        ASSERT_TRUE(std::isfinite(d));
        ASSERT_GE(d, 224.0);
        ASSERT_LE(d, 448.0);
    }
}

// Unusable parameters set programmatically (parsing rejects them in JSON) are refused by the
// fill itself: a zero or NaN scale, a fraction outside [0, 1], an unordered range.
TEST(TestFillInputs, UnusableRecipeParametersAreUnsupported)
{
    expectUnsupported<float>(FillRecipe::large(0.0f), "\"large\"");
    expectUnsupported<float>(FillRecipe::large(std::numeric_limits<float>::quiet_NaN()),
                             "\"large\"");
    expectUnsupported<float>(FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 1.5f),
                             "\"special\"");
    expectUnsupported<float>(FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, -0.1f),
                             "\"special\"");
    expectUnsupported<float>(FillRecipe::nonzero(1.0f, -1.0f), "\"nonzero\"");
}

// Sub-byte graphs are filled once unpacked and once bit-packed and the two must hold the same
// values; a packed fill that diverged would compare a different input on each side. Checked
// for a row-major and a column-major dense layout, where the packed slot of a coordinate
// differs from its logical index.
TEST(TestFillInputs, PackedSubByteTensorsHoldTheSameValuesAsUnpackedOnes)
{
    using hipdnn_data_sdk::utilities::PackedFp4Tensor;
    using hipdnn_data_sdk::utilities::PackedFp6Tensor;

    const std::vector<FillRecipe> recipes = {FillRecipe::nonzero(-1.0f, 1.0f),
                                             FillRecipe::zeros(),
                                             FillRecipe::large(100.0f),
                                             FillRecipe::subnormal()};
    for(const auto& recipe : recipes)
    {
        SCOPED_TRACE(InputFillRecipes::kindToString(recipe.kind));
        expectPackedMatchesUnpacked<types::fp4_e2m1, PackedFp4Tensor>(recipe, 4);
        expectPackedMatchesUnpacked<types::fp6_e2m3, PackedFp6Tensor<types::fp6_e2m3>>(recipe, 6);
        expectPackedMatchesUnpacked<types::fp6_e3m2, PackedFp6Tensor<types::fp6_e3m2>>(recipe, 6);
    }
}

// A packed tensor of a type that cannot hold the kind is refused like its unpacked twin.
TEST(TestFillInputs, PackedSubByteTensorOfAnUnsupportedKindIsRefused)
{
    const std::vector<int64_t> dims = {4, 8};
    const std::vector<int64_t> strides = {8, 1};
    const auto gr = buildMatmulGraph(DataType::FP4_E2M1, dims, strides);
    InputTensorMap packed;
    packed[1] = std::make_shared<hipdnn_data_sdk::utilities::PackedFp4Tensor>(dims, strides);
    InputFillRecipes recipes;
    recipes.setNonFinite(1, FillRecipe::SpecialValue::QUIET_NAN, 0.25f);

    const auto result = fillInputs(*gr.graph, packed, {1}, recipes, nullptr);

    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("fp4_e2m1"), std::string::npos) << result.reason;
}

// ── Domain fills: the device and the existing kinds ─────────────────────────────────────────

// rocRAND can only draw uniform floats, so every non-FREE kind must come out of the host
// generator even for a tensor large enough to be sent to the device; a decline that failed
// would put the new kinds through a path that cannot make them (or, with no device, throw). The
// bytes must be the ones a host-only fill makes. Runs without a device: a decline never
// reaches HIP.
TEST(TestFillInputs, DeviceFillerDeclinesEveryNewKindAndTheHostFillsIt)
{
    const std::vector<int64_t> dims = {256, 256};
    const std::vector<int64_t> strides = {256, 1};
    ASSERT_GE(256u * 256u, DeviceInputFiller::minElements());

    const std::vector<FillRecipe> recipes
        = {FillRecipe::nonzero(-1.0f, 1.0f),
           FillRecipe::zeros(),
           FillRecipe::large(64.0f),
           FillRecipe::subnormal(),
           FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 0.25f)};
    for(const auto& recipe : recipes)
    {
        SCOPED_TRACE(InputFillRecipes::kindToString(recipe.kind));
        const auto gr = buildMatmulGraph(DataType::FLOAT, dims, strides);
        const std::vector<int64_t> ownedUids = {1};
        auto hostInputs = makeTensorsFromGraph(gr, ownedUids);
        auto deviceInputs = makeTensorsFromGraph(gr, ownedUids);
        InputFillRecipes hostRecipes;
        hostRecipes.set(1, recipe).setSeed(1, 5);
        InputFillRecipes deviceRecipes;
        deviceRecipes.set(1, recipe).setSeed(1, 5);

        DeviceInputFiller device;
        EXPECT_FALSE(device.tryFill(*deviceInputs.at(1), recipe, 5));

        const auto hostResult = fillInputs(*gr.graph, hostInputs, ownedUids, hostRecipes, nullptr);
        const auto deviceResult
            = fillInputs(*gr.graph, deviceInputs, ownedUids, deviceRecipes, &device);
        ASSERT_TRUE(hostResult.filled) << hostResult.reason;
        ASSERT_TRUE(deviceResult.filled) << deviceResult.reason;
        EXPECT_EQ(deviceResult.deviceFilled, 0u);

        auto& expected = *hostInputs.at(1);
        auto& actual = *deviceInputs.at(1);
        ASSERT_EQ(expected.elementSpace(), actual.elementSpace());
        EXPECT_EQ(std::memcmp(expected.rawHostData(),
                              actual.rawHostData(),
                              expected.elementSpace() * expected.elementSize()),
                  0);
    }
}

// FREE goes through the tensor's own draw, not the typed generators: its bytes for a given seed
// are what recorded runs and the device-off reproduction rely on.
TEST(TestFillInputs, FreeFillStillMatchesTheTensorsOwnRandomFill)
{
    const auto filled = fillSingle<float>(FillRecipe::free(-2.0f, 3.0f), 9);
    ASSERT_TRUE(filled.result.filled) << filled.result.reason;

    const std::vector<int64_t> dims = {64, 64};
    const std::vector<int64_t> strides = {64, 1};
    hipdnn_data_sdk::utilities::Tensor<float> direct(dims, strides);
    direct.fillTensorWithRandomValues(-2.0f, 3.0f, 9);
    const auto* data = static_cast<const float*>(direct.rawHostData());
    const std::vector<float> expected(data, data + direct.elementCount());

    EXPECT_TRUE(sameBytes(filled.values, expected));
}

// The block-scale normalization reads the operand's range through recipeMagnitude(); each new
// kind must report the largest finite magnitude it can produce, or the scale it derives is
// off. For an x of e5m2 (every kind is supported there): zeros and subnormal are ~0, so the
// scale stays at [0.5, 2]; large(8) peaks at 8, nonzero(-4, 4) and special's range at 4 and 2.
TEST(TestFillInputs, OperandRecipeKindsFeedTheScaleNormalization)
{
    struct Case
    {
        FillRecipe x;
        FillRecipe scale;
    };
    const std::vector<Case> cases = {
        {FillRecipe::zeros(), FillRecipe::free(0.5f, 2.0f)},
        {FillRecipe::subnormal(), FillRecipe::free(0.5f, 2.0f)},
        {FillRecipe::large(8.0f), FillRecipe::free(0.0625f, 0.25f)},
        {FillRecipe::nonzero(-4.0f, 4.0f), FillRecipe::free(0.125f, 0.5f)},
        {FillRecipe::nonFinite(FillRecipe::SpecialValue::QUIET_NAN, 0.1f, -2.0f, 2.0f),
         FillRecipe::free(0.25f, 1.0f)},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(InputFillRecipes::kindToString(c.x.kind));
        InputFillRecipes overrides;
        overrides.set(1, c.x);
        const auto recipes = fillBlockScaleDequantizeMatmul(DataType::FP8_E5M2, overrides);
        EXPECT_EQ(recipes.fill(2), c.scale);
    }
}
// NOLINTEND(readability-identifier-naming)
