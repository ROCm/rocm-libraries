// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input-init/FillInputs.hpp"
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests;

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

// NOLINTEND(readability-identifier-naming)
