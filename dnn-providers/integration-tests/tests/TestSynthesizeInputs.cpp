// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/golden/input_init/SynthesizeInputs.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::golden;

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

// ── SDPA forward (no structured optionals) ──────────────────────────────────

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

// ── SDPA forward with structured seq_len_q ──────────────────────────────────

GraphResult buildSdpaFwdWithStructuredGraph()
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
// O and stats are leaf inputs (not virtual) → DERIVED → refuses

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

SynthesisResult runSynthesis(const GraphResult& gr, const std::set<int64_t>& outputUids)
{
    const auto leafUids = gr.leafInputUids(outputUids);
    auto inputs = makeTensors(leafUids);
    std::mt19937 rng(42);

    SynthesisTracker tracker(leafUids, inputs);
    for(uint32_t i = 0; i < gr.graph->nodes()->size(); ++i)
    {
        const SynthesisResult nodeResult
            = synthesizeNodeInputs(*gr.graph->nodes()->Get(i), tracker, rng);
        if(!nodeResult.filled)
        {
            return nodeResult;
        }
    }
    return tracker.finish("test");
}

} // namespace

// ── Test cases ──────────────────────────────────────────────────────────────

TEST(TestSynthesizeInputs, SingleConvFwd)
{
    const auto gr = buildConvFwdGraph();
    const auto result = runSynthesis(gr, {3});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestSynthesizeInputs, ConvPlusBiasFused)
{
    const auto gr = buildConvBiasGraph();
    const auto result = runSynthesis(gr, {5});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestSynthesizeInputs, ConvPlusBiasPlusReluFused)
{
    const auto gr = buildConvBiasReluGraph();
    const auto result = runSynthesis(gr, {6});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestSynthesizeInputs, SdpaFwdNoStructuredOptionals)
{
    const auto gr = buildSdpaFwdGraph();
    const auto result = runSynthesis(gr, {4});

    EXPECT_TRUE(result.filled) << result.reason;
}

TEST(TestSynthesizeInputs, SdpaFwdWithStructuredInputRefuses)
{
    const auto gr = buildSdpaFwdWithStructuredGraph();
    const auto result = runSynthesis(gr, {4});

    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("seq_len_q"), std::string::npos);
    EXPECT_NE(result.reason.find("structured"), std::string::npos);
}

TEST(TestSynthesizeInputs, SdpaBwdStandaloneRefusesDerived)
{
    const auto gr = buildSdpaBwdStandaloneGraph();
    const auto result = runSynthesis(gr, {7, 8, 9});

    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("derived"), std::string::npos);
}

TEST(TestSynthesizeInputs, SdpaFwdBwdFusedSucceeds)
{
    const auto gr = buildSdpaFwdBwdFusedGraph();
    const auto result = runSynthesis(gr, {7, 8, 9});

    EXPECT_TRUE(result.filled) << result.reason;
}

// NOLINTEND(readability-identifier-naming)
