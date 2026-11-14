// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <nlohmann/json.hpp>

#include "BundleFixtureFiles.hpp"
#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/input-init/FillInputs.hpp"
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
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
    if(f.kind == FillRecipe::Kind::VALUES)
    {
        *os << "VALUES(" << (f.repeat ? "repeat" : "exact") << ")";
        return;
    }
    *os << "FREE[" << f.lo << ", " << f.hi << "]";
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

// ── SDPA backward at attention shapes ───────────────────────────────────────
// Q, O, dO, dQ [1, 2, 4, 8]; K, V, dK, dV [1, 2, 6, 8]. uids: q=1, k=2, v=3, o=4,
// do=5, stats=6, dq=7, dk=8, dv=9, runtime scale=10, additive mask=11.

constexpr int64_t kSdpaHeads = 2;
constexpr int64_t kSdpaSeqQ = 4;
constexpr int64_t kSdpaSeqKv = 6;
constexpr int64_t kSdpaHeadDim = 8;

struct SdpaBwdOptions
{
    DataType ioType = DataType::FLOAT;
    bool rank3Stats = false;
    std::optional<float> attnScale;
    bool runtimeScale = false;
    bool causal = false;
    bool attnMask = false;
    std::optional<float> dropoutProbability;
    bool seed = false;
    bool offset = false;
    bool dropoutMask = false;
    bool dropoutScale = false;
    bool dropoutScaleInv = false;
};

GraphResult buildSdpaBwdGraph(const SdpaBwdOptions& opts)
{
    using hipdnn_data_sdk::utilities::generateStrides;

    GraphResult r;
    auto& b = r.builder;

    const std::vector<int64_t> qDims = {1, kSdpaHeads, kSdpaSeqQ, kSdpaHeadDim};
    const std::vector<int64_t> kvDims = {1, kSdpaHeads, kSdpaSeqKv, kSdpaHeadDim};
    const auto statsDims = opts.rank3Stats ? std::vector<int64_t>{1, kSdpaHeads, kSdpaSeqQ}
                                           : std::vector<int64_t>{1, kSdpaHeads, kSdpaSeqQ, 1};
    const std::vector<int64_t> scalarDims = {1};
    const std::vector<int64_t> maskDims = {1, 1, kSdpaSeqQ, kSdpaSeqKv};
    const auto qStrides = generateStrides(qDims);
    const auto kvStrides = generateStrides(kvDims);
    const auto statsStrides = generateStrides(statsDims);
    const auto maskStrides = generateStrides(maskDims);
    const auto io = opts.ioType;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "q", io, &qStrides, &qDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "k", io, &kvStrides, &kvDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "v", io, &kvStrides, &kvDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 4, "o", io, &qStrides, &qDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 5, "do", io, &qStrides, &qDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 6, "stats", DataType::FLOAT, &statsStrides, &statsDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 7, "dq", io, &qStrides, &qDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 8, "dk", io, &kvStrides, &kvDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 9, "dv", io, &kvStrides, &kvDims));
    if(opts.runtimeScale)
    {
        tensors.push_back(CreateTensorAttributesDirect(b,
                                                       10,
                                                       "scale",
                                                       DataType::FLOAT,
                                                       &scalarDims,
                                                       &scalarDims,
                                                       /*virtual_=*/false,
                                                       TensorValue::NONE,
                                                       /*value=*/0,
                                                       /*is_runtime_pass_by_value=*/true));
    }
    if(opts.attnMask)
    {
        tensors.push_back(
            CreateTensorAttributesDirect(b, 11, "mask", DataType::FLOAT, &maskStrides, &maskDims));
    }
    if(opts.seed)
    {
        tensors.push_back(
            CreateTensorAttributesDirect(b, 12, "seed", DataType::INT64, &scalarDims, &scalarDims));
    }
    if(opts.offset)
    {
        tensors.push_back(CreateTensorAttributesDirect(
            b, 13, "offset", DataType::INT64, &scalarDims, &scalarDims));
    }

    if(opts.dropoutMask)
    {
        tensors.push_back(CreateTensorAttributesDirect(
            b, 14, "dropout_mask", DataType::FLOAT, &maskStrides, &maskDims));
    }
    if(opts.dropoutScale)
    {
        tensors.push_back(CreateTensorAttributesDirect(
            b, 15, "dropout_scale", DataType::FLOAT, &scalarDims, &scalarDims));
    }
    if(opts.dropoutScaleInv)
    {
        tensors.push_back(CreateTensorAttributesDirect(
            b, 16, "dropout_scale_inv", DataType::FLOAT, &scalarDims, &scalarDims));
    }

    SdpaBackwardAttributesBuilder attrs(b);
    attrs.add_q_tensor_uid(1);
    attrs.add_k_tensor_uid(2);
    attrs.add_v_tensor_uid(3);
    attrs.add_o_tensor_uid(4);
    attrs.add_do_tensor_uid(5);
    attrs.add_stats_tensor_uid(6);
    attrs.add_dq_tensor_uid(7);
    attrs.add_dk_tensor_uid(8);
    attrs.add_dv_tensor_uid(9);
    if(opts.runtimeScale)
    {
        attrs.add_scale_tensor_uid(10);
    }
    if(opts.attnScale.has_value())
    {
        attrs.add_attn_scale_value(*opts.attnScale);
    }
    if(opts.causal)
    {
        attrs.add_right_bound(0);
    }
    if(opts.attnMask)
    {
        attrs.add_attn_mask_tensor_uid(11);
    }
    if(opts.dropoutProbability.has_value())
    {
        attrs.add_dropout_probability(*opts.dropoutProbability);
    }
    if(opts.seed)
    {
        attrs.add_seed_tensor_uid(12);
    }
    if(opts.offset)
    {
        attrs.add_offset_tensor_uid(13);
    }
    if(opts.dropoutMask)
    {
        attrs.add_dropout_mask_tensor_uid(14);
    }
    if(opts.dropoutScale)
    {
        attrs.add_dropout_scale_tensor_uid(15);
    }
    if(opts.dropoutScaleInv)
    {
        attrs.add_dropout_scale_inv_tensor_uid(16);
    }
    auto bwd = attrs.Finish();

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(
        CreateNodeDirect(b, "sdpa_bwd", io, NodeAttributes::SdpaBackwardAttributes, bwd.Union()));

    auto graph
        = CreateGraphDirect(b, "test", io, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// Fills the graph with O and stats forced to 0.5, which no forward of these
// shapes produces everywhere, so a derived value is told apart from a filled one.
FillResult fillSdpaBwd(const GraphResult& gr,
                       InputTensorMap& inputs,
                       const std::vector<int64_t>& ownedUids,
                       InputFillRecipes recipes = {})
{
    recipes.set(4, FillRecipe::fixed(0.5f));
    recipes.set(6, FillRecipe::fixed(0.5f));
    return fillInputs(*gr.graph, inputs, ownedUids, recipes, nullptr);
}

InputTensorMap fillSdpaBwdWithFixedOAndStats(const GraphResult& gr, InputFillRecipes recipes = {})
{
    const auto leafUids = gr.leafInputUids({7, 8, 9});
    auto inputs = makeTensorsFromGraph(gr, leafUids);
    const auto result = fillSdpaBwd(gr, inputs, leafUids, std::move(recipes));
    EXPECT_TRUE(result.filled) << result.reason;
    return inputs;
}

// Recomputes the forward from the filled Q, K and V (and the [1, 1, Sq, Skv] float
// mask, if any): every row of exp(Q K^T * scale + mask - lse) sums to 1 over its
// unmasked keys, and O is that row's weighted sum of V. A causal graph masks keys
// after the query (top-left aligned).
template <class T>
void expectOAndStatsAreForwardOfQkv(
    InputTensorMap& inputs, float scale, bool causal, float oTolerance, const float* mask = nullptr)
{
    const auto& q = typedTensor<T>(inputs, 1);
    const auto& k = typedTensor<T>(inputs, 2);
    const auto& v = typedTensor<T>(inputs, 3);
    const auto& o = typedTensor<T>(inputs, 4);
    // [1, H, Sq, 1], packed.
    const auto* lse = static_cast<const float*>(inputs.at(6)->rawHostData());

    for(int64_t h = 0; h < kSdpaHeads; ++h)
    {
        for(int64_t sq = 0; sq < kSdpaSeqQ; ++sq)
        {
            const float rowLse = lse[(h * kSdpaSeqQ) + sq];
            float rowSum = 0.0f;
            std::vector<float> expectedO(kSdpaHeadDim, 0.0f);
            for(int64_t skv = 0; skv < kSdpaSeqKv; ++skv)
            {
                if(causal && skv > sq)
                {
                    continue;
                }
                float score = 0.0f;
                for(int64_t d = 0; d < kSdpaHeadDim; ++d)
                {
                    score += static_cast<float>(q.getHostValue(0, h, sq, d))
                             * static_cast<float>(k.getHostValue(0, h, skv, d));
                }
                const float maskTerm = mask != nullptr ? mask[(sq * kSdpaSeqKv) + skv] : 0.0f;
                const float p = std::exp((score * scale) + maskTerm - rowLse);
                rowSum += p;
                for(int64_t d = 0; d < kSdpaHeadDim; ++d)
                {
                    expectedO[static_cast<size_t>(d)]
                        += p * static_cast<float>(v.getHostValue(0, h, skv, d));
                }
            }
            EXPECT_NEAR(rowSum, 1.0f, 1e-4f) << "h=" << h << " sq=" << sq;
            for(int64_t d = 0; d < kSdpaHeadDim; ++d)
            {
                EXPECT_NEAR(static_cast<float>(o.getHostValue(0, h, sq, d)),
                            expectedO[static_cast<size_t>(d)],
                            oTolerance)
                    << "h=" << h << " sq=" << sq << " d=" << d;
            }
        }
    }
}

void expectAllEqual(hipdnn_data_sdk::utilities::ITensor& tensor, float value)
{
    const auto* data = static_cast<const float*>(tensor.rawHostData());
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        ASSERT_EQ(data[i], value) << "element " << i;
    }
}

// O and stats are replaced by the forward of Q, K and V even over an explicit
// override, which is what a bundle's metadata "inputs" entry for them becomes.
// Causal and scaled, with [B, H, Sq, 1] stats, as the SdpaBackward sweeps declare them.
template <class T>
void expectCausalScaledOAndStatsAreDerived(DataType ioType, float oTolerance)
{
    SdpaBwdOptions opts;
    opts.ioType = ioType;
    opts.attnScale = 0.35f;
    opts.causal = true;
    const auto gr = buildSdpaBwdGraph(opts);
    auto inputs = fillSdpaBwdWithFixedOAndStats(gr);

    expectOAndStatsAreForwardOfQkv<T>(inputs, 0.35f, /*causal=*/true, oTolerance);
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

TEST(TestFillInputs, SdpaBwdOAndStatsAreTheForwardOfQkv)
{
    // O is stored as bf16: 8 significant bits on values below 1.
    expectCausalScaledOAndStatsAreDerived<hipdnn_data_sdk::types::bfloat16>(DataType::BFLOAT16,
                                                                            1e-2f);
}

TEST(TestFillInputs, SdpaBwdHalfOAndStatsAreTheForwardOfQkv)
{
    // O is stored as fp16: 11 significant bits on values below 1.
    expectCausalScaledOAndStatsAreDerived<hipdnn_data_sdk::types::half>(DataType::HALF, 1e-3f);
}

// A runtime scale is read from its filled value, not taken as 1.
TEST(TestFillInputs, SdpaBwdDerivationUsesFilledRuntimeScale)
{
    SdpaBwdOptions opts;
    opts.runtimeScale = true;
    const auto gr = buildSdpaBwdGraph(opts);
    InputFillRecipes recipes;
    recipes.set(10, FillRecipe::fixed(0.25f));
    auto inputs = fillSdpaBwdWithFixedOAndStats(gr, recipes);

    expectOAndStatsAreForwardOfQkv<float>(inputs, 0.25f, /*causal=*/false, /*oTolerance=*/1e-5f);
}

// The filled additive mask enters the forward, broadcast over the batch and heads.
TEST(TestFillInputs, SdpaBwdDerivationAddsTheAttnMask)
{
    SdpaBwdOptions opts;
    opts.attnMask = true;
    const auto gr = buildSdpaBwdGraph(opts);
    auto inputs = fillSdpaBwdWithFixedOAndStats(gr);

    expectOAndStatsAreForwardOfQkv<float>(inputs,
                                          1.0f,
                                          /*causal=*/false,
                                          /*oTolerance=*/1e-5f,
                                          static_cast<const float*>(inputs.at(11)->rawHostData()));
}

TEST(TestFillInputs, SdpaBwdWithDropoutIsRefused)
{
    SdpaBwdOptions opts;
    opts.dropoutProbability = 0.1f;
    const auto gr = buildSdpaBwdGraph(opts);
    const auto leafUids = gr.leafInputUids({7, 8, 9});
    auto inputs = makeTensorsFromGraph(gr, leafUids);

    const auto result = fillSdpaBwd(gr, inputs, leafUids);

    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("dropout"), std::string::npos) << result.reason;
}

// A zero dropout probability is no dropout, so O and stats are still derived.
TEST(TestFillInputs, SdpaBwdWithZeroDropoutIsDerived)
{
    SdpaBwdOptions opts;
    opts.dropoutProbability = 0.0f;
    const auto gr = buildSdpaBwdGraph(opts);
    auto inputs = fillSdpaBwdWithFixedOAndStats(gr);

    expectOAndStatsAreForwardOfQkv<float>(inputs, 1.0f, /*causal=*/false, /*oTolerance=*/1e-5f);
}

// A seed and offset the zero-probability node names are not read, so O and stats are still derived.
TEST(TestFillInputs, SdpaBwdWithZeroDropoutAndSeedOffsetIsDerived)
{
    SdpaBwdOptions opts;
    opts.dropoutProbability = 0.0f;
    opts.seed = true;
    opts.offset = true;
    const auto gr = buildSdpaBwdGraph(opts);
    auto inputs = fillSdpaBwdWithFixedOAndStats(gr);

    expectOAndStatsAreForwardOfQkv<float>(inputs, 1.0f, /*causal=*/false, /*oTolerance=*/1e-5f);
}

// Either of a seed and an offset alone is not read at probability zero either.
TEST(TestFillInputs, SdpaBwdWithZeroDropoutAndOneOfSeedOrOffsetIsDerived)
{
    for(const bool seed : {true, false})
    {
        SCOPED_TRACE(seed ? "seed only" : "offset only");
        SdpaBwdOptions opts;
        opts.dropoutProbability = 0.0f;
        opts.seed = seed;
        opts.offset = !seed;
        const auto gr = buildSdpaBwdGraph(opts);
        auto inputs = fillSdpaBwdWithFixedOAndStats(gr);

        expectOAndStatsAreForwardOfQkv<float>(inputs, 1.0f, /*causal=*/false, /*oTolerance=*/1e-5f);
    }
}

// Probability zero excuses the probability, seed and offset only: a dropout mask, scale or
// scale-inverse tensor the node names still means dropout, which the fill does not model.
TEST(TestFillInputs, SdpaBwdWithZeroDropoutAndADropoutMaskOrScaleIsRefused)
{
    for(const std::string tensor : {"dropout mask", "dropout scale", "dropout scale inverse"})
    {
        SCOPED_TRACE(tensor);
        SdpaBwdOptions opts;
        opts.dropoutProbability = 0.0f;
        opts.dropoutMask = tensor == "dropout mask";
        opts.dropoutScale = tensor == "dropout scale";
        opts.dropoutScaleInv = tensor == "dropout scale inverse";
        const auto gr = buildSdpaBwdGraph(opts);
        const auto leafUids = gr.leafInputUids({7, 8, 9});
        auto inputs = makeTensorsFromGraph(gr, leafUids);

        const auto result = fillSdpaBwd(gr, inputs, leafUids);

        EXPECT_FALSE(result.filled);
        EXPECT_NE(result.reason.find("dropout"), std::string::npos) << result.reason;
    }
}

// A seed and offset without a probability keep the dropout refusal.
TEST(TestFillInputs, SdpaBwdWithSeedOffsetAndNoProbabilityIsRefused)
{
    SdpaBwdOptions opts;
    opts.seed = true;
    opts.offset = true;
    const auto gr = buildSdpaBwdGraph(opts);
    auto inputs = makeTensorsFromGraph(gr, gr.leafInputUids({7, 8, 9}));

    const auto result = fillSdpaBwd(gr, inputs, gr.leafInputUids({7, 8, 9}));

    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("dropout"), std::string::npos) << result.reason;
}

// Each of a seed and an offset alone keeps the dropout refusal when there is no probability.
TEST(TestFillInputs, SdpaBwdWithOneOfSeedOrOffsetAndNoProbabilityIsRefused)
{
    for(const bool seed : {true, false})
    {
        SCOPED_TRACE(seed ? "seed only" : "offset only");
        SdpaBwdOptions opts;
        opts.seed = seed;
        opts.offset = !seed;
        const auto gr = buildSdpaBwdGraph(opts);
        auto inputs = makeTensorsFromGraph(gr, gr.leafInputUids({7, 8, 9}));

        const auto result = fillSdpaBwd(gr, inputs, gr.leafInputUids({7, 8, 9}));

        EXPECT_FALSE(result.filled);
        EXPECT_NE(result.reason.find("dropout"), std::string::npos) << result.reason;
    }
}

// Unlike a fused graph, where neither O nor stats is an input, owning O and stats
// but not Q leaves them underivable: Q's values are not known at fill time.
TEST(TestFillInputs, SdpaBwdWithComputedQIsRefused)
{
    const auto gr = buildSdpaBwdGraph({});
    auto ownedUids = gr.leafInputUids({7, 8, 9});
    ownedUids.erase(std::remove(ownedUids.begin(), ownedUids.end(), 1), ownedUids.end());
    auto inputs = makeTensorsFromGraph(gr, ownedUids);

    const auto result = fillSdpaBwd(gr, inputs, ownedUids);

    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("Q is computed by the graph"), std::string::npos) << result.reason;
}

// The CPU reference takes the LSE only as [B, H, Sq, 1]. Rank-3 stats are left as
// filled rather than refused: the backward reference rejects the graph with the
// same check and fails it, which a skip at fill time would hide.
TEST(TestFillInputs, SdpaBwdWithRank3StatsKeepsFilledOAndStats)
{
    SdpaBwdOptions opts;
    opts.rank3Stats = true;
    const auto gr = buildSdpaBwdGraph(opts);
    auto inputs = fillSdpaBwdWithFixedOAndStats(gr);

    expectAllEqual(*inputs.at(4), 0.5f);
    expectAllEqual(*inputs.at(6), 0.5f);
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

// ── VALUES fill ─────────────────────────────────────────────────────────────

namespace
{

// uids: x=1 (the tensor under test), y=2 (output). No nodes, so only the recipe
// decides x's contents.
GraphResult buildValuesGraph(DataType xType,
                             const std::vector<int64_t>& dims,
                             const std::vector<int64_t>& strides)
{
    GraphResult r;
    auto& b = r.builder;
    const std::vector<int64_t> outDims = {1};
    const std::vector<int64_t> outStrides = {1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", xType, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "y", DataType::FLOAT, &outStrides, &outDims));

    const std::vector<flatbuffers::Offset<Node>> nodes;
    auto graph = CreateGraphDirect(
        b, "values", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

GraphResult buildValuesGraph(DataType xType)
{
    return buildValuesGraph(xType, {4, 1, 1, 1}, {1, 1, 1, 1});
}

struct ValuesFill
{
    FillResult result;
    InputTensorMap inputs;
};

ValuesFill fillX(const GraphResult& gr, const FillRecipe& recipe)
{
    const std::vector<int64_t> leafUids = {1};
    ValuesFill out{FillResult{}, makeTensorsFromGraph(gr, leafUids)};
    InputFillRecipes recipes;
    recipes.set(1, recipe);
    out.result = fillInputs(*gr.graph, out.inputs, leafUids, recipes, nullptr);
    return out;
}

template <class T>
std::vector<T> logicalElements(InputTensorMap& inputs, int64_t uid)
{
    std::vector<T> elements;
    for(auto* element : *inputs.at(uid))
    {
        elements.push_back(*static_cast<T*>(element));
    }
    return elements;
}

FillRecipe ints(std::vector<int64_t> numbers, bool repeat = true)
{
    return FillRecipe::values(std::move(numbers), repeat);
}

FillRecipe doubles(std::vector<double> numbers, bool repeat = true)
{
    return FillRecipe::values(std::move(numbers), repeat);
}

void expectInvalid(const ValuesFill& fill, const std::string& reasonPart)
{
    EXPECT_FALSE(fill.result.filled);
    EXPECT_EQ(fill.result.status, FillResult::Status::INVALID) << fill.result.reason;
    EXPECT_NE(fill.result.reason.find("uid 1: "), std::string::npos) << fill.result.reason;
    EXPECT_NE(fill.result.reason.find(reasonPart), std::string::npos) << fill.result.reason;
}

// uids: x=1 (float), z=3 (gdn output). The GDN node has no input fills registered.
GraphResult buildUnsupportedOpGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 3, "z", DataType::FLOAT, &kStrides, &kDims));

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "gdn", DataType::FLOAT, NodeAttributes::GdnAttributes, flatbuffers::Offset<void>()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

} // namespace

TEST(TestFillInputs, ValuesFillInt32RepeatsTheList)
{
    auto fill = fillX(buildValuesGraph(DataType::INT32), ints({64, 32}));

    ASSERT_TRUE(fill.result.filled) << fill.result.reason;
    EXPECT_EQ(fill.result.status, FillResult::Status::OK);
    EXPECT_EQ(logicalElements<int32_t>(fill.inputs, 1), (std::vector<int32_t>{64, 32, 64, 32}));
}

TEST(TestFillInputs, ValuesFillInt64IsExact)
{
    constexpr int64_t K_TWO_TO_53_PLUS_ONE = (int64_t{1} << 53) + 1;
    auto fill = fillX(buildValuesGraph(DataType::INT64),
                      ints({K_TWO_TO_53_PLUS_ONE,
                            -5,
                            std::numeric_limits<int64_t>::max(),
                            std::numeric_limits<int64_t>::min()},
                           false));

    ASSERT_TRUE(fill.result.filled) << fill.result.reason;
    EXPECT_EQ(logicalElements<int64_t>(fill.inputs, 1),
              (std::vector<int64_t>{K_TWO_TO_53_PLUS_ONE,
                                    -5,
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::min()}));
}

TEST(TestFillInputs, ValuesFillUint8Int8AndBoolean)
{
    auto uint8Fill = fillX(buildValuesGraph(DataType::UINT8), ints({0, 255}));
    ASSERT_TRUE(uint8Fill.result.filled) << uint8Fill.result.reason;
    EXPECT_EQ(logicalElements<uint8_t>(uint8Fill.inputs, 1),
              (std::vector<uint8_t>{0, 255, 0, 255}));

    auto int8Fill = fillX(buildValuesGraph(DataType::INT8), ints({-128, 127}));
    ASSERT_TRUE(int8Fill.result.filled) << int8Fill.result.reason;
    EXPECT_EQ(logicalElements<int8_t>(int8Fill.inputs, 1),
              (std::vector<int8_t>{-128, 127, -128, 127}));

    auto boolFill = fillX(buildValuesGraph(DataType::BOOLEAN), ints({1, 0, 0, 1}, false));
    ASSERT_TRUE(boolFill.result.filled) << boolFill.result.reason;
    EXPECT_EQ(logicalElements<bool>(boolFill.inputs, 1),
              (std::vector<bool>{true, false, false, true}));
}

TEST(TestFillInputs, ValuesFillFloatingPointTypes)
{
    namespace types = hipdnn_data_sdk::types;

    auto floatFill = fillX(buildValuesGraph(DataType::FLOAT), doubles({0.5, -2.25}));
    ASSERT_TRUE(floatFill.result.filled) << floatFill.result.reason;
    EXPECT_EQ(logicalElements<float>(floatFill.inputs, 1),
              (std::vector<float>{0.5f, -2.25f, 0.5f, -2.25f}));

    auto halfFill = fillX(buildValuesGraph(DataType::HALF), doubles({0.5, -2.25}));
    ASSERT_TRUE(halfFill.result.filled) << halfFill.result.reason;
    const auto halves = logicalElements<types::half>(halfFill.inputs, 1);
    ASSERT_EQ(halves.size(), 4u);
    EXPECT_EQ(static_cast<float>(halves[0]), 0.5f);
    EXPECT_EQ(static_cast<float>(halves[1]), -2.25f);

    auto bf16Fill = fillX(buildValuesGraph(DataType::BFLOAT16), ints({3, -4}));
    ASSERT_TRUE(bf16Fill.result.filled) << bf16Fill.result.reason;
    const auto bf16s = logicalElements<types::bfloat16>(bf16Fill.inputs, 1);
    ASSERT_EQ(bf16s.size(), 4u);
    EXPECT_EQ(static_cast<float>(bf16s[2]), 3.0f);
    EXPECT_EQ(static_cast<float>(bf16s[3]), -4.0f);
}

TEST(TestFillInputs, ValuesFillDoubleAndFp8)
{
    namespace types = hipdnn_data_sdk::types;
    constexpr int64_t K_TWO_TO_53 = int64_t{1} << 53;

    auto doubleInts = fillX(buildValuesGraph(DataType::DOUBLE), ints({K_TWO_TO_53, -3}));
    ASSERT_TRUE(doubleInts.result.filled) << doubleInts.result.reason;
    EXPECT_EQ(logicalElements<double>(doubleInts.inputs, 1),
              (std::vector<double>{9007199254740992.0, -3.0, 9007199254740992.0, -3.0}));

    auto doubleFractions = fillX(buildValuesGraph(DataType::DOUBLE), doubles({0.1}));
    ASSERT_TRUE(doubleFractions.result.filled) << doubleFractions.result.reason;
    EXPECT_EQ(logicalElements<double>(doubleFractions.inputs, 1), std::vector<double>(4, 0.1));

    auto fp8Fill = fillX(buildValuesGraph(DataType::FP8_E4M3), doubles({0.5, -2.0}));
    ASSERT_TRUE(fp8Fill.result.filled) << fp8Fill.result.reason;
    const auto fp8s = logicalElements<types::fp8_e4m3>(fp8Fill.inputs, 1);
    ASSERT_EQ(fp8s.size(), 4u);
    EXPECT_EQ(static_cast<float>(fp8s[0]), 0.5f);
    EXPECT_EQ(static_cast<float>(fp8s[1]), -2.0f);
    EXPECT_EQ(static_cast<float>(fp8s[2]), 0.5f);
}

// With repeat true, a list longer than the tensor would leave values unused.
TEST(TestFillInputs, ValuesFillRejectsAListLongerThanTheTensor)
{
    expectInvalid(fillX(buildValuesGraph(DataType::INT32), ints({1, 2, 3, 4, 5})),
                  "5 values for 4 elements");

    auto exact = fillX(buildValuesGraph(DataType::INT32), ints({1, 2, 3, 4}));
    ASSERT_TRUE(exact.result.filled) << exact.result.reason;
}

// The block-scale default for the scale normalizes against the largest magnitude in
// the operand's VALUES list: 12 gives e = 3, so the scale range is [0.5, 2] * 2^-3.
TEST(TestFillInputs, ValuesOperandRenormalizesBlockScale)
{
    InputFillRecipes overrides;
    overrides.set(1, ints({-12, 3}));

    const auto recipes = fillBlockScaleDequantizeMatmul(DataType::FP8_E4M3, overrides);

    EXPECT_EQ(recipes.fill(2), FillRecipe::free(0.0625f, 0.25f));
}

TEST(TestFillInputs, ValuesFillWithoutRepeatNeedsOneValuePerElement)
{
    const auto graph = buildValuesGraph(DataType::FLOAT, kDims, kStrides);

    auto exact = fillX(graph, doubles({0, 1, 2, 3, 4, 5}, false));
    ASSERT_TRUE(exact.result.filled) << exact.result.reason;
    EXPECT_EQ(logicalElements<float>(exact.inputs, 1), (std::vector<float>{0, 1, 2, 3, 4, 5}));

    expectInvalid(fillX(graph, doubles({0, 1, 2, 3, 4}, false)), "5 values for 6 elements");
    expectInvalid(fillX(graph, doubles({0, 1, 2, 3, 4, 5, 6}, false)), "7 values for 6 elements");
}

// Element i of the list goes to logical element i, whatever the strides, so a
// column-major x gets the same logical contents as a row-major one.
TEST(TestFillInputs, ValuesFillFollowsLogicalOrderOnPermutedStrides)
{
    const std::vector<int64_t> columnMajor = {1, 2};
    auto fill = fillX(buildValuesGraph(DataType::INT32, kDims, columnMajor),
                      ints({0, 1, 2, 3, 4, 5}, false));
    ASSERT_TRUE(fill.result.filled) << fill.result.reason;

    auto& x = typedTensor<int32_t>(fill.inputs, 1);
    for(int64_t row = 0; row < kDims[0]; ++row)
    {
        for(int64_t col = 0; col < kDims[1]; ++col)
        {
            EXPECT_EQ(x.getHostValue(row, col), (row * kDims[1]) + col)
                << "at (" << row << ", " << col << ")";
        }
    }
}

TEST(TestFillInputs, ValuesFillRejectsWhatTheTensorCannotHold)
{
    expectInvalid(fillX(buildValuesGraph(DataType::INT8), ints({128})), "out of range");
    expectInvalid(fillX(buildValuesGraph(DataType::UINT8), ints({-1})), "out of range");
    expectInvalid(fillX(buildValuesGraph(DataType::INT32), ints({int64_t{1} << 31})),
                  "out of range");
    expectInvalid(fillX(buildValuesGraph(DataType::BOOLEAN), ints({2})), "out of range");
    expectInvalid(fillX(buildValuesGraph(DataType::INT32), doubles({2.5})), "non-integer");
}

// A list that holds any non-integer number is stored as doubles, so even its integers
// are refused by an integer tensor.
TEST(TestFillInputs, ValuesFillRejectsAMixedListOnAnIntegerTensor)
{
    const auto spec
        = parseInputFillSpec(nlohmann::json::parse(R"({"kind": "values", "values": [1, 2.5]})"));
    ASSERT_TRUE(spec.recipe.has_value());

    expectInvalid(fillX(buildValuesGraph(DataType::INT32), *spec.recipe), "non-integer");
    expectInvalid(fillX(buildValuesGraph(DataType::INT32), doubles({1.0, 2.0})), "non-integer");
}

TEST(TestFillInputs, ValuesFillRejectsSubByteTypes)
{
    for(const auto dataType :
        {DataType::FP4_E2M1, DataType::FP6_E2M3, DataType::FP6_E3M2, DataType::INT4})
    {
        expectInvalid(fillX(buildValuesGraph(dataType), ints({1})), EnumNameDataType(dataType));
    }
}

TEST(TestFillInputs, InvalidAndUnsupportedFillsStayDistinct)
{
    const auto invalid = fillX(buildValuesGraph(DataType::INT8), ints({1000}));
    EXPECT_EQ(invalid.result.status, FillResult::Status::INVALID);

    const auto gr = buildUnsupportedOpGraph();
    const auto unsupported = runFill(gr, {3});
    EXPECT_FALSE(unsupported.filled);
    EXPECT_EQ(unsupported.status, FillResult::Status::UNSUPPORTED) << unsupported.reason;
    EXPECT_NE(unsupported.reason.find("no input fill registered"), std::string::npos)
        << unsupported.reason;
}

// Every tensor draws its seed in uid order whatever its recipe, so a VALUES tensor
// leaves the random contents of every other tensor unchanged.
TEST(TestFillInputs, ValuesFillLeavesOtherTensorsRandomContentsUnchanged)
{
    const auto gr = buildConvBiasGraph();
    const auto leafUids = gr.leafInputUids({5});
    ASSERT_EQ(leafUids.size(), 3u);

    auto plainInputs = makeTensors(leafUids);
    InputFillRecipes plainRecipes;
    ASSERT_TRUE(fillInputs(*gr.graph, plainInputs, leafUids, plainRecipes, nullptr).filled);

    auto valuesInputs = makeTensors(leafUids);
    InputFillRecipes valuesRecipes;
    valuesRecipes.set(2, doubles({7.0}));
    const auto result = fillInputs(*gr.graph, valuesInputs, leafUids, valuesRecipes, nullptr);
    ASSERT_TRUE(result.filled) << result.reason;

    EXPECT_EQ(logicalElements<float>(valuesInputs, 2),
              std::vector<float>(static_cast<std::size_t>(kDims[0] * kDims[1]), 7.0f));
    for(const int64_t uid : {1, 4})
    {
        auto& expected = *plainInputs.at(uid);
        auto& actual = *valuesInputs.at(uid);
        EXPECT_EQ(std::memcmp(expected.rawHostData(),
                              actual.rawHostData(),
                              expected.elementSpace() * expected.elementSize()),
                  0)
            << "uid " << uid;
    }
}

// ── Aliases ─────────────────────────────────────────────────────────────────

namespace
{

// One SDPA backward node: q=1, k=2, v=3, o=4, do=5, stats=6 in; dq=7, dk=8, dv=9 out.
nlohmann::json sdpaBackwardGraphJson()
{
    const auto tensor = [](int64_t uid, bool stats) {
        return nlohmann::json{
            {"name", ""},
            {"uid", uid},
            {"dims", stats ? std::vector<int64_t>{1, 1, 2, 1} : std::vector<int64_t>{1, 1, 2, 4}},
            {"strides",
             stats ? std::vector<int64_t>{2, 2, 1, 1} : std::vector<int64_t>{8, 8, 4, 1}},
            {"data_type", "float"},
            {"virtual", false}};
    };
    auto tensors = nlohmann::json::array();
    for(int64_t uid = 1; uid <= 9; ++uid)
    {
        tensors.push_back(tensor(uid, uid == 6));
    }
    auto graph = nlohmann::json::parse(R"({
        "name": "sdpa_backward", "compute_data_type": "float", "io_data_type": "float",
        "intermediate_data_type": "float",
        "nodes": [{
            "name": "sdpa_backward", "type": "SdpaBackwardAttributes",
            "compute_data_type": "float",
            "inputs": {
                "q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3, "o_tensor_uid": 4,
                "do_tensor_uid": 5, "stats_tensor_uid": 6, "scale_tensor_uid": null,
                "attn_mask_tensor_uid": null, "seq_len_q_tensor_uid": null,
                "seq_len_kv_tensor_uid": null, "seed_tensor_uid": null,
                "offset_tensor_uid": null, "dropout_mask_tensor_uid": null,
                "dropout_scale_tensor_uid": null, "dropout_scale_inv_tensor_uid": null},
            "outputs": {
                "dq_tensor_uid": 7, "dk_tensor_uid": 8, "dv_tensor_uid": 9,
                "dbias_tensor_uid": null},
            "parameters": {
                "alibi_mask": false, "padding_mask": false, "causal_mask": false,
                "causal_mask_bottom_right": false, "dropout_probability": null,
                "attn_scale_value": null, "left_bound": null, "right_bound": null,
                "diagonal_alignment": "TOP_LEFT"}}]
    })");
    graph["tensors"] = std::move(tensors);
    return graph;
}

bool sameHostBytes(hipdnn_data_sdk::utilities::ITensor& a, hipdnn_data_sdk::utilities::ITensor& b)
{
    return a.elementSpace() == b.elementSpace() && a.elementSize() == b.elementSize()
           && std::memcmp(a.rawHostData(), b.rawHostData(), a.elementSpace() * a.elementSize())
                  == 0;
}

// w (uid 2) of buildConvBiasGraph() as a view of x (uid 1).
const std::vector<TensorAlias> kWViewOfX = {{2, 1, 0}};

} // namespace

TEST(TestFillInputs, AliasViewIsNotFilledAndNeedsNoTensor)
{
    const auto gr = buildConvBiasGraph();
    const auto leafUids = gr.leafInputUids({5});
    auto inputs = makeTensors({1, 4});
    InputFillRecipes recipes;

    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr, kWViewOfX);
    ASSERT_TRUE(result.filled) << result.reason;
    EXPECT_EQ(inputs.count(2), 0u);
    EXPECT_EQ(inputs.size(), 2u);
}

// The view keeps its place in the seed sequence, so the owner and every other input
// hold what they hold without the alias.
TEST(TestFillInputs, AliasLeavesEveryOtherTensorsContentsUnchanged)
{
    const auto gr = buildConvBiasGraph();
    const auto leafUids = gr.leafInputUids({5});
    ASSERT_EQ(leafUids.size(), 3u);

    auto plainInputs = makeTensors(leafUids);
    InputFillRecipes plainRecipes;
    ASSERT_TRUE(fillInputs(*gr.graph, plainInputs, leafUids, plainRecipes, nullptr).filled);

    auto aliasedInputs = makeTensors({1, 4});
    InputFillRecipes aliasedRecipes;
    const auto result
        = fillInputs(*gr.graph, aliasedInputs, leafUids, aliasedRecipes, nullptr, kWViewOfX);
    ASSERT_TRUE(result.filled) << result.reason;

    for(const int64_t uid : {1, 4})
    {
        EXPECT_TRUE(sameHostBytes(*plainInputs.at(uid), *aliasedInputs.at(uid))) << "uid " << uid;
    }
}

// Two views of one owner: the owner is filled once, as without the aliases, whatever the
// order of the alias entries.
TEST(TestFillInputs, AliasOwnerWithTwoViewsIsFilledOnce)
{
    const auto gr = buildConvBiasGraph();
    const auto leafUids = gr.leafInputUids({5});

    auto plainInputs = makeTensors(leafUids);
    InputFillRecipes plainRecipes;
    ASSERT_TRUE(fillInputs(*gr.graph, plainInputs, leafUids, plainRecipes, nullptr).filled);

    for(const auto& aliases : {std::vector<TensorAlias>{{2, 1, 0}, {4, 1, 0}},
                               std::vector<TensorAlias>{{4, 1, 0}, {2, 1, 0}}})
    {
        SCOPED_TRACE("first view " + std::to_string(aliases.front().uid));
        auto inputs = makeTensors({1});
        InputFillRecipes recipes;
        const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr, aliases);
        ASSERT_TRUE(result.filled) << result.reason;
        EXPECT_EQ(inputs.size(), 1u);
        EXPECT_TRUE(sameHostBytes(*plainInputs.at(1), *inputs.at(1)));
    }
}

// A seed-only metadata entry for a view sets no recipe, so the fill stays valid, and the
// view's seed changes no other tensor.
TEST(TestFillInputs, AliasViewWithASeedOnlyEntryChangesNothing)
{
    const auto gr = buildConvBiasGraph();
    const auto leafUids = gr.leafInputUids({5});

    auto expectedInputs = makeTensors({1, 4});
    InputFillRecipes expectedRecipes;
    ASSERT_TRUE(fillInputs(*gr.graph, expectedInputs, leafUids, expectedRecipes, nullptr, kWViewOfX)
                    .filled);

    auto inputs = makeTensors({1, 4});
    InputFillRecipes recipes;
    recipes.loadFromJson({{2, nlohmann::json{{"seed", 7u}}}});
    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr, kWViewOfX);
    ASSERT_TRUE(result.filled) << result.reason;
    for(const int64_t uid : {1, 4})
    {
        EXPECT_TRUE(sameHostBytes(*expectedInputs.at(uid), *inputs.at(uid))) << "uid " << uid;
    }
}

// Strides {4, 1} on dims {2, 3} leave element 3 of the allocation outside the tensor.
TEST(TestFillInputs, AliasOwnerGapsHoldZeros)
{
    const auto gr = buildConvBiasGraph();
    const auto leafUids = gr.leafInputUids({5});
    const std::vector<int64_t> gapStrides = {4, 1};

    InputTensorMap inputs = makeTensors({4});
    inputs[1] = std::make_shared<hipdnn_data_sdk::utilities::Tensor<float>>(kDims, gapStrides);
    auto& owner = *inputs.at(1);
    ASSERT_EQ(owner.elementSpace(), 7u);
    std::memset(owner.rawHostData(), 0xFF, owner.elementSpace() * owner.elementSize());

    InputFillRecipes recipes;
    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr, kWViewOfX);
    ASSERT_TRUE(result.filled) << result.reason;

    const auto* data = static_cast<const float*>(owner.rawHostData());
    EXPECT_EQ(data[3], 0.0f);
    for(const std::size_t i : {0u, 1u, 2u, 4u, 5u, 6u})
    {
        EXPECT_GE(data[i], -1.0f) << "element " << i;
        EXPECT_LE(data[i], 1.0f) << "element " << i;
    }
}

// x is large enough for the device fill, and is an owner, so it is filled on the host;
// w is not an owner and still goes to the device.
TEST(TestFillInputs, AliasOwnerIsFilledOnTheHostWithADeviceFiller)
{
    SKIP_IF_NO_DEVICES();
    if(!DeviceInputFiller::isSupported())
    {
        GTEST_SKIP() << "rocRAND not available. Skipping test.";
    }

    const auto graph = buildMixedSizeConvBiasGraph();
    const std::vector<int64_t> leafUids = {1, 2, 4};
    const std::vector<TensorAlias> biasViewOfX = {{4, 1, 0}};

    auto hostInputs = makeTensorsFromGraph(graph, {1, 2});
    InputFillRecipes hostRecipes;
    ASSERT_TRUE(
        fillInputs(*graph.graph, hostInputs, leafUids, hostRecipes, nullptr, biasViewOfX).filled);

    auto deviceInputs = makeTensorsFromGraph(graph, {1, 2});
    InputFillRecipes deviceRecipes;
    DeviceInputFiller device;
    const auto result
        = fillInputs(*graph.graph, deviceInputs, leafUids, deviceRecipes, &device, biasViewOfX);
    ASSERT_TRUE(result.filled) << result.reason;
    EXPECT_EQ(result.deviceFilled, 1u);

    const auto& x = typedTensor<float>(deviceInputs, 1);
    EXPECT_NE(x.memory().location(), hipdnn_data_sdk::utilities::MemoryLocation::DEVICE);
    EXPECT_TRUE(sameHostBytes(*hostInputs.at(1), *deviceInputs.at(1)));
}

TEST(TestFillInputs, AliasViewWithASetRecipeIsInvalid)
{
    const auto gr = buildConvBiasGraph();
    const auto leafUids = gr.leafInputUids({5});
    auto inputs = makeTensors({1, 4});
    InputFillRecipes recipes;
    recipes.set(2, FillRecipe::fixed(1.0f));

    const auto result = fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr, kWViewOfX);
    EXPECT_FALSE(result.filled);
    EXPECT_EQ(result.status, FillResult::Status::INVALID);
    EXPECT_NE(result.reason.find("uid 2"), std::string::npos) << result.reason;
}

TEST(TestFillInputs, AliasOwnerWithoutATensorThrows)
{
    const auto gr = buildConvBiasGraph();
    const auto leafUids = gr.leafInputUids({5});
    auto inputs = makeTensors({1, 4});
    InputFillRecipes recipes;
    const std::vector<TensorAlias> ownerIsAnOutput = {{2, 5, 0}};

    EXPECT_THROW(fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr, ownerIsAnOutput),
                 std::invalid_argument);
}

TEST(TestFillInputs, FillDerivesInputsOnlyForAGraphWhoseFillReplacesInputs)
{
    flatbuffers::DetachedBuffer buffer;
    ASSERT_TRUE(bundle::detail::buildGraphBuffer(sdpaBackwardGraphJson(), buffer));
    const auto& sdpaBackward = *GetGraph(buffer.data());
    EXPECT_TRUE(fillDerivesInputs(sdpaBackward));

    const auto convBias = buildConvBiasGraph();
    EXPECT_FALSE(fillDerivesInputs(*convBias.graph));

    InputTensorMap inputs;
    InputFillRecipes recipes;
    EXPECT_THROW(fillInputs(sdpaBackward, inputs, {1, 2, 3}, recipes, nullptr, {{3, 2, 0}}),
                 std::invalid_argument);
}

// ── Absent tensors ──────────────────────────────────────────────────────────

namespace
{

namespace fx = hipdnn_integration_tests::bundle::fixtures;
namespace sdpa_utils = hipdnn_integration_tests::test_utils;

// The loaded case `caseId` under `root`; nullopt, with a test failure, when it does not load.
std::optional<bundle::IntegrationTestBundle> loadSweepCase(const std::filesystem::path& root,
                                                           const std::string& caseId,
                                                           bundle::SweepManifestCache& sweeps)
{
    for(const auto& discovered : bundle::discoverBundles(root))
    {
        if(discovered.testName != caseId)
        {
            continue;
        }
        auto result = bundle::loadIntegrationTestBundle(discovered, sweeps);
        if(auto* loaded = std::get_if<bundle::IntegrationTestBundle>(&result))
        {
            return std::move(*loaded);
        }
        ADD_FAILURE() << caseId << " did not load";
        return std::nullopt;
    }
    ADD_FAILURE() << "no case " << caseId << " under " << root;
    return std::nullopt;
}

// Fills the inputs of `loaded` as the harness does on the host: every tensor that is not
// virtual and not an output, with the metadata `seed` and the recipes and seeds of the
// metadata `inputs`.
InputTensorMap fillLoadedInputs(const bundle::IntegrationTestBundle& loaded)
{
    const auto wrapper = loaded.graphWrapper();
    const std::set<int64_t> outputUids(loaded.outputTensorUids.begin(),
                                       loaded.outputTensorUids.end());
    InputTensorMap inputs;
    std::vector<int64_t> leafUids;
    for(const auto& [uid, attributes] : wrapper.getTensorMap())
    {
        if(attributes->virtual_() || outputUids.count(uid) != 0)
        {
            continue;
        }
        leafUids.push_back(uid);
        inputs.emplace(uid, hipdnn_test_sdk::detail::createTensorFromAttribute(*attributes));
    }
    InputFillRecipes recipes;
    if(loaded.metadata.seed.has_value())
    {
        recipes.setGlobalSeed(static_cast<unsigned int>(*loaded.metadata.seed));
    }
    if(loaded.metadata.inputs.has_value())
    {
        recipes.loadFromJson(*loaded.metadata.inputs);
    }
    const auto result = fillInputs(wrapper.getGraph(), inputs, leafUids, recipes, nullptr);
    EXPECT_TRUE(result.filled) << result.reason;
    return inputs;
}

} // namespace

// With a seed for each input, removing the sink (the lowest uid) leaves the contents of
// Q, K and V unchanged. Without the seeds, the stream moves and Q changes.
TEST(TestFillInputs, AbsentTensorLeavesTheOtherSeededInputsUnchanged)
{
    const auto scratch = hipdnn_test_sdk::utilities::claimScratchDirectory("fill_absent_tensors");
    const auto graph = sdpa_utils::optionalSdpaFwdGraphJson(/*withStats=*/true, /*withSink=*/true);
    const std::vector<int64_t> sinkAbsent = {sdpa_utils::K_OPTIONAL_SDPA_SINK_UID};
    const nlohmann::json seeds
        = {{"1", {{"seed", 11}}}, {"2", {{"seed", 12}}}, {"3", {{"seed", 13}}}};

    auto seededAll = fx::optionalTensorSweepCase("seeded_all", graph);
    seededAll["metadata"]["inputs"] = seeds;
    seededAll["metadata"]["inputs"]["0"] = nlohmann::json{{"seed", 10}};
    auto seededSinkAbsent = fx::optionalTensorSweepCase("seeded_sink_absent", graph, sinkAbsent);
    seededSinkAbsent["metadata"]["inputs"] = seeds;
    fx::writeTemplateSweep(
        scratch.path() / "quick" / "SdpaFwd" / "Optional",
        fx::templateOfGraph(graph),
        nlohmann::json::array(
            {seededAll,
             seededSinkAbsent,
             fx::optionalTensorSweepCase("unseeded_all", graph),
             fx::optionalTensorSweepCase("unseeded_sink_absent", graph, sinkAbsent)}));

    bundle::SweepManifestCache sweeps;
    const auto all = loadSweepCase(scratch.path(), "seeded_all", sweeps);
    const auto absent = loadSweepCase(scratch.path(), "seeded_sink_absent", sweeps);
    const auto unseededAll = loadSweepCase(scratch.path(), "unseeded_all", sweeps);
    const auto unseededAbsent = loadSweepCase(scratch.path(), "unseeded_sink_absent", sweeps);
    ASSERT_TRUE(all.has_value() && absent.has_value() && unseededAll.has_value()
                && unseededAbsent.has_value());

    const auto allInputs = fillLoadedInputs(*all);
    const auto absentInputs = fillLoadedInputs(*absent);
    ASSERT_EQ(allInputs.count(sdpa_utils::K_OPTIONAL_SDPA_SINK_UID), 1u);
    EXPECT_EQ(absentInputs.count(sdpa_utils::K_OPTIONAL_SDPA_SINK_UID), 0u);
    for(const int64_t uid : {sdpa_utils::K_OPTIONAL_SDPA_Q_UID,
                             sdpa_utils::K_OPTIONAL_SDPA_K_UID,
                             sdpa_utils::K_OPTIONAL_SDPA_V_UID})
    {
        ASSERT_EQ(absentInputs.count(uid), 1u) << "uid " << uid;
        EXPECT_TRUE(sameHostBytes(*allInputs.at(uid), *absentInputs.at(uid))) << "uid " << uid;
    }

    EXPECT_EQ(fx::hostVariantPackUids(*absent),
              (std::set<int64_t>{sdpa_utils::K_OPTIONAL_SDPA_Q_UID,
                                 sdpa_utils::K_OPTIONAL_SDPA_K_UID,
                                 sdpa_utils::K_OPTIONAL_SDPA_V_UID,
                                 sdpa_utils::K_OPTIONAL_SDPA_O_UID,
                                 sdpa_utils::K_OPTIONAL_SDPA_STATS_UID}));

    const auto unseededAllInputs = fillLoadedInputs(*unseededAll);
    const auto unseededAbsentInputs = fillLoadedInputs(*unseededAbsent);
    EXPECT_FALSE(sameHostBytes(*unseededAllInputs.at(sdpa_utils::K_OPTIONAL_SDPA_Q_UID),
                               *unseededAbsentInputs.at(sdpa_utils::K_OPTIONAL_SDPA_Q_UID)));
}

// ── Resample backward: the max-pooling index is the forward's argmax ─────────
// uids: dy=1, index=2, dx=3. The scratch x the derivation pools is filled
// uniformly in [-1, 1] from the index's seed, so with the seed pinned the test
// rebuilds the same x and checks each index against an argmax computed here.

namespace
{

constexpr unsigned int kResampleIndexSeed = 1234;

struct ResampleBwdOptions
{
    std::vector<int64_t> dxDims = {1, 2, 5, 6};
    std::vector<int64_t> dyDims = {1, 2, 3, 3};
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 1};
    std::vector<int64_t> stride = {2, 2};
    std::vector<int64_t> window = {3, 3};
    ResampleMode mode = ResampleMode::MAXPOOL;
    PaddingMode paddingMode = PaddingMode::ZERO_PAD;
    DataType indexType = DataType::INT32;
    // Bundles may lay dy, index and dx out NHWC/NDHWC rather than packed in dims
    // order. Keep in sync between buildResampleBwdGraph() and
    // expectIndexIsForwardArgmax(), which must build its own scratch x with the
    // same strides the derivation builds dx's scratch x with.
    bool nhwcStrides = false;
};

// Packed-in-dims-order strides when nhwcStrides is false, matching every test
// below until this field is set; channel-last strides (NHWC for 4D, NDHWC for 5D)
// when it is true.
std::vector<int64_t> resampleBwdStrides(const std::vector<int64_t>& dims, bool nhwcStrides)
{
    using hipdnn_data_sdk::utilities::generateStrides;
    using hipdnn_data_sdk::utilities::TensorLayout;

    if(!nhwcStrides)
    {
        return generateStrides(dims);
    }
    return generateStrides(
        dims, dims.size() == 5 ? TensorLayout::NDHWC.strideOrder : TensorLayout::NHWC.strideOrder);
}

GraphResult buildResampleBwdGraph(const ResampleBwdOptions& opts)
{
    GraphResult r;
    auto& b = r.builder;

    const auto dyStrides = resampleBwdStrides(opts.dyDims, opts.nhwcStrides);
    const auto dxStrides = resampleBwdStrides(opts.dxDims, opts.nhwcStrides);

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(b, 1, "dy", DataType::FLOAT, &dyStrides, &opts.dyDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "index", opts.indexType, &dyStrides, &opts.dyDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 3, "dx", DataType::FLOAT, &dxStrides, &opts.dxDims));

    auto bwd = CreateResampleBwdAttributesDirect(b,
                                                 1,
                                                 3,
                                                 2,
                                                 &opts.prePadding,
                                                 &opts.postPadding,
                                                 &opts.stride,
                                                 &opts.window,
                                                 opts.mode,
                                                 opts.paddingMode);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "resample_bwd", DataType::FLOAT, NodeAttributes::ResampleBwdAttributes, bwd.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);

    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// The index is filled with a fixed 7, which a forward over these shapes never
// writes everywhere, so a derived index is told apart from a filled one.
FillResult fillResampleBwd(const GraphResult& gr,
                           InputTensorMap& inputs,
                           const std::vector<int64_t>& ownedUids)
{
    InputFillRecipes recipes;
    recipes.setGlobalSeed(42);
    recipes.setSeed(2, kResampleIndexSeed);
    recipes.set(2, FillRecipe::fixed(7.0f));
    return fillInputs(*gr.graph, inputs, ownedUids, recipes, nullptr);
}

InputTensorMap fillResampleBwdInputs(const GraphResult& gr)
{
    const auto leafUids = gr.leafInputUids({3});
    auto inputs = makeTensorsFromGraph(gr, leafUids);
    const auto result = fillResampleBwd(gr, inputs, leafUids);
    EXPECT_TRUE(result.filled) << result.reason;
    return inputs;
}

// Checks every index against the argmax of the x the derivation pooled: the first
// maximum in window order, -1 when it is a zero-padding element, and with
// NEG_INF_PAD only in-bounds elements competing.
void expectIndexIsForwardArgmax(InputTensorMap& inputs, const ResampleBwdOptions& opts)
{
    hipdnn_data_sdk::utilities::Tensor<float> x(opts.dxDims,
                                                resampleBwdStrides(opts.dxDims, opts.nhwcStrides));
    x.fillWithRandomValues(-1.0f, 1.0f, kResampleIndexSeed);
    auto& index = typedTensor<int32_t>(inputs, 2);

    const std::size_t spatialRank = opts.dxDims.size() - 2;
    int64_t spatialSize = 1;
    for(std::size_t i = 2; i < opts.dxDims.size(); ++i)
    {
        spatialSize *= opts.dxDims[i];
    }

    std::size_t checked = 0;
    std::size_t padded = 0;
    hipdnn_data_sdk::utilities::iterateAlongDimensions(
        opts.dyDims, [&](const std::vector<int64_t>& dyIndices) {
            bool selected = false;
            float best = 0.0f;
            int32_t bestIndex = -1;
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                opts.window, [&](const std::vector<int64_t>& windowIndices) {
                    std::vector<int64_t> xIndices = {dyIndices[0], dyIndices[1]};
                    bool inBounds = true;
                    int64_t flat = 0;
                    for(std::size_t i = 0; i < spatialRank; ++i)
                    {
                        const int64_t at = (dyIndices[i + 2] * opts.stride[i]) + windowIndices[i]
                                           - opts.prePadding[i];
                        inBounds = inBounds && at >= 0 && at < opts.dxDims[i + 2];
                        xIndices.push_back(at);
                        flat = (flat * opts.dxDims[i + 2]) + at;
                    }
                    if(!inBounds && opts.paddingMode == PaddingMode::NEG_INF_PAD)
                    {
                        return;
                    }
                    const float value = inBounds ? x.getHostValue(xIndices) : 0.0f;
                    if(!selected || value > best)
                    {
                        selected = true;
                        best = value;
                        bestIndex = inBounds ? static_cast<int32_t>(flat) : -1;
                    }
                });

            const int32_t actual = index.getHostValue(dyIndices);
            EXPECT_EQ(actual, bestIndex);
            EXPECT_GE(actual, -1);
            EXPECT_LT(actual, spatialSize);
            padded += actual == -1 ? 1 : 0;
            ++checked;
        });

    int64_t dyElements = 1;
    for(const int64_t d : opts.dyDims)
    {
        dyElements *= d;
    }
    EXPECT_EQ(checked, static_cast<std::size_t>(dyElements));
    if(opts.paddingMode == PaddingMode::NEG_INF_PAD)
    {
        EXPECT_EQ(padded, 0u);
    }
}

void expectIndexAllEqual(InputTensorMap& inputs, int32_t value)
{
    auto& index = typedTensor<int32_t>(inputs, 2);
    const std::vector<int64_t> dims = index.dims();
    hipdnn_data_sdk::utilities::iterateAlongDimensions(
        dims, [&](const std::vector<int64_t>& at) { EXPECT_EQ(index.getHostValue(at), value); });
}

} // namespace

TEST(TestFillInputs, ResampleBwdMaxpoolIndexIsForwardArgmax2dZeroPad)
{
    const ResampleBwdOptions opts;
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexIsForwardArgmax(inputs, opts);
}

TEST(TestFillInputs, ResampleBwdMaxpoolIndexIsForwardArgmax2dNegInfPad)
{
    ResampleBwdOptions opts;
    opts.paddingMode = PaddingMode::NEG_INF_PAD;
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexIsForwardArgmax(inputs, opts);
}

TEST(TestFillInputs, ResampleBwdMaxpoolIndexIsForwardArgmax3d)
{
    ResampleBwdOptions opts;
    opts.dxDims = {1, 2, 3, 4, 4};
    opts.dyDims = {1, 2, 2, 2, 2};
    opts.prePadding = {0, 0, 0};
    opts.postPadding = {0, 0, 0};
    opts.stride = {1, 2, 2};
    opts.window = {2, 2, 2};
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexIsForwardArgmax(inputs, opts);
}

// Unlike the default shape above, dy(0, 0)'s window has exactly one in-bounds
// element (x(0, 0, 0, 0)), and with the pinned seed that element's fill is
// negative. Zero-padding's implicit 0 outranks it, so the derived index is -1
// there; a hard-coded NEG_INF_PAD would instead pick the lone in-bounds element
// (index 0) and fail expectIndexIsForwardArgmax's per-element check.
TEST(TestFillInputs, ResampleBwdZeroPadIndexIsMinusOneWhenPaddingOutranksSoleValidElement)
{
    ResampleBwdOptions opts;
    opts.dxDims = {1, 1, 5, 5};
    opts.dyDims = {1, 1, 3, 3};
    opts.prePadding = {1, 1};
    opts.postPadding = {1, 1};
    opts.stride = {2, 2};
    opts.window = {2, 2};
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexIsForwardArgmax(inputs, opts);

    auto& index = typedTensor<int32_t>(inputs, 2);
    EXPECT_EQ(index.getHostValue(std::vector<int64_t>{0, 0, 0, 0}), -1);
}

// Same shape and seed as above, under NEG_INF_PAD: the sole in-bounds element at
// dy(0, 0) is the only candidate, so its index wins regardless of its sign. A
// hard-coded ZERO_PAD would instead yield -1 there and fail the same check.
TEST(TestFillInputs, ResampleBwdNegInfPadPicksSoleValidElementOverPadding)
{
    ResampleBwdOptions opts;
    opts.dxDims = {1, 1, 5, 5};
    opts.dyDims = {1, 1, 3, 3};
    opts.prePadding = {1, 1};
    opts.postPadding = {1, 1};
    opts.stride = {2, 2};
    opts.window = {2, 2};
    opts.paddingMode = PaddingMode::NEG_INF_PAD;
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexIsForwardArgmax(inputs, opts);

    auto& index = typedTensor<int32_t>(inputs, 2);
    EXPECT_EQ(index.getHostValue(std::vector<int64_t>{0, 0, 0, 0}), 0);
}

// 15 of the 39 shipped Default bundle cases lay dy, index and dx out NHWC/NDHWC.
// The derivation's indexView must be built from the index tensor's own strides;
// a mutant that rebuilt it from dims-only (packed) strides instead would pass
// every test above, which stays packed throughout.
TEST(TestFillInputs, ResampleBwdMaxpoolIndexIsForwardArgmax2dNhwc)
{
    ResampleBwdOptions opts;
    opts.nhwcStrides = true;
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexIsForwardArgmax(inputs, opts);
}

TEST(TestFillInputs, ResampleBwdMaxpoolIndexIsForwardArgmax3dNdhwc)
{
    ResampleBwdOptions opts;
    opts.dxDims = {1, 2, 3, 4, 4};
    opts.dyDims = {1, 2, 2, 2, 2};
    opts.prePadding = {0, 0, 0};
    opts.postPadding = {0, 0, 0};
    opts.stride = {1, 2, 2};
    opts.window = {2, 2, 2};
    opts.nhwcStrides = true;
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexIsForwardArgmax(inputs, opts);
}

// The device and reference bundles are filled separately; both must hold the
// same indices.
TEST(TestFillInputs, ResampleBwdIndexIsDeterministicAcrossFills)
{
    const ResampleBwdOptions opts;
    const auto gr = buildResampleBwdGraph(opts);
    const auto leafUids = gr.leafInputUids({3});

    auto fillOnce = [&]() {
        auto inputs = makeTensorsFromGraph(gr, leafUids);
        InputFillRecipes recipes;
        recipes.setGlobalSeed(42);
        EXPECT_TRUE(fillInputs(*gr.graph, inputs, leafUids, recipes, nullptr).filled);
        return inputs;
    };
    auto first = fillOnce();
    auto second = fillOnce();

    auto& a = typedTensor<int32_t>(first, 2);
    auto& b = typedTensor<int32_t>(second, 2);
    const std::vector<int64_t> dims = a.dims();
    hipdnn_data_sdk::utilities::iterateAlongDimensions(dims, [&](const std::vector<int64_t>& at) {
        EXPECT_EQ(a.getHostValue(at), b.getHostValue(at));
    });
}

TEST(TestFillInputs, ResampleBwdAvgpoolLeavesIndexFilled)
{
    ResampleBwdOptions opts;
    opts.mode = ResampleMode::AVGPOOL_INCLUDE_PADDING;
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexAllEqual(inputs, 7);
}

// In a fused forward-backward graph the forward node computes the index.
TEST(TestFillInputs, ResampleBwdComputedIndexIsNotDerived)
{
    const ResampleBwdOptions opts;
    const auto gr = buildResampleBwdGraph(opts);
    const std::vector<int64_t> owned = {1};
    auto inputs = makeTensorsFromGraph(gr, owned);
    const auto result = fillResampleBwd(gr, inputs, owned);
    EXPECT_TRUE(result.filled) << result.reason;
    EXPECT_EQ(inputs.count(2), 0u);
}

TEST(TestFillInputs, ResampleBwdNonInt32IndexIsUnsupported)
{
    ResampleBwdOptions opts;
    opts.indexType = DataType::INT64;
    const auto gr = buildResampleBwdGraph(opts);
    const auto leafUids = gr.leafInputUids({3});
    auto inputs = makeTensorsFromGraph(gr, leafUids);
    const auto result = fillResampleBwd(gr, inputs, leafUids);
    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("resample_bwd"), std::string::npos) << result.reason;
    EXPECT_NE(result.reason.find("INT64"), std::string::npos) << result.reason;
}

// The CPU forward rejects a window whose rank does not match the tensors; the
// index keeps its fill so the CPU backward reference, which rejects it too,
// fails the graph instead of it being skipped.
TEST(TestFillInputs, ResampleBwdRejectedShapeKeepsFill)
{
    ResampleBwdOptions opts;
    opts.window = {3};
    const auto gr = buildResampleBwdGraph(opts);
    auto inputs = fillResampleBwdInputs(gr);
    expectIndexAllEqual(inputs, 7);
}

// NOLINTEND(readability-identifier-naming)
