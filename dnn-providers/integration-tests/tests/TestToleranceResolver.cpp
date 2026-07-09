// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "harness/tolerance/ToleranceResolver.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_flatbuffers_sdk::data_objects;
namespace tol = hipdnn_integration_tests::tolerance;
namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

const std::vector<int64_t> kDims = {2, 3};
const std::vector<int64_t> kStrides = {3, 1};

struct GraphResult
{
    flatbuffers::FlatBufferBuilder builder;
    const Graph* graph = nullptr;
};

fb::GraphWrapper wrapGraph(const GraphResult& r)
{
    return fb::GraphWrapper(r.builder.GetBufferPointer(), r.builder.GetSize());
}

// ── Single conv fwd ─────────────────────────────────────────────────────────

GraphResult buildSingleConvFwd()
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

// ── BN inference → conv fwd (discriminating: loosest op is NOT the last) ────

GraphResult buildBnInferenceConvFwd()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "mean", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 3, "inv_var", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 4, "scale", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 5, "bias", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 6, "bn_y", DataType::FLOAT, &kStrides, &kDims, true));
    tensors.push_back(CreateTensorAttributesDirect(b, 7, "w", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 8, "y", DataType::FLOAT, &kStrides, &kDims));

    auto bn = CreateBatchnormInferenceAttributes(b, 1, 2, 3, 4, 5, 6);
    auto conv = CreateConvolutionFwdAttributesDirect(b, 6, 7, 8);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "bn", DataType::FLOAT, NodeAttributes::BatchnormInferenceAttributes, bn.Union()));
    nodes.push_back(CreateNodeDirect(
        b, "conv", DataType::FLOAT, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── Conv fwd + trailing pointwise ───────────────────────────────────────────

GraphResult buildConvPlusPointwise()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(CreateTensorAttributesDirect(b, 2, "w", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 3, "conv_y", DataType::FLOAT, &kStrides, &kDims, true));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 4, "out", DataType::FLOAT, &kStrides, &kDims));

    auto conv = CreateConvolutionFwdAttributesDirect(b, 1, 2, 3);
    auto pw = CreatePointwiseAttributes(b,
                                        PointwiseMode::RELU_FWD,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        3,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        4);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "conv", DataType::FLOAT, NodeAttributes::ConvolutionFwdAttributes, conv.Union()));
    nodes.push_back(CreateNodeDirect(
        b, "relu", DataType::FLOAT, NodeAttributes::PointwiseAttributes, pw.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── All-pointwise graph ─────────────────────────────────────────────────────

GraphResult buildAllPointwise()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(b, 2, "out", DataType::FLOAT, &kStrides, &kDims));

    auto pw = CreatePointwiseAttributes(b,
                                        PointwiseMode::RELU_FWD,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        1,
                                        flatbuffers::nullopt,
                                        flatbuffers::nullopt,
                                        2);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "relu", DataType::FLOAT, NodeAttributes::PointwiseAttributes, pw.Union()));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── Empty graph (zero nodes) ────────────────────────────────────────────────

GraphResult buildEmptyGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    std::vector<flatbuffers::Offset<Node>> nodes;

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

// ── Unknown / unmapped op (uses NONE attributes) ────────────────────────────

GraphResult buildUnknownOpGraph()
{
    GraphResult r;
    auto& b = r.builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(b, 1, "x", DataType::FLOAT, &kStrides, &kDims));

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(b, "unknown", DataType::FLOAT, NodeAttributes::NONE, 0));

    auto graph = CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    b.Finish(graph);
    r.graph = GetGraph(b.GetBufferPointer());
    return r;
}

} // namespace

// Known fp32 reference values from TestTolerances.hpp
constexpr float kConvFwdFp32 = 1e-5f;
constexpr float kBnInferenceFp32 = 2e-4f;
constexpr float kPointwiseFp32 = 1e-5f;
constexpr float kFallback = 1e-3f;

// ── Single non-Pointwise op: both policies agree ────────────────────────────

TEST(TestToleranceResolver, SingleConvFwd_MaxAcrossNodes)
{
    auto g = buildSingleConvFwd();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kConvFwdFp32);
}

TEST(TestToleranceResolver, SingleConvFwd_OutputOpTolerance)
{
    auto g = buildSingleConvFwd();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kConvFwdFp32);
}

// ── Discriminating case: BN inference (2e-4) then conv fwd (1e-5) ───────────
// MAX picks the loose one (2e-4), OUTPUT_OP picks the last non-Pointwise (1e-5).

TEST(TestToleranceResolver, BnInferenceConvFwd_MaxAcrossNodes)
{
    auto g = buildBnInferenceConvFwd();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kBnInferenceFp32);
}

TEST(TestToleranceResolver, BnInferenceConvFwd_OutputOpTolerance)
{
    auto g = buildBnInferenceConvFwd();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kConvFwdFp32);
}

// ── Conv + trailing Pointwise: OUTPUT_OP skips Pointwise, picks conv ────────

TEST(TestToleranceResolver, ConvPlusPointwise_MaxAcrossNodes)
{
    auto g = buildConvPlusPointwise();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT),
                    std::max(kConvFwdFp32, kPointwiseFp32));
}

TEST(TestToleranceResolver, ConvPlusPointwise_OutputOpTolerance)
{
    auto g = buildConvPlusPointwise();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kConvFwdFp32);
}

// ── All-Pointwise: OUTPUT_OP falls back to MAX ──────────────────────────────

TEST(TestToleranceResolver, AllPointwise_MaxAcrossNodes)
{
    auto g = buildAllPointwise();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kPointwiseFp32);
}

TEST(TestToleranceResolver, AllPointwise_OutputOpTolerance)
{
    auto g = buildAllPointwise();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kPointwiseFp32);
}

// ── Empty graph: both return 1e-3 floor ─────────────────────────────────────

TEST(TestToleranceResolver, EmptyGraph_MaxAcrossNodes)
{
    auto g = buildEmptyGraph();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kFallback);
}

TEST(TestToleranceResolver, EmptyGraph_OutputOpTolerance)
{
    auto g = buildEmptyGraph();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kFallback);
}

// ── Unknown op: conservative 1e-3 fallback ──────────────────────────────────

TEST(TestToleranceResolver, UnknownOp_MaxAcrossNodes)
{
    auto g = buildUnknownOpGraph();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kFallback);
}

TEST(TestToleranceResolver, UnknownOp_OutputOpTolerance)
{
    auto g = buildUnknownOpGraph();
    auto w = wrapGraph(g);
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kFallback);
}

// NOLINTEND(readability-identifier-naming)
