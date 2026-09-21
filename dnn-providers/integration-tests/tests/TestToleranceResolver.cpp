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

// Tolerance functions only inspect attributes_type() per node — shapes and uids
// are irrelevant. This builder creates a minimal valid graph from just node types.
fb::GraphWrapper buildGraph(flatbuffers::FlatBufferBuilder& builder,
                            const std::vector<NodeAttributes>& nodeTypes)
{
    builder.Clear();

    const std::vector<int64_t> dims = {1};
    const std::vector<int64_t> strides = {1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "t", DataType::FLOAT, &strides, &dims));

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.reserve(nodeTypes.size());
    for(const auto attr : nodeTypes)
    {
        nodes.push_back(CreateNodeDirect(builder, "n", DataType::FLOAT, attr, 0));
    }

    auto graph = CreateGraphDirect(
        builder, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    builder.Finish(graph);
    return fb::GraphWrapper(builder.GetBufferPointer(), builder.GetSize());
}

} // namespace

// Known fp32 reference values from TestTolerances.hpp
constexpr float kConvFwdFp32 = 1e-5f;
constexpr float kBnInferenceFp32 = 2e-4f;
constexpr float kPointwiseFp32 = 1e-5f;
constexpr float kFallback = 1e-3f;

// ── Single non-Pointwise op: both policies agree ────────────────────────────

TEST(TestToleranceResolver, SingleConvFwdMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::ConvolutionFwdAttributes});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kConvFwdFp32);
}

TEST(TestToleranceResolver, SingleConvFwdOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::ConvolutionFwdAttributes});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kConvFwdFp32);
}

// ── Discriminating case: BN inference (2e-4) then conv fwd (1e-5) ───────────
// MAX picks the loose one (2e-4), OUTPUT_OP picks the last non-Pointwise (1e-5).

TEST(TestToleranceResolver, BnInferenceConvFwdMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(
        b,
        {NodeAttributes::BatchnormInferenceAttributes, NodeAttributes::ConvolutionFwdAttributes});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kBnInferenceFp32);
}

TEST(TestToleranceResolver, BnInferenceConvFwdOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(
        b,
        {NodeAttributes::BatchnormInferenceAttributes, NodeAttributes::ConvolutionFwdAttributes});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kConvFwdFp32);
}

// ── All-Pointwise: OUTPUT_OP falls back to MAX ──────────────────────────────

TEST(TestToleranceResolver, AllPointwiseMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::PointwiseAttributes});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kPointwiseFp32);
}

TEST(TestToleranceResolver, AllPointwiseOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::PointwiseAttributes});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kPointwiseFp32);
}

// ── Empty graph: both return 1e-3 floor ─────────────────────────────────────

TEST(TestToleranceResolver, EmptyGraphMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kFallback);
}

TEST(TestToleranceResolver, EmptyGraphOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kFallback);
}

// ── Unknown op: conservative 1e-3 fallback ──────────────────────────────────

TEST(TestToleranceResolver, UnknownOpMaxAcrossNodes)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::NONE});
    EXPECT_FLOAT_EQ(tol::maxAcrossNodes(w, DataType::FLOAT), kFallback);
}

TEST(TestToleranceResolver, UnknownOpOutputOpTolerance)
{
    flatbuffers::FlatBufferBuilder b;
    auto w = buildGraph(b, {NodeAttributes::NONE});
    EXPECT_FLOAT_EQ(tol::outputOpTolerance(w, DataType::FLOAT), kFallback);
}

// NOLINTEND(readability-identifier-naming)
