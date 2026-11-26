// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::GetGraph;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
using hipdnn_test_sdk::utilities::createReductionGraph;
using hipdnn_test_sdk::utilities::createValidMxMatmulGraph;
using hipdnn_test_sdk::utilities::ReductionGraphSpec;

namespace
{

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>& values)
{
    return {values.begin(), values.end()};
}

std::vector<int64_t> tensorStrides(const flatbuffers::FlatBufferBuilder& builder,
                                   const std::string& name)
{
    const auto* graph = GetGraph(builder.GetBufferPointer());
    for(const auto* tensor : *graph->tensors())
    {
        if(tensor->name()->str() == name)
        {
            return {tensor->strides()->begin(), tensor->strides()->end()};
        }
    }
    ADD_FAILURE() << "no tensor named " << name;
    return {};
}

// Builds the default MX matmul graph (scale A {32, 4}, scale B {4, 32}) with the given
// scale strides; an empty vector keeps the helper's default.
flatbuffers::FlatBufferBuilder
    createMxMatmulGraphWithScaleStrides(const std::vector<int64_t>& scaleAStrides,
                                        const std::vector<int64_t>& scaleBStrides)
{
    return createValidMxMatmulGraph({32, 128},
                                    {1, 32},
                                    {128, 32},
                                    {32, 1},
                                    {32, 32},
                                    {32, 1},
                                    {32, 4},
                                    {4, 32},
                                    DataType::FP8_E4M3,
                                    DataType::HALF,
                                    DataType::FP8_E8M0,
                                    DataType::FP8_E8M0,
                                    DataType::FLOAT,
                                    32,
                                    false,
                                    false,
                                    DataType::UNSET,
                                    scaleAStrides,
                                    scaleBStrides);
}

} // namespace

// ============================================================================
// ReductionGraphSpec and createReductionGraph
// ============================================================================

TEST(TestReductionGraphSpec, PackedSetsModeDimsPackedStridesAndDataType)
{
    const auto spec
        = ReductionGraphSpec::packed(ReductionMode::MUL, {2, 3, 4}, {2, 1, 1}, DataType::HALF);

    EXPECT_EQ(spec.mode, ReductionMode::MUL);
    EXPECT_EQ(spec.x.dims, (std::vector<int64_t>{2, 3, 4}));
    ASSERT_TRUE(spec.x.strides.has_value());
    EXPECT_EQ(*spec.x.strides, (std::vector<int64_t>{12, 4, 1}));
    EXPECT_EQ(spec.y.dims, (std::vector<int64_t>{2, 1, 1}));
    ASSERT_TRUE(spec.y.strides.has_value());
    EXPECT_EQ(*spec.y.strides, (std::vector<int64_t>{1, 1, 1}));
    EXPECT_EQ(spec.dataType, DataType::HALF);
    EXPECT_EQ(spec.outputDataType(), DataType::HALF);
}

TEST(TestCreateReductionGraph, LeavesOutTensorNotInGraph)
{
    ReductionGraphSpec spec;
    spec.y.inGraph = false;

    const auto builder = createReductionGraph(spec);
    const auto* graph = GetGraph(builder.GetBufferPointer());

    ASSERT_EQ(graph->tensors()->size(), 1u);
    EXPECT_EQ(graph->tensors()->Get(0)->uid(), ReductionGraphSpec::X_UID);
    // The node still names y, so its output uid dangles.
    const auto* attributes = graph->nodes()->Get(0)->attributes_as_ReductionAttributes();
    ASSERT_NE(attributes, nullptr);
    EXPECT_EQ(attributes->out_tensor_uid(), ReductionGraphSpec::Y_UID);
}

TEST(TestCreateReductionGraph, WritesNoStridesWhenUnset)
{
    ReductionGraphSpec spec;
    spec.x.strides = std::nullopt;

    const auto builder = createReductionGraph(spec);
    const auto* tensors = GetGraph(builder.GetBufferPointer())->tensors();

    ASSERT_EQ(tensors->size(), 2u);
    EXPECT_EQ(tensors->Get(0)->strides(), nullptr);
    ASSERT_NE(tensors->Get(1)->strides(), nullptr);
    EXPECT_EQ(toVector(*tensors->Get(1)->strides()), *spec.y.strides);
}

TEST(TestCreateReductionGraph, EmitsInPlaceTensorOnce)
{
    ReductionGraphSpec spec;
    spec.y.uid = spec.x.uid;

    const auto builder = createReductionGraph(spec);
    const auto* graph = GetGraph(builder.GetBufferPointer());

    ASSERT_EQ(graph->tensors()->size(), 1u);
    EXPECT_EQ(graph->tensors()->Get(0)->name()->str(), "x");
    const auto* attributes = graph->nodes()->Get(0)->attributes_as_ReductionAttributes();
    ASSERT_NE(attributes, nullptr);
    EXPECT_EQ(attributes->in_tensor_uid(), ReductionGraphSpec::X_UID);
    EXPECT_EQ(attributes->out_tensor_uid(), ReductionGraphSpec::X_UID);
}

// ============================================================================
// createValidMxMatmulGraph scale strides
// ============================================================================

TEST(TestCreateValidMxMatmulGraph, DefaultsToRowMajorScaleStrides)
{
    const auto builder = createValidMxMatmulGraph();

    EXPECT_EQ(tensorStrides(builder, "scale_a"), (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(tensorStrides(builder, "scale_b"), (std::vector<int64_t>{32, 1}));
}

TEST(TestCreateValidMxMatmulGraph, UsesGivenScaleStrides)
{
    const auto builder = createMxMatmulGraphWithScaleStrides({1, 32}, {1, 4});

    EXPECT_EQ(tensorStrides(builder, "scale_a"), (std::vector<int64_t>{1, 32}));
    EXPECT_EQ(tensorStrides(builder, "scale_b"), (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(tensorStrides(builder, "x_a"), (std::vector<int64_t>{1, 32}));
    EXPECT_EQ(tensorStrides(builder, "x_b"), (std::vector<int64_t>{32, 1}));
}

TEST(TestCreateValidMxMatmulGraph, OverridesEachScaleSeparately)
{
    const auto onlyA = createMxMatmulGraphWithScaleStrides({1, 32}, {});
    EXPECT_EQ(tensorStrides(onlyA, "scale_a"), (std::vector<int64_t>{1, 32}));
    EXPECT_EQ(tensorStrides(onlyA, "scale_b"), (std::vector<int64_t>{32, 1}));

    const auto onlyB = createMxMatmulGraphWithScaleStrides({}, {1, 4});
    EXPECT_EQ(tensorStrides(onlyB, "scale_a"), (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(tensorStrides(onlyB, "scale_b"), (std::vector<int64_t>{1, 4}));
}
