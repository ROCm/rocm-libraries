// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::GetGraph;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
using hipdnn_test_sdk::utilities::createReductionGraph;
using hipdnn_test_sdk::utilities::ReductionGraphSpec;

namespace
{

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>& values)
{
    return {values.begin(), values.end()};
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
