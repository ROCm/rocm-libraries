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
using hipdnn_flatbuffers_sdk::data_objects::PaddingMode;
using hipdnn_flatbuffers_sdk::data_objects::ResampleMode;
using hipdnn_test_sdk::utilities::createResampleFwdGraph;
using hipdnn_test_sdk::utilities::ResampleFwdGraphSpec;

namespace
{

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>& values)
{
    return {values.begin(), values.end()};
}

} // namespace

// ============================================================================
// ResampleFwdGraphSpec and createResampleFwdGraph
// ============================================================================

TEST(TestResampleFwdGraphSpec, PackedSizesFloorModeOutputAndMatchesPaddingMode)
{
    // Y: H = (7 + 2 - 3) / 2 + 1 = 4, W = (5 - 2) / 1 + 1 = 4.
    const auto spec = ResampleFwdGraphSpec::packed(ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                   {1, 2, 7, 5},
                                                   {3, 2},
                                                   {2, 1},
                                                   {1, 0},
                                                   DataType::HALF);

    EXPECT_EQ(spec.mode, ResampleMode::AVGPOOL_INCLUDE_PADDING);
    EXPECT_EQ(spec.paddingMode, PaddingMode::ZERO_PAD);
    EXPECT_EQ(spec.x.dims, (std::vector<int64_t>{1, 2, 7, 5}));
    EXPECT_EQ(spec.x.strides, (std::vector<int64_t>{70, 35, 5, 1}));
    EXPECT_EQ(spec.x.dataType, DataType::HALF);
    EXPECT_EQ(spec.y.dims, (std::vector<int64_t>{1, 2, 4, 4}));
    EXPECT_EQ(spec.y.strides, (std::vector<int64_t>{32, 16, 4, 1}));
    EXPECT_EQ(spec.y.dataType, DataType::HALF);
    EXPECT_EQ(spec.window, (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(spec.stride, (std::vector<int64_t>{2, 1}));
    EXPECT_EQ(spec.prePadding, (std::vector<int64_t>{1, 0}));
    EXPECT_EQ(spec.postPadding, (std::vector<int64_t>{1, 0}));
    EXPECT_FALSE(spec.index.has_value());
    EXPECT_FALSE(spec.generateIndex.has_value());
}

TEST(TestResampleFwdGraphSpec, PackedWithIndexTypeAddsChannelsFirstIndexShapedLikeY)
{
    const auto spec = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                                   {2, 3, 6, 6},
                                                   {3, 3},
                                                   {2, 2},
                                                   {1, 1},
                                                   DataType::FLOAT,
                                                   DataType::INT8);

    EXPECT_EQ(spec.paddingMode, PaddingMode::NEG_INF_PAD);
    ASSERT_TRUE(spec.index.has_value());
    EXPECT_EQ(spec.index->uid, ResampleFwdGraphSpec::INDEX_UID);
    EXPECT_EQ(spec.index->dims, spec.y.dims);
    EXPECT_EQ(spec.index->strides, (std::vector<int64_t>{27, 9, 3, 1}));
    EXPECT_EQ(spec.index->dataType, DataType::INT8);
    EXPECT_EQ(spec.generateIndex, std::optional<bool>(true));
}

TEST(TestResampleFwdGraphSpec, MakeChannelsLastRestridesXAndYButNotTheIndex)
{
    auto spec = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                             {2, 3, 6, 6},
                                             {3, 3},
                                             {2, 2},
                                             {1, 1},
                                             DataType::FLOAT,
                                             DataType::INT32);
    spec.makeChannelsLast();

    EXPECT_EQ(spec.x.strides, (std::vector<int64_t>{108, 1, 18, 3}));
    EXPECT_EQ(spec.y.strides, (std::vector<int64_t>{27, 1, 9, 3}));
    ASSERT_TRUE(spec.index.has_value());
    EXPECT_EQ(spec.index->strides, (std::vector<int64_t>{27, 9, 3, 1}));
}

TEST(TestCreateResampleFwdGraph, WritesTensorsAndNodeAttributes)
{
    auto spec = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                             {2, 3, 6, 6},
                                             {3, 3},
                                             {2, 2},
                                             {1, 1},
                                             DataType::FLOAT,
                                             DataType::INT32);
    spec.computeDataType = DataType::HALF;

    const auto builder = createResampleFwdGraph(spec);
    const auto* graph = GetGraph(builder.GetBufferPointer());

    ASSERT_EQ(graph->tensors()->size(), 3u);
    EXPECT_EQ(graph->tensors()->Get(2)->uid(), ResampleFwdGraphSpec::INDEX_UID);
    EXPECT_EQ(graph->tensors()->Get(2)->data_type(), DataType::INT32);
    ASSERT_EQ(graph->nodes()->size(), 1u);
    EXPECT_EQ(graph->nodes()->Get(0)->compute_data_type(), DataType::HALF);
    EXPECT_FALSE(graph->is_override_shape_enabled());

    const auto* attributes = graph->nodes()->Get(0)->attributes_as_ResampleFwdAttributes();
    ASSERT_NE(attributes, nullptr);
    EXPECT_EQ(attributes->x_tensor_uid(), ResampleFwdGraphSpec::X_UID);
    EXPECT_EQ(attributes->y_tensor_uid(), ResampleFwdGraphSpec::Y_UID);
    ASSERT_TRUE(attributes->index_tensor_uid().has_value());
    EXPECT_EQ(attributes->index_tensor_uid().value(), ResampleFwdGraphSpec::INDEX_UID);
    EXPECT_EQ(toVector(*attributes->window()), (std::vector<int64_t>{3, 3}));
    EXPECT_EQ(toVector(*attributes->stride()), (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(toVector(*attributes->pre_padding()), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(toVector(*attributes->post_padding()), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(attributes->resample_mode(), ResampleMode::MAXPOOL);
    EXPECT_EQ(attributes->padding_mode(), PaddingMode::NEG_INF_PAD);
    ASSERT_TRUE(attributes->generate_index().has_value());
    EXPECT_TRUE(attributes->generate_index().value());
}

TEST(TestCreateResampleFwdGraph, LeavesOutTensorNotInGraph)
{
    ResampleFwdGraphSpec spec;
    spec.y.inGraph = false;

    const auto builder = createResampleFwdGraph(spec);
    const auto* graph = GetGraph(builder.GetBufferPointer());

    ASSERT_EQ(graph->tensors()->size(), 1u);
    EXPECT_EQ(graph->tensors()->Get(0)->uid(), ResampleFwdGraphSpec::X_UID);
    // The node still names y, so its output uid dangles.
    const auto* attributes = graph->nodes()->Get(0)->attributes_as_ResampleFwdAttributes();
    ASSERT_NE(attributes, nullptr);
    EXPECT_EQ(attributes->y_tensor_uid(), ResampleFwdGraphSpec::Y_UID);
    EXPECT_FALSE(attributes->index_tensor_uid().has_value());
}

TEST(TestCreateResampleFwdGraph, WritesNoStridesOrPoolingVectorsWhenUnset)
{
    ResampleFwdGraphSpec spec;
    spec.x.strides = std::nullopt;
    spec.window = std::nullopt;
    spec.generateIndex = false;

    const auto builder = createResampleFwdGraph(spec);
    const auto* graph = GetGraph(builder.GetBufferPointer());

    ASSERT_EQ(graph->tensors()->size(), 2u);
    EXPECT_EQ(graph->tensors()->Get(0)->strides(), nullptr);
    ASSERT_NE(graph->tensors()->Get(1)->strides(), nullptr);
    const auto* attributes = graph->nodes()->Get(0)->attributes_as_ResampleFwdAttributes();
    ASSERT_NE(attributes, nullptr);
    EXPECT_EQ(attributes->window(), nullptr);
    ASSERT_NE(attributes->stride(), nullptr);
    ASSERT_TRUE(attributes->generate_index().has_value());
    EXPECT_FALSE(attributes->generate_index().value());
}

TEST(TestCreateResampleFwdGraph, EmitsAliasedTensorOnce)
{
    ResampleFwdGraphSpec spec;
    spec.y.uid = spec.x.uid;

    const auto builder = createResampleFwdGraph(spec);
    const auto* graph = GetGraph(builder.GetBufferPointer());

    ASSERT_EQ(graph->tensors()->size(), 1u);
    const auto* attributes = graph->nodes()->Get(0)->attributes_as_ResampleFwdAttributes();
    ASSERT_NE(attributes, nullptr);
    EXPECT_EQ(attributes->x_tensor_uid(), attributes->y_tensor_uid());
}

TEST(TestCreateResampleFwdGraph, WritesNodeCountAndOverrideShapeFlag)
{
    ResampleFwdGraphSpec spec;
    spec.nodeCount = 2;
    spec.overrideShapeEnabled = true;

    const auto builder = createResampleFwdGraph(spec);
    const auto* graph = GetGraph(builder.GetBufferPointer());

    EXPECT_EQ(graph->nodes()->size(), 2u);
    EXPECT_TRUE(graph->is_override_shape_enabled());
}
