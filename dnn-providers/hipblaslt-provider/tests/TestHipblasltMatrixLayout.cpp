// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltMatrixLayout.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipblaslt_plugin;

TEST(TestHipblasltMatrixLayout, CanCreateAndDestroy)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper const graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;
    ASSERT_NE(tensorAttr, nullptr);

    EXPECT_NO_THROW({
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
            tensorAttr);
        HipblasltMatrixLayout const matLayout(tensorWrapper);
        EXPECT_EQ(matLayout.uid(), tensorWrapper.uid());
        EXPECT_NE(matLayout.matrixLayout(), nullptr);
    });
}

TEST(TestHipblasltMatrixLayout, TensorDescriptorIsValid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper const graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
        tensorMap.begin()->second);
    HipblasltMatrixLayout const matLayout(tensorWrapper);

    // The descriptor should be non-null and can be used in HipBLASLt API calls
    EXPECT_NE(matLayout.matrixLayout(), nullptr);
}

TEST(TestHipblasltMatrixLayout, CanCreateAndDestroyDefaultMatrix)
{
    EXPECT_NO_THROW({
        HipblasltMatrixLayout const matLayout;
        EXPECT_EQ(matLayout.uid(), 0);
        EXPECT_EQ(matLayout.matrixLayout(), nullptr);
    });
}

TEST(TestHipblasltMatrixLayout, RawDimensionsCreatesValidLayout)
{
    EXPECT_NO_THROW({
        // Byte (R_8I) layout from explicit dims, as used for scale transposes.
        HipblasltMatrixLayout const matLayout(HIP_R_8I, 4, 16, 4);
        // No associated tensor, so uid defaults to 0 while the handle is valid.
        EXPECT_EQ(matLayout.uid(), 0);
        EXPECT_NE(matLayout.matrixLayout(), nullptr);
    });
}

TEST(TestHipblasltMatrixLayout, TensorWithEmptyShape)
{
    std::vector<int64_t> const strides = {1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, &strides, nullptr);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
        attrPtr);
    EXPECT_THROW(HipblasltMatrixLayout const matLayout(tensorWrapper), std::invalid_argument);
}

TEST(TestHipblasltMatrixLayout, TensorWithEmptyStride)
{
    std::vector<int64_t> const dims = {10, 16};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, nullptr, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
        attrPtr);
    EXPECT_THROW(HipblasltMatrixLayout const matLayout(tensorWrapper), std::invalid_argument);
}

TEST(TestHipblasltMatrixLayout, TensorWithInvalidShapeRank)
{
    std::vector<int64_t> const dims = {10};
    std::vector<int64_t> const strides = {1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, &strides, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
        attrPtr);
    EXPECT_THROW(HipblasltMatrixLayout const matLayout(tensorWrapper),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipblasltMatrixLayout, TensorWithInvalidMatrixType)
{
    std::vector<int64_t> const dims = {8, 2, 4};
    std::vector<int64_t> const strides = {1, 8, 16};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, &strides, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
        attrPtr);
    EXPECT_THROW(HipblasltMatrixLayout const matLayout(tensorWrapper),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipblasltMatrixLayout, SetBatchCount)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper const graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;

    EXPECT_NO_THROW({
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
            tensorAttr);
        HipblasltMatrixLayout matLayout(tensorWrapper);
        matLayout.setBatchCount(2);
    });
}

TEST(TestHipblasltMatrixLayout, setStridedBatchOffset)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper const graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;

    EXPECT_NO_THROW({
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
            tensorAttr);
        HipblasltMatrixLayout matLayout(tensorWrapper);
        matLayout.setStridedBatchOffset(2);
    });
}

namespace
{
// Builds a FLOAT tensor with the given dims and strides and keeps the builder alive.
struct FloatTensorHolder
{
    flatbuffers::FlatBufferBuilder builder;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attrs = nullptr;

    FloatTensorHolder(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
    {
        builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            1,
            "t",
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            &strides,
            &dims));
        attrs = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
            builder.GetBufferPointer());
    }
};
} // namespace

TEST(TestHipblasltMatrixLayout, NonPositiveLeadingDimensionThrows)
{
    struct Row
    {
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
    };

    const std::vector<Row> rows = {
        // Row-major: the leading dimension is the row stride.
        {{6, 8}, {-1, 1}},
        {{6, 8}, {0, 1}},
        {{2, 6, 8}, {48, -1, 1}},
        // Column-major: the leading dimension is the column stride.
        {{6, 8}, {1, -1}},
        {{6, 8}, {1, 0}},
        {{2, 6, 8}, {48, 1, -1}},
    };

    for(const auto& row : rows)
    {
        const FloatTensorHolder holder(row.dims, row.strides);
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
            holder.attrs);
        EXPECT_THROW(HipblasltMatrixLayout const matLayout(tensorWrapper),
                     hipdnn_plugin_sdk::HipdnnPluginException)
            << "strides " << row.strides[row.strides.size() - 2] << ","
            << row.strides[row.strides.size() - 1];
    }
}

TEST(TestHipblasltMatrixLayout, NonPositiveStrideOfSizeOneDimIsNotALeadingDimension)
{
    struct Row
    {
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
    };

    const std::vector<Row> rows = {
        // A size-1 row count takes ld from the column count; its stride is ignored.
        {{1, 8}, {-1, 1}},
        {{1, 8}, {0, 1}},
        // A size-1 column count is row-major; the stride of that dim is ignored.
        {{6, 1}, {1, -5}},
        {{6, 1}, {1, 0}},
        {{1, 1}, {-3, -4}},
    };

    for(const auto& row : rows)
    {
        const FloatTensorHolder holder(row.dims, row.strides);
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper const tensorWrapper(
            holder.attrs);
        EXPECT_NO_THROW({
            HipblasltMatrixLayout const matLayout(tensorWrapper);
            EXPECT_NE(matLayout.matrixLayout(), nullptr);
        }) << "dims "
           << row.dims[0] << "x" << row.dims[1] << " strides " << row.strides[0] << ","
           << row.strides[1];
    }
}
