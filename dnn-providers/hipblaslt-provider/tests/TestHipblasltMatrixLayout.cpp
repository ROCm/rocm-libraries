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
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;
    ASSERT_NE(tensorAttr, nullptr);

    EXPECT_NO_THROW({
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper tensorWrapper(
            tensorAttr);
        HipblasltMatrixLayout matLayout(tensorWrapper);
        EXPECT_EQ(matLayout.uid(), tensorWrapper.uid());
        EXPECT_NE(matLayout.matrixLayout(), nullptr);
    });
}

TEST(TestHipblasltMatrixLayout, TensorDescriptorIsValid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper tensorWrapper(
        tensorMap.begin()->second);
    HipblasltMatrixLayout matLayout(tensorWrapper);

    // The descriptor should be non-null and can be used in HipBLASLt API calls
    EXPECT_NE(matLayout.matrixLayout(), nullptr);
}

TEST(TestHipblasltMatrixLayout, CanCreateAndDestroyDefaultMatrix)
{
    EXPECT_NO_THROW({
        HipblasltMatrixLayout matLayout;
        EXPECT_EQ(matLayout.uid(), 0);
        EXPECT_EQ(matLayout.matrixLayout(), nullptr);
    });
}

TEST(TestHipblasltMatrixLayout, TensorWithEmptyShape)
{
    std::vector<int64_t> strides = {1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, &strides, nullptr);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper tensorWrapper(attrPtr);
    EXPECT_THROW(HipblasltMatrixLayout matLayout(tensorWrapper), std::invalid_argument);
}

TEST(TestHipblasltMatrixLayout, TensorWithEmptyStride)
{
    std::vector<int64_t> dims = {10, 16};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, nullptr, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper tensorWrapper(attrPtr);
    EXPECT_THROW(HipblasltMatrixLayout matLayout(tensorWrapper), std::invalid_argument);
}

TEST(TestHipblasltMatrixLayout, TensorWithInvalidShapeRank)
{
    std::vector<int64_t> dims = {10};
    std::vector<int64_t> strides = {1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, &strides, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper tensorWrapper(attrPtr);
    EXPECT_THROW(HipblasltMatrixLayout matLayout(tensorWrapper),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipblasltMatrixLayout, TensorWithInvalidMatrixType)
{
    std::vector<int64_t> dims = {8, 2, 4};
    std::vector<int64_t> strides = {1, 8, 16};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, &strides, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper tensorWrapper(attrPtr);
    EXPECT_THROW(HipblasltMatrixLayout matLayout(tensorWrapper),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipblasltMatrixLayout, SetBatchCount)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;

    EXPECT_NO_THROW({
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper tensorWrapper(
            tensorAttr);
        HipblasltMatrixLayout matLayout(tensorWrapper);
        matLayout.setBatchCount(2);
    });
}

TEST(TestHipblasltMatrixLayout, setStridedBatchOffset)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;

    EXPECT_NO_THROW({
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper tensorWrapper(
            tensorAttr);
        HipblasltMatrixLayout matLayout(tensorWrapper);
        matLayout.setStridedBatchOffset(2);
    });
}
