// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenTensor.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_plugin;

TEST(TestMiopenTensor, CanCreateAndDestroy)
{
    // Use a real tensor attributes from a valid batchnorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    // Get the first tensor attributes from the tensor map
    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;
    ASSERT_NE(tensorAttr, nullptr);

    // Construct and destroy MiopenTensor
    EXPECT_NO_THROW({
        MiopenTensor tensor(*tensorAttr);
        EXPECT_EQ(tensor.uid(), tensorAttr->uid());
        EXPECT_NE(tensor.tensorDescriptor(), nullptr);
    });
}

TEST(TestMiopenTensor, TensorDescriptorIsValid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;
    MiopenTensor tensor(*tensorAttr);

    // The descriptor should be non-null and can be used in MIOpen API calls
    EXPECT_NE(tensor.tensorDescriptor(), nullptr);
}

TEST(TestMiopenTensor, ConstructorWithDimsAndStridesSucceeds)
{
    constexpr int64_t UID = 42;
    const std::vector<int64_t> dims = {2, 16, 224};
    const std::vector<int64_t> strides = {static_cast<int64_t>(16) * 224, 224, 1};

    EXPECT_NO_THROW({
        MiopenTensor tensor(
            UID, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, dims, strides);
        EXPECT_NE(tensor.tensorDescriptor(), nullptr);
    });
}

TEST(TestMiopenTensor, ConstructorThrowsOnDimsStridesSizeMismatch)
{
    constexpr int64_t UID = 1;
    const std::vector<int64_t> dims = {2, 16, 224};
    const std::vector<int64_t> strides = {224, 1}; // Wrong size

    EXPECT_THROW(
        MiopenTensor(UID, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, dims, strides),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenTensor, ConstructorThrowsOnNegativeDimension)
{
    constexpr int64_t UID = 1;
    const std::vector<int64_t> dims = {2, -16, 224}; // Negative dimension
    const std::vector<int64_t> strides = {static_cast<int64_t>(16) * 224, 224, 1};

    EXPECT_THROW(
        MiopenTensor(UID, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, dims, strides),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenTensor, ConstructorThrowsOnNegativeStride)
{
    constexpr int64_t UID = 1;
    const std::vector<int64_t> dims = {2, 16, 224};
    const std::vector<int64_t> strides
        = {static_cast<int64_t>(16) * 224, -224, 1}; // Negative stride

    EXPECT_THROW(
        MiopenTensor(UID, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, dims, strides),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenTensor, ConstructorSetsCorrectUid)
{
    constexpr int64_t EXPECTED_UID = 12345;
    const std::vector<int64_t> dims = {2, 16};
    const std::vector<int64_t> strides = {16, 1};

    MiopenTensor tensor(
        EXPECTED_UID, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, dims, strides);

    EXPECT_EQ(tensor.uid(), EXPECTED_UID);
}
