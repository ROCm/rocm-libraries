// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenUtils.hpp"
#include "common/MiopenHandleFixture.hpp"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

using namespace miopen_plugin;
using namespace miopen_utils;

TEST(TestMiopenUtils, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{42, reinterpret_cast<void*>(0x1234)}, {99, reinterpret_cast<void*>(0x5678)}};

    auto result = miopen_utils::findDeviceBuffer(99, buffers.data(), 2);
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(TestMiopenUtils, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(
        miopen_utils::findDeviceBuffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, TensorDataTypeToMiopenDataType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::FLOAT), miopenFloat);
    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::HALF), miopenHalf);
    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::BFLOAT16), miopenBFloat16);
}

TEST(TestMiopenUtils, TensorDataTypeToMiopenDataTypeThrowsOnUnsupported)
{
    // Use a value not in the enum
    EXPECT_THROW(miopen_utils::tensorDataTypeToMiopenDataType(
                     static_cast<hipdnn_flatbuffers_sdk::data_objects::DataType>(-1)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, FindTensorAttributesReturnsCorrectValue)
{
    flatbuffers::FlatBufferBuilder builder1;
    auto attrOffset1
        = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(builder1, 1);
    builder1.Finish(attrOffset1);

    flatbuffers::FlatBufferBuilder builder2;
    auto attrOffset2
        = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(builder2, 2);
    builder2.Finish(attrOffset2);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder1.GetBufferPointer());
    auto attrPtr2 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder2.GetBufferPointer());

    auto attrMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{
            {1, attrPtr1}, {2, attrPtr2}};

    EXPECT_EQ(miopen_utils::findTensorAttributes(attrMap, 1).uid(), 1);
    EXPECT_EQ(miopen_utils::findTensorAttributes(attrMap, 2).uid(), 2);
}

TEST(TestMiopenUtils, FindTensorAttributesThrowsIfNotFound)
{
    auto attrMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{};

    EXPECT_THROW(miopen_utils::findTensorAttributes(attrMap, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, GetSpatialDimCountReturnsCorrectValue)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, nullptr, &dims);
    builder.Finish(attrOffset);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    EXPECT_EQ(miopen_utils::getSpatialDimCount(*attrPtr1), 3);
}

TEST(TestMiopenUtils, GetSpatialDimCountThrowsOnInvalidDims)
{
    const std::vector<int64_t> dims = {1, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET, nullptr, &dims);
    builder.Finish(attrOffset);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(miopen_utils::getSpatialDimCount(*attrPtr1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, CreateBatchnormTensor4dPassthrough)
{
    // 4D NCHW tensor should pass through unchanged
    const std::vector<int64_t> dims = {2, 3, 14, 14};
    const std::vector<int64_t> strides = {588, 196, 14, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 42, "", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    auto tensorMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{
            {42, attrPtr}};

    auto result = miopen_utils::createBatchnormTensor(tensorMap, 42);

    EXPECT_EQ(result.uid(), 42);

    // Verify tensor is unchanged (4D with exact dims/strides)
    int numDims = 0;
    miopenGetTensorDescriptorSize(result.tensorDescriptor(), &numDims);
    ASSERT_EQ(numDims, 4);

    std::vector<int> resultDims(4);
    std::vector<int> resultStrides(4);
    miopenDataType_t dataType;
    miopenGetTensorDescriptor(
        result.tensorDescriptor(), &dataType, resultDims.data(), resultStrides.data());

    EXPECT_EQ(resultDims[0], 2);
    EXPECT_EQ(resultDims[1], 3);
    EXPECT_EQ(resultDims[2], 14);
    EXPECT_EQ(resultDims[3], 14);

    EXPECT_EQ(resultStrides[0], 588);
    EXPECT_EQ(resultStrides[1], 196);
    EXPECT_EQ(resultStrides[2], 14);
    EXPECT_EQ(resultStrides[3], 1);
}

TEST(TestMiopenUtils, CreateBatchnormTensor5dPassthrough)
{
    // 5D NCDHW tensor should pass through unchanged
    const std::vector<int64_t> dims = {2, 3, 4, 14, 14};
    const std::vector<int64_t> strides = {2352, 784, 196, 14, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 42, "", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    auto tensorMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{
            {42, attrPtr}};

    auto result = miopen_utils::createBatchnormTensor(tensorMap, 42);

    EXPECT_EQ(result.uid(), 42);

    // Verify tensor is unchanged (5D with exact dims/strides)
    int numDims = 0;
    miopenGetTensorDescriptorSize(result.tensorDescriptor(), &numDims);
    ASSERT_EQ(numDims, 5);

    std::vector<int> resultDims(5);
    std::vector<int> resultStrides(5);
    miopenDataType_t dataType;
    miopenGetTensorDescriptor(
        result.tensorDescriptor(), &dataType, resultDims.data(), resultStrides.data());

    EXPECT_EQ(resultDims[0], 2);
    EXPECT_EQ(resultDims[1], 3);
    EXPECT_EQ(resultDims[2], 4);
    EXPECT_EQ(resultDims[3], 14);
    EXPECT_EQ(resultDims[4], 14);

    EXPECT_EQ(resultStrides[0], 2352);
    EXPECT_EQ(resultStrides[1], 784);
    EXPECT_EQ(resultStrides[2], 196);
    EXPECT_EQ(resultStrides[3], 14);
    EXPECT_EQ(resultStrides[4], 1);
}

TEST(TestMiopenUtils, CreateBatchnormTensor3dNclPadsToNchw)
{
    // NCL (channels-first): dims [N, C, L], strides [C*L, L, 1]
    // C stride (14) > L stride (1) = channels-first
    const std::vector<int64_t> dims = {1, 3, 14};
    const std::vector<int64_t> strides = {42, 14, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 42, "", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    auto tensorMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{
            {42, attrPtr}};

    auto result = miopen_utils::createBatchnormTensor(tensorMap, 42);

    // Should be padded to 4D with W=1 and stride[3]=1 for NCHW
    int numDims = 0;
    miopenGetTensorDescriptorSize(result.tensorDescriptor(), &numDims);
    ASSERT_EQ(numDims, 4);

    std::vector<int> resultDims(4);
    std::vector<int> resultStrides(4);
    miopenDataType_t dataType;
    miopenGetTensorDescriptor(
        result.tensorDescriptor(), &dataType, resultDims.data(), resultStrides.data());

    EXPECT_EQ(resultDims[0], 1);
    EXPECT_EQ(resultDims[1], 3);
    EXPECT_EQ(resultDims[2], 14);
    EXPECT_EQ(resultDims[3], 1);

    EXPECT_EQ(resultStrides[3], 1); // W stride = 1 for channels-first
}

TEST(TestMiopenUtils, CreateBatchnormTensor3dNlcPadsToNhwc)
{
    // NLC (channels-last): dims [N, C, L], strides [C*L, 1, C]
    // C stride (1) < L stride (3) = channels-last
    const std::vector<int64_t> dims = {1, 3, 14};
    const std::vector<int64_t> strides = {42, 1, 3};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 42, "", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims);
    builder.Finish(attrOffset);

    auto attrPtr = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    auto tensorMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{
            {42, attrPtr}};

    auto result = miopen_utils::createBatchnormTensor(tensorMap, 42);

    // Should be padded to 4D with W=1 and stride[3]=C for NHWC
    int numDims = 0;
    miopenGetTensorDescriptorSize(result.tensorDescriptor(), &numDims);
    ASSERT_EQ(numDims, 4);

    std::vector<int> resultDims(4);
    std::vector<int> resultStrides(4);
    miopenDataType_t dataType;
    miopenGetTensorDescriptor(
        result.tensorDescriptor(), &dataType, resultDims.data(), resultStrides.data());

    EXPECT_EQ(resultDims[0], 1);
    EXPECT_EQ(resultDims[1], 3);
    EXPECT_EQ(resultDims[2], 14);
    EXPECT_EQ(resultDims[3], 1);

    EXPECT_EQ(resultStrides[3], 3); // W stride = C (channel size) for channels-last
}

TEST(TestMiopenUtils, MapPointwiseModeStandardRelu)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationRELU);
    EXPECT_DOUBLE_EQ(result.alpha, 0.0);
}

TEST(TestMiopenUtils, MapPointwiseModeReluNonZeroLowerClipOnly)
{
    const float lowerClip = 0.1f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, lowerClip);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ASSERT_THROW(mapPointwiseModeToMiopenActivation(*attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, MapPointwiseModeReluZeroLowerClipOnly)
{
    const float lowerClip = 0.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, lowerClip);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = mapPointwiseModeToMiopenActivation(*attr));

    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(lowerClip));
}

TEST(TestMiopenUtils, MapPointwiseModeClippedRelu)
{
    const float upperClip = 6.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        flatbuffers::nullopt,
        upperClip);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationCLIPPEDRELU);
    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(upperClip));
}

TEST(TestMiopenUtils, MapPointwiseModeLeakyRelu)
{
    const float slope = 0.01f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        slope);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationLEAKYRELU);
    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(slope));
}

TEST(TestMiopenUtils, MapPointwiseModeClamp)
{
    const float lowerClip = -1.0f;
    const float upperClip = 6.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        lowerClip,
        upperClip);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationCLAMP);
    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(lowerClip));
    EXPECT_DOUBLE_EQ(result.beta, static_cast<double>(upperClip));
}

TEST(TestMiopenUtils, MapPointwiseModeSigmoid)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationLOGISTIC);
}

TEST(TestMiopenUtils, MapPointwiseModeTanh)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_BWD);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationTANH);
    EXPECT_DOUBLE_EQ(result.alpha, 1.0);
    EXPECT_DOUBLE_EQ(result.beta, 1.0);
}

TEST(TestMiopenUtils, MapPointwiseModeEluWithCustomAlpha)
{
    const float eluAlpha = 2.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        0,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        eluAlpha);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationELU);
    EXPECT_DOUBLE_EQ(result.alpha, static_cast<double>(eluAlpha));
}

TEST(TestMiopenUtils, MapPointwiseModeEluWithDefaultAlpha)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_BWD);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationELU);
    EXPECT_DOUBLE_EQ(result.alpha, 1.0);
}

TEST(TestMiopenUtils, MapPointwiseModeSoftplusWithoutBeta)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationSOFTRELU);
}

TEST(TestMiopenUtils, MapPointwiseModeSoftplusWithBetaOne)
{
    const float beta = 1.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        0,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        beta);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationSOFTRELU);
}

TEST(TestMiopenUtils, MapPointwiseModeSoftplusWithInvalidBeta)
{
    const float beta = 2.0f;
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        0,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        beta);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ASSERT_THROW(miopen_utils::mapPointwiseModeToMiopenActivation(*attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenUtils, MapPointwiseModeAbs)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ABS);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationABS);
}

TEST(TestMiopenUtils, MapPointwiseModeIdentity)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ActivationParams result;
    ASSERT_NO_THROW(result = miopen_utils::mapPointwiseModeToMiopenActivation(*attr));
    EXPECT_EQ(result.mode, miopenActivationPASTHRU);
}

TEST(TestMiopenUtils, MapPointwiseModeUnsupported)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD);
    builder.Finish(attrOffset);
    const auto* attr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    ASSERT_THROW(miopen_utils::mapPointwiseModeToMiopenActivation(*attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// =============================================================================
// GPU-required tests for ScopedTuningPolicy
// =============================================================================

class TestGpuScopedTuningPolicy : public test_common::MiopenHandleFixture
{
};

TEST_F(TestGpuScopedTuningPolicy, SetsSearchPolicyWhenBenchmarkingEnabled)
{
    {
        const ScopedTuningPolicy guard(_miopenHandle, true);

        miopenTuningPolicy_t currentPolicy;
        auto status = miopenGetTuningPolicy(_miopenHandle, &currentPolicy);
        ASSERT_EQ(status, miopenStatusSuccess);
        EXPECT_EQ(currentPolicy, miopenTuningPolicySearch);
    }
}

TEST_F(TestGpuScopedTuningPolicy, SetsNonePolicyWhenBenchmarkingDisabled)
{
    {
        const ScopedTuningPolicy guard(_miopenHandle, false);

        miopenTuningPolicy_t currentPolicy;
        auto status = miopenGetTuningPolicy(_miopenHandle, &currentPolicy);
        ASSERT_EQ(status, miopenStatusSuccess);
        EXPECT_EQ(currentPolicy, miopenTuningPolicyNone);
    }
}

TEST_F(TestGpuScopedTuningPolicy, RestoresOriginalPolicyOnDestruction)
{
    // Set a non-default policy first
    auto preSetStatus = miopenSetTuningPolicy(_miopenHandle, miopenTuningPolicyDbUpdate);
    ASSERT_EQ(preSetStatus, miopenStatusSuccess);

    {
        const ScopedTuningPolicy guard(_miopenHandle, true);

        // Verify it was changed during scope
        miopenTuningPolicy_t duringPolicy;
        miopenGetTuningPolicy(_miopenHandle, &duringPolicy);
        EXPECT_EQ(duringPolicy, miopenTuningPolicySearch);
    }

    // Verify it was restored to original
    miopenTuningPolicy_t afterPolicy;
    auto status = miopenGetTuningPolicy(_miopenHandle, &afterPolicy);
    ASSERT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(afterPolicy, miopenTuningPolicyDbUpdate);
}

TEST_F(TestGpuScopedTuningPolicy, RestoresToNoneWhenOriginalWasNone)
{
    // Ensure policy starts as None
    miopenSetTuningPolicy(_miopenHandle, miopenTuningPolicyNone);

    {
        const ScopedTuningPolicy guard(_miopenHandle, true);
    }

    miopenTuningPolicy_t afterPolicy;
    auto status = miopenGetTuningPolicy(_miopenHandle, &afterPolicy);
    ASSERT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(afterPolicy, miopenTuningPolicyNone);
}

TEST_F(TestGpuScopedTuningPolicy, NestedScopesRestoreCorrectly)
{
    miopenSetTuningPolicy(_miopenHandle, miopenTuningPolicyNone);

    {
        const ScopedTuningPolicy outerGuard(_miopenHandle, true);

        miopenTuningPolicy_t afterOuterSet;
        miopenGetTuningPolicy(_miopenHandle, &afterOuterSet);
        EXPECT_EQ(afterOuterSet, miopenTuningPolicySearch);

        {
            const ScopedTuningPolicy innerGuard(_miopenHandle, false);

            miopenTuningPolicy_t afterInnerSet;
            miopenGetTuningPolicy(_miopenHandle, &afterInnerSet);
            EXPECT_EQ(afterInnerSet, miopenTuningPolicyNone);
        }

        // After inner scope, should restore to Search (what outer set)
        miopenTuningPolicy_t afterInnerDestroy;
        miopenGetTuningPolicy(_miopenHandle, &afterInnerDestroy);
        EXPECT_EQ(afterInnerDestroy, miopenTuningPolicySearch);
    }

    // After outer scope, should restore to None (original)
    miopenTuningPolicy_t finalPolicy;
    miopenGetTuningPolicy(_miopenHandle, &finalPolicy);
    EXPECT_EQ(finalPolicy, miopenTuningPolicyNone);
}
