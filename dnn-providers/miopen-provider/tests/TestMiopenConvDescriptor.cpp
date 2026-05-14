// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <limits>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"

using namespace miopen_plugin;

TEST(TestMiopenConvDescriptor, CreateValidDescriptorFwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1);

    miopenStatus_t status;
    int returnedSpatialDimCount = 0;
    status = miopenGetConvolutionSpatialDim(convDesc.convDescriptor(), &returnedSpatialDimCount);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(returnedSpatialDimCount, spatialDimCount);

    std::vector<int> returnedPadding(spatialDimCount);
    std::vector<int> returnedStride(spatialDimCount);
    std::vector<int> returnedDilation(spatialDimCount);
    status = miopenGetConvolutionNdDescriptor(convDesc.convDescriptor(),
                                              static_cast<int>(spatialDimCount),
                                              nullptr,
                                              returnedPadding.data(),
                                              returnedStride.data(),
                                              returnedDilation.data(),
                                              nullptr);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_TRUE(std::equal(returnedPadding.begin(), returnedPadding.end(), prePadding.begin()));
    EXPECT_TRUE(std::equal(returnedStride.begin(), returnedStride.end(), stride.begin()));
    EXPECT_TRUE(std::equal(returnedDilation.begin(), returnedDilation.end(), dilation.begin()));
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongSpatialDimCountFwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(
                     static_cast<size_t>(std::numeric_limits<int>::max()) + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() - 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(prePadding.size(), *attrPtr, 1));
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongConvModeFwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CONVOLUTION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnAsymmetricPaddingFwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongPaddingFwd)
{
    const std::vector<int64_t> prePadding{0, -1, 0};
    const std::vector<int64_t> postPadding{0, -1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongStrideFwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 0, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongDilationFwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 0, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, CreateValidDescriptorBwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>(
            builder.GetBufferPointer());

    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1);

    miopenStatus_t status;
    int returnedSpatialDimCount = 0;
    status = miopenGetConvolutionSpatialDim(convDesc.convDescriptor(), &returnedSpatialDimCount);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(returnedSpatialDimCount, spatialDimCount);

    std::vector<int> returnedPadding(spatialDimCount);
    std::vector<int> returnedStride(spatialDimCount);
    std::vector<int> returnedDilation(spatialDimCount);
    status = miopenGetConvolutionNdDescriptor(convDesc.convDescriptor(),
                                              static_cast<int>(spatialDimCount),
                                              nullptr,
                                              returnedPadding.data(),
                                              returnedStride.data(),
                                              returnedDilation.data(),
                                              nullptr);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_TRUE(std::equal(returnedPadding.begin(), returnedPadding.end(), prePadding.begin()));
    EXPECT_TRUE(std::equal(returnedStride.begin(), returnedStride.end(), stride.begin()));
    EXPECT_TRUE(std::equal(returnedDilation.begin(), returnedDilation.end(), dilation.begin()));
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongSpatialDimCountBwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(
                     static_cast<size_t>(std::numeric_limits<int>::max()) + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() - 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(prePadding.size(), *attrPtr, 1));
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongConvModeBwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CONVOLUTION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnAsymmetricPaddingBwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongPaddingBwd)
{
    const std::vector<int64_t> prePadding{0, -1, 0};
    const std::vector<int64_t> postPadding{0, -1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongStrideBwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 0, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongDilationBwd)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 0, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, CreateValidDescriptorWrw)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>(
            builder.GetBufferPointer());

    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1);

    miopenStatus_t status;
    int returnedSpatialDimCount = 0;
    status = miopenGetConvolutionSpatialDim(convDesc.convDescriptor(), &returnedSpatialDimCount);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(returnedSpatialDimCount, spatialDimCount);

    std::vector<int> returnedPadding(spatialDimCount);
    std::vector<int> returnedStride(spatialDimCount);
    std::vector<int> returnedDilation(spatialDimCount);
    status = miopenGetConvolutionNdDescriptor(convDesc.convDescriptor(),
                                              static_cast<int>(spatialDimCount),
                                              nullptr,
                                              returnedPadding.data(),
                                              returnedStride.data(),
                                              returnedDilation.data(),
                                              nullptr);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_TRUE(std::equal(returnedPadding.begin(), returnedPadding.end(), prePadding.begin()));
    EXPECT_TRUE(std::equal(returnedStride.begin(), returnedStride.end(), stride.begin()));
    EXPECT_TRUE(std::equal(returnedDilation.begin(), returnedDilation.end(), dilation.begin()));
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongSpatialDimCountWrw)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(
                     static_cast<size_t>(std::numeric_limits<int>::max()) + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() - 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(prePadding.size(), *attrPtr, 1));
    EXPECT_THROW(MiopenConvDescriptor convDesc(prePadding.size() + 1, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongConvModeWrw)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CONVOLUTION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnAsymmetricPaddingWrw)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongPaddingWrw)
{
    const std::vector<int64_t> prePadding{0, -1, 0};
    const std::vector<int64_t> postPadding{0, -1, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongStrideWrw)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 0, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, ThrowsOnWrongDilationWrw)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 0, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, AcceptsValidGroupCount)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1));
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 2));
    EXPECT_NO_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 4));
}

TEST(TestMiopenConvDescriptor, VerifiesGroupCountSetCorrectly)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;
    int groupCount = 4;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, groupCount);

    int returnedGroupCount = 0;
    miopenStatus_t status
        = miopenGetConvolutionGroupCount(convDesc.convDescriptor(), &returnedGroupCount);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(returnedGroupCount, groupCount);
}

TEST(TestMiopenConvDescriptor, ThrowsOnInvalidGroupCount)
{
    const std::vector<int64_t> prePadding{0, 0, 0};
    const std::vector<int64_t> postPadding{0, 0, 0};
    const std::vector<int64_t> stride{1, 1, 1};
    const std::vector<int64_t> dilation{1, 1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 3;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 0),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, -1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenConvDescriptor, SetsDeterministicAttributeWhenEnabled)
{
    const std::vector<int64_t> prePadding{0, 0};
    const std::vector<int64_t> postPadding{0, 0};
    const std::vector<int64_t> stride{1, 1};
    const std::vector<int64_t> dilation{1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 2;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    // Create descriptor with deterministic enabled
    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1, true);

    // Verify the deterministic attribute is set to 1
    int deterministicValue = 0;
    miopenStatus_t status = miopenGetConvolutionAttribute(
        convDesc.convDescriptor(), MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, &deterministicValue);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(deterministicValue, 1);
}

TEST(TestMiopenConvDescriptor, DeterministicAttributeDefaultsToDisabled)
{
    const std::vector<int64_t> prePadding{0, 0};
    const std::vector<int64_t> postPadding{0, 0};
    const std::vector<int64_t> stride{1, 1};
    const std::vector<int64_t> dilation{1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 2;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>(
            builder.GetBufferPointer());

    // Create descriptor with default (deterministic disabled)
    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1);

    // Verify the deterministic attribute is 0 (disabled)
    int deterministicValue = -1;
    miopenStatus_t status = miopenGetConvolutionAttribute(
        convDesc.convDescriptor(), MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, &deterministicValue);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(deterministicValue, 0);
}

TEST(TestMiopenConvDescriptor, SetsDeterministicAttributeForBwdDescriptor)
{
    const std::vector<int64_t> prePadding{0, 0};
    const std::vector<int64_t> postPadding{0, 0};
    const std::vector<int64_t> stride{1, 1};
    const std::vector<int64_t> dilation{1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 2;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>(
            builder.GetBufferPointer());

    // Create descriptor with deterministic enabled
    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1, true);

    // Verify the deterministic attribute is set to 1
    int deterministicValue = 0;
    miopenStatus_t status = miopenGetConvolutionAttribute(
        convDesc.convDescriptor(), MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, &deterministicValue);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(deterministicValue, 1);
}

TEST(TestMiopenConvDescriptor, SetsDeterministicAttributeForWrwDescriptor)
{
    const std::vector<int64_t> prePadding{0, 0};
    const std::vector<int64_t> postPadding{0, 0};
    const std::vector<int64_t> stride{1, 1};
    const std::vector<int64_t> dilation{1, 1};
    const auto convMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    size_t spatialDimCount = 2;

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder, 0, 0, 0, &prePadding, &postPadding, &stride, &dilation, convMode);
    builder.Finish(attrOffset);
    auto attrPtr
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>(
            builder.GetBufferPointer());

    // Create descriptor with deterministic enabled
    MiopenConvDescriptor convDesc(spatialDimCount, *attrPtr, 1, true);

    // Verify the deterministic attribute is set to 1
    int deterministicValue = 0;
    miopenStatus_t status = miopenGetConvolutionAttribute(
        convDesc.convDescriptor(), MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, &deterministicValue);
    EXPECT_EQ(status, miopenStatusSuccess);
    EXPECT_EQ(deterministicValue, 1);
}
