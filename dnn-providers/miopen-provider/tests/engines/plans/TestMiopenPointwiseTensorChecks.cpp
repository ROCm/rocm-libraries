// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenPointwiseTensorChecks.hpp"

#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace miopen_plugin;

namespace
{

flatbuffers::FlatBufferBuilder
    buildTensorAttrBuffer(int64_t uid,
                          hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                          const std::vector<int64_t>* dims,
                          const std::vector<int64_t>* strides,
                          bool isVirtual = false)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, uid, "t", dataType, strides, dims, isVirtual);
    builder.Finish(attrOffset);
    return builder;
}

} // namespace

TEST(TestMiopenPointwiseTensorChecks, ValidatePointwiseIoTensorsAcceptsUniformFloatTensors)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    const std::vector<int64_t> strides = {48, 16, 4, 1};

    auto b1 = buildTensorAttrBuffer(
        1, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &dims, &strides);
    auto b2 = buildTensorAttrBuffer(
        2, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &dims, &strides);

    const auto* t1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b1.GetBufferPointer());
    const auto* t2 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b2.GetBufferPointer());

    EXPECT_NO_THROW(pointwise_applicability::validatePointwiseIoTensors({t1, t2}, "Test"));
}

TEST(TestMiopenPointwiseTensorChecks, ValidatePointwiseIoTensorsRejectsVirtualTensor)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    const std::vector<int64_t> strides = {48, 16, 4, 1};

    auto b1 = buildTensorAttrBuffer(1,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                    &dims,
                                    &strides,
                                    /*isVirtual=*/true);
    const auto* t1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b1.GetBufferPointer());

    EXPECT_THROW(pointwise_applicability::validatePointwiseIoTensors({t1}, "Test"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenPointwiseTensorChecks, ValidatePointwiseIoTensorsRejectsUnsupportedDtype)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    const std::vector<int64_t> strides = {48, 16, 4, 1};

    auto b1 = buildTensorAttrBuffer(
        1, hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16, &dims, &strides);
    const auto* t1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b1.GetBufferPointer());

    EXPECT_THROW(pointwise_applicability::validatePointwiseIoTensors({t1}, "Test"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenPointwiseTensorChecks, ValidatePointwiseIoTensorsRejectsMismatchedDtypes)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    const std::vector<int64_t> strides = {48, 16, 4, 1};

    auto b1 = buildTensorAttrBuffer(
        1, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &dims, &strides);
    auto b2 = buildTensorAttrBuffer(
        2, hipdnn_flatbuffers_sdk::data_objects::DataType::HALF, &dims, &strides);

    const auto* t1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b1.GetBufferPointer());
    const auto* t2 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b2.GetBufferPointer());

    EXPECT_THROW(pointwise_applicability::validatePointwiseIoTensors({t1, t2}, "Test"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenPointwiseTensorChecks, ValidatePointwiseIoTensorsRejectsNullDims)
{
    const std::vector<int64_t> strides = {48, 16, 4, 1};

    auto b1 = buildTensorAttrBuffer(
        1, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, nullptr, &strides);
    const auto* t1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b1.GetBufferPointer());

    EXPECT_THROW(pointwise_applicability::validatePointwiseIoTensors({t1}, "Test"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenPointwiseTensorChecks, ValidatePointwiseIoTensorsRejectsNullStrides)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};

    auto b1 = buildTensorAttrBuffer(
        1, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &dims, nullptr);
    const auto* t1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b1.GetBufferPointer());

    EXPECT_THROW(pointwise_applicability::validatePointwiseIoTensors({t1}, "Test"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenPointwiseTensorChecks, ValidatePointwiseIoTensorsRejectsDimsStridesSizeMismatch)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    const std::vector<int64_t> strides = {16, 4, 1};

    auto b1 = buildTensorAttrBuffer(
        1, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &dims, &strides);
    const auto* t1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        b1.GetBufferPointer());

    EXPECT_THROW(pointwise_applicability::validatePointwiseIoTensors({t1}, "Test"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
