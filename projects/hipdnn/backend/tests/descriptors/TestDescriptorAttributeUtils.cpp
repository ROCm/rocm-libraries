// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/DescriptorAttributeUtils.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include <array>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <vector>

namespace hipdnn_backend
{
namespace testing
{

// --- setInt64Vector ---

TEST(TestDescriptorAttributeUtils, SetInt64VectorThrowsOnNegativeElementCount)
{
    std::vector<int64_t> target;
    std::array<int64_t, 3> data = {1, 2, 3};

    ASSERT_THROW_HIPDNN_STATUS(setInt64Vector(target, HIPDNN_TYPE_INT64, -1, data.data(), "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetInt64VectorThrowsOnZeroElementCount)
{
    std::vector<int64_t> target;
    std::array<int64_t, 3> data = {1, 2, 3};

    ASSERT_THROW_HIPDNN_STATUS(setInt64Vector(target, HIPDNN_TYPE_INT64, 0, data.data(), "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetInt64VectorThrowsOnNullArrayOfElements)
{
    std::vector<int64_t> target;

    ASSERT_THROW_HIPDNN_STATUS(setInt64Vector(target, HIPDNN_TYPE_INT64, 1, nullptr, "test"),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetInt64VectorThrowsOnNullErrorPrefix)
{
    std::vector<int64_t> target;
    std::array<int64_t, 3> data = {1, 2, 3};

    ASSERT_THROW_HIPDNN_STATUS(setInt64Vector(target, HIPDNN_TYPE_INT64, 3, data.data(), nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetInt64VectorThrowsOnWrongAttributeType)
{
    std::vector<int64_t> target;
    std::array<int64_t, 3> data = {1, 2, 3};

    ASSERT_THROW_HIPDNN_STATUS(setInt64Vector(target, HIPDNN_TYPE_BOOLEAN, 3, data.data(), "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetInt64VectorSuccess)
{
    std::vector<int64_t> target;
    std::array<int64_t, 3> data = {10, 20, 30};

    ASSERT_NO_THROW(setInt64Vector(target, HIPDNN_TYPE_INT64, 3, data.data(), "test"));
    ASSERT_EQ(target.size(), 3u);
    EXPECT_EQ(target[0], 10);
    EXPECT_EQ(target[1], 20);
    EXPECT_EQ(target[2], 30);
}

// --- getInt64Vector ---

TEST(TestDescriptorAttributeUtils, GetInt64VectorThrowsOnNegativeRequestedElementCount)
{
    const std::vector<int64_t> source = {1, 2, 3};
    std::array<int64_t, 3> output = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_INT64, -1, &count, output.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorQueryReturnsSizeOnNullArray)
{
    const std::vector<int64_t> source = {1, 2, 3};
    int64_t count = 0;

    ASSERT_NO_THROW(getInt64Vector(source, HIPDNN_TYPE_INT64, 3, &count, nullptr, "test"));
    ASSERT_EQ(count, 3);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorQueryReturnsSizeOnZeroRequestedCount)
{
    const std::vector<int64_t> source = {10, 20, 30, 40};
    int64_t count = 0;
    std::array<int64_t, 4> output = {};

    ASSERT_NO_THROW(getInt64Vector(source, HIPDNN_TYPE_INT64, 0, &count, output.data(), "test"));
    ASSERT_EQ(count, 4);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorQueryThrowsWhenBothPointersNull)
{
    const std::vector<int64_t> source = {1, 2, 3};

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_INT64, 3, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorThrowsOnNullErrorPrefix)
{
    const std::vector<int64_t> source = {1, 2, 3};
    std::array<int64_t, 3> output = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_INT64, 3, &count, output.data(), nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorThrowsOnWrongAttributeType)
{
    const std::vector<int64_t> source = {1, 2, 3};
    std::array<int64_t, 3> output = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_BOOLEAN, 3, &count, output.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorSuccess)
{
    const std::vector<int64_t> source = {10, 20, 30};
    std::array<int64_t, 3> output = {};
    int64_t count = 0;

    ASSERT_NO_THROW(getInt64Vector(source, HIPDNN_TYPE_INT64, 3, &count, output.data(), "test"));
    ASSERT_EQ(count, 3);
    EXPECT_EQ(output[0], 10);
    EXPECT_EQ(output[1], 20);
    EXPECT_EQ(output[2], 30);
}

// --- setScalar ---

TEST(TestDescriptorAttributeUtils, SetScalarThrowsOnNullArrayOfElements)
{
    int64_t target = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        setScalar(target, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetScalarThrowsOnNullErrorPrefix)
{
    int64_t target = 0;
    int64_t value = 42;

    ASSERT_THROW_HIPDNN_STATUS(
        setScalar(target, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, &value, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetScalarThrowsOnWrongAttributeType)
{
    int64_t target = 0;
    int64_t value = 42;

    ASSERT_THROW_HIPDNN_STATUS(
        setScalar(target, HIPDNN_TYPE_INT64, HIPDNN_TYPE_BOOLEAN, 1, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetScalarThrowsOnWrongElementCount)
{
    int64_t target = 0;
    int64_t value = 42;

    ASSERT_THROW_HIPDNN_STATUS(
        setScalar(target, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 2, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetScalarSuccess)
{
    int64_t target = 0;
    int64_t value = 42;

    ASSERT_NO_THROW(setScalar(target, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, &value, "test"));
    ASSERT_EQ(target, 42);
}

// --- getScalar ---

TEST(TestDescriptorAttributeUtils, GetScalarQueryReturnsOneOnNullArray)
{
    const int64_t source = 42;
    int64_t count = 0;

    ASSERT_NO_THROW(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetScalarQueryReturnsOneOnZeroRequestedCount)
{
    const int64_t source = 42;
    int64_t count = 0;
    int64_t output = 0;

    ASSERT_NO_THROW(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 0, &count, &output, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetScalarQueryThrowsWhenBothPointersNull)
{
    const int64_t source = 42;

    ASSERT_THROW_HIPDNN_STATUS(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetScalarThrowsOnNullErrorPrefix)
{
    const int64_t source = 42;
    int64_t count = 0;
    int64_t output = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, &count, &output, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetScalarThrowsOnWrongAttributeType)
{
    const int64_t source = 42;
    int64_t count = 0;
    int64_t output = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_BOOLEAN, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetScalarSuccess)
{
    const int64_t source = 42;
    int64_t count = 0;
    int64_t output = 0;

    ASSERT_NO_THROW(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, &count, &output, "test"));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(output, 42);
}

// --- setDataType ---

TEST(TestDescriptorAttributeUtils, SetDataTypeThrowsOnNullArrayOfElements)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    auto target = DataType::UNSET;

    ASSERT_THROW_HIPDNN_STATUS(setDataType(target, HIPDNN_TYPE_DATA_TYPE, 1, nullptr, "test"),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetDataTypeThrowsOnNullErrorPrefix)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    auto target = DataType::UNSET;
    auto value = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(setDataType(target, HIPDNN_TYPE_DATA_TYPE, 1, &value, nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetDataTypeThrowsOnWrongAttributeType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    auto target = DataType::UNSET;
    auto value = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(setDataType(target, HIPDNN_TYPE_INT64, 1, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetDataTypeThrowsOnWrongElementCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    auto target = DataType::UNSET;
    auto value = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(setDataType(target, HIPDNN_TYPE_DATA_TYPE, 2, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetDataTypeSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    auto target = DataType::UNSET;
    auto value = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(setDataType(target, HIPDNN_TYPE_DATA_TYPE, 1, &value, "test"));
    ASSERT_EQ(target, DataType::FLOAT);
}

// --- getDataType ---

TEST(TestDescriptorAttributeUtils, GetDataTypeQueryReturnsOneOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    int64_t count = 0;

    ASSERT_NO_THROW(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeQueryReturnsOneOnZeroRequestedCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    int64_t count = 0;
    hipdnnDataType_t output = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 0, &count, &output, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeQueryThrowsWhenBothPointersNull)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    ASSERT_THROW_HIPDNN_STATUS(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 1, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeThrowsOnNullErrorPrefix)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    int64_t count = 0;
    hipdnnDataType_t output = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 1, &count, &output, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeThrowsOnWrongAttributeType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    int64_t count = 0;
    hipdnnDataType_t output = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    int64_t count = 0;
    hipdnnDataType_t output = {};

    ASSERT_NO_THROW(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 1, &count, &output, "test"));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(output, HIPDNN_DATA_FLOAT);
}

// --- setConvMode ---

TEST(TestDescriptorAttributeUtils, SetConvModeThrowsOnNullArrayOfElements)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;

    ASSERT_THROW_HIPDNN_STATUS(
        setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetConvModeThrowsOnNullErrorPrefix)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CROSS_CORRELATION;

    ASSERT_THROW_HIPDNN_STATUS(
        setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &value, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetConvModeThrowsOnWrongAttributeType)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CROSS_CORRELATION;

    ASSERT_THROW_HIPDNN_STATUS(setConvMode(target, HIPDNN_TYPE_INT64, 1, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetConvModeThrowsOnWrongElementCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CROSS_CORRELATION;

    ASSERT_THROW_HIPDNN_STATUS(setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 2, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetConvModeSuccessCrossCorrelation)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CROSS_CORRELATION;

    ASSERT_NO_THROW(setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &value, "test"));
    ASSERT_EQ(target, ConvMode::CROSS_CORRELATION);
}

TEST(TestDescriptorAttributeUtils, SetConvModeSuccessConvolution)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CONVOLUTION;

    ASSERT_NO_THROW(setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &value, "test"));
    ASSERT_EQ(target, ConvMode::CONVOLUTION);
}

// --- getConvMode ---

TEST(TestDescriptorAttributeUtils, GetConvModeQueryReturnsOneOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    int64_t count = 0;

    ASSERT_NO_THROW(getConvMode(
        ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetConvModeQueryReturnsOneOnZeroRequestedCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION;

    ASSERT_NO_THROW(getConvMode(
        ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 0, &count, &output, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetConvModeQueryThrowsWhenBothPointersNull)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;

    ASSERT_THROW_HIPDNN_STATUS(
        getConvMode(
            ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetConvModeThrowsOnNullErrorPrefix)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION;

    ASSERT_THROW_HIPDNN_STATUS(
        getConvMode(
            ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &count, &output, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetConvModeThrowsOnWrongAttributeType)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION;

    ASSERT_THROW_HIPDNN_STATUS(
        getConvMode(ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetConvModeSuccessCrossCorrelation)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION;

    ASSERT_NO_THROW(getConvMode(
        ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &count, &output, "test"));
    ASSERT_EQ(output, HIPDNN_CROSS_CORRELATION);
}

TEST(TestDescriptorAttributeUtils, GetConvModeSuccessConvolution)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CROSS_CORRELATION;

    ASSERT_NO_THROW(getConvMode(
        ConvMode::CONVOLUTION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &count, &output, "test"));
    ASSERT_EQ(output, HIPDNN_CONVOLUTION);
}

// --- getOperationType ---

TEST(TestDescriptorAttributeUtils, GetOperationTypeQueryReturnsOneOnNullArray)
{
    int64_t count = 0;

    ASSERT_NO_THROW(getOperationType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT,
                                     HIPDNN_TYPE_OPERATION_TYPE_EXT,
                                     1,
                                     &count,
                                     nullptr,
                                     "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetOperationTypeQueryReturnsOneOnZeroRequestedCount)
{
    int64_t count = 0;
    hipdnnOperationType_ext_t output = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;

    ASSERT_NO_THROW(getOperationType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT,
                                     HIPDNN_TYPE_OPERATION_TYPE_EXT,
                                     0,
                                     &count,
                                     &output,
                                     "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetOperationTypeQueryThrowsWhenBothPointersNull)
{
    ASSERT_THROW_HIPDNN_STATUS(getOperationType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT,
                                                HIPDNN_TYPE_OPERATION_TYPE_EXT,
                                                1,
                                                nullptr,
                                                nullptr,
                                                "test"),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetOperationTypeThrowsOnNullErrorPrefix)
{
    int64_t count = 0;
    hipdnnOperationType_ext_t output = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;

    ASSERT_THROW_HIPDNN_STATUS(getOperationType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT,
                                                HIPDNN_TYPE_OPERATION_TYPE_EXT,
                                                1,
                                                &count,
                                                &output,
                                                nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetOperationTypeThrowsOnWrongAttributeType)
{
    int64_t count = 0;
    hipdnnOperationType_ext_t output = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;

    ASSERT_THROW_HIPDNN_STATUS(getOperationType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT,
                                                HIPDNN_TYPE_INT64,
                                                1,
                                                &count,
                                                &output,
                                                "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetOperationTypeSuccessConvForward)
{
    int64_t count = 0;
    hipdnnOperationType_ext_t output = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;

    ASSERT_NO_THROW(getOperationType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT,
                                     HIPDNN_TYPE_OPERATION_TYPE_EXT,
                                     1,
                                     &count,
                                     &output,
                                     "test"));
    ASSERT_EQ(output, HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT);
}

TEST(TestDescriptorAttributeUtils, GetOperationTypeSuccessBatchnormInference)
{
    int64_t count = 0;
    hipdnnOperationType_ext_t output = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;

    ASSERT_NO_THROW(getOperationType(HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_EXT,
                                     HIPDNN_TYPE_OPERATION_TYPE_EXT,
                                     1,
                                     &count,
                                     &output,
                                     "test"));
    ASSERT_EQ(output, HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_EXT);
}

// --- setTensorDescriptor ---

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorThrowsOnNullArrayOfElements)
{
    std::shared_ptr<TensorDescriptor> descTarget;
    int64_t uidTarget = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        setTensorDescriptor(
            descTarget, uidTarget, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorThrowsOnNullErrorPrefix)
{
    std::shared_ptr<TensorDescriptor> descTarget;
    int64_t uidTarget = 0;
    auto wrapper = test_utilities::createFinalizedTensor(1);
    const auto* ptr = wrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        setTensorDescriptor(
            descTarget, uidTarget, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorThrowsOnWrongAttributeType)
{
    std::shared_ptr<TensorDescriptor> descTarget;
    int64_t uidTarget = 0;
    auto wrapper = test_utilities::createFinalizedTensor(1);
    const auto* ptr = wrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        setTensorDescriptor(descTarget, uidTarget, HIPDNN_TYPE_INT64, 1, &ptr, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorThrowsOnWrongElementCount)
{
    std::shared_ptr<TensorDescriptor> descTarget;
    int64_t uidTarget = 0;
    auto wrapper = test_utilities::createFinalizedTensor(1);
    const auto* ptr = wrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        setTensorDescriptor(descTarget, uidTarget, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &ptr, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorThrowsOnUnfinalizedTensor)
{
    std::shared_ptr<TensorDescriptor> descTarget;
    int64_t uidTarget = 0;
    auto wrapper = test_utilities::createDescriptor<TensorDescriptor>();
    const auto* ptr = wrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        setTensorDescriptor(descTarget, uidTarget, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorSuccess)
{
    std::shared_ptr<TensorDescriptor> descTarget;
    int64_t uidTarget = 0;
    auto wrapper = test_utilities::createFinalizedTensor(42);
    const auto* ptr = wrapper.get();

    ASSERT_NO_THROW(setTensorDescriptor(descTarget,
                                        uidTarget,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        static_cast<const void*>(&ptr),
                                        "test"));
    ASSERT_NE(descTarget, nullptr);
    ASSERT_EQ(uidTarget, 42);
}

// --- getTensorDescriptor ---

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorQueryReturnsOneOnNullArray)
{
    auto wrapper = test_utilities::createFinalizedTensor(1);
    auto source = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");
    int64_t count = 0;

    ASSERT_NO_THROW(
        getTensorDescriptor(source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorQueryReturnsOneOnZeroRequestedCount)
{
    auto wrapper = test_utilities::createFinalizedTensor(1);
    auto source = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");
    int64_t count = 0;
    HipdnnBackendDescriptor* output = nullptr;

    ASSERT_NO_THROW(getTensorDescriptor(
        source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, static_cast<void*>(&output), "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorQueryThrowsWhenBothPointersNull)
{
    auto wrapper = test_utilities::createFinalizedTensor(1);
    auto source = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");

    ASSERT_THROW_HIPDNN_STATUS(
        getTensorDescriptor(source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorThrowsOnNullErrorPrefix)
{
    auto wrapper = test_utilities::createFinalizedTensor(1);
    auto source = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");
    int64_t count = 0;
    HipdnnBackendDescriptor* output = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        getTensorDescriptor(source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &output, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorThrowsOnWrongAttributeType)
{
    auto wrapper = test_utilities::createFinalizedTensor(1);
    auto source = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");
    int64_t count = 0;
    HipdnnBackendDescriptor* output = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        getTensorDescriptor(source, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorSuccess)
{
    auto wrapper = test_utilities::createFinalizedTensor(1);
    auto source = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");
    int64_t count = 0;
    HipdnnBackendDescriptor* rawOutput = nullptr;

    ASSERT_NO_THROW(getTensorDescriptor(
        source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, static_cast<void*>(&rawOutput), "test"));
    const std::unique_ptr<HipdnnBackendDescriptor> output(rawOutput);
    ASSERT_EQ(count, 1);
    ASSERT_NE(output, nullptr);
}

// --- setString ---

TEST(TestDescriptorAttributeUtils, SetStringThrowsOnWrongAttributeType)
{
    std::string target;
    const char* data = "hello";

    ASSERT_THROW_HIPDNN_STATUS(setString(target, HIPDNN_TYPE_INT64, 5, data, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetStringThrowsOnNullArrayOfElements)
{
    std::string target;

    ASSERT_THROW_HIPDNN_STATUS(setString(target, HIPDNN_TYPE_CHAR, 5, nullptr, "test"),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetStringThrowsOnNegativeElementCount)
{
    std::string target;
    const char* data = "hello";

    ASSERT_THROW_HIPDNN_STATUS(setString(target, HIPDNN_TYPE_CHAR, -1, data, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetStringSuccess)
{
    std::string target;
    const char* data = "hello";

    ASSERT_NO_THROW(setString(target, HIPDNN_TYPE_CHAR, 5, data, "test"));
    ASSERT_EQ(target, "hello");
}

// --- getString ---

TEST(TestDescriptorAttributeUtils, GetStringReturnsSizeWhenBufferNull)
{
    const std::string source = "hello";
    int64_t count = 0;

    ASSERT_NO_THROW(getString(source, HIPDNN_TYPE_CHAR, 0, &count, nullptr, "test"));
    ASSERT_EQ(count, 6); // "hello" + null terminator
}

TEST(TestDescriptorAttributeUtils, GetStringCopiesData)
{
    const std::string source = "hello";
    std::array<char, 16> buffer = {};
    int64_t count = 0;

    ASSERT_NO_THROW(getString(source, HIPDNN_TYPE_CHAR, 16, &count, buffer.data(), "test"));
    ASSERT_EQ(count, 6);
    ASSERT_STREQ(buffer.data(), "hello");
}

TEST(TestDescriptorAttributeUtils, GetStringTruncatesWhenBufferSmall)
{
    const std::string source = "hello";
    std::array<char, 4> buffer = {};
    int64_t count = 0;

    ASSERT_NO_THROW(getString(source, HIPDNN_TYPE_CHAR, 4, &count, buffer.data(), "test"));
    ASSERT_EQ(count, 4);
    ASSERT_STREQ(buffer.data(), "hel"); // 3 chars + null
}

TEST(TestDescriptorAttributeUtils, GetStringThrowsOnWrongAttributeType)
{
    const std::string source = "hello";
    std::array<char, 16> buffer = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getString(source, HIPDNN_TYPE_INT64, 16, &count, buffer.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

// --- setKnobValueUnion ---

TEST(TestDescriptorAttributeUtils, SetKnobValueUnionInt64)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion target;
    int64_t value = 42;

    ASSERT_NO_THROW(setKnobValueUnion(target, HIPDNN_TYPE_INT64, 1, &value, "test", 256));
    ASSERT_EQ(target.type, KnobValue::IntValue);
    ASSERT_EQ(target.AsIntValue()->value, 42);
}

TEST(TestDescriptorAttributeUtils, SetKnobValueUnionDouble)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion target;
    double value = 3.14;

    ASSERT_NO_THROW(setKnobValueUnion(target, HIPDNN_TYPE_DOUBLE, 1, &value, "test", 256));
    ASSERT_EQ(target.type, KnobValue::FloatValue);
    ASSERT_DOUBLE_EQ(target.AsFloatValue()->value, 3.14);
}

TEST(TestDescriptorAttributeUtils, SetKnobValueUnionString)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion target;
    const char* value = "hello";

    ASSERT_NO_THROW(setKnobValueUnion(target, HIPDNN_TYPE_CHAR, 5, value, "test", 256));
    ASSERT_EQ(target.type, KnobValue::StringValue);
    ASSERT_EQ(target.AsStringValue()->value, "hello");
}

TEST(TestDescriptorAttributeUtils, SetKnobValueUnionThrowsOnUnsupportedType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion target;
    int64_t value = 1;

    ASSERT_THROW_HIPDNN_STATUS(
        setKnobValueUnion(target, HIPDNN_TYPE_BOOLEAN, 1, &value, "test", 256),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetKnobValueUnionThrowsOnNullArray)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion target;

    ASSERT_THROW_HIPDNN_STATUS(
        setKnobValueUnion(target, HIPDNN_TYPE_INT64, 1, nullptr, "test", 256),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetKnobValueUnionThrowsOnWrongElementCount)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion target;
    int64_t value = 42;

    ASSERT_THROW_HIPDNN_STATUS(setKnobValueUnion(target, HIPDNN_TYPE_INT64, 2, &value, "test", 256),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetKnobValueUnionStringExceedsMaxLength)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion target;
    const char* value = "too long";

    ASSERT_THROW_HIPDNN_STATUS(setKnobValueUnion(target, HIPDNN_TYPE_CHAR, 8, value, "test", 4),
                               HIPDNN_STATUS_BAD_PARAM);
}

// --- getKnobValueUnion ---

TEST(TestDescriptorAttributeUtils, GetKnobValueUnionInt64)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion source;
    IntValueT intVal;
    intVal.value = 42;
    source.Set(intVal);

    int64_t output = 0;
    int64_t count = 0;
    ASSERT_NO_THROW(getKnobValueUnion(source, HIPDNN_TYPE_INT64, 1, &count, &output, "test"));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(output, 42);
}

TEST(TestDescriptorAttributeUtils, GetKnobValueUnionDouble)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion source;
    FloatValueT floatVal;
    floatVal.value = 2.718;
    source.Set(floatVal);

    double output = 0.0;
    int64_t count = 0;
    ASSERT_NO_THROW(getKnobValueUnion(source, HIPDNN_TYPE_DOUBLE, 1, &count, &output, "test"));
    ASSERT_EQ(count, 1);
    ASSERT_DOUBLE_EQ(output, 2.718);
}

TEST(TestDescriptorAttributeUtils, GetKnobValueUnionString)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion source;
    StringValueT strVal;
    strVal.value = "hello";
    source.Set(std::move(strVal));

    std::array<char, 16> buffer = {};
    int64_t count = 0;
    ASSERT_NO_THROW(getKnobValueUnion(source, HIPDNN_TYPE_CHAR, 16, &count, buffer.data(), "test"));
    ASSERT_EQ(count, 6); // "hello" + null
    ASSERT_STREQ(buffer.data(), "hello");
}

TEST(TestDescriptorAttributeUtils, GetKnobValueUnionStringSizeQuery)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion source;
    StringValueT strVal;
    strVal.value = "hello";
    source.Set(std::move(strVal));

    int64_t count = 0;
    ASSERT_NO_THROW(getKnobValueUnion(source, HIPDNN_TYPE_CHAR, 0, &count, nullptr, "test"));
    ASSERT_EQ(count, 6);
}

TEST(TestDescriptorAttributeUtils, GetKnobValueUnionInt64TypeMismatch)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion source;
    IntValueT intVal;
    intVal.value = 42;
    source.Set(intVal);

    double output = 0.0;
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getKnobValueUnion(source, HIPDNN_TYPE_DOUBLE, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetKnobValueUnionThrowsOnNoneType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    const KnobValueUnion source; // default is NONE

    int64_t output = 0;
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getKnobValueUnion(source, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(TestDescriptorAttributeUtils, GetKnobValueUnionInt64SizeQuery)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion source;
    IntValueT intVal;
    intVal.value = 99;
    source.Set(intVal);

    int64_t count = 0;
    ASSERT_NO_THROW(getKnobValueUnion(source, HIPDNN_TYPE_INT64, 0, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

// --- setBoundedString ---

TEST(TestDescriptorAttributeUtils, SetBoundedStringSuccess)
{
    std::string target;
    const char* data = "hello";

    ASSERT_NO_THROW(setBoundedString(target, HIPDNN_TYPE_CHAR, 5, data, "test", 10, 1));
    ASSERT_EQ(target, "hello");
}

TEST(TestDescriptorAttributeUtils, SetBoundedStringThrowsWhenExceedsMax)
{
    std::string target;
    const char* data = "toolong";

    ASSERT_THROW_HIPDNN_STATUS(setBoundedString(target, HIPDNN_TYPE_CHAR, 7, data, "test", 4, 0),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetBoundedStringThrowsWhenBelowMin)
{
    std::string target;
    const char* data = "hi";

    ASSERT_THROW_HIPDNN_STATUS(setBoundedString(target, HIPDNN_TYPE_CHAR, 2, data, "test", 10, 5),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetBoundedStringThrowsOnWrongType)
{
    std::string target;
    const char* data = "hello";

    ASSERT_THROW_HIPDNN_STATUS(setBoundedString(target, HIPDNN_TYPE_INT64, 5, data, "test", 10, 0),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetBoundedStringThrowsOnNullArray)
{
    std::string target;

    ASSERT_THROW_HIPDNN_STATUS(
        setBoundedString(target, HIPDNN_TYPE_CHAR, 5, nullptr, "test", 10, 0),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// --- setScalarVector / getScalarVector ---

TEST(TestDescriptorAttributeUtils, SetScalarVectorSuccess)
{
    std::vector<float> target;
    std::array<float, 3> data = {1.0f, 2.0f, 3.0f};

    ASSERT_NO_THROW(
        setScalarVector(target, HIPDNN_TYPE_FLOAT, HIPDNN_TYPE_FLOAT, 3, data.data(), "test"));
    ASSERT_EQ(target.size(), 3u);
    EXPECT_FLOAT_EQ(target[0], 1.0f);
    EXPECT_FLOAT_EQ(target[1], 2.0f);
    EXPECT_FLOAT_EQ(target[2], 3.0f);
}

TEST(TestDescriptorAttributeUtils, SetScalarVectorThrowsOnWrongType)
{
    std::vector<float> target;
    std::array<float, 3> data = {1.0f, 2.0f, 3.0f};

    ASSERT_THROW_HIPDNN_STATUS(
        setScalarVector(target, HIPDNN_TYPE_FLOAT, HIPDNN_TYPE_INT64, 3, data.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetScalarVectorThrowsOnNullArray)
{
    std::vector<float> target;

    ASSERT_THROW_HIPDNN_STATUS(
        setScalarVector(target, HIPDNN_TYPE_FLOAT, HIPDNN_TYPE_FLOAT, 3, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetScalarVectorThrowsOnZeroCount)
{
    std::vector<float> target;
    std::array<float, 3> data = {1.0f, 2.0f, 3.0f};

    ASSERT_THROW_HIPDNN_STATUS(
        setScalarVector(target, HIPDNN_TYPE_FLOAT, HIPDNN_TYPE_FLOAT, 0, data.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetScalarVectorSuccess)
{
    const std::vector<float> source = {10.0f, 20.0f, 30.0f};
    std::array<float, 3> output = {};
    int64_t count = 0;

    ASSERT_NO_THROW(getScalarVector(
        source, HIPDNN_TYPE_FLOAT, HIPDNN_TYPE_FLOAT, 3, &count, output.data(), "test"));
    ASSERT_EQ(count, 3);
    EXPECT_FLOAT_EQ(output[0], 10.0f);
    EXPECT_FLOAT_EQ(output[1], 20.0f);
    EXPECT_FLOAT_EQ(output[2], 30.0f);
}

TEST(TestDescriptorAttributeUtils, GetScalarVectorThrowsOnWrongType)
{
    const std::vector<float> source = {1.0f, 2.0f};
    std::array<float, 2> output = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getScalarVector(
            source, HIPDNN_TYPE_FLOAT, HIPDNN_TYPE_INT64, 2, &count, output.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetScalarVectorQueryReturnsSizeOnNullArray)
{
    const std::vector<float> source = {1.0f, 2.0f, 3.0f};
    int64_t count = 0;

    ASSERT_NO_THROW(
        getScalarVector(source, HIPDNN_TYPE_FLOAT, HIPDNN_TYPE_FLOAT, 3, &count, nullptr, "test"));
    ASSERT_EQ(count, 3);
}

TEST(TestDescriptorAttributeUtils, GetScalarVectorQueryReturnsSizeOnZeroCount)
{
    const std::vector<float> source = {1.0f, 2.0f, 3.0f};
    std::array<float, 3> output = {};
    int64_t count = 0;

    ASSERT_NO_THROW(getScalarVector(
        source, HIPDNN_TYPE_FLOAT, HIPDNN_TYPE_FLOAT, 0, &count, output.data(), "test"));
    ASSERT_EQ(count, 3);
}

// --- setPointwiseMode / getPointwiseMode ---

TEST(TestDescriptorAttributeUtils, SetPointwiseModeSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    auto target = PointwiseMode::UNSET;
    auto value = HIPDNN_POINTWISE_RELU_FWD;

    ASSERT_NO_THROW(setPointwiseMode(target, HIPDNN_TYPE_POINTWISE_MODE, 1, &value, "test"));
    ASSERT_EQ(target, PointwiseMode::RELU_FWD);
}

TEST(TestDescriptorAttributeUtils, SetPointwiseModeThrowsOnWrongType)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    auto target = PointwiseMode::UNSET;
    auto value = HIPDNN_POINTWISE_RELU_FWD;

    ASSERT_THROW_HIPDNN_STATUS(setPointwiseMode(target, HIPDNN_TYPE_INT64, 1, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetPointwiseModeThrowsOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    auto target = PointwiseMode::UNSET;

    ASSERT_THROW_HIPDNN_STATUS(
        setPointwiseMode(target, HIPDNN_TYPE_POINTWISE_MODE, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetPointwiseModeThrowsOnWrongElementCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    auto target = PointwiseMode::UNSET;
    auto value = HIPDNN_POINTWISE_RELU_FWD;

    ASSERT_THROW_HIPDNN_STATUS(
        setPointwiseMode(target, HIPDNN_TYPE_POINTWISE_MODE, 2, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetPointwiseModeSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    int64_t count = 0;
    hipdnnPointwiseMode_t output = HIPDNN_POINTWISE_ABS;

    ASSERT_NO_THROW(getPointwiseMode(
        PointwiseMode::RELU_FWD, HIPDNN_TYPE_POINTWISE_MODE, 1, &count, &output, "test"));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(output, HIPDNN_POINTWISE_RELU_FWD);
}

TEST(TestDescriptorAttributeUtils, GetPointwiseModeThrowsOnWrongType)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    int64_t count = 0;
    hipdnnPointwiseMode_t output = HIPDNN_POINTWISE_ABS;

    ASSERT_THROW_HIPDNN_STATUS(
        getPointwiseMode(PointwiseMode::RELU_FWD, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetPointwiseModeQueryReturnsSizeOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    int64_t count = 0;

    ASSERT_NO_THROW(getPointwiseMode(
        PointwiseMode::RELU_FWD, HIPDNN_TYPE_POINTWISE_MODE, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetPointwiseModeQueryReturnsSizeOnZeroCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    int64_t count = 0;
    hipdnnPointwiseMode_t output = HIPDNN_POINTWISE_ABS;

    ASSERT_NO_THROW(getPointwiseMode(
        PointwiseMode::RELU_FWD, HIPDNN_TYPE_POINTWISE_MODE, 0, &count, &output, "test"));
    ASSERT_EQ(count, 1);
}

// --- setOptionalScalar / getOptionalScalar ---

TEST(TestDescriptorAttributeUtils, SetOptionalScalarSuccess)
{
    std::optional<int64_t> target;
    int64_t value = 99;

    ASSERT_NO_THROW(
        setOptionalScalar<HIPDNN_TYPE_INT64>(target, HIPDNN_TYPE_INT64, 1, &value, "test"));
    ASSERT_TRUE(target.has_value());
    ASSERT_EQ(target.value(), 99);
}

TEST(TestDescriptorAttributeUtils, SetOptionalScalarThrowsOnWrongType)
{
    std::optional<int64_t> target;
    int64_t value = 99;

    ASSERT_THROW_HIPDNN_STATUS(
        setOptionalScalar<HIPDNN_TYPE_INT64>(target, HIPDNN_TYPE_BOOLEAN, 1, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetOptionalScalarThrowsOnWrongElementCount)
{
    std::optional<int64_t> target;
    int64_t value = 99;

    ASSERT_THROW_HIPDNN_STATUS(
        setOptionalScalar<HIPDNN_TYPE_INT64>(target, HIPDNN_TYPE_INT64, 2, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetOptionalScalarSuccessWhenSet)
{
    const std::optional<int64_t> source = int64_t{42};
    int64_t output = 0;
    int64_t count = 0;

    ASSERT_NO_THROW(getOptionalScalar<HIPDNN_TYPE_INT64>(
        source, HIPDNN_TYPE_INT64, 1, &count, &output, "test"));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(output, 42);
}

TEST(TestDescriptorAttributeUtils, GetOptionalScalarSuccessWhenUnset)
{
    const std::optional<int64_t> source; // has no value
    int64_t output = 0;
    int64_t count = 99;

    ASSERT_NO_THROW(getOptionalScalar<HIPDNN_TYPE_INT64>(
        source, HIPDNN_TYPE_INT64, 1, &count, &output, "test"));
    ASSERT_EQ(count, 0);
}

TEST(TestDescriptorAttributeUtils, GetOptionalScalarThrowsOnWrongType)
{
    const std::optional<int64_t> source = int64_t{42};
    int64_t output = 0;
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(getOptionalScalar<HIPDNN_TYPE_INT64>(
                                   source, HIPDNN_TYPE_BOOLEAN, 1, &count, &output, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetOptionalScalarQueryReturnsSizeOnNullArray)
{
    const std::optional<int64_t> source = int64_t{42};
    int64_t count = 0;

    ASSERT_NO_THROW(getOptionalScalar<HIPDNN_TYPE_INT64>(
        source, HIPDNN_TYPE_INT64, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

// --- setNormFwdPhase / getNormFwdPhase ---

TEST(TestDescriptorAttributeUtils, SetNormFwdPhaseSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
    auto target = NormFwdPhase::NOT_SET;
    auto value = HIPDNN_NORM_FWD_TRAINING;

    ASSERT_NO_THROW(setNormFwdPhase(target, HIPDNN_TYPE_NORM_FWD_PHASE, 1, &value, "test"));
    ASSERT_EQ(target, NormFwdPhase::TRAINING);
}

TEST(TestDescriptorAttributeUtils, SetNormFwdPhaseThrowsOnWrongType)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
    auto target = NormFwdPhase::NOT_SET;
    auto value = HIPDNN_NORM_FWD_TRAINING;

    ASSERT_THROW_HIPDNN_STATUS(setNormFwdPhase(target, HIPDNN_TYPE_INT64, 1, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetNormFwdPhaseThrowsOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
    auto target = NormFwdPhase::NOT_SET;

    ASSERT_THROW_HIPDNN_STATUS(
        setNormFwdPhase(target, HIPDNN_TYPE_NORM_FWD_PHASE, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetNormFwdPhaseThrowsOnWrongElementCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
    auto target = NormFwdPhase::NOT_SET;
    auto value = HIPDNN_NORM_FWD_TRAINING;

    ASSERT_THROW_HIPDNN_STATUS(
        setNormFwdPhase(target, HIPDNN_TYPE_NORM_FWD_PHASE, 2, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetNormFwdPhaseSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
    int64_t count = 0;
    hipdnnNormFwdPhase_t output = HIPDNN_NORM_FWD_INFERENCE;

    ASSERT_NO_THROW(getNormFwdPhase(
        NormFwdPhase::TRAINING, HIPDNN_TYPE_NORM_FWD_PHASE, 1, &count, &output, "test"));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(output, HIPDNN_NORM_FWD_TRAINING);
}

TEST(TestDescriptorAttributeUtils, GetNormFwdPhaseThrowsOnWrongType)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
    int64_t count = 0;
    hipdnnNormFwdPhase_t output = HIPDNN_NORM_FWD_INFERENCE;

    ASSERT_THROW_HIPDNN_STATUS(
        getNormFwdPhase(NormFwdPhase::TRAINING, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetNormFwdPhaseQueryReturnsSizeOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
    int64_t count = 0;

    ASSERT_NO_THROW(getNormFwdPhase(
        NormFwdPhase::TRAINING, HIPDNN_TYPE_NORM_FWD_PHASE, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetNormFwdPhaseQueryReturnsSizeOnZeroCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
    int64_t count = 0;
    hipdnnNormFwdPhase_t output = HIPDNN_NORM_FWD_INFERENCE;

    ASSERT_NO_THROW(getNormFwdPhase(
        NormFwdPhase::TRAINING, HIPDNN_TYPE_NORM_FWD_PHASE, 0, &count, &output, "test"));
    ASSERT_EQ(count, 1);
}

// --- setOptionalTensorDescriptor / getOptionalTensorDescriptor ---

TEST(TestDescriptorAttributeUtils, SetOptionalTensorDescriptorSuccess)
{
    std::shared_ptr<TensorDescriptor> descTarget;
    std::optional<int64_t> uidTarget;
    auto wrapper = test_utilities::createFinalizedTensor(7);
    const auto* ptr = wrapper.get();

    ASSERT_NO_THROW(setOptionalTensorDescriptor(descTarget,
                                                uidTarget,
                                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                1,
                                                static_cast<const void*>(&ptr),
                                                "test"));
    ASSERT_NE(descTarget, nullptr);
    ASSERT_TRUE(uidTarget.has_value());
    ASSERT_EQ(uidTarget.value(), 7);
}

TEST(TestDescriptorAttributeUtils, SetOptionalTensorDescriptorThrowsOnWrongType)
{
    std::shared_ptr<TensorDescriptor> descTarget;
    std::optional<int64_t> uidTarget;
    auto wrapper = test_utilities::createFinalizedTensor(7);
    const auto* ptr = wrapper.get();

    ASSERT_THROW_HIPDNN_STATUS(
        setOptionalTensorDescriptor(
            descTarget, uidTarget, HIPDNN_TYPE_INT64, 1, static_cast<const void*>(&ptr), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetOptionalTensorDescriptorSuccessWhenSet)
{
    auto wrapper = test_utilities::createFinalizedTensor(5);
    auto source = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");
    int64_t count = 0;
    HipdnnBackendDescriptor* rawOutput = nullptr;

    ASSERT_NO_THROW(getOptionalTensorDescriptor(
        source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, static_cast<void*>(&rawOutput), "test"));
    const std::unique_ptr<HipdnnBackendDescriptor> output(rawOutput);
    ASSERT_EQ(count, 1);
    ASSERT_NE(output, nullptr);
}

TEST(TestDescriptorAttributeUtils, GetOptionalTensorDescriptorReturnsZeroCountWhenNull)
{
    const std::shared_ptr<TensorDescriptor> source; // null
    int64_t count = 99;

    ASSERT_NO_THROW(getOptionalTensorDescriptor(
        source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 0);
}

TEST(TestDescriptorAttributeUtils, GetOptionalTensorDescriptorThrowsOnWrongType)
{
    auto wrapper = test_utilities::createFinalizedTensor(5);
    auto source = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");
    int64_t count = 0;
    HipdnnBackendDescriptor* rawOutput = nullptr;

    ASSERT_THROW_HIPDNN_STATUS(
        getOptionalTensorDescriptor(
            source, HIPDNN_TYPE_INT64, 1, &count, static_cast<void*>(&rawOutput), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

// --- setTensorDescriptorArray / getTensorDescriptorArray ---

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorArraySuccess)
{
    std::vector<std::shared_ptr<TensorDescriptor>> descTarget;
    std::vector<int64_t> uidTarget;
    auto wrapper1 = test_utilities::createFinalizedTensor(10);
    auto wrapper2 = test_utilities::createFinalizedTensor(20);
    std::array<HipdnnBackendDescriptor*, 2> ptrs = {wrapper1.get(), wrapper2.get()};

    ASSERT_NO_THROW(setTensorDescriptorArray(descTarget,
                                             uidTarget,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             2,
                                             static_cast<const void*>(ptrs.data()),
                                             "test"));
    ASSERT_EQ(descTarget.size(), 2u);
    ASSERT_EQ(uidTarget.size(), 2u);
    ASSERT_EQ(uidTarget[0], 10);
    ASSERT_EQ(uidTarget[1], 20);
}

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorArrayThrowsOnWrongType)
{
    std::vector<std::shared_ptr<TensorDescriptor>> descTarget;
    std::vector<int64_t> uidTarget;
    auto wrapper = test_utilities::createFinalizedTensor(10);
    std::array<HipdnnBackendDescriptor*, 1> ptrs = {wrapper.get()};

    ASSERT_THROW_HIPDNN_STATUS(setTensorDescriptorArray(descTarget,
                                                        uidTarget,
                                                        HIPDNN_TYPE_INT64,
                                                        1,
                                                        static_cast<const void*>(ptrs.data()),
                                                        "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetTensorDescriptorArrayThrowsOnNullArray)
{
    std::vector<std::shared_ptr<TensorDescriptor>> descTarget;
    std::vector<int64_t> uidTarget;

    ASSERT_THROW_HIPDNN_STATUS(
        setTensorDescriptorArray(
            descTarget, uidTarget, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorArraySuccess)
{
    auto wrapper1 = test_utilities::createFinalizedTensor(10);
    auto wrapper2 = test_utilities::createFinalizedTensor(20);
    std::vector<std::shared_ptr<TensorDescriptor>> source;
    source.push_back(HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper1.get(), HIPDNN_STATUS_BAD_PARAM, "unpack"));
    source.push_back(HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper2.get(), HIPDNN_STATUS_BAD_PARAM, "unpack"));

    std::array<HipdnnBackendDescriptor*, 2> output = {nullptr, nullptr};
    int64_t count = 0;

    ASSERT_NO_THROW(getTensorDescriptorArray(source,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             2,
                                             &count,
                                             static_cast<void*>(output.data()),
                                             "test"));
    ASSERT_EQ(count, 2);
    ASSERT_NE(output[0], nullptr);
    ASSERT_NE(output[1], nullptr);

    // cleanup packed descriptors
    delete output[0];
    delete output[1];
}

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorArrayThrowsOnWrongType)
{
    auto wrapper = test_utilities::createFinalizedTensor(10);
    std::vector<std::shared_ptr<TensorDescriptor>> source;
    source.push_back(HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack"));

    std::array<HipdnnBackendDescriptor*, 1> output = {nullptr};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getTensorDescriptorArray(
            source, HIPDNN_TYPE_INT64, 1, &count, static_cast<void*>(output.data()), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetTensorDescriptorArrayQueryReturnsSizeOnNullArray)
{
    auto wrapper = test_utilities::createFinalizedTensor(10);
    std::vector<std::shared_ptr<TensorDescriptor>> source;
    source.push_back(HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack"));

    int64_t count = 0;

    ASSERT_NO_THROW(getTensorDescriptorArray(
        source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

// --- setDiagonalAlignment / getDiagonalAlignment ---

TEST(TestDescriptorAttributeUtils, SetDiagonalAlignmentSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    auto target = DiagonalAlignment::TOP_LEFT;
    auto value = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;

    ASSERT_NO_THROW(
        setDiagonalAlignment(target, HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT, 1, &value, "test"));
    ASSERT_EQ(target, DiagonalAlignment::BOTTOM_RIGHT);
}

TEST(TestDescriptorAttributeUtils, SetDiagonalAlignmentThrowsOnWrongType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    auto target = DiagonalAlignment::TOP_LEFT;
    auto value = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;

    ASSERT_THROW_HIPDNN_STATUS(setDiagonalAlignment(target, HIPDNN_TYPE_INT64, 1, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetDiagonalAlignmentThrowsOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    auto target = DiagonalAlignment::TOP_LEFT;

    ASSERT_THROW_HIPDNN_STATUS(
        setDiagonalAlignment(target, HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetDiagonalAlignmentThrowsOnWrongElementCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    auto target = DiagonalAlignment::TOP_LEFT;
    auto value = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;

    ASSERT_THROW_HIPDNN_STATUS(
        setDiagonalAlignment(target, HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT, 2, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetDiagonalAlignmentSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    int64_t count = 0;
    hipdnnDiagonalAlignment_t output = HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;

    ASSERT_NO_THROW(getDiagonalAlignment(DiagonalAlignment::BOTTOM_RIGHT,
                                         HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                                         1,
                                         &count,
                                         &output,
                                         "test"));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(output, HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT);
}

TEST(TestDescriptorAttributeUtils, GetDiagonalAlignmentThrowsOnWrongType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    int64_t count = 0;
    hipdnnDiagonalAlignment_t output = HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;

    ASSERT_THROW_HIPDNN_STATUS(
        getDiagonalAlignment(
            DiagonalAlignment::BOTTOM_RIGHT, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetDiagonalAlignmentQueryReturnsSizeOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    int64_t count = 0;

    ASSERT_NO_THROW(getDiagonalAlignment(DiagonalAlignment::TOP_LEFT,
                                         HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                                         1,
                                         &count,
                                         nullptr,
                                         "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetDiagonalAlignmentQueryReturnsSizeOnZeroCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    int64_t count = 0;
    hipdnnDiagonalAlignment_t output = HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;

    ASSERT_NO_THROW(getDiagonalAlignment(DiagonalAlignment::TOP_LEFT,
                                         HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                                         0,
                                         &count,
                                         &output,
                                         "test"));
    ASSERT_EQ(count, 1);
}

// --- setAttentionImplementation / getAttentionImplementation ---

TEST(TestDescriptorAttributeUtils, SetAttentionImplementationSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    auto target = AttentionImplementation::AUTO;
    auto value = HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT;

    ASSERT_NO_THROW(setAttentionImplementation(
        target, HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT, 1, &value, "test"));
    ASSERT_EQ(target, AttentionImplementation::UNIFIED);
}

TEST(TestDescriptorAttributeUtils, SetAttentionImplementationThrowsOnWrongType)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    auto target = AttentionImplementation::AUTO;
    auto value = HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT;

    ASSERT_THROW_HIPDNN_STATUS(
        setAttentionImplementation(target, HIPDNN_TYPE_INT64, 1, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetAttentionImplementationThrowsOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    auto target = AttentionImplementation::AUTO;

    ASSERT_THROW_HIPDNN_STATUS(
        setAttentionImplementation(
            target, HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetAttentionImplementationThrowsOnWrongElementCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    auto target = AttentionImplementation::AUTO;
    auto value = HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT;

    ASSERT_THROW_HIPDNN_STATUS(
        setAttentionImplementation(
            target, HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT, 2, &value, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetAttentionImplementationSuccess)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    int64_t count = 0;
    hipdnnAttentionImplementation_t output = HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;

    ASSERT_NO_THROW(getAttentionImplementation(AttentionImplementation::COMPOSITE,
                                               HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT,
                                               1,
                                               &count,
                                               &output,
                                               "test"));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(output, HIPDNN_ATTENTION_IMPLEMENTATION_COMPOSITE_EXT);
}

TEST(TestDescriptorAttributeUtils, GetAttentionImplementationThrowsOnWrongType)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    int64_t count = 0;
    hipdnnAttentionImplementation_t output = HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;

    ASSERT_THROW_HIPDNN_STATUS(
        getAttentionImplementation(
            AttentionImplementation::COMPOSITE, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetAttentionImplementationQueryReturnsSizeOnNullArray)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    int64_t count = 0;

    ASSERT_NO_THROW(getAttentionImplementation(AttentionImplementation::AUTO,
                                               HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT,
                                               1,
                                               &count,
                                               nullptr,
                                               "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetAttentionImplementationQueryReturnsSizeOnZeroCount)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    int64_t count = 0;
    hipdnnAttentionImplementation_t output = HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;

    ASSERT_NO_THROW(getAttentionImplementation(AttentionImplementation::AUTO,
                                               HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT,
                                               0,
                                               &count,
                                               &output,
                                               "test"));
    ASSERT_EQ(count, 1);
}

// --- findTensorInMap ---

TEST(TestDescriptorAttributeUtils, FindTensorInMapSuccess)
{
    auto wrapper = test_utilities::createFinalizedTensor(42);
    auto tensorDesc = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        wrapper.get(), HIPDNN_STATUS_BAD_PARAM, "unpack");

    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> tensorMap;
    tensorMap[42] = tensorDesc;

    std::shared_ptr<TensorDescriptor> result;
    ASSERT_NO_THROW(result = findTensorInMap(tensorMap, 42, "test"));
    ASSERT_NE(result, nullptr);
}

TEST(TestDescriptorAttributeUtils, FindTensorInMapThrowsWhenNotFound)
{
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> tensorMap;

    ASSERT_THROW_HIPDNN_STATUS(findTensorInMap(tensorMap, 99, "test"),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

// --- copyKnobValueUnion ---

TEST(TestDescriptorAttributeUtils, CopyKnobValueUnionInt)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion src;
    IntValueT intVal;
    intVal.value = 77;
    src.Set(intVal);

    KnobValueUnion dst;
    ASSERT_NO_THROW(copyKnobValueUnion(src, dst, "test"));
    ASSERT_EQ(dst.type, KnobValue::IntValue);
    ASSERT_EQ(dst.AsIntValue()->value, 77);
}

TEST(TestDescriptorAttributeUtils, CopyKnobValueUnionDouble)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion src;
    FloatValueT floatVal;
    floatVal.value = 1.5;
    src.Set(floatVal);

    KnobValueUnion dst;
    ASSERT_NO_THROW(copyKnobValueUnion(src, dst, "test"));
    ASSERT_EQ(dst.type, KnobValue::FloatValue);
    ASSERT_DOUBLE_EQ(dst.AsFloatValue()->value, 1.5);
}

TEST(TestDescriptorAttributeUtils, CopyKnobValueUnionString)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    KnobValueUnion src;
    StringValueT strVal;
    strVal.value = "copy_test";
    src.Set(std::move(strVal));

    KnobValueUnion dst;
    ASSERT_NO_THROW(copyKnobValueUnion(src, dst, "test"));
    ASSERT_EQ(dst.type, KnobValue::StringValue);
    ASSERT_EQ(dst.AsStringValue()->value, "copy_test");
}

TEST(TestDescriptorAttributeUtils, CopyKnobValueUnionThrowsOnNoneType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    const KnobValueUnion src; // default is NONE

    KnobValueUnion dst;
    ASSERT_THROW_HIPDNN_STATUS(copyKnobValueUnion(src, dst, "test"), HIPDNN_STATUS_INTERNAL_ERROR);
}

} // namespace testing
} // namespace hipdnn_backend
