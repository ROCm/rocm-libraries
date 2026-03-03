// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/DescriptorAttributeUtils.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include <array>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
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

// --- getInt64Vector ---

TEST(TestDescriptorAttributeUtils, GetInt64VectorThrowsOnNegativeRequestedElementCount)
{
    std::vector<int64_t> source = {1, 2, 3};
    std::array<int64_t, 3> output = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_INT64, -1, &count, output.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorQueryReturnsSizeOnNullArray)
{
    std::vector<int64_t> source = {1, 2, 3};
    int64_t count = 0;

    ASSERT_NO_THROW(getInt64Vector(source, HIPDNN_TYPE_INT64, 3, &count, nullptr, "test"));
    ASSERT_EQ(count, 3);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorQueryReturnsSizeOnZeroRequestedCount)
{
    std::vector<int64_t> source = {10, 20, 30, 40};
    int64_t count = 0;
    std::array<int64_t, 4> output = {};

    ASSERT_NO_THROW(getInt64Vector(source, HIPDNN_TYPE_INT64, 0, &count, output.data(), "test"));
    ASSERT_EQ(count, 4);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorQueryThrowsWhenBothPointersNull)
{
    std::vector<int64_t> source = {1, 2, 3};

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_INT64, 3, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorThrowsOnNullErrorPrefix)
{
    std::vector<int64_t> source = {1, 2, 3};
    std::array<int64_t, 3> output = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_INT64, 3, &count, output.data(), nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetInt64VectorThrowsOnWrongAttributeType)
{
    std::vector<int64_t> source = {1, 2, 3};
    std::array<int64_t, 3> output = {};
    int64_t count = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getInt64Vector(source, HIPDNN_TYPE_BOOLEAN, 3, &count, output.data(), "test"),
        HIPDNN_STATUS_BAD_PARAM);
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

// --- getScalar ---

TEST(TestDescriptorAttributeUtils, GetScalarQueryReturnsOneOnNullArray)
{
    int64_t source = 42;
    int64_t count = 0;

    ASSERT_NO_THROW(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetScalarQueryReturnsOneOnZeroRequestedCount)
{
    int64_t source = 42;
    int64_t count = 0;
    int64_t output = 0;

    ASSERT_NO_THROW(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 0, &count, &output, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetScalarQueryThrowsWhenBothPointersNull)
{
    int64_t source = 42;

    ASSERT_THROW_HIPDNN_STATUS(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetScalarThrowsOnNullErrorPrefix)
{
    int64_t source = 42;
    int64_t count = 0;
    int64_t output = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_INT64, 1, &count, &output, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetScalarThrowsOnWrongAttributeType)
{
    int64_t source = 42;
    int64_t count = 0;
    int64_t output = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        getScalar(source, HIPDNN_TYPE_INT64, HIPDNN_TYPE_BOOLEAN, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

// --- setDataType ---

TEST(TestDescriptorAttributeUtils, SetDataTypeThrowsOnNullArrayOfElements)
{
    using hipdnn_data_sdk::data_objects::DataType;
    auto target = DataType::UNSET;

    ASSERT_THROW_HIPDNN_STATUS(setDataType(target, HIPDNN_TYPE_DATA_TYPE, 1, nullptr, "test"),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetDataTypeThrowsOnNullErrorPrefix)
{
    using hipdnn_data_sdk::data_objects::DataType;
    auto target = DataType::UNSET;
    auto value = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(setDataType(target, HIPDNN_TYPE_DATA_TYPE, 1, &value, nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetDataTypeThrowsOnWrongAttributeType)
{
    using hipdnn_data_sdk::data_objects::DataType;
    auto target = DataType::UNSET;
    auto value = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(setDataType(target, HIPDNN_TYPE_INT64, 1, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetDataTypeThrowsOnWrongElementCount)
{
    using hipdnn_data_sdk::data_objects::DataType;
    auto target = DataType::UNSET;
    auto value = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(setDataType(target, HIPDNN_TYPE_DATA_TYPE, 2, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

// --- getDataType ---

TEST(TestDescriptorAttributeUtils, GetDataTypeQueryReturnsOneOnNullArray)
{
    using hipdnn_data_sdk::data_objects::DataType;
    int64_t count = 0;

    ASSERT_NO_THROW(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeQueryReturnsOneOnZeroRequestedCount)
{
    using hipdnn_data_sdk::data_objects::DataType;
    int64_t count = 0;
    hipdnnDataType_t output = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 0, &count, &output, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeQueryThrowsWhenBothPointersNull)
{
    using hipdnn_data_sdk::data_objects::DataType;

    ASSERT_THROW_HIPDNN_STATUS(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 1, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeThrowsOnNullErrorPrefix)
{
    using hipdnn_data_sdk::data_objects::DataType;
    int64_t count = 0;
    hipdnnDataType_t output = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_DATA_TYPE, 1, &count, &output, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetDataTypeThrowsOnWrongAttributeType)
{
    using hipdnn_data_sdk::data_objects::DataType;
    int64_t count = 0;
    hipdnnDataType_t output = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        getDataType(DataType::FLOAT, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

// --- setConvMode ---

TEST(TestDescriptorAttributeUtils, SetConvModeThrowsOnNullArrayOfElements)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;

    ASSERT_THROW_HIPDNN_STATUS(
        setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 1, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetConvModeThrowsOnNullErrorPrefix)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;

    ASSERT_THROW_HIPDNN_STATUS(
        setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &value, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, SetConvModeThrowsOnWrongAttributeType)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;

    ASSERT_THROW_HIPDNN_STATUS(setConvMode(target, HIPDNN_TYPE_INT64, 1, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetConvModeThrowsOnWrongElementCount)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;

    ASSERT_THROW_HIPDNN_STATUS(setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 2, &value, "test"),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, SetConvModeSuccessCrossCorrelation)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;

    ASSERT_NO_THROW(setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &value, "test"));
    ASSERT_EQ(target, ConvMode::CROSS_CORRELATION);
}

TEST(TestDescriptorAttributeUtils, SetConvModeSuccessConvolution)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    auto target = ConvMode::UNSET;
    auto value = HIPDNN_CONVOLUTION_MODE_CONVOLUTION;

    ASSERT_NO_THROW(setConvMode(target, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &value, "test"));
    ASSERT_EQ(target, ConvMode::CONVOLUTION);
}

// --- getConvMode ---

TEST(TestDescriptorAttributeUtils, GetConvModeQueryReturnsOneOnNullArray)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    int64_t count = 0;

    ASSERT_NO_THROW(getConvMode(
        ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &count, nullptr, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetConvModeQueryReturnsOneOnZeroRequestedCount)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION_MODE_CONVOLUTION;

    ASSERT_NO_THROW(getConvMode(
        ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 0, &count, &output, "test"));
    ASSERT_EQ(count, 1);
}

TEST(TestDescriptorAttributeUtils, GetConvModeQueryThrowsWhenBothPointersNull)
{
    using hipdnn_data_sdk::data_objects::ConvMode;

    ASSERT_THROW_HIPDNN_STATUS(
        getConvMode(
            ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, nullptr, nullptr, "test"),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetConvModeThrowsOnNullErrorPrefix)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION_MODE_CONVOLUTION;

    ASSERT_THROW_HIPDNN_STATUS(
        getConvMode(
            ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &count, &output, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestDescriptorAttributeUtils, GetConvModeThrowsOnWrongAttributeType)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION_MODE_CONVOLUTION;

    ASSERT_THROW_HIPDNN_STATUS(
        getConvMode(ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_INT64, 1, &count, &output, "test"),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDescriptorAttributeUtils, GetConvModeSuccessCrossCorrelation)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION_MODE_CONVOLUTION;

    ASSERT_NO_THROW(getConvMode(
        ConvMode::CROSS_CORRELATION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &count, &output, "test"));
    ASSERT_EQ(output, HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION);
}

TEST(TestDescriptorAttributeUtils, GetConvModeSuccessConvolution)
{
    using hipdnn_data_sdk::data_objects::ConvMode;
    int64_t count = 0;
    hipdnnConvolutionMode_t output = HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;

    ASSERT_NO_THROW(getConvMode(
        ConvMode::CONVOLUTION, HIPDNN_TYPE_CONVOLUTION_MODE, 1, &count, &output, "test"));
    ASSERT_EQ(output, HIPDNN_CONVOLUTION_MODE_CONVOLUTION);
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

    ASSERT_NO_THROW(setTensorDescriptor(
        descTarget, uidTarget, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ptr, "test"));
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

    ASSERT_NO_THROW(
        getTensorDescriptor(source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, &output, "test"));
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
    HipdnnBackendDescriptor* output = nullptr;

    ASSERT_NO_THROW(
        getTensorDescriptor(source, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &output, "test"));
    ASSERT_EQ(count, 1);
    ASSERT_NE(output, nullptr);
}

} // namespace testing
} // namespace hipdnn_backend
