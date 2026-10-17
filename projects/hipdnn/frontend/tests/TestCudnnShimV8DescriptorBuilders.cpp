// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Tensor, ConvDesc, PointWiseDesc and MatMulDesc builders against the shim's
// backend C API and the mock hipDNN backend.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Tensor.h>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_ConvDesc.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_MatMulDesc.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_PointWiseDesc.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::ConvolutionMode_t;
using fe::DataType_t;
using fe::PointwiseMode_t;
using fe::TensorReordering_t;

constexpr std::array<std::int64_t, 4> DIMS = {2, 3, 4, 5};
constexpr std::array<std::int64_t, 4> STRIDES = {60, 20, 5, 1};
constexpr auto RANK = static_cast<std::int64_t>(DIMS.size());

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

struct BuildFailure
{
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    std::string what;
};

template <typename Builder>
BuildFailure buildFailure(Builder& builder)
{
    try
    {
        [[maybe_unused]] auto built = builder.build();
        ADD_FAILURE() << "build() did not throw";
    }
    catch(fe::cudnnException& e)
    {
        return {e.getCudnnStatus(), e.what()};
    }
    return {};
}

template <typename T>
T getScalar(cudnnBackendDescriptor_t descriptor,
            cudnnBackendAttributeName_t name,
            cudnnBackendAttributeType_t type)
{
    T value{};
    std::int64_t count = 0;
    EXPECT_EQ(
        cudnnBackendGetAttribute(descriptor, name, type, 1, &count, static_cast<void*>(&value)),
        CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(count, 1);
    return value;
}

std::vector<std::int64_t> getInt64s(cudnnBackendDescriptor_t descriptor,
                                    cudnnBackendAttributeName_t name)
{
    std::array<std::int64_t, CUDNN_DIM_MAX + 1> values{};
    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                       name,
                                       CUDNN_TYPE_INT64,
                                       static_cast<std::int64_t>(values.size()),
                                       &count,
                                       values.data()),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    const auto size = static_cast<std::ptrdiff_t>(
        std::clamp<std::int64_t>(count, 0, static_cast<std::int64_t>(values.size())));
    return {values.begin(), values.begin() + size};
}

fe::Tensor tensor(std::int64_t uid)
{
    return fe::TensorBuilder()
        .setDataType(DataType_t::FLOAT)
        .setDim(RANK, DIMS.data())
        .setStride(RANK, STRIDES.data())
        .setId(uid)
        .setAlignment(16)
        .build();
}

class TestCudnnShimV8DescriptorBuilders : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        lastError();
    }
};

TEST_F(TestCudnnShimV8DescriptorBuilders, TensorBuilderSetsShimAttributes)
{
    const fe::Tensor fromC = fe::TensorBuilder()
                                 .setDataType(CUDNN_DATA_HALF)
                                 .setDim(RANK, DIMS.data())
                                 .setStride(RANK, STRIDES.data())
                                 .setId(7)
                                 .setAlignment(16)
                                 .setVirtual()
                                 .build();
    const fe::Tensor fromFe = fe::TensorBuilder()
                                  .setDataType(DataType_t::HALF)
                                  .setDim(RANK, DIMS.data())
                                  .setStride(RANK, STRIDES.data())
                                  .setId(8)
                                  .setAlignment(32)
                                  .setByValue()
                                  .build();
    cudnnBackendDescriptor_t virtualTensor = fromC.get_raw_desc();
    cudnnBackendDescriptor_t byValueTensor = fromFe.get_raw_desc();

    EXPECT_EQ(getScalar<cudnnDataType_t>(
                  virtualTensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE),
              CUDNN_DATA_HALF);
    EXPECT_EQ(getScalar<cudnnDataType_t>(
                  byValueTensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE),
              CUDNN_DATA_HALF);
    EXPECT_EQ(getInt64s(virtualTensor, CUDNN_ATTR_TENSOR_DIMENSIONS),
              (std::vector<std::int64_t>(DIMS.begin(), DIMS.end())));
    EXPECT_EQ(getInt64s(virtualTensor, CUDNN_ATTR_TENSOR_STRIDES),
              (std::vector<std::int64_t>(STRIDES.begin(), STRIDES.end())));
    EXPECT_EQ(getScalar<std::int64_t>(virtualTensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64),
              7);
    EXPECT_EQ(getScalar<std::int64_t>(byValueTensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64),
              8);
    EXPECT_EQ(
        getScalar<std::int64_t>(virtualTensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64),
        16);
    EXPECT_EQ(
        getScalar<std::int64_t>(byValueTensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64),
        32);
    EXPECT_TRUE(getScalar<bool>(virtualTensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN));
    EXPECT_TRUE(getScalar<bool>(byValueTensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN));
    EXPECT_EQ(fromFe.getDataType(), fromC.getDataType());
}

TEST_F(TestCudnnShimV8DescriptorBuilders, TensorPrecheckNamesAttribute)
{
    fe::TensorBuilder builder;
    builder.setDataType(DataType_t::FLOAT)
        .setDim(RANK, DIMS.data())
        .setStride(RANK, STRIDES.data())
        .setId(1);

    const BuildFailure failure = buildFailure(builder);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT"), std::string::npos)
        << failure.what;
}

TEST_F(TestCudnnShimV8DescriptorBuilders, TensorVectorCountIsNotSupported)
{
    fe::TensorBuilder builder;
    builder.setDataType(DataType_t::FLOAT)
        .setDim(RANK, DIMS.data())
        .setStride(RANK, STRIDES.data())
        .setId(1)
        .setAlignment(16)
        .setVectorCountAndDimension(4, 1);

    const BuildFailure failure = buildFailure(builder);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    // The builder appends the shim's last-error string to its own message.
    EXPECT_NE(failure.what.find("cudnnFinalize failed"), std::string::npos) << failure.what;
    EXPECT_NE(failure.what.find("CUDNN_ATTR_TENSOR_VECTOR_COUNT other than 1"), std::string::npos)
        << failure.what;
}

TEST_F(TestCudnnShimV8DescriptorBuilders, TensorReorderTypeIsNotSupported)
{
    // The builder writes the FE enum's bytes as the C attribute value.
    const TensorReordering_t feValue = TensorReordering_t::INT8x32;
    const cudnnBackendTensorReordering_t cValue = CUDNN_TENSOR_REORDERING_INT8x32;
    static_assert(sizeof(feValue) == sizeof(cValue));
    EXPECT_EQ(std::memcmp(&feValue, &cValue, sizeof(cValue)), 0);

    fe::TensorBuilder builder;
    builder.setDataType(DataType_t::FLOAT)
        .setDim(RANK, DIMS.data())
        .setStride(RANK, STRIDES.data())
        .setId(1)
        .setAlignment(16)
        .setReorderType(feValue);

    const BuildFailure failure = buildFailure(builder);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("cudnnFinalize failed"), std::string::npos) << failure.what;
    EXPECT_NE(failure.what.find("CUDNN_ATTR_TENSOR_REORDERING_MODE other than NONE"),
              std::string::npos)
        << failure.what;
}

TEST_F(TestCudnnShimV8DescriptorBuilders, TensorConstValueForwardsBytesByValue)
{
    const fe::Tensor constant = fe::TensorBuilder()
                                    .setDataType(DataType_t::FLOAT)
                                    .setDim(1, DIMS.data())
                                    .setStride(1, &STRIDES.back())
                                    .setId(3)
                                    .setAlignment(16)
                                    .setConstValue(2.5F)
                                    .build();
    cudnnBackendDescriptor_t raw = constant.get_raw_desc();

    const void* reported
        = getScalar<const void*>(raw, CUDNN_ATTR_TENSOR_CONSTANT_VALUE, CUDNN_TYPE_VOID_PTR);
    ASSERT_NE(reported, nullptr);
    float value = 0.0F;
    std::memcpy(&value, reported, sizeof(value));
    EXPECT_EQ(value, 2.5F);
    EXPECT_TRUE(getScalar<bool>(raw, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN));
}

TEST_F(TestCudnnShimV8DescriptorBuilders, TensorDescribeMatchesUpstream)
{
    const std::array<std::int64_t, 2> dims = {1, 2};
    const std::array<std::int64_t, 2> strides = {2, 1};
    const fe::Tensor built = fe::TensorBuilder()
                                 .setDataType(DataType_t::FLOAT)
                                 .setDim(2, dims.data())
                                 .setStride(2, strides.data())
                                 .setId(4)
                                 .setAlignment(16)
                                 .build();

    const std::string text = built.describe();
    EXPECT_NE(text.find("CUDNN_BACKEND_TENSOR_DESCRIPTOR :"), std::string::npos) << text;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    EXPECT_NE(text.find("Datatype: [\"FLOAT\"]"), std::string::npos) << text;
#else
    EXPECT_NE(text.find("Datatype: " + std::to_string(static_cast<int>(DataType_t::FLOAT))),
              std::string::npos)
        << text;
#endif
    EXPECT_NE(text.find("Dim [ 1,2 ]"), std::string::npos) << text;
}

TEST_F(TestCudnnShimV8DescriptorBuilders, ConvDescBuilderSetsShimAttributes)
{
    const std::array<std::int64_t, 2> prePadding = {1, 2};
    const std::array<std::int64_t, 2> postPadding = {3, 4};
    const std::array<std::int64_t, 2> dilation = {1, 2};
    const std::array<std::int64_t, 2> stride = {2, 3};
    const fe::ConvDesc conv = fe::ConvDescBuilder()
                                  .setComputeType(DataType_t::FLOAT)
                                  .setMathMode(CUDNN_CROSS_CORRELATION)
                                  .setSpatialDimCount(2)
                                  .setPrePadding(2, prePadding.data())
                                  .setPostPadding(2, postPadding.data())
                                  .setDilation(2, dilation.data())
                                  .setSpatialStride(2, stride.data())
                                  .build();
    cudnnBackendDescriptor_t raw = conv.get_raw_desc();

    EXPECT_EQ(
        getScalar<cudnnDataType_t>(raw, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE),
        CUDNN_DATA_FLOAT);
    EXPECT_EQ(getScalar<cudnnConvolutionMode_t>(
                  raw, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE),
              CUDNN_CROSS_CORRELATION);
    EXPECT_EQ(getScalar<std::int64_t>(raw, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64),
              2);
    EXPECT_EQ(getInt64s(raw, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS),
              (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(getInt64s(raw, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS),
              (std::vector<std::int64_t>{3, 4}));
    EXPECT_EQ(getInt64s(raw, CUDNN_ATTR_CONVOLUTION_DILATIONS), (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(getInt64s(raw, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES),
              (std::vector<std::int64_t>{2, 3}));
    EXPECT_EQ(conv.getMathMode(), CUDNN_CROSS_CORRELATION);

    // Upstream's default mode is CONVOLUTION; NOT_SET falls back to cross-correlation.
    const fe::ConvDesc byDefault = fe::ConvDescBuilder()
                                       .setSpatialDimCount(2)
                                       .setPrePadding(2, prePadding.data())
                                       .setPostPadding(2, postPadding.data())
                                       .setDilation(2, dilation.data())
                                       .setSpatialStride(2, stride.data())
                                       .build();
    EXPECT_EQ(byDefault.getMathMode(), CUDNN_CONVOLUTION);
    const fe::ConvDesc notSet = fe::ConvDescBuilder()
                                    .setMathMode(ConvolutionMode_t::NOT_SET)
                                    .setSpatialDimCount(2)
                                    .setPrePadding(2, prePadding.data())
                                    .setPostPadding(2, postPadding.data())
                                    .setDilation(2, dilation.data())
                                    .setSpatialStride(2, stride.data())
                                    .build();
    EXPECT_EQ(notSet.getMathMode(), CUDNN_CROSS_CORRELATION);
    EXPECT_EQ(getScalar<cudnnConvolutionMode_t>(notSet.get_raw_desc(),
                                                CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                                                CUDNN_TYPE_CONVOLUTION_MODE),
              CUDNN_CROSS_CORRELATION);
}

TEST_F(TestCudnnShimV8DescriptorBuilders, ConvDescDescribeKeepsUpstreamSeparators)
{
    const std::array<std::int64_t, 2> zeros = {0, 0};
    const std::array<std::int64_t, 2> ones = {1, 1};
    const fe::ConvDesc conv = fe::ConvDescBuilder()
                                  .setComputeType(DataType_t::FLOAT)
                                  .setMathMode(ConvolutionMode_t::CROSS_CORRELATION)
                                  .setSpatialDimCount(2)
                                  .setPrePadding(2, zeros.data())
                                  .setPostPadding(2, zeros.data())
                                  .setDilation(2, ones.data())
                                  .setSpatialStride(2, ones.data())
                                  .build();

    const std::string text = conv.describe();
    EXPECT_NE(text.find("PadLower [ 0,0 ] PadUpper [,0,0 ]"), std::string::npos) << text;
}

TEST_F(TestCudnnShimV8DescriptorBuilders, PointwiseReluSetsEveryParameter)
{
    const fe::PointWiseDesc relu = fe::PointWiseDescBuilder()
                                       .setMode(PointwiseMode_t::RELU_FWD)
                                       .setComputeType(DataType_t::FLOAT)
                                       .build();
    cudnnBackendDescriptor_t raw = relu.get_raw_desc();

    EXPECT_EQ(getScalar<cudnnNanPropagation_t>(
                  raw, CUDNN_ATTR_POINTWISE_NAN_PROPAGATION, CUDNN_TYPE_NAN_PROPOGATION),
              CUDNN_PROPAGATE_NAN);
    EXPECT_EQ(getScalar<double>(raw, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, CUDNN_TYPE_DOUBLE), 0.0);
    EXPECT_EQ(getScalar<double>(raw, CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP, CUDNN_TYPE_DOUBLE),
              static_cast<double>(std::numeric_limits<float>::max()));
    EXPECT_EQ(getScalar<double>(raw, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, CUDNN_TYPE_DOUBLE),
              0.0);
    EXPECT_EQ(relu.getPortCount(), 2);
}

TEST_F(TestCudnnShimV8DescriptorBuilders, PointwiseBuilderDefaultsElideAtLowering)
{
    const fe::PointWiseDesc relu = fe::PointWiseDescBuilder()
                                       .setMode(PointwiseMode_t::RELU_FWD)
                                       .setComputeType(DataType_t::FLOAT)
                                       .build();
    const fe::Tensor x = tensor(1);
    const fe::Tensor y = tensor(2);

    const auto set = [](cudnnBackendDescriptor_t descriptor,
                        cudnnBackendAttributeName_t name,
                        cudnnBackendAttributeType_t type,
                        const auto& value) {
        EXPECT_EQ(
            cudnnBackendSetAttribute(descriptor, name, type, 1, static_cast<const void*>(&value)),
            CUDNN_STATUS_SUCCESS)
            << lastError();
    };

    cudnnBackendDescriptor_t op = nullptr;
    ASSERT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &op),
              CUDNN_STATUS_SUCCESS);
    set(op,
        CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
        CUDNN_TYPE_BACKEND_DESCRIPTOR,
        relu.get_raw_desc());
    set(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, x.get_raw_desc());
    set(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, y.get_raw_desc());
    EXPECT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS) << lastError();

    cudnnBackendDescriptor_t graph = nullptr;
    ASSERT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &graph),
              CUDNN_STATUS_SUCCESS);
    set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, _handle);
    set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, op);
    EXPECT_EQ(cudnnBackendFinalize(graph), CUDNN_STATUS_SUCCESS) << lastError();

    const std::vector<hipdnn_shim_test::CapturedSetAttribute> calls
        = setAttributesOn(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    const auto has = [&calls](hipdnnBackendAttributeName_t name) {
        return std::any_of(
            calls.begin(), calls.end(), [name](const auto& call) { return call.name == name; });
    };
    EXPECT_TRUE(has(HIPDNN_ATTR_POINTWISE_MODE));
    EXPECT_FALSE(has(HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP));
    EXPECT_FALSE(has(HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP));
    EXPECT_FALSE(has(HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE));

    EXPECT_EQ(cudnnBackendDestroyDescriptor(graph), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendDestroyDescriptor(op), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8DescriptorBuilders, PointwiseUnsupportedModeIsRejected)
{
    fe::PointWiseDescBuilder builder;
    builder.setMode(PointwiseMode_t::COS).setComputeType(DataType_t::FLOAT);

    const BuildFailure failure = buildFailure(builder);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("cudnnFinalize Failed"), std::string::npos) << failure.what;
}

TEST_F(TestCudnnShimV8DescriptorBuilders, MatMulPaddingValueIsNotSupported)
{
    fe::MatMulDescBuilder builder;
    builder.setComputeType(DataType_t::FLOAT).setPaddingValue(0.0);

    const BuildFailure failure = buildFailure(builder);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("CUDNN_ATTR_MATMUL_PADDING_VALUE is not supported"),
              std::string::npos)
        << failure.what;
}

TEST_F(TestCudnnShimV8DescriptorBuilders, MatMulWithoutComputeTypeFails)
{
    fe::MatMulDescBuilder builder;

    const BuildFailure failure = buildFailure(builder);
    EXPECT_EQ(failure.status, CUDNN_STATUS_INVALID_VALUE);
    // The conversion fails before any C call, so no shim last-error text is appended.
    EXPECT_EQ(failure.what.rfind("CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute "
                                 "CUDNN_ATTR_MATMUL_COMP_TYPE Failed cudnn_status: ",
                                 0),
              0U)
        << failure.what;
}

} // namespace
