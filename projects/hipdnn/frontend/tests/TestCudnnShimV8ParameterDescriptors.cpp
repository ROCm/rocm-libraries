// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include "fake_backend/MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace
{

namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;
using hipdnn_frontend::compatibility::cudnn_frontend::ConvolutionMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::DataType_t;
using hipdnn_frontend::compatibility::cudnn_frontend::PaddingMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::PointwiseMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::ReductionMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::ResampleMode_t;

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

class TestCudnnShimV8ParameterDescriptors : public hipdnn_shim_test::ShimMockBackendFixture
{
protected:
    std::vector<cudnnBackendDescriptor_t> _owned;

    void SetUp() override
    {
        ShimMockBackendFixture::SetUp();
        lastError();
    }

    void TearDown() override
    {
        for(cudnnBackendDescriptor_t descriptor : _owned)
        {
            cudnnBackendDestroyDescriptor(descriptor);
        }
        _owned.clear();
        ShimMockBackendFixture::TearDown();
    }

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(type, &descriptor), CUDNN_STATUS_SUCCESS);
        _owned.push_back(descriptor);
        return descriptor;
    }

    template <typename T>
    static cudnnStatus_t set(cudnnBackendDescriptor_t descriptor,
                             cudnnBackendAttributeName_t name,
                             cudnnBackendAttributeType_t type,
                             const T& value)
    {
        return cudnnBackendSetAttribute(descriptor, name, type, 1, &value);
    }

    template <typename T>
    static cudnnStatus_t setArray(cudnnBackendDescriptor_t descriptor,
                                  cudnnBackendAttributeName_t name,
                                  cudnnBackendAttributeType_t type,
                                  const std::vector<T>& values)
    {
        // A non-null pointer even for count 0, as upstream's fixed arrays give.
        static const T s_empty{};
        return cudnnBackendSetAttribute(descriptor,
                                        name,
                                        type,
                                        static_cast<std::int64_t>(values.size()),
                                        values.empty() ? &s_empty : values.data());
    }

    template <typename Derived>
    static const Derived& object(cudnnBackendDescriptor_t descriptor,
                                 cudnnBackendDescriptorType_t type)
    {
        return backend::descriptorCast<Derived>(*backend::liveHandle(descriptor)->object, type);
    }

    static void expectFinalizeFails(cudnnBackendDescriptor_t descriptor,
                                    cudnnStatus_t status,
                                    const std::string& named)
    {
        EXPECT_EQ(cudnnBackendFinalize(descriptor), status);
        EXPECT_NE(lastError().find(named), std::string::npos) << named;
    }

    // ConvDescBuilder_v8::build sends everything; `skip` leaves one out.
    cudnnBackendDescriptor_t convolution(std::optional<cudnnBackendAttributeName_t> skip
                                         = std::nullopt,
                                         std::int64_t spatialDims = 2)
    {
        cudnnBackendDescriptor_t conv = create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        if(skip != CUDNN_ATTR_CONVOLUTION_COMP_TYPE)
        {
            EXPECT_EQ(
                set(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT),
                CUDNN_STATUS_SUCCESS);
        }
        if(skip != CUDNN_ATTR_CONVOLUTION_CONV_MODE)
        {
            EXPECT_EQ(set(conv,
                          CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                          CUDNN_TYPE_CONVOLUTION_MODE,
                          CUDNN_CROSS_CORRELATION),
                      CUDNN_STATUS_SUCCESS);
        }
        if(skip != CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS)
        {
            EXPECT_EQ(set(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, spatialDims),
                      CUDNN_STATUS_SUCCESS);
        }
        const std::vector<std::int64_t> pre = {1, 2};
        const std::vector<std::int64_t> post = {3, 4};
        const std::vector<std::int64_t> dilations = {1, 1};
        const std::vector<std::int64_t> strides = {2, 2};
        EXPECT_EQ(setArray(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, pre),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(setArray(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, post),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(setArray(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dilations),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(setArray(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, strides),
                  CUDNN_STATUS_SUCCESS);
        return conv;
    }

    cudnnBackendDescriptor_t pointwise(std::optional<cudnnPointwiseMode_t> mode,
                                       std::optional<cudnnDataType_t> mathPrecision)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        if(mode)
        {
            EXPECT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, *mode),
                      CUDNN_STATUS_SUCCESS);
        }
        if(mathPrecision)
        {
            EXPECT_EQ(set(descriptor,
                          CUDNN_ATTR_POINTWISE_MATH_PREC,
                          CUDNN_TYPE_DATA_TYPE,
                          *mathPrecision),
                      CUDNN_STATUS_SUCCESS);
        }
        return descriptor;
    }

    cudnnBackendDescriptor_t reduction(std::optional<cudnnReduceTensorOp_t> mode,
                                       std::optional<cudnnDataType_t> computeType)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        if(mode)
        {
            EXPECT_EQ(set(descriptor,
                          CUDNN_ATTR_REDUCTION_OPERATOR,
                          CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                          *mode),
                      CUDNN_STATUS_SUCCESS);
        }
        if(computeType)
        {
            EXPECT_EQ(
                set(descriptor, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, *computeType),
                CUDNN_STATUS_SUCCESS);
        }
        return descriptor;
    }

    // ResampleDescBuilder_v8::build order: mode, compute type, NaN option,
    // padding mode, spatial dims, then the four FRACTION arrays with
    // count = spatial dims.
    cudnnBackendDescriptor_t resample(std::optional<cudnnBackendAttributeName_t> skip
                                      = std::nullopt,
                                      cudnnResampleMode_t mode = CUDNN_RESAMPLE_MAXPOOL,
                                      std::int64_t spatialDims = 2,
                                      std::vector<cudnnFraction_t> windowDims = {{4, 2}, {3, 1}})
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        if(skip != CUDNN_ATTR_RESAMPLE_MODE)
        {
            EXPECT_EQ(set(descriptor, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, mode),
                      CUDNN_STATUS_SUCCESS);
        }
        if(skip != CUDNN_ATTR_RESAMPLE_COMP_TYPE)
        {
            EXPECT_EQ(set(descriptor,
                          CUDNN_ATTR_RESAMPLE_COMP_TYPE,
                          CUDNN_TYPE_DATA_TYPE,
                          CUDNN_DATA_FLOAT),
                      CUDNN_STATUS_SUCCESS);
        }
        EXPECT_EQ(set(descriptor,
                      CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
                      CUDNN_TYPE_NAN_PROPOGATION,
                      CUDNN_PROPAGATE_NAN),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(set(descriptor,
                      CUDNN_ATTR_RESAMPLE_PADDING_MODE,
                      CUDNN_TYPE_PADDING_MODE,
                      CUDNN_NEG_INF_PAD),
                  CUDNN_STATUS_SUCCESS);
        if(skip != CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS)
        {
            EXPECT_EQ(
                set(descriptor, CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, CUDNN_TYPE_INT64, spatialDims),
                CUDNN_STATUS_SUCCESS);
        }
        const auto count = static_cast<std::size_t>(spatialDims);
        windowDims.resize(count, {1, 1});
        const std::vector<cudnnFraction_t> zeros(count, {0, 1});
        const std::vector<cudnnFraction_t> strides(count, {2, 1});
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_FRACTION, windowDims),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, CUDNN_TYPE_FRACTION, zeros),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, CUDNN_TYPE_FRACTION, zeros),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(setArray(descriptor, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, strides),
                  CUDNN_STATUS_SUCCESS);
        return descriptor;
    }
};

TEST_F(TestCudnnShimV8ParameterDescriptors, ConvolutionRecordsLowering)
{
    cudnnBackendDescriptor_t conv = convolution();
    ASSERT_EQ(cudnnBackendFinalize(conv), CUDNN_STATUS_SUCCESS);
    const auto& lowered
        = object<backend::ConvolutionDescriptor>(conv, CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    EXPECT_EQ(lowered.computeType(), DataType_t::FLOAT);
    EXPECT_EQ(lowered.mode(), ConvolutionMode_t::CROSS_CORRELATION);
    EXPECT_EQ(lowered.prePaddings(), (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(lowered.postPaddings(), (std::vector<std::int64_t>{3, 4}));
    EXPECT_EQ(lowered.dilations(), (std::vector<std::int64_t>{1, 1}));
    EXPECT_EQ(lowered.filterStrides(), (std::vector<std::int64_t>{2, 2}));
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ConvolutionRequiredAttributesNamed)
{
    for(const cudnnBackendAttributeName_t required : {CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                      CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                                                      CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS})
    {
        const std::string name = backend::attributeName(required);
        SCOPED_TRACE(name);
        expectFinalizeFails(convolution(required), CUDNN_STATUS_BAD_PARAM, name);
    }
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ConvolutionSpatialDimsMustBePositive)
{
    expectFinalizeFails(convolution(std::nullopt, 0),
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS");
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ConvolutionArrayCountsMatchSpatialDims)
{
    for(const cudnnBackendAttributeName_t array : {CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                                   CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                                   CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                                   CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES})
    {
        const std::string name = backend::attributeName(array);
        SCOPED_TRACE(name);
        cudnnBackendDescriptor_t conv = convolution();
        ASSERT_EQ(setArray(conv, array, CUDNN_TYPE_INT64, std::vector<std::int64_t>{1, 1, 1}),
                  CUDNN_STATUS_SUCCESS);
        expectFinalizeFails(conv, CUDNN_STATUS_BAD_PARAM, name);
    }
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ConvolutionFirstFilterStrideMustBePositive)
{
    cudnnBackendDescriptor_t conv = convolution();
    ASSERT_EQ(setArray(conv,
                       CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                       CUDNN_TYPE_INT64,
                       std::vector<std::int64_t>{0, 1}),
              CUDNN_STATUS_SUCCESS);
    expectFinalizeFails(conv, CUDNN_STATUS_BAD_PARAM, "CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES");
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ConvolutionUnlowerableComputeTypeNotSupported)
{
    cudnnBackendDescriptor_t conv = convolution();
    ASSERT_EQ(set(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_INT8x32),
              CUDNN_STATUS_SUCCESS);
    expectFinalizeFails(conv, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_ATTR_CONVOLUTION_COMP_TYPE");
}

TEST_F(TestCudnnShimV8ParameterDescriptors, PointwiseRequiresModeAndMathPrecision)
{
    expectFinalizeFails(pointwise(std::nullopt, CUDNN_DATA_FLOAT),
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_POINTWISE_MODE");
    expectFinalizeFails(pointwise(CUDNN_POINTWISE_ADD, std::nullopt),
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_POINTWISE_MATH_PREC");
}

TEST_F(TestCudnnShimV8ParameterDescriptors, PointwiseModeSupportFollowsHipdnnMapping)
{
    // The cuDNN modes with no case in Types.hpp toBackendPointwiseMode.
    constexpr std::array<cudnnPointwiseMode_t, 3> UNSUPPORTED
        = {CUDNN_POINTWISE_COS, CUDNN_POINTWISE_MOD, CUDNN_POINTWISE_POW};
    int supported = 0;
    for(int raw = CUDNN_POINTWISE_ABS; raw <= CUDNN_POINTWISE_TANH_FWD; ++raw)
    {
        const auto mode = static_cast<cudnnPointwiseMode_t>(raw);
        SCOPED_TRACE(raw);
        cudnnBackendDescriptor_t descriptor = pointwise(mode, CUDNN_DATA_FLOAT);
        if(std::find(UNSUPPORTED.begin(), UNSUPPORTED.end(), mode) != UNSUPPORTED.end())
        {
            expectFinalizeFails(
                descriptor, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_ATTR_POINTWISE_MODE");
            continue;
        }
        ASSERT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            object<backend::PointwiseDescriptor>(descriptor, CUDNN_BACKEND_POINTWISE_DESCRIPTOR)
                .mode(),
            *backend::toFe(mode));
        ++supported;
    }
    EXPECT_EQ(supported, 47);
}

TEST_F(TestCudnnShimV8ParameterDescriptors, PointwiseUpstreamDefaultsElided)
{
    // PointWiseDescBuilder_v8 sends the RELU parameters at their defaults, with
    // the upper clip clamped to FLT_MAX for FLOAT math.
    cudnnBackendDescriptor_t relu = pointwise(CUDNN_POINTWISE_RELU_FWD, CUDNN_DATA_FLOAT);
    ASSERT_EQ(set(relu,
                  CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                  CUDNN_TYPE_NAN_PROPOGATION,
                  CUDNN_PROPAGATE_NAN),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(relu, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, CUDNN_TYPE_DOUBLE, 0.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(relu,
                  CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                  CUDNN_TYPE_DOUBLE,
                  static_cast<double>(std::numeric_limits<float>::max())),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(relu, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, CUDNN_TYPE_DOUBLE, 0.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(relu), CUDNN_STATUS_SUCCESS);
    const auto& lowered
        = object<backend::PointwiseDescriptor>(relu, CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    EXPECT_FALSE(lowered.reluLowerClip().has_value());
    EXPECT_FALSE(lowered.reluUpperClip().has_value());
    EXPECT_FALSE(lowered.reluLowerClipSlope().has_value());

    cudnnBackendDescriptor_t others = pointwise(CUDNN_POINTWISE_ADD, CUDNN_DATA_DOUBLE);
    ASSERT_EQ(set(others,
                  CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                  CUDNN_TYPE_DOUBLE,
                  std::numeric_limits<double>::max()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(others, CUDNN_ATTR_POINTWISE_ELU_ALPHA, CUDNN_TYPE_DOUBLE, 1.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(others, CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, CUDNN_TYPE_DOUBLE, 1.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(others, CUDNN_ATTR_POINTWISE_SWISH_BETA, CUDNN_TYPE_DOUBLE, 1.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(others, CUDNN_ATTR_POINTWISE_AXIS, CUDNN_TYPE_INT64, std::int64_t{-1}),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(others), CUDNN_STATUS_SUCCESS);
    const auto& defaulted
        = object<backend::PointwiseDescriptor>(others, CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    EXPECT_FALSE(defaulted.reluUpperClip().has_value());
    EXPECT_FALSE(defaulted.eluAlpha().has_value());
    EXPECT_FALSE(defaulted.softplusBeta().has_value());
    EXPECT_FALSE(defaulted.swishBeta().has_value());
    EXPECT_FALSE(defaulted.axis().has_value());
}

TEST_F(TestCudnnShimV8ParameterDescriptors, PointwiseExplicitParametersRecorded)
{
    cudnnBackendDescriptor_t descriptor = pointwise(CUDNN_POINTWISE_RELU_FWD, CUDNN_DATA_FLOAT);
    ASSERT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, CUDNN_TYPE_DOUBLE, -1.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP, CUDNN_TYPE_DOUBLE, 6.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, CUDNN_TYPE_DOUBLE, 0.1),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_ELU_ALPHA, CUDNN_TYPE_DOUBLE, 0.5),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, CUDNN_TYPE_DOUBLE, 2.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_SWISH_BETA, CUDNN_TYPE_DOUBLE, 3.0),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_AXIS, CUDNN_TYPE_INT64, std::int64_t{2}),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
    const auto& lowered
        = object<backend::PointwiseDescriptor>(descriptor, CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    EXPECT_EQ(lowered.computeType(), DataType_t::FLOAT);
    EXPECT_EQ(lowered.reluLowerClip(), std::optional<double>(-1.0));
    EXPECT_EQ(lowered.reluUpperClip(), std::optional<double>(6.0));
    EXPECT_EQ(lowered.reluLowerClipSlope(), std::optional<double>(0.1));
    EXPECT_EQ(lowered.eluAlpha(), std::optional<double>(0.5));
    EXPECT_EQ(lowered.softplusBeta(), std::optional<double>(2.0));
    EXPECT_EQ(lowered.swishBeta(), std::optional<double>(3.0));
    EXPECT_EQ(lowered.axis(), std::optional<std::int64_t>(2));
}

TEST_F(TestCudnnShimV8ParameterDescriptors, PointwiseNonFiniteParametersRejected)
{
    for(const cudnnBackendAttributeName_t parameter : {CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                                                       CUDNN_ATTR_POINTWISE_ELU_ALPHA,
                                                       CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                                                       CUDNN_ATTR_POINTWISE_SWISH_BETA})
    {
        const std::string name = backend::attributeName(parameter);
        SCOPED_TRACE(name);
        for(const double value :
            {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        {
            cudnnBackendDescriptor_t descriptor = pointwise(CUDNN_POINTWISE_ADD, CUDNN_DATA_FLOAT);
            ASSERT_EQ(set(descriptor, parameter, CUDNN_TYPE_DOUBLE, value), CUDNN_STATUS_SUCCESS);
            expectFinalizeFails(descriptor, CUDNN_STATUS_BAD_PARAM, name);
        }
    }
}

TEST_F(TestCudnnShimV8ParameterDescriptors, PointwiseNanPropagationStored)
{
    cudnnBackendDescriptor_t descriptor = pointwise(CUDNN_POINTWISE_RELU_BWD, CUDNN_DATA_FLOAT);
    ASSERT_EQ(set(descriptor,
                  CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                  CUDNN_TYPE_NAN_PROPOGATION,
                  CUDNN_PROPAGATE_NAN),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
    std::int32_t nan = -1;
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(descriptor,
                                       CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                                       CUDNN_TYPE_NAN_PROPOGATION,
                                       1,
                                       &count,
                                       &nan),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(nan, static_cast<std::int32_t>(CUDNN_PROPAGATE_NAN));
}

TEST_F(TestCudnnShimV8ParameterDescriptors, MatmulRequiresComputeType)
{
    expectFinalizeFails(create(CUDNN_BACKEND_MATMUL_DESCRIPTOR),
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_MATMUL_COMP_TYPE");

    cudnnBackendDescriptor_t matmul = create(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
    ASSERT_EQ(set(matmul, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(matmul), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        object<backend::MatmulDescriptor>(matmul, CUDNN_BACKEND_MATMUL_DESCRIPTOR).computeType(),
        DataType_t::FLOAT);
}

TEST_F(TestCudnnShimV8ParameterDescriptors, MatmulPaddingValueNotSupported)
{
    cudnnBackendDescriptor_t matmul = create(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
    ASSERT_EQ(set(matmul, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(matmul, CUDNN_ATTR_MATMUL_PADDING_VALUE, CUDNN_TYPE_DOUBLE, 0.0),
              CUDNN_STATUS_SUCCESS);
    expectFinalizeFails(matmul, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_ATTR_MATMUL_PADDING_VALUE");
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ReductionRequiresOperatorAndComputeType)
{
    expectFinalizeFails(reduction(std::nullopt, CUDNN_DATA_FLOAT),
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_REDUCTION_OPERATOR");
    expectFinalizeFails(reduction(CUDNN_REDUCE_TENSOR_ADD, std::nullopt),
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_REDUCTION_COMP_TYPE");
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ReductionOperatorSupportFollowsHipdnnMapping)
{
    // Types.hpp toBackendReductionMode has a case for every cuDNN reduction operator.
    int supported = 0;
    for(int raw = CUDNN_REDUCE_TENSOR_ADD; raw <= CUDNN_REDUCE_TENSOR_NORM2; ++raw)
    {
        const auto mode = static_cast<cudnnReduceTensorOp_t>(raw);
        SCOPED_TRACE(raw);
        cudnnBackendDescriptor_t descriptor = reduction(mode, CUDNN_DATA_FLOAT);
        ASSERT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        const auto& lowered
            = object<backend::ReductionDescriptor>(descriptor, CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        EXPECT_EQ(lowered.mode(), *backend::toFe(mode));
        EXPECT_EQ(lowered.computeType(), DataType_t::FLOAT);
        ++supported;
    }
    EXPECT_EQ(supported, 9);
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ReductionIsDeterministicStored)
{
    cudnnBackendDescriptor_t descriptor = reduction(CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT);
    ASSERT_EQ(set(descriptor, CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC, CUDNN_TYPE_BOOLEAN, true),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
    std::array<unsigned char, 2> read = {0, 0x55};
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(descriptor,
                                       CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
                                       CUDNN_TYPE_BOOLEAN,
                                       1,
                                       &count,
                                       read.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(read, (std::array<unsigned char, 2>{1, 0x55}));
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ResampleRecordsIntegralFractions)
{
    cudnnBackendDescriptor_t descriptor = resample();
    ASSERT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
    const auto& lowered
        = object<backend::ResampleDescriptor>(descriptor, CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
    EXPECT_EQ(lowered.mode(), ResampleMode_t::MAXPOOL);
    EXPECT_EQ(lowered.computeType(), DataType_t::FLOAT);
    EXPECT_EQ(lowered.paddingMode(), std::optional<PaddingMode_t>(PaddingMode_t::NEG_INF_PAD));
    EXPECT_EQ(lowered.windowDims(), (std::vector<std::int64_t>{2, 3}));
    EXPECT_EQ(lowered.prePaddings(), (std::vector<std::int64_t>{0, 0}));
    EXPECT_EQ(lowered.postPaddings(), (std::vector<std::int64_t>{0, 0}));
    EXPECT_EQ(lowered.strides(), (std::vector<std::int64_t>{2, 2}));

    std::int32_t nan = -1;
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(descriptor,
                                       CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
                                       CUDNN_TYPE_NAN_PROPOGATION,
                                       1,
                                       &count,
                                       &nan),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(nan, static_cast<std::int32_t>(CUDNN_PROPAGATE_NAN));
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ResampleRequiredAttributesNamed)
{
    for(const cudnnBackendAttributeName_t required :
        {CUDNN_ATTR_RESAMPLE_MODE, CUDNN_ATTR_RESAMPLE_COMP_TYPE, CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS})
    {
        const std::string name = backend::attributeName(required);
        SCOPED_TRACE(name);
        expectFinalizeFails(resample(required), CUDNN_STATUS_BAD_PARAM, name);
    }
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ResampleZeroSpatialDimsRejectedAtFinalize)
{
    // Every SET succeeds (resample() checks each one); only finalize fails.
    cudnnBackendDescriptor_t descriptor
        = resample(std::nullopt, CUDNN_RESAMPLE_MAXPOOL, 0, std::vector<cudnnFraction_t>{});
    expectFinalizeFails(descriptor, CUDNN_STATUS_BAD_PARAM, "CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS");
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ResampleArrayCountsMatchSpatialDims)
{
    for(const cudnnBackendAttributeName_t array : {CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                                                   CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                                   CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                                   CUDNN_ATTR_RESAMPLE_STRIDES})
    {
        const std::string name = backend::attributeName(array);
        SCOPED_TRACE(name);
        cudnnBackendDescriptor_t descriptor = resample();
        ASSERT_EQ(
            setArray(descriptor, array, CUDNN_TYPE_FRACTION, std::vector<cudnnFraction_t>{{1, 1}}),
            CUDNN_STATUS_SUCCESS);
        expectFinalizeFails(descriptor, CUDNN_STATUS_BAD_PARAM, name);
    }
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ResampleZeroDenominatorRejectedAtSet)
{
    cudnnBackendDescriptor_t descriptor = resample();
    EXPECT_EQ(setArray(descriptor,
                       CUDNN_ATTR_RESAMPLE_STRIDES,
                       CUDNN_TYPE_FRACTION,
                       std::vector<cudnnFraction_t>{{2, 1}, {3, 0}}),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_RESAMPLE_STRIDES"), std::string::npos);
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ResampleFractionalValueNotSupported)
{
    cudnnBackendDescriptor_t descriptor = resample(
        std::nullopt, CUDNN_RESAMPLE_MAXPOOL, 2, std::vector<cudnnFraction_t>{{2, 1}, {3, 2}});
    expectFinalizeFails(descriptor, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_ATTR_RESAMPLE_WINDOW_DIMS");

    cudnnBackendDescriptor_t negative = resample(
        std::nullopt, CUDNN_RESAMPLE_MAXPOOL, 2, std::vector<cudnnFraction_t>{{-4, -2}, {6, 3}});
    ASSERT_EQ(cudnnBackendFinalize(negative), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(object<backend::ResampleDescriptor>(negative, CUDNN_BACKEND_RESAMPLE_DESCRIPTOR)
                  .windowDims(),
              (std::vector<std::int64_t>{2, 2}));
}

TEST_F(TestCudnnShimV8ParameterDescriptors, ResampleNearestAndBilinearNotSupported)
{
    for(const cudnnResampleMode_t mode : {CUDNN_RESAMPLE_NEAREST, CUDNN_RESAMPLE_BILINEAR})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        expectFinalizeFails(
            resample(std::nullopt, mode), CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_ATTR_RESAMPLE_MODE");
    }
    for(const cudnnResampleMode_t mode : {CUDNN_RESAMPLE_AVGPOOL,
                                          CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING,
                                          CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING,
                                          CUDNN_RESAMPLE_MAXPOOL})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        EXPECT_EQ(cudnnBackendFinalize(resample(std::nullopt, mode)), CUDNN_STATUS_SUCCESS);
    }
}

TEST_F(TestCudnnShimV8ParameterDescriptors, RngRequiresDistribution)
{
    expectFinalizeFails(create(CUDNN_BACKEND_RNG_DESCRIPTOR),
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_RNG_DISTRIBUTION");
}

TEST_F(TestCudnnShimV8ParameterDescriptors, RngStoresDoubleParameters)
{
    cudnnBackendDescriptor_t rng = create(CUDNN_BACKEND_RNG_DESCRIPTOR);
    ASSERT_EQ(set(rng,
                  CUDNN_ATTR_RNG_DISTRIBUTION,
                  CUDNN_TYPE_RNG_DISTRIBUTION,
                  CUDNN_RNG_DISTRIBUTION_BERNOULLI),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(rng, CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY, CUDNN_TYPE_DOUBLE, 0.25),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(rng, CUDNN_ATTR_RNG_NORMAL_DIST_MEAN, CUDNN_TYPE_DOUBLE, -1.5),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(rng), CUDNN_STATUS_SUCCESS);

    double probability = 0.0;
    double mean = 0.0;
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(rng,
                                       CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &count,
                                       &probability),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendGetAttribute(
                  rng, CUDNN_ATTR_RNG_NORMAL_DIST_MEAN, CUDNN_TYPE_DOUBLE, 1, &count, &mean),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(probability, 0.25);
    EXPECT_EQ(mean, -1.5);
}

} // namespace
