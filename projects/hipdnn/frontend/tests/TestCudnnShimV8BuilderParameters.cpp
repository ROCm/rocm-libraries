// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v0.x parameter and VariantPack builders: build statuses, getters, and what reaches hipDNN.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8BuilderFixture.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{

using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::buildFailure;
using hipdnn_shim_test::Dims;
namespace fe = hipdnn_shim_test::v8fe;

using fe::DataType_t;
using fe::DescriptorType_t;
using fe::PointwiseMode_t;

using Fractions = std::vector<std::pair<std::int64_t, std::int64_t>>;

const Dims X_DIMS = {1, 4, 8, 8};
const Dims W_DIMS = {8, 4, 3, 3};
const Dims PW_DIMS = {2, 3, 4, 5};

Fractions fractions(const cudnnFraction_t* values, std::int64_t count)
{
    Fractions result;
    for(std::int64_t i = 0; i < count; ++i)
    {
        const cudnnFraction_t& value = values[i];
        result.emplace_back(value.numerator, value.denominator);
    }
    return result;
}

template <typename Builder>
void expectBuildStatus(Builder& builder, cudnnStatus_t status)
{
    const BuildFailure failure = buildFailure(builder);
    EXPECT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, status) << failure.what;
}

class TestCudnnShimV8BuilderParameters : public hipdnn_shim_test::ShimV8BuilderFixture
{
protected:
    static void describeConvolution(fe::ConvDescBuilder& builder,
                                    const Dims& pre,
                                    const Dims& post,
                                    const Dims& dilation,
                                    const Dims& stride)
    {
        const auto count = static_cast<std::int64_t>(pre.size());
        builder.setSpatialDimCount(count)
            .setPrePadding(count, pre.data())
            .setPostPadding(count, post.data())
            .setDilation(count, dilation.data())
            .setSpatialStride(count, stride.data());
    }

    // MAXPOOL, NEG_INF_PAD, window 3x3, pre {1,0}, post {0,1}, stride {2,2}.
    static void configureMaxPool(fe::ResampleDescBuilder_v8& builder)
    {
        const std::array<std::int64_t, 2> window = {3, 3};
        const std::array<std::int64_t, 2> pre = {1, 0};
        const std::array<cudnnFraction_t, 2> post = {{{0, 1}, {1, 1}}};
        const std::array<std::int64_t, 2> stride = {2, 2};
        builder.setResampleMode(fe::ResampleMode_t::MAXPOOL)
            .setComputeType(DataType_t::FLOAT)
            .setPaddingMode(CUDNN_NEG_INF_PAD)
            .setSpatialDim(2, window.data())
            .setPrePadding(2, pre.data())
            .setPostPadding(2, post.data())
            .setSpatialStride(2, stride.data());
    }

    // MAXPOOL, NEG_INF_PAD, zero paddings, stride 2, the given window fractions.
    static void configureFractionalPool(fe::ResampleDescBuilder_v8& builder,
                                        const std::array<cudnnFraction_t, 2>& window)
    {
        const std::array<cudnnFraction_t, 2> zeros = {{{0, 1}, {0, 1}}};
        const std::array<cudnnFraction_t, 2> strides = {{{2, 1}, {2, 1}}};
        builder.setResampleMode(CUDNN_RESAMPLE_MAXPOOL)
            .setComputeType(CUDNN_DATA_FLOAT)
            .setNanPropagation(CUDNN_PROPAGATE_NAN)
            .setPaddingMode(fe::PaddingMode_t::NEG_INF_PAD)
            .setSpatialDim(2, window.data())
            .setPrePadding(2, zeros.data())
            .setPostPadding(2, zeros.data())
            .setSpatialStride(2, strides.data());
    }

    // The hipDNN operation of `type` created last, after lowering a one-op graph.
    hipdnnBackendDescriptor_t lower(const fe::Operation& op,
                                    hipdnnBackendDescriptorType_t type) const
    {
        [[maybe_unused]] const fe::OperationGraph built = graph({&op});
        const std::vector<hipdnnBackendDescriptor_t> all = created(type);
        EXPECT_FALSE(all.empty());
        return all.empty() ? nullptr : all.back();
    }

    hipdnnBackendDescriptor_t lowerConvolution(const fe::ConvDesc& conv, const Dims& yDims) const
    {
        const fe::Operation op
            = fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                  .setxDesc(tensor(1, X_DIMS))
                  .setwDesc(tensor(2, W_DIMS))
                  .setyDesc(tensor(3, yDims))
                  .setcDesc(conv)
                  .build();
        return lower(op, HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    }

    // Backward modes take dy and dx in place of y.
    hipdnnBackendDescriptor_t lowerPointwise(const fe::PointWiseDesc& pointwise,
                                             bool backward = false) const
    {
        fe::OperationBuilder builder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
        builder.setpwDesc(pointwise).setxDesc(tensor(1, PW_DIMS));
        if(backward)
        {
            builder.setdyDesc(tensor(2, PW_DIMS)).setdxDesc(tensor(3, PW_DIMS));
        }
        else
        {
            builder.setyDesc(tensor(2, PW_DIMS));
        }
        const fe::Operation op = builder.build();
        return lower(op, HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    }

    hipdnnBackendDescriptor_t lowerReduction(const fe::ReductionDesc& reduction) const
    {
        const fe::Operation op
            = fe::OperationBuilder(DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR)
                  .setxDesc(tensor(1, X_DIMS))
                  .setyDesc(tensor(2, {1, 4, 1, 1}))
                  .setreductionDesc(reduction)
                  .build();
        return lower(op, HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    }

    hipdnnBackendDescriptor_t lowerResample(const fe::ResampleDesc_v8& resample,
                                            const Dims& yDims) const
    {
        const fe::Operation op
            = fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                  .setxDesc(tensor(1, X_DIMS))
                  .setyDesc(tensor(2, yDims))
                  .setResampleDesc(resample)
                  .build();
        return lower(op, HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
    }

    std::vector<std::int64_t> int64s(hipdnnBackendDescriptor_t op,
                                     hipdnnBackendAttributeName_t name) const
    {
        const auto call = callOn(op, name);
        EXPECT_TRUE(call.has_value()) << "attribute " << name << " not set";
        return call ? call->values<std::int64_t>() : std::vector<std::int64_t>{};
    }

    template <typename T>
    T scalarOn(hipdnnBackendDescriptor_t op, hipdnnBackendAttributeName_t name) const
    {
        const auto call = callOn(op, name);
        EXPECT_TRUE(call.has_value()) << "attribute " << name << " not set";
        return call ? call->value<T>() : T{};
    }

    bool isSet(hipdnnBackendDescriptor_t op, hipdnnBackendAttributeName_t name) const
    {
        return callOn(op, name).has_value();
    }

    bool variantPackCreatedOnHipdnn() const
    {
        return !created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty();
    }
};

// ---- ConvDesc ----

TEST_F(TestCudnnShimV8BuilderParameters, ConvDescGettersReturnSetValues)
{
    fe::ConvDescBuilder builder;
    builder.setComputeType(DataType_t::FLOAT).setMathMode(CUDNN_CROSS_CORRELATION);
    describeConvolution(builder, {1, 2}, {3, 4}, {1, 2}, {2, 3});
    const fe::ConvDesc conv = builder.build();

    ASSERT_EQ(conv.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(conv.getComputeType(), DataType_t::FLOAT);
    EXPECT_EQ(conv.getComputePrecision(), DataType_t::FLOAT);
    EXPECT_EQ(conv.getMathMode(), CUDNN_CROSS_CORRELATION);
    EXPECT_EQ(conv.getSpatialDimCount(), 2);
    EXPECT_EQ(conv.getDimensionCount(), 2);
    EXPECT_EQ(Dims(conv.getPadding(), conv.getPadding() + 2), (Dims{1, 2}));
    EXPECT_EQ(Dims(conv.getDilation(), conv.getDilation() + 2), (Dims{1, 2}));
    EXPECT_EQ(Dims(conv.getSpatialStride(), conv.getSpatialStride() + 2), (Dims{2, 3}));
    EXPECT_EQ(Dims(conv.getStride(), conv.getStride() + 2), (Dims{2, 3}));
}

TEST_F(TestCudnnShimV8BuilderParameters, ConvDescParametersReachHipdnn)
{
    fe::ConvDescBuilder builder;
    builder.setComputeType(CUDNN_DATA_FLOAT).setMathMode(fe::ConvolutionMode_t::CROSS_CORRELATION);
    describeConvolution(builder, {1, 2}, {3, 4}, {1, 2}, {2, 3});
    const fe::ConvDesc conv = builder.build();

    // H: (8 + 1 + 3 - 3) / 2 + 1 = 5; W: (8 + 2 + 4 - 5) / 3 + 1 = 4.
    hipdnnBackendDescriptor_t op = lowerConvolution(conv, {1, 8, 5, 4});
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS), (Dims{1, 2}));
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS), (Dims{3, 4}));
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_CONVOLUTION_DILATIONS), (Dims{1, 2}));
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES), (Dims{2, 3}));
    EXPECT_EQ(scalarOn<hipdnnConvolutionMode_t>(op, HIPDNN_ATTR_CONVOLUTION_CONV_MODE),
              HIPDNN_CROSS_CORRELATION);
    EXPECT_EQ(scalarOn<hipdnnDataType_t>(op, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE), HIPDNN_DATA_FLOAT);
}

TEST_F(TestCudnnShimV8BuilderParameters, ConvDescModeDefaultsToConvolutionAndNotSetFallsBack)
{
    fe::ConvDescBuilder byDefault;
    describeConvolution(byDefault, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    const fe::ConvDesc defaulted = byDefault.build();
    EXPECT_EQ(defaulted.getMathMode(), CUDNN_CONVOLUTION);
    EXPECT_EQ(defaulted.getComputeType(), DataType_t::FLOAT);
    hipdnnBackendDescriptor_t defaultOp = lowerConvolution(defaulted, {1, 8, 8, 8});
    EXPECT_EQ(scalarOn<hipdnnConvolutionMode_t>(defaultOp, HIPDNN_ATTR_CONVOLUTION_CONV_MODE),
              HIPDNN_CONVOLUTION);
    EXPECT_EQ(scalarOn<hipdnnDataType_t>(defaultOp, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE),
              HIPDNN_DATA_FLOAT);

    fe::ConvDescBuilder notSetBuilder;
    notSetBuilder.setMathMode(fe::ConvolutionMode_t::NOT_SET);
    describeConvolution(notSetBuilder, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    const fe::ConvDesc notSet = notSetBuilder.build();
    EXPECT_EQ(notSet.getMathMode(), CUDNN_CROSS_CORRELATION);
    hipdnnBackendDescriptor_t notSetOp = lowerConvolution(notSet, {1, 8, 8, 8});
    EXPECT_EQ(scalarOn<hipdnnConvolutionMode_t>(notSetOp, HIPDNN_ATTR_CONVOLUTION_CONV_MODE),
              HIPDNN_CROSS_CORRELATION);
}

TEST_F(TestCudnnShimV8BuilderParameters, ConvDescDescribeKeepsUpstreamSeparators)
{
    fe::ConvDescBuilder builder;
    builder.setComputeType(DataType_t::FLOAT).setMathMode(fe::ConvolutionMode_t::CROSS_CORRELATION);
    describeConvolution(builder, {0, 0}, {0, 0}, {1, 1}, {1, 1});
    const fe::ConvDesc conv = builder.build();

    const std::string text = conv.describe();
    EXPECT_NE(text.find("PadLower [ 0,0 ] PadUpper [,0,0 ]"), std::string::npos) << text;
}

TEST_F(TestCudnnShimV8BuilderParameters, ConvDescRequiredFieldStatuses)
{
    const std::array<std::int64_t, 2> ones = {1, 1};
    struct Case
    {
        const char* label;
        std::function<void(fe::ConvDescBuilder&)> describe;
        cudnnStatus_t status;
    };
    const std::vector<Case> cases = {
        {"no spatial dim count",
         [&ones](fe::ConvDescBuilder& b) {
             b.setPrePadding(2, ones.data())
                 .setPostPadding(2, ones.data())
                 .setDilation(2, ones.data())
                 .setSpatialStride(2, ones.data());
         },
         CUDNN_STATUS_BAD_PARAM},
        {"zero spatial dim count",
         [](fe::ConvDescBuilder& b) {
             describeConvolution(b, {1, 1}, {1, 1}, {1, 1}, {1, 1});
             b.setSpatialDimCount(0);
         },
         CUDNN_STATUS_BAD_PARAM},
        {"no strides",
         [&ones](fe::ConvDescBuilder& b) {
             b.setSpatialDimCount(2)
                 .setPrePadding(2, ones.data())
                 .setPostPadding(2, ones.data())
                 .setDilation(2, ones.data());
         },
         CUDNN_STATUS_BAD_PARAM},
        {"zero first stride",
         [](fe::ConvDescBuilder& b) { describeConvolution(b, {1, 1}, {1, 1}, {1, 1}, {0, 1}); },
         CUDNN_STATUS_BAD_PARAM},
        {"negative first stride",
         [](fe::ConvDescBuilder& b) { describeConvolution(b, {1, 1}, {1, 1}, {1, 1}, {-1, 1}); },
         CUDNN_STATUS_BAD_PARAM},
        {"compute type NOT_SET",
         [](fe::ConvDescBuilder& b) {
             describeConvolution(b, {1, 1}, {1, 1}, {1, 1}, {1, 1});
             b.setComputeType(DataType_t::NOT_SET);
         },
         CUDNN_STATUS_INVALID_VALUE},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        fe::ConvDescBuilder builder;
        c.describe(builder);
        expectBuildStatus(builder, c.status);
    }
}

TEST_F(TestCudnnShimV8BuilderParameters, ConvDescUnlowerableComputeTypeIsNotSupported)
{
    fe::ConvDescBuilder builder;
    builder.setComputeType(DataType_t::INT8x32);
    describeConvolution(builder, {1, 1}, {1, 1}, {1, 1}, {1, 1});

    expectBuildStatus(builder, CUDNN_STATUS_NOT_SUPPORTED);
}

// Upstream's argument checks still apply; arguments that pass them get NOT_SUPPORTED, never a
// SUCCESS that would leave the caller with an unreordered filter.
TEST_F(TestCudnnShimV8BuilderParameters, ReorderFilterAndBiasInt8x32Statuses)
{
    const auto conv = [](std::size_t spatialDims) {
        const Dims ones(spatialDims, 1);
        fe::ConvDescBuilder builder;
        builder.setComputeType(DataType_t::FLOAT).setMathMode(CUDNN_CROSS_CORRELATION);
        describeConvolution(builder, ones, ones, ones, ones);
        return fe::ConvDesc(builder.build());
    };
    const fe::ConvDesc conv1d = conv(1);
    const fe::ConvDesc conv2d = conv(2);
    const fe::Tensor filter3d = tensor(1, {8, 4, 3});
    const fe::Tensor filter4d = tensor(2, W_DIMS);
    int filter = 0;
    int reorderedFilter = 0;
    int bias = 0;
    int reorderedBias = 0;

    struct Case
    {
        const char* label;
        const fe::Tensor* tensor;
        const fe::ConvDesc* conv;
        void* filter;
        void* reorderedFilter;
        void* bias;
        void* reorderedBias;
        cudnnStatus_t status;
    };
    const std::vector<Case> cases = {
        {"filter without reordered filter",
         &filter4d,
         &conv2d,
         &filter,
         nullptr,
         nullptr,
         nullptr,
         CUDNN_STATUS_BAD_PARAM},
        {"bias without reordered bias",
         &filter4d,
         &conv2d,
         &filter,
         &reorderedFilter,
         &bias,
         nullptr,
         CUDNN_STATUS_BAD_PARAM},
        {"3-D filter, 2-D convolution",
         &filter3d,
         &conv2d,
         &filter,
         &reorderedFilter,
         &bias,
         &reorderedBias,
         CUDNN_STATUS_BAD_PARAM},
        {"3-D filter, 1-D convolution",
         &filter3d,
         &conv1d,
         &filter,
         &reorderedFilter,
         &bias,
         &reorderedBias,
         CUDNN_STATUS_BAD_PARAM},
        {"4-D filter, 2-D convolution",
         &filter4d,
         &conv2d,
         &filter,
         &reorderedFilter,
         &bias,
         &reorderedBias,
         CUDNN_STATUS_NOT_SUPPORTED},
        {"no bias",
         &filter4d,
         &conv2d,
         &filter,
         &reorderedFilter,
         nullptr,
         nullptr,
         CUDNN_STATUS_NOT_SUPPORTED},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        EXPECT_EQ(
            fe::cudnnReorderFilterAndBiasInt8x32(
                _handle, *c.tensor, *c.conv, c.filter, c.reorderedFilter, c.bias, c.reorderedBias),
            c.status);
    }
}

// ---- PointWiseDesc ----

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseModeSupportFollowsHipdnnMapping)
{
    // The cuDNN modes hipDNN has no pointwise mode for.
    const std::set<cudnnPointwiseMode_t> unsupported
        = {CUDNN_POINTWISE_COS, CUDNN_POINTWISE_MOD, CUDNN_POINTWISE_POW};
    std::set<PointwiseMode_t> built;
    for(int raw = CUDNN_POINTWISE_ABS; raw <= CUDNN_POINTWISE_TANH_FWD; ++raw)
    {
        const auto mode = static_cast<cudnnPointwiseMode_t>(raw);
        SCOPED_TRACE(raw);
        fe::PointWiseDescBuilder builder;
        builder.setMode(mode).setComputeType(DataType_t::FLOAT);
        if(unsupported.count(mode) != 0)
        {
            expectBuildStatus(builder, CUDNN_STATUS_NOT_SUPPORTED);
            continue;
        }
        const fe::PointWiseDesc desc = builder.build();
        ASSERT_EQ(desc.get_status(), CUDNN_STATUS_SUCCESS);
        EXPECT_NE(desc.getPointWiseMode(), PointwiseMode_t::NOT_SET);
        built.insert(desc.getPointWiseMode());
    }
    EXPECT_EQ(built.size(), 47U);

    fe::PointWiseDescBuilder cosine;
    cosine.setMode(PointwiseMode_t::COS).setComputeType(DataType_t::FLOAT);
    expectBuildStatus(cosine, CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseRequiredFieldStatuses)
{
    fe::PointWiseDescBuilder noMode;
    noMode.setComputeType(DataType_t::FLOAT);
    expectBuildStatus(noMode, CUDNN_STATUS_INVALID_VALUE);

    fe::PointWiseDescBuilder noComputeType;
    noComputeType.setMode(PointwiseMode_t::ADD).setComputeType(DataType_t::NOT_SET);
    expectBuildStatus(noComputeType, CUDNN_STATUS_INVALID_VALUE);
}

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseDefaultReluLowersNoParameters)
{
    const fe::PointWiseDesc relu = fe::PointWiseDescBuilder()
                                       .setMode(PointwiseMode_t::RELU_FWD)
                                       .setComputeType(DataType_t::FLOAT)
                                       .build();
    EXPECT_EQ(relu.getPortCount(), 2);

    hipdnnBackendDescriptor_t op = lowerPointwise(relu);
    EXPECT_EQ(scalarOn<hipdnnPointwiseMode_t>(op, HIPDNN_ATTR_POINTWISE_MODE),
              HIPDNN_POINTWISE_RELU_FWD);
    EXPECT_EQ(scalarOn<hipdnnDataType_t>(op, HIPDNN_ATTR_POINTWISE_MATH_PREC), HIPDNN_DATA_FLOAT);
    EXPECT_FALSE(isSet(op, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP));
    EXPECT_FALSE(isSet(op, HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP));
    EXPECT_FALSE(isSet(op, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE));
}

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseDefaultParametersElideAtLowering)
{
    struct Case
    {
        const char* label;
        PointwiseMode_t mode;
        DataType_t computeType;
        std::function<void(fe::PointWiseDescBuilder&)> describe;
        std::vector<hipdnnBackendAttributeName_t> absent;
    };
    const std::vector<Case> cases = {
        // An upper clip at or above FLT_MAX, the DBL_MAX default included, is not sent.
        {"relu double",
         PointwiseMode_t::RELU_FWD,
         DataType_t::DOUBLE,
         [](fe::PointWiseDescBuilder&) {},
         {HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
          HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
          HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE}},
        {"relu explicit zero lower clip",
         PointwiseMode_t::RELU_FWD,
         DataType_t::FLOAT,
         [](fe::PointWiseDescBuilder& b) { b.setReluLowerClip(0.0); },
         {HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP}},
        {"relu upper clip at float max",
         PointwiseMode_t::RELU_FWD,
         DataType_t::FLOAT,
         [](fe::PointWiseDescBuilder& b) {
             b.setReluUpperClip(static_cast<double>(std::numeric_limits<float>::max()));
         },
         {HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP}},
        {"relu upper clip infinite",
         PointwiseMode_t::RELU_FWD,
         DataType_t::FLOAT,
         [](fe::PointWiseDescBuilder& b) {
             b.setReluUpperClip(std::numeric_limits<double>::infinity());
         },
         {HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP}},
        {"elu",
         PointwiseMode_t::ELU_FWD,
         DataType_t::FLOAT,
         [](fe::PointWiseDescBuilder&) {},
         {HIPDNN_ATTR_POINTWISE_ELU_ALPHA}},
        {"softplus",
         PointwiseMode_t::SOFTPLUS_FWD,
         DataType_t::FLOAT,
         [](fe::PointWiseDescBuilder&) {},
         {HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA}},
        {"swish",
         PointwiseMode_t::SWISH_FWD,
         DataType_t::FLOAT,
         [](fe::PointWiseDescBuilder&) {},
         {HIPDNN_ATTR_POINTWISE_SWISH_BETA}},
        {"gen index",
         PointwiseMode_t::GEN_INDEX,
         DataType_t::FLOAT,
         [](fe::PointWiseDescBuilder&) {},
         {HIPDNN_ATTR_POINTWISE_AXIS}},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        fe::PointWiseDescBuilder builder;
        builder.setMode(c.mode).setComputeType(c.computeType);
        c.describe(builder);
        const fe::PointWiseDesc desc = builder.build();
        hipdnnBackendDescriptor_t op = lowerPointwise(desc);
        ASSERT_NE(op, nullptr);
        EXPECT_TRUE(isSet(op, HIPDNN_ATTR_POINTWISE_MODE));
        for(const hipdnnBackendAttributeName_t name : c.absent)
        {
            EXPECT_FALSE(isSet(op, name)) << name;
        }
    }
}

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseExplicitParametersReachHipdnn)
{
    for(const PointwiseMode_t mode : {PointwiseMode_t::RELU_FWD, PointwiseMode_t::RELU_BWD})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const fe::PointWiseDesc relu = fe::PointWiseDescBuilder()
                                           .setMode(mode)
                                           .setMathPrecision(CUDNN_DATA_FLOAT)
                                           .setReluLowerClip(-1.0)
                                           .setReluUpperClip(6.0)
                                           .setReluLowerClipSlope(0.1)
                                           .build();
        hipdnnBackendDescriptor_t reluOp = lowerPointwise(relu, mode == PointwiseMode_t::RELU_BWD);
        EXPECT_EQ(scalarOn<hipdnnDataType_t>(reluOp, HIPDNN_ATTR_POINTWISE_MATH_PREC),
                  HIPDNN_DATA_FLOAT);
        EXPECT_EQ(scalarOn<float>(reluOp, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP), -1.0F);
        EXPECT_EQ(scalarOn<float>(reluOp, HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP), 6.0F);
        EXPECT_EQ(scalarOn<float>(reluOp, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE), 0.1F);
    }

    const fe::PointWiseDesc clipped = fe::PointWiseDescBuilder()
                                          .setMode(PointwiseMode_t::RELU_FWD)
                                          .setComputeType(DataType_t::FLOAT)
                                          .setClipping(-2.0, 3.0)
                                          .build();
    hipdnnBackendDescriptor_t clippedOp = lowerPointwise(clipped);
    EXPECT_EQ(scalarOn<float>(clippedOp, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP), -2.0F);
    EXPECT_EQ(scalarOn<float>(clippedOp, HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP), 3.0F);

    struct Case
    {
        const char* label;
        PointwiseMode_t mode;
        bool backward;
        std::function<void(fe::PointWiseDescBuilder&)> describe;
        hipdnnBackendAttributeName_t name;
        float expected;
    };
    const std::vector<Case> cases = {
        {"elu alpha",
         PointwiseMode_t::ELU_FWD,
         false,
         [](fe::PointWiseDescBuilder& b) { b.setEluAlpha(0.5); },
         HIPDNN_ATTR_POINTWISE_ELU_ALPHA,
         0.5F},
        {"elu backward alpha",
         PointwiseMode_t::ELU_BWD,
         true,
         [](fe::PointWiseDescBuilder& b) { b.setEluAlpha(0.5); },
         HIPDNN_ATTR_POINTWISE_ELU_ALPHA,
         0.5F},
        {"softplus beta",
         PointwiseMode_t::SOFTPLUS_FWD,
         false,
         [](fe::PointWiseDescBuilder& b) { b.setSoftplusBeta(2.0); },
         HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
         2.0F},
        {"softplus backward beta",
         PointwiseMode_t::SOFTPLUS_BWD,
         true,
         [](fe::PointWiseDescBuilder& b) { b.setSoftplusBeta(2.0); },
         HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
         2.0F},
        {"swish beta",
         PointwiseMode_t::SWISH_FWD,
         false,
         [](fe::PointWiseDescBuilder& b) { b.setSwishBeta(3.0); },
         HIPDNN_ATTR_POINTWISE_SWISH_BETA,
         3.0F},
        {"swish backward beta",
         PointwiseMode_t::SWISH_BWD,
         true,
         [](fe::PointWiseDescBuilder& b) { b.setSwishBeta(2.0); },
         HIPDNN_ATTR_POINTWISE_SWISH_BETA,
         2.0F},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        fe::PointWiseDescBuilder builder;
        builder.setMode(c.mode).setComputeType(DataType_t::FLOAT);
        c.describe(builder);
        const fe::PointWiseDesc desc = builder.build();
        EXPECT_EQ(scalarOn<float>(lowerPointwise(desc, c.backward), c.name), c.expected);
    }

    const fe::PointWiseDesc genIndex = fe::PointWiseDescBuilder()
                                           .setMode(PointwiseMode_t::GEN_INDEX)
                                           .setComputeType(DataType_t::FLOAT)
                                           .setAxis(2)
                                           .build();
    EXPECT_EQ(scalarOn<std::int64_t>(lowerPointwise(genIndex), HIPDNN_ATTR_POINTWISE_AXIS), 2);
}

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseZeroLowerClipElidedBesideSlope)
{
    const fe::PointWiseDesc leaky = fe::PointWiseDescBuilder()
                                        .setMode(PointwiseMode_t::RELU_FWD)
                                        .setComputeType(DataType_t::FLOAT)
                                        .setReluLowerClip(0.0)
                                        .setReluLowerClipSlope(0.01)
                                        .build();
    hipdnnBackendDescriptor_t op = lowerPointwise(leaky);
    EXPECT_FALSE(isSet(op, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP));
    EXPECT_EQ(scalarOn<float>(op, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE), 0.01F);
}

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseNonFiniteParametersAreBadParam)
{
    struct Case
    {
        const char* label;
        PointwiseMode_t mode;
        std::function<void(fe::PointWiseDescBuilder&, double)> set;
    };
    const std::vector<Case> cases = {
        {"relu lower clip slope",
         PointwiseMode_t::RELU_FWD,
         [](fe::PointWiseDescBuilder& b, double v) { b.setReluLowerClipSlope(v); }},
        {"relu backward lower clip slope",
         PointwiseMode_t::RELU_BWD,
         [](fe::PointWiseDescBuilder& b, double v) { b.setReluLowerClipSlope(v); }},
        {"elu alpha",
         PointwiseMode_t::ELU_FWD,
         [](fe::PointWiseDescBuilder& b, double v) { b.setEluAlpha(v); }},
        {"elu backward alpha",
         PointwiseMode_t::ELU_BWD,
         [](fe::PointWiseDescBuilder& b, double v) { b.setEluAlpha(v); }},
        {"softplus beta",
         PointwiseMode_t::SOFTPLUS_FWD,
         [](fe::PointWiseDescBuilder& b, double v) { b.setSoftplusBeta(v); }},
        {"softplus backward beta",
         PointwiseMode_t::SOFTPLUS_BWD,
         [](fe::PointWiseDescBuilder& b, double v) { b.setSoftplusBeta(v); }},
        {"swish beta",
         PointwiseMode_t::SWISH_FWD,
         [](fe::PointWiseDescBuilder& b, double v) { b.setSwishBeta(v); }},
        {"swish backward beta",
         PointwiseMode_t::SWISH_BWD,
         [](fe::PointWiseDescBuilder& b, double v) { b.setSwishBeta(v); }},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        for(const double value : {std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()})
        {
            SCOPED_TRACE(value);
            fe::PointWiseDescBuilder builder;
            builder.setMode(c.mode).setComputeType(DataType_t::FLOAT);
            c.set(builder, value);
            expectBuildStatus(builder, CUDNN_STATUS_BAD_PARAM);
        }
    }
}

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseReluClipsAndOtherModesParameters)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for(const PointwiseMode_t mode : {PointwiseMode_t::RELU_FWD, PointwiseMode_t::RELU_BWD})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        fe::PointWiseDescBuilder lower;
        lower.setMode(mode).setComputeType(DataType_t::FLOAT).setReluLowerClip(nan);
        expectBuildStatus(lower, CUDNN_STATUS_BAD_PARAM);
        fe::PointWiseDescBuilder upper;
        upper.setMode(mode).setComputeType(DataType_t::FLOAT).setReluUpperClip(nan);
        expectBuildStatus(upper, CUDNN_STATUS_BAD_PARAM);

        // Infinite clips are unbounded sides; other modes' parameters are not sent, as upstream.
        const fe::PointWiseDesc unbounded = fe::PointWiseDescBuilder()
                                                .setMode(mode)
                                                .setComputeType(DataType_t::FLOAT)
                                                .setClipping(-inf, inf)
                                                .setEluAlpha(nan)
                                                .setSoftplusBeta(nan)
                                                .setSwishBeta(nan)
                                                .build();
        EXPECT_EQ(unbounded.get_status(), CUDNN_STATUS_SUCCESS) << unbounded.get_error();
    }
}

TEST_F(TestCudnnShimV8BuilderParameters, PointwiseNanPropagationSetterKeepsMode)
{
    // The cudnnNanPropagation_t overload of setMode sets the NaN option, not the mode.
    const fe::PointWiseDesc desc = fe::PointWiseDescBuilder()
                                       .setMode(PointwiseMode_t::RELU_BWD)
                                       .setComputeType(DataType_t::FLOAT)
                                       .setMode(CUDNN_PROPAGATE_NAN)
                                       .build();
    ASSERT_EQ(desc.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(desc.getPointWiseMode(), PointwiseMode_t::RELU_BWD);

    const fe::Operation op = fe::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                 .setpwDesc(desc)
                                 .setxDesc(tensor(1, PW_DIMS))
                                 .setdyDesc(tensor(2, PW_DIMS))
                                 .setdxDesc(tensor(3, PW_DIMS))
                                 .build();
    hipdnnBackendDescriptor_t lowered = lower(op, HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    EXPECT_EQ(scalarOn<hipdnnPointwiseMode_t>(lowered, HIPDNN_ATTR_POINTWISE_MODE),
              HIPDNN_POINTWISE_RELU_BWD);
}

// ---- MatMulDesc ----

TEST_F(TestCudnnShimV8BuilderParameters, MatMulRequiresComputeType)
{
    fe::MatMulDescBuilder unset;
    expectBuildStatus(unset, CUDNN_STATUS_INVALID_VALUE);

    const fe::MatMulDesc matmul = fe::MatMulDescBuilder().setComputeType(DataType_t::FLOAT).build();
    ASSERT_EQ(matmul.get_status(), CUDNN_STATUS_SUCCESS);
    const fe::Operation op = fe::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                 .setaMatDesc(tensor(1, {1, 4, 8}))
                                 .setbMatDesc(tensor(2, {1, 8, 6}))
                                 .setcMatDesc(tensor(3, {1, 4, 6}))
                                 .setmatmulDesc(matmul)
                                 .build();
    hipdnnBackendDescriptor_t lowered = lower(op, HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
    EXPECT_EQ(scalarOn<hipdnnDataType_t>(lowered, HIPDNN_ATTR_MATMUL_COMP_TYPE), HIPDNN_DATA_FLOAT);
}

TEST_F(TestCudnnShimV8BuilderParameters, MatMulPaddingValueIsNotSupported)
{
    for(const double padding : {0.0, 1.5})
    {
        SCOPED_TRACE(padding);
        fe::MatMulDescBuilder builder;
        builder.setComputeType(DataType_t::FLOAT).setPaddingValue(padding);
        expectBuildStatus(builder, CUDNN_STATUS_NOT_SUPPORTED);
    }
}

// ---- ReductionDesc ----

TEST_F(TestCudnnShimV8BuilderParameters, ReductionRequiredFieldStatuses)
{
    fe::ReductionDescBuilder neither;
    expectBuildStatus(neither, CUDNN_STATUS_INVALID_VALUE);

    fe::ReductionDescBuilder noOperator;
    noOperator.setComputeType(DataType_t::FLOAT);
    expectBuildStatus(noOperator, CUDNN_STATUS_INVALID_VALUE);

    fe::ReductionDescBuilder noComputeType;
    noComputeType.setReductionOp(CUDNN_REDUCE_TENSOR_ADD);
    expectBuildStatus(noComputeType, CUDNN_STATUS_INVALID_VALUE);
}

TEST_F(TestCudnnShimV8BuilderParameters, ReductionOperatorSupportFollowsHipdnnMapping)
{
    std::set<std::string> descriptions;
    std::set<hipdnnReduceTensorOp_t> lowered;
    for(int raw = CUDNN_REDUCE_TENSOR_ADD; raw <= CUDNN_REDUCE_TENSOR_NORM2; ++raw)
    {
        SCOPED_TRACE(raw);
        const auto mode = static_cast<cudnnReduceTensorOp_t>(raw);
        const fe::ReductionDesc reduction = fe::ReductionDescBuilder()
                                                .setReductionOp(mode)
                                                .setMathPrecision(CUDNN_DATA_FLOAT)
                                                .build();
        ASSERT_EQ(reduction.get_status(), CUDNN_STATUS_SUCCESS);
        descriptions.insert(reduction.describe());
        lowered.insert(scalarOn<hipdnnReduceTensorOp_t>(lowerReduction(reduction),
                                                        HIPDNN_ATTR_REDUCTION_OPERATOR));
    }
    EXPECT_EQ(descriptions.size(), 9U);
    EXPECT_EQ(lowered.size(), 9U);
}

TEST_F(TestCudnnShimV8BuilderParameters, ReductionParametersReachHipdnn)
{
    const fe::ReductionDesc deterministic = fe::ReductionDescBuilder()
                                                .setComputeType(DataType_t::FLOAT)
                                                .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                                                .setIsDeterministic(true)
                                                .build();
    ASSERT_EQ(deterministic.get_status(), CUDNN_STATUS_SUCCESS);
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    EXPECT_NE(deterministic.describe().find(R"(Reduction mode ["MAX"] Is deterministic true)"),
              std::string::npos)
        << deterministic.describe();
#endif
    hipdnnBackendDescriptor_t op = lowerReduction(deterministic);
    EXPECT_EQ(scalarOn<hipdnnReduceTensorOp_t>(op, HIPDNN_ATTR_REDUCTION_OPERATOR),
              HIPDNN_REDUCE_TENSOR_MAX);
    EXPECT_EQ(scalarOn<hipdnnDataType_t>(op, HIPDNN_ATTR_REDUCTION_COMP_TYPE), HIPDNN_DATA_FLOAT);
    EXPECT_TRUE(scalarOn<bool>(op, HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC));

    const fe::ReductionDesc byDefault = fe::ReductionDescBuilder()
                                            .setComputeType(DataType_t::FLOAT)
                                            .setReductionOp(fe::ReductionMode_t::ADD)
                                            .build();
    EXPECT_FALSE(scalarOn<bool>(lowerReduction(byDefault), HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC));
}

// ---- ResampleDesc ----

TEST_F(TestCudnnShimV8BuilderParameters, ResampleGettersReturnSetValues)
{
    fe::ResampleDescBuilder_v8 builder;
    configureMaxPool(builder);
    const fe::ResampleDesc_v8 resample = builder.build();

    ASSERT_EQ(resample.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(resample.getMode(), fe::ResampleMode_t::MAXPOOL);
    EXPECT_EQ(resample.getPaddingMode(), fe::PaddingMode_t::NEG_INF_PAD);
    EXPECT_EQ(resample.getComputeType(), DataType_t::FLOAT);
    EXPECT_EQ(resample.getNanOpt(), CUDNN_PROPAGATE_NAN);
    EXPECT_EQ(resample.getSpatialDimCount(), 2);
    EXPECT_EQ(fractions(resample.getWindowDim(), 2), (Fractions{{3, 1}, {3, 1}}));
    EXPECT_EQ(fractions(resample.getPrePadding(), 2), (Fractions{{1, 1}, {0, 1}}));
    EXPECT_EQ(fractions(resample.getPostPadding(), 2), (Fractions{{0, 1}, {1, 1}}));
    EXPECT_EQ(fractions(resample.getSpatialStride(), 2), (Fractions{{2, 1}, {2, 1}}));
}

TEST_F(TestCudnnShimV8BuilderParameters, ResampleIntegralFractionsReachHipdnn)
{
    fe::ResampleDescBuilder_v8 builder;
    configureFractionalPool(builder, {{{4, 2}, {3, 1}}});
    const fe::ResampleDesc_v8 resample = builder.build();
    ASSERT_EQ(resample.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(resample.getNanOpt(), CUDNN_PROPAGATE_NAN);

    // H: (8 - 2) / 2 + 1 = 4; W: (8 - 3) / 2 + 1 = 3.
    hipdnnBackendDescriptor_t op = lowerResample(resample, {1, 4, 4, 3});
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS), (Dims{2, 3}));
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS), (Dims{0, 0}));
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_RESAMPLE_POST_PADDINGS), (Dims{0, 0}));
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_RESAMPLE_STRIDES), (Dims{2, 2}));
    EXPECT_EQ(scalarOn<hipdnnResampleMode_t>(op, HIPDNN_ATTR_RESAMPLE_MODE),
              HIPDNN_RESAMPLE_MAXPOOL);
    EXPECT_EQ(scalarOn<hipdnnPaddingMode_t>(op, HIPDNN_ATTR_RESAMPLE_PADDING_MODE),
              HIPDNN_PADDING_NEG_INF_PAD);
    EXPECT_EQ(scalarOn<hipdnnDataType_t>(op, HIPDNN_ATTR_RESAMPLE_COMP_TYPE), HIPDNN_DATA_FLOAT);
}

TEST_F(TestCudnnShimV8BuilderParameters, ResampleRequiredFieldStatuses)
{
    const std::array<std::int64_t, 2> window = {2, 2};
    struct Case
    {
        const char* label;
        std::function<void(fe::ResampleDescBuilder_v8&)> breakIt;
        cudnnStatus_t status;
    };
    const std::vector<Case> cases = {
        {"mode NOT_SET",
         [](fe::ResampleDescBuilder_v8& b) { b.setResampleMode(fe::ResampleMode_t::NOT_SET); },
         CUDNN_STATUS_INVALID_VALUE},
        {"compute type NOT_SET",
         [](fe::ResampleDescBuilder_v8& b) { b.setComputeType(DataType_t::NOT_SET); },
         CUDNN_STATUS_INVALID_VALUE},
        {"padding mode NOT_SET",
         [](fe::ResampleDescBuilder_v8& b) { b.setPaddingMode(fe::PaddingMode_t::NOT_SET); },
         CUDNN_STATUS_INVALID_VALUE},
        {"zero spatial dims",
         [&window](fe::ResampleDescBuilder_v8& b) { b.setSpatialDim(0, window.data()); },
         CUDNN_STATUS_BAD_PARAM},
        {"negative spatial dims",
         [&window](fe::ResampleDescBuilder_v8& b) { b.setSpatialDim(-1, window.data()); },
         CUDNN_STATUS_BAD_PARAM},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        fe::ResampleDescBuilder_v8 builder;
        configureMaxPool(builder);
        c.breakIt(builder);
        expectBuildStatus(builder, c.status);
    }

    // Upstream leaves the spatial dimension count 0 until setSpatialDim, with or without the
    // paddings and stride, through either enum family.
    const std::array<std::int64_t, 2> zeros = {0, 0};
    fe::ResampleDescBuilder_v8 noWindow;
    noWindow.setResampleMode(fe::ResampleMode_t::MAXPOOL)
        .setComputeType(DataType_t::FLOAT)
        .setPaddingMode(fe::PaddingMode_t::NEG_INF_PAD)
        .setPrePadding(2, zeros.data())
        .setPostPadding(2, zeros.data())
        .setSpatialStride(2, window.data());
    expectBuildStatus(noWindow, CUDNN_STATUS_BAD_PARAM);

    fe::ResampleDescBuilder_v8 onlyModes;
    onlyModes.setResampleMode(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)
        .setComputeType(DataType_t::FLOAT)
        .setPaddingMode(CUDNN_ZERO_PAD);
    expectBuildStatus(onlyModes, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderParameters, ResampleZeroDenominatorIsBadParam)
{
    const std::array<cudnnFraction_t, 2> zeroDenominator = {{{2, 1}, {3, 0}}};
    struct Case
    {
        const char* label;
        std::function<void(fe::ResampleDescBuilder_v8&)> set;
    };
    const std::vector<Case> cases = {
        {"window",
         [&zeroDenominator](fe::ResampleDescBuilder_v8& b) {
             b.setSpatialDim(2, zeroDenominator.data());
         }},
        {"pre padding",
         [&zeroDenominator](fe::ResampleDescBuilder_v8& b) {
             b.setPrePadding(2, zeroDenominator.data());
         }},
        {"post padding",
         [&zeroDenominator](fe::ResampleDescBuilder_v8& b) {
             b.setPostPadding(2, zeroDenominator.data());
         }},
        {"stride",
         [&zeroDenominator](fe::ResampleDescBuilder_v8& b) {
             b.setSpatialStride(2, zeroDenominator.data());
         }},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        fe::ResampleDescBuilder_v8 builder;
        configureMaxPool(builder);
        c.set(builder);
        expectBuildStatus(builder, CUDNN_STATUS_BAD_PARAM);
    }
}

TEST_F(TestCudnnShimV8BuilderParameters, ResampleFractionalValueIsNotSupported)
{
    fe::ResampleDescBuilder_v8 fractional;
    configureFractionalPool(fractional, {{{2, 1}, {3, 2}}});
    expectBuildStatus(fractional, CUDNN_STATUS_NOT_SUPPORTED);

    fe::ResampleDescBuilder_v8 negative;
    configureFractionalPool(negative, {{{-4, -2}, {6, 3}}});
    const fe::ResampleDesc_v8 resample = negative.build();
    ASSERT_EQ(resample.get_status(), CUDNN_STATUS_SUCCESS);
    hipdnnBackendDescriptor_t op = lowerResample(resample, {1, 4, 4, 4});
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS), (Dims{2, 2}));
}

TEST_F(TestCudnnShimV8BuilderParameters, ResampleNearestAndBilinearAreNotSupported)
{
    for(const fe::ResampleMode_t mode : {fe::ResampleMode_t::NEAREST, fe::ResampleMode_t::BILINEAR})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        fe::ResampleDescBuilder_v8 builder;
        configureMaxPool(builder);
        builder.setResampleMode(mode);
        expectBuildStatus(builder, CUDNN_STATUS_NOT_SUPPORTED);
    }
    for(const cudnnResampleMode_t mode : {CUDNN_RESAMPLE_NEAREST, CUDNN_RESAMPLE_BILINEAR})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        fe::ResampleDescBuilder_v8 builder;
        configureMaxPool(builder);
        builder.setResampleMode(mode);
        expectBuildStatus(builder, CUDNN_STATUS_NOT_SUPPORTED);
    }
    for(const cudnnResampleMode_t mode : {CUDNN_RESAMPLE_AVGPOOL,
                                          CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING,
                                          CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING,
                                          CUDNN_RESAMPLE_MAXPOOL})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        fe::ResampleDescBuilder_v8 builder;
        configureMaxPool(builder);
        builder.setResampleMode(mode);
        const fe::ResampleDesc_v8 resample = builder.build();
        EXPECT_EQ(resample.get_status(), CUDNN_STATUS_SUCCESS);
    }
}

// EDGE_VAL_PAD is the one padding mode hipDNN cannot lower.
TEST_F(TestCudnnShimV8BuilderParameters, ResampleEdgeValuePaddingIsNotSupported)
{
    fe::ResampleDescBuilder_v8 cudnnPadding;
    configureMaxPool(cudnnPadding);
    cudnnPadding.setPaddingMode(CUDNN_EDGE_VAL_PAD);
    fe::ResampleDescBuilder_v8 fePadding;
    configureMaxPool(fePadding);
    fePadding.setPaddingMode(fe::PaddingMode_t::EDGE_VAL_PAD);
    for(fe::ResampleDescBuilder_v8* builder : {&cudnnPadding, &fePadding})
    {
        const BuildFailure failure = buildFailure(*builder);
        EXPECT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED) << failure.what;
        EXPECT_NE(failure.what.find("padding mode"), std::string::npos) << failure.what;
    }
}

TEST_F(TestCudnnShimV8BuilderParameters, ResampleCudnnAvgpoolLowersAsIncludePadding)
{
    fe::ResampleDescBuilder_v8 builder;
    configureMaxPool(builder);
    builder.setResampleMode(CUDNN_RESAMPLE_AVGPOOL).setPaddingMode(fe::PaddingMode_t::ZERO_PAD);
    const fe::ResampleDesc_v8 resample = builder.build();

    ASSERT_EQ(resample.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(resample.getMode(), fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING);
    // H: (8 + 1 + 0 - 3) / 2 + 1 = 4; W: (8 + 0 + 1 - 3) / 2 + 1 = 4.
    hipdnnBackendDescriptor_t op = lowerResample(resample, {1, 4, 4, 4});
    EXPECT_EQ(scalarOn<hipdnnResampleMode_t>(op, HIPDNN_ATTR_RESAMPLE_MODE),
              HIPDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING);
    EXPECT_EQ(scalarOn<hipdnnPaddingMode_t>(op, HIPDNN_ATTR_RESAMPLE_PADDING_MODE),
              HIPDNN_PADDING_ZERO_PAD);
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS), (Dims{1, 0}));
    EXPECT_EQ(int64s(op, HIPDNN_ATTR_RESAMPLE_POST_PADDINGS), (Dims{0, 1}));
}

// ---- RngDesc ----

TEST_F(TestCudnnShimV8BuilderParameters, RngRequiresDistribution)
{
    fe::RngDescBuilder unset;
    unset.setBernoulliDistProbability(0.5);
    expectBuildStatus(unset, CUDNN_STATUS_INVALID_VALUE);
}

TEST_F(TestCudnnShimV8BuilderParameters, RngGettersReturnSetValues)
{
    const fe::RngDesc fromPairs = fe::RngDescBuilder()
                                      .setRngDistribution(CUDNN_RNG_DISTRIBUTION_UNIFORM)
                                      .setNormalDistParams(1.5, 2.5)
                                      .setUniformDistParams(7.0, -3.0)
                                      .setBernoulliDistProbability(0.25)
                                      .build();
    ASSERT_EQ(fromPairs.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(fromPairs.getDistribution(), fe::RngDistribution_t::UNIFORM);
    EXPECT_EQ(fromPairs.getNormalDistMean(), 1.5);
    EXPECT_EQ(fromPairs.getNormalDistStdDev(), 2.5);
    EXPECT_EQ(fromPairs.getUniformDistMax(), 7.0);
    EXPECT_EQ(fromPairs.getUniformDistMin(), -3.0);
    EXPECT_EQ(fromPairs.getBernoulliDistProbability(), 0.25);

    const fe::RngDesc fromSingles = fe::RngDescBuilder()
                                        .setRngDistribution(fe::RngDistribution_t::NORMAL)
                                        .setNormalDistMean(4.0)
                                        .setNormalDistStdDev(5.0)
                                        .setUniformDistMax(6.0)
                                        .setUniformDistMin(-6.0)
                                        .setBernoulliDistProbability(0.75)
                                        .build();
    ASSERT_EQ(fromSingles.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(fromSingles.getDistribution(), fe::RngDistribution_t::NORMAL);
    EXPECT_EQ(fromSingles.getNormalDistMean(), 4.0);
    EXPECT_EQ(fromSingles.getNormalDistStdDev(), 5.0);
    EXPECT_EQ(fromSingles.getUniformDistMax(), 6.0);
    EXPECT_EQ(fromSingles.getUniformDistMin(), -6.0);
    EXPECT_EQ(fromSingles.getBernoulliDistProbability(), 0.75);

    const fe::RngDesc bernoulli = fe::RngDescBuilder()
                                      .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                      .setBernoulliDistProbability(0.25)
                                      .setNormalDistMean(-1.5)
                                      .build();
    ASSERT_EQ(bernoulli.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(bernoulli.getDistribution(), fe::RngDistribution_t::BERNOULLI);
    EXPECT_EQ(bernoulli.getBernoulliDistProbability(), 0.25);
    EXPECT_EQ(bernoulli.getNormalDistMean(), -1.5);
}

// ---- VariantPack ----

TEST_F(TestCudnnShimV8BuilderParameters, VariantPackForwardsToHipdnn)
{
    int x = 0;
    int y = 0;
    int workspace = 0;
    std::array<void*, 2> pointers = {&x, &y};
    const std::array<std::int64_t, 2> uids = {11, 22};

    fe::VariantPackBuilder builder;
    builder.setDataPointers(2, pointers.data())
        .setUids(2, uids.data())
        .setWorkspacePointer(&workspace);
    const fe::VariantPack pack = builder.build();
    ASSERT_EQ(pack.get_status(), CUDNN_STATUS_SUCCESS);

    const std::vector<hipdnnBackendDescriptor_t> packs
        = created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    ASSERT_EQ(packs.size(), 1U);
    const auto dataPointers = callOn(packs[0], HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
    const auto uniqueIds = callOn(packs[0], HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
    const auto workspacePointer = callOn(packs[0], HIPDNN_ATTR_VARIANT_PACK_WORKSPACE);
    ASSERT_TRUE(dataPointers && uniqueIds && workspacePointer);
    EXPECT_EQ(dataPointers->values<void*>(), (std::vector<void*>{&x, &y}));
    EXPECT_EQ(uniqueIds->values<std::int64_t>(), (std::vector<std::int64_t>{11, 22}));
    EXPECT_EQ(workspacePointer->value<void*>(), static_cast<void*>(&workspace));
    const std::vector<hipdnnBackendDescriptor_t> done = finalized();
    EXPECT_NE(std::find(done.begin(), done.end(), packs[0]), done.end());
}

TEST_F(TestCudnnShimV8BuilderParameters, VariantPackFromUidPointerSet)
{
    int a = 0;
    int b = 0;
    int c = 0;
    const std::set<std::pair<std::uint64_t, void*>> uidPointers = {{30, &c}, {10, &a}, {20, &b}};

    fe::VariantPackBuilder builder;
    builder.setDataPointers(uidPointers);
    const fe::VariantPack pack = builder.build();
    ASSERT_EQ(pack.get_status(), CUDNN_STATUS_SUCCESS);
    EXPECT_NE(pack.describe().find("has 3 data pointers"), std::string::npos) << pack.describe();

    const std::vector<hipdnnBackendDescriptor_t> packs
        = created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    ASSERT_EQ(packs.size(), 1U);
    EXPECT_EQ(attributeOn(packs[0], HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS).values<void*>(),
              (std::vector<void*>{&a, &b, &c}));
    EXPECT_EQ(attributeOn(packs[0], HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS).values<std::int64_t>(),
              (std::vector<std::int64_t>{10, 20, 30}));
    EXPECT_EQ(attributeOn(packs[0], HIPDNN_ATTR_VARIANT_PACK_WORKSPACE).value<void*>(), nullptr);
}

TEST_F(TestCudnnShimV8BuilderParameters, VariantPackUidCountMismatchIsBadParam)
{
    int x = 0;
    int y = 0;
    std::array<void*, 2> pointers = {&x, &y};
    const std::array<std::int64_t, 1> uids = {11};

    fe::VariantPackBuilder builder;
    builder.setDataPointers(2, pointers.data()).setUids(1, uids.data());

    expectBuildStatus(builder, CUDNN_STATUS_BAD_PARAM);
    EXPECT_FALSE(variantPackCreatedOnHipdnn());
}

TEST_F(TestCudnnShimV8BuilderParameters, EmptyVariantPackIsBadParam)
{
    fe::VariantPackBuilder builder;

    expectBuildStatus(builder, CUDNN_STATUS_BAD_PARAM);
    EXPECT_FALSE(variantPackCreatedOnHipdnn());
}

} // namespace
