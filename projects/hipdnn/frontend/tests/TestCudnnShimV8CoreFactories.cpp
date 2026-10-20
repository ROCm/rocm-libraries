// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v8 record factories called directly, without the C API: every
// validation and normalization rule, check order, and the caller's labels in
// the messages.
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

#include <gtest/gtest.h>

#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace v8 = hipdnn_frontend::compatibility::cudnn_frontend::detail::v8;
using hipdnn_frontend::compatibility::cudnn_frontend::ConvolutionMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::DataType_t;
using hipdnn_frontend::compatibility::cudnn_frontend::NormFwdPhase_t;
using hipdnn_frontend::compatibility::cudnn_frontend::PaddingMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::PointwiseMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::ReductionMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::ResampleMode_t;
using hipdnn_frontend::compatibility::cudnn_frontend::RngDistribution_t;
using v8::OperationTarget;
using v8::PointwiseModeClass;

constexpr v8::TensorLabels TENSOR_LABELS{"dataType",
                                         "dimensions",
                                         "strides",
                                         "uniqueId",
                                         "alignment",
                                         "vectorCount",
                                         "reordering",
                                         "constant"};
constexpr v8::ConvolutionLabels CONV_LABELS{"computeType",
                                            "mode",
                                            "spatialDims",
                                            "prePaddings",
                                            "postPaddings",
                                            "dilations",
                                            "filterStrides"};
constexpr v8::PointwiseLabels POINTWISE_LABELS{
    "mode", "computeType", "slope", "eluAlpha", "softplusBeta", "swishBeta"};
constexpr v8::ResampleLabels RESAMPLE_LABELS{"mode",
                                             "computeType",
                                             "padding",
                                             "spatialDims",
                                             "window",
                                             "prePaddings",
                                             "postPaddings",
                                             "strides"};
constexpr v8::ConvolutionOperationLabels CONV_OP_LABELS{
    "convolution", "alpha", "beta", {"x", "w", "y"}, "spatialDims"};
constexpr v8::PointwiseOperationLabels POINTWISE_OP_LABELS{
    "pointwise", "x", "b", "t", "y", "dy", "dx", "alpha1", "alpha2"};
constexpr v8::ResampleBackwardOperationLabels RESAMPLE_BWD_LABELS{
    "resample", "dx", "dy", "index", "alpha", "beta", "x", "y"};
constexpr v8::NormForwardOperationLabels NORM_FWD_LABELS{"mode",
                                                         "phase",
                                                         "x",
                                                         "scale",
                                                         "bias",
                                                         "epsilon",
                                                         "y",
                                                         "mean",
                                                         "invVariance",
                                                         "inputRunningMean",
                                                         "inputRunningVariance",
                                                         "expAvgFactor",
                                                         "outputRunningMean",
                                                         "outputRunningVariance",
                                                         "peerStats"};
constexpr v8::NormBackwardOperationLabels NORM_BWD_LABELS{"mode",
                                                          "x",
                                                          "dy",
                                                          "scale",
                                                          "mean",
                                                          "invVariance",
                                                          "epsilon",
                                                          "dx",
                                                          "dscale",
                                                          "dbias",
                                                          "peerStats"};

template <typename Fn>
void expectShimError(Fn&& fn, cudnnStatus_t status, const std::string& message)
{
    try
    {
        std::forward<Fn>(fn)();
        ADD_FAILURE() << "expected ShimError: " << message;
    }
    catch(const v8::ShimError& error)
    {
        EXPECT_EQ(error.status(), status);
        EXPECT_EQ(std::string(error.what()), message);
    }
}

v8::TensorInputs tensorInputs(std::vector<std::int64_t> dims = {1, 4, 8, 8},
                              std::int64_t uniqueId = 1)
{
    v8::TensorInputs inputs;
    inputs.dataType = CUDNN_DATA_FLOAT;
    inputs.strides.assign(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        inputs.strides[i - 1] = inputs.strides[i] * dims[i];
    }
    inputs.dimensions = std::move(dims);
    inputs.uniqueId = uniqueId;
    inputs.alignment = 16;
    return inputs;
}

v8::TensorPtr tensor(std::vector<std::int64_t> dims = {1, 4, 8, 8}, std::int64_t uniqueId = 1)
{
    return v8::makeTensorRecord(tensorInputs(std::move(dims), uniqueId), TENSOR_LABELS);
}

std::shared_ptr<const v8::PointwiseRecord> pointwise(cudnnPointwiseMode_t mode)
{
    v8::PointwiseInputs inputs;
    inputs.mode = mode;
    inputs.computeType = CUDNN_DATA_FLOAT;
    return v8::makePointwiseRecord(inputs, POINTWISE_LABELS);
}

std::shared_ptr<const v8::ConvolutionRecord> convolution()
{
    v8::ConvolutionInputs inputs;
    inputs.computeType = CUDNN_DATA_FLOAT;
    inputs.mode = CUDNN_CROSS_CORRELATION;
    inputs.spatialDims = 2;
    inputs.prePaddings = {1, 2};
    inputs.postPaddings = {3, 4};
    inputs.dilations = {1, 1};
    inputs.filterStrides = {2, 2};
    return v8::makeConvRecord(std::move(inputs), CONV_LABELS);
}

v8::ResampleInputs resampleInputs(cudnnResampleMode_t mode = CUDNN_RESAMPLE_MAXPOOL)
{
    v8::ResampleInputs inputs;
    inputs.mode = mode;
    inputs.computeType = CUDNN_DATA_FLOAT;
    inputs.paddingMode = CUDNN_NEG_INF_PAD;
    inputs.spatialDims = 2;
    inputs.windowDims = {{4, 2}, {3, 1}};
    inputs.prePaddings = {{0, 1}, {0, 1}};
    inputs.postPaddings = {{0, 1}, {0, 1}};
    inputs.strides = {{2, 1}, {-4, -2}};
    return inputs;
}

v8::NormForwardOperationInputs normForward(cudnnBackendNormMode_t mode,
                                           cudnnBackendNormFwdPhase_t phase)
{
    v8::NormForwardOperationInputs inputs;
    inputs.mode = mode;
    inputs.phase = phase;
    inputs.x = tensor({2, 8, 3, 3}, 1);
    inputs.scale = tensor({1, 8, 3, 3}, 2);
    inputs.bias = tensor({1, 8, 3, 3}, 3);
    inputs.epsilon = tensor({1, 1, 1, 1}, 4);
    inputs.y = tensor({2, 8, 3, 3}, 5);
    return inputs;
}

v8::NormBackwardOperationInputs normBackward(cudnnBackendNormMode_t mode)
{
    v8::NormBackwardOperationInputs inputs;
    inputs.mode = mode;
    inputs.x = tensor({2, 8, 3, 3}, 1);
    inputs.dy = tensor({2, 8, 3, 3}, 2);
    inputs.scale = tensor({1, 8, 1, 1}, 3);
    inputs.mean = tensor({1, 8, 1, 1}, 4);
    inputs.invVariance = tensor({1, 8, 1, 1}, 5);
    inputs.dx = tensor({2, 8, 3, 3}, 6);
    inputs.dscale = tensor({1, 8, 1, 1}, 7);
    inputs.dbias = tensor({1, 8, 1, 1}, 8);
    return inputs;
}

TEST(TestCudnnShimV8CoreFactories, TensorRecordKeepsLoweringFacts)
{
    v8::TensorInputs inputs = tensorInputs({2, 3, 4, 5}, 7);
    inputs.alignment = 4;
    inputs.isVirtual = true;
    const v8::TensorPtr record = v8::makeTensorRecord(std::move(inputs), TENSOR_LABELS);

    EXPECT_EQ(record->kind(), CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    EXPECT_EQ(record->dataType(), DataType_t::FLOAT);
    EXPECT_EQ(record->dimensions(), (std::vector<std::int64_t>{2, 3, 4, 5}));
    EXPECT_EQ(record->strides(), (std::vector<std::int64_t>{60, 20, 5, 1}));
    EXPECT_EQ(record->uniqueId(), 7);
    EXPECT_EQ(record->alignment(), 4);
    EXPECT_EQ(record->forwardedAlignment(), std::optional<std::int64_t>(4));
    EXPECT_TRUE(record->isVirtual());
    EXPECT_FALSE(record->isRuntimePassByValue());
    EXPECT_FALSE(record->constantValue().has_value());
    EXPECT_EQ(record->raggedOffset(), nullptr);

    EXPECT_FALSE(tensor()->forwardedAlignment().has_value());
}

TEST(TestCudnnShimV8CoreFactories, TensorChecksRunInOrderWithCallerLabels)
{
    const auto expectTensorError
        = [](v8::TensorInputs inputs, cudnnStatus_t status, const std::string& message) {
              expectShimError(
                  [&] { v8::makeTensorRecord(std::move(inputs), TENSOR_LABELS); }, status, message);
          };

    v8::TensorInputs inputs = tensorInputs();
    inputs.dataType.reset();
    inputs.dimensions.clear();
    expectTensorError(inputs, CUDNN_STATUS_BAD_PARAM, "dataType is required");

    inputs = tensorInputs();
    inputs.dimensions.clear();
    inputs.uniqueId = -1;
    expectTensorError(inputs, CUDNN_STATUS_BAD_PARAM, "dimensions is required (non-empty)");

    inputs = tensorInputs({4, 4});
    inputs.strides = {1};
    expectTensorError(
        inputs, CUDNN_STATUS_BAD_PARAM, "dimensions has 2 elements but strides has 1");

    inputs = tensorInputs({4, 0});
    expectTensorError(inputs,
                      CUDNN_STATUS_BAD_PARAM,
                      "tensor dimension 1 is 0 with stride 1; dimensions must be positive and "
                      "strides non-negative");

    inputs = tensorInputs();
    inputs.uniqueId.reset();
    expectTensorError(inputs, CUDNN_STATUS_BAD_PARAM, "uniqueId is required");

    inputs = tensorInputs();
    inputs.uniqueId = -1;
    inputs.dataType = CUDNN_DATA_INT8x32;
    expectTensorError(inputs, CUDNN_STATUS_BAD_PARAM, "uniqueId must be >= 0");

    inputs = tensorInputs();
    inputs.alignment = 0;
    expectTensorError(inputs, CUDNN_STATUS_BAD_PARAM, "alignment must be > 0");

    inputs = tensorInputs();
    inputs.dataType = CUDNN_DATA_INT8x32;
    inputs.vectorCount = 4;
    expectTensorError(inputs,
                      CUDNN_STATUS_NOT_SUPPORTED,
                      "dataType value " + std::to_string(static_cast<int>(CUDNN_DATA_INT8x32))
                          + " is not supported by the hipDNN cuDNN shim");

    inputs = tensorInputs();
    inputs.vectorCount = 4;
    inputs.reorderingMode = CUDNN_TENSOR_REORDERING_INT8x32;
    expectTensorError(
        inputs, CUDNN_STATUS_NOT_SUPPORTED, "vectorCount other than 1 is not supported");

    inputs = tensorInputs();
    inputs.reorderingMode = CUDNN_TENSOR_REORDERING_INT8x32;
    expectTensorError(
        inputs, CUDNN_STATUS_NOT_SUPPORTED, "reordering other than NONE is not supported");

    inputs = tensorInputs();
    inputs.constantValue = static_cast<const void*>(nullptr);
    expectTensorError(inputs, CUDNN_STATUS_BAD_PARAM, "constant is null");
}

TEST(TestCudnnShimV8CoreFactories, TensorConstantCopiedAndWinsOverByValue)
{
    v8::TensorPtr record;
    {
        const double value = 2.5;
        v8::TensorInputs inputs = tensorInputs({1});
        inputs.dataType = CUDNN_DATA_DOUBLE;
        inputs.isByValue = true;
        inputs.constantValue = static_cast<const void*>(&value);
        record = v8::makeTensorRecord(std::move(inputs), TENSOR_LABELS);
    }
    ASSERT_TRUE(record->constantValue().has_value());
    ASSERT_EQ(record->constantValue()->size(), sizeof(double));
    double copied = 0.0;
    std::memcpy(&copied, record->constantValue()->data(), sizeof(copied));
    EXPECT_EQ(copied, 2.5);
    EXPECT_FALSE(record->isRuntimePassByValue());

    const bool flag = true;
    v8::TensorInputs boolean = tensorInputs({1});
    boolean.dataType = CUDNN_DATA_BOOLEAN;
    boolean.constantValue = static_cast<const void*>(&flag);
    EXPECT_EQ(v8::makeTensorRecord(std::move(boolean), TENSOR_LABELS)->constantValue()->size(), 1U);

    v8::TensorInputs runtime = tensorInputs({1});
    runtime.isByValue = true;
    EXPECT_TRUE(v8::makeTensorRecord(std::move(runtime), TENSOR_LABELS)->isRuntimePassByValue());
}

TEST(TestCudnnShimV8CoreFactories, TensorRaggedOffsetKept)
{
    const v8::TensorPtr offsets = tensor({3}, 9);
    v8::TensorInputs inputs = tensorInputs();
    inputs.raggedOffset = offsets;
    EXPECT_EQ(v8::makeTensorRecord(std::move(inputs), TENSOR_LABELS)->raggedOffset(), offsets);
}

TEST(TestCudnnShimV8CoreFactories, ConvRecordValidatesSpatialArrays)
{
    const std::shared_ptr<const v8::ConvolutionRecord> record = convolution();
    EXPECT_EQ(record->kind(), CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    EXPECT_EQ(record->computeType(), DataType_t::FLOAT);
    EXPECT_EQ(record->mode(), ConvolutionMode_t::CROSS_CORRELATION);
    EXPECT_EQ(record->spatialDims(), 2);
    EXPECT_EQ(record->prePaddings(), (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(record->postPaddings(), (std::vector<std::int64_t>{3, 4}));
    EXPECT_EQ(record->filterStrides(), (std::vector<std::int64_t>{2, 2}));

    const auto base = [] {
        v8::ConvolutionInputs inputs;
        inputs.computeType = CUDNN_DATA_FLOAT;
        inputs.mode = CUDNN_CROSS_CORRELATION;
        inputs.spatialDims = 2;
        inputs.prePaddings = {0, 0};
        inputs.postPaddings = {0, 0};
        inputs.dilations = {1, 1};
        inputs.filterStrides = {1, 1};
        return inputs;
    };
    const auto expectConvError
        = [](v8::ConvolutionInputs inputs, cudnnStatus_t status, const std::string& message) {
              expectShimError(
                  [&] { v8::makeConvRecord(std::move(inputs), CONV_LABELS); }, status, message);
          };

    v8::ConvolutionInputs inputs = base();
    inputs.mode.reset();
    inputs.spatialDims = 0;
    expectConvError(inputs, CUDNN_STATUS_BAD_PARAM, "mode is required");

    inputs = base();
    inputs.spatialDims = 0;
    expectConvError(inputs, CUDNN_STATUS_BAD_PARAM, "spatialDims must be >= 1");

    inputs = base();
    inputs.dilations = {1};
    expectConvError(inputs,
                    CUDNN_STATUS_BAD_PARAM,
                    "dilations has 1 elements, expected the spatial dimension count 2");

    inputs = base();
    inputs.filterStrides = {0, 1};
    inputs.computeType = CUDNN_DATA_INT8x32;
    expectConvError(inputs, CUDNN_STATUS_BAD_PARAM, "filterStrides[0] must be > 0");

    inputs = base();
    inputs.computeType = CUDNN_DATA_INT8x32;
    expectConvError(inputs,
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "computeType value " + std::to_string(static_cast<int>(CUDNN_DATA_INT8x32))
                        + " is not supported by the hipDNN cuDNN shim");
}

TEST(TestCudnnShimV8CoreFactories, PointwiseReluLowerClipKeptUnlessSlopeSet)
{
    v8::PointwiseInputs inputs;
    inputs.mode = CUDNN_POINTWISE_RELU_FWD;
    inputs.computeType = CUDNN_DATA_FLOAT;
    inputs.reluLowerClip = 0.0;
    inputs.reluUpperClip = DBL_MAX;
    inputs.reluLowerClipSlope = 0.0;
    std::shared_ptr<const v8::PointwiseRecord> record
        = v8::makePointwiseRecord(inputs, POINTWISE_LABELS);
    EXPECT_EQ(record->mode(), PointwiseMode_t::RELU_FWD);
    EXPECT_EQ(record->reluLowerClip(), std::optional<double>(0.0));
    EXPECT_FALSE(record->reluUpperClip().has_value());
    EXPECT_FALSE(record->reluLowerClipSlope().has_value());

    inputs.reluLowerClipSlope = 0.01;
    record = v8::makePointwiseRecord(inputs, POINTWISE_LABELS);
    EXPECT_FALSE(record->reluLowerClip().has_value());
    EXPECT_EQ(record->reluLowerClipSlope(), std::optional<double>(0.01));

    inputs.reluLowerClip = -1.0;
    EXPECT_EQ(v8::makePointwiseRecord(inputs, POINTWISE_LABELS)->reluLowerClip(),
              std::optional<double>(-1.0));
}

TEST(TestCudnnShimV8CoreFactories, PointwiseDefaultsAndAxisElided)
{
    v8::PointwiseInputs inputs;
    inputs.mode = CUDNN_POINTWISE_ADD;
    inputs.computeType = CUDNN_DATA_FLOAT;
    inputs.reluUpperClip = static_cast<double>(std::numeric_limits<float>::max());
    inputs.eluAlpha = 1.0;
    inputs.softplusBeta = 1.0;
    inputs.swishBeta = 1.0;
    inputs.axis = -1;
    std::shared_ptr<const v8::PointwiseRecord> record
        = v8::makePointwiseRecord(inputs, POINTWISE_LABELS);
    EXPECT_FALSE(record->reluLowerClip().has_value());
    EXPECT_FALSE(record->reluUpperClip().has_value());
    EXPECT_FALSE(record->eluAlpha().has_value());
    EXPECT_FALSE(record->softplusBeta().has_value());
    EXPECT_FALSE(record->swishBeta().has_value());
    EXPECT_FALSE(record->axis().has_value());

    inputs.reluUpperClip = 6.0;
    inputs.eluAlpha = 0.5;
    inputs.softplusBeta = 2.0;
    inputs.swishBeta = 3.0;
    inputs.axis = 2;
    record = v8::makePointwiseRecord(inputs, POINTWISE_LABELS);
    EXPECT_EQ(record->reluUpperClip(), std::optional<double>(6.0));
    EXPECT_EQ(record->eluAlpha(), std::optional<double>(0.5));
    EXPECT_EQ(record->softplusBeta(), std::optional<double>(2.0));
    EXPECT_EQ(record->swishBeta(), std::optional<double>(3.0));
    EXPECT_EQ(record->axis(), std::optional<std::int64_t>(2));
}

TEST(TestCudnnShimV8CoreFactories, PointwiseRejectsMissingAndNonFinite)
{
    v8::PointwiseInputs inputs;
    inputs.computeType = CUDNN_DATA_FLOAT;
    expectShimError([&] { v8::makePointwiseRecord(inputs, POINTWISE_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "mode is required");

    inputs.mode = CUDNN_POINTWISE_ELU_FWD;
    inputs.eluAlpha = std::nan("");
    expectShimError([&] { v8::makePointwiseRecord(inputs, POINTWISE_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "eluAlpha must be finite");
}

TEST(TestCudnnShimV8CoreFactories, MatmulReductionAndRngRecords)
{
    expectShimError([] { v8::makeMatmulRecord({std::nullopt, true}, {"computeType", "padding"}); },
                    CUDNN_STATUS_BAD_PARAM,
                    "computeType is required");
    expectShimError(
        [] { v8::makeMatmulRecord({CUDNN_DATA_FLOAT, true}, {"computeType", "padding"}); },
        CUDNN_STATUS_NOT_SUPPORTED,
        "padding is not supported");
    EXPECT_EQ(
        v8::makeMatmulRecord({CUDNN_DATA_FLOAT, false}, {"computeType", "padding"})->computeType(),
        DataType_t::FLOAT);

    const std::shared_ptr<const v8::ReductionRecord> reduction = v8::makeReductionRecord(
        {CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, true}, {"operator", "computeType"});
    EXPECT_EQ(reduction->mode(), ReductionMode_t::MAX);
    EXPECT_TRUE(reduction->isDeterministic());

    expectShimError([] { v8::makeRngRecord(std::nullopt, "distribution"); },
                    CUDNN_STATUS_BAD_PARAM,
                    "distribution is required");
    EXPECT_EQ(v8::makeRngRecord(CUDNN_RNG_DISTRIBUTION_BERNOULLI, "distribution")->distribution(),
              RngDistribution_t::BERNOULLI);
}

TEST(TestCudnnShimV8CoreFactories, ResampleFractionsMustBeIntegral)
{
    const std::shared_ptr<const v8::ResampleRecord> record
        = v8::makeResampleRecord(resampleInputs(), RESAMPLE_LABELS);
    EXPECT_EQ(record->mode(), ResampleMode_t::MAXPOOL);
    EXPECT_EQ(record->paddingMode(), PaddingMode_t::NEG_INF_PAD);
    EXPECT_EQ(record->windowDims(), (std::vector<std::int64_t>{2, 3}));
    EXPECT_EQ(record->strides(), (std::vector<std::int64_t>{2, 2}));

    v8::ResampleInputs inputs = resampleInputs();
    inputs.windowDims = {{2, 1}, {3, 2}};
    expectShimError([&] { v8::makeResampleRecord(inputs, RESAMPLE_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "window[1] = 3/2 is not an integer; fractional values are not supported");

    inputs = resampleInputs();
    inputs.strides = {{std::numeric_limits<std::int64_t>::min(), -1}, {1, 1}};
    expectShimError([&] { v8::makeResampleRecord(inputs, RESAMPLE_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "strides[0] = " + std::to_string(std::numeric_limits<std::int64_t>::min())
                        + "/-1 is not an integer; fractional values are not supported");

    inputs = resampleInputs();
    inputs.paddingMode.reset();
    inputs.spatialDims = 0;
    expectShimError([&] { v8::makeResampleRecord(inputs, RESAMPLE_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "padding is required");

    inputs = resampleInputs(CUDNN_RESAMPLE_NEAREST);
    inputs.windowDims = {{3, 2}, {1, 1}};
    expectShimError([&] { v8::makeResampleRecord(inputs, RESAMPLE_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "mode value " + std::to_string(static_cast<int>(CUDNN_RESAMPLE_NEAREST))
                        + " is not supported by the hipDNN cuDNN shim");
}

TEST(TestCudnnShimV8CoreFactories, ConvolutionOperationRankAndScaling)
{
    const std::array<v8::TensorPtr, 3> tensors{
        tensor({1, 4, 8, 8}, 1), tensor({4, 4, 3, 3}, 2), tensor({1, 4, 8, 8}, 3)};
    const std::shared_ptr<const v8::ConvolutionBackwardFilterOperationRecord> wgrad
        = v8::makeOperationRecord<v8::ConvolutionBackwardFilterOperationRecord>(
            {convolution(), tensors, 1.0, 0.0}, CONV_OP_LABELS);
    EXPECT_EQ(wgrad->kind(), CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
    EXPECT_EQ(wgrad->target(), OperationTarget::CONVOLUTION_WGRAD);
    EXPECT_EQ(wgrad->dy(), tensors[0]);
    EXPECT_EQ(wgrad->x(), tensors[1]);
    EXPECT_EQ(wgrad->dw(), tensors[2]);

    expectShimError(
        [&] {
            v8::makeOperationRecord<v8::ConvolutionForwardOperationRecord>(
                {nullptr, tensors, std::nullopt, std::nullopt}, CONV_OP_LABELS);
        },
        CUDNN_STATUS_BAD_PARAM,
        "convolution is required");
    expectShimError(
        [&] {
            v8::makeOperationRecord<v8::ConvolutionForwardOperationRecord>(
                {convolution(), {tensors[0], tensors[1], nullptr}, 0.5, std::nullopt},
                CONV_OP_LABELS);
        },
        CUDNN_STATUS_BAD_PARAM,
        "y is required");
    expectShimError(
        [&] {
            v8::makeOperationRecord<v8::ConvolutionForwardOperationRecord>(
                {convolution(), {tensors[0], tensor({4, 3, 3}, 2), tensors[2]}, 0.5, std::nullopt},
                CONV_OP_LABELS);
        },
        CUDNN_STATUS_BAD_PARAM,
        "w has rank 3, expected spatialDims + 2 = 4");
    expectShimError(
        [&] {
            v8::makeOperationRecord<v8::ConvolutionForwardOperationRecord>(
                {convolution(), tensors, 0.5, std::nullopt}, CONV_OP_LABELS);
        },
        CUDNN_STATUS_NOT_SUPPORTED,
        "alpha = 0.5 is not supported by the hipDNN cuDNN shim; only 1 is");
    expectShimError(
        [&] {
            v8::makeOperationRecord<v8::ConvolutionForwardOperationRecord>(
                {convolution(), tensors, std::nullopt, 1.0}, CONV_OP_LABELS);
        },
        CUDNN_STATUS_NOT_SUPPORTED,
        "beta = 1 is not supported by the hipDNN cuDNN shim; only 0 is");
}

TEST(TestCudnnShimV8CoreFactories, PointwiseOperationSlotMap)
{
    const v8::TensorPtr x = tensor({4}, 1);
    const v8::TensorPtr b = tensor({4}, 2);
    const v8::TensorPtr t = tensor({4}, 3);
    const v8::TensorPtr y = tensor({4}, 4);

    v8::PointwiseOperationInputs forward;
    forward.pointwise = pointwise(CUDNN_POINTWISE_ADD);
    forward.x = x;
    forward.b = b;
    forward.y = y;
    const std::shared_ptr<const v8::PointwiseOperationRecord> binary
        = v8::makeOperationRecord(forward, POINTWISE_OP_LABELS);
    EXPECT_EQ(binary->target(), OperationTarget::POINTWISE);
    EXPECT_EQ(binary->modeClass(), PointwiseModeClass::MATH);
    EXPECT_EQ(binary->portCount(), 3);
    EXPECT_EQ(binary->in0(), x);
    EXPECT_EQ(binary->in1(), b);
    EXPECT_EQ(binary->in2(), nullptr);
    EXPECT_EQ(binary->out0(), y);

    forward.pointwise = pointwise(CUDNN_POINTWISE_BINARY_SELECT);
    forward.t = t;
    const std::shared_ptr<const v8::PointwiseOperationRecord> select
        = v8::makeOperationRecord(forward, POINTWISE_OP_LABELS);
    EXPECT_EQ(select->portCount(), 4);
    EXPECT_EQ(select->in2(), t);

    v8::PointwiseOperationInputs backward;
    backward.pointwise = pointwise(CUDNN_POINTWISE_RELU_BWD);
    backward.x = x;
    backward.dy = b;
    backward.dx = y;
    const std::shared_ptr<const v8::PointwiseOperationRecord> relu
        = v8::makeOperationRecord(backward, POINTWISE_OP_LABELS);
    EXPECT_EQ(relu->modeClass(), PointwiseModeClass::ACTIVATION_BACKWARD);
    EXPECT_EQ(relu->in0(), b);
    EXPECT_EQ(relu->in1(), x);
    EXPECT_EQ(relu->out0(), y);
}

TEST(TestCudnnShimV8CoreFactories, PointwiseOperationRejectsUnusedSlotsBeforeScaling)
{
    v8::PointwiseOperationInputs unary;
    unary.pointwise = pointwise(CUDNN_POINTWISE_SQRT);
    unary.x = tensor({4}, 1);
    unary.y = tensor({4}, 2);
    unary.b = tensor({4}, 3);
    unary.alpha1 = 2.0;
    expectShimError([&] { v8::makeOperationRecord(unary, POINTWISE_OP_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "b is not used by a pointwise math mode with 2 ports");

    unary.b = nullptr;
    unary.alpha1.reset();
    unary.alpha2 = 2.0;
    expectShimError([&] { v8::makeOperationRecord(unary, POINTWISE_OP_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "alpha2 = 2 is not supported by the hipDNN cuDNN shim; only 1 is");

    v8::PointwiseOperationInputs backward;
    backward.pointwise = pointwise(CUDNN_POINTWISE_RELU_BWD);
    backward.x = tensor({4}, 1);
    backward.dy = tensor({4}, 2);
    expectShimError([&] { v8::makeOperationRecord(backward, POINTWISE_OP_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "dx is required");
    backward.dx = tensor({4}, 3);
    backward.y = tensor({4}, 4);
    expectShimError([&] { v8::makeOperationRecord(backward, POINTWISE_OP_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "y is not used by a pointwise activation-backward mode with 3 ports");
}

TEST(TestCudnnShimV8CoreFactories, MatmulAndResampleOperationRules)
{
    const v8::MatmulOperationLabels matmulLabels{"matmul", "a", "b", "c", "m", "n", "k"};
    v8::MatmulOperationInputs matmul;
    matmul.matmul = v8::makeMatmulRecord({CUDNN_DATA_FLOAT, false}, {"computeType", "padding"});
    matmul.a = tensor({1, 4, 8}, 1);
    matmul.b = tensor({1, 8, 4}, 2);
    matmul.c = tensor({1, 4, 4}, 3);
    EXPECT_EQ(v8::makeOperationRecord(matmul, matmulLabels)->target(), OperationTarget::MATMUL);
    matmul.hasNOverride = true;
    expectShimError([&] { v8::makeOperationRecord(matmul, matmulLabels); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "n is not supported by the hipDNN cuDNN shim");

    v8::ResampleBackwardOperationInputs maxpool;
    maxpool.resample = v8::makeResampleRecord(resampleInputs(), RESAMPLE_LABELS);
    maxpool.dx = tensor({1, 4, 8, 8}, 1);
    maxpool.dy = tensor({1, 4, 4, 4}, 2);
    maxpool.hasX = true;
    expectShimError([&] { v8::makeOperationRecord(maxpool, RESAMPLE_BWD_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "index is required for CUDNN_RESAMPLE_MAXPOOL");
    maxpool.index = tensor({1, 4, 4, 4}, 3);
    expectShimError([&] { v8::makeOperationRecord(maxpool, RESAMPLE_BWD_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "x is not supported by the hipDNN cuDNN shim for resample backward");
    maxpool.hasX = false;
    EXPECT_EQ(v8::makeOperationRecord(maxpool, RESAMPLE_BWD_LABELS)->index(), maxpool.index);

    v8::ResampleBackwardOperationInputs average = maxpool;
    average.resample = v8::makeResampleRecord(
        resampleInputs(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING), RESAMPLE_LABELS);
    average.index = nullptr;
    EXPECT_EQ(v8::makeOperationRecord(average, RESAMPLE_BWD_LABELS)->index(), nullptr);
}

TEST(TestCudnnShimV8CoreFactories, NormForwardTargetsAndKeptSlots)
{
    v8::NormForwardOperationInputs inference
        = normForward(CUDNN_BATCH_NORM, CUDNN_NORM_FWD_INFERENCE);
    inference.mean = tensor({1, 8, 1, 1}, 6);
    inference.invVariance = tensor({1, 8, 1, 1}, 7);
    const std::shared_ptr<const v8::NormForwardOperationRecord> batch
        = v8::makeOperationRecord(inference, NORM_FWD_LABELS);
    EXPECT_EQ(batch->target(), OperationTarget::BATCHNORM_INFERENCE);
    EXPECT_EQ(batch->phase(), NormFwdPhase_t::INFERENCE);
    EXPECT_EQ(batch->epsilon(), nullptr);
    EXPECT_EQ(batch->mean(), inference.mean);
    EXPECT_EQ(batch->normalizedDimCount(), 0);

    const std::shared_ptr<const v8::NormForwardOperationRecord> layer = v8::makeOperationRecord(
        normForward(CUDNN_LAYER_NORM, CUDNN_NORM_FWD_TRAINING), NORM_FWD_LABELS);
    EXPECT_EQ(layer->target(), OperationTarget::LAYERNORM);
    EXPECT_EQ(layer->normalizedDimCount(), 3);

    v8::NormForwardOperationInputs rms = normForward(CUDNN_RMS_NORM, CUDNN_NORM_FWD_TRAINING);
    rms.bias = nullptr;
    rms.invVariance = tensor({2, 1, 1, 1}, 8);
    const std::shared_ptr<const v8::NormForwardOperationRecord> rmsRecord
        = v8::makeOperationRecord(rms, NORM_FWD_LABELS);
    EXPECT_EQ(rmsRecord->target(), OperationTarget::RMSNORM);
    EXPECT_EQ(rmsRecord->bias(), nullptr);
    EXPECT_EQ(rmsRecord->invVariance(), rms.invVariance);
}

TEST(TestCudnnShimV8CoreFactories, NormForwardGroupsAndRejections)
{
    v8::NormForwardOperationInputs batch = normForward(CUDNN_BATCH_NORM, CUDNN_NORM_FWD_TRAINING);
    batch.inputRunningMean = tensor({1, 8, 1, 1}, 6);
    batch.expAvgFactor = tensor({1, 1, 1, 1}, 7);
    expectShimError([&] { v8::makeOperationRecord(batch, NORM_FWD_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "inputRunningVariance is required when inputRunningMean is set");

    v8::NormForwardOperationInputs layer = normForward(CUDNN_LAYER_NORM, CUDNN_NORM_FWD_TRAINING);
    layer.mean = tensor({2, 1, 1, 1}, 6);
    layer.hasPeerStats = true;
    expectShimError([&] { v8::makeOperationRecord(layer, NORM_FWD_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "invVariance is required when mean is set");

    layer.mean = nullptr;
    expectShimError([&] { v8::makeOperationRecord(layer, NORM_FWD_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "peerStats is not supported by the hipDNN cuDNN shim");

    layer.hasPeerStats = false;
    layer.outputRunningMean = tensor({1, 8, 1, 1}, 6);
    expectShimError([&] { v8::makeOperationRecord(layer, NORM_FWD_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "outputRunningMean is not supported by the hipDNN cuDNN shim for this "
                    "normalization mode and phase");

    v8::NormForwardOperationInputs rms = normForward(CUDNN_RMS_NORM, CUDNN_NORM_FWD_TRAINING);
    rms.mean = tensor({2, 1, 1, 1}, 6);
    rms.invVariance = tensor({2, 1, 1, 1}, 7);
    expectShimError([&] { v8::makeOperationRecord(rms, NORM_FWD_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "mean is not supported by the hipDNN cuDNN shim for CUDNN_RMS_NORM");

    expectShimError(
        [] {
            v8::makeOperationRecord(normForward(CUDNN_GROUP_NORM, CUDNN_NORM_FWD_TRAINING),
                                    NORM_FWD_LABELS);
        },
        CUDNN_STATUS_NOT_SUPPORTED,
        "mode CUDNN_GROUP_NORM is not supported by the hipDNN cuDNN shim");
}

TEST(TestCudnnShimV8CoreFactories, NormBackwardTargetsAndRules)
{
    v8::NormBackwardOperationInputs batch = normBackward(CUDNN_BATCH_NORM);
    batch.epsilon = tensor({1, 1, 1, 1}, 9);
    const std::shared_ptr<const v8::NormBackwardOperationRecord> batchRecord
        = v8::makeOperationRecord(batch, NORM_BWD_LABELS);
    EXPECT_EQ(batchRecord->target(), OperationTarget::BATCHNORM_BACKWARD);
    EXPECT_EQ(batchRecord->epsilon(), nullptr);
    batch.mean = nullptr;
    expectShimError([&] { v8::makeOperationRecord(batch, NORM_BWD_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "mean is required");

    v8::NormBackwardOperationInputs layer = normBackward(CUDNN_LAYER_NORM);
    layer.scale = tensor({1, 8, 3, 3}, 3);
    layer.epsilon = tensor({1, 1, 1, 1}, 9);
    const std::shared_ptr<const v8::NormBackwardOperationRecord> layerRecord
        = v8::makeOperationRecord(layer, NORM_BWD_LABELS);
    EXPECT_EQ(layerRecord->target(), OperationTarget::LAYERNORM_BACKWARD);
    EXPECT_EQ(layerRecord->epsilon(), layer.epsilon);
    EXPECT_EQ(layerRecord->normalizedDimCount(), 3);
    layer.invVariance = nullptr;
    expectShimError([&] { v8::makeOperationRecord(layer, NORM_BWD_LABELS); },
                    CUDNN_STATUS_BAD_PARAM,
                    "invVariance is required when mean is set");

    v8::NormBackwardOperationInputs rms = normBackward(CUDNN_RMS_NORM);
    rms.dbias = nullptr;
    expectShimError([&] { v8::makeOperationRecord(rms, NORM_BWD_LABELS); },
                    CUDNN_STATUS_NOT_SUPPORTED,
                    "mean is not supported by the hipDNN cuDNN shim for CUDNN_RMS_NORM");
    rms.mean = nullptr;
    const std::shared_ptr<const v8::NormBackwardOperationRecord> rmsRecord
        = v8::makeOperationRecord(rms, NORM_BWD_LABELS);
    EXPECT_EQ(rmsRecord->target(), OperationTarget::RMSNORM_BACKWARD);
    EXPECT_EQ(rmsRecord->dbias(), nullptr);
}

TEST(TestCudnnShimV8CoreFactories, RecordAsChecksTheKindTag)
{
    const std::shared_ptr<const v8::Record> record = tensor();
    EXPECT_EQ(&v8::recordAs<v8::TensorRecord>(*record), record.get());
    expectShimError([&] { v8::recordAs<v8::ConvolutionRecord>(*record); },
                    CUDNN_STATUS_BAD_PARAM,
                    "record of descriptor type "
                        + std::to_string(static_cast<int>(CUDNN_BACKEND_TENSOR_DESCRIPTOR))
                        + " does not have the expected type");
    EXPECT_THROW(v8::recordAs<v8::OperationRecord>(*record), v8::ShimError);

    v8::PointwiseOperationInputs inputs;
    inputs.pointwise = pointwise(CUDNN_POINTWISE_SQRT);
    inputs.x = tensor({4}, 1);
    inputs.y = tensor({4}, 2);
    const std::shared_ptr<const v8::Record> operation
        = v8::makeOperationRecord(inputs, POINTWISE_OP_LABELS);
    EXPECT_EQ(v8::recordAs<v8::OperationRecord>(*operation).target(), OperationTarget::POINTWISE);
    EXPECT_EQ(v8::recordAs<v8::PointwiseOperationRecord>(operation).get(), operation.get());
    EXPECT_EQ(v8::recordAs<v8::PointwiseOperationRecord>(std::shared_ptr<const v8::Record>()),
              nullptr);
}

} // namespace
