// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRMSNorm.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

template <typename T1, typename T2>
struct TypePair
{
    using First = T1;
    using Second = T2;
};

using TypesRMSNormFwdNchw = ::testing::Types<TypePair<float, float>,
                                             TypePair<half, float>,
                                             TypePair<bfloat16, float>,
                                             TypePair<double, double>>;

template <class T>
class CpuFpReferenceRMSNormFwdNchw : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceRMSNormFwdNchw, TypesRMSNormFwdNchw, );

TYPED_TEST(CpuFpReferenceRMSNormFwdNchw, RMSNormFwdNchw)
{
    Tensor<typename TypeParam::First> inputTensor({1, 3, 224, 224});
    Tensor<typename TypeParam::First> outputTensor({1, 3, 224, 224});
    Tensor<typename TypeParam::Second> scaleTensor({1, 3, 224, 224});

    inputTensor.fillWithValue(safeTestTypeCast<typename TypeParam::First>(1.0));
    scaleTensor.fillWithValue(safeTestTypeCast<typename TypeParam::Second>(1.0));

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

using TypesRMSNormFwdNhwc = ::testing::Types<TypePair<float, float>, TypePair<half, bfloat16>>;

template <class T>
class CpuFpReferenceRMSNormFwdNhwc : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceRMSNormFwdNhwc, TypesRMSNormFwdNhwc, );

TYPED_TEST(CpuFpReferenceRMSNormFwdNhwc, RMSNormFwdNhwc)
{
    Tensor<typename TypeParam::First> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<typename TypeParam::First> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<typename TypeParam::Second> scaleTensor({1, 3, 32, 32});

    inputTensor.fillWithValue(safeTestTypeCast<typename TypeParam::First>(1.0));
    scaleTensor.fillWithValue(safeTestTypeCast<typename TypeParam::Second>(1.0));

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdSanityValidationNchw)
{
    // RMSNorm: y = x / sqrt(mean(x^2) + epsilon) * scale
    // Reduction is over all non-batch dims; here spatial is 1x1 so reduction is
    // effectively over C only.
    const std::vector<int64_t> dims = {1, 4, 1, 1};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor(dims); // full-shape scale

    // x = [1, 2, 3, 4] across 4 channels at a single spatial position
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 1, 0, 0);
    inputTensor.setHostValue(3.0, 0, 2, 0, 0);
    inputTensor.setHostValue(4.0, 0, 3, 0, 0);

    for(int i = 0; i < 4; i++)
    {
        scaleTensor.setHostValue(1.0, 0, i);
    }

    const double epsilon = 1e-5;

    // mean_C(x^2) = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    // rms = sqrt(7.5 + 1e-5) = 2.73861278753...
    // inv_rms = 1 / rms = 0.36514837167...
    // y[c] = x[c] * inv_rms * scale
    // y[0] = 1.0 * 0.36514837167 = 0.36514837167
    // y[1] = 2.0 * 0.36514837167 = 0.73029674335
    // y[2] = 3.0 * 0.36514837167 = 1.09544511502
    // y[3] = 4.0 * 0.36514837167 = 1.46059348669

    const double invRmsExpected = 1.0 / std::sqrt(7.5 + epsilon);
    const std::vector<double> expectedOutput
        = {1.0 * invRmsExpected, 2.0 * invRmsExpected, 3.0 * invRmsExpected, 4.0 * invRmsExpected};

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, epsilon);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 0), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 2, 0, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 3, 0, 0), expectedOutput[3], tolerance);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdWithInvRms)
{
    const std::vector<int64_t> dims = {1, 4, 1, 1};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor(dims); // full-shape scale
    Tensor<float> invRmsTensor({1, 1, 1, 1});

    // x = [1, 2, 3, 4] across 4 channels
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 1, 0, 0);
    inputTensor.setHostValue(3.0, 0, 2, 0, 0);
    inputTensor.setHostValue(4.0, 0, 3, 0, 0);

    for(int i = 0; i < 4; i++)
    {
        scaleTensor.setHostValue(2.0, 0, i);
    }

    const double epsilon = 1e-5;

    const double invRmsExpected = 1.0 / std::sqrt(7.5 + epsilon);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, epsilon, &invRmsTensor);

    auto tolerance = 1e-5;

    EXPECT_NEAR(
        static_cast<double>(invRmsTensor.getHostValue(0, 0, 0, 0)), invRmsExpected, tolerance);

    // y = x * inv_rms * scale (scale = 2.0)
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 1.0 * invRmsExpected * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 0), 2.0 * invRmsExpected * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 2, 0, 0), 3.0 * invRmsExpected * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 3, 0, 0), 4.0 * invRmsExpected * 2.0, tolerance);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdPartialSuffixScale)
{
    // Partial-suffix scale: scale matches input's trailing two dims (H, W) and is 1
    // elsewhere. matchCount=2 -> reductionStart=2, so leadingDims=[N, C] and each
    // (N, C) position gets its own invRms. Distinct from the full-shape scale tests
    // above, which collapse all non-batch dims to a single rms.
    const std::vector<int64_t> inputDims = {1, 2, 1, 2};
    const std::vector<int64_t> scaleDims = {1, 1, 1, 2}; // partial-suffix scale
    const std::vector<int64_t> invRmsDims = {1, 2, 1, 1}; // leading=[N, C], reduction collapses

    Tensor<double> inputTensor(inputDims);
    Tensor<double> outputTensor(inputDims);
    Tensor<double> scaleTensor(scaleDims);
    Tensor<float> invRmsTensor(invRmsDims);

    // Channel 0: x = [1, 2]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    // Channel 1: x = [3, 4]
    inputTensor.setHostValue(3.0, 0, 1, 0, 0);
    inputTensor.setHostValue(4.0, 0, 1, 0, 1);

    // Scale broadcasts over (N, C, H), varies over W: [1.0, 2.0].
    scaleTensor.setHostValue(1.0, 0, 0, 0, 0);
    scaleTensor.setHostValue(2.0, 0, 0, 0, 1);

    const double epsilon = 1e-5;

    // Per-channel reduction over [H, W] (size 1×2):
    //   C=0: mean(1^2, 2^2)/2 = 2.5  -> invRms_C0 = 1 / sqrt(2.5 + eps)
    //   C=1: mean(3^2, 4^2)/2 = 12.5 -> invRms_C1 = 1 / sqrt(12.5 + eps)
    const double invRmsC0 = 1.0 / std::sqrt(2.5 + epsilon);
    const double invRmsC1 = 1.0 / std::sqrt(12.5 + epsilon);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, epsilon, &invRmsTensor);

    const auto tolerance = 1e-6;

    EXPECT_NEAR(static_cast<double>(invRmsTensor.getHostValue(0, 0, 0, 0)), invRmsC0, tolerance);
    EXPECT_NEAR(static_cast<double>(invRmsTensor.getHostValue(0, 1, 0, 0)), invRmsC1, tolerance);

    // y[n, c, h, w] = scale[0, 0, 0, w] * x[n, c, h, w] * invRms[n, c, 0, 0]
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 1.0 * invRmsC0 * 1.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), 2.0 * invRmsC0 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 0), 3.0 * invRmsC1 * 1.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 1), 4.0 * invRmsC1 * 2.0, tolerance);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormFwd2D)
{
    Tensor<float> inputTensor({4, 3});
    Tensor<float> outputTensor({4, 3});
    Tensor<float> scaleTensor({1, 3});

    inputTensor.fillWithValue(1.0f);
    scaleTensor.fillWithValue(1.0f);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormFwd3D)
{
    Tensor<float> inputTensor({2, 3, 10});
    Tensor<float> outputTensor({2, 3, 10});
    Tensor<float> scaleTensor({1, 3, 10});

    inputTensor.fillWithValue(2.0f);
    scaleTensor.fillWithValue(2.0f);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormFwdNcdhw)
{
    Tensor<float> inputTensor({2, 3, 4, 5, 6});
    Tensor<float> outputTensor({2, 3, 4, 5, 6});
    Tensor<float> scaleTensor({1, 3, 4, 5, 6});

    inputTensor.fillWithValue(1.5f);
    scaleTensor.fillWithValue(1.0f);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormFwdNdhwc)
{
    Tensor<float> inputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> outputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> scaleTensor({1, 3, 4, 5, 6});

    inputTensor.fillWithValue(1.5f);
    scaleTensor.fillWithValue(2.0f);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdConstantInput)
{
    // When all inputs are the same constant c, mean(x^2) = c^2
    // inv_rms = 1/sqrt(c^2+eps) ~ 1/c
    // y = x * inv_rms * scale = c * (1/c) * scale = scale
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor(dims); // full-shape scale

    const double c = 3.0;
    inputTensor.fillWithValue(c);
    scaleTensor.fillWithValue(2.0);

    const double epsilon = 1e-5;

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, epsilon);

    const double invRms = 1.0 / std::sqrt(c * c + epsilon);
    const double expectedY = c * invRms * 2.0;

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedY, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedY, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedY, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedY, tolerance);
}

TEST(TestCpuFpReferenceRMSNorm, RMSNormFwdWithBias)
{
    // bias is added per-element after scale multiplication:
    // y = x / rms * scale + bias
    const std::vector<int64_t> dims = {1, 2, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputTensor(dims);
    Tensor<float> scaleTensor(dims); // full-shape scale
    Tensor<float> biasTensor(dims); // full-shape bias

    // Constant input: all 1s so rms = sqrt(1 + eps) ~ 1
    inputTensor.fillWithValue(1.0f);

    // Per-channel scale/bias (replicated across the spatial positions).
    // channel 0: scale=2, bias=0.5
    scaleTensor.setHostValue(2.0f, 0, 0, 0, 0);
    scaleTensor.setHostValue(2.0f, 0, 0, 0, 1);
    scaleTensor.setHostValue(2.0f, 0, 0, 1, 0);
    scaleTensor.setHostValue(2.0f, 0, 0, 1, 1);
    biasTensor.setHostValue(0.5f, 0, 0, 0, 0);
    biasTensor.setHostValue(0.5f, 0, 0, 0, 1);
    biasTensor.setHostValue(0.5f, 0, 0, 1, 0);
    biasTensor.setHostValue(0.5f, 0, 0, 1, 1);
    // channel 1: scale=3, bias=-1.0
    scaleTensor.setHostValue(3.0f, 0, 1, 0, 0);
    scaleTensor.setHostValue(3.0f, 0, 1, 0, 1);
    scaleTensor.setHostValue(3.0f, 0, 1, 1, 0);
    scaleTensor.setHostValue(3.0f, 0, 1, 1, 1);
    biasTensor.setHostValue(-1.0f, 0, 1, 0, 0);
    biasTensor.setHostValue(-1.0f, 0, 1, 0, 1);
    biasTensor.setHostValue(-1.0f, 0, 1, 1, 0);
    biasTensor.setHostValue(-1.0f, 0, 1, 1, 1);

    const double epsilon = 0.0; // zero epsilon so inv_rms = 1

    Tensor<float>* noInvRms = nullptr;
    CpuFpReferenceRMSNorm::forward(
        inputTensor, scaleTensor, outputTensor, epsilon, noInvRms, &biasTensor);

    // y = x * invRms * scale + bias = 1 * 1 * scale + bias
    const float expectedC0 = 2.0f + 0.5f; // 2.5
    const float expectedC1 = 3.0f + -1.0f; // 2.0

    auto tolerance = 1e-5f;
    // Channel 0 outputs
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedC0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedC0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedC0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedC0, tolerance);
    // Channel 1 outputs
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 0), expectedC1, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 1), expectedC1, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 1, 0), expectedC1, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 1, 1), expectedC1, tolerance);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdDegenerateAllOnesNonBatch)
{
    // Degenerate case: all-1 scale, all-1 input non-batch dims.
    // Reduction volume = 1 per batch element, so mean(x^2) = x^2 and invRms = 1/|x|.
    // y = x * invRms * scale = sign(x) (with scale = 1, eps = 0).
    const std::vector<int64_t> dims = {2, 1, 1, 1};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor(dims);
    Tensor<float> invRmsTensor(dims); // invRms shape == input shape

    inputTensor.setHostValue(3.0, 0, 0, 0, 0);
    inputTensor.setHostValue(4.0, 1, 0, 0, 0);
    scaleTensor.fillWithValue(1.0);

    const double epsilon = 0.0;

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, epsilon, &invRmsTensor);

    const auto tolerance = 1e-6;

    EXPECT_NEAR(static_cast<double>(invRmsTensor.getHostValue(0, 0, 0, 0)), 1.0 / 3.0, tolerance);
    EXPECT_NEAR(static_cast<double>(invRmsTensor.getHostValue(1, 0, 0, 0)), 1.0 / 4.0, tolerance);

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 1.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 0, 0), 1.0, tolerance);
}

TEST(TestCpuFpReferenceRMSNorm, RMSNormFwdBiasIsOptional)
{
    // Passing nullptr bias should give the same result as no-bias call
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputNoBias(dims);
    Tensor<float> outputNullBias(dims);
    Tensor<float> scaleTensor(dims); // full-shape scale

    inputTensor.fillWithValue(2.0f);
    scaleTensor.fillWithValue(1.5f);
    const double epsilon = 1e-5;

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputNoBias, epsilon);
    Tensor<float>* noInvRms2 = nullptr;
    Tensor<float>* noBias = nullptr;
    CpuFpReferenceRMSNorm::forward(
        inputTensor, scaleTensor, outputNullBias, epsilon, noInvRms2, noBias);

    auto tolerance = 1e-6f;
    EXPECT_NEAR(
        outputNoBias.getHostValue(0, 0, 0, 0), outputNullBias.getHostValue(0, 0, 0, 0), tolerance);
    EXPECT_NEAR(
        outputNoBias.getHostValue(0, 0, 1, 1), outputNullBias.getHostValue(0, 0, 1, 1), tolerance);
}

TEST(TestCpuFpReferenceRMSNorm, RMSNormFwdRejectsRankMismatch)
{
    const Tensor<float> inputTensor({2, 3, 4, 4});
    Tensor<float> outputTensor({2, 3, 4, 4});
    const Tensor<float> scaleTensor({1, 3, 4}); // wrong rank
    EXPECT_THROW(CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdScaleNormAxis2)
{
    Tensor<double> inputTensor({2, 2, 2, 2});
    Tensor<double> outputTensor({2, 2, 2, 2});
    Tensor<double> scaleTensor({1, 1, 2, 2});

    // Batch 0, Channel 0
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(1.0, 0, 0, 1, 0);
    inputTensor.setHostValue(2.0, 0, 0, 1, 1);

    // Batch 0, Channel 1
    inputTensor.setHostValue(3.0, 0, 1, 0, 0);
    inputTensor.setHostValue(4.0, 0, 1, 0, 1);
    inputTensor.setHostValue(3.0, 0, 1, 1, 0);
    inputTensor.setHostValue(4.0, 0, 1, 1, 1);

    // Batch 1, Channel 0
    inputTensor.setHostValue(1.0, 1, 0, 0, 0);
    inputTensor.setHostValue(2.0, 1, 0, 0, 1);
    inputTensor.setHostValue(1.0, 1, 0, 1, 0);
    inputTensor.setHostValue(2.0, 1, 0, 1, 1);

    // Batch 1, Channel 1
    inputTensor.setHostValue(3.0, 1, 1, 0, 0);
    inputTensor.setHostValue(4.0, 1, 1, 0, 1);
    inputTensor.setHostValue(3.0, 1, 1, 1, 0);
    inputTensor.setHostValue(4.0, 1, 1, 1, 1);

    // Scale broadcasts over (N, C); varies over (H, W).
    scaleTensor.setHostValue(2.0, 0, 0, 0, 0);
    scaleTensor.setHostValue(2.5, 0, 0, 0, 1);
    scaleTensor.setHostValue(3.0, 0, 0, 1, 0);
    scaleTensor.setHostValue(3.5, 0, 0, 1, 1);

    const double epsilon = 1e-5;

    // matchCount=2 -> reduction over (H, W) for each (n, c). Each (n, c) gets one invRms:
    //   C=0: x^2={1,4,1,4}, mean=2.5   -> invRms_C0 = 1 / sqrt(2.5 + eps)
    //   C=1: x^2={9,16,9,16}, mean=12.5 -> invRms_C1 = 1 / sqrt(12.5 + eps)
    const double invRmsC0 = 1.0 / std::sqrt(2.5 + epsilon);
    const double invRmsC1 = 1.0 / std::sqrt(12.5 + epsilon);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, epsilon);

    auto tolerance = 1e-6;

    // y[n, c, h, w] = scale[0, 0, h, w] * x[n, c, h, w] * invRms[n, c, 0, 0]

    // Batch 0, Channel 0
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 1.0 * invRmsC0 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), 2.0 * invRmsC0 * 2.5, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), 1.0 * invRmsC0 * 3.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), 2.0 * invRmsC0 * 3.5, tolerance);

    // Batch 0, Channel 1
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 0), 3.0 * invRmsC1 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 1), 4.0 * invRmsC1 * 2.5, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 1, 0), 3.0 * invRmsC1 * 3.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 1, 1), 4.0 * invRmsC1 * 3.5, tolerance);

    // Batch 1, Channel 0
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 0, 0), 1.0 * invRmsC0 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 0, 1), 2.0 * invRmsC0 * 2.5, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 1, 0), 1.0 * invRmsC0 * 3.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 1, 1), 2.0 * invRmsC0 * 3.5, tolerance);

    // Batch 1, Channel 1
    EXPECT_NEAR(outputTensor.getHostValue(1, 1, 0, 0), 3.0 * invRmsC1 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 1, 0, 1), 4.0 * invRmsC1 * 2.5, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 1, 1, 0), 3.0 * invRmsC1 * 3.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 1, 1, 1), 4.0 * invRmsC1 * 3.5, tolerance);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdScaleNormAxis3)
{
    Tensor<double> inputTensor({2, 2, 2, 2});
    Tensor<double> outputTensor({2, 2, 2, 2});
    Tensor<double> scaleTensor({1, 1, 1, 2});

    // Batch 0, Channel 0
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(1.0, 0, 0, 1, 0);
    inputTensor.setHostValue(2.0, 0, 0, 1, 1);

    // Batch 0, Channel 1
    inputTensor.setHostValue(3.0, 0, 1, 0, 0);
    inputTensor.setHostValue(4.0, 0, 1, 0, 1);
    inputTensor.setHostValue(3.0, 0, 1, 1, 0);
    inputTensor.setHostValue(4.0, 0, 1, 1, 1);

    // Batch 1, Channel 0
    inputTensor.setHostValue(1.0, 1, 0, 0, 0);
    inputTensor.setHostValue(2.0, 1, 0, 0, 1);
    inputTensor.setHostValue(1.0, 1, 0, 1, 0);
    inputTensor.setHostValue(2.0, 1, 0, 1, 1);

    // Batch 1, Channel 1
    inputTensor.setHostValue(3.0, 1, 1, 0, 0);
    inputTensor.setHostValue(4.0, 1, 1, 0, 1);
    inputTensor.setHostValue(3.0, 1, 1, 1, 0);
    inputTensor.setHostValue(4.0, 1, 1, 1, 1);

    // Scale broadcasts over (N, C, H); varies over W.
    scaleTensor.setHostValue(2.0, 0, 0, 0, 0);
    scaleTensor.setHostValue(2.5, 0, 0, 0, 1);

    const double epsilon = 1e-5;

    // matchCount=1 -> reduction over W only for each (n, c, h). Input is symmetric
    // across N and H, so invRms collapses to per-channel here:
    //   C=0: x^2={1,4}, mean=2.5   -> invRms_C0 = 1 / sqrt(2.5 + eps)
    //   C=1: x^2={9,16}, mean=12.5 -> invRms_C1 = 1 / sqrt(12.5 + eps)
    const double invRmsC0 = 1.0 / std::sqrt(2.5 + epsilon);
    const double invRmsC1 = 1.0 / std::sqrt(12.5 + epsilon);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, epsilon);

    auto tolerance = 1e-6;

    // y[n, c, h, w] = scale[0, 0, 0, w] * x[n, c, h, w] * invRms[n, c, h, 0]

    // Batch 0, Channel 0
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 1.0 * invRmsC0 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), 2.0 * invRmsC0 * 2.5, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), 1.0 * invRmsC0 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), 2.0 * invRmsC0 * 2.5, tolerance);

    // Batch 0, Channel 1
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 0), 3.0 * invRmsC1 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 1), 4.0 * invRmsC1 * 2.5, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 1, 0), 3.0 * invRmsC1 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 1, 1), 4.0 * invRmsC1 * 2.5, tolerance);

    // Batch 1, Channel 0
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 0, 0), 1.0 * invRmsC0 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 0, 1), 2.0 * invRmsC0 * 2.5, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 1, 0), 1.0 * invRmsC0 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 0, 1, 1), 2.0 * invRmsC0 * 2.5, tolerance);

    // Batch 1, Channel 1
    EXPECT_NEAR(outputTensor.getHostValue(1, 1, 0, 0), 3.0 * invRmsC1 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 1, 0, 1), 4.0 * invRmsC1 * 2.5, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 1, 1, 0), 3.0 * invRmsC1 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(1, 1, 1, 1), 4.0 * invRmsC1 * 2.5, tolerance);
}

template <typename DyT, typename XT, typename ScaleT, typename DxT, typename ComputeT>
struct BwdTypeGroup
{
    using Dy = DyT;
    using X = XT;
    using Scale = ScaleT;
    using Dx = DxT;
    using Compute = ComputeT;
};

using RMSNormBwdTypes = ::testing::Types<BwdTypeGroup<float, float, float, float, float>,
                                         BwdTypeGroup<half, half, float, half, float>,
                                         BwdTypeGroup<bfloat16, bfloat16, float, bfloat16, float>>;

template <class T>
class CpuFpReferenceRMSNormBwdNchw : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceRMSNormBwdNchw, RMSNormBwdTypes, );

TYPED_TEST(CpuFpReferenceRMSNormBwdNchw, RMSNormBwdNchw)
{
    using DyT = typename TypeParam::Dy;
    using XT = typename TypeParam::X;
    using ScaleT = typename TypeParam::Scale;
    using DxT = typename TypeParam::Dx;
    using ComputeT = typename TypeParam::Compute;

    const std::vector<int64_t> dims = {4, 16, 8, 8};

    Tensor<DyT> dy(dims, TensorLayout::NCHW);
    Tensor<XT> x(dims, TensorLayout::NCHW);
    Tensor<ScaleT> scale({1, 16, 8, 8});
    Tensor<ComputeT> invRms({4, 1, 1, 1});

    Tensor<DxT> dx(dims, TensorLayout::NCHW);
    Tensor<ScaleT> dscale({1, 16, 8, 8});
    Tensor<ScaleT> dbias({1, 16, 8, 8});

    dy.fillWithValue(safeTestTypeCast<DyT>(1.0));
    x.fillWithValue(safeTestTypeCast<XT>(1.0));
    scale.fillWithValue(safeTestTypeCast<ScaleT>(1.0));
    invRms.fillWithValue(safeTestTypeCast<ComputeT>(0.5));

    CpuFpReferenceRMSNorm::backward<DyT, XT, ScaleT, DxT, ComputeT>(
        dy, x, scale, invRms, dx, dscale, &dbias);
}

template <class T>
class CpuFpReferenceRMSNormBwdNhwc : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceRMSNormBwdNhwc, RMSNormBwdTypes, );

TYPED_TEST(CpuFpReferenceRMSNormBwdNhwc, RMSNormBwdNhwc)
{
    using DyT = typename TypeParam::Dy;
    using XT = typename TypeParam::X;
    using ScaleT = typename TypeParam::Scale;
    using DxT = typename TypeParam::Dx;
    using ComputeT = typename TypeParam::Compute;

    const std::vector<int64_t> dims = {4, 8, 8, 16};

    Tensor<DyT> dy(dims, TensorLayout::NHWC);
    Tensor<XT> x(dims, TensorLayout::NHWC);
    Tensor<ScaleT> scale({1, 8, 8, 16});
    Tensor<ComputeT> invRms({4, 1, 1, 1});

    Tensor<DxT> dx(dims, TensorLayout::NHWC);
    Tensor<ScaleT> dscale({1, 8, 8, 16});
    Tensor<ScaleT> dbias({1, 8, 8, 16});

    dy.fillWithValue(safeTestTypeCast<DyT>(1.0));
    x.fillWithValue(safeTestTypeCast<XT>(1.0));
    scale.fillWithValue(safeTestTypeCast<ScaleT>(1.0));
    invRms.fillWithValue(safeTestTypeCast<ComputeT>(0.5));

    CpuFpReferenceRMSNorm::backward<DyT, XT, ScaleT, DxT, ComputeT>(
        dy, x, scale, invRms, dx, dscale, &dbias);
}

TEST(TestCpuFpReferenceRMSNorm, RMSNormBwdRejectsRankMismatch)
{
    const Tensor<float> dy({2, 3, 4, 4});
    const Tensor<float> x({2, 3, 4, 4});
    const Tensor<float> scale({2, 3, 4, 4});
    const Tensor<float> invRms({1, 3, 4}); // wrong rank
    Tensor<float> dx({2, 3, 4, 4});
    Tensor<float> dScale({2, 3, 4, 4});

    EXPECT_THROW(CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dScale),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormBwd2D)
{
    Tensor<float> dy({4, 3});
    Tensor<float> x({4, 3});
    Tensor<float> scale({1, 3});
    Tensor<float> invRms({4, 1});

    Tensor<float> dx({4, 3});
    Tensor<float> dscale({1, 3});

    dy.fillWithValue(1.0f);
    x.fillWithValue(1.0f);
    scale.fillWithValue(1.0f);
    invRms.fillWithValue(0.5f);

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormBwd3D)
{
    Tensor<float> dy({2, 3, 10});
    Tensor<float> x({2, 3, 10});
    Tensor<float> scale({1, 3, 10});
    Tensor<float> invRms({2, 1, 1});

    Tensor<float> dx({2, 3, 10});
    Tensor<float> dscale({1, 3, 10});

    dy.fillWithValue(0.5f);
    x.fillWithValue(2.0f);
    scale.fillWithValue(1.0f);
    invRms.fillWithValue(0.1f);

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormBwdNcdhw)
{
    Tensor<float> dy({2, 3, 4, 5, 6});
    Tensor<float> x({2, 3, 4, 5, 6});
    Tensor<float> scale({1, 3, 4, 5, 6});
    Tensor<float> invRms({2, 1, 1, 1, 1});

    Tensor<float> dx({2, 3, 4, 5, 6});
    Tensor<float> dscale({1, 3, 4, 5, 6});
    Tensor<float> dbias({1, 3, 4, 5, 6});

    dy.fillWithValue(1.0f);
    x.fillWithValue(1.2f);
    scale.fillWithValue(1.0f);
    invRms.fillWithValue(0.8f);

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale, &dbias);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormBwdNdhwc)
{
    Tensor<float> dy({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> x({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> scale({1, 1, 1, 1, 6}, TensorLayout::NDHWC);
    Tensor<float> invRms({2, 3, 4, 5, 1}, TensorLayout::NDHWC);

    Tensor<float> dx({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> dscale({1, 1, 1, 1, 6}, TensorLayout::NDHWC);

    dy.fillWithValue(0.1f);
    x.fillWithValue(1.0f);
    scale.fillWithValue(1.0f);
    invRms.fillWithValue(0.9f);

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormBwdSanityValidation)
{
    const std::vector<int64_t> dims = {1, 4, 1, 1};
    const double epsilon = 1e-5;

    Tensor<double> dy(dims);
    Tensor<double> x(dims);
    Tensor<double> scale(dims);
    Tensor<double> dx(dims);
    Tensor<double> dscale(dims);
    Tensor<double> invRms({1, 1, 1, 1});

    // Inputs: x = [1, 2, 3, 4], scale = [1, 1, 1, 1], dy = [0.1, 0.2, 0.3, 0.4]
    for(int i = 0; i < 4; i++)
    {
        x.setHostValue(static_cast<double>(i + 1), 0, i, 0, 0);
        dy.setHostValue(static_cast<double>(i + 1) * 0.1, 0, i, 0, 0);
        scale.setHostValue(1.0, 0, i, 0, 0);
    }

    // Forward stats: mean(x^2) = 7.5
    const double invRmsVal = 1.0 / std::sqrt(7.5 + epsilon);
    invRms.setHostValue(invRmsVal, 0, 0, 0, 0);

    // Manual gradient calculation:
    // 1. Dot product: sum(dy * scale * x) = (0.1*1*1 + 0.2*1*2 + 0.3*1*3 + 0.4*1*4) = 0.1+0.4+0.9+1.6 = 3.0
    // 2. Mean projection: m = 3.0 / 4 = 0.75
    // 3. dx[i] = (dy[i]*scale[i]*invRms) - (m * x[i] * invRms^3)
    const double projection = 0.75;
    const double invRmsCubed = invRmsVal * invRmsVal * invRmsVal;

    std::vector<double> expectedDx(4);
    std::vector<double> expectedDScale(4);
    for(size_t i = 0; i < 4; i++)
    {
        const auto valX = static_cast<double>(i + 1);
        const double valDy = valX * 0.1;
        expectedDx[i] = (valDy * 1.0 * invRmsVal) - (projection * valX * invRmsCubed);
        expectedDScale[i] = valDy * (valX * invRmsVal);
    }

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale);

    const double tolerance = 1e-6;
    for(size_t i = 0; i < 4; i++)
    {
        EXPECT_NEAR(dx.getHostValue(0, i, 0, 0), expectedDx[i], tolerance);
        EXPECT_NEAR(dscale.getHostValue(0, i, 0, 0), expectedDScale[i], tolerance);
    }
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormBwdDbiasValidation)
{
    const std::vector<int64_t> dims = {1, 2, 1, 1};
    Tensor<double> dy(dims);
    Tensor<double> x(dims);
    Tensor<double> scale(dims);
    Tensor<double> dx(dims);
    Tensor<double> dscale(dims);
    Tensor<double> dbias(dims);
    Tensor<double> invRms({1, 1, 1, 1});

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    dy.setHostValue(0.5, 0, 0);
    dy.setHostValue(0.5, 0, 1);
    scale.setHostValue(1.0, 0, 0);
    scale.setHostValue(1.0, 0, 1);

    const double invRmsVal = 1.0 / std::sqrt(2.5);
    invRms.setHostValue(invRmsVal, 0, 0, 0, 0);

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale, &dbias);

    // dbias is simply the sum of dy over the batch. Since batch=1, dbias == dy.
    EXPECT_NEAR(dbias.getHostValue(0, 0, 0, 0), 0.5, 1e-10);
    EXPECT_NEAR(dbias.getHostValue(0, 1, 0, 0), 0.5, 1e-10);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormBwdPartialSuffixScale)
{
    const std::vector<int64_t> dims = {1, 2, 1, 2};
    const std::vector<int64_t> scaleDims = {1, 1, 1, 2};
    const std::vector<int64_t> invRmsDims = {1, 2, 1, 1};

    Tensor<double> dy(dims);
    Tensor<double> x(dims);
    Tensor<double> scale(scaleDims);
    Tensor<double> dx(dims);
    Tensor<double> dscale(scaleDims);
    Tensor<double> invRms(invRmsDims);

    // C=0: x=[1, 2], dy=[0.1, 0.1]
    x.setHostValue(1.0, 0, 0, 0, 0);
    x.setHostValue(2.0, 0, 0, 0, 1);
    dy.setHostValue(0.1, 0, 0, 0, 0);
    dy.setHostValue(0.1, 0, 0, 0, 1);

    // C=1: x=[3, 4], dy=[0.2, 0.2]
    x.setHostValue(3.0, 0, 1, 0, 0);
    x.setHostValue(4.0, 0, 1, 0, 1);
    dy.setHostValue(0.2, 0, 1, 0, 0);
    dy.setHostValue(0.2, 0, 1, 0, 1);

    // Scale varies over scale: [1.0, 0.5]
    scale.setHostValue(1.0, 0, 0, 0, 0);
    scale.setHostValue(0.5, 0, 0, 0, 1);

    // Pre-calculated invRms (mean(1,4) = 2.5, mean(9,16) = 12.5)
    const double invRmsC0 = 1.0 / std::sqrt(2.5);
    const double invRmsC1 = 1.0 / std::sqrt(12.5);
    invRms.setHostValue(invRmsC0, 0, 0, 0, 0);
    invRms.setHostValue(invRmsC1, 0, 1, 0, 0);

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale);

    // dscale[w] = sum_{n,c,h} (dy[n,c,h,w] * x[n,c,h,w] * invRms[n,c,h,w])
    // dscale[0] = (dy[0,0,0,0]*x[0,0,0,0]*invRmsC0) + (dy[0,1,0,0]*x[0,1,0,0]*invRmsC1)
    const double expectedDs0 = (0.1 * 1.0 * invRmsC0) + (0.2 * 3.0 * invRmsC1);
    // dscale[1] = (dy[0,0,0,1]*x[0,0,0,1]*invRmsC0) + (dy[0,1,0,1]*x[0,1,0,1]*invRmsC1)
    const double expectedDs1 = (0.1 * 2.0 * invRmsC0) + (0.2 * 4.0 * invRmsC1);

    EXPECT_NEAR(dscale.getHostValue(0, 0, 0, 0), expectedDs0, 1e-10);
    EXPECT_NEAR(dscale.getHostValue(0, 0, 0, 1), expectedDs1, 1e-10);

    // For C=0: x=[1, 2], dy=[0.1, 0.1], scale=[1.0, 0.5], N=2
    // sum(dy*w*x) = (0.1*1.0*1.0) + (0.1*0.5*2.0) = 0.1 + 0.1 = 0.2
    // m_C0 = 0.2 / 2 = 0.1
    const double mC0 = 0.1;
    const double invRmsC03 = invRmsC0 * invRmsC0 * invRmsC0;
    const double expDxC0W0 = (0.1 * 1.0 * invRmsC0) - (mC0 * 1.0 * invRmsC03);
    const double expDxC0W1 = (0.1 * 0.5 * invRmsC0) - (mC0 * 2.0 * invRmsC03);

    // For C=1: x=[3, 4], dy=[0.2, 0.2], scale=[1.0, 0.5], N=2
    // sum(dy*w*x) = (0.2*1.0*3.0) + (0.2*0.5*4.0) = 0.6 + 0.4 = 1.0
    // m_C1 = 1.0 / 2 = 0.5
    const double mC1 = 0.5;
    const double invRmsC13 = invRmsC1 * invRmsC1 * invRmsC1;
    const double expDxC1W0 = (0.2 * 1.0 * invRmsC1) - (mC1 * 3.0 * invRmsC13);
    const double expDxC1W1 = (0.2 * 0.5 * invRmsC1) - (mC1 * 4.0 * invRmsC13);

    EXPECT_NEAR(dx.getHostValue(0, 0, 0, 0), expDxC0W0, 1e-10);
    EXPECT_NEAR(dx.getHostValue(0, 0, 0, 1), expDxC0W1, 1e-10);
    EXPECT_NEAR(dx.getHostValue(0, 1, 0, 0), expDxC1W0, 1e-10);
    EXPECT_NEAR(dx.getHostValue(0, 1, 0, 1), expDxC1W1, 1e-10);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormBwdConstantInput)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};
    Tensor<double> dy(dims);
    Tensor<double> x(dims);
    Tensor<double> scale(dims);
    Tensor<double> dx(dims);
    Tensor<double> dscale(dims);
    Tensor<double> dbias(dims);
    Tensor<double> invRms({1, 1, 1, 1});

    const double valX = 3.0;
    const double valDy = 0.5;
    const double valScale = 2.0;
    const double epsilon = 1e-5;

    x.fillWithValue(valX);
    dy.fillWithValue(valDy);
    scale.fillWithValue(valScale);

    const double invRmsVal = 1.0 / std::sqrt(valX * valX + epsilon);
    invRms.fillWithValue(invRmsVal);

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale, &dbias);

    const double expectedDx = (valDy * valScale * invRmsVal)
                              - (valDy * valScale * valX * valX * std::pow(invRmsVal, 3));
    const double expectedDScale = valDy * valX * invRmsVal;
    const double expectedDBias = valDy;

    const double tolerance = 1e-10;
    for(size_t i = 0; i < 2; i++)
    {
        for(size_t j = 0; j < 2; j++)
        {
            EXPECT_NEAR(dx.getHostValue(0, 0, i, j), expectedDx, tolerance);
            EXPECT_NEAR(dscale.getHostValue(0, 0, i, j), expectedDScale, tolerance);
            EXPECT_NEAR(dbias.getHostValue(0, 0, i, j), expectedDBias, tolerance);
        }
    }
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormBwdScaleNormAxis2)
{
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> scaleDims = {1, 1, 2, 2};
    const std::vector<int64_t> invRmsDims = {2, 2, 1, 1};

    Tensor<double> dy(dims);
    Tensor<double> x(dims);
    Tensor<double> scale(scaleDims);
    Tensor<double> dx(dims);
    Tensor<double> dscale(scaleDims);
    Tensor<double> invRms(invRmsDims);

    // Setup:
    // C=0: x=1.0, dy=0.1 -> Mean(x^2)=1.0, invRms=1.0
    // C=1: x=2.0, dy=0.1 -> Mean(x^2)=4.0, invRms=0.5
    // scale=1.0 everywhere
    x.fillWithValue(1.0);
    for(int n = 0; n < 2; ++n)
    {
        for(int h = 0; h < 2; ++h)
        {
            for(int w = 0; w < 2; ++w)
            {
                x.setHostValue(2.0, n, 1, h, w);
            }
        }
    }

    dy.fillWithValue(0.1);
    scale.fillWithValue(1.0);

    const double rstdC0 = 1.0 / std::sqrt(1.0);
    const double rstdC1 = 1.0 / std::sqrt(4.0);
    for(int n = 0; n < 2; ++n)
    {
        invRms.setHostValue(rstdC0, n, 0, 0, 0);
        invRms.setHostValue(rstdC1, n, 1, 0, 0);
    }

    CpuFpReferenceRMSNorm::backward(dy, x, scale, invRms, dx, dscale);

    // Verify dscale
    // Each (h,w) position in dscale = sum_{n} (dy[n,0,h,w]*x[n,0,h,w]*rstd[n,0] + dy[n,1,h,w]*x[n,1,h,w]*rstd[n,1])
    // dscale = 2 * (0.1 * 1.0 * 1.0) + 2 * (0.1 * 2.0 * 0.5) = 0.2 + 0.2 = 0.4
    const double tolerance = 1e-10;
    const double expectedDs = 2.0 * (0.1 * 1.0 * rstdC0) + 2.0 * (0.1 * 2.0 * rstdC1);
    for(int h = 0; h < 2; ++h)
    {
        for(int w = 0; w < 2; ++w)
        {
            EXPECT_NEAR(dscale.getHostValue(0, 0, h, w), expectedDs, tolerance);
        }
    }

    // Verify dx
    // For C=0: dy=0.1, w=1.0, x=1.0, rstd=1.0, N_elements=4 (H*W)
    //   sum(dy*w*x) = 4 * (0.1 * 1.0 * 1.0) = 0.4
    //   dx = 1.0 * (0.1*1.0 - 1.0 * 0.4 * 1.0^2 / 4) = 1.0 * (0.1 - 0.1) = 0.0
    // For C=1: dy=0.1, w=1.0, x=2.0, rstd=0.5, N_elements=4
    //   sum(dy*w*x) = 4 * (0.1 * 1.0 * 2.0) = 0.8
    //   dx = 0.5 * (0.1*1.0 - 2.0 * 0.8 * 0.5^2 / 4) = 0.5 * (0.1 - 0.1) = 0.0
    for(int n = 0; n < 2; ++n)
    {
        for(int h = 0; h < 2; ++h)
        {
            for(int w = 0; w < 2; ++w)
            {
                EXPECT_NEAR(dx.getHostValue(n, 0, h, w), 0.0, tolerance);
                EXPECT_NEAR(dx.getHostValue(n, 1, h, w), 0.0, tolerance);
            }
        }
    }
}
