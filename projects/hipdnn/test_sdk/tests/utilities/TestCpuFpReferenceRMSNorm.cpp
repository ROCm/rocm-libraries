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
    Tensor<typename TypeParam::Second> scaleTensor({1, 3});

    inputTensor.fillWithValue(safeTestTypeCast<typename TypeParam::First>(1.0));
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(safeTestTypeCast<typename TypeParam::Second>(1.0), 0, i);
    }

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
    Tensor<typename TypeParam::Second> scaleTensor({1, 3});

    inputTensor.fillWithValue(safeTestTypeCast<typename TypeParam::First>(1.0));
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(safeTestTypeCast<typename TypeParam::Second>(1.0), 0, i);
    }

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdSanityValidationNchw)
{
    // RMSNorm: y = x / sqrt(mean_C(x^2) + epsilon) * scale
    // Reduction is over the channel dimension; each (batch, spatial) position has its own RMS.
    const std::vector<int64_t> dims = {1, 4, 1, 1};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 4});

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
    Tensor<double> scaleTensor({1, 4});
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

TEST(TestCpuFpReferenceRMSNormFp64, RMSNormFwdMultipleChannels)
{
    const std::vector<int64_t> dims = {1, 2, 1, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 2});

    // Channel 0: x = [1, 2]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);

    // Channel 1: x = [3, 4]
    inputTensor.setHostValue(3.0, 0, 1, 0, 0);
    inputTensor.setHostValue(4.0, 0, 1, 0, 1);

    scaleTensor.setHostValue(1.0, 0, 0);
    scaleTensor.setHostValue(2.0, 0, 1);

    const double epsilon = 1e-5;

    // Position (0,0,0): channels [1,3], mean(x^2) = (1+9)/2 = 5
    const double invRmsPos0 = 1.0 / std::sqrt(5.0 + epsilon);
    // Position (0,0,1): channels [2,4], mean(x^2) = (4+16)/2 = 10
    const double invRmsPos1 = 1.0 / std::sqrt(10.0 + epsilon);

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, epsilon);

    auto tolerance = 1e-6;

    // Channel 0 (scale=1.0): y = x * inv_rms * 1.0
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 1.0 * invRmsPos0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), 2.0 * invRmsPos1, tolerance);

    // Channel 1 (scale=2.0): y = x * inv_rms * 2.0
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 0), 3.0 * invRmsPos0 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 1), 4.0 * invRmsPos1 * 2.0, tolerance);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormFwd2D)
{
    // Test with 2D tensor (batch, channel)
    Tensor<float> inputTensor({4, 3});
    Tensor<float> outputTensor({4, 3});
    Tensor<float> scaleTensor({1, 3});

    inputTensor.fillWithValue(1.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
    }

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormFwd3D)
{
    // Test with 3D tensor (batch, channel, length)
    Tensor<float> inputTensor({2, 3, 10});
    Tensor<float> outputTensor({2, 3, 10});
    Tensor<float> scaleTensor({1, 3});

    inputTensor.fillWithValue(2.0f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(2.0f, 0, i);
    }

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormFwdNcdhw)
{
    // Test with 5D tensor (batch, channel, depth, height, width)
    Tensor<float> inputTensor({2, 3, 4, 5, 6});
    Tensor<float> outputTensor({2, 3, 4, 5, 6});
    Tensor<float> scaleTensor({1, 3});

    inputTensor.fillWithValue(1.5f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.0f, 0, i);
    }

    CpuFpReferenceRMSNorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRMSNormFp32, RMSNormFwdNdhwc)
{
    Tensor<float> inputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> outputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> scaleTensor({1, 3});

    inputTensor.fillWithValue(1.5f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(2.0f, 0, i);
    }

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
    Tensor<double> scaleTensor({1, 1});

    const double c = 3.0;
    inputTensor.fillWithValue(c);
    scaleTensor.setHostValue(2.0, 0, 0);

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
    // bias is added per-channel after scale multiplication:
    // y = x / rms * scale + bias
    const std::vector<int64_t> dims = {1, 2, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputTensor(dims);
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> biasTensor({1, 2});

    // Constant input: all 1s so rms = 1 + eps ~ 1
    inputTensor.fillWithValue(1.0f);
    scaleTensor.setHostValue(2.0f, 0, 0); // channel 0: scale=2, bias=0.5
    scaleTensor.setHostValue(3.0f, 0, 1); // channel 1: scale=3, bias=-1.0
    biasTensor.setHostValue(0.5f, 0, 0);
    biasTensor.setHostValue(-1.0f, 0, 1);

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

TEST(TestCpuFpReferenceRMSNorm, RMSNormFwdBiasIsOptional)
{
    // Passing nullptr bias should give the same result as no-bias call
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputNoBias(dims);
    Tensor<float> outputNullBias(dims);
    Tensor<float> scaleTensor({1, 1});

    inputTensor.fillWithValue(2.0f);
    scaleTensor.setHostValue(1.5f, 0, 0);
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
