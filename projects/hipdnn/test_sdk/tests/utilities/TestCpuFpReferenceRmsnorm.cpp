// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRmsnorm.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

template <typename T1, typename T2>
struct TypePair
{
    using First = T1;
    using Second = T2;
};

using TypesRmsnormFwdNchw = ::testing::Types<TypePair<float, float>,
                                             TypePair<half, float>,
                                             TypePair<bfloat16, float>,
                                             TypePair<double, double>>;

template <class T>
class CpuFpReferenceRmsnormFwdNchw : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceRmsnormFwdNchw, TypesRmsnormFwdNchw, );

TYPED_TEST(CpuFpReferenceRmsnormFwdNchw, RmsnormFwdNchw)
{
    Tensor<typename TypeParam::First> inputTensor({1, 3, 224, 224});
    Tensor<typename TypeParam::First> outputTensor({1, 3, 224, 224});
    Tensor<typename TypeParam::Second> scaleTensor({1, 3});

    inputTensor.fillWithValue(static_cast<typename TypeParam::First>(1.0));
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(static_cast<typename TypeParam::Second>(1.0), 0, i);
    }

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

using TypesRmsnormFwdNhwc = ::testing::Types<TypePair<float, float>, TypePair<half, bfloat16>>;

template <class T>
class CpuFpReferenceRmsnormFwdNhwc : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceRmsnormFwdNhwc, TypesRmsnormFwdNhwc, );

TYPED_TEST(CpuFpReferenceRmsnormFwdNhwc, RmsnormFwdNhwc)
{
    Tensor<typename TypeParam::First> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<typename TypeParam::First> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<typename TypeParam::Second> scaleTensor({1, 3});

    inputTensor.fillWithValue(static_cast<typename TypeParam::First>(1.0));
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(static_cast<typename TypeParam::Second>(1.0), 0, i);
    }

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRmsnormFp64, RmsnormFwdSanityValidationNchw)
{
    // RMSNorm: y = x / sqrt(mean(x^2) + epsilon) * scale
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});

    // x = [1, 2, 3, 4]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0, 0, 0);

    double epsilon = 1e-5;

    // mean(x^2) = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    // rms = sqrt(7.5 + 1e-5) = 2.73861278753...
    // inv_rms = 1 / rms = 0.36514837167...
    // y = x * inv_rms * scale
    // y[0] = 1.0 * 0.36514837167 = 0.36514837167
    // y[1] = 2.0 * 0.36514837167 = 0.73029674335
    // y[2] = 3.0 * 0.36514837167 = 1.09544511502
    // y[3] = 4.0 * 0.36514837167 = 1.46059348669

    double invRmsExpected = 1.0 / std::sqrt(7.5 + epsilon);
    const std::vector<double> expectedOutput
        = {1.0 * invRmsExpected, 2.0 * invRmsExpected, 3.0 * invRmsExpected, 4.0 * invRmsExpected};

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, epsilon);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], tolerance);
}

TEST(TestCpuFpReferenceRmsnormFp64, RmsnormFwdWithInvRms)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<float> invRmsTensor({1, 1});

    // x = [1, 2, 3, 4]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    scaleTensor.setHostValue(2.0, 0, 0);

    double epsilon = 1e-5;

    double invRmsExpected = 1.0 / std::sqrt(7.5 + epsilon);

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, epsilon, &invRmsTensor);

    auto tolerance = 1e-5;

    EXPECT_NEAR(static_cast<double>(invRmsTensor.getHostValue(0, 0)), invRmsExpected, tolerance);

    // y = x * inv_rms * scale (scale = 2.0)
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 1.0 * invRmsExpected * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), 2.0 * invRmsExpected * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), 3.0 * invRmsExpected * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), 4.0 * invRmsExpected * 2.0, tolerance);
}

TEST(TestCpuFpReferenceRmsnormFp64, RmsnormFwdMultipleChannels)
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

    double epsilon = 1e-5;

    // Channel 0: mean(x^2) = (1+4)/2 = 2.5, inv_rms = 1/sqrt(2.5+eps)
    double invRms0 = 1.0 / std::sqrt(2.5 + epsilon);
    // Channel 1: mean(x^2) = (9+16)/2 = 12.5, inv_rms = 1/sqrt(12.5+eps)
    double invRms1 = 1.0 / std::sqrt(12.5 + epsilon);

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, epsilon);

    auto tolerance = 1e-6;

    // Channel 0 (scale=1.0): y = x * inv_rms * 1.0
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), 1.0 * invRms0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), 2.0 * invRms0, tolerance);

    // Channel 1 (scale=2.0): y = x * inv_rms * 2.0
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 0), 3.0 * invRms1 * 2.0, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 1, 0, 1), 4.0 * invRms1 * 2.0, tolerance);
}

TEST(TestCpuFpReferenceRmsnormFp32, RmsnormFwd2D)
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

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRmsnormFp32, RmsnormFwd3D)
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

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRmsnormFp32, RmsnormFwdNcdhw)
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

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRmsnormFp32, RmsnormFwdNdhwc)
{
    Tensor<float> inputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> outputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> scaleTensor({1, 3});

    inputTensor.fillWithValue(1.5f);
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(2.0f, 0, i);
    }

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceRmsnormFp64, RmsnormFwdConstantInput)
{
    // When all inputs are the same constant c, mean(x^2) = c^2
    // inv_rms = 1/sqrt(c^2+eps) ~ 1/c
    // y = x * inv_rms * scale = c * (1/c) * scale = scale
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});

    double c = 3.0;
    inputTensor.fillWithValue(c);
    scaleTensor.setHostValue(2.0, 0, 0);

    double epsilon = 1e-5;

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputTensor, epsilon);

    double invRms = 1.0 / std::sqrt(c * c + epsilon);
    double expectedY = c * invRms * 2.0;

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedY, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedY, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedY, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedY, tolerance);
}

TEST(TestCpuFpReferenceRmsnorm, RmsnormFwdWithBias)
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

    double epsilon = 0.0; // zero epsilon so inv_rms = 1

    Tensor<float>* noInvRms = nullptr;
    CpuFpReferenceRmsnorm::forward(
        inputTensor, scaleTensor, outputTensor, epsilon, noInvRms, &biasTensor);

    // y = x * invRms * scale + bias = 1 * 1 * scale + bias
    float expectedC0 = 2.0f + 0.5f; // 2.5
    float expectedC1 = 3.0f + -1.0f; // 2.0

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

TEST(TestCpuFpReferenceRmsnorm, RmsnormFwdBiasIsOptional)
{
    // Passing nullptr bias should give the same result as no-bias call
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputNoBias(dims);
    Tensor<float> outputNullBias(dims);
    Tensor<float> scaleTensor({1, 1});

    inputTensor.fillWithValue(2.0f);
    scaleTensor.setHostValue(1.5f, 0, 0);
    double epsilon = 1e-5;

    CpuFpReferenceRmsnorm::forward(inputTensor, scaleTensor, outputNoBias, epsilon);
    Tensor<float>* noInvRms2 = nullptr;
    Tensor<float>* noBias = nullptr;
    CpuFpReferenceRmsnorm::forward(
        inputTensor, scaleTensor, outputNullBias, epsilon, noInvRms2, noBias);

    auto tolerance = 1e-6f;
    EXPECT_NEAR(
        outputNoBias.getHostValue(0, 0, 0, 0), outputNullBias.getHostValue(0, 0, 0, 0), tolerance);
    EXPECT_NEAR(
        outputNoBias.getHostValue(0, 0, 1, 1), outputNullBias.getHostValue(0, 0, 1, 1), tolerance);
}
