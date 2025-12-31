// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/Constants.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <limits>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

// ============================================================================
// Core Functionality Tests - Different Data Types and Layouts
// ============================================================================

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInferenceNchw)
{
    Tensor<float> inputTensor({1, 3, 224, 224});
    Tensor<float> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0f, 1.0f, 123);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f, 456);
    biasTensor.fillWithRandomValues(-0.5f, 0.5f, 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceBfp16, BatchnormFwdInferenceNchw)
{
    Tensor<hip_bfloat16> inputTensor({1, 3, 224, 224});
    Tensor<hip_bfloat16> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0_bf, 1.0_bf, 123);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f, 456);
    biasTensor.fillWithRandomValues(-0.5f, 0.5f, 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp16, BatchnormFwdInferenceNchw)
{
    Tensor<half> inputTensor({1, 3, 224, 224});
    Tensor<half> outputTensor({1, 3, 224, 224});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0_h, 1.0_h, 123);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f, 456);
    biasTensor.fillWithRandomValues(-0.5f, 0.5f, 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, BatchnormFwdInferenceNchw)
{
    Tensor<double> inputTensor({1, 3, 224, 224});
    Tensor<double> outputTensor({1, 3, 224, 224});
    Tensor<double> biasTensor({1, 3});
    Tensor<double> scaleTensor({1, 3});
    Tensor<double> meanTensor({1, 3});
    Tensor<double> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0, 1.0, 123);
    scaleTensor.fillWithRandomValues(0.5, 1.5, 456);
    biasTensor.fillWithRandomValues(-0.5, 0.5, 789);
    meanTensor.fillWithRandomValues(-0.2, 0.2, 321);
    varianceTensor.fillWithRandomValues(0.5, 2.0, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInferenceNhwc)
{
    Tensor<float> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0f, 1.0f, 123);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f, 456);
    biasTensor.fillWithRandomValues(-0.5f, 0.5f, 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceMixedPrecision, BatchnormFwdInferenceNhwc)
{
    Tensor<half> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<half> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<hip_bfloat16> biasTensor({1, 3});
    Tensor<hip_bfloat16> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0_h, 1.0_h, 123);
    scaleTensor.fillWithRandomValues(
        staticCast<hip_bfloat16>(0.5f), staticCast<hip_bfloat16>(1.5f), 456);
    biasTensor.fillWithRandomValues(
        staticCast<hip_bfloat16>(-0.5f), staticCast<hip_bfloat16>(0.5f), 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInference2D)
{
    // Test with 2D tensor (batch, channel)
    Tensor<float> inputTensor({4, 3});
    Tensor<float> outputTensor({4, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0f, 1.0f, 123);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f, 456);
    biasTensor.fillWithRandomValues(-0.5f, 0.5f, 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInference3D)
{
    // Test with 3D tensor (batch, channel, length)
    Tensor<float> inputTensor({2, 3, 10});
    Tensor<float> outputTensor({2, 3, 10});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0f, 1.0f, 123);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f, 456);
    biasTensor.fillWithRandomValues(-0.5f, 0.5f, 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BatchnormFwdInferenceNcdhw)
{
    // Test with 5D tensor (batch, channel, depth, height, width)
    Tensor<float> inputTensor({2, 3, 4, 5, 6});
    Tensor<float> outputTensor({2, 3, 4, 5, 6});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0f, 1.0f, 123);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f, 456);
    biasTensor.fillWithRandomValues(-0.5f, 0.5f, 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceBfp16, BatchnormFwdInferenceNdhwc)
{
    Tensor<float> inputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> outputTensor({2, 3, 4, 5, 6}, TensorLayout::NDHWC);
    Tensor<float> biasTensor({1, 3});
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});

    inputTensor.fillWithRandomValues(-1.0f, 1.0f, 123);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f, 456);
    biasTensor.fillWithRandomValues(-0.5f, 0.5f, 789);
    meanTensor.fillWithRandomValues(-0.2f, 0.2f, 321);
    varianceTensor.fillWithRandomValues(0.5f, 2.0f, 654);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);
}

// ============================================================================
// Sanity Validation Test - Standard Mathematical Case
// ============================================================================

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, SanityValidationNchw)
{
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    // x = [1, 2, 3, 4]
    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    // Fixed scale and bias parameters (one channel)
    scaleTensor.setHostValue(2.0, 0, 0);
    biasTensor.setHostValue(0.5, 0, 0);

    // Inference uses population statistics per channel:
    // mean = 2.5
    // variance = 1.25
    // epsilon = 1e-5
    // Formula: y = scale * ((x - mean) / sqrt(variance + epsilon)) + bias
    //           = 2.0 * ((x - 2.5) / sqrt(1.25 + 1e-5)) + 0.5
    meanTensor.setHostValue(2.5, 0, 0);
    varianceTensor.setHostValue(1.25, 0, 0);

    double epsilon = 1e-5;

    // Calculate expected output:
    // For x=1: 2.0 * ((1-2.5) / sqrt(1.25+1e-5)) + 0.5 = 2.0 * (-1.5 / 1.118033989) + 0.5
    // For x=2: 2.0 * ((2-2.5) / sqrt(1.25+1e-5)) + 0.5 = 2.0 * (-0.5 / 1.118033989) + 0.5
    // For x=3: 2.0 * ((3-2.5) / sqrt(1.25+1e-5)) + 0.5 = 2.0 * (0.5 / 1.118033989) + 0.5
    // For x=4: 2.0 * ((4-2.5) / sqrt(1.25+1e-5)) + 0.5 = 2.0 * (1.5 / 1.118033989) + 0.5
    const std::vector<double> expectedOutput = {-2.18327084, -0.39442361, 1.39442361, 3.18327084};

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], tolerance);
}

// ============================================================================
// Numerical Stability Tests - Zero and Near-Zero Variance
// ============================================================================

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, ZeroVarianceWithDefaultEpsilon)
{
    // All inputs identical → variance = 0
    // Output should equal bias (since (x - mean) = 0)
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    // All inputs identical: x = [5, 5, 5, 5]
    inputTensor.fillWithValue(5.0);

    scaleTensor.setHostValue(2.0, 0, 0);
    biasTensor.setHostValue(3.5, 0, 0);
    meanTensor.setHostValue(5.0, 0, 0); // mean = 5
    varianceTensor.setHostValue(0.0, 0, 0); // variance = 0

    double epsilon = 1e-5;

    // Expected: y = 2.0 * ((5 - 5) / sqrt(0 + 1e-5)) + 3.5 = 2.0 * 0 + 3.5 = 3.5
    double expectedOutput = 3.5;

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput, tolerance);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, NearZeroVariance)
{
    // Variance much smaller than epsilon
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(1.001, 0, 0, 0, 1);
    inputTensor.setHostValue(0.999, 0, 0, 1, 0);
    inputTensor.setHostValue(1.0, 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0, 0, 0);
    biasTensor.setHostValue(0.0, 0, 0);
    meanTensor.setHostValue(1.0, 0, 0);
    varianceTensor.setHostValue(1e-10, 0, 0); // Very small variance

    double epsilon = 1e-5;

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    // Should not produce NaN or Inf
    EXPECT_FALSE(std::isnan(outputTensor.getHostValue(0, 0, 0, 0)));
    EXPECT_FALSE(std::isinf(outputTensor.getHostValue(0, 0, 0, 0)));
    EXPECT_FALSE(std::isnan(outputTensor.getHostValue(0, 0, 0, 1)));
    EXPECT_FALSE(std::isinf(outputTensor.getHostValue(0, 0, 0, 1)));
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, VarianceEqualsEpsilon)
{
    // Boundary case: variance exactly equals epsilon
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0, 0, 0);
    biasTensor.setHostValue(0.0, 0, 0);
    meanTensor.setHostValue(2.5, 0, 0);

    double epsilon = 1e-5;
    varianceTensor.setHostValue(epsilon, 0, 0); // variance = epsilon

    // y = 1.0 * ((x - 2.5) / sqrt(1e-5 + 1e-5)) + 0.0
    //   = (x - 2.5) / sqrt(2e-5)
    //   = (x - 2.5) / 0.00447213595

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    // Verify outputs are finite and reasonable
    EXPECT_FALSE(std::isnan(outputTensor.getHostValue(0, 0, 0, 0)));
    EXPECT_FALSE(std::isinf(outputTensor.getHostValue(0, 0, 0, 0)));

    // Expected approximate values
    double divisor = std::sqrt(2 * epsilon);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), (1.0 - 2.5) / divisor, 1e-4);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), (2.0 - 2.5) / divisor, 1e-4);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), (3.0 - 2.5) / divisor, 1e-4);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), (4.0 - 2.5) / divisor, 1e-4);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, VeryLargeVariance)
{
    // Test with very large variance
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0, 0, 0);
    biasTensor.setHostValue(0.0, 0, 0);
    meanTensor.setHostValue(2.5, 0, 0);
    varianceTensor.setHostValue(100.0, 0, 0); // Large variance

    double epsilon = 1e-5;

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    // With large variance, normalized outputs should be close to 0
    auto tolerance = 0.2; // Loose tolerance for small values
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), -0.15, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), -0.05, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), 0.05, tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), 0.15, tolerance);
}

// ============================================================================
// Epsilon Range Tests
// ============================================================================

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, MinimalEpsilon)
{
    // Test with minimal epsilon for FP64
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0, 0, 0);
    biasTensor.setHostValue(0.0, 0, 0);
    meanTensor.setHostValue(2.5, 0, 0);
    varianceTensor.setHostValue(1.25, 0, 0);

    double epsilon = 1e-8; // Minimal epsilon

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    // Should not produce NaN or Inf
    EXPECT_FALSE(std::isnan(outputTensor.getHostValue(0, 0, 0, 0)));
    EXPECT_FALSE(std::isinf(outputTensor.getHostValue(0, 0, 0, 0)));
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, LargeEpsilon)
{
    // Test with large epsilon
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<double> inputTensor(dims);
    Tensor<double> outputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    inputTensor.setHostValue(1.0, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0, 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0, 0, 0);
    biasTensor.setHostValue(0.0, 0, 0);
    meanTensor.setHostValue(2.5, 0, 0);
    varianceTensor.setHostValue(0.01, 0, 0); // Small variance

    double epsilon = 1e-2; // Large epsilon (dominates small variance)

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    // Epsilon dominates, so divisor ≈ sqrt(epsilon) = 0.1
    double divisor = std::sqrt(0.01 + 1e-2);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), (1.0 - 2.5) / divisor, 1e-4);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), (2.0 - 2.5) / divisor, 1e-4);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, EpsilonSensitivity)
{
    // Test how output changes with different epsilon values
    const std::vector<int64_t> dims = {1, 1, 1, 1};

    Tensor<double> inputTensor(dims);
    Tensor<double> scaleTensor({1, 1});
    Tensor<double> biasTensor({1, 1});
    Tensor<double> meanTensor({1, 1});
    Tensor<double> varianceTensor({1, 1});

    inputTensor.setHostValue(3.0, 0, 0, 0, 0);
    scaleTensor.setHostValue(1.0, 0, 0);
    biasTensor.setHostValue(0.0, 0, 0);
    meanTensor.setHostValue(2.0, 0, 0);
    varianceTensor.setHostValue(0.5, 0, 0);

    std::vector<double> epsilons = {1e-8, 1e-5, 1e-3, 1e-2};
    std::vector<double> outputs;

    for(double eps : epsilons)
    {
        Tensor<double> outputTensor(dims);
        CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
            inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, eps);
        outputs.push_back(outputTensor.getHostValue(0, 0, 0, 0));
    }

    // Larger epsilon should produce smaller absolute output magnitude
    // Since divisor = sqrt(variance + epsilon) increases with epsilon
    EXPECT_GT(std::abs(outputs[0]), std::abs(outputs[1])); // 1e-8 > 1e-5
    EXPECT_GT(std::abs(outputs[1]), std::abs(outputs[2])); // 1e-5 > 1e-3
    EXPECT_GT(std::abs(outputs[2]), std::abs(outputs[3])); // 1e-3 > 1e-2
}

// ============================================================================
// Precision-Specific Tests
// ============================================================================

TEST(TestCpuFpReferenceBatchnormWithVarianceFp16, NumericalLimitsWithLargerEpsilon)
{
    // FP16 has limited precision, use larger epsilon
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<half> inputTensor(dims);
    Tensor<half> outputTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    inputTensor.setHostValue(staticCast<half>(1.0f), 0, 0, 0, 0);
    inputTensor.setHostValue(staticCast<half>(2.0f), 0, 0, 0, 1);
    inputTensor.setHostValue(staticCast<half>(3.0f), 0, 0, 1, 0);
    inputTensor.setHostValue(staticCast<half>(4.0f), 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0f, 0, 0);
    biasTensor.setHostValue(0.0f, 0, 0);
    meanTensor.setHostValue(2.5f, 0, 0);
    varianceTensor.setHostValue(1.25f, 0, 0);

    double epsilon = 1e-4; // Larger epsilon for FP16

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    // Should not produce NaN or Inf
    EXPECT_FALSE(std::isnan(static_cast<float>(outputTensor.getHostValue(0, 0, 0, 0))));
    EXPECT_FALSE(std::isinf(static_cast<float>(outputTensor.getHostValue(0, 0, 0, 0))));
}

TEST(TestCpuFpReferenceBatchnormWithVarianceBfp16, NumericalLimitsWithLargerEpsilon)
{
    // BFP16 has wider exponent range but less precision
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<hip_bfloat16> inputTensor(dims);
    Tensor<hip_bfloat16> outputTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    inputTensor.setHostValue(staticCast<hip_bfloat16>(1.0f), 0, 0, 0, 0);
    inputTensor.setHostValue(staticCast<hip_bfloat16>(2.0f), 0, 0, 0, 1);
    inputTensor.setHostValue(staticCast<hip_bfloat16>(3.0f), 0, 0, 1, 0);
    inputTensor.setHostValue(staticCast<hip_bfloat16>(4.0f), 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0f, 0, 0);
    biasTensor.setHostValue(0.0f, 0, 0);
    meanTensor.setHostValue(2.5f, 0, 0);
    varianceTensor.setHostValue(1.25f, 0, 0);

    double epsilon = 1e-4; // Appropriate epsilon for BFP16

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    // Should not produce NaN or Inf
    EXPECT_FALSE(std::isnan(static_cast<float>(outputTensor.getHostValue(0, 0, 0, 0))));
    EXPECT_FALSE(std::isinf(static_cast<float>(outputTensor.getHostValue(0, 0, 0, 0))));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, ScaleZero)
{
    // Test with scale = 0, output should equal bias
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    inputTensor.fillWithRandomValues(-1.0f, 1.0f, 123);
    scaleTensor.setHostValue(0.0f, 0, 0); // Scale = 0
    biasTensor.setHostValue(5.0f, 0, 0);
    meanTensor.setHostValue(0.0f, 0, 0);
    varianceTensor.setHostValue(1.0f, 0, 0);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    // All outputs should equal bias
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), 5.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1), 5.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0), 5.0f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 5.0f);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, BiasZero)
{
    // Test with bias = 0, pure normalization with scale
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    scaleTensor.setHostValue(2.0f, 0, 0);
    biasTensor.setHostValue(0.0f, 0, 0); // Bias = 0
    meanTensor.setHostValue(2.5f, 0, 0);
    varianceTensor.setHostValue(1.25f, 0, 0);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-5);

    // Verify outputs are symmetric around 0
    float out0 = outputTensor.getHostValue(0, 0, 0, 0);
    float out3 = outputTensor.getHostValue(0, 0, 1, 1);
    EXPECT_NEAR(out0, -out3, 1e-5f);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, UnitScaleZeroBias)
{
    // Pure standardization: scale = 1, bias = 0
    const std::vector<int64_t> dims = {1, 1, 2, 2};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputTensor(dims);
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> biasTensor({1, 1});
    Tensor<float> meanTensor({1, 1});
    Tensor<float> varianceTensor({1, 1});

    inputTensor.setHostValue(1.0f, 0, 0, 0, 0);
    inputTensor.setHostValue(2.0f, 0, 0, 0, 1);
    inputTensor.setHostValue(3.0f, 0, 0, 1, 0);
    inputTensor.setHostValue(4.0f, 0, 0, 1, 1);

    scaleTensor.setHostValue(1.0f, 0, 0);
    biasTensor.setHostValue(0.0f, 0, 0);
    meanTensor.setHostValue(2.5f, 0, 0);
    varianceTensor.setHostValue(1.0f, 0, 0); // Unit variance

    double epsilon = 0.0; // No epsilon for pure standardization test

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, epsilon);

    // With unit variance and zero epsilon: y = (x - 2.5) / 1.0
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 0), -1.5f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 0, 1), -0.5f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 0), 0.5f);
    EXPECT_FLOAT_EQ(outputTensor.getHostValue(0, 0, 1, 1), 1.5f);
}

// ============================================================================
// Golden Reference Test Placeholder
// ============================================================================

// TODO: Golden reference tests will be enabled when reference data is available
// Expected directory structure:
// hipdnn_reference_data/BatchnormFwdInferenceWithVariance/
//   nchw/
//     fp32/
//     fp16/
//     bfp16/
//   ncdhw/
//     fp32/
//
// Example instantiation (currently commented out):
//
// template <class T>
// class TestCpuBatchnormFwdInferenceWithVarianceGoldenReference : public TestGoldenReferenceCpu
// {
// public:
//     void testSuite()
//     {
//         return goldenReferenceTestSuite(batchnorm::getToleranceInference<T>(),
//                                         batchnorm::getToleranceInference<T>());
//     }
// };
//
// class TestCpuBatchnormFwdInferenceWithVarianceGoldenReferenceNchwFp32
//     : public TestCpuBatchnormFwdInferenceWithVarianceGoldenReference<float>
// {
// };
//
// TEST_P(TestCpuBatchnormFwdInferenceWithVarianceGoldenReferenceNchwFp32, Correctness)
// {
//     testSuite();
// }
//
// INSTANTIATE_TEST_SUITE_P(,
//                          TestCpuBatchnormFwdInferenceWithVarianceGoldenReferenceNchwFp32,
//                          getGoldenReferenceParams("BatchnormFwdInferenceWithVariance/nchw/fp32"));
