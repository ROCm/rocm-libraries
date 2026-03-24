// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

// ============================================================================
// Type Definitions
// ============================================================================

using DataTypes = ::testing::Types<float, half, bfloat16, double>;

// ============================================================================
// Test Fixture
// ============================================================================

template <typename T>
class CpuFpReferenceBatchnormWithVariance : public ::testing::Test
{
protected:
    // Helper to get parameter type for mixed precision
    using ParamType
        = std::conditional_t<std::is_same_v<T, half> || std::is_same_v<T, bfloat16>, float, T>;
};

TYPED_TEST_SUITE(CpuFpReferenceBatchnormWithVariance, DataTypes, );

// ============================================================================
// Section 1: Basic Functionality Tests
// ============================================================================
// Tests that batchnorm inference with variance works correctly across
// different tensor shapes, layouts, and data types.

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, BatchnormFwdInferenceWithVarianceNchw)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    const Tensor<DataType> inputTensor({1, 3, 224, 224});
    Tensor<DataType> outputTensor({1, 3, 224, 224});
    const Tensor<ParamType> biasTensor({1, 3});
    const Tensor<ParamType> scaleTensor({1, 3});
    const Tensor<ParamType> meanTensor({1, 3});
    const Tensor<ParamType> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, BatchnormFwdInferenceWithVarianceNhwc)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    const Tensor<DataType> inputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    Tensor<DataType> outputTensor({6, 3, 32, 32}, TensorLayout::NHWC);
    const Tensor<ParamType> biasTensor({1, 3});
    const Tensor<ParamType> scaleTensor({1, 3});
    const Tensor<ParamType> meanTensor({1, 3});
    const Tensor<ParamType> varianceTensor({1, 3});

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, BatchnormFwdInferenceWithVariance2D)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    Tensor<DataType> inputTensor({4, 3});
    Tensor<DataType> outputTensor({4, 3});
    Tensor<ParamType> scaleTensor({1, 3});
    Tensor<ParamType> biasTensor({1, 3});
    Tensor<ParamType> meanTensor({1, 3});
    Tensor<ParamType> varianceTensor({1, 3});

    inputTensor.fillWithValue(safeTestTypeCast<DataType>(1.0));
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, i);
        biasTensor.setHostValue(safeTestTypeCast<ParamType>(0.0), 0, i);
        meanTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, i);
        varianceTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, BatchnormFwdInferenceWithVariance3D)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    Tensor<DataType> inputTensor({2, 3, 10});
    Tensor<DataType> outputTensor({2, 3, 10});
    Tensor<ParamType> scaleTensor({1, 3});
    Tensor<ParamType> biasTensor({1, 3});
    Tensor<ParamType> meanTensor({1, 3});
    Tensor<ParamType> varianceTensor({1, 3});

    inputTensor.fillWithValue(safeTestTypeCast<DataType>(2.0));
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(safeTestTypeCast<ParamType>(2.0), 0, i);
        biasTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, i);
        meanTensor.setHostValue(safeTestTypeCast<ParamType>(2.0), 0, i);
        varianceTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, BatchnormFwdInferenceWithVarianceNcdhw)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    Tensor<DataType> inputTensor({2, 3, 4, 5, 6});
    Tensor<DataType> outputTensor({2, 3, 4, 5, 6});
    Tensor<ParamType> scaleTensor({1, 3});
    Tensor<ParamType> biasTensor({1, 3});
    Tensor<ParamType> meanTensor({1, 3});
    Tensor<ParamType> varianceTensor({1, 3});

    inputTensor.fillWithValue(safeTestTypeCast<DataType>(1.5));
    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, i);
        biasTensor.setHostValue(safeTestTypeCast<ParamType>(0.5), 0, i);
        meanTensor.setHostValue(safeTestTypeCast<ParamType>(1.5), 0, i);
        varianceTensor.setHostValue(safeTestTypeCast<ParamType>(0.5), 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);
}

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, ZeroVarianceHandling)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    const std::vector<int64_t> dims = {1, 1, 2, 2};
    Tensor<DataType> inputTensor(dims);
    Tensor<DataType> outputTensor(dims);
    Tensor<ParamType> scaleTensor({1, 1});
    Tensor<ParamType> biasTensor({1, 1});
    Tensor<ParamType> meanTensor({1, 1});
    Tensor<ParamType> varianceTensor({1, 1});

    // All input values identical
    inputTensor.setHostValue(safeTestTypeCast<DataType>(3.0), 0, 0, 0, 0);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(3.0), 0, 0, 0, 1);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(3.0), 0, 0, 1, 0);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(3.0), 0, 0, 1, 1);

    scaleTensor.setHostValue(safeTestTypeCast<ParamType>(2.0), 0, 0);
    biasTensor.setHostValue(safeTestTypeCast<ParamType>(0.5), 0, 0);
    meanTensor.setHostValue(safeTestTypeCast<ParamType>(3.0), 0, 0);
    varianceTensor.setHostValue(safeTestTypeCast<ParamType>(0.0), 0, 0); // Zero variance

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);

    // When variance is 0, inv_variance = 1/sqrt(epsilon)
    // For all elements: y = 2.0 * (3.0 - 3.0) * (1/sqrt(epsilon)) + 0.5 = 0.5
    const double tolerance
        = std::is_same_v<DataType, half> || std::is_same_v<DataType, bfloat16> ? 1e-3 : 1e-5;

    EXPECT_NEAR(static_cast<double>(outputTensor.getHostValue(0, 0, 0, 0)), 0.5, tolerance);
    EXPECT_NEAR(static_cast<double>(outputTensor.getHostValue(0, 0, 0, 1)), 0.5, tolerance);
    EXPECT_NEAR(static_cast<double>(outputTensor.getHostValue(0, 0, 1, 0)), 0.5, tolerance);
    EXPECT_NEAR(static_cast<double>(outputTensor.getHostValue(0, 0, 1, 1)), 0.5, tolerance);
}

// ============================================================================
// Section 2: Epsilon Variation Tests
// ============================================================================
// Tests that verify epsilon parameter is handled correctly across all data types.
// Ensures numerical stability with different epsilon values.

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, CustomEpsilonSmall)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    const std::vector<int64_t> dims = {1, 1, 2, 2};
    Tensor<DataType> inputTensor(dims);
    Tensor<DataType> outputTensor(dims);
    Tensor<ParamType> scaleTensor({1, 1});
    Tensor<ParamType> biasTensor({1, 1});
    Tensor<ParamType> meanTensor({1, 1});
    Tensor<ParamType> varianceTensor({1, 1});

    // x = [1, 2, 3, 4]
    inputTensor.setHostValue(safeTestTypeCast<DataType>(1.0), 0, 0, 0, 0);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(2.0), 0, 0, 0, 1);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(3.0), 0, 0, 1, 0);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(4.0), 0, 0, 1, 1);

    scaleTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, 0);
    biasTensor.setHostValue(safeTestTypeCast<ParamType>(0.0), 0, 0);
    meanTensor.setHostValue(safeTestTypeCast<ParamType>(2.5), 0, 0);
    varianceTensor.setHostValue(safeTestTypeCast<ParamType>(1.25), 0, 0);

    // With epsilon=1e-10: inv_var = 1/sqrt(1.25 + 1e-10) ≈ 0.894427191
    const double smallEpsilon = 1e-10;
    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(inputTensor,
                                                      scaleTensor,
                                                      biasTensor,
                                                      meanTensor,
                                                      varianceTensor,
                                                      outputTensor,
                                                      smallEpsilon);

    // y = scale * (x - mean) * inv_var + bias = 1.0 * (x - 2.5) * 0.894427191 + 0.0
    const std::vector<double> expectedOutput = {
        -1.3416407865, // 1.0 * (1 - 2.5) * 0.894427191 + 0.0
        -0.4472135955, // 1.0 * (2 - 2.5) * 0.894427191 + 0.0
        0.4472135955, // 1.0 * (3 - 2.5) * 0.894427191 + 0.0
        1.3416407865 // 1.0 * (4 - 2.5) * 0.894427191 + 0.0
    };

    const double tolerance
        = std::is_same_v<DataType, half> || std::is_same_v<DataType, bfloat16> ? 1e-2 : 1e-6;

    EXPECT_NEAR(
        static_cast<double>(outputTensor.getHostValue(0, 0, 0, 0)), expectedOutput[0], tolerance);
    EXPECT_NEAR(
        static_cast<double>(outputTensor.getHostValue(0, 0, 0, 1)), expectedOutput[1], tolerance);
    EXPECT_NEAR(
        static_cast<double>(outputTensor.getHostValue(0, 0, 1, 0)), expectedOutput[2], tolerance);
    EXPECT_NEAR(
        static_cast<double>(outputTensor.getHostValue(0, 0, 1, 1)), expectedOutput[3], tolerance);
}

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, CustomEpsilonLarge)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    const std::vector<int64_t> dims = {1, 1, 2, 2};
    Tensor<DataType> inputTensor(dims);
    Tensor<DataType> outputTensor(dims);
    Tensor<ParamType> scaleTensor({1, 1});
    Tensor<ParamType> biasTensor({1, 1});
    Tensor<ParamType> meanTensor({1, 1});
    Tensor<ParamType> varianceTensor({1, 1});

    inputTensor.setHostValue(safeTestTypeCast<DataType>(1.0), 0, 0, 0, 0);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(2.0), 0, 0, 0, 1);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(3.0), 0, 0, 1, 0);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(4.0), 0, 0, 1, 1);

    scaleTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, 0);
    biasTensor.setHostValue(safeTestTypeCast<ParamType>(0.0), 0, 0);
    meanTensor.setHostValue(safeTestTypeCast<ParamType>(2.5), 0, 0);
    varianceTensor.setHostValue(safeTestTypeCast<ParamType>(1.25), 0, 0);

    // With epsilon=0.1: inv_var = 1/sqrt(1.25 + 0.1) = 1/sqrt(1.35) ≈ 0.860662386
    const double largeEpsilon = 0.1;
    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(inputTensor,
                                                      scaleTensor,
                                                      biasTensor,
                                                      meanTensor,
                                                      varianceTensor,
                                                      outputTensor,
                                                      largeEpsilon);

    // y = scale * (x - mean) * inv_var + bias = 1.0 * (x - 2.5) * 0.860662386 + 0.0
    const std::vector<double> expectedOutput = {
        -1.2909935825, // 1.0 * (1 - 2.5) * 0.860662386 + 0.0
        -0.4303311928, // 1.0 * (2 - 2.5) * 0.860662386 + 0.0
        0.4303311928, // 1.0 * (3 - 2.5) * 0.860662386 + 0.0
        1.2909935825 // 1.0 * (4 - 2.5) * 0.860662386 + 0.0
    };

    const double tolerance
        = std::is_same_v<DataType, half> || std::is_same_v<DataType, bfloat16> ? 1e-2 : 1e-6;

    EXPECT_NEAR(
        static_cast<double>(outputTensor.getHostValue(0, 0, 0, 0)), expectedOutput[0], tolerance);
    EXPECT_NEAR(
        static_cast<double>(outputTensor.getHostValue(0, 0, 0, 1)), expectedOutput[1], tolerance);
    EXPECT_NEAR(
        static_cast<double>(outputTensor.getHostValue(0, 0, 1, 0)), expectedOutput[2], tolerance);
    EXPECT_NEAR(
        static_cast<double>(outputTensor.getHostValue(0, 0, 1, 1)), expectedOutput[3], tolerance);
}

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, EpsilonProducesDifferentResults)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    const std::vector<int64_t> dims = {2, 3, 4, 4};
    Tensor<DataType> inputTensor(dims);
    Tensor<DataType> outputSmallEps(dims);
    Tensor<DataType> outputLargeEps(dims);
    Tensor<ParamType> scaleTensor({1, 3});
    Tensor<ParamType> biasTensor({1, 3});
    Tensor<ParamType> meanTensor({1, 3});
    Tensor<ParamType> varianceTensor({1, 3});

    auto min = safeTestTypeCast<DataType>(-5.0f);
    auto max = safeTestTypeCast<DataType>(5.0f);
    inputTensor.fillWithRandomValues(min, max, 42);

    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, i);
        biasTensor.setHostValue(safeTestTypeCast<ParamType>(0.0), 0, i);
        meanTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, i);
        varianceTensor.setHostValue(safeTestTypeCast<ParamType>(0.5), 0, i);
    }

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputSmallEps, 1e-10);

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputLargeEps, 0.1);

    // Verify that different epsilon values produce measurably different outputs
    bool foundDifference = false;

    for(int b = 0; b < 2 && !foundDifference; b++)
    {
        for(int c = 0; c < 3 && !foundDifference; c++)
        {
            for(int h = 0; h < 4 && !foundDifference; h++)
            {
                for(int w = 0; w < 4 && !foundDifference; w++)
                {
                    auto valSmall = static_cast<double>(outputSmallEps.getHostValue(b, c, h, w));
                    auto valLarge = static_cast<double>(outputLargeEps.getHostValue(b, c, h, w));
                    if(hipdnn_data_sdk::types::abs(valSmall - valLarge) > 0.001)
                    {
                        foundDifference = true;
                    }
                }
            }
        }
    }

    EXPECT_TRUE(foundDifference) << "Epsilon should produce measurably different results";
}

TYPED_TEST(CpuFpReferenceBatchnormWithVariance, NoNaNOrInfProduced)
{
    using DataType = TypeParam;
    using ParamType = typename CpuFpReferenceBatchnormWithVariance<DataType>::ParamType;

    const std::vector<int64_t> dims = {1, 1, 2, 2};
    Tensor<DataType> inputTensor(dims);
    Tensor<DataType> outputTensor(dims);
    Tensor<ParamType> scaleTensor({1, 1});
    Tensor<ParamType> biasTensor({1, 1});
    Tensor<ParamType> meanTensor({1, 1});
    Tensor<ParamType> varianceTensor({1, 1});

    inputTensor.setHostValue(safeTestTypeCast<DataType>(1.0), 0, 0, 0, 0);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(2.0), 0, 0, 0, 1);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(3.0), 0, 0, 1, 0);
    inputTensor.setHostValue(safeTestTypeCast<DataType>(4.0), 0, 0, 1, 1);

    scaleTensor.setHostValue(safeTestTypeCast<ParamType>(1.0), 0, 0);
    biasTensor.setHostValue(safeTestTypeCast<ParamType>(0.0), 0, 0);
    meanTensor.setHostValue(safeTestTypeCast<ParamType>(2.5), 0, 0);
    varianceTensor.setHostValue(safeTestTypeCast<ParamType>(1.25), 0, 0);

    // Test with very small epsilon to verify no numerical instability
    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor, 1e-10);

    // Verify all outputs are finite (no NaN or Inf)
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            auto val = static_cast<double>(outputTensor.getHostValue(0, 0, i, j));
            EXPECT_TRUE(hipdnn_data_sdk::types::isfinite(val))
                << "NaN/Inf detected at [" << i << "," << j << "]";
        }
    }
}

// ============================================================================
// Section 3: Specialized Validation Tests
// ============================================================================
// These tests validate specific scenarios with exact numerical expectations
// and are NOT parameterized by data type because:
// - SanityNchw: Uses FP64 golden reference with hardcoded expected values
// - CompareVarianceVsInvVariance: Validates API equivalence (focused on FP32)

TEST(TestCpuFpReferenceBatchnormWithVarianceFp64, BatchnormFwdInferenceWithVarianceSanityNchw)
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

    // fixed scale and bias parameters (one channel)
    scaleTensor.setHostValue(2.0, 0, 0);
    biasTensor.setHostValue(0.5, 0, 0);

    // inference uses population statistics per channel:
    // mean = (1+2+3+4)/4 = 2.5
    // variance = [(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2] / 4 = 5.0 / 4 = 1.25
    // inv_variance = 1 / sqrt(1.25 + 1e-5) = 0.894423613312618
    //
    // With variance input, we compute inv_variance from variance internally
    meanTensor.setHostValue(2.5, 0, 0);
    varianceTensor.setHostValue(1.25, 0, 0);

    // output is calculated via a pointwise linear transform on x:
    // y = scale * (x - mean) * inv_variance + bias = 2 * (x - 2.5) * inv_variance + 0.5
    const std::vector<double> expectedOutput = {-2.18327084, -0.39442361, 1.39442361, 3.18327084};

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputTensor);

    auto tolerance = 1e-6;

    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 0), expectedOutput[0], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 0, 1), expectedOutput[1], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 0), expectedOutput[2], tolerance);
    EXPECT_NEAR(outputTensor.getHostValue(0, 0, 1, 1), expectedOutput[3], tolerance);
}

TEST(TestCpuFpReferenceBatchnormWithVarianceFp32, CompareVarianceVsInvVarianceImplementationsNchw)
{
    // This test verifies that fwdInferenceWithVariance produces the same results
    // as fwdInference when given variance vs inv_variance inputs
    const std::vector<int64_t> dims = {2, 3, 4, 4};

    Tensor<float> inputTensor(dims);
    Tensor<float> outputFromVariance(dims);
    Tensor<float> outputFromInvVariance(dims);
    Tensor<float> scaleTensor({1, 3});
    Tensor<float> biasTensor({1, 3});
    Tensor<float> meanTensor({1, 3});
    Tensor<float> varianceTensor({1, 3});
    Tensor<float> invVarianceTensor({1, 3});

    // Initialize with random values
    inputTensor.fillWithRandomValues(-5.0f, 5.0f, 42);

    for(int i = 0; i < 3; i++)
    {
        scaleTensor.setHostValue(1.5f + (static_cast<float>(i) * 0.5f), 0, i);
        biasTensor.setHostValue(0.5f - (static_cast<float>(i) * 0.2f), 0, i);
        meanTensor.setHostValue(1.0f + (static_cast<float>(i) * 0.3f), 0, i);

        // Set variance and compute corresponding inv_variance
        auto var = 2.0f + (static_cast<float>(i) * 0.5f);
        varianceTensor.setHostValue(var, 0, i);
        auto invVar
            = 1.0f
              / hipdnn_data_sdk::types::sqrt(var + static_cast<float>(BATCHNORM_DEFAULT_EPSILON));
        invVarianceTensor.setHostValue(invVar, 0, i);
    }

    // Call both implementations
    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(
        inputTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, outputFromVariance);

    CpuFpReferenceBatchnorm::fwdInference(
        inputTensor, scaleTensor, biasTensor, meanTensor, invVarianceTensor, outputFromInvVariance);

    // Compare results
    auto tolerance = 1e-5f;
    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 4; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    auto valFromVariance = outputFromVariance.getHostValue(b, c, h, w);
                    auto valFromInvVariance = outputFromInvVariance.getHostValue(b, c, h, w);
                    EXPECT_NEAR(valFromVariance, valFromInvVariance, tolerance)
                        << "Mismatch at [" << b << "," << c << "," << h << "," << w << "]";
                }
            }
        }
    }
}
