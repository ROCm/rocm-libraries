// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_data_sdk/utilities/Constants.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

// ============================================================================
// Hand-computed golden reference tests
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropSanityValidation2D)
{
    // Input shape: [2, 4] — normalize over last dim (4 features per sample)
    // x = [[1, 2, 3, 4],
    //      [2, 4, 6, 8]]
    //
    // Sample 0: mean=2.5, var=1.25, rstd=1/sqrt(1.25+1e-5)=0.894423613312618
    //   xhat = [-1.34163542, -0.44721181, 0.44721181, 1.34163542]
    //   y = scale * xhat + bias = 2*xhat + 0.5
    //     = [-2.18327084, -0.39442361, 1.39442361, 3.18327084]
    //
    // Sample 1: mean=5.0, var=5.0, rstd=1/sqrt(5.0+1e-5)=0.447212474619866
    //   xhat = [-1.34163742, -0.44721247, 0.44721247, 1.34163742]
    //   y = 2*xhat + 0.5
    //     = [-2.18327485, -0.39442495, 1.39442495, 3.18327485]

    Tensor<double> x({2, 4});
    Tensor<double> y({2, 4});
    Tensor<double> scale({4});
    Tensor<double> bias({4});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 0, 3);
    x.setHostValue(2.0, 1, 0);
    x.setHostValue(4.0, 1, 1);
    x.setHostValue(6.0, 1, 2);
    x.setHostValue(8.0, 1, 3);

    for(int i = 0; i < 4; i++)
    {
        scale.setHostValue(2.0, i);
        bias.setHostValue(0.5, i);
    }

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    // Sample 0 statistics
    EXPECT_NEAR(mean.getHostValue(0), 2.5, tolerance);
    EXPECT_NEAR(rstd.getHostValue(0), 0.894423613312618, tolerance);

    // Sample 0 outputs
    EXPECT_NEAR(y.getHostValue(0, 0), -2.18327084, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 1), -0.39442361, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 2), 1.39442361, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 3), 3.18327084, tolerance);

    // Sample 1 statistics
    EXPECT_NEAR(mean.getHostValue(1), 5.0, tolerance);
    EXPECT_NEAR(rstd.getHostValue(1), 0.447212474619866, tolerance);

    // Sample 1 outputs
    EXPECT_NEAR(y.getHostValue(1, 0), -2.18327485, 1e-5);
    EXPECT_NEAR(y.getHostValue(1, 1), -0.39442495, 1e-5);
    EXPECT_NEAR(y.getHostValue(1, 2), 1.39442495, 1e-5);
    EXPECT_NEAR(y.getHostValue(1, 3), 3.18327485, 1e-5);
}

TEST(TestCpuFpReferenceLayernormFp64, FpropSanityValidation3D)
{
    // Input shape: [1, 2, 3] — normalize over last 2 dims (2x3 = 6 features)
    // x = [[[1, 2, 3], [4, 5, 6]]]
    //
    // mean = (1+2+3+4+5+6)/6 = 3.5
    // var = ((1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2) / 6
    //     = (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 6 = 17.5 / 6 = 2.916666...
    // rstd = 1/sqrt(2.916666... + 1e-5) = 0.585540...
    //
    // With scale=1, bias=0 (identity affine):
    //   y = xhat = (x - mean) * rstd

    Tensor<double> x({1, 2, 3});
    Tensor<double> y({1, 2, 3});
    Tensor<double> scale({2, 3});
    Tensor<double> bias({2, 3});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});

    x.setHostValue(1.0, 0, 0, 0);
    x.setHostValue(2.0, 0, 0, 1);
    x.setHostValue(3.0, 0, 0, 2);
    x.setHostValue(4.0, 0, 1, 0);
    x.setHostValue(5.0, 0, 1, 1);
    x.setHostValue(6.0, 0, 1, 2);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    EXPECT_NEAR(mean.getHostValue(0), 3.5, tolerance);

    const double expectedVar = 17.5 / 6.0;
    const double expectedRstd = 1.0 / std::sqrt(expectedVar + 1e-5);
    EXPECT_NEAR(rstd.getHostValue(0), expectedRstd, tolerance);

    // Verify each output element: y = (x - mean) * rstd
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            const double xVal = x.getHostValue(0, i, j);
            const double expected = (xVal - 3.5) * expectedRstd;
            EXPECT_NEAR(y.getHostValue(0, i, j), expected, tolerance);
        }
    }
}

// ============================================================================
// Corner case: all zeros input
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropAllZeros)
{
    // All-zero input: mean=0, var=0, rstd=1/sqrt(epsilon)
    // xhat = (0 - 0) * rstd = 0
    // y = scale * 0 + bias = bias

    Tensor<double> x({2, 4});
    Tensor<double> y({2, 4});
    Tensor<double> scale({4});
    Tensor<double> bias({4});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});

    x.fillWithValue(0.0);

    for(int i = 0; i < 4; i++)
    {
        scale.setHostValue(2.0, i);
        bias.setHostValue(3.0, i);
    }

    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, epsilon, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();
    // rstd = 1/sqrt(eps) is large, so float precision causes larger absolute error
    auto rstdTolerance = 1e-3;

    for(int b = 0; b < 2; b++)
    {
        EXPECT_NEAR(mean.getHostValue(b), 0.0, tolerance);
        EXPECT_NEAR(rstd.getHostValue(b), 1.0 / std::sqrt(epsilon), rstdTolerance);

        // Output should be bias since scale * 0 + bias = bias
        for(int f = 0; f < 4; f++)
        {
            EXPECT_NEAR(y.getHostValue(b, f), 3.0, tolerance);
        }
    }
}

// ============================================================================
// Corner case: all ones input
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropAllOnes)
{
    // All-one input: mean=1, var=0, rstd=1/sqrt(epsilon)
    // xhat = (1 - 1) * rstd = 0
    // y = scale * 0 + bias = bias

    Tensor<double> x({3, 5});
    Tensor<double> y({3, 5});
    Tensor<double> scale({5});
    Tensor<double> bias({5});
    Tensor<double> mean({3});
    Tensor<double> rstd({3});

    x.fillWithValue(1.0);
    scale.fillWithValue(1.5);
    bias.fillWithValue(0.25);

    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, epsilon, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();
    // rstd = 1/sqrt(eps) is large, so float precision causes larger absolute error
    auto rstdTolerance = 1e-3;

    for(int b = 0; b < 3; b++)
    {
        EXPECT_NEAR(mean.getHostValue(b), 1.0, tolerance);
        EXPECT_NEAR(rstd.getHostValue(b), 1.0 / std::sqrt(epsilon), rstdTolerance);

        // All elements identical -> normalized to 0 -> y = bias
        for(int f = 0; f < 5; f++)
        {
            EXPECT_NEAR(y.getHostValue(b, f), 0.25, tolerance);
        }
    }
}

// ============================================================================
// Corner case: constant input (all same non-trivial value)
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropConstantInput)
{
    // All elements = 7.0: mean=7, var=0
    // Same behavior as all-ones but shifted — output should still be bias

    Tensor<double> x({2, 3});
    Tensor<double> y({2, 3});
    Tensor<double> scale({3});
    Tensor<double> bias({3});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});

    x.fillWithValue(7.0);
    scale.fillWithValue(1.0);
    bias.fillWithValue(-1.0);

    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, epsilon, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    for(int b = 0; b < 2; b++)
    {
        EXPECT_NEAR(mean.getHostValue(b), 7.0, tolerance);

        for(int f = 0; f < 3; f++)
        {
            EXPECT_NEAR(y.getHostValue(b, f), -1.0, tolerance);
        }
    }
}

// ============================================================================
// Corner case: single element per normalized group
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropSingleFeature)
{
    // Shape [3, 1]: normalize over dim of size 1
    // mean = x, var = 0, xhat = 0, y = bias

    Tensor<double> x({3, 1});
    Tensor<double> y({3, 1});
    Tensor<double> scale({1});
    Tensor<double> bias({1});
    Tensor<double> mean({3});
    Tensor<double> rstd({3});

    x.setHostValue(5.0, 0, 0);
    x.setHostValue(-3.0, 1, 0);
    x.setHostValue(0.0, 2, 0);

    scale.setHostValue(2.0, 0);
    bias.setHostValue(1.0, 0);

    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, epsilon, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    // Each sample has 1 feature, so mean=x, var=0, xhat=0, y=bias
    EXPECT_NEAR(mean.getHostValue(0), 5.0, tolerance);
    EXPECT_NEAR(mean.getHostValue(1), -3.0, tolerance);
    EXPECT_NEAR(mean.getHostValue(2), 0.0, tolerance);

    EXPECT_NEAR(y.getHostValue(0, 0), 1.0, tolerance);
    EXPECT_NEAR(y.getHostValue(1, 0), 1.0, tolerance);
    EXPECT_NEAR(y.getHostValue(2, 0), 1.0, tolerance);
}

// ============================================================================
// Corner case: identity transform (scale=1, bias=0) — output should be xhat
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropIdentityTransform)
{
    // With scale=1, bias=0, output should be the standardized values
    // Verify that the output has zero mean and unit variance

    Tensor<double> x({1, 6});
    Tensor<double> y({1, 6});
    Tensor<double> scale({6});
    Tensor<double> bias({6});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});

    // Known values
    x.setHostValue(10.0, 0, 0);
    x.setHostValue(20.0, 0, 1);
    x.setHostValue(30.0, 0, 2);
    x.setHostValue(40.0, 0, 3);
    x.setHostValue(50.0, 0, 4);
    x.setHostValue(60.0, 0, 5);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    EXPECT_NEAR(mean.getHostValue(0), 35.0, tolerance);

    // Verify output has approximately zero mean
    double outputMean = 0.0;
    for(int i = 0; i < 6; i++)
    {
        outputMean += y.getHostValue(0, i);
    }
    outputMean /= 6.0;
    EXPECT_NEAR(outputMean, 0.0, tolerance);

    // Verify output has approximately unit variance
    double outputVar = 0.0;
    for(int i = 0; i < 6; i++)
    {
        const double diff = y.getHostValue(0, i) - outputMean;
        outputVar += diff * diff;
    }
    outputVar /= 6.0;
    EXPECT_NEAR(outputVar, 1.0, tolerance);
}

// ============================================================================
// Corner case: negative values and mixed signs
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropNegativeAndMixedValues)
{
    // Test with negative and mixed-sign values to ensure correct handling
    // x = [-3, -1, 1, 3]
    // mean = 0, var = (9+1+1+9)/4 = 5.0
    // rstd = 1/sqrt(5 + 1e-5)
    // xhat = x * rstd (since mean=0)

    Tensor<double> x({1, 4});
    Tensor<double> y({1, 4});
    Tensor<double> scale({4});
    Tensor<double> bias({4});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});

    x.setHostValue(-3.0, 0, 0);
    x.setHostValue(-1.0, 0, 1);
    x.setHostValue(1.0, 0, 2);
    x.setHostValue(3.0, 0, 3);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    EXPECT_NEAR(mean.getHostValue(0), 0.0, tolerance);

    const double expectedRstd = 1.0 / std::sqrt(5.0 + 1e-5);
    EXPECT_NEAR(rstd.getHostValue(0), expectedRstd, tolerance);

    // Output should be antisymmetric: y[0]=-y[3], y[1]=-y[2]
    EXPECT_NEAR(y.getHostValue(0, 0), -y.getHostValue(0, 3), tolerance);
    EXPECT_NEAR(y.getHostValue(0, 1), -y.getHostValue(0, 2), tolerance);

    // And the actual values
    EXPECT_NEAR(y.getHostValue(0, 0), -3.0 * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 1), -1.0 * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 2), 1.0 * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 3), 3.0 * expectedRstd, tolerance);
}

// ============================================================================
// Corner case: large epsilon
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropLargeEpsilon)
{
    // Large epsilon dominates variance, effectively suppressing normalization
    // x = [1, 2, 3, 4], mean=2.5, var=1.25
    // rstd = 1/sqrt(1.25 + 100) = 1/sqrt(101.25) ≈ 0.099380...
    // Output should be close to bias since rstd is very small

    Tensor<double> x({1, 4});
    Tensor<double> y({1, 4});
    Tensor<double> scale({4});
    Tensor<double> bias({4});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 0, 3);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    const double largeEpsilon = 100.0;
    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, largeEpsilon, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    EXPECT_NEAR(mean.getHostValue(0), 2.5, tolerance);

    const double expectedRstd = 1.0 / std::sqrt(1.25 + largeEpsilon);
    EXPECT_NEAR(rstd.getHostValue(0), expectedRstd, tolerance);

    // With large epsilon, rstd is small so all outputs should be near 0 (bias)
    for(int i = 0; i < 4; i++)
    {
        EXPECT_NEAR(y.getHostValue(0, i), 0.0, 0.2);
    }
}

// ============================================================================
// Numerical stability: large values with small variance
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropLargeValueNumericalStability)
{
    // Values clustered around 1e15 with small relative differences.
    // Naive one-pass (E[x²] - E[x]²) would suffer catastrophic cancellation here
    // because E[x²] ≈ 1e30 and E[x]² ≈ 1e30, so their difference loses all precision.
    // Welford's algorithm computes variance from centered deltas, avoiding this.
    //
    // x = [1e15 + 1, 1e15 + 2, 1e15 + 3, 1e15 + 4]
    // mean = 1e15 + 2.5
    // var = 1.25 (same as small-value case, independent of offset)

    Tensor<double> x({1, 4});
    Tensor<double> y({1, 4});
    Tensor<double> scale({4});
    Tensor<double> bias({4});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});

    const double offset = 1e15;
    x.setHostValue(offset + 1.0, 0, 0);
    x.setHostValue(offset + 2.0, 0, 1);
    x.setHostValue(offset + 3.0, 0, 2);
    x.setHostValue(offset + 4.0, 0, 3);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop<double, double, double, double, double>(
        x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    EXPECT_NEAR(mean.getHostValue(0), offset + 2.5, tolerance);

    // Variance should be exactly 1.25, same as if offset were 0
    const double expectedRstd = 1.0 / std::sqrt(1.25 + 1e-5);
    EXPECT_NEAR(rstd.getHostValue(0), expectedRstd, tolerance);

    // Normalized output should match the small-value case exactly
    EXPECT_NEAR(y.getHostValue(0, 0), -1.5 * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 1), -0.5 * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 2), 0.5 * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 3), 1.5 * expectedRstd, tolerance);
}

// ============================================================================
// No scale/bias (optional tensors not provided)
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropWithoutScaleBias)
{
    // Without scale and bias, output should be pure xhat
    // x = [1, 2, 3, 4], mean=2.5, var=1.25
    // y = xhat = (x - 2.5) / sqrt(1.25 + 1e-5)

    Tensor<double> x({1, 4});
    Tensor<double> y({1, 4});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 0, 3);

    CpuFpReferenceLayernorm::fprop<double, double, double, double>(
        x,
        static_cast<TensorBase<double>*>(nullptr),
        static_cast<TensorBase<double>*>(nullptr),
        y,
        1e-5,
        1,
        &mean,
        &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    const double expectedRstd = 1.0 / std::sqrt(1.25 + 1e-5);
    EXPECT_NEAR(mean.getHostValue(0), 2.5, tolerance);
    EXPECT_NEAR(rstd.getHostValue(0), expectedRstd, tolerance);

    EXPECT_NEAR(y.getHostValue(0, 0), (1.0 - 2.5) * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 1), (2.0 - 2.5) * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 2), (3.0 - 2.5) * expectedRstd, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 3), (4.0 - 2.5) * expectedRstd, tolerance);
}

// ============================================================================
// Realistic shape: typical transformer hidden dim
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp32, FpropTypicalTransformerShape)
{
    // Batch=2, SeqLen=8, Hidden=64 — normalize over last dim (hidden)
    Tensor<float> x({2, 8, 64});
    Tensor<float> y({2, 8, 64});
    Tensor<float> scale({64});
    Tensor<float> bias({64});
    Tensor<float> mean({2, 8});
    Tensor<float> rstd({2, 8});

    x.fillWithRandomValues(-1.0f, 1.0f, 42);
    scale.fillWithValue(1.0f);
    bias.fillWithValue(0.0f);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    // With identity scale/bias, verify output properties per normalized group:
    // - output mean ≈ 0
    // - output variance ≈ 1
    auto tolerance = layernorm::getTolerance<float>();

    for(int b = 0; b < 2; b++)
    {
        for(int s = 0; s < 8; s++)
        {
            float outMean = 0.0f;
            for(int h = 0; h < 64; h++)
            {
                outMean += y.getHostValue(b, s, h);
            }
            outMean /= 64.0f;
            EXPECT_NEAR(outMean, 0.0f, tolerance);

            float outVar = 0.0f;
            for(int h = 0; h < 64; h++)
            {
                const float diff = y.getHostValue(b, s, h) - outMean;
                outVar += diff * diff;
            }
            outVar /= 64.0f;
            EXPECT_NEAR(outVar, 1.0f, tolerance);
        }
    }
}

// ============================================================================
// Transformer padding: PAD (zero) positions don't affect real token outputs
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropPadPositionsDoNotAffectRealTokens)
{
    // Simulates a typical NLP scenario with variable-length sequences zero-padded
    // to a fixed length. With normalizedDimCount=1, LayerNorm normalizes each
    // position independently over the hidden dimension (d_model). PAD positions
    // cannot corrupt real token outputs because each position computes its own
    // mean and variance.
    //
    // Shape: [N=2, seq=6, d_model=4]
    //
    // batch 0: "The cat sat [PAD] [PAD] [PAD]"
    //   pos 0 "The": [0.8, 0.2, 0.5, 0.1]   <- real
    //   pos 1 "cat": [0.1, 0.9, 0.3, 0.7]   <- real
    //   pos 2 "sat": [0.3, 0.1, 0.8, 0.4]   <- real
    //   pos 3  PAD : [0.0, 0.0, 0.0, 0.0]   <- padding
    //   pos 4  PAD : [0.0, 0.0, 0.0, 0.0]   <- padding
    //   pos 5  PAD : [0.0, 0.0, 0.0, 0.0]   <- padding
    //
    // batch 1: "The quick brown fox [PAD] [PAD]"
    //   pos 0 "The"  : [0.8, 0.2, 0.5, 0.1]   <- real
    //   pos 1 "quick": [0.6, 0.7, 0.2, 0.9]   <- real
    //   pos 2 "brown": [0.4, 0.3, 0.6, 0.5]   <- real
    //   pos 3 "fox"  : [0.9, 0.8, 0.1, 0.3]   <- real
    //   pos 4  PAD   : [0.0, 0.0, 0.0, 0.0]   <- padding
    //   pos 5  PAD   : [0.0, 0.0, 0.0, 0.0]   <- padding

    Tensor<double> x({2, 6, 4});
    Tensor<double> y({2, 6, 4});
    // scale shape {4} matches the last 1 dimension (d_model) of x, which is the
    // normalized dimension selected by normalizedDimCount=1 below. This means
    // LayerNorm computes mean/variance independently for each (batch, seq_pos)
    // pair over the 4 d_model features — exactly why PAD positions at different
    // sequence positions cannot affect real token outputs.
    Tensor<double> scale({4});
    Tensor<double> bias({4});
    Tensor<double> mean({2, 6});
    Tensor<double> rstd({2, 6});

    x.fillWithValue(0.0); // Initialize all to zero (PAD positions stay zero)

    // Batch 0: real tokens at positions 0-2
    x.setHostValue(0.8, 0, 0, 0);
    x.setHostValue(0.2, 0, 0, 1);
    x.setHostValue(0.5, 0, 0, 2);
    x.setHostValue(0.1, 0, 0, 3);

    x.setHostValue(0.1, 0, 1, 0);
    x.setHostValue(0.9, 0, 1, 1);
    x.setHostValue(0.3, 0, 1, 2);
    x.setHostValue(0.7, 0, 1, 3);

    x.setHostValue(0.3, 0, 2, 0);
    x.setHostValue(0.1, 0, 2, 1);
    x.setHostValue(0.8, 0, 2, 2);
    x.setHostValue(0.4, 0, 2, 3);

    // Batch 1: real tokens at positions 0-3
    x.setHostValue(0.8, 1, 0, 0);
    x.setHostValue(0.2, 1, 0, 1);
    x.setHostValue(0.5, 1, 0, 2);
    x.setHostValue(0.1, 1, 0, 3);

    x.setHostValue(0.6, 1, 1, 0);
    x.setHostValue(0.7, 1, 1, 1);
    x.setHostValue(0.2, 1, 1, 2);
    x.setHostValue(0.9, 1, 1, 3);

    x.setHostValue(0.4, 1, 2, 0);
    x.setHostValue(0.3, 1, 2, 1);
    x.setHostValue(0.6, 1, 2, 2);
    x.setHostValue(0.5, 1, 2, 3);

    x.setHostValue(0.9, 1, 3, 0);
    x.setHostValue(0.8, 1, 3, 1);
    x.setHostValue(0.1, 1, 3, 2);
    x.setHostValue(0.3, 1, 3, 3);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    const double epsilon = 1e-5;
    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, epsilon, 1, &mean, &rstd);

    const auto tolerance = 1e-6;

    // Key assertion: "The" appears at (0,0) and (1,0) with identical input
    // [0.8, 0.2, 0.5, 0.1]. Despite different padding patterns in their batches
    // (3 PADs vs 2 PADs), the outputs must be identical — each position normalizes
    // independently over d_model, so neighboring PAD positions have no effect.
    //
    // "The": mean = (0.8+0.2+0.5+0.1)/4 = 0.4
    //        var  = (0.16+0.04+0.01+0.09)/4 = 0.075
    const double theMean = 0.4;
    const double theRstd = 1.0 / std::sqrt(0.075 + epsilon);

    EXPECT_NEAR(mean.getHostValue(0, 0), theMean, tolerance);
    EXPECT_NEAR(rstd.getHostValue(0, 0), theRstd, tolerance);
    EXPECT_NEAR(mean.getHostValue(1, 0), theMean, tolerance);
    EXPECT_NEAR(rstd.getHostValue(1, 0), theRstd, tolerance);

    for(int d = 0; d < 4; d++)
    {
        EXPECT_NEAR(y.getHostValue(0, 0, d), y.getHostValue(1, 0, d), tolerance);
    }

    // PAD positions: all-zero input -> mean=0, var=0, output=0 (identity affine)
    for(int pad = 3; pad < 6; pad++)
    {
        EXPECT_NEAR(mean.getHostValue(0, pad), 0.0, tolerance);
        for(int d = 0; d < 4; d++)
        {
            EXPECT_NEAR(y.getHostValue(0, pad, d), 0.0, tolerance);
        }
    }
    for(int pad = 4; pad < 6; pad++)
    {
        EXPECT_NEAR(mean.getHostValue(1, pad), 0.0, tolerance);
        for(int d = 0; d < 4; d++)
        {
            EXPECT_NEAR(y.getHostValue(1, pad, d), 0.0, tolerance);
        }
    }
}

// ============================================================================
// Type compatibility: various type combinations
// ============================================================================

template <typename T1, typename T2>
struct TypePair
{
    using First = T1;
    using Second = T2;
};

using LayernormFpropTypes = ::testing::Types<TypePair<float, float>,
                                             TypePair<half, float>,
                                             TypePair<bfloat16, float>,
                                             TypePair<double, double>>;

template <class T>
class CpuFpReferenceLayernormFpropTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceLayernormFpropTyped, LayernormFpropTypes, );

TYPED_TEST(CpuFpReferenceLayernormFpropTyped, FpropRunsWithoutError)
{
    using XType = typename TypeParam::First;
    using ScaleType = typename TypeParam::Second;

    Tensor<XType> x({2, 8, 32});
    Tensor<XType> y({2, 8, 32});
    Tensor<ScaleType> scale({32});
    Tensor<ScaleType> bias({32});
    Tensor<ScaleType> mean({2, 8});
    Tensor<ScaleType> rstd({2, 8});

    x.fillWithValue(safeTestTypeCast<XType>(1.0));
    scale.fillWithValue(safeTestTypeCast<ScaleType>(1.0));
    bias.fillWithValue(safeTestTypeCast<ScaleType>(0.0));

    // Should not throw
    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    // All identical values -> all normalized to 0 -> y = bias = 0
    for(int b = 0; b < 2; b++)
    {
        for(int s = 0; s < 8; s++)
        {
            for(int h = 0; h < 32; h++)
            {
                auto yVal = static_cast<float>(y.getHostValue(b, s, h));
                EXPECT_NEAR(yVal, 0.0f, 0.01f);
            }
        }
    }
}

// ============================================================================
// Multi-dim normalization: normalize over last 2 dims
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropNormalizeLastTwoDims)
{
    // Shape [2, 3, 4], normalize over last 2 dims (3x4 = 12 features)
    // Each of the 2 batches is independently normalized over 12 elements

    Tensor<double> x({2, 3, 4});
    Tensor<double> y({2, 3, 4});
    Tensor<double> scale({3, 4});
    Tensor<double> bias({3, 4});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});

    // Batch 0: values 1..12
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            x.setHostValue(static_cast<double>(i * 4 + j + 1), 0, i, j);
        }
    }

    // Batch 1: values 13..24
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            x.setHostValue(static_cast<double>(i * 4 + j + 13), 1, i, j);
        }
    }

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    // Batch 0: mean = (1+2+...+12)/12 = 78/12 = 6.5
    EXPECT_NEAR(mean.getHostValue(0), 6.5, tolerance);

    // Batch 1: mean = (13+14+...+24)/12 = 222/12 = 18.5
    EXPECT_NEAR(mean.getHostValue(1), 18.5, tolerance);

    // Verify output zero-mean property for each batch
    for(int b = 0; b < 2; b++)
    {
        double outSum = 0.0;
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                outSum += y.getHostValue(b, i, j);
            }
        }
        EXPECT_NEAR(outSum / 12.0, 0.0, tolerance);
    }
}

// ============================================================================
// Corner case: "diagonal-like" pattern — one hot-per-row
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropOneHotRows)
{
    // Shape [3, 3], each row is a one-hot vector
    // Row 0: [1, 0, 0], mean=1/3, var=(4/9+1/9+1/9)/3=2/9
    // Row 1: [0, 1, 0], same statistics by symmetry
    // Row 2: [0, 0, 1], same statistics by symmetry
    //
    // All rows should produce the same set of output values (permuted)

    Tensor<double> x({3, 3});
    Tensor<double> y({3, 3});
    Tensor<double> scale({3});
    Tensor<double> bias({3});
    Tensor<double> mean({3});
    Tensor<double> rstd({3});

    x.fillWithValue(0.0);
    x.setHostValue(1.0, 0, 0);
    x.setHostValue(1.0, 1, 1);
    x.setHostValue(1.0, 2, 2);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    // All rows have same mean = 1/3
    const double expectedMean = 1.0 / 3.0;
    for(int b = 0; b < 3; b++)
    {
        EXPECT_NEAR(mean.getHostValue(b), expectedMean, tolerance);
    }

    // All rows have same rstd
    EXPECT_NEAR(rstd.getHostValue(0), rstd.getHostValue(1), tolerance);
    EXPECT_NEAR(rstd.getHostValue(1), rstd.getHostValue(2), tolerance);

    // The "hot" position output should be identical across rows
    // y[0,0] should equal y[1,1] and y[2,2]
    EXPECT_NEAR(y.getHostValue(0, 0), y.getHostValue(1, 1), tolerance);
    EXPECT_NEAR(y.getHostValue(1, 1), y.getHostValue(2, 2), tolerance);

    // The "cold" position output should be identical across rows
    EXPECT_NEAR(y.getHostValue(0, 1), y.getHostValue(0, 2), tolerance);
    EXPECT_NEAR(y.getHostValue(0, 1), y.getHostValue(1, 0), tolerance);
}

// ============================================================================
// Verify per-feature scale and bias
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropPerFeatureScaleBias)
{
    // Verify that different scale/bias per feature are applied correctly
    // x = [[1, 2, 3, 4]]
    // scale = [1, 2, 3, 4], bias = [0, 10, 20, 30]

    Tensor<double> x({1, 4});
    Tensor<double> y({1, 4});
    Tensor<double> scale({4});
    Tensor<double> bias({4});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 0, 3);

    scale.setHostValue(1.0, 0);
    scale.setHostValue(2.0, 1);
    scale.setHostValue(3.0, 2);
    scale.setHostValue(4.0, 3);

    bias.setHostValue(0.0, 0);
    bias.setHostValue(10.0, 1);
    bias.setHostValue(20.0, 2);
    bias.setHostValue(30.0, 3);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    // mean=2.5, var=1.25, rstd=1/sqrt(1.25+1e-5)
    const double expectedMean = 2.5;
    const double expectedRstd = 1.0 / std::sqrt(1.25 + 1e-5);
    EXPECT_NEAR(mean.getHostValue(0), expectedMean, tolerance);

    // y[i] = scale[i] * (x[i] - mean) * rstd + bias[i]
    // Use slightly relaxed tolerance due to float compute precision
    auto yTolerance = 1e-5;
    for(int i = 0; i < 4; i++)
    {
        const double xVal = x.getHostValue(0, i);
        const double scaleVal = scale.getHostValue(i);
        const double biasVal = bias.getHostValue(i);
        const double expected = scaleVal * (xVal - expectedMean) * expectedRstd + biasVal;
        EXPECT_NEAR(y.getHostValue(0, i), expected, yTolerance);
    }
}

// ============================================================================
// Without mean/rstd outputs (optional outputs not provided)
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropWithoutMeanRstdOutputs)
{
    Tensor<double> x({2, 4});
    Tensor<double> y({2, 4});
    Tensor<double> scale({4});
    Tensor<double> bias({4});

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 0, 3);
    x.setHostValue(2.0, 1, 0);
    x.setHostValue(4.0, 1, 1);
    x.setHostValue(6.0, 1, 2);
    x.setHostValue(8.0, 1, 3);

    scale.fillWithValue(2.0);
    bias.fillWithValue(0.5);

    // Call without mean/rstd pointers — should work fine
    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1);

    // Verify same results as the golden test (Sample 0)
    auto tolerance = layernorm::getTolerance<double>();
    EXPECT_NEAR(y.getHostValue(0, 0), -2.18327084, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 1), -0.39442361, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 2), 1.39442361, tolerance);
    EXPECT_NEAR(y.getHostValue(0, 3), 3.18327084, tolerance);
}

// ============================================================================
// Error handling: invalid dimensions
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp32, FpropThrowsOnInvalidNormalizedDimCount)
{
    const Tensor<float> x({2, 4});
    Tensor<float> y({2, 4});
    Tensor<float> scale({4});
    Tensor<float> bias({4});

    // normalizedDimCount = 0 is invalid
    EXPECT_THROW(CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 0),
                 std::runtime_error);

    // normalizedDimCount > ndim is invalid
    EXPECT_THROW(CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3),
                 std::runtime_error);
}

// ============================================================================
// Error handling: scalar (0D) tensor
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp32, FpropThrowsOnScalarTensor)
{
    const Tensor<float> x({});
    Tensor<float> y({});
    Tensor<float> scale({});
    Tensor<float> bias({});

    EXPECT_THROW(CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1),
                 std::runtime_error);
}

// ============================================================================
// Dimensionality coverage: 1D tensor
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, Fprop1D)
{
    // Shape [5] — normalize entire 1D tensor (single group, no batch dim)
    // x = [2, 4, 6, 8, 10]
    // mean = 6, var = (16+4+0+4+16)/5 = 8
    // rstd = 1/sqrt(8 + 1e-5)

    Tensor<double> x({5});
    Tensor<double> y({5});
    Tensor<double> scale({5});
    Tensor<double> bias({5});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});

    x.setHostValue(2.0, 0);
    x.setHostValue(4.0, 1);
    x.setHostValue(6.0, 2);
    x.setHostValue(8.0, 3);
    x.setHostValue(10.0, 4);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    EXPECT_NEAR(mean.getHostValue(0), 6.0, tolerance);

    const double expectedRstd = 1.0 / std::sqrt(8.0 + 1e-5);
    EXPECT_NEAR(rstd.getHostValue(0), expectedRstd, tolerance);

    // y = (x - 6) * rstd
    for(int i = 0; i < 5; i++)
    {
        const double xVal = x.getHostValue(i);
        EXPECT_NEAR(y.getHostValue(i), (xVal - 6.0) * expectedRstd, tolerance);
    }

    // Output should have zero mean and unit variance
    double outSum = 0.0;
    for(int i = 0; i < 5; i++)
    {
        outSum += y.getHostValue(i);
    }
    EXPECT_NEAR(outSum / 5.0, 0.0, tolerance);
}

// ============================================================================
// Dimensionality coverage: 4D tensor
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, Fprop4DNormalizeLast1)
{
    // Shape [2, 3, 2, 4], normalizedDimCount=1: normalize over last dim (4)
    // Batch dims = [2, 3, 2], 12 independent groups of 4 elements each

    Tensor<double> x({2, 3, 2, 4});
    Tensor<double> y({2, 3, 2, 4});
    Tensor<double> scale({4});
    Tensor<double> bias({4});
    Tensor<double> mean({2, 3, 2});
    Tensor<double> rstd({2, 3, 2});

    // Fill with sequential values so each group has a known pattern
    double val = 1.0;
    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 2; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    x.setHostValue(val, b, c, h, w);
                    val += 1.0;
                }
            }
        }
    }

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 1, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    // Verify each of the 12 groups has zero-mean, unit-variance output
    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 2; h++)
            {
                double outMean = 0.0;
                for(int w = 0; w < 4; w++)
                {
                    outMean += y.getHostValue(b, c, h, w);
                }
                outMean /= 4.0;
                EXPECT_NEAR(outMean, 0.0, tolerance);

                double outVar = 0.0;
                for(int w = 0; w < 4; w++)
                {
                    const double diff = y.getHostValue(b, c, h, w) - outMean;
                    outVar += diff * diff;
                }
                outVar /= 4.0;
                EXPECT_NEAR(outVar, 1.0, tolerance);
            }
        }
    }

    // Spot-check first group: x = [1, 2, 3, 4], mean = 2.5
    EXPECT_NEAR(mean.getHostValue(0, 0, 0), 2.5, tolerance);
}

TEST(TestCpuFpReferenceLayernormFp64, Fprop4DNormalizeLast3)
{
    // Shape [2, 3, 2, 4], normalizedDimCount=3: normalize over last 3 dims (3*2*4=24)
    // Batch dims = [2], 2 independent groups of 24 elements each

    Tensor<double> x({2, 3, 2, 4});
    Tensor<double> y({2, 3, 2, 4});
    Tensor<double> scale({3, 2, 4});
    Tensor<double> bias({3, 2, 4});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});

    // Batch 0: values 1..24, Batch 1: values 25..48
    double val = 1.0;
    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 2; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    x.setHostValue(val, b, c, h, w);
                    val += 1.0;
                }
            }
        }
    }

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 3, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    // Batch 0: mean = (1+2+...+24)/24 = 300/24 = 12.5
    EXPECT_NEAR(mean.getHostValue(0), 12.5, tolerance);

    // Batch 1: mean = (25+26+...+48)/24 = 876/24 = 36.5
    EXPECT_NEAR(mean.getHostValue(1), 36.5, tolerance);

    // Verify zero-mean output per batch
    for(int b = 0; b < 2; b++)
    {
        double outSum = 0.0;
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 2; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    outSum += y.getHostValue(b, c, h, w);
                }
            }
        }
        EXPECT_NEAR(outSum / 24.0, 0.0, tolerance);
    }
}

// ============================================================================
// Full-tensor normalization (normalizedDimCount == ndim)
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, FpropFullTensorNormalization)
{
    // Shape [2, 3], normalizedDimCount=2 means normalize entire tensor per "batch"
    // But there are no batch dims here, so entire tensor is one group
    // x = [[1, 2, 3], [4, 5, 6]]
    // mean = 3.5, var = 17.5/6 = 2.9166...

    Tensor<double> x({2, 3});
    Tensor<double> y({2, 3});
    Tensor<double> scale({2, 3});
    Tensor<double> bias({2, 3});

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 1, 0);
    x.setHostValue(5.0, 1, 1);
    x.setHostValue(6.0, 1, 2);

    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2);

    // Verify output has zero mean (sum to 0)
    double sum = 0.0;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            sum += y.getHostValue(i, j);
        }
    }
    EXPECT_NEAR(sum / 6.0, 0.0, 1e-5);

    // Verify output has unit variance
    double var = 0.0;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            var += y.getHostValue(i, j) * y.getHostValue(i, j);
        }
    }
    EXPECT_NEAR(var / 6.0, 1.0, 1e-5);
}

// ============================================================================
// Dimensionality coverage: 5D tensor
// ============================================================================

TEST(TestCpuFpReferenceLayernormFp64, Fprop5DNormalizeLast2)
{
    // Shape [2, 2, 3, 4, 5], normalizedDimCount=2: normalize over last 2 dims (4*5=20)
    // Batch dims = [2, 2, 3], 12 independent groups of 20 elements each

    Tensor<double> x({2, 2, 3, 4, 5});
    Tensor<double> y({2, 2, 3, 4, 5});
    Tensor<double> scale({4, 5});
    Tensor<double> bias({4, 5});
    Tensor<double> mean({2, 2, 3});
    Tensor<double> rstd({2, 2, 3});

    x.fillWithRandomValues(-5.0, 5.0, 123);
    scale.fillWithValue(1.0);
    bias.fillWithValue(0.0);

    CpuFpReferenceLayernorm::fprop(x, &scale, &bias, y, LAYERNORM_DEFAULT_EPSILON, 2, &mean, &rstd);

    auto tolerance = layernorm::getTolerance<double>();

    // Verify each of the 12 groups has zero-mean, unit-variance output
    for(int b0 = 0; b0 < 2; b0++)
    {
        for(int b1 = 0; b1 < 2; b1++)
        {
            for(int b2 = 0; b2 < 3; b2++)
            {
                double outMean = 0.0;
                for(int d3 = 0; d3 < 4; d3++)
                {
                    for(int d4 = 0; d4 < 5; d4++)
                    {
                        outMean += y.getHostValue(b0, b1, b2, d3, d4);
                    }
                }
                outMean /= 20.0;
                EXPECT_NEAR(outMean, 0.0, tolerance);

                double outVar = 0.0;
                for(int d3 = 0; d3 < 4; d3++)
                {
                    for(int d4 = 0; d4 < 5; d4++)
                    {
                        const double diff = y.getHostValue(b0, b1, b2, d3, d4) - outMean;
                        outVar += diff * diff;
                    }
                }
                outVar /= 20.0;
                EXPECT_NEAR(outVar, 1.0, tolerance);
            }
        }
    }
}
