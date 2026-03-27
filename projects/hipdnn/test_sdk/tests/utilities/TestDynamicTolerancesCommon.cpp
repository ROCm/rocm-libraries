// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>

using namespace hipdnn_test_sdk::utilities;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

// =================================================================================================
// TestComputeGamma — verifies computeGamma against hand-derived formulas
//
// computeGamma selects between two regimes based on nU = 2*k*epsilon:
//   nU < 0.01  → linear:      gamma = nU / (1 - nU)
//   nU >= 0.01 → statistical: gamma = 6 * sqrt(2*k) * epsilon
//
// Tests below compute expected values from the raw formulas, NOT by calling computeGamma.
// =================================================================================================

// --- Linear regime (nU < 0.01) ---

TEST(TestComputeGamma, LinearRegime_Float_K1)
{
    // float eps = 2^-23, k=1: nU = 2 * 2^-23 = 2^-22 ≈ 2.38e-7 << 0.01
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double nU = 2.0 * 1.0 * u;
    const double expected = nU / (1.0 - nU);

    EXPECT_NEAR(computeGamma(1, u), expected, 1e-15);
}

TEST(TestComputeGamma, LinearRegime_Float_K100)
{
    // float eps = 2^-23, k=100: nU = 200 * 2^-23 ≈ 2.38e-5 << 0.01
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double nU = 2.0 * 100.0 * u;
    const double expected = nU / (1.0 - nU);

    EXPECT_NEAR(computeGamma(100, u), expected, 1e-15);
}

TEST(TestComputeGamma, LinearRegime_Half_K1)
{
    // half eps = 2^-10, k=1: nU = 2 * 2^-10 = 2^-9 ≈ 0.00195 < 0.01
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    const double nU = 2.0 * 1.0 * u;
    const double expected = nU / (1.0 - nU); // = 1/511 ≈ 0.001957

    EXPECT_NEAR(computeGamma(1, u), expected, 1e-10);
}

TEST(TestComputeGamma, LinearRegime_Half_K5)
{
    // half eps = 2^-10, k=5: nU = 10 * 2^-10 = 10/1024 ≈ 0.00977 < 0.01 (boundary)
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    const double nU = 2.0 * 5.0 * u;
    const double expected = nU / (1.0 - nU); // = 5/507 ≈ 0.009862

    EXPECT_NEAR(computeGamma(5, u), expected, 1e-10);
}

TEST(TestComputeGamma, LinearRegime_Double_K1000)
{
    // double eps = 2^-52, k=1000: nU = 2000 * 2^-52 ≈ 4.44e-13 << 0.01
    const auto u = std::numeric_limits<double>::epsilon();
    const double nU = 2.0 * 1000.0 * u;
    const double expected = nU / (1.0 - nU);

    EXPECT_NEAR(computeGamma(1000, u), expected, 1e-20);
}

// --- Statistical regime (nU >= 0.01) ---

TEST(TestComputeGamma, StatisticalRegime_Bf16_K1)
{
    // bf16 eps = 2^-7, k=1: nU = 2 * 2^-7 = 2^-6 = 0.015625 >= 0.01
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    const double expected = 6.0 * std::sqrt(2.0 * 1.0) * u; // 6 * sqrt(2) * 2^-7

    EXPECT_NEAR(computeGamma(1, u), expected, 1e-10);
}

TEST(TestComputeGamma, StatisticalRegime_Bf16_K10)
{
    // bf16 eps = 2^-7, k=10: nU = 20 * 2^-7 ≈ 0.156 >= 0.01
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    const double expected = 6.0 * std::sqrt(2.0 * 10.0) * u; // 6 * sqrt(20) * 2^-7

    EXPECT_NEAR(computeGamma(10, u), expected, 1e-10);
}

TEST(TestComputeGamma, StatisticalRegime_Bf16_K100)
{
    // bf16 eps = 2^-7, k=100: nU = 200 * 2^-7 ≈ 1.56 >= 0.01
    // gamma = 6 * sqrt(200) * 2^-7 ≈ 0.663 (exceeds 0.5 — useful for overflow tests)
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    const double expected = 6.0 * std::sqrt(2.0 * 100.0) * u;

    EXPECT_NEAR(computeGamma(100, u), expected, 1e-10);
    EXPECT_GT(computeGamma(100, u), 0.5); // Confirms this triggers validateGamma overflow
}

TEST(TestComputeGamma, StatisticalRegime_Half_K6)
{
    // half eps = 2^-10, k=6: nU = 12 * 2^-10 ≈ 0.01172 >= 0.01 (just above threshold)
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    const double expected = 6.0 * std::sqrt(2.0 * 6.0) * u; // 6 * sqrt(12) * 2^-10

    EXPECT_NEAR(computeGamma(6, u), expected, 1e-10);
}

TEST(TestComputeGamma, StatisticalRegime_Half_K10)
{
    // half eps = 2^-10, k=10: nU = 20 * 2^-10 ≈ 0.0195 >= 0.01
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    const double expected = 6.0 * std::sqrt(2.0 * 10.0) * u;

    EXPECT_NEAR(computeGamma(10, u), expected, 1e-10);
}

// --- Threshold boundary ---

TEST(TestComputeGamma, ThresholdBoundary_ExactlyAtThreshold)
{
    // Construct epsilon such that nU = 2*k*eps = exactly 0.01
    // k=1, eps=0.005 → nU = 0.01 → should use statistical (nU >= 0.01)
    const double eps = 0.005;
    const double expected = 6.0 * std::sqrt(2.0) * eps;

    EXPECT_NEAR(computeGamma(1, eps), expected, 1e-15);
}

TEST(TestComputeGamma, ThresholdBoundary_JustBelowThreshold)
{
    // k=1, eps=0.004999 → nU = 0.009998 < 0.01 → should use linear
    const double eps = 0.004999;
    const double nU = 2.0 * eps;
    const double expected = nU / (1.0 - nU);

    EXPECT_NEAR(computeGamma(1, eps), expected, 1e-15);
}

// --- Edge cases ---

TEST(TestComputeGamma, ZeroAccumulations)
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    // k=0: nU = 0 < 0.01 → linear: 0/(1-0) = 0
    EXPECT_DOUBLE_EQ(computeGamma(0, u), 0.0);
}

TEST(TestComputeGamma, ZeroEpsilon)
{
    // Hypothetical perfect precision: gamma should be 0
    EXPECT_DOUBLE_EQ(computeGamma(100, 0.0), 0.0);
}

// =================================================================================================
// TestValidateGamma
// =================================================================================================

TEST(TestValidateGamma, DoesNotThrowBelowThreshold)
{
    EXPECT_NO_THROW(validateGamma(0.0));
    EXPECT_NO_THROW(validateGamma(0.1));
    EXPECT_NO_THROW(validateGamma(0.499));
}

TEST(TestValidateGamma, ThrowsAtThreshold)
{
    EXPECT_THROW(validateGamma(0.5), std::overflow_error);
}

TEST(TestValidateGamma, ThrowsAboveThreshold)
{
    EXPECT_THROW(validateGamma(0.9), std::overflow_error);
    EXPECT_THROW(validateGamma(1.0), std::overflow_error);
}

TEST(TestValidateGamma, CustomThreshold)
{
    EXPECT_NO_THROW(validateGamma(0.09, 0.1));
    EXPECT_THROW(validateGamma(0.1, 0.1), std::overflow_error);
    EXPECT_THROW(validateGamma(0.5, 0.1), std::overflow_error);
}

// =================================================================================================
// TestComputeInputCastingError
// =================================================================================================

TEST(TestComputeInputCastingError, HigherPrecisionInput_ReturnsError)
{
    // double→float downcast: inputEps < computeEps → error = 2 * signal * computeEps
    auto computeEps = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double signal = 100.0;
    const double expected = 2.0 * signal * computeEps;

    EXPECT_NEAR((computeInputCastingError<double, float>(signal)), expected, 1e-15);
}

TEST(TestComputeInputCastingError, LowerPrecisionInput_ReturnsZero)
{
    // half→float upcast: inputEps > computeEps → no error
    EXPECT_DOUBLE_EQ((computeInputCastingError<half, float>(100.0)), 0.0);
}

TEST(TestComputeInputCastingError, SamePrecision_ReturnsZero)
{
    // float→float: no casting → no error
    EXPECT_DOUBLE_EQ((computeInputCastingError<float, float>(100.0)), 0.0);
}

TEST(TestComputeInputCastingError, SingleInputFactor)
{
    // Self-product (e.g., x^2): numDistinctInputs=1
    auto computeEps = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double signal = 50.0;
    const double expected = 1.0 * signal * computeEps;

    EXPECT_NEAR((computeInputCastingError<double, float>(signal, 1)), expected, 1e-15);
}

// =================================================================================================
// TestComputeOutputCastingError
// =================================================================================================

TEST(TestComputeOutputCastingError, LowerPrecisionOutput_ReturnsError)
{
    // float→half downcast: outputEps > computeEps → error = magnitude * outputEps
    auto outputEps = static_cast<double>(std::numeric_limits<half>::epsilon());
    const double magnitude = 200.0;
    const double expected = magnitude * outputEps;

    EXPECT_NEAR((computeOutputCastingError<half, float>(magnitude)), expected, 1e-15);
}

TEST(TestComputeOutputCastingError, HigherPrecisionOutput_ReturnsZero)
{
    // float→double upcast: outputEps < computeEps → no error
    EXPECT_DOUBLE_EQ((computeOutputCastingError<double, float>(200.0)), 0.0);
}

TEST(TestComputeOutputCastingError, SamePrecision_ReturnsZero)
{
    EXPECT_DOUBLE_EQ((computeOutputCastingError<float, float>(200.0)), 0.0);
}

TEST(TestComputeOutputCastingError, Bf16Output_ReturnsError)
{
    // float→bf16 downcast
    auto outputEps = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    const double magnitude = 100.0;
    const double expected = magnitude * outputEps;

    EXPECT_NEAR((computeOutputCastingError<bfloat16, float>(magnitude)), expected, 1e-15);
}

// =================================================================================================
// TestValidateToleranceRange
// =================================================================================================

TEST(TestValidateToleranceRange, WithinRange_NoThrow)
{
    EXPECT_NO_THROW((validateToleranceRange<float>(1.0)));
    EXPECT_NO_THROW((validateToleranceRange<half>(100.0)));
    EXPECT_NO_THROW((validateToleranceRange<bfloat16>(100.0)));
}

TEST(TestValidateToleranceRange, ExceedsHalfMax_Throws)
{
    // half max ≈ 65504
    EXPECT_THROW((validateToleranceRange<half>(70000.0)), std::overflow_error);
}

TEST(TestValidateToleranceRange, ExceedsBf16Max_Throws)
{
    // bf16 max ≈ 3.39e+38 (same exponent range as float, less mantissa)
    // Use a value larger than float max
    EXPECT_THROW((validateToleranceRange<bfloat16>(1e39)), std::overflow_error);
}

TEST(TestValidateToleranceRange, AtExactMax_NoThrow)
{
    auto halfMax = static_cast<double>(std::numeric_limits<half>::max());
    EXPECT_NO_THROW((validateToleranceRange<half>(halfMax)));
}
