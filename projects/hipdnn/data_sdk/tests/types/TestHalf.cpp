// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestHalf.cpp
/// @brief Type-specific tests for half (FP16).
///
/// Common tests (type properties, construction, arithmetic, comparison, math functions,
/// numeric_limits) are in TestPortableTypes.cpp. This file contains half-specific tests
/// for specific numeric_limits values and named constants.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

using hipdnn_data_sdk::types::half;
using namespace hipdnn_data_sdk::types;

class TestHalf : public ::testing::Test
{
protected:
    static constexpr float K_TOLERANCE = 0.001f;

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(half a, half b, float tol = K_TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST_F(TestHalf, NumericLimitsSpecificValues)
{
    // half max is 0x7BFF = (2 - 2^-10) * 2^15 = 65504.0 exactly
    const half maxVal = std::numeric_limits<half>::max();
    auto maxFloat = static_cast<float>(maxVal);
    EXPECT_EQ(maxFloat, 65504.0f);
    EXPECT_EQ(maxVal.data, 0x7BFF);

    // half min (smallest positive normal) is 0x0400 = 2^-14 ≈ 6.1035e-5
    const half minVal = std::numeric_limits<half>::min();
    auto minFloat = static_cast<float>(minVal);
    EXPECT_GT(minFloat, 6.0e-5f);
    EXPECT_LT(minFloat, 6.2e-5f);
    EXPECT_EQ(minVal.data, 0x0400);

    // half lowest is -max = 0xFBFF = -65504.0 exactly
    const half lowestVal = std::numeric_limits<half>::lowest();
    auto lowestFloat = static_cast<float>(lowestVal);
    EXPECT_EQ(lowestFloat, -65504.0f);
    EXPECT_EQ(lowestVal.data, 0xFBFF);

    // half epsilon is 2^-10 ≈ 0.0009765625
    const half eps = std::numeric_limits<half>::epsilon();
    auto epsFloat = static_cast<float>(eps);
    EXPECT_TRUE(nearEqual(epsFloat, 0.0009765625f, 0.0001f));
    EXPECT_EQ(eps.data, 0x1400);

    // half round_error is 0.5
    EXPECT_EQ(static_cast<float>(std::numeric_limits<half>::round_error()), 0.5f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST_F(TestHalf, NamedConstants)
{
    // Access named constants via numeric_limits
    EXPECT_TRUE(isinf(std::numeric_limits<half>::infinity()));
    EXPECT_FALSE(signbit(std::numeric_limits<half>::infinity()));
    EXPECT_TRUE(isnan(std::numeric_limits<half>::quiet_NaN()));
    EXPECT_TRUE(isnan(std::numeric_limits<half>::signaling_NaN()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::max()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::min()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::lowest()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::epsilon()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::denorm_min()));
}

// ============================================================================
// Large Value Construction Tests
// ============================================================================

TEST_F(TestHalf, ConstructLargeValues)
{
    const half d(1000.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1000.0f, 1.0f));

    const half b(3.14159265358979);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 3.14159f, 0.002f));
}

TEST_F(TestHalf, ConstructFromInt64)
{
    const half d(int64_t{1000});
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1000.0f, 1.0f));
}

// ============================================================================
// Additional Math Function Tests
// ============================================================================

TEST_F(TestHalf, Pow)
{
    const half base(2.0f);
    const half exponent(3.0f);
    EXPECT_TRUE(nearEqual(pow(base, exponent), half(8.0f)));
}

TEST_F(TestHalf, Copysign)
{
    const half a(3.0f);
    const half b(-1.0f);
    EXPECT_TRUE(nearEqual(copysign(a, b), half(-3.0f)));
    EXPECT_TRUE(nearEqual(copysign(b, a), half(1.0f)));
}

TEST_F(TestHalf, Sin)
{
    const half a(0.0f);
    EXPECT_TRUE(nearEqual(sin(a), half(0.0f)));
}

TEST_F(TestHalf, Cos)
{
    const half a(0.0f);
    EXPECT_TRUE(nearEqual(cos(a), half(1.0f)));
}

TEST_F(TestHalf, Fma)
{
    const half a(2.0f);
    const half b(3.0f);
    const half c(1.0f);
    EXPECT_TRUE(nearEqual(fma(a, b, c), half(7.0f)));
}

// ============================================================================
// Signaling NaN Test
// ============================================================================

TEST_F(TestHalf, SignalingNaN)
{
    const half snan = half::from_bits(0x7C01);
    EXPECT_TRUE(isnan(snan));
}
