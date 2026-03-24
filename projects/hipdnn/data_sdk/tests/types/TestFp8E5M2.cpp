// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E5M2.cpp
/// @brief Type-specific tests for fp8_e5m2 (OCP E5M2 format).
///
/// Common tests (type properties, construction, arithmetic, comparison, math functions,
/// numeric_limits) are in TestPortableTypes.cpp. This file contains fp8_e5m2-specific
/// tests for specific numeric_limits values and named constants.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

using hipdnn_data_sdk::types::fp8_e5m2;
using namespace hipdnn_data_sdk::types;

class TestFp8E5M2 : public ::testing::Test
{
protected:
    // FP8 E5M2 has limited precision (only 2 mantissa bits), use larger tolerance
    static constexpr float K_TOLERANCE = 0.5f;

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(fp8_e5m2 a, fp8_e5m2 b, float tol = K_TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

// ============================================================================
// E5M2-Specific Tests
// ============================================================================

TEST_F(TestFp8E5M2, SignalingNaN)
{
    const fp8_e5m2 snan = fp8_e5m2::from_bits(0x7D);
    EXPECT_TRUE(isnan(snan));
}

// ============================================================================
// Wide Dynamic Range Test
// ============================================================================

TEST_F(TestFp8E5M2, WideDynamicRange)
{
    // Test a larger value (E5M2 has wider range than E4M3)
    const fp8_e5m2 d(1024.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1024.0f, 128.0f));
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST_F(TestFp8E5M2, NumericLimitsSpecificValues)
{
    // E5M2 max is 0x7B = (1 + 3/4) * 2^15 = 1.75 * 32768 = 57344.0
    const fp8_e5m2 maxVal = std::numeric_limits<fp8_e5m2>::max();
    auto maxFloat = static_cast<float>(maxVal);
    EXPECT_EQ(maxFloat, 57344.0f);
    EXPECT_EQ(maxVal.data, 0x7B);

    // E5M2 min (smallest positive normal) is 0x04 = 2^(1-15) = 2^-14 ≈ 6.1035e-5
    const fp8_e5m2 minVal = std::numeric_limits<fp8_e5m2>::min();
    auto minFloat = static_cast<float>(minVal);
    EXPECT_GT(minFloat, 6.0e-5f);
    EXPECT_LT(minFloat, 6.2e-5f);
    EXPECT_EQ(minVal.data, 0x04);

    // E5M2 lowest is -max = 0xFB = -57344.0
    const fp8_e5m2 lowestVal = std::numeric_limits<fp8_e5m2>::lowest();
    auto lowestFloat = static_cast<float>(lowestVal);
    EXPECT_EQ(lowestFloat, -57344.0f);
    EXPECT_EQ(lowestVal.data, 0xFB);

    // E5M2 epsilon is 2^-2 = 0.25 (2 mantissa bits)
    const fp8_e5m2 eps = std::numeric_limits<fp8_e5m2>::epsilon();
    auto epsFloat = static_cast<float>(eps);
    EXPECT_EQ(epsFloat, 0.25f);

    // E5M2 round_error is 0.5
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp8_e5m2>::round_error()), 0.5f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST_F(TestFp8E5M2, NamedConstants)
{
    // Access named constants via numeric_limits
    EXPECT_TRUE(isinf(std::numeric_limits<fp8_e5m2>::infinity()));
    EXPECT_FALSE(signbit(std::numeric_limits<fp8_e5m2>::infinity()));
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e5m2>::quiet_NaN()));
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e5m2>::signaling_NaN()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::max()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::min()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::lowest()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::epsilon()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::denorm_min()));
}

// ============================================================================
// Construction Tests with Specific Values
// ============================================================================

TEST_F(TestFp8E5M2, ConstructSubnormal)
{
    const fp8_e5m2 val(0x1p-15f);
    EXPECT_EQ(static_cast<float>(val), 0x1p-15f);
}

TEST_F(TestFp8E5M2, ConstructFromInt64)
{
    const fp8_e5m2 d(int64_t{16});
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 16.0f));
}
