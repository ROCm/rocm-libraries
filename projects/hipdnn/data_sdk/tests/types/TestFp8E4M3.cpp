// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E4M3.cpp
/// @brief Type-specific tests for fp8_e4m3 (OCP E4M3 format).
///
/// Common tests (type properties, construction, arithmetic, comparison, math functions,
/// numeric_limits) are in TestPortableTypes.cpp. This file contains fp8_e4m3-specific
/// tests for no-infinity behavior, saturation, and specific numeric_limits values.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

using hipdnn_data_sdk::types::fp8_e4m3;
using namespace hipdnn_data_sdk::types;

class TestFp8E4M3 : public ::testing::Test
{
protected:
    // FP8 E4M3 has limited precision, use larger tolerance
    static constexpr float K_TOLERANCE = 0.2f;

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(fp8_e4m3 a, fp8_e4m3 b, float tol = K_TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

// ============================================================================
// E4M3-Specific: No Infinity
// ============================================================================

TEST_F(TestFp8E4M3, NoInfinity)
{
    // E4M3 OCP format has no infinity, large values saturate to max
    const fp8_e4m3 nan = fp8_e4m3::from_bits(0x7F);
    EXPECT_FALSE(isinf(nan)); // 0x7F is NaN, not infinity
    EXPECT_TRUE(isnan(nan));
}

// ============================================================================
// Saturation Tests (E4M3 specific)
// ============================================================================

TEST_F(TestFp8E4M3, SaturationOnOverflow)
{
    // Values beyond 448 should saturate to max
    const fp8_e4m3 large(1000.0f);
    EXPECT_TRUE(isfinite(large));
    EXPECT_TRUE(nearEqual(static_cast<float>(large), 448.0f, 10.0f));
}

TEST_F(TestFp8E4M3, MaxRepresentableValue)
{
    // Test max representable value (448)
    const fp8_e4m3 d(448.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 448.0f, 1.0f));
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST_F(TestFp8E4M3, NumericLimitsSpecificValues)
{
    // E4M3 OCP max is 0x7E = (1 + 6/8) * 2^8 = 1.75 * 256 = 448.0
    const fp8_e4m3 maxVal = std::numeric_limits<fp8_e4m3>::max();
    auto maxFloat = static_cast<float>(maxVal);
    EXPECT_EQ(maxFloat, 448.0f);
    EXPECT_EQ(maxVal.data, 0x7E);

    // E4M3 min (smallest positive normal) is 0x08 = 2^(1-7) = 2^-6 = 0.015625
    const fp8_e4m3 minVal = std::numeric_limits<fp8_e4m3>::min();
    auto minFloat = static_cast<float>(minVal);
    EXPECT_EQ(minFloat, 0.015625f);
    EXPECT_EQ(minVal.data, 0x08);

    // E4M3 lowest is -max = 0xFE = -448.0
    const fp8_e4m3 lowestVal = std::numeric_limits<fp8_e4m3>::lowest();
    auto lowestFloat = static_cast<float>(lowestVal);
    EXPECT_EQ(lowestFloat, -448.0f);
    EXPECT_EQ(lowestVal.data, 0xFE);

    // E4M3 epsilon is 2^-3 = 0.125 (3 mantissa bits)
    const fp8_e4m3 eps = std::numeric_limits<fp8_e4m3>::epsilon();
    auto epsFloat = static_cast<float>(eps);
    EXPECT_EQ(epsFloat, 0.125f);

    // E4M3 round_error is 0.5
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp8_e4m3>::round_error()), 0.5f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST_F(TestFp8E4M3, NamedConstants)
{
    // Access named constants via numeric_limits
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e4m3>::quiet_NaN()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::max()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::min()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::lowest()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::epsilon()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::denorm_min()));
}

// ============================================================================
// Construction Tests with Specific Values
// ============================================================================

TEST_F(TestFp8E4M3, ConstructSubnormal)
{
    const fp8_e4m3 val(0x1p-7f);
    EXPECT_EQ(static_cast<float>(val), 0x1p-7f);
}

TEST_F(TestFp8E4M3, ConstructFromInt64)
{
    const fp8_e4m3 d(int64_t{16});
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 16.0f));
}
