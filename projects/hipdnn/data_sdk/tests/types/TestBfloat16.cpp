// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestBfloat16.cpp
/// @brief Type-specific tests for bfloat16.
///
/// Common tests (type properties, construction, arithmetic, comparison, math functions,
/// numeric_limits) are in TestPortableTypes.cpp. This file contains bfloat16-specific
/// tests for rounding modes, truncation, and specific numeric_limits values.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

using hipdnn_data_sdk::types::bfloat16;
using namespace hipdnn_data_sdk::types;

class TestBfloat16 : public ::testing::Test
{
protected:
    static constexpr float K_TOLERANCE = 0.01f;

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(bfloat16 a, bfloat16 b, float tol = K_TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

// ============================================================================
// Round-to-Nearest-Even Tests
// ============================================================================

TEST_F(TestBfloat16, RoundToNearestEvenRoundDown)
{
    // Test value that should round down (remainder < 0.5 in the truncated bits)
    // 1.0 in float32: 0x3F800000
    // Adding a small value that's less than half the bfloat16 LSB
    // bfloat16 1.0 = 0x3F80, next value = 0x3F81
    // Half the distance = 0x00008000 in the lower 16 bits
    // Value with 0x00004000 in lower bits should round down
    uint32_t floatBits = 0x3F804000; // 1.0 + small amount (rounds down)
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    const bfloat16 bf(f);
    EXPECT_EQ(bf.data, 0x3F80); // Should round down to 1.0
}

TEST_F(TestBfloat16, RoundToNearestEvenRoundUp)
{
    // Test value that should round up (remainder > 0.5 in the truncated bits)
    // Value with 0x0000C000 in lower bits should round up
    uint32_t floatBits = 0x3F80C000; // 1.0 + larger amount (rounds up)
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    const bfloat16 bf(f);
    EXPECT_EQ(bf.data, 0x3F81); // Should round up
}

TEST_F(TestBfloat16, RoundToNearestEvenTieToEven)
{
    // Test tie-breaking: exactly 0.5 between two values should round to even
    // Value ending in ...0 with exactly 0x8000 remainder should stay (already even)
    uint32_t floatBitsEven = 0x3F808000; // Tie, result LSB is 0 -> stays at 0x3F80
    float fEven;
    std::memcpy(&fEven, &floatBitsEven, sizeof(float));
    const bfloat16 bfEven(fEven);
    EXPECT_EQ(bfEven.data, 0x3F80); // Tie breaks to even (LSB = 0)

    // Value ending in ...1 with exactly 0x8000 remainder should round up
    uint32_t floatBitsOdd = 0x3F818000; // Tie, result LSB would be 1 -> rounds up to even
    float fOdd;
    std::memcpy(&fOdd, &floatBitsOdd, sizeof(float));
    const bfloat16 bfOdd(fOdd);
    EXPECT_EQ(bfOdd.data, 0x3F82); // Tie breaks to even (rounds up)
}

TEST_F(TestBfloat16, TruncationVsRNE)
{
    // Test that truncation and RNE produce different results for appropriate inputs
    // Value with 0x8000 remainder where LSB is 1: RNE rounds up, truncation doesn't
    uint32_t floatBits = 0x3F818000; // LSB=1, remainder=0x8000
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));

    // RNE should round up (tie-break to even)
    const uint16_t rneResult = hipdnn_data_sdk::types::detail::float_to_bfloat16_bits_rne(f);
    EXPECT_EQ(rneResult, 0x3F82);

    // Truncation should not round
    const uint16_t truncResult = hipdnn_data_sdk::types::detail::float_to_bfloat16_bits_truncate(f);
    EXPECT_EQ(truncResult, 0x3F81);

    // Default should match RNE
    const uint16_t defaultResult = hipdnn_data_sdk::types::detail::float_to_bfloat16_bits(f);
    EXPECT_EQ(defaultResult, rneResult);
}

// ============================================================================
// bfloat16_truncate Type Tests
// ============================================================================

TEST_F(TestBfloat16, Bfloat16TruncateType)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;
    using hipdnn_data_sdk::types::Bfloat16RoundingMode;

    // Verify type properties
    EXPECT_EQ(sizeof(bfloat16_truncate), 2);
    EXPECT_TRUE(std::is_trivially_copyable_v<bfloat16_truncate>);
    EXPECT_TRUE(std::is_standard_layout_v<bfloat16_truncate>);

    // Verify rounding mode is set correctly
    EXPECT_EQ(bfloat16_truncate::rounding_mode, Bfloat16RoundingMode::Truncate);
    EXPECT_EQ(bfloat16::rounding_mode, Bfloat16RoundingMode::RNE);
}

TEST_F(TestBfloat16, Bfloat16TruncateRounding)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;

    // Value with 0x8000 remainder where LSB is 1: RNE rounds up, truncation doesn't
    uint32_t floatBits = 0x3F818000;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));

    // RNE type should round up
    const bfloat16 rneVal(f);
    EXPECT_EQ(rneVal.data, 0x3F82);

    // Truncate type should not round
    const bfloat16_truncate truncVal(f);
    EXPECT_EQ(truncVal.data, 0x3F81);
}

TEST_F(TestBfloat16, Bfloat16TruncateInterop)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;

    // Test implicit conversion between rounding modes
    const bfloat16 rneVal = bfloat16::from_bits(0x4000); // 2.0
    const bfloat16_truncate truncVal = rneVal; // Implicit conversion
    EXPECT_EQ(truncVal.data, rneVal.data);

    // Convert back
    const bfloat16 backToRne = truncVal;
    EXPECT_EQ(backToRne.data, rneVal.data);

    // Both should convert to the same float
    EXPECT_EQ(static_cast<float>(rneVal), static_cast<float>(truncVal));
}

TEST_F(TestBfloat16, Bfloat16TruncateMathFunctions)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;

    const bfloat16_truncate a(-5.0f);
    const bfloat16_truncate b(4.0f);

    // Math functions should work with truncate type
    EXPECT_TRUE(nearEqual(static_cast<float>(abs(a)), 5.0f));
    EXPECT_TRUE(nearEqual(static_cast<float>(sqrt(b)), 2.0f));
    EXPECT_FALSE(isnan(a));
    EXPECT_TRUE(isfinite(a));
}

TEST_F(TestBfloat16, Bfloat16TruncateNumericLimits)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;

    // numeric_limits should work for truncate type too
    EXPECT_TRUE(std::numeric_limits<bfloat16_truncate>::is_specialized);
    EXPECT_EQ(std::numeric_limits<bfloat16_truncate>::max().data, 0x7F7F);
    EXPECT_EQ(std::numeric_limits<bfloat16_truncate>::min().data, 0x0080);
    EXPECT_EQ(std::numeric_limits<bfloat16_truncate>::lowest().data, 0xFF7F);
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST_F(TestBfloat16, NumericLimitsSpecificValues)
{
    // bfloat16 max is 0x7F7F = (2 - 2^-7) * 2^127 ≈ 3.3895e+38
    const bfloat16 maxVal = std::numeric_limits<bfloat16>::max();
    auto maxFloat = static_cast<float>(maxVal);
    EXPECT_GT(maxFloat, 3.3e38f);
    EXPECT_LT(maxFloat, std::numeric_limits<float>::max());
    EXPECT_EQ(maxVal.data, 0x7F7F);

    // bfloat16 min (smallest positive normal) is 0x0080 = 2^-126 ≈ 1.175e-38
    const bfloat16 minVal = std::numeric_limits<bfloat16>::min();
    auto minFloat = static_cast<float>(minVal);
    EXPECT_GT(minFloat, 1.1e-38f);
    EXPECT_LT(minFloat, 1.2e-38f);
    EXPECT_EQ(minVal.data, 0x0080);

    // bfloat16 lowest is -max = 0xFF7F ≈ -3.3895e+38
    const bfloat16 lowestVal = std::numeric_limits<bfloat16>::lowest();
    auto lowestFloat = static_cast<float>(lowestVal);
    EXPECT_LT(lowestFloat, -3.3e38f);
    EXPECT_GT(lowestFloat, -std::numeric_limits<float>::max());
    EXPECT_EQ(lowestVal.data, 0xFF7F);

    // bfloat16 epsilon is 2^-7 = 0.0078125
    const bfloat16 eps = std::numeric_limits<bfloat16>::epsilon();
    auto epsFloat = static_cast<float>(eps);
    EXPECT_TRUE(nearEqual(epsFloat, 0.0078125f, 0.0001f));
    EXPECT_EQ(eps.data, 0x3C00);

    // bfloat16 round_error is 0.5
    EXPECT_EQ(static_cast<float>(std::numeric_limits<bfloat16>::round_error()), 0.5f);
}

// ============================================================================
// Large Value Construction Tests
// ============================================================================

TEST_F(TestBfloat16, ConstructLargeValues)
{
    const bfloat16 d(1e10f);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1e10f, 1e8f));

    const bfloat16 b(3.14159265358979);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 3.14159f, 0.02f));
}

TEST_F(TestBfloat16, ConstructFromInt64)
{
    const bfloat16 d(int64_t{1000});
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1000.0f));
}

// ============================================================================
// Additional Math Function Tests
// ============================================================================

TEST_F(TestBfloat16, Pow)
{
    const bfloat16 base(2.0f);
    const bfloat16 exp(3.0f);
    EXPECT_TRUE(nearEqual(pow(base, exp), bfloat16(8.0f)));
}

TEST_F(TestBfloat16, Copysign)
{
    const bfloat16 a(3.0f);
    const bfloat16 b(-1.0f);
    EXPECT_TRUE(nearEqual(copysign(a, b), bfloat16(-3.0f)));
    EXPECT_TRUE(nearEqual(copysign(b, a), bfloat16(1.0f)));
}

TEST_F(TestBfloat16, Sin)
{
    const bfloat16 a(0.0f);
    EXPECT_TRUE(nearEqual(sin(a), bfloat16(0.0f)));
}

TEST_F(TestBfloat16, Cos)
{
    const bfloat16 a(0.0f);
    EXPECT_TRUE(nearEqual(cos(a), bfloat16(1.0f)));
}

TEST_F(TestBfloat16, Fma)
{
    const bfloat16 a(2.0f);
    const bfloat16 b(3.0f);
    const bfloat16 c(1.0f);
    EXPECT_TRUE(nearEqual(fma(a, b, c), bfloat16(7.0f)));
}
