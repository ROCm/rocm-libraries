// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E8M0.cpp
/// @brief Type-specific unit tests for fp8_e8m0 (MX scale format).
///
/// This file contains tests for fp8_e8m0-specific behavior that cannot be generalized:
/// - Unsigned type behavior (no sign bit, clamping of negative values)
/// - No zero representation (scale=0 = 2^-127)
/// - Power of 2 only representation
///
/// Generic tests (construction, conversion, numeric_limits basics, stream output)
/// are in TestPortableTypes.cpp. Cross-type conversions are tested in TestCrossTypeConversion.cpp.
///
/// @see fp8_e8m0 struct for format specification.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <vector>

using namespace hipdnn_data_sdk::types;

namespace
{
// E8M0 bit patterns for test values
constexpr uint8_t E8M0_BITS_HALF = 0x7E; // 2^-1 = 0.5
constexpr uint8_t E8M0_BITS_ONE = 0x7F; // 2^0  = 1.0
constexpr uint8_t E8M0_BITS_TWO = 0x80; // 2^1  = 2.0
constexpr uint8_t E8M0_BITS_FOUR = 0x81; // 2^2  = 4.0
constexpr uint8_t E8M0_BITS_MIN = 0x00; // 2^-127 (minimum positive value)
constexpr uint8_t E8M0_BITS_MAX = 0xFE; // 2^127 (maximum value)
constexpr uint8_t E8M0_BITS_NAN = 0xFF; // NaN
} // anonymous namespace

// ============================================================================
// Construction Tests (E8M0-specific behavior)
// ============================================================================

TEST(TestFp8E8M0, DefaultConstruction)
{
    const fp8_e8m0 val;
    EXPECT_EQ(val.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, ConstructFromFloatBitPatterns)
{
    // Test specific E8M0 bit patterns for powers of 2
    const fp8_e8m0 half(0.5f);
    EXPECT_EQ(half.data, E8M0_BITS_HALF);

    const fp8_e8m0 one(1.0f);
    EXPECT_EQ(one.data, E8M0_BITS_ONE);

    const fp8_e8m0 two(2.0f);
    EXPECT_EQ(two.data, E8M0_BITS_TWO);

    const fp8_e8m0 four(4.0f);
    EXPECT_EQ(four.data, E8M0_BITS_FOUR);
}

TEST(TestFp8E8M0, CopyAssignment)
{
    const fp8_e8m0 a(2.0f);
    fp8_e8m0 b;
    b = a;
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Power of 2 Representation Tests (E8M0-specific)
// ============================================================================

TEST(TestFp8E8M0, RoundTripConversion)
{
    // Test that powers of 2 round-trip correctly
    const std::vector<float> powersOfTwo = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};

    for(const float val : powersOfTwo)
    {
        const fp8_e8m0 e8m0(val);
        auto result = static_cast<float>(e8m0);
        EXPECT_EQ(result, val) << "Failed for value: " << val;
    }
}

TEST(TestFp8E8M0, NonPowerOfTwoTruncation)
{
    // E8M0 truncates to the lower power of 2 (floor behavior, not rounding)
    // 3.0 is between 2 (2^1) and 4 (2^2) - truncates to 2.0
    const fp8_e8m0 three(3.0f);
    EXPECT_EQ(static_cast<float>(three), 2.0f);

    // 3.9 is very close to 4.0 but still truncates to 2.0
    const fp8_e8m0 almostFour(3.9f);
    EXPECT_EQ(static_cast<float>(almostFour), 2.0f);
}

TEST(TestFp8E8M0, MaximumValue)
{
    // Maximum value: exponent = 254, value = 2^(254-127) = 2^127
    const fp8_e8m0 maxVal = fp8_e8m0::from_bits(E8M0_BITS_MAX);
    EXPECT_EQ(static_cast<float>(maxVal), 0x1p127f);
}

// ============================================================================
// Special Values Tests (E8M0-specific: no zero, no sign, no infinity)
// ============================================================================

TEST(TestFp8E8M0, MinimumValue)
{
    // E8M0 has no zero - scale=0 represents min (2^-127)
    const fp8_e8m0 minVal = fp8_e8m0::from_bits(0x00);
    EXPECT_EQ(minVal.data, E8M0_BITS_MIN);

    auto minFloat = static_cast<float>(minVal);
    EXPECT_EQ(minFloat, 0x1p-127f);

    // Round-trip: convert float back to fp8_e8m0
    const fp8_e8m0 roundTrip(minFloat);
    EXPECT_EQ(roundTrip.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, Signbit)
{
    // E8M0 is unsigned - signbit always returns false
    EXPECT_FALSE(signbit(fp8_e8m0::from_bits(E8M0_BITS_MIN)));
    EXPECT_FALSE(signbit(fp8_e8m0::from_bits(E8M0_BITS_MAX)));
    EXPECT_FALSE(signbit(fp8_e8m0::from_bits(E8M0_BITS_NAN))); // even NaN
}

TEST(TestFp8E8M0, Isinf)
{
    // E8M0 has no infinity - isinf always returns false
    EXPECT_FALSE(isinf(fp8_e8m0::from_bits(E8M0_BITS_MIN)));
    EXPECT_FALSE(isinf(fp8_e8m0::from_bits(E8M0_BITS_MAX)));
    EXPECT_FALSE(isinf(fp8_e8m0::from_bits(E8M0_BITS_NAN))); // NaN is not infinity
}

TEST(TestFp8E8M0, Isfinite)
{
    // isfinite returns true for all values except NaN
    EXPECT_TRUE(isfinite(fp8_e8m0::from_bits(E8M0_BITS_MIN)));
    EXPECT_TRUE(isfinite(fp8_e8m0::from_bits(E8M0_BITS_MAX)));
    EXPECT_FALSE(isfinite(fp8_e8m0::from_bits(E8M0_BITS_NAN))); // NaN
}

// ============================================================================
// Clamping Tests (E8M0-specific: unsigned type)
// ============================================================================

TEST(TestFp8E8M0, NegativeValuesClampToMin)
{
    // E8M0 is unsigned - negative float values clamp to min
    const fp8_e8m0 neg(-1.0f);
    EXPECT_EQ(neg.data, E8M0_BITS_MIN);

    const fp8_e8m0 negLarge(-1000.0f);
    EXPECT_EQ(negLarge.data, E8M0_BITS_MIN);

    // Negative integral values also clamp to min
    const fp8_e8m0 negInt(-4);
    EXPECT_EQ(negInt.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, ZeroClampedToMin)
{
    // E8M0 has no zero representation - 0.0f clamps to min
    const fp8_e8m0 zero(0.0f);
    EXPECT_EQ(zero.data, E8M0_BITS_MIN);

    // Negative zero also clamps to min
    const fp8_e8m0 negZero(-0.0f);
    EXPECT_EQ(negZero.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, VerySmallFloatClampedToMin)
{
    // Float denormal values clamp to min
    const fp8_e8m0 tiny(1e-40f);
    EXPECT_EQ(tiny.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, InfinityClampedToMax)
{
    const fp8_e8m0 posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(posInf.data, E8M0_BITS_MAX); // Max value
}

TEST(TestFp8E8M0, NegativeInfinityClampedToMin)
{
    // E8M0 is unsigned - negative infinity clamps to min
    const fp8_e8m0 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(negInf.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, NaNFromFloat)
{
    const fp8_e8m0 nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(nan.data, E8M0_BITS_NAN);
    EXPECT_TRUE(isnan(nan));
}

// ============================================================================
// numeric_limits Tests (E8M0-specific)
// ============================================================================

TEST(TestFp8E8M0, NumericLimitsInfinityReturnsMax)
{
    // E8M0 has no infinity, so infinity() returns max()
    const fp8_e8m0 inf = std::numeric_limits<fp8_e8m0>::infinity();
    const fp8_e8m0 maxVal = std::numeric_limits<fp8_e8m0>::max();
    EXPECT_EQ(inf.data, maxVal.data);
}

TEST(TestFp8E8M0, NumericLimitsEpsilonValue)
{
    // E8M0 epsilon = 1.0 (smallest difference at 1.0 is to 2.0)
    const fp8_e8m0 eps = std::numeric_limits<fp8_e8m0>::epsilon();
    EXPECT_EQ(eps.data, E8M0_BITS_ONE); // scale=127 = 1.0
    EXPECT_EQ(static_cast<float>(eps), 1.0f);
}

TEST(TestFp8E8M0, NumericLimitsDenormMin)
{
    // E8M0 has no denormals, denorm_min equals min
    const fp8_e8m0 denormMin = std::numeric_limits<fp8_e8m0>::denorm_min();
    const fp8_e8m0 minVal = std::numeric_limits<fp8_e8m0>::min();
    EXPECT_EQ(denormMin.data, minVal.data);
}

TEST(TestFp8E8M0, NumericLimitsSignalingNaN)
{
    // E8M0 has only one NaN representation
    const fp8_e8m0 snan = std::numeric_limits<fp8_e8m0>::signaling_NaN();
    EXPECT_TRUE(isnan(snan));
    EXPECT_EQ(snan.data, E8M0_BITS_NAN);
}

TEST(TestFp8E8M0, LowestEqualsMin)
{
    // E8M0 is unsigned and has no zero, so lowest() equals min()
    EXPECT_EQ(std::numeric_limits<fp8_e8m0>::lowest().data,
              std::numeric_limits<fp8_e8m0>::min().data);
}

TEST(TestFp8E8M0, NumericLimitsRoundError)
{
    const fp8_e8m0 roundErr = std::numeric_limits<fp8_e8m0>::round_error();
    EXPECT_EQ(roundErr.data, E8M0_BITS_ONE);
    EXPECT_EQ(static_cast<float>(roundErr), 1.0f);
}
