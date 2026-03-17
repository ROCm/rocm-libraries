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
    fp8_e8m0 const val;
    EXPECT_EQ(val.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, ConstructFromFloatBitPatterns)
{
    // Test specific E8M0 bit patterns for powers of 2
    fp8_e8m0 const half(0.5f);
    EXPECT_EQ(half.data, E8M0_BITS_HALF);

    fp8_e8m0 const one(1.0f);
    EXPECT_EQ(one.data, E8M0_BITS_ONE);

    fp8_e8m0 const two(2.0f);
    EXPECT_EQ(two.data, E8M0_BITS_TWO);

    fp8_e8m0 const four(4.0f);
    EXPECT_EQ(four.data, E8M0_BITS_FOUR);
}

TEST(TestFp8E8M0, CopyAssignment)
{
    fp8_e8m0 const a(2.0f);
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
    std::vector<float> const powersOfTwo = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};

    for(float const val : powersOfTwo)
    {
        fp8_e8m0 const e8m0(val);
        auto result = static_cast<float>(e8m0);
        EXPECT_EQ(result, val) << "Failed for value: " << val;
    }
}

TEST(TestFp8E8M0, NonPowerOfTwoTruncation)
{
    // E8M0 truncates to the lower power of 2 (floor behavior, not rounding)
    // 3.0 is between 2 (2^1) and 4 (2^2) - truncates to 2.0
    fp8_e8m0 const three(3.0f);
    EXPECT_EQ(static_cast<float>(three), 2.0f);

    // 3.9 is very close to 4.0 but still truncates to 2.0
    fp8_e8m0 const almostFour(3.9f);
    EXPECT_EQ(static_cast<float>(almostFour), 2.0f);
}

TEST(TestFp8E8M0, MaximumValue)
{
    // Maximum value: exponent = 254, value = 2^(254-127) = 2^127
    fp8_e8m0 const maxVal = fp8_e8m0::from_bits(E8M0_BITS_MAX);
    EXPECT_EQ(static_cast<float>(maxVal), 0x1p127f);
}

// ============================================================================
// Special Values Tests (E8M0-specific: no zero, no sign, no infinity)
// ============================================================================

TEST(TestFp8E8M0, MinimumValue)
{
    // E8M0 has no zero - scale=0 represents min (2^-127)
    fp8_e8m0 const minVal = fp8_e8m0::from_bits(0x00);
    EXPECT_EQ(minVal.data, E8M0_BITS_MIN);

    auto minFloat = static_cast<float>(minVal);
    EXPECT_EQ(minFloat, 0x1p-127f);

    // Round-trip: convert float back to fp8_e8m0
    fp8_e8m0 const roundTrip(minFloat);
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
    fp8_e8m0 const neg(-1.0f);
    EXPECT_EQ(neg.data, E8M0_BITS_MIN);

    fp8_e8m0 const negLarge(-1000.0f);
    EXPECT_EQ(negLarge.data, E8M0_BITS_MIN);

    // Negative integral values also clamp to min
    fp8_e8m0 const negInt(-4);
    EXPECT_EQ(negInt.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, ZeroClampedToMin)
{
    // E8M0 has no zero representation - 0.0f clamps to min
    fp8_e8m0 const zero(0.0f);
    EXPECT_EQ(zero.data, E8M0_BITS_MIN);

    // Negative zero also clamps to min
    fp8_e8m0 const negZero(-0.0f);
    EXPECT_EQ(negZero.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, VerySmallFloatClampedToMin)
{
    // Float denormal values clamp to min
    fp8_e8m0 const tiny(1e-40f);
    EXPECT_EQ(tiny.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, InfinityClampedToMax)
{
    fp8_e8m0 const posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(posInf.data, E8M0_BITS_MAX); // Max value
}

TEST(TestFp8E8M0, NegativeInfinityClampedToMin)
{
    // E8M0 is unsigned - negative infinity clamps to min
    fp8_e8m0 const negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(negInf.data, E8M0_BITS_MIN);
}

TEST(TestFp8E8M0, NaNFromFloat)
{
    fp8_e8m0 const nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(nan.data, E8M0_BITS_NAN);
    EXPECT_TRUE(isnan(nan));
}

// ============================================================================
// numeric_limits Tests (E8M0-specific)
// ============================================================================

TEST(TestFp8E8M0, NumericLimitsInfinityReturnsMax)
{
    // E8M0 has no infinity, so infinity() returns max()
    fp8_e8m0 const inf = std::numeric_limits<fp8_e8m0>::infinity();
    fp8_e8m0 const maxVal = std::numeric_limits<fp8_e8m0>::max();
    EXPECT_EQ(inf.data, maxVal.data);
}

TEST(TestFp8E8M0, NumericLimitsEpsilonValue)
{
    // E8M0 epsilon = 1.0 (smallest difference at 1.0 is to 2.0)
    fp8_e8m0 const eps = std::numeric_limits<fp8_e8m0>::epsilon();
    EXPECT_EQ(eps.data, E8M0_BITS_ONE); // scale=127 = 1.0
    EXPECT_EQ(static_cast<float>(eps), 1.0f);
}

TEST(TestFp8E8M0, NumericLimitsDenormMin)
{
    // E8M0 has no denormals, denorm_min equals min
    fp8_e8m0 const denormMin = std::numeric_limits<fp8_e8m0>::denorm_min();
    fp8_e8m0 const minVal = std::numeric_limits<fp8_e8m0>::min();
    EXPECT_EQ(denormMin.data, minVal.data);
}

TEST(TestFp8E8M0, NumericLimitsSignalingNaN)
{
    // E8M0 has only one NaN representation
    fp8_e8m0 const snan = std::numeric_limits<fp8_e8m0>::signaling_NaN();
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
    fp8_e8m0 const roundErr = std::numeric_limits<fp8_e8m0>::round_error();
    EXPECT_EQ(roundErr.data, E8M0_BITS_ONE);
    EXPECT_EQ(static_cast<float>(roundErr), 1.0f);
}
