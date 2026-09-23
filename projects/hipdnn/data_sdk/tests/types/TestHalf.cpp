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
#include <cstdint>
#include <cstring>
#include <ios>
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

// ============================================================================
// Subnormal (denormal) Conversion Tests
// ============================================================================
//
// Half subnormals are the 1023 values k * 2^-24 for k in [1, 1023]. Every float in
// (2^-25, 2^-14) must convert to one of them; a conversion that returns zero instead
// silently destroys small-magnitude data (regression guard for the double-shifted
// significand in float_to_half_bits).

namespace
{
/// Exact float equal to k * 2^-24 (k <= 2^24, so the product is exact in float).
float subnormalUnits(int k)
{
    return std::ldexp(static_cast<float>(k), -24);
}

/// Exact float equal to the midpoint between (k) and (k + 1) subnormal units.
float subnormalMidpoint(int k)
{
    return std::ldexp(static_cast<float>(2 * k + 1), -25);
}

/// float with the given IEEE 754 bit pattern.
float floatFromBits(uint32_t bits)
{
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
} // namespace

TEST_F(TestHalf, SubnormalIsNotFlushedToZero)
{
    // Values well inside the subnormal range must not collapse to zero.
    EXPECT_NE(half(1e-5f).data & 0x7FFFU, 0U);
    EXPECT_NE(half(5e-5f).data & 0x7FFFU, 0U);
    EXPECT_NE(half(6e-5f).data & 0x7FFFU, 0U);
    EXPECT_NE(half(-1e-5f).data & 0x7FFFU, 0U);

    // ... and must still be classified as finite, non-zero, and correctly signed.
    EXPECT_TRUE(isfinite(half(1e-5f)));
    EXPECT_NE(static_cast<float>(half(1e-5f)), 0.0f);
    EXPECT_LT(static_cast<float>(half(-1e-5f)), 0.0f);
}

TEST_F(TestHalf, SubnormalExactBitPatterns)
{
    // Round-to-nearest-even results, in units of 2^-24.
    EXPECT_EQ(half(1e-5f).data, 0x00A8); // 168 units
    EXPECT_EQ(half(5e-5f).data, 0x0347); // 839 units
    EXPECT_EQ(half(6e-5f).data, 0x03EF); // 1007 units
    EXPECT_EQ(half(1e-7f).data, 0x0002); // 2 units

    // Sign is carried through unchanged.
    EXPECT_EQ(half(-1e-5f).data, 0x80A8);
    EXPECT_EQ(half(-5e-5f).data, 0x8347);

    // Decoding the result reproduces the encoded multiple of 2^-24 exactly.
    EXPECT_EQ(static_cast<float>(half(1e-5f)), subnormalUnits(168));
    EXPECT_EQ(static_cast<float>(half(5e-5f)), subnormalUnits(839));
    EXPECT_EQ(static_cast<float>(half(6e-5f)), subnormalUnits(1007));
}

TEST_F(TestHalf, SubnormalExactMultiplesConvertExactly)
{
    for(int k = 1; k <= 1023; ++k)
    {
        const half value(subnormalUnits(k));
        EXPECT_EQ(value.data, static_cast<uint16_t>(k)) << "k = " << k;
        EXPECT_EQ(static_cast<float>(value), subnormalUnits(k)) << "k = " << k;

        const half negative(-subnormalUnits(k));
        EXPECT_EQ(negative.data, static_cast<uint16_t>(0x8000U | static_cast<uint32_t>(k)))
            << "k = " << k;
    }
}

TEST_F(TestHalf, SubnormalRoundToNearestEven)
{
    // k == 0 covers the underflow tie at 2^-25; k == 1023 covers rounding up out of the
    // subnormal range into the smallest normal (0x0400).
    for(int k = 0; k <= 1023; ++k)
    {
        const float midpoint = subnormalMidpoint(k);
        const auto lower = static_cast<uint16_t>(k);
        const auto upper = static_cast<uint16_t>(k + 1);

        // Just below the midpoint rounds down, just above rounds up.
        EXPECT_EQ(half(std::nextafter(midpoint, 0.0f)).data, lower) << "k = " << k;
        EXPECT_EQ(half(std::nextafter(midpoint, 1.0f)).data, upper) << "k = " << k;

        // Exact midpoint is a tie: round to the even significand.
        const uint16_t tie = ((k % 2) == 0) ? lower : upper;
        EXPECT_EQ(half(midpoint).data, tie) << "k = " << k;
        EXPECT_EQ(half(-midpoint).data, static_cast<uint16_t>(0x8000U | tie)) << "k = " << k;
    }
}

TEST_F(TestHalf, SubnormalRoundsUpIntoSmallestNormal)
{
    // Anything above 1023.5 * 2^-24 rounds up to 2^-14, the smallest normal.
    const float justBelowNormal = std::nextafter(subnormalUnits(1024), 0.0f);
    EXPECT_EQ(half(justBelowNormal).data, 0x0400);
    EXPECT_EQ(half(justBelowNormal).data, std::numeric_limits<half>::min().data);
    EXPECT_EQ(half(subnormalUnits(1024)).data, 0x0400);
}

TEST_F(TestHalf, SubnormalUnderflowToSignedZero)
{
    // Strictly below half the smallest subnormal: rounds to zero, sign preserved.
    const float belowHalfUlp = std::nextafter(subnormalMidpoint(0), 0.0f);
    EXPECT_EQ(half(belowHalfUlp).data, 0x0000);
    EXPECT_EQ(half(-belowHalfUlp).data, 0x8000);
    EXPECT_TRUE(signbit(half(-belowHalfUlp)));

    // Exactly half the smallest subnormal is a tie to even, i.e. zero.
    EXPECT_EQ(half(subnormalMidpoint(0)).data, 0x0000);

    // Just above it is the smallest subnormal.
    EXPECT_EQ(half(std::nextafter(subnormalMidpoint(0), 1.0f)).data, 0x0001);
}

TEST_F(TestHalf, RoundTripAllPatterns)
{
    // For every finite bit pattern: decode to float, re-encode, and verify the bit pattern
    // is identical. The other narrow types all have this check; half did not, which is how
    // the subnormal encode path stayed broken.
    for(int bits = 0; bits <= 0xFFFF; ++bits)
    {
        const auto pattern = static_cast<uint16_t>(bits);
        const half decoded = half::from_bits(pattern);
        if(isnan(decoded))
        {
            continue; // NaN payload propagation is covered by the NaN tests
        }
        EXPECT_EQ(half(static_cast<float>(decoded)).data, pattern)
            << "Bit pattern 0x" << std::hex << bits;
    }
}

TEST_F(TestHalf, DenormMinIsRepresentableFromFloat)
{
    const half denormMin = std::numeric_limits<half>::denorm_min();
    EXPECT_EQ(denormMin.data, 0x0001);

    const auto asFloat = static_cast<float>(denormMin);
    EXPECT_EQ(asFloat, subnormalUnits(1));
    EXPECT_EQ(half(asFloat).data, 0x0001);
    EXPECT_LT(asFloat, static_cast<float>(std::numeric_limits<half>::min()));
    EXPECT_GT(asFloat, 0.0f);
}

// ============================================================================
// NaN Conversion Tests
// ============================================================================
//
// float -> half discards the low 13 payload bits. A NaN whose payload lives entirely in
// those bits must still convert to a NaN, not to the infinity encoding, and IEEE 754
// requires the result to be quiet. Both properties match HIP's __float2half.

TEST_F(TestHalf, NanFromFloatNeverEncodesInfinity)
{
    for(uint32_t payload = 1; payload <= 0x1FFFU; ++payload)
    {
        const half positive(floatFromBits(0x7F800000U | payload));
        EXPECT_TRUE(isnan(positive)) << "payload 0x" << std::hex << payload;
        EXPECT_FALSE(isinf(positive)) << "payload 0x" << std::hex << payload;
        EXPECT_EQ(positive.data, 0x7E00) << "payload 0x" << std::hex << payload;

        const half negative(floatFromBits(0xFF800000U | payload));
        EXPECT_TRUE(isnan(negative)) << "payload 0x" << std::hex << payload;
        EXPECT_EQ(negative.data, 0xFE00) << "payload 0x" << std::hex << payload;
    }
}

TEST_F(TestHalf, NanFromFloatIsQuieted)
{
    // Signaling float NaN (payload MSB clear) must convert to a quiet half NaN.
    const half fromSignaling(floatFromBits(0x7F800001U));
    EXPECT_TRUE(isnan(fromSignaling));
    EXPECT_NE(fromSignaling.data & 0x0200U, 0U);

    const half fromLimitsSignaling(std::numeric_limits<float>::signaling_NaN());
    EXPECT_TRUE(isnan(fromLimitsSignaling));
    EXPECT_NE(fromLimitsSignaling.data & 0x0200U, 0U);
}

TEST_F(TestHalf, NanPayloadTopBitsSurviveConversion)
{
    // The 10 payload bits that fit are carried through, with the quiet bit forced on.
    EXPECT_EQ(half(std::numeric_limits<float>::quiet_NaN()).data, 0x7E00);
    EXPECT_EQ(half(floatFromBits(0x7FE00000U)).data, 0x7F00);
    EXPECT_EQ(half(floatFromBits(0x7FFFFFFFU)).data, 0x7FFF);
    EXPECT_EQ(half(floatFromBits(0xFFFFFFFFU)).data, 0xFFFF);
}

TEST_F(TestHalf, InfinityFromFloatIsUnchanged)
{
    EXPECT_EQ(half(std::numeric_limits<float>::infinity()).data, 0x7C00);
    EXPECT_EQ(half(-std::numeric_limits<float>::infinity()).data, 0xFC00);
    EXPECT_TRUE(isinf(half(std::numeric_limits<float>::infinity())));
    EXPECT_FALSE(isnan(half(std::numeric_limits<float>::infinity())));
}
