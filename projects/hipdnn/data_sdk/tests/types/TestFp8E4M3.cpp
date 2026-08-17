// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E4M3.cpp
/// @brief Type-specific tests for fp8_e4m3 (OCP E4M3 format).
///
/// Common tests (type properties, construction, math functions, numeric_limits) are in
/// TestPortableTypes.cpp. This file covers exact-value round-trip, exhaustive 256-bit-pattern
/// round-trip, OCP E4M3-essential semantics (dual NaN encoding at 0x7F/0xFF, no infinity,
/// saturate-to-MAX, signed zero), rounding (parametrized), saturation, underflow, and
/// numeric_limits checks.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// fp8_e4m3 Type-Specific Tests
// ============================================================================

TEST(TestFp8E4M3, RoundTripExactValues)
{
    // These values are exactly representable in fp8_e4m3 (OCP E4M3 format).
    const std::vector<float> exactValues = {
        // Powers of two and simple fractions
        0.0f,
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        4.0f,
        -4.0f,
        8.0f,
        -8.0f,
        0.5f,
        -0.5f,
        0.25f,
        -0.25f,
        // Format limits
        448.0f, // MAX
        -448.0f, // LOWEST
        0.015625f, // MIN_NORMAL = 2^-6
        -0.015625f,
        0.001953125f, // denorm_min = 2^-9
        -0.001953125f,
    };

    for(const float val : exactValues)
    {
        const fp8_e4m3 v(val);
        EXPECT_EQ(static_cast<float>(v), val) << "Round-trip failed for " << val;
    }
}

// ============================================================================
// Rounding Tests
// ============================================================================

namespace
{
struct RoundingTestCase
{
    float input;
    float expected;

    friend std::ostream& operator<<(std::ostream& os, const RoundingTestCase& tc)
    {
        return os << tc.input << " -> " << tc.expected;
    }
};
} // namespace

class TestFp8E4M3Rounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp8E4M3Rounding, Rounding)
{
    auto [input, expected] = GetParam();
    const fp8_e4m3 val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

// Midpoint values that are exactly halfway between two representable fp8_e4m3
// encodings; round-to-nearest-even selects the encoding with an even mantissa.
// Lookup table: 0x40=1.0, 0x41=1.125, 0x42=1.25, 0x7E=448.0 (MAX);
// midpoint to phantom 512 is 480.
INSTANTIATE_TEST_SUITE_P(
    MidpointRounding,
    TestFp8E4M3Rounding,
    ::testing::Values(
        // 0x40 (mant=0, even) vs 0x41 (mant=1, odd): midpoint=1.0625 -> round down -> 1.0
        RoundingTestCase{1.0625f, 1.0f},
        // 0x41 (mant=1, odd) vs 0x42 (mant=2, even): midpoint=1.1875 -> round up -> 1.25
        RoundingTestCase{1.1875f, 1.25f},
        // 0x7E (mant=6, even) vs phantom 0x80-space: midpoint=480.0 -> round to even -> MAX=448.0
        RoundingTestCase{480.0f, 448.0f}));

// Values in the subnormal/underflow range.
// OCP E4M3 denorms: value = mant * 2^-9 (mant=1..7); denorm_min = 2^-9 ~ 0.001953125.
// Underflow threshold = half of denorm_min = 2^-10 ~ 9.765625e-4.
// At exactly 2^-10 the result rounds to 0 (even mantissa tie-break).
// fp32 subnormal inputs (fp32Exp==0) flush directly to zero regardless of value.
INSTANTIATE_TEST_SUITE_P(
    SubnormalRounding,
    TestFp8E4M3Rounding,
    ::testing::Values(
        // Well below underflow threshold (2^-10): flushes to zero
        RoundingTestCase{1e-4f, 0.0f},
        // At exactly the underflow threshold 2^-10: round to even -> zero
        RoundingTestCase{9.765625e-4f, 0.0f},
        // Slightly above 2^-10: rounds to denorm_min = 2^-9
        RoundingTestCase{1.0e-3f, 0.001953125f},
        // Midpoint between 0x01 (mant=1 odd) and 0x02 (mant=2 even): rounds up to 0x02
        RoundingTestCase{0.0029296875f, 0.00390625f},
        // Even-LSB subnormal midpoint: 0x02 (mant=2, even) vs 0x03 (mant=3, odd).
        // Midpoint=0.0048828125; ties-to-even -> round DOWN to 0x02=0.00390625.
        RoundingTestCase{0.0048828125f, 0.00390625f},
        // Denormal-to-normal carry, below midpoint: 0x07=0.013671875 (mant=7, odd),
        // 0x08=0.015625 (smallest normal). Below midpoint -> 0x07.
        RoundingTestCase{0.01416015625f, 0.013671875f},
        // Exact carry midpoint: subnormal mant=7 odd -> round up -> mant=8 overflow ->
        // carry into smallest normal 0x08=0.015625.
        RoundingTestCase{0.0146484375f, 0.015625f},
        // Above carry midpoint: definitively rounds up to smallest normal.
        RoundingTestCase{0.01513671875f, 0.015625f},
        RoundingTestCase{-0.01513671875f, -0.015625f}));

// Values at the boundary between two adjacent encodings (not at exact midpoints).
// Verifies correct rounding direction in the non-tie case.
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp8E4M3Rounding,
                         ::testing::Values(
                             // Slightly above 1.0 (0x40): closer to 1.0 than 1.125 -> 1.0
                             RoundingTestCase{1.06f, 1.0f},
                             // Slightly above midpoint 1.0625: closer to 1.125 -> 1.125
                             RoundingTestCase{1.07f, 1.125f},
                             // Between 416.0 (0x7D) and 448.0 (0x7E): closer to 416 -> 416
                             RoundingTestCase{420.0f, 416.0f},
                             // Between 416.0 and 448.0: closer to 448 -> 448
                             RoundingTestCase{440.0f, 448.0f},
                             // Negative boundary
                             RoundingTestCase{-1.06f, -1.0f}));

// Rounding where mantissa overflow from the round-up cascades into the exponent,
// or where the final exponent overflows into saturation.
//
// OCP E4M3 mant bits = 3; when fp8Mant rounds from 7 to 8 (overflow), exp increments.
// If exp exceeds the highest valid normal (15), the value saturates to MAX.
// Saturation midpoint: MAX=448 (exp=15, mant=6), phantom next = 512. Midpoint=480.0;
// mant at MAX is 6 (even) -> round down to MAX=448.
//
// Inputs in [480, 488) encode normally to (exp=15, mant=7) = bit pattern 0x7F, the OCP
// E4M3 NaN encoding. The post-encoding NaN-zone check detects this collision and
// saturates to MAX=448.0 (0x7E) instead.
INSTANTIATE_TEST_SUITE_P(
    RoundingOverflow,
    TestFp8E4M3Rounding,
    ::testing::Values(
        // Mantissa overflow: exp increments, stays in range
        // 0x47 (mant=7, odd) = 1.875; midpoint to 0x48=2.0 -> mant overflow -> 2.0
        RoundingTestCase{1.9375f, 2.0f},
        // Just below the saturation midpoint: rounds to MAX
        RoundingTestCase{449.0f, 448.0f},
        // NaN-zone collision: 481.0 encodes to (exp=15, mant=7)=0x7F (NaN) -> saturate to 448.
        RoundingTestCase{481.0f, 448.0f},
        // Negative counterpart for mantissa-overflow cascade
        RoundingTestCase{-1.9375f, -2.0f},
        // Negative mirror of NaN-zone collision: -481.0 -> 0xFF -> saturate to -448.
        RoundingTestCase{-481.0f, -448.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp8E4M3, SaturationPositive)
{
    const fp8_e4m3 val1(1e6f);
    EXPECT_EQ(static_cast<float>(val1), 448.0f);

    const fp8_e4m3 val2(449.0f);
    EXPECT_EQ(static_cast<float>(val2), 448.0f);

    const fp8_e4m3 val3(896.0f);
    EXPECT_EQ(static_cast<float>(val3), 448.0f);

    const fp8_e4m3 val4(std::nextafter(448.0f, 1e9f));
    EXPECT_EQ(static_cast<float>(val4), 448.0f);

    // Rounding-into-saturation: 480.0 midpoint with mant=6 (even) -> round down to MAX=448.
    const fp8_e4m3 val5(480.0f);
    EXPECT_EQ(static_cast<float>(val5), 448.0f);
}

TEST(TestFp8E4M3, SaturationNegative)
{
    const fp8_e4m3 val1(-1e6f);
    EXPECT_EQ(static_cast<float>(val1), -448.0f);

    const fp8_e4m3 val2(-449.0f);
    EXPECT_EQ(static_cast<float>(val2), -448.0f);

    const fp8_e4m3 val3(-896.0f);
    EXPECT_EQ(static_cast<float>(val3), -448.0f);

    // Rounding-into-saturation: -480.0 mirrors the positive case above.
    const fp8_e4m3 val4(-480.0f);
    EXPECT_EQ(static_cast<float>(val4), -448.0f);
}

TEST(TestFp8E4M3, SaturationInfinity)
{
    const fp8_e4m3 posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 448.0f);

    const fp8_e4m3 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -448.0f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp8E4M3, Underflow)
{
    const fp8_e4m3 val1(1e-10f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    const fp8_e4m3 val2(-1e-10f);
    EXPECT_EQ(static_cast<float>(val2), 0.0f);

    // fp32 denormal inputs flush to zero (fp32Exp==0 path)
    const fp8_e4m3 val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    const fp8_e4m3 val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    const fp8_e4m3 val5(1e-4f);
    EXPECT_EQ(static_cast<float>(val5), 0.0f);

    // The exact underflow threshold 2^-10 rounds to zero (even mantissa tie-break).
    const fp8_e4m3 val6(9.765625e-4f);
    EXPECT_EQ(static_cast<float>(val6), 0.0f);

    // Just above the threshold rounds up to denorm_min = 2^-9.
    const fp8_e4m3 val7(1.5f * 9.765625e-4f);
    EXPECT_EQ(val7.data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(val7), 0.001953125f);

    const fp8_e4m3 val8(-1.5f * 9.765625e-4f);
    EXPECT_EQ(val8.data, static_cast<uint8_t>(0x81));
    EXPECT_EQ(static_cast<float>(val8), -0.001953125f);
}

// ============================================================================
// E4M3-OCP-Specific: NaN, Infinity, Signed Zero, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp8E4M3, RoundTripAllPatterns)
{
    // For every bit pattern: decode to float via the lookup table, re-encode back,
    // and verify that the resulting bit pattern is identical.
    // OCP E4M3 NaN encodings: 0x7F (positive) and 0xFF (negative); re-encoding a
    // float NaN preserves the sign bit.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const auto decoded = fp8_e4m3::from_bits(pattern);
        const auto f = static_cast<float>(decoded);

        // NaN patterns: low 7 bits all set (0x7F positive, 0xFF negative).
        // Sign is preserved on re-encode.
        if((pattern & 0x7F) == 0x7F)
        {
            EXPECT_TRUE(std::isnan(f))
                << "Pattern 0x" << std::hex << bits << " should decode to NaN";
            EXPECT_EQ(fp8_e4m3(f).data, pattern);
            continue;
        }

        // All other patterns must survive the round-trip exactly.
        const fp8_e4m3 reencoded(f);
        EXPECT_EQ(reencoded.data, pattern) << "Round-trip failed for bit pattern 0x" << std::hex
                                           << bits << " (float value " << std::dec << f << ")";
    }
}

TEST(TestFp8E4M3, SignedZeroSupported)
{
    // OCP E4M3 supports both +0 (0x00) and -0 (0x80); both compare equal to 0.0f.
    EXPECT_EQ(fp8_e4m3(0.0f).data, static_cast<uint8_t>(0x00));
    EXPECT_EQ(fp8_e4m3(-0.0f).data, static_cast<uint8_t>(0x80));
    EXPECT_EQ(static_cast<float>(fp8_e4m3::from_bits(0x80)), -0.0f);
}

TEST(TestFp8E4M3, NegateZeroProducesNegativeZero)
{
    // OCP E4M3 supports -0; negating +0 flips the sign bit -> 0x80 (-0).
    auto val = -fp8_e4m3(0.0f);
    EXPECT_EQ(val.data, static_cast<uint8_t>(0x80));
    EXPECT_EQ(static_cast<float>(val), -0.0f);
    EXPECT_FALSE(isnan(val));
}

TEST(TestFp8E4M3, NegateNanProducesNan)
{
    // OCP E4M3: negating NaN flips the sign bit; both 0x7F and 0xFF are valid NaN.
    auto posNan = fp8_e4m3::from_bits(0x7F);
    auto negNan = -posNan;
    EXPECT_EQ(negNan.data, static_cast<uint8_t>(0xFF));
    EXPECT_TRUE(isnan(negNan));

    auto fromNeg = fp8_e4m3::from_bits(0xFF);
    auto backToPos = -fromNeg;
    EXPECT_EQ(backToPos.data, static_cast<uint8_t>(0x7F));
    EXPECT_TRUE(isnan(backToPos));
}

TEST(TestFp8E4M3, DualNanEncoding)
{
    // Float NaN preserves sign on conversion: positive NaN -> 0x7F, negative -> 0xFF.
    auto v = fp8_e4m3(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(v.data, static_cast<uint8_t>(0x7F));
    EXPECT_TRUE(isnan(v));

    auto vNeg = fp8_e4m3(-std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(vNeg.data, static_cast<uint8_t>(0xFF));
    EXPECT_TRUE(isnan(vNeg));

    // Signaling NaN: OCP E4M3 has no SNaN distinction, also maps to 0x7F.
    auto vs = fp8_e4m3(std::numeric_limits<float>::signaling_NaN());
    EXPECT_EQ(vs.data, static_cast<uint8_t>(0x7F));
    EXPECT_TRUE(isnan(vs));

    // Every bit pattern except those with low 7 bits all set (0x7F, 0xFF) is not NaN.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        if((bits & 0x7F) == 0x7F)
        {
            continue;
        }
        EXPECT_FALSE(isnan(fp8_e4m3::from_bits(static_cast<uint8_t>(bits))))
            << "Bit pattern 0x" << std::hex << bits << " should not be NaN";
    }
}

TEST(TestFp8E4M3, IsinfAlwaysFalse)
{
    // No bit pattern in fp8_e4m3 represents infinity (OCP E4M3 has no Inf).
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        EXPECT_FALSE(isinf(fp8_e4m3::from_bits(static_cast<uint8_t>(bits))))
            << "Bit pattern 0x" << std::hex << bits << " should not be infinity";
    }
}

TEST(TestFp8E4M3, MathFunctions)
{
    // isfinite: all values except NaN are finite
    EXPECT_TRUE(isfinite(fp8_e4m3(1.0f)));
    EXPECT_TRUE(isfinite(fp8_e4m3(0.0f)));
    EXPECT_TRUE(isfinite(fp8_e4m3(448.0f)));
    EXPECT_FALSE(isfinite(fp8_e4m3::from_bits(0x7F)));
    EXPECT_FALSE(isfinite(fp8_e4m3::from_bits(0xFF)));

    // signbit: positive values have sign bit 0, negative have sign bit 1
    EXPECT_FALSE(signbit(fp8_e4m3(1.0f)));
    EXPECT_TRUE(signbit(fp8_e4m3(-1.0f)));
    EXPECT_FALSE(signbit(fp8_e4m3(0.0f)));
    EXPECT_TRUE(signbit(fp8_e4m3(-0.0f)));
    // Positive NaN (0x7F) has sign bit clear; negative NaN (0xFF) has sign bit set.
    EXPECT_FALSE(signbit(fp8_e4m3::from_bits(0x7F)));
    EXPECT_TRUE(signbit(fp8_e4m3::from_bits(0xFF)));
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp8E4M3, NumericLimitsSpecificValues)
{
    using L = std::numeric_limits<fp8_e4m3>;

    // max() = 448.0
    EXPECT_EQ(static_cast<float>(L::max()), 448.0f);
    EXPECT_EQ(L::max().data, static_cast<uint8_t>(0x7E));

    // min() = 2^-6 = 0.015625
    EXPECT_EQ(static_cast<float>(L::min()), 0.015625f);
    EXPECT_EQ(L::min().data, static_cast<uint8_t>(0x08));

    // lowest() = -448.0
    EXPECT_EQ(static_cast<float>(L::lowest()), -448.0f);
    EXPECT_EQ(L::lowest().data, static_cast<uint8_t>(0xFE));

    // epsilon() = 2^-3 = 0.125
    EXPECT_EQ(static_cast<float>(L::epsilon()), 0.125f);
    EXPECT_EQ(L::epsilon().data, static_cast<uint8_t>(0x20));

    // round_error() = 0.5
    EXPECT_EQ(static_cast<float>(L::round_error()), 0.5f);
    EXPECT_EQ(L::round_error().data, static_cast<uint8_t>(0x30));

    // denorm_min() returns bit pattern 0x01 = 2^-9 = 0.001953125 (minimum positive subnormal).
    EXPECT_EQ(L::denorm_min().data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(L::denorm_min()), 0.001953125f);

    // infinity() returns max() since OCP E4M3 has no infinity.
    EXPECT_EQ(L::infinity().data, L::max().data);

    // quiet_NaN() and signaling_NaN() both return 0x7F (positive NaN).
    EXPECT_EQ(L::quiet_NaN().data, static_cast<uint8_t>(0x7F));
    EXPECT_EQ(L::signaling_NaN().data, static_cast<uint8_t>(0x7F));

    // has_infinity = false
    EXPECT_FALSE(L::has_infinity);

    // has_quiet_NaN = true
    EXPECT_TRUE(L::has_quiet_NaN);

    // has_denorm = denorm_present (OCP E4M3 supports subnormals)
    EXPECT_EQ(L::has_denorm, std::denorm_present);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST(TestFp8E4M3, NamedConstants)
{
    using L = std::numeric_limits<fp8_e4m3>;

    // OCP E4M3 has no infinity - returns max
    EXPECT_EQ(L::infinity().data, L::max().data);

    // OCP E4M3 has dual NaN encoding at 0x7F (positive) and 0xFF (negative);
    // quiet_NaN() and signaling_NaN() return the canonical 0x7F.
    EXPECT_EQ(L::quiet_NaN().data, static_cast<uint8_t>(0x7F));
    EXPECT_EQ(L::signaling_NaN().data, static_cast<uint8_t>(0x7F));

    // OCP E4M3 supports -0; denorm_min is the smallest positive subnormal
    EXPECT_FALSE(L::has_infinity);
    EXPECT_TRUE(L::has_quiet_NaN);
    EXPECT_EQ(L::has_denorm, std::denorm_present);
}
