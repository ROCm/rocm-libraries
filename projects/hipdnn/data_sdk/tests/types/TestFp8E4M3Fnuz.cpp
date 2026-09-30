// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E4M3Fnuz.cpp
/// @brief Type-specific tests for fp8_e4m3_fnuz (FNUZ E4M3 format).
///
/// Common tests (type properties, construction, math functions, numeric_limits) are in
/// TestPortableTypes.cpp. This file covers exact-value round-trip, exhaustive 256-bit-pattern
/// round-trip, FNUZ-essential semantics (single NaN at 0x80, no -0, no infinity,
/// saturate-to-MAX), rounding (parametrized), saturation, underflow, and numeric_limits checks.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// fp8_e4m3_fnuz Type-Specific Tests
// ============================================================================

TEST(TestFp8E4M3Fnuz, RoundTripExactValues)
{
    // These values are exactly representable in fp8_e4m3_fnuz
    const std::vector<float> exactValues = {
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
        240.0f, // MAX
        -240.0f, // LOWEST
        0.0078125f, // MIN_NORMAL = 2^-7
        -0.0078125f,
    };

    for(const float val : exactValues)
    {
        const fp8_e4m3_fnuz v(val);
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

class TestFp8E4M3FnuzRounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp8E4M3FnuzRounding, Rounding)
{
    auto [input, expected] = GetParam();
    const fp8_e4m3_fnuz val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

// Midpoint values that are exactly halfway between two representable fp8_e4m3_fnuz
// encodings; round-to-nearest-even selects the encoding with an even mantissa.
// Lookup table: 0x08=0.0078125, 0x09=0.0087890625, 0x40=1.0, 0x41=1.125, 0x42=1.25,
//               0x48=2.0, 0x49=2.25, 0x7E=224.0, 0x7F=240.0.
INSTANTIATE_TEST_SUITE_P(
    MidpointRounding,
    TestFp8E4M3FnuzRounding,
    ::testing::Values(
        // 0x08 (mant=0, even) vs 0x09 (mant=1):
        // midpoint = (0.0078125 + 0.0087890625) / 2 = 0.00830078125 (exact in float32)
        // round down to 0x08 (even mantissa)
        RoundingTestCase{0.00830078125f, 0.0078125f},
        // 0x09 (mant=1, odd) vs 0x0A (mant=2):
        // midpoint = (0.0087890625 + 0.009765625) / 2 = 0.009277343750 (exact in float32)
        // round up to 0x0A (even mantissa)
        RoundingTestCase{0.00927734375f, 0.009765625f},
        // 0x40 (mant=0, even) vs 0x41 (mant=1): midpoint=1.0625 -> round down -> 1.0
        RoundingTestCase{1.0625f, 1.0f},
        // 0x41 (mant=1, odd) vs 0x42 (mant=2): midpoint=1.1875 -> round up -> 1.25
        RoundingTestCase{1.1875f, 1.25f},
        // 0x48 (mant=0, even) vs 0x49 (mant=1): midpoint=2.125 -> round down -> 2.0
        RoundingTestCase{2.125f, 2.0f},
        // 0x7E (mant=6, even) vs 0x7F (mant=7): midpoint=232.0 -> round down -> 224.0
        RoundingTestCase{232.0f, 224.0f}));

// Values in the subnormal/underflow range.
// Values with |value| <= 2^-11 = 0.00048828125 underflow to zero.
// Values in (2^-11, 2^-7) encode as FNUZ subnormals (mant x 2^-10).
// Round-to-nearest-even applies within the subnormal range.
INSTANTIATE_TEST_SUITE_P(
    SubnormalRounding,
    TestFp8E4M3FnuzRounding,
    ::testing::Values(
        // Well below underflow threshold (2^-11): flushes to zero
        RoundingTestCase{1e-4f, 0.0f},
        // In subnormal range: 0.007 ~ 7.168 x 2^-10, rounds to mant=7 (0x07 = 0.0068359375)
        RoundingTestCase{0.007f, 0.0068359375f},
        // Exactly 4 x 2^-10 = 0.00390625: encodes exactly as subnormal mant=4
        RoundingTestCase{0.00390625f, 0.00390625f},
        // Very small: underflows to zero
        RoundingTestCase{1e-10f, 0.0f},
        // Negative: same underflow to zero
        RoundingTestCase{-1e-4f, 0.0f},
        // Negative subnormal: -0.003 ~ -3.072 x 2^-10, rounds to mant=3 -> -0.0029296875
        RoundingTestCase{-0.003f, -0.0029296875f}));

// Values at the boundary between two adjacent encodings (not at exact midpoints).
// Verifies correct rounding direction in the non-tie case.
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp8E4M3FnuzRounding,
                         ::testing::Values(
                             // Slightly above 1.0 (0x40): closer to 1.0 than 1.125 -> 1.0
                             RoundingTestCase{1.06f, 1.0f},
                             // Slightly above midpoint 1.0625: closer to 1.125 -> 1.125
                             RoundingTestCase{1.07f, 1.125f},
                             // Between 224.0 (0x7E) and 240.0 (0x7F): closer to 224 -> 224
                             RoundingTestCase{228.0f, 224.0f},
                             // Between 224.0 and 240.0: closer to 240 -> 240
                             RoundingTestCase{236.0f, 240.0f},
                             // Negative boundary
                             RoundingTestCase{-1.06f, -1.0f}));

// Rounding where mantissa overflow from the round-up cascades into the exponent.
// The critical case is a value whose fp8 mantissa is 7 (odd), falling exactly at the
// midpoint above 240.0 (the MAX): the round-up causes mant=8 (overflow) -> exp=16
// (overflow) -> saturate to MAX=240.0.
INSTANTIATE_TEST_SUITE_P(
    RoundingOverflow,
    TestFp8E4M3FnuzRounding,
    ::testing::Values(
        // Mantissa overflow: exp increments, but stays within range
        // 0x47 (mant=7, odd) = 1.875; midpoint to 0x48=2.0 causes mant overflow -> 2.0
        RoundingTestCase{1.9375f, 2.0f},
        // 0x4F (mant=7, odd) = 3.75; midpoint to 0x50=4.0 causes mant overflow -> 4.0
        RoundingTestCase{3.875f, 4.0f},
        // Exponent overflow into saturation:
        // 248.0 is exactly the midpoint between MAX=240.0 and phantom 256.0.
        // fp8exp=15, fp8Mant=7 (odd) -> round up -> mant=8 overflow -> exp=16 -> saturate
        RoundingTestCase{248.0f, 240.0f},
        // Negative counterparts
        RoundingTestCase{-1.9375f, -2.0f},
        RoundingTestCase{-3.875f, -4.0f},
        RoundingTestCase{-248.0f, -240.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp8E4M3Fnuz, SaturationPositive)
{
    const fp8_e4m3_fnuz val1(1e6f);
    EXPECT_EQ(static_cast<float>(val1), 240.0f);

    const fp8_e4m3_fnuz val2(241.0f);
    EXPECT_EQ(static_cast<float>(val2), 240.0f);

    const fp8_e4m3_fnuz val3(480.0f);
    EXPECT_EQ(static_cast<float>(val3), 240.0f);

    const fp8_e4m3_fnuz val4(std::nextafter(240.0f, 1e9f));
    EXPECT_EQ(static_cast<float>(val4), 240.0f);

    // Rounding-into-saturation: 248.0 is the exact midpoint between MAX=240 and
    // phantom 256.0. Mantissa is 7 (odd) so round-up overflows exp -> saturate.
    const fp8_e4m3_fnuz val5(248.0f);
    EXPECT_EQ(static_cast<float>(val5), 240.0f);
}

TEST(TestFp8E4M3Fnuz, SaturationNegative)
{
    const fp8_e4m3_fnuz val1(-1e6f);
    EXPECT_EQ(static_cast<float>(val1), -240.0f);

    const fp8_e4m3_fnuz val2(-241.0f);
    EXPECT_EQ(static_cast<float>(val2), -240.0f);

    const fp8_e4m3_fnuz val3(-480.0f);
    EXPECT_EQ(static_cast<float>(val3), -240.0f);

    // Rounding-into-saturation: -248.0 mirrors the positive case above
    const fp8_e4m3_fnuz val4(-248.0f);
    EXPECT_EQ(static_cast<float>(val4), -240.0f);
}

TEST(TestFp8E4M3Fnuz, SaturationInfinity)
{
    const fp8_e4m3_fnuz posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 240.0f);

    const fp8_e4m3_fnuz negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -240.0f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp8E4M3Fnuz, Underflow)
{
    const fp8_e4m3_fnuz val1(1e-10f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    const fp8_e4m3_fnuz val2(-1e-10f);
    EXPECT_EQ(static_cast<float>(val2), 0.0f);

    const fp8_e4m3_fnuz val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    const fp8_e4m3_fnuz val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    const fp8_e4m3_fnuz val5(1e-4f);
    EXPECT_EQ(static_cast<float>(val5), 0.0f);

    // The exact midpoint 2^-11 rounds to zero (even mantissa tie-break)
    const fp8_e4m3_fnuz val6(0.00048828125f);
    EXPECT_EQ(static_cast<float>(val6), 0.0f);

    // Just above the midpoint rounds up to denorm_min = 2^-10
    const fp8_e4m3_fnuz val7(1.5f * 0.00048828125f);
    EXPECT_EQ(val7.data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(val7), 0.0009765625f);

    const fp8_e4m3_fnuz val8(-1.5f * 0.00048828125f);
    EXPECT_EQ(val8.data, static_cast<uint8_t>(0x81));
    EXPECT_EQ(static_cast<float>(val8), -0.0009765625f);
}

// ============================================================================
// E4M3-FNUZ-Specific: NaN, Infinity, Negative Zero, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp8E4M3Fnuz, RoundTripAllPatterns)
{
    // For every bit pattern: decode to float via the lookup table, re-encode back,
    // and verify that the resulting bit pattern is identical.
    // Bit pattern 0x80 is the single NaN encoding; re-encoding a float NaN returns
    // 0x80 by definition, so it participates in the round-trip correctly.
    // Subnormal bit patterns 0x01-0x07 and 0x81-0x87 decode to their true subnormal
    // float values, which the encoder re-encodes back to the same bit pattern.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const auto decoded = fp8_e4m3_fnuz::from_bits(pattern);
        const auto f = static_cast<float>(decoded);

        // NaN round-trip: float NaN re-encodes to 0x80.
        if(pattern == 0x80)
        {
            EXPECT_TRUE(std::isnan(f)) << "Pattern 0x80 should decode to NaN";
            EXPECT_EQ(fp8_e4m3_fnuz(f).data, static_cast<uint8_t>(0x80));
            continue;
        }

        // All patterns must survive the round-trip exactly.
        const fp8_e4m3_fnuz reencoded(f);
        EXPECT_EQ(reencoded.data, pattern) << "Round-trip failed for bit pattern 0x" << std::hex
                                           << bits << " (float value " << std::dec << f << ")";
    }
}

TEST(TestFp8E4M3Fnuz, NoNegativeZero)
{
    // FNUZ has no -0; -0.0f input must produce bit pattern 0x00
    EXPECT_EQ(fp8_e4m3_fnuz(-0.0f).data, static_cast<uint8_t>(0x00));
    EXPECT_EQ(fp8_e4m3_fnuz(0.0f).data, static_cast<uint8_t>(0x00));
}

TEST(TestFp8E4M3Fnuz, NegateZeroProducesNaN)
{
    // FNUZ has no -0; negating zero flips the sign bit: 0x00 -> 0x80 = NaN
    auto val = -fp8_e4m3_fnuz(0.0f);
    EXPECT_TRUE(isnan(val));
    EXPECT_EQ(val.data, static_cast<uint8_t>(0x80));
}

TEST(TestFp8E4M3Fnuz, NegateNaNProducesZero)
{
    // NaN (0x80) negated flips the sign bit: 0x80 -> 0x00 = zero
    auto val = -fp8_e4m3_fnuz::from_bits(0x80);
    EXPECT_EQ(static_cast<float>(val), 0.0f);
    EXPECT_EQ(val.data, static_cast<uint8_t>(0x00));
}

TEST(TestFp8E4M3Fnuz, SingleNanEncoding)
{
    // Float NaN must map to 0x80
    auto v = fp8_e4m3_fnuz(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(v.data, static_cast<uint8_t>(0x80));
    EXPECT_TRUE(isnan(v));

    // signaling NaN also maps to 0x80
    auto vs = fp8_e4m3_fnuz(std::numeric_limits<float>::signaling_NaN());
    EXPECT_EQ(vs.data, static_cast<uint8_t>(0x80));
    EXPECT_TRUE(isnan(vs));

    // Every bit pattern except 0x80 is not NaN
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        if(bits == 0x80)
        {
            continue;
        }
        EXPECT_FALSE(isnan(fp8_e4m3_fnuz::from_bits(static_cast<uint8_t>(bits))))
            << "Bit pattern 0x" << std::hex << bits << " should not be NaN";
    }
}

TEST(TestFp8E4M3Fnuz, IsinfAlwaysFalse)
{
    // No bit pattern in fp8_e4m3_fnuz represents infinity
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        EXPECT_FALSE(isinf(fp8_e4m3_fnuz::from_bits(static_cast<uint8_t>(bits))))
            << "Bit pattern 0x" << std::hex << bits << " should not be infinity";
    }
}

TEST(TestFp8E4M3Fnuz, MathFunctions)
{
    // isfinite: all values except NaN are finite
    EXPECT_TRUE(isfinite(fp8_e4m3_fnuz(1.0f)));
    EXPECT_TRUE(isfinite(fp8_e4m3_fnuz(0.0f)));
    EXPECT_TRUE(isfinite(fp8_e4m3_fnuz(240.0f)));
    EXPECT_FALSE(isfinite(fp8_e4m3_fnuz::from_bits(0x80)));

    // signbit: positive values have sign bit 0, negative have sign bit 1
    EXPECT_FALSE(signbit(fp8_e4m3_fnuz(1.0f)));
    EXPECT_TRUE(signbit(fp8_e4m3_fnuz(-1.0f)));
    EXPECT_FALSE(signbit(fp8_e4m3_fnuz(0.0f)));
    // The sole NaN encoding 0x80 has the sign bit set
    EXPECT_TRUE(signbit(fp8_e4m3_fnuz::from_bits(0x80)));
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp8E4M3Fnuz, NumericLimitsSpecificValues)
{
    using L = std::numeric_limits<fp8_e4m3_fnuz>;

    // max() = 240.0
    EXPECT_EQ(static_cast<float>(L::max()), 240.0f);
    EXPECT_EQ(L::max().data, static_cast<uint8_t>(0x7F));

    // min() = 2^-7 = 0.0078125
    EXPECT_EQ(static_cast<float>(L::min()), 0.0078125f);
    EXPECT_EQ(L::min().data, static_cast<uint8_t>(0x08));

    // lowest() = -240.0
    EXPECT_EQ(static_cast<float>(L::lowest()), -240.0f);
    EXPECT_EQ(L::lowest().data, static_cast<uint8_t>(0xFF));

    // epsilon() = 2^-3 = 0.125
    EXPECT_EQ(static_cast<float>(L::epsilon()), 0.125f);
    EXPECT_EQ(L::epsilon().data, static_cast<uint8_t>(0x28));

    // round_error() = 0.5
    EXPECT_EQ(static_cast<float>(L::round_error()), 0.5f);
    EXPECT_EQ(L::round_error().data, static_cast<uint8_t>(0x38));

    // denorm_min() returns bit pattern 0x01 = 1 x 2^-10 = 0.0009765625 (minimum positive subnormal).
    EXPECT_EQ(L::denorm_min().data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(L::denorm_min()), 0.0009765625f);

    // infinity() returns max() since there is no infinity
    EXPECT_EQ(L::infinity().data, L::max().data);

    // quiet_NaN() and signaling_NaN() both return 0x80
    EXPECT_EQ(L::quiet_NaN().data, static_cast<uint8_t>(0x80));
    EXPECT_EQ(L::signaling_NaN().data, static_cast<uint8_t>(0x80));

    // has_infinity = false
    EXPECT_FALSE(L::has_infinity);

    // has_quiet_NaN = true
    EXPECT_TRUE(L::has_quiet_NaN);

    // has_denorm = denorm_present (FNUZ supports subnormals)
    EXPECT_EQ(L::has_denorm, std::denorm_present);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST(TestFp8E4M3Fnuz, NamedConstants)
{
    using L = std::numeric_limits<fp8_e4m3_fnuz>;

    // FNUZ has no infinity - returns max
    EXPECT_EQ(L::infinity().data, L::max().data);

    // FNUZ has a single NaN at 0x80
    EXPECT_EQ(L::quiet_NaN().data, static_cast<uint8_t>(0x80));
    EXPECT_EQ(L::signaling_NaN().data, static_cast<uint8_t>(0x80));

    // FNUZ has no -0; denorm_min is the smallest positive subnormal
    EXPECT_FALSE(L::has_infinity);
    EXPECT_TRUE(L::has_quiet_NaN);
    EXPECT_EQ(L::has_denorm, std::denorm_present);
}
