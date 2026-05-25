// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E5M2Fnuz.cpp
/// @brief Type-specific tests for fp8_e5m2_fnuz (FNUZ E5M2 format).
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
// fp8_e5m2_fnuz Type-Specific Tests
// ============================================================================

TEST(TestFp8E5M2Fnuz, RoundTripExactValues)
{
    // These values are exactly representable in fp8_e5m2_fnuz
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
        57344.0f, // MAX
        -57344.0f, // LOWEST
        3.0517578125e-5f, // MIN_NORMAL = 2^-15
        -3.0517578125e-5f,
    };

    for(const float val : exactValues)
    {
        const fp8_e5m2_fnuz v(val);
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

class TestFp8E5M2FnuzRounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp8E5M2FnuzRounding, Rounding)
{
    auto [input, expected] = GetParam();
    const fp8_e5m2_fnuz val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

// Midpoint values that are exactly halfway between two representable fp8_e5m2_fnuz
// encodings; round-to-nearest-even selects the encoding with an even mantissa.
// Lookup table: 0x40=1.0, 0x41=1.25, 0x42=1.5, 0x44=2.0, 0x45=2.5,
//               0x7E=49152.0, 0x7F=57344.0.
INSTANTIATE_TEST_SUITE_P(
    MidpointRounding,
    TestFp8E5M2FnuzRounding,
    ::testing::Values(
        // 0x40 (mant=0, even) vs 0x41 (mant=1): midpoint=1.125 -> round down -> 1.0
        RoundingTestCase{1.125f, 1.0f},
        // 0x41 (mant=1, odd) vs 0x42 (mant=2): midpoint=1.375 -> round up -> 1.5
        RoundingTestCase{1.375f, 1.5f},
        // 0x44 (mant=0, even) vs 0x45 (mant=1): midpoint=2.25 -> round down -> 2.0
        RoundingTestCase{2.25f, 2.0f},
        // 0x7E (mant=2, even) vs 0x7F (mant=3): midpoint=53248.0 -> round down -> 49152.0
        RoundingTestCase{53248.0f, 49152.0f}));

// Values in the subnormal/underflow range.
// Values with |value| <= 2^-18 = 3.814697265625e-6 underflow to zero.
// Values in (2^-18, 2^-15) encode as FNUZ subnormals (mant x 2^-17).
// Round-to-nearest-even applies within the subnormal range.
INSTANTIATE_TEST_SUITE_P(
    SubnormalRounding,
    TestFp8E5M2FnuzRounding,
    ::testing::Values(
        // Well below underflow threshold (2^-18): flushes to zero
        RoundingTestCase{1e-6f, 0.0f},
        // In subnormal range: 2e-5 ~ 2.621 x 2^-17, rounds to mant=3 (0x03 = 2.288818359375e-5)
        RoundingTestCase{2e-5f, 2.288818359375e-05f},
        // Exactly 2 x 2^-17 = 1.52587890625e-5: encodes exactly as subnormal mant=2
        RoundingTestCase{1.52587890625e-5f, 1.52587890625e-5f},
        // Very small: underflows to zero
        RoundingTestCase{1e-10f, 0.0f},
        // Negative: same underflow to zero
        RoundingTestCase{-1e-6f, 0.0f},
        // Negative subnormal: -1e-5 ~ -1.31 x 2^-17, rounds to mant=1 -> -7.62939453125e-6
        RoundingTestCase{-1e-5f, -7.62939453125e-06f}));

// Values at the boundary between two adjacent encodings (not at exact midpoints).
// Verifies correct rounding direction in the non-tie case.
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp8E5M2FnuzRounding,
                         ::testing::Values(
                             // Slightly above 1.0 (0x40): closer to 1.0 than 1.25 -> 1.0
                             RoundingTestCase{1.1f, 1.0f},
                             // Slightly above midpoint 1.125: closer to 1.25 -> 1.25
                             RoundingTestCase{1.15f, 1.25f},
                             // Between 49152.0 (0x7E) and 57344.0 (0x7F): closer to 49152 -> 49152
                             RoundingTestCase{50000.0f, 49152.0f},
                             // Between 49152.0 and 57344.0: closer to 57344 -> 57344
                             RoundingTestCase{56000.0f, 57344.0f},
                             // Negative boundary
                             RoundingTestCase{-1.1f, -1.0f}));

// Rounding where mantissa overflow from the round-up cascades into the exponent.
// The critical case is a value whose fp8 mantissa is 3 (odd), falling exactly at the
// midpoint above 57344.0 (the MAX): the round-up causes mant=4 (overflow) -> exp=32
// (overflow) -> saturate to MAX=57344.0.
INSTANTIATE_TEST_SUITE_P(
    RoundingOverflow,
    TestFp8E5M2FnuzRounding,
    ::testing::Values(
        // Mantissa overflow: exp increments, but stays within range
        // 0x43 (mant=3, odd) = 1.75; midpoint to 0x44=2.0 causes mant overflow -> 2.0
        RoundingTestCase{1.875f, 2.0f},
        // 0x47 (mant=3, odd) = 3.5; midpoint to 0x48=4.0 causes mant overflow -> 4.0
        RoundingTestCase{3.75f, 4.0f},
        // Exponent overflow into saturation:
        // 61440.0 is exactly the midpoint between MAX=57344.0 and phantom 65536.0.
        // fp8exp=31, fp8Mant=3 (odd) -> round up -> mant=4 overflow -> exp=32 -> saturate
        RoundingTestCase{61440.0f, 57344.0f},
        // Negative counterparts
        RoundingTestCase{-1.875f, -2.0f},
        RoundingTestCase{-3.75f, -4.0f},
        RoundingTestCase{-61440.0f, -57344.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp8E5M2Fnuz, SaturationPositive)
{
    const fp8_e5m2_fnuz val1(1e10f);
    EXPECT_EQ(static_cast<float>(val1), 57344.0f);

    const fp8_e5m2_fnuz val2(60000.0f);
    EXPECT_EQ(static_cast<float>(val2), 57344.0f);

    const fp8_e5m2_fnuz val3(57345.0f);
    EXPECT_EQ(static_cast<float>(val3), 57344.0f);

    const fp8_e5m2_fnuz val4(std::nextafter(57344.0f, 1e9f));
    EXPECT_EQ(static_cast<float>(val4), 57344.0f);

    // Rounding-into-saturation: 61440.0 is the exact midpoint between MAX=57344 and
    // phantom 65536.0. Mantissa is 3 (odd) so round-up overflows exp -> saturate.
    const fp8_e5m2_fnuz val5(61440.0f);
    EXPECT_EQ(static_cast<float>(val5), 57344.0f);
}

TEST(TestFp8E5M2Fnuz, SaturationNegative)
{
    const fp8_e5m2_fnuz val1(-1e10f);
    EXPECT_EQ(static_cast<float>(val1), -57344.0f);

    const fp8_e5m2_fnuz val2(-60000.0f);
    EXPECT_EQ(static_cast<float>(val2), -57344.0f);

    const fp8_e5m2_fnuz val3(-57345.0f);
    EXPECT_EQ(static_cast<float>(val3), -57344.0f);

    // Rounding-into-saturation: -61440.0 mirrors the positive case above
    const fp8_e5m2_fnuz val4(-61440.0f);
    EXPECT_EQ(static_cast<float>(val4), -57344.0f);
}

TEST(TestFp8E5M2Fnuz, SaturationInfinity)
{
    const fp8_e5m2_fnuz posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 57344.0f);

    const fp8_e5m2_fnuz negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -57344.0f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp8E5M2Fnuz, Underflow)
{
    const fp8_e5m2_fnuz val1(1e-10f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    const fp8_e5m2_fnuz val2(-1e-10f);
    EXPECT_EQ(static_cast<float>(val2), 0.0f);

    const fp8_e5m2_fnuz val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    const fp8_e5m2_fnuz val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    const fp8_e5m2_fnuz val5(1e-6f);
    EXPECT_EQ(static_cast<float>(val5), 0.0f);

    // The exact midpoint 2^-18 rounds to zero (even mantissa tie-break)
    const fp8_e5m2_fnuz val6(3.814697265625e-06f);
    EXPECT_EQ(static_cast<float>(val6), 0.0f);

    // Just above the midpoint rounds up to denorm_min = 2^-17
    const fp8_e5m2_fnuz val7(1.5f * 3.814697265625e-06f);
    EXPECT_EQ(val7.data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(val7), 7.62939453125e-06f);

    const fp8_e5m2_fnuz val8(-1.5f * 3.814697265625e-06f);
    EXPECT_EQ(val8.data, static_cast<uint8_t>(0x81));
    EXPECT_EQ(static_cast<float>(val8), -7.62939453125e-06f);
}

// ============================================================================
// E5M2-FNUZ-Specific: NaN, Infinity, Negative Zero, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp8E5M2Fnuz, RoundTripAllPatterns)
{
    // For every bit pattern: decode to float via the lookup table, re-encode back,
    // and verify that the resulting bit pattern is identical.
    // Bit pattern 0x80 is the single NaN encoding; re-encoding a float NaN returns
    // 0x80 by definition, so it participates in the round-trip correctly.
    // Subnormal bit patterns 0x01-0x03 and 0x81-0x83 decode to their true subnormal
    // float values, which the encoder re-encodes back to the same bit pattern.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const auto decoded = fp8_e5m2_fnuz::from_bits(pattern);
        const auto f = static_cast<float>(decoded);

        // NaN round-trip: float NaN re-encodes to 0x80.
        if(pattern == 0x80)
        {
            EXPECT_TRUE(std::isnan(f)) << "Pattern 0x80 should decode to NaN";
            EXPECT_EQ(fp8_e5m2_fnuz(f).data, static_cast<uint8_t>(0x80));
            continue;
        }

        // All patterns must survive the round-trip exactly.
        const fp8_e5m2_fnuz reencoded(f);
        EXPECT_EQ(reencoded.data, pattern) << "Round-trip failed for bit pattern 0x" << std::hex
                                           << bits << " (float value " << std::dec << f << ")";
    }
}

TEST(TestFp8E5M2Fnuz, NoNegativeZero)
{
    // FNUZ has no -0; -0.0f input must produce bit pattern 0x00
    EXPECT_EQ(fp8_e5m2_fnuz(-0.0f).data, static_cast<uint8_t>(0x00));
    EXPECT_EQ(fp8_e5m2_fnuz(0.0f).data, static_cast<uint8_t>(0x00));
}

TEST(TestFp8E5M2Fnuz, NegateZeroProducesNaN)
{
    // FNUZ has no -0; negating zero flips the sign bit: 0x00 -> 0x80 = NaN
    auto val = -fp8_e5m2_fnuz(0.0f);
    EXPECT_TRUE(isnan(val));
    EXPECT_EQ(val.data, static_cast<uint8_t>(0x80));
}

TEST(TestFp8E5M2Fnuz, NegateNaNProducesZero)
{
    // NaN (0x80) negated flips the sign bit: 0x80 -> 0x00 = zero
    auto val = -fp8_e5m2_fnuz::from_bits(0x80);
    EXPECT_EQ(static_cast<float>(val), 0.0f);
    EXPECT_EQ(val.data, static_cast<uint8_t>(0x00));
}

TEST(TestFp8E5M2Fnuz, SingleNanEncoding)
{
    // Float NaN must map to 0x80
    auto v = fp8_e5m2_fnuz(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(v.data, static_cast<uint8_t>(0x80));
    EXPECT_TRUE(isnan(v));

    // signaling NaN also maps to 0x80
    auto vs = fp8_e5m2_fnuz(std::numeric_limits<float>::signaling_NaN());
    EXPECT_EQ(vs.data, static_cast<uint8_t>(0x80));
    EXPECT_TRUE(isnan(vs));

    // Every bit pattern except 0x80 is not NaN
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        if(bits == 0x80)
        {
            continue;
        }
        EXPECT_FALSE(isnan(fp8_e5m2_fnuz::from_bits(static_cast<uint8_t>(bits))))
            << "Bit pattern 0x" << std::hex << bits << " should not be NaN";
    }
}

TEST(TestFp8E5M2Fnuz, IsinfAlwaysFalse)
{
    // No bit pattern in fp8_e5m2_fnuz represents infinity
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        EXPECT_FALSE(isinf(fp8_e5m2_fnuz::from_bits(static_cast<uint8_t>(bits))))
            << "Bit pattern 0x" << std::hex << bits << " should not be infinity";
    }
}

TEST(TestFp8E5M2Fnuz, MathFunctions)
{
    // isfinite: all values except NaN are finite
    EXPECT_TRUE(isfinite(fp8_e5m2_fnuz(1.0f)));
    EXPECT_TRUE(isfinite(fp8_e5m2_fnuz(0.0f)));
    EXPECT_TRUE(isfinite(fp8_e5m2_fnuz(57344.0f)));
    EXPECT_FALSE(isfinite(fp8_e5m2_fnuz::from_bits(0x80)));

    // signbit: positive values have sign bit 0, negative have sign bit 1
    EXPECT_FALSE(signbit(fp8_e5m2_fnuz(1.0f)));
    EXPECT_TRUE(signbit(fp8_e5m2_fnuz(-1.0f)));
    EXPECT_FALSE(signbit(fp8_e5m2_fnuz(0.0f)));
    // The sole NaN encoding 0x80 has the sign bit set
    EXPECT_TRUE(signbit(fp8_e5m2_fnuz::from_bits(0x80)));
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp8E5M2Fnuz, NumericLimitsSpecificValues)
{
    using L = std::numeric_limits<fp8_e5m2_fnuz>;

    // max() = 57344.0
    EXPECT_EQ(static_cast<float>(L::max()), 57344.0f);
    EXPECT_EQ(L::max().data, static_cast<uint8_t>(0x7F));

    // min() = 2^-15 = 3.0517578125e-5
    EXPECT_EQ(static_cast<float>(L::min()), 3.0517578125e-5f);
    EXPECT_EQ(L::min().data, static_cast<uint8_t>(0x04));

    // lowest() = -57344.0
    EXPECT_EQ(static_cast<float>(L::lowest()), -57344.0f);
    EXPECT_EQ(L::lowest().data, static_cast<uint8_t>(0xFF));

    // epsilon() = 2^-2 = 0.25
    EXPECT_EQ(static_cast<float>(L::epsilon()), 0.25f);
    EXPECT_EQ(L::epsilon().data, static_cast<uint8_t>(0x38));

    // round_error() = 0.5
    EXPECT_EQ(static_cast<float>(L::round_error()), 0.5f);
    EXPECT_EQ(L::round_error().data, static_cast<uint8_t>(0x3C));

    // denorm_min() returns bit pattern 0x01 = 1 x 2^-17 = 7.62939453125e-6 (minimum positive subnormal).
    EXPECT_EQ(L::denorm_min().data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(L::denorm_min()), 7.62939453125e-06f);

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

TEST(TestFp8E5M2Fnuz, NamedConstants)
{
    using L = std::numeric_limits<fp8_e5m2_fnuz>;

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
