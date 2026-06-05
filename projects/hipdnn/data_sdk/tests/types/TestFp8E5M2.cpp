// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp8E5M2.cpp
/// @brief Type-specific tests for fp8_e5m2 (OCP E5M2 format).
///
/// Common tests (type properties, construction, math functions, numeric_limits) are in
/// TestPortableTypes.cpp. This file covers exact-value round-trip, exhaustive 256-bit-pattern
/// round-trip, OCP E5M2-essential semantics (Inf at 0x7C/0xFC, multi-encoding NaN family
/// where exp=31 && mant!=0, saturate-to-MAX vs Inf, signed zero), rounding (parametrized),
/// saturation, underflow, and numeric_limits checks.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// fp8_e5m2 Type-Specific Tests
// ============================================================================

TEST(TestFp8E5M2, RoundTripExactValues)
{
    // These values are exactly representable in fp8_e5m2 (OCP E5M2 format)
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
        6.103515625e-5f, // MIN_NORMAL = 2^-14
        -6.103515625e-5f,
        1.52587890625e-5f, // denorm_min = 2^-16
        -1.52587890625e-5f,
    };

    for(const float val : exactValues)
    {
        const fp8_e5m2 v(val);
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

class TestFp8E5M2Rounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp8E5M2Rounding, Rounding)
{
    auto [input, expected] = GetParam();
    const fp8_e5m2 val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

// Midpoint values that are exactly halfway between two representable fp8_e5m2
// encodings; round-to-nearest-even selects the encoding with an even mantissa.
// Lookup table: 0x40=1.0, 0x41=1.25, 0x42=1.5, 0x7A=49152.0, 0x7B=57344.0 (MAX).
INSTANTIATE_TEST_SUITE_P(
    MidpointRounding,
    TestFp8E5M2Rounding,
    ::testing::Values(
        // 0x40 (mant=0, even) vs 0x41 (mant=1, odd): midpoint=1.125 -> round down -> 1.0
        RoundingTestCase{1.125f, 1.0f},
        // 0x41 (mant=1, odd) vs 0x42 (mant=2, even): midpoint=1.375 -> round up -> 1.5
        RoundingTestCase{1.375f, 1.5f},
        // 0x7A (mant=2, even) vs 0x7B (mant=3, odd): midpoint=53248.0 -> round down -> 49152.0
        RoundingTestCase{53248.0f, 49152.0f}));

// Values in the subnormal/underflow range.
// OCP E5M2 denorms: value = mant * 2^-16 (mant=1..3); denorm_min = 2^-16 = 1.52587890625e-5.
// Underflow threshold = half of denorm_min = 2^-17 = 7.62939453125e-6.
// At exactly 2^-17 the result rounds to 0 (even mantissa tie-break).
INSTANTIATE_TEST_SUITE_P(
    SubnormalRounding,
    TestFp8E5M2Rounding,
    ::testing::Values(
        // Well below underflow threshold (2^-17): flushes to zero
        RoundingTestCase{1e-6f, 0.0f},
        // At exactly the underflow threshold 2^-17: round to even -> zero
        RoundingTestCase{7.62939453125e-6f, 0.0f},
        // Slightly above 2^-17: rounds to denorm_min = 2^-16
        RoundingTestCase{8e-6f, 1.52587890625e-5f},
        // Midpoint between 0x01 (mant=1 odd) and 0x02 (mant=2 even): rounds up to 0x02
        RoundingTestCase{2.288818359375e-5f, 3.0517578125e-5f},
        // Even-LSB subnormal midpoint: 0x02 (mant=2, even) vs 0x03 (mant=3, odd).
        // Midpoint = 3.814697265625e-5; ties-to-even -> round DOWN to 0x02 = 3.0517578125e-5.
        RoundingTestCase{3.814697265625e-5f, 3.0517578125e-5f},
        // Denormal-to-normal carry, below midpoint: 0x03=4.57763671875e-5 (mant=3, odd),
        // 0x04=6.103515625e-5 (smallest normal). Below midpoint -> 0x03.
        RoundingTestCase{4.96246337890625e-5f, 4.57763671875e-5f},
        // Exact carry midpoint: subnormal mant=3 odd -> round up -> mant=4 overflow ->
        // carry into smallest normal 0x04 = 6.103515625e-5.
        RoundingTestCase{5.340576171875e-5f, 6.103515625e-5f},
        // Above carry midpoint: definitively rounds up to smallest normal.
        RoundingTestCase{5.7220458984375e-5f, 6.103515625e-5f},
        RoundingTestCase{-5.7220458984375e-5f, -6.103515625e-5f}));

// Values at the boundary between two adjacent encodings (not at exact midpoints).
// Verifies correct rounding direction in the non-tie case.
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp8E5M2Rounding,
                         ::testing::Values(
                             // Slightly above 1.0 (0x40): closer to 1.0 than 1.25 -> 1.0
                             RoundingTestCase{1.1f, 1.0f},
                             // Slightly above midpoint 1.125: closer to 1.25 -> 1.25
                             RoundingTestCase{1.15f, 1.25f},
                             // Between 49152.0 (0x7A) and 57344.0 (0x7B): closer to 49152 -> 49152
                             RoundingTestCase{50000.0f, 49152.0f},
                             // Between 49152.0 and 57344.0: closer to 57344 -> 57344
                             RoundingTestCase{56000.0f, 57344.0f},
                             // Negative boundary
                             RoundingTestCase{-1.1f, -1.0f}));

// Rounding where mantissa overflow from the round-up cascades into the exponent,
// or where the final exponent overflows into saturation or Inf.
//
// OCP E5M2 mant bits = 2; when fp8Mant rounds from 3 to 4 (overflow), exp increments.
// If exp reaches the reserved Inf/NaN exponent (31), the value produces Inf
// (saturate=false) or MAX (saturate=true).
// Saturation midpoint: MAX=57344 (exp=30, mant=3=odd), phantom next = 65536 (Inf territory).
// Midpoint = 61440; mant=3 is odd -> round UP -> Inf or MAX.
INSTANTIATE_TEST_SUITE_P(
    RoundingOverflow,
    TestFp8E5M2Rounding,
    ::testing::Values(
        // Mantissa overflow: exp increments, stays in range.
        // 0x43 (mant=3, odd) = 1.75; midpoint to 0x44=2.0 -> mant overflow -> 2.0
        RoundingTestCase{1.875f, 2.0f},
        // Exponent overflow into saturation:
        // 61440 is midpoint between MAX=57344 (mant=3 odd) and phantom 65536 -> round up -> MAX
        RoundingTestCase{61440.0f, 57344.0f},
        // Negative counterpart for mantissa-overflow cascade
        RoundingTestCase{-1.875f, -2.0f},
        // Negative counterpart for saturation
        RoundingTestCase{-61440.0f, -57344.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp8E5M2, SaturationPositive)
{
    const fp8_e5m2 val1(1e10f);
    EXPECT_EQ(static_cast<float>(val1), 57344.0f);

    const fp8_e5m2 val2(60000.0f);
    EXPECT_EQ(static_cast<float>(val2), 57344.0f);

    const fp8_e5m2 val3(57345.0f);
    EXPECT_EQ(static_cast<float>(val3), 57344.0f);

    const fp8_e5m2 val4(std::nextafter(57344.0f, 1e9f));
    EXPECT_EQ(static_cast<float>(val4), 57344.0f);

    // Rounding-into-saturation: 61440.0 (midpoint, mant=3 odd) -> rounds up -> MAX.
    const fp8_e5m2 val5(61440.0f);
    EXPECT_EQ(static_cast<float>(val5), 57344.0f);
}

TEST(TestFp8E5M2, SaturationNegative)
{
    const fp8_e5m2 val1(-1e10f);
    EXPECT_EQ(static_cast<float>(val1), -57344.0f);

    const fp8_e5m2 val2(-60000.0f);
    EXPECT_EQ(static_cast<float>(val2), -57344.0f);

    const fp8_e5m2 val3(-57345.0f);
    EXPECT_EQ(static_cast<float>(val3), -57344.0f);

    // Rounding-into-saturation mirrors positive case
    const fp8_e5m2 val4(-61440.0f);
    EXPECT_EQ(static_cast<float>(val4), -57344.0f);
}

TEST(TestFp8E5M2, SaturationInfinity)
{
    const fp8_e5m2 posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 57344.0f);

    const fp8_e5m2 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -57344.0f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp8E5M2, Underflow)
{
    const fp8_e5m2 val1(1e-10f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    const fp8_e5m2 val2(-1e-10f);
    EXPECT_EQ(static_cast<float>(val2), 0.0f);

    // fp32 denormal inputs flush to zero (fp32Exp==0 path)
    const fp8_e5m2 val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    const fp8_e5m2 val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    const fp8_e5m2 val5(1e-6f);
    EXPECT_EQ(static_cast<float>(val5), 0.0f);

    // The exact underflow threshold 2^-17 rounds to zero (even mantissa tie-break).
    const fp8_e5m2 val6(7.62939453125e-6f);
    EXPECT_EQ(static_cast<float>(val6), 0.0f);

    // Just above the threshold rounds up to denorm_min = 2^-16.
    const fp8_e5m2 val7(1.5f * 7.62939453125e-6f);
    EXPECT_EQ(val7.data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(val7), 1.52587890625e-5f);

    const fp8_e5m2 val8(-1.5f * 7.62939453125e-6f);
    EXPECT_EQ(val8.data, static_cast<uint8_t>(0x81));
    EXPECT_EQ(static_cast<float>(val8), -1.52587890625e-5f);
}

// ============================================================================
// E5M2-OCP-Specific: Infinity, NaN, Signed Zero, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp8E5M2, RoundTripAllPatterns)
{
    // For every bit pattern: decode to float via the lookup table, re-encode back,
    // and verify that the resulting bit pattern is identical.
    // OCP E5M2 NaN: any pattern with (bits & 0x7C) == 0x7C && (bits & 0x03) != 0.
    // OCP E5M2 Inf: (bits & 0x7F) == 0x7C (i.e. exp=31, mant=0). The default-saturating
    // ctor cannot round-trip Inf back to its bit pattern (it saturates to MAX), so Inf
    // patterns are checked for decode only.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const auto decoded = fp8_e5m2::from_bits(pattern);
        const auto f = static_cast<float>(decoded);

        // NaN patterns: OCP E5M2 NaN has exp=31 and mant != 0.
        // Multiple input NaN encodings collapse to the canonical 0x7F/0xFF on re-encode,
        // so the round-trip check is "still NaN" rather than bit-pattern equality.
        if(((pattern & 0x7C) == 0x7C) && ((pattern & 0x03) != 0))
        {
            EXPECT_TRUE(std::isnan(f))
                << "Pattern 0x" << std::hex << bits << " should decode to NaN";
            EXPECT_TRUE(isnan(fp8_e5m2(f)));
            continue;
        }

        // Inf patterns (exp=31, mant=0): decode to Inf, re-encode saturates to MAX/LOWEST.
        if((pattern & 0x7F) == 0x7C)
        {
            EXPECT_TRUE(std::isinf(f))
                << "Pattern 0x" << std::hex << bits << " should decode to Inf";
            const uint8_t expectedSaturated = std::signbit(f) ? 0xFB : 0x7B;
            EXPECT_EQ(fp8_e5m2(f).data, expectedSaturated)
                << "Inf at 0x" << std::hex << bits << " should saturate to 0x"
                << static_cast<int>(expectedSaturated);
            continue;
        }

        // All finite, non-NaN patterns must survive the round-trip exactly.
        const fp8_e5m2 reencoded(f);
        EXPECT_EQ(reencoded.data, pattern) << "Round-trip failed for bit pattern 0x" << std::hex
                                           << bits << " (float value " << std::dec << f << ")";
    }
}

TEST(TestFp8E5M2, SignedZeroSupported)
{
    // OCP E5M2 supports both +0 (0x00) and -0 (0x80); both compare equal to 0.0f.
    EXPECT_EQ(fp8_e5m2(0.0f).data, static_cast<uint8_t>(0x00));
    EXPECT_EQ(fp8_e5m2(-0.0f).data, static_cast<uint8_t>(0x80));
    EXPECT_EQ(static_cast<float>(fp8_e5m2::from_bits(0x80)), -0.0f);
}

TEST(TestFp8E5M2, NegateZeroProducesNegativeZero)
{
    // OCP E5M2 supports -0; negating +0 flips the sign bit -> 0x80 (-0).
    auto val = -fp8_e5m2(0.0f);
    EXPECT_EQ(val.data, static_cast<uint8_t>(0x80));
    EXPECT_EQ(static_cast<float>(val), -0.0f);
    EXPECT_FALSE(isnan(val));
}

TEST(TestFp8E5M2, NegateNanProducesNan)
{
    // OCP E5M2: negating any NaN encoding flips the sign bit; the result is still NaN.
    auto posNan = fp8_e5m2::from_bits(0x7F);
    auto negNan = -posNan;
    EXPECT_EQ(negNan.data, static_cast<uint8_t>(0xFF));
    EXPECT_TRUE(isnan(negNan));

    // Non-canonical NaN encodings (0x7D, 0x7E) also negate correctly.
    auto altPosNan = fp8_e5m2::from_bits(0x7D);
    auto altNegNan = -altPosNan;
    EXPECT_EQ(altNegNan.data, static_cast<uint8_t>(0xFD));
    EXPECT_TRUE(isnan(altNegNan));
}

TEST(TestFp8E5M2, NegateInfinityProducesInfinity)
{
    // OCP E5M2: negating +Inf (0x7C) gives -Inf (0xFC), and vice versa.
    auto posInf = fp8_e5m2::from_bits(0x7C);
    auto negInf = -posInf;
    EXPECT_EQ(negInf.data, static_cast<uint8_t>(0xFC));
    EXPECT_TRUE(isinf(negInf));
    EXPECT_TRUE(signbit(negInf));

    auto fromNeg = fp8_e5m2::from_bits(0xFC);
    auto backToPos = -fromNeg;
    EXPECT_EQ(backToPos.data, static_cast<uint8_t>(0x7C));
    EXPECT_TRUE(isinf(backToPos));
    EXPECT_FALSE(signbit(backToPos));
}

TEST(TestFp8E5M2, InfinityTruthTable)
{
    // OCP E5M2 has infinity at 0x7C (+Inf) and 0xFC (-Inf).
    // All other bit patterns must NOT be infinity.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const bool expectedInf = (bits == 0x7C || bits == 0xFC);
        EXPECT_EQ(isinf(fp8_e5m2::from_bits(static_cast<uint8_t>(bits))), expectedInf)
            << "isinf wrong for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E5M2, NanTruthTable)
{
    // OCP E5M2: isnan is true when exp field = 11111 (=31) AND mant != 0.
    // Bit pattern structure: [S|EEEEE|MM]; NaN: (bits & 0x7C) == 0x7C && (bits & 0x03) != 0.
    for(int bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const bool expectedNan = ((pattern & 0x7C) == 0x7C) && ((pattern & 0x03) != 0);
        EXPECT_EQ(isnan(fp8_e5m2::from_bits(pattern)), expectedNan)
            << "isnan wrong for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E5M2, MathFunctions)
{
    // isfinite: finite for all non-Inf, non-NaN
    EXPECT_TRUE(isfinite(fp8_e5m2(1.0f)));
    EXPECT_TRUE(isfinite(fp8_e5m2(0.0f)));
    EXPECT_TRUE(isfinite(fp8_e5m2(57344.0f)));
    EXPECT_FALSE(isfinite(fp8_e5m2::from_bits(0x7C))); // +Inf
    EXPECT_FALSE(isfinite(fp8_e5m2::from_bits(0x7F))); // NaN

    // isinf: true only for 0x7C and 0xFC
    EXPECT_TRUE(isinf(fp8_e5m2::from_bits(0x7C)));
    EXPECT_TRUE(isinf(fp8_e5m2::from_bits(0xFC)));
    EXPECT_FALSE(isinf(fp8_e5m2(1.0f)));
    EXPECT_FALSE(isinf(fp8_e5m2::from_bits(0x7F))); // NaN, not Inf

    // isnan: true for exp=31 && mant != 0 (OCP E5M2 does not distinguish QNAN/SNAN)
    EXPECT_TRUE(isnan(fp8_e5m2::from_bits(0x7F))); // canonical NaN
    EXPECT_TRUE(isnan(fp8_e5m2::from_bits(0x7D))); // alternate NaN encoding
    EXPECT_FALSE(isnan(fp8_e5m2(1.0f)));
    EXPECT_FALSE(isnan(fp8_e5m2::from_bits(0x7C))); // Inf, not NaN

    // signbit
    EXPECT_FALSE(signbit(fp8_e5m2(1.0f)));
    EXPECT_TRUE(signbit(fp8_e5m2(-1.0f)));
    EXPECT_FALSE(signbit(fp8_e5m2(0.0f)));
    EXPECT_TRUE(signbit(fp8_e5m2(-0.0f)));
    EXPECT_FALSE(signbit(fp8_e5m2::from_bits(0x7C))); // +Inf
    EXPECT_TRUE(signbit(fp8_e5m2::from_bits(0xFC))); // -Inf
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp8E5M2, NumericLimitsSpecificValues)
{
    using L = std::numeric_limits<fp8_e5m2>;

    // max() = 57344.0
    EXPECT_EQ(static_cast<float>(L::max()), 57344.0f);
    EXPECT_EQ(L::max().data, static_cast<uint8_t>(0x7B));

    // min() = 2^-14 = 6.103515625e-5
    EXPECT_EQ(static_cast<float>(L::min()), 6.103515625e-5f);
    EXPECT_EQ(L::min().data, static_cast<uint8_t>(0x04));

    // lowest() = -57344.0
    EXPECT_EQ(static_cast<float>(L::lowest()), -57344.0f);
    EXPECT_EQ(L::lowest().data, static_cast<uint8_t>(0xFB));

    // epsilon() = 2^-2 = 0.25
    EXPECT_EQ(static_cast<float>(L::epsilon()), 0.25f);
    EXPECT_EQ(L::epsilon().data, static_cast<uint8_t>(0x34));

    // round_error() = 0.5
    EXPECT_EQ(static_cast<float>(L::round_error()), 0.5f);
    EXPECT_EQ(L::round_error().data, static_cast<uint8_t>(0x38));

    // denorm_min() returns bit pattern 0x01 = 2^-16 = 1.52587890625e-5
    // (minimum positive subnormal).
    EXPECT_EQ(L::denorm_min().data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(static_cast<float>(L::denorm_min()), 1.52587890625e-5f);

    // infinity() = 0x7C (OCP E5M2 has infinity).
    EXPECT_EQ(L::infinity().data, static_cast<uint8_t>(0x7C));
    EXPECT_TRUE(isinf(L::infinity()));
    EXPECT_FALSE(signbit(L::infinity()));

    // quiet_NaN() and signaling_NaN() both return 0x7F (canonical NaN).
    // (OCP E5M2 does not distinguish between signaling and quiet NaN.)
    EXPECT_EQ(L::quiet_NaN().data, static_cast<uint8_t>(0x7F));
    EXPECT_EQ(L::signaling_NaN().data, static_cast<uint8_t>(0x7F));
    EXPECT_TRUE(isnan(L::quiet_NaN()));
    EXPECT_TRUE(isnan(L::signaling_NaN()));

    // has_infinity = true
    EXPECT_TRUE(L::has_infinity);

    // has_quiet_NaN = true; has_signaling_NaN = false (OCP E5M2 has no SNaN distinction).
    EXPECT_TRUE(L::has_quiet_NaN);
    EXPECT_FALSE(L::has_signaling_NaN);

    // has_denorm = denorm_present (OCP E5M2 supports subnormals).
    EXPECT_EQ(L::has_denorm, std::denorm_present);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST(TestFp8E5M2, NamedConstants)
{
    using L = std::numeric_limits<fp8_e5m2>;

    // OCP E5M2 has infinity at 0x7C (+Inf) / 0xFC (-Inf).
    EXPECT_TRUE(L::has_infinity);
    EXPECT_EQ(L::infinity().data, static_cast<uint8_t>(0x7C));
    EXPECT_TRUE(isinf(L::infinity()));

    // OCP E5M2 NaN family (exp=31, mant!=0); canonical encoding is 0x7F.
    // OCP E5M2 does not distinguish signaling NaN from quiet NaN.
    EXPECT_EQ(L::quiet_NaN().data, static_cast<uint8_t>(0x7F));
    EXPECT_EQ(L::signaling_NaN().data, static_cast<uint8_t>(0x7F));
    EXPECT_TRUE(L::has_quiet_NaN);
    EXPECT_FALSE(L::has_signaling_NaN);

    // OCP E5M2 supports -0; denorm_min is the smallest positive subnormal.
    EXPECT_EQ(L::has_denorm, std::denorm_present);
}
