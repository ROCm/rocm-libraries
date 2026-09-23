// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp6E3M2.cpp
/// @brief Type-specific tests for fp6_e3m2, fp6x2_e3m2, and fp6x4_e3m2 (OCP MX E3M2 format).
///
/// Common tests (type properties, construction, math functions, numeric_limits) are in
/// TestPortableTypes.cpp. This file contains fp6_e3m2-specific tests for round-to-nearest-even
/// behavior, saturation, underflow, rounding edge cases, and the packed storage types.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// fp6_e3m2 Type-Specific Tests
// ============================================================================

TEST(TestFp6E3M2, RoundTripAllValues)
{
    // clang-format off
    const std::vector<float> exactValues = {
        // Positive subnormals and zero
        0.0f,    0.0625f, 0.125f,  0.1875f,
        // Positive normals
        0.25f,   0.3125f, 0.375f,  0.4375f,
        0.5f,    0.625f,  0.75f,   0.875f,
        1.0f,    1.25f,   1.5f,    1.75f,
        2.0f,    2.5f,    3.0f,    3.5f,
        4.0f,    5.0f,    6.0f,    7.0f,
        8.0f,    10.0f,   12.0f,   14.0f,
        16.0f,   20.0f,   24.0f,   28.0f,
        // Negative subnormals and zero
        -0.0f,   -0.0625f, -0.125f, -0.1875f,
        // Negative normals
        -0.25f,  -0.3125f, -0.375f, -0.4375f,
        -0.5f,   -0.625f,  -0.75f,  -0.875f,
        -1.0f,   -1.25f,   -1.5f,   -1.75f,
        -2.0f,   -2.5f,    -3.0f,   -3.5f,
        -4.0f,   -5.0f,    -6.0f,   -7.0f,
        -8.0f,   -10.0f,   -12.0f,  -14.0f,
        -16.0f,  -20.0f,   -24.0f,  -28.0f
    };
    // clang-format on

    for(const float val : exactValues)
    {
        const fp6_e3m2 f6(val);
        EXPECT_EQ(static_cast<float>(f6), val);
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

class TestFp6E3M2Rounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp6E3M2Rounding, Rounding)
{
    auto [input, expected] = GetParam();
    const fp6_e3m2 val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

INSTANTIATE_TEST_SUITE_P(MidpointRounding,
                         TestFp6E3M2Rounding,
                         ::testing::Values(RoundingTestCase{0.09375f, 0.125f},
                                           RoundingTestCase{0.15625f, 0.125f},
                                           RoundingTestCase{1.125f, 1.0f},
                                           RoundingTestCase{1.375f, 1.5f},
                                           RoundingTestCase{2.25f, 2.0f},
                                           RoundingTestCase{2.75f, 3.0f},
                                           RoundingTestCase{4.5f, 4.0f},
                                           RoundingTestCase{5.5f, 6.0f},
                                           RoundingTestCase{9.0f, 8.0f},
                                           RoundingTestCase{18.0f, 16.0f}));

INSTANTIATE_TEST_SUITE_P(SubnormalRounding,
                         TestFp6E3M2Rounding,
                         ::testing::Values(RoundingTestCase{0.05f, 0.0625f},
                                           RoundingTestCase{0.07f, 0.0625f},
                                           RoundingTestCase{0.1f, 0.125f},
                                           RoundingTestCase{0.03f, 0.0f},
                                           RoundingTestCase{0.16f, 0.1875f},
                                           RoundingTestCase{-0.05f, -0.0625f},
                                           RoundingTestCase{-0.1f, -0.125f}));

INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp6E3M2Rounding,
                         ::testing::Values(RoundingTestCase{26.0f, 24.0f},
                                           RoundingTestCase{27.0f, 28.0f},
                                           RoundingTestCase{22.0f, 24.0f},
                                           RoundingTestCase{11.0f, 12.0f},
                                           RoundingTestCase{13.0f, 12.0f},
                                           RoundingTestCase{-26.0f, -24.0f}));

INSTANTIATE_TEST_SUITE_P(
    RoundingOverflow,
    TestFp6E3M2Rounding,
    ::testing::Values(
        // Mantissa overflow from rounding causes exponent increment
        RoundingTestCase{1.875f, 2.0f}, // e=3 to e=4: mant overflow 11->00, exp++
        RoundingTestCase{3.75f, 4.0f}, // e=4 to e=5: mant overflow 11->00, exp++
        RoundingTestCase{7.5f, 8.0f}, // e=5 to e=6: mant overflow 11->00, exp++
        RoundingTestCase{15.0f, 16.0f}, // e=6 to e=7: mant overflow 11->00, exp++
        // Exponent overflow from rounding causes saturation to max
        RoundingTestCase{30.0f, 28.0f}, // e=7 rounds up but exp>7 saturates to max
        RoundingTestCase{-1.875f, -2.0f}, // Negative: e=3 to e=4
        RoundingTestCase{-7.5f, -8.0f}, // Negative: e=5 to e=6
        RoundingTestCase{-30.0f, -28.0f})); // Negative: saturates to min

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp6E3M2, SaturationPositive)
{
    const fp6_e3m2 val1(30.0f);
    EXPECT_EQ(static_cast<float>(val1), 28.0f);

    const fp6_e3m2 val2(100.0f);
    EXPECT_EQ(static_cast<float>(val2), 28.0f);
}

TEST(TestFp6E3M2, SaturationNegative)
{
    const fp6_e3m2 val1(-30.0f);
    EXPECT_EQ(static_cast<float>(val1), -28.0f);

    const fp6_e3m2 val2(-100.0f);
    EXPECT_EQ(static_cast<float>(val2), -28.0f);
}

TEST(TestFp6E3M2, SaturationInfinity)
{
    const fp6_e3m2 posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 28.0f);

    const fp6_e3m2 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -28.0f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp6E3M2, Underflow)
{
    const fp6_e3m2 val1(0.03f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    const fp6_e3m2 val2(-0.03f);
    EXPECT_EQ(static_cast<float>(val2), -0.0f);
    EXPECT_TRUE(std::signbit(static_cast<float>(val2)));

    // fp32 subnormal inputs (exercises shift > 24 path with fp32Exp == 0)
    const fp6_e3m2 val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    const fp6_e3m2 val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    const fp6_e3m2 val5(1e-40f);
    EXPECT_EQ(static_cast<float>(val5), 0.0f);
}

// ============================================================================
// E3M2-Specific: No NaN or Infinity (Exhaustive Check)
// ============================================================================

TEST(TestFp6E3M2, IsnanAlwaysFalse)
{
    for(uint8_t bits = 0; bits < 64; ++bits)
    {
        const fp6_e3m2 val = fp6_e3m2::from_bits(bits);
        EXPECT_FALSE(isnan(val));
    }
}

TEST(TestFp6E3M2, IsinfAlwaysFalse)
{
    for(uint8_t bits = 0; bits < 64; ++bits)
    {
        const fp6_e3m2 val = fp6_e3m2::from_bits(bits);
        EXPECT_FALSE(isinf(val));
    }
}

TEST(TestFp6E3M2, IsfiniteAlwaysTrue)
{
    for(uint8_t bits = 0; bits < 64; ++bits)
    {
        const fp6_e3m2 val = fp6_e3m2::from_bits(bits);
        EXPECT_TRUE(isfinite(val));
    }
}

TEST(TestFp6E3M2, NanConversionToZero)
{
    // Per OCP MX Spec, conversion from NaN is implementation-defined.
    // This implementation returns zero.
    const fp6_e3m2 fromNan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(static_cast<float>(fromNan), 0.0f);
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp6E3M2, NumericLimitsSpecificValues)
{
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e3m2>::max()), 28.0f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e3m2>::min()), 0.25f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e3m2>::lowest()), -28.0f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e3m2>::epsilon()), 0.25f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e3m2>::round_error()), 0.5f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e3m2>::denorm_min()), 0.0625f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST(TestFp6E3M2, NamedConstants)
{
    // E3M2 has no infinity - returns max
    auto maxVal = std::numeric_limits<fp6_e3m2>::max();
    EXPECT_EQ(std::numeric_limits<fp6_e3m2>::infinity().data, maxVal.data);

    // E3M2 has no NaN - returns zero (consistent with float-to-E3M2 conversion)
    auto zeroVal = fp6_e3m2();
    EXPECT_EQ(std::numeric_limits<fp6_e3m2>::quiet_NaN().data, zeroVal.data);
    EXPECT_EQ(std::numeric_limits<fp6_e3m2>::signaling_NaN().data, zeroVal.data);
}

// ============================================================================
// fp6x2_e3m2 Packed Storage Type Tests
// ============================================================================

TEST(TestFp6x2E3M2, DefaultConstruction)
{
    const fp6x2_e3m2 val;
    EXPECT_EQ(val.data, 0x0000);
    EXPECT_EQ(static_cast<float>(val.lo()), 0.0f);
    EXPECT_EQ(static_cast<float>(val.hi()), 0.0f);
}

TEST(TestFp6x2E3M2, SingleValueConstruction)
{
    const fp6_e3m2 elem(4.0f);
    const fp6x2_e3m2 packed(elem);
    EXPECT_EQ(static_cast<float>(packed.lo()), 4.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 0.0f);
}

TEST(TestFp6x2E3M2, TwoValueConstruction)
{
    const fp6_e3m2 lo(1.0f);
    const fp6_e3m2 hi(2.0f);
    const fp6x2_e3m2 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 2.0f);
}

TEST(TestFp6x2E3M2, DataLayout)
{
    // Verify storage model: lo = bits [5:0], hi = bits [11:6]
    // lo = 0x0C (1.0, exp=3, mant=0), hi = 0x10 (2.0, exp=4, mant=0)
    // data = 0x0C | (0x10 << 6) = 0x0C | 0x400 = 0x40C
    const fp6x2_e3m2 packed(fp6_e3m2::from_bits(0x0C), fp6_e3m2::from_bits(0x10));
    EXPECT_EQ(packed.data, 0x040C);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 2.0f);
}

TEST(TestFp6x2E3M2, NegativeValuesInLo)
{
    const fp6_e3m2 neg(-2.0f);
    const fp6x2_e3m2 packed(neg);
    EXPECT_EQ(static_cast<float>(packed.lo()), -2.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 0.0f);
}

TEST(TestFp6x2E3M2, NegativeValuesInHi)
{
    const fp6_e3m2 lo(1.0f);
    const fp6_e3m2 hi(-4.0f);
    const fp6x2_e3m2 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), -4.0f);
}

TEST(TestFp6x2E3M2, BothNegativeValues)
{
    const fp6_e3m2 lo(-1.5f);
    const fp6_e3m2 hi(-5.0f);
    const fp6x2_e3m2 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), -1.5f);
    EXPECT_EQ(static_cast<float>(packed.hi()), -5.0f);
}

TEST(TestFp6x2E3M2, CopyConstruct)
{
    const fp6x2_e3m2 a(fp6_e3m2(1.0f), fp6_e3m2(2.0f));
    const fp6x2_e3m2 b(a);
    EXPECT_EQ(a.data, b.data);
}

TEST(TestFp6x2E3M2, CopyAssignment)
{
    const fp6x2_e3m2 a(fp6_e3m2(1.0f), fp6_e3m2(2.0f));
    fp6x2_e3m2 b;
    b = a;
    EXPECT_EQ(a.data, b.data);
    EXPECT_EQ(static_cast<float>(b.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(b.hi()), 2.0f);
}

// ============================================================================
// fp6x4_e3m2 Packed Storage Type Tests (4 values in 3 bytes)
// ============================================================================

TEST(TestFp6x4E3M2, DefaultConstruction)
{
    fp6x4_e3m2 val;
    EXPECT_EQ(val.data[0], 0);
    EXPECT_EQ(val.data[1], 0);
    EXPECT_EQ(val.data[2], 0);

    auto loPair = val.lo_pair();
    auto hiPair = val.hi_pair();
    EXPECT_EQ(static_cast<float>(loPair.lo()), 0.0f);
    EXPECT_EQ(static_cast<float>(loPair.hi()), 0.0f);
    EXPECT_EQ(static_cast<float>(hiPair.lo()), 0.0f);
    EXPECT_EQ(static_cast<float>(hiPair.hi()), 0.0f);
}

TEST(TestFp6x4E3M2, TwoPairConstruction)
{
    const fp6x2_e3m2 loPair(fp6_e3m2(1.0f), fp6_e3m2(2.0f));
    const fp6x2_e3m2 hiPair(fp6_e3m2(4.0f), fp6_e3m2(8.0f));
    const fp6x4_e3m2 packed(loPair, hiPair);

    auto extractedLo = packed.lo_pair();
    auto extractedHi = packed.hi_pair();

    EXPECT_EQ(static_cast<float>(extractedLo.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(extractedLo.hi()), 2.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.lo()), 4.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.hi()), 8.0f);
}

TEST(TestFp6x4E3M2, NegativeValues)
{
    const fp6x2_e3m2 loPair(fp6_e3m2(-1.0f), fp6_e3m2(-2.0f));
    const fp6x2_e3m2 hiPair(fp6_e3m2(-4.0f), fp6_e3m2(-8.0f));
    const fp6x4_e3m2 packed(loPair, hiPair);

    auto extractedLo = packed.lo_pair();
    auto extractedHi = packed.hi_pair();

    EXPECT_EQ(static_cast<float>(extractedLo.lo()), -1.0f);
    EXPECT_EQ(static_cast<float>(extractedLo.hi()), -2.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.lo()), -4.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.hi()), -8.0f);
}

TEST(TestFp6x4E3M2, MixedValues)
{
    const fp6x2_e3m2 loPair(fp6_e3m2(0.5f), fp6_e3m2(28.0f));
    const fp6x2_e3m2 hiPair(fp6_e3m2(-0.5f), fp6_e3m2(-28.0f));
    const fp6x4_e3m2 packed(loPair, hiPair);

    auto extractedLo = packed.lo_pair();
    auto extractedHi = packed.hi_pair();

    EXPECT_EQ(static_cast<float>(extractedLo.lo()), 0.5f);
    EXPECT_EQ(static_cast<float>(extractedLo.hi()), 28.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.lo()), -0.5f);
    EXPECT_EQ(static_cast<float>(extractedHi.hi()), -28.0f);
}

TEST(TestFp6x4E3M2, CopyConstruct)
{
    const fp6x2_e3m2 loPair(fp6_e3m2(1.0f), fp6_e3m2(2.0f));
    const fp6x2_e3m2 hiPair(fp6_e3m2(4.0f), fp6_e3m2(8.0f));
    fp6x4_e3m2 a(loPair, hiPair);
    fp6x4_e3m2 b(a);

    EXPECT_EQ(a.data[0], b.data[0]);
    EXPECT_EQ(a.data[1], b.data[1]);
    EXPECT_EQ(a.data[2], b.data[2]);
}

TEST(TestFp6x4E3M2, CopyAssignment)
{
    const fp6x2_e3m2 loPair(fp6_e3m2(1.0f), fp6_e3m2(2.0f));
    const fp6x2_e3m2 hiPair(fp6_e3m2(4.0f), fp6_e3m2(8.0f));
    fp6x4_e3m2 a(loPair, hiPair);
    fp6x4_e3m2 b;
    b = a;

    EXPECT_EQ(a.data[0], b.data[0]);
    EXPECT_EQ(a.data[1], b.data[1]);
    EXPECT_EQ(a.data[2], b.data[2]);

    auto extractedLo = b.lo_pair();
    auto extractedHi = b.hi_pair();
    EXPECT_EQ(static_cast<float>(extractedLo.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(extractedLo.hi()), 2.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.lo()), 4.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.hi()), 8.0f);
}
