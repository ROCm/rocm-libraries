// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp6E2M3.cpp
/// @brief Type-specific tests for fp6_e2m3, fp6x2_e2m3, and fp6x4_e2m3 (OCP MX E2M3 format).
///
/// Common tests (type properties, construction, math functions, numeric_limits) are in
/// TestPortableTypes.cpp. This file contains fp6_e2m3-specific tests for round-to-nearest-even
/// behavior, saturation, underflow, rounding edge cases, and the packed storage types.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// fp6_e2m3 Type-Specific Tests
// ============================================================================

TEST(TestFp6E2M3, RoundTripAllValues)
{
    // clang-format off
    std::vector<float> exactValues = {
        // Positive subnormals and zero
        0.0f,   0.125f, 0.25f,  0.375f, 0.5f,   0.625f, 0.75f,  0.875f,
        // Positive normals
        1.0f,   1.125f, 1.25f,  1.375f, 1.5f,   1.625f, 1.75f,  1.875f,
        2.0f,   2.25f,  2.5f,   2.75f,  3.0f,   3.25f,  3.5f,   3.75f,
        4.0f,   4.5f,   5.0f,   5.5f,   6.0f,   6.5f,   7.0f,   7.5f,
        // Negative subnormals and zero
        -0.0f,  -0.125f, -0.25f, -0.375f, -0.5f, -0.625f, -0.75f, -0.875f,
        // Negative normals
        -1.0f,  -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
        -2.0f,  -2.25f,  -2.5f,  -2.75f,  -3.0f, -3.25f,  -3.5f,  -3.75f,
        -4.0f,  -4.5f,   -5.0f,  -5.5f,   -6.0f, -6.5f,   -7.0f,  -7.5f
    };
    // clang-format on

    for(float val : exactValues)
    {
        fp6_e2m3 f6(val);
        EXPECT_EQ(static_cast<float>(f6), val);
    }
}

// ============================================================================
// Rounding Tests
// ============================================================================

struct RoundingTestCase
{
    float input;
    float expected;

    friend std::ostream& operator<<(std::ostream& os, const RoundingTestCase& tc)
    {
        return os << tc.input << " -> " << tc.expected;
    }
};

class TestFp6E2M3Rounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp6E2M3Rounding, Rounding)
{
    auto [input, expected] = GetParam();
    fp6_e2m3 val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

INSTANTIATE_TEST_SUITE_P(MidpointRounding,
                         TestFp6E2M3Rounding,
                         ::testing::Values(RoundingTestCase{0.0625f, 0.0f},
                                           RoundingTestCase{0.1875f, 0.25f},
                                           RoundingTestCase{0.3125f, 0.25f},
                                           RoundingTestCase{1.0625f, 1.0f},
                                           RoundingTestCase{1.1875f, 1.25f},
                                           RoundingTestCase{2.125f, 2.0f},
                                           RoundingTestCase{2.375f, 2.5f},
                                           RoundingTestCase{4.25f, 4.0f},
                                           RoundingTestCase{4.75f, 5.0f}));

INSTANTIATE_TEST_SUITE_P(SubnormalRounding,
                         TestFp6E2M3Rounding,
                         ::testing::Values(RoundingTestCase{0.1f, 0.125f},
                                           RoundingTestCase{0.15f, 0.125f},
                                           RoundingTestCase{0.2f, 0.25f},
                                           RoundingTestCase{0.05f, 0.0f},
                                           RoundingTestCase{0.8f, 0.75f},
                                           RoundingTestCase{-0.1f, -0.125f},
                                           RoundingTestCase{-0.2f, -0.25f}));

INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp6E2M3Rounding,
                         ::testing::Values(RoundingTestCase{7.25f, 7.0f},
                                           RoundingTestCase{7.1f, 7.0f},
                                           RoundingTestCase{5.25f, 5.0f},
                                           RoundingTestCase{3.125f, 3.0f},
                                           RoundingTestCase{3.375f, 3.5f},
                                           RoundingTestCase{-7.25f, -7.0f}));

INSTANTIATE_TEST_SUITE_P(
    RoundingOverflow,
    TestFp6E2M3Rounding,
    ::testing::Values(
        // Mantissa overflow from rounding causes exponent increment
        RoundingTestCase{1.9375f, 2.0f}, // e=1 to e=2: mant overflow 111->000, exp++
        RoundingTestCase{3.875f, 4.0f}, // e=2 to e=3: mant overflow 111->000, exp++
        // Exponent overflow from rounding causes saturation to max
        RoundingTestCase{7.625f, 7.5f}, // e=3 rounds up but exp>3 saturates to max
        RoundingTestCase{-1.9375f, -2.0f}, // Negative: e=1 to e=2
        RoundingTestCase{-3.875f, -4.0f}, // Negative: e=2 to e=3
        RoundingTestCase{-7.625f, -7.5f})); // Negative: saturates to min

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp6E2M3, SaturationPositive)
{
    fp6_e2m3 val1(8.0f);
    EXPECT_EQ(static_cast<float>(val1), 7.5f);

    fp6_e2m3 val2(100.0f);
    EXPECT_EQ(static_cast<float>(val2), 7.5f);
}

TEST(TestFp6E2M3, SaturationNegative)
{
    fp6_e2m3 val1(-8.0f);
    EXPECT_EQ(static_cast<float>(val1), -7.5f);

    fp6_e2m3 val2(-100.0f);
    EXPECT_EQ(static_cast<float>(val2), -7.5f);
}

TEST(TestFp6E2M3, SaturationInfinity)
{
    fp6_e2m3 posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 7.5f);

    fp6_e2m3 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -7.5f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp6E2M3, Underflow)
{
    fp6_e2m3 val1(0.05f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    fp6_e2m3 val2(-0.05f);
    EXPECT_EQ(static_cast<float>(val2), -0.0f);
    EXPECT_TRUE(std::signbit(static_cast<float>(val2)));

    // fp32 subnormal inputs (exercises shift > 24 path with fp32Exp == 0)
    fp6_e2m3 val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    fp6_e2m3 val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    fp6_e2m3 val5(1e-40f);
    EXPECT_EQ(static_cast<float>(val5), 0.0f);
}

// ============================================================================
// E2M3-Specific: No NaN or Infinity (Exhaustive Check)
// ============================================================================

TEST(TestFp6E2M3, IsnanAlwaysFalse)
{
    for(uint8_t bits = 0; bits < 64; ++bits)
    {
        fp6_e2m3 val = fp6_e2m3::from_bits(bits);
        EXPECT_FALSE(isnan(val));
    }
}

TEST(TestFp6E2M3, IsinfAlwaysFalse)
{
    for(uint8_t bits = 0; bits < 64; ++bits)
    {
        fp6_e2m3 val = fp6_e2m3::from_bits(bits);
        EXPECT_FALSE(isinf(val));
    }
}

TEST(TestFp6E2M3, IsfiniteAlwaysTrue)
{
    for(uint8_t bits = 0; bits < 64; ++bits)
    {
        fp6_e2m3 val = fp6_e2m3::from_bits(bits);
        EXPECT_TRUE(isfinite(val));
    }
}

TEST(TestFp6E2M3, NanConversionToZero)
{
    // Per OCP MX Spec, conversion from NaN is implementation-defined.
    // This implementation returns zero.
    fp6_e2m3 fromNan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(static_cast<float>(fromNan), 0.0f);
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp6E2M3, NumericLimitsSpecificValues)
{
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e2m3>::max()), 7.5f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e2m3>::min()), 1.0f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e2m3>::lowest()), -7.5f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e2m3>::epsilon()), 0.125f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp6_e2m3>::denorm_min()), 0.125f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST(TestFp6E2M3, NamedConstants)
{
    // E2M3 has no infinity - returns max
    auto maxVal = std::numeric_limits<fp6_e2m3>::max();
    EXPECT_EQ(std::numeric_limits<fp6_e2m3>::infinity().data, maxVal.data);

    // E2M3 has no NaN - returns zero (consistent with float-to-E2M3 conversion)
    auto zeroVal = fp6_e2m3();
    EXPECT_EQ(std::numeric_limits<fp6_e2m3>::quiet_NaN().data, zeroVal.data);
    EXPECT_EQ(std::numeric_limits<fp6_e2m3>::signaling_NaN().data, zeroVal.data);
}

// ============================================================================
// fp6x2_e2m3 Packed Storage Type Tests
// ============================================================================

TEST(TestFp6x2E2M3, DefaultConstruction)
{
    fp6x2_e2m3 val;
    EXPECT_EQ(val.data, 0x0000);
    EXPECT_EQ(static_cast<float>(val.lo()), 0.0f);
    EXPECT_EQ(static_cast<float>(val.hi()), 0.0f);
}

TEST(TestFp6x2E2M3, SingleValueConstruction)
{
    fp6_e2m3 elem(3.0f);
    fp6x2_e2m3 packed(elem);
    EXPECT_EQ(static_cast<float>(packed.lo()), 3.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 0.0f);
}

TEST(TestFp6x2E2M3, TwoValueConstruction)
{
    fp6_e2m3 lo(1.0f);
    fp6_e2m3 hi(2.0f);
    fp6x2_e2m3 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 2.0f);
}

TEST(TestFp6x2E2M3, DataLayout)
{
    // Verify storage model: lo = bits [5:0], hi = bits [11:6]
    // lo = 0x08 (1.0), hi = 0x10 (2.0) -> data = 0x08 | (0x10 << 6) = 0x08 | 0x400 = 0x408
    fp6x2_e2m3 packed(fp6_e2m3::from_bits(0x08), fp6_e2m3::from_bits(0x10));
    EXPECT_EQ(packed.data, 0x0408);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 2.0f);
}

TEST(TestFp6x2E2M3, NegativeValuesInLo)
{
    fp6_e2m3 neg(-2.0f);
    fp6x2_e2m3 packed(neg);
    EXPECT_EQ(static_cast<float>(packed.lo()), -2.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 0.0f);
}

TEST(TestFp6x2E2M3, NegativeValuesInHi)
{
    fp6_e2m3 lo(1.0f);
    fp6_e2m3 hi(-3.0f);
    fp6x2_e2m3 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), -3.0f);
}

TEST(TestFp6x2E2M3, BothNegativeValues)
{
    fp6_e2m3 lo(-1.5f);
    fp6_e2m3 hi(-4.0f);
    fp6x2_e2m3 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), -1.5f);
    EXPECT_EQ(static_cast<float>(packed.hi()), -4.0f);
}

TEST(TestFp6x2E2M3, CopyConstruct)
{
    fp6x2_e2m3 a(fp6_e2m3(1.0f), fp6_e2m3(2.0f));
    fp6x2_e2m3 b(a);
    EXPECT_EQ(a.data, b.data);
}

TEST(TestFp6x2E2M3, CopyAssignment)
{
    fp6x2_e2m3 a(fp6_e2m3(1.0f), fp6_e2m3(2.0f));
    fp6x2_e2m3 b;
    b = a;
    EXPECT_EQ(a.data, b.data);
    EXPECT_EQ(static_cast<float>(b.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(b.hi()), 2.0f);
}

// ============================================================================
// fp6x4_e2m3 Packed Storage Type Tests (4 values in 3 bytes)
// ============================================================================

TEST(TestFp6x4E2M3, DefaultConstruction)
{
    fp6x4_e2m3 val;
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

TEST(TestFp6x4E2M3, TwoPairConstruction)
{
    fp6x2_e2m3 loPair(fp6_e2m3(1.0f), fp6_e2m3(2.0f));
    fp6x2_e2m3 hiPair(fp6_e2m3(3.0f), fp6_e2m3(4.0f));
    fp6x4_e2m3 packed(loPair, hiPair);

    auto extractedLo = packed.lo_pair();
    auto extractedHi = packed.hi_pair();

    EXPECT_EQ(static_cast<float>(extractedLo.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(extractedLo.hi()), 2.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.lo()), 3.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.hi()), 4.0f);
}

TEST(TestFp6x4E2M3, NegativeValues)
{
    fp6x2_e2m3 loPair(fp6_e2m3(-1.0f), fp6_e2m3(-2.0f));
    fp6x2_e2m3 hiPair(fp6_e2m3(-3.0f), fp6_e2m3(-4.0f));
    fp6x4_e2m3 packed(loPair, hiPair);

    auto extractedLo = packed.lo_pair();
    auto extractedHi = packed.hi_pair();

    EXPECT_EQ(static_cast<float>(extractedLo.lo()), -1.0f);
    EXPECT_EQ(static_cast<float>(extractedLo.hi()), -2.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.lo()), -3.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.hi()), -4.0f);
}

TEST(TestFp6x4E2M3, MixedValues)
{
    fp6x2_e2m3 loPair(fp6_e2m3(0.5f), fp6_e2m3(7.5f));
    fp6x2_e2m3 hiPair(fp6_e2m3(-0.5f), fp6_e2m3(-7.5f));
    fp6x4_e2m3 packed(loPair, hiPair);

    auto extractedLo = packed.lo_pair();
    auto extractedHi = packed.hi_pair();

    EXPECT_EQ(static_cast<float>(extractedLo.lo()), 0.5f);
    EXPECT_EQ(static_cast<float>(extractedLo.hi()), 7.5f);
    EXPECT_EQ(static_cast<float>(extractedHi.lo()), -0.5f);
    EXPECT_EQ(static_cast<float>(extractedHi.hi()), -7.5f);
}

TEST(TestFp6x4E2M3, CopyConstruct)
{
    fp6x2_e2m3 loPair(fp6_e2m3(1.0f), fp6_e2m3(2.0f));
    fp6x2_e2m3 hiPair(fp6_e2m3(3.0f), fp6_e2m3(4.0f));
    fp6x4_e2m3 a(loPair, hiPair);
    fp6x4_e2m3 b(a);

    EXPECT_EQ(a.data[0], b.data[0]);
    EXPECT_EQ(a.data[1], b.data[1]);
    EXPECT_EQ(a.data[2], b.data[2]);
}

TEST(TestFp6x4E2M3, CopyAssignment)
{
    fp6x2_e2m3 loPair(fp6_e2m3(1.0f), fp6_e2m3(2.0f));
    fp6x2_e2m3 hiPair(fp6_e2m3(3.0f), fp6_e2m3(4.0f));
    fp6x4_e2m3 a(loPair, hiPair);
    fp6x4_e2m3 b;
    b = a;

    EXPECT_EQ(a.data[0], b.data[0]);
    EXPECT_EQ(a.data[1], b.data[1]);
    EXPECT_EQ(a.data[2], b.data[2]);

    auto extractedLo = b.lo_pair();
    auto extractedHi = b.hi_pair();
    EXPECT_EQ(static_cast<float>(extractedLo.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(extractedLo.hi()), 2.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.lo()), 3.0f);
    EXPECT_EQ(static_cast<float>(extractedHi.hi()), 4.0f);
}
