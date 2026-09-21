// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestFp4E2M1.cpp
/// @brief Type-specific tests for fp4_e2m1 and fp4x2_e2m1 (OCP MX E2M1 format).
///
/// Common tests (type properties, construction, math functions, numeric_limits) are in
/// TestPortableTypes.cpp. This file contains fp4_e2m1-specific tests for round-to-nearest-even
/// behavior, saturation, underflow, rounding edge cases, and the fp4x2_e2m1 packed storage type.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// fp4_e2m1 Type-Specific Tests
// ============================================================================

TEST(TestFp4E2M1, RoundTripAllValues)
{
    // clang-format off
    std::vector<float> const exactValues = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    // clang-format on

    for(const float val : exactValues)
    {
        const fp4_e2m1 f4(val);
        EXPECT_EQ(static_cast<float>(f4), val);
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

class TestFp4E2M1Rounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp4E2M1Rounding, Rounding)
{
    auto [input, expected] = GetParam();
    const fp4_e2m1 val(input);
    EXPECT_EQ(static_cast<float>(val), expected);
}

INSTANTIATE_TEST_SUITE_P(MidpointRounding,
                         TestFp4E2M1Rounding,
                         ::testing::Values(RoundingTestCase{0.25f, 0.0f},
                                           RoundingTestCase{0.75f, 1.0f},
                                           RoundingTestCase{1.25f, 1.0f},
                                           RoundingTestCase{2.5f, 2.0f},
                                           RoundingTestCase{5.0f, 4.0f}));

INSTANTIATE_TEST_SUITE_P(SubnormalRounding,
                         TestFp4E2M1Rounding,
                         ::testing::Values(RoundingTestCase{0.3f, 0.5f},
                                           RoundingTestCase{0.4f, 0.5f},
                                           RoundingTestCase{0.24f, 0.0f},
                                           RoundingTestCase{0.6f, 0.5f},
                                           RoundingTestCase{-0.3f, -0.5f},
                                           RoundingTestCase{-0.4f, -0.5f}));

INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp4E2M1Rounding,
                         ::testing::Values(RoundingTestCase{5.5f, 6.0f},
                                           RoundingTestCase{5.1f, 6.0f},
                                           RoundingTestCase{3.5f, 4.0f},
                                           RoundingTestCase{2.25f, 2.0f},
                                           RoundingTestCase{2.75f, 3.0f},
                                           RoundingTestCase{-5.5f, -6.0f}));

INSTANTIATE_TEST_SUITE_P(RoundingOverflow,
                         TestFp4E2M1Rounding,
                         ::testing::Values(
                             // Mantissa overflow from rounding causes exponent increment
                             RoundingTestCase{1.75f, 2.0f}, // e=1 to e=2: mant overflow 1->0, exp++
                             RoundingTestCase{-1.75f, -2.0f})); // Negative: e=1 to e=2

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp4E2M1, SaturationPositive)
{
    const fp4_e2m1 val1(7.0f);
    EXPECT_EQ(static_cast<float>(val1), 6.0f);

    const fp4_e2m1 val2(100.0f);
    EXPECT_EQ(static_cast<float>(val2), 6.0f);
}

TEST(TestFp4E2M1, SaturationNegative)
{
    const fp4_e2m1 val1(-7.0f);
    EXPECT_EQ(static_cast<float>(val1), -6.0f);

    const fp4_e2m1 val2(-100.0f);
    EXPECT_EQ(static_cast<float>(val2), -6.0f);
}

TEST(TestFp4E2M1, SaturationInfinity)
{
    const fp4_e2m1 posInf(std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(posInf), 6.0f);

    const fp4_e2m1 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(static_cast<float>(negInf), -6.0f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp4E2M1, Underflow)
{
    const fp4_e2m1 val1(0.1f);
    EXPECT_EQ(static_cast<float>(val1), 0.0f);

    const fp4_e2m1 val2(-0.1f);
    EXPECT_EQ(static_cast<float>(val2), -0.0f);
    EXPECT_TRUE(std::signbit(static_cast<float>(val2)));

    // fp32 subnormal inputs (exercises shift > 24 path with fp32Exp == 0)
    const fp4_e2m1 val3(std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val3), 0.0f);

    const fp4_e2m1 val4(-std::numeric_limits<float>::denorm_min());
    EXPECT_EQ(static_cast<float>(val4), 0.0f);

    const fp4_e2m1 val5(1e-40f);
    EXPECT_EQ(static_cast<float>(val5), 0.0f);
}

// ============================================================================
// E2M1-Specific: No NaN or Infinity (Exhaustive Check)
// ============================================================================

TEST(TestFp4E2M1, IsnanAlwaysFalse)
{
    for(uint8_t bits = 0; bits < 16; ++bits)
    {
        const fp4_e2m1 val = fp4_e2m1::from_bits(bits);
        EXPECT_FALSE(isnan(val));
    }
}

TEST(TestFp4E2M1, IsinfAlwaysFalse)
{
    for(uint8_t bits = 0; bits < 16; ++bits)
    {
        const fp4_e2m1 val = fp4_e2m1::from_bits(bits);
        EXPECT_FALSE(isinf(val));
    }
}

TEST(TestFp4E2M1, IsfiniteAlwaysTrue)
{
    for(uint8_t bits = 0; bits < 16; ++bits)
    {
        const fp4_e2m1 val = fp4_e2m1::from_bits(bits);
        EXPECT_TRUE(isfinite(val));
    }
}

TEST(TestFp4E2M1, NanConversionToZero)
{
    // Per OCP MX Spec, conversion from NaN is implementation-defined.
    // This implementation returns zero.
    const fp4_e2m1 fromNan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(static_cast<float>(fromNan), 0.0f);
}

// ============================================================================
// Specific numeric_limits Value Tests
// ============================================================================

TEST(TestFp4E2M1, NumericLimitsSpecificValues)
{
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp4_e2m1>::max()), 6.0f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp4_e2m1>::min()), 1.0f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp4_e2m1>::lowest()), -6.0f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp4_e2m1>::epsilon()), 0.5f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp4_e2m1>::round_error()), 0.5f);
    EXPECT_EQ(static_cast<float>(std::numeric_limits<fp4_e2m1>::denorm_min()), 0.5f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST(TestFp4E2M1, NamedConstants)
{
    // E2M1 has no infinity - returns max
    auto maxVal = std::numeric_limits<fp4_e2m1>::max();
    EXPECT_EQ(std::numeric_limits<fp4_e2m1>::infinity().data, maxVal.data);

    // E2M1 has no NaN - returns zero (consistent with float-to-E2M1 conversion)
    auto zeroVal = fp4_e2m1();
    EXPECT_EQ(std::numeric_limits<fp4_e2m1>::quiet_NaN().data, zeroVal.data);
    EXPECT_EQ(std::numeric_limits<fp4_e2m1>::signaling_NaN().data, zeroVal.data);
}

// ============================================================================
// fp4x2_e2m1 Packed Storage Type Tests
// ============================================================================

TEST(TestFp4x2E2M1, DefaultConstruction)
{
    const fp4x2_e2m1 val;
    EXPECT_EQ(val.data, 0x00);
    EXPECT_EQ(static_cast<float>(val.lo()), 0.0f);
    EXPECT_EQ(static_cast<float>(val.hi()), 0.0f);
}

TEST(TestFp4x2E2M1, SingleValueConstruction)
{
    const fp4_e2m1 elem(3.0f);
    const fp4x2_e2m1 packed(elem);
    EXPECT_EQ(static_cast<float>(packed.lo()), 3.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 0.0f);
}

TEST(TestFp4x2E2M1, TwoValueConstruction)
{
    const fp4_e2m1 lo(1.0f);
    const fp4_e2m1 hi(2.0f);
    const fp4x2_e2m1 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 2.0f);
}

TEST(TestFp4x2E2M1, DataLayout)
{
    // Verify storage model: lo = low nibble, hi = high nibble
    // lo = 0x02 (1.0), hi = 0x06 (4.0) -> byte = 0x62
    const fp4x2_e2m1 packed(fp4_e2m1::from_bits(0x02), fp4_e2m1::from_bits(0x06));
    EXPECT_EQ(packed.data, 0x62);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 4.0f);
}

TEST(TestFp4x2E2M1, NegativeValuesInLo)
{
    const fp4_e2m1 neg(-2.0f);
    const fp4x2_e2m1 packed(neg);
    EXPECT_EQ(static_cast<float>(packed.lo()), -2.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), 0.0f);
}

TEST(TestFp4x2E2M1, NegativeValuesInHi)
{
    const fp4_e2m1 lo(1.0f);
    const fp4_e2m1 hi(-3.0f);
    const fp4x2_e2m1 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(packed.hi()), -3.0f);
}

TEST(TestFp4x2E2M1, BothNegativeValues)
{
    const fp4_e2m1 lo(-1.5f);
    const fp4_e2m1 hi(-4.0f);
    const fp4x2_e2m1 packed(lo, hi);
    EXPECT_EQ(static_cast<float>(packed.lo()), -1.5f);
    EXPECT_EQ(static_cast<float>(packed.hi()), -4.0f);
}

TEST(TestFp4x2E2M1, CopyConstruct)
{
    const fp4x2_e2m1 a(fp4_e2m1(1.0f), fp4_e2m1(2.0f));
    const fp4x2_e2m1 b(a);
    EXPECT_EQ(a.data, b.data);
}

TEST(TestFp4x2E2M1, CopyAssignment)
{
    const fp4x2_e2m1 a(fp4_e2m1(1.0f), fp4_e2m1(2.0f));
    fp4x2_e2m1 b;
    b = a;
    EXPECT_EQ(a.data, b.data);
    EXPECT_EQ(static_cast<float>(b.lo()), 1.0f);
    EXPECT_EQ(static_cast<float>(b.hi()), 2.0f);
}
