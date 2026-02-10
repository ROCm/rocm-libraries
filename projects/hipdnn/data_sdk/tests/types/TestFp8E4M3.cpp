// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/All.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

using hipdnn_data_sdk::types::fp8_e4m3;
using namespace hipdnn_data_sdk::types;

class TestFp8E4M3 : public ::testing::Test
{
protected:
    // FP8 E4M3 has limited precision, use larger tolerance
    static constexpr float K_TOLERANCE = 0.2f; // NOLINT(readability-identifier-naming)

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(fp8_e4m3 a, fp8_e4m3 b, float tol = K_TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

// ============================================================================
// Type Properties Tests
// ============================================================================

TEST_F(TestFp8E4M3, TypeProperties)
{
    EXPECT_EQ(sizeof(fp8_e4m3), 1);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_default_constructible_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_copy_constructible_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_move_constructible_v<fp8_e4m3>);
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(TestFp8E4M3, ConstructFromFloat)
{
    fp8_e4m3 a(1.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));

    fp8_e4m3 b(0.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 0.0f));

    fp8_e4m3 c(-2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(c), -2.0f));

    // Test max representable value (448)
    fp8_e4m3 d(448.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 448.0f, 1.0f));
}

TEST_F(TestFp8E4M3, ConstructFromDouble)
{
    fp8_e4m3 a(1.0);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));

    fp8_e4m3 b(2.0);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.0f));
}

TEST_F(TestFp8E4M3, ConstructFromIntegral)
{
    fp8_e4m3 a(4);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 4.0f));

    fp8_e4m3 b(-8);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -8.0f));

    fp8_e4m3 c(0u);
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 0.0f));

    fp8_e4m3 d(int64_t{16});
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 16.0f));
}

TEST_F(TestFp8E4M3, FromBits)
{
    // 1.0 in E4M3: sign=0, exp=7 (0x7), mantissa=0 -> 0x38
    fp8_e4m3 one = fp8_e4m3::from_bits(0x38);
    EXPECT_TRUE(nearEqual(static_cast<float>(one), 1.0f));

    // 0.0 in E4M3
    fp8_e4m3 zero = fp8_e4m3::from_bits(0x00);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);

    // NaN in E4M3 is 0x7F
    fp8_e4m3 nan = fp8_e4m3::from_bits(0x7F);
    EXPECT_TRUE(isnan(nan));
}

TEST_F(TestFp8E4M3, CopyConstruct)
{
    fp8_e4m3 a(2.0f);
    fp8_e4m3 b(a);
    EXPECT_EQ(a.data, b.data);
    EXPECT_EQ(static_cast<float>(a), static_cast<float>(b));
}

// ============================================================================
// Conversion Tests
// ============================================================================

TEST_F(TestFp8E4M3, ExplicitConversionToFloat)
{
    fp8_e4m3 a(1.5f);
    auto f = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(f, 1.5f));
}

TEST_F(TestFp8E4M3, ExplicitConversionToDouble)
{
    fp8_e4m3 a(2.0f);
    auto d = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 2.0f));
}

// ============================================================================
// Arithmetic Operator Tests
// ============================================================================

TEST_F(TestFp8E4M3, Addition)
{
    fp8_e4m3 a(1.0f);
    fp8_e4m3 b(2.0f);
    fp8_e4m3 c = a + b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 3.0f));
}

TEST_F(TestFp8E4M3, Subtraction)
{
    fp8_e4m3 a(4.0f);
    fp8_e4m3 b(2.0f);
    fp8_e4m3 c = a - b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 2.0f));
}

TEST_F(TestFp8E4M3, Multiplication)
{
    fp8_e4m3 a(2.0f);
    fp8_e4m3 b(4.0f);
    fp8_e4m3 c = a * b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 8.0f));
}

TEST_F(TestFp8E4M3, Division)
{
    fp8_e4m3 a(8.0f);
    fp8_e4m3 b(2.0f);
    fp8_e4m3 c = a / b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 4.0f));
}

TEST_F(TestFp8E4M3, UnaryNegation)
{
    fp8_e4m3 a(4.0f);
    fp8_e4m3 b = -a;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -4.0f));

    fp8_e4m3 c(-2.0f);
    fp8_e4m3 d = -c;
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 2.0f));
}

TEST_F(TestFp8E4M3, UnaryPlus)
{
    fp8_e4m3 a(4.0f);
    fp8_e4m3 b = +a;
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Compound Assignment Tests
// ============================================================================

TEST_F(TestFp8E4M3, CompoundAddition)
{
    fp8_e4m3 a(1.0f);
    a += fp8_e4m3(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 3.0f));
}

TEST_F(TestFp8E4M3, CompoundSubtraction)
{
    fp8_e4m3 a(4.0f);
    a -= fp8_e4m3(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 2.0f));
}

TEST_F(TestFp8E4M3, CompoundMultiplication)
{
    fp8_e4m3 a(2.0f);
    a *= fp8_e4m3(4.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 8.0f));
}

TEST_F(TestFp8E4M3, CompoundDivision)
{
    fp8_e4m3 a(8.0f);
    a /= fp8_e4m3(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 4.0f));
}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TEST_F(TestFp8E4M3, Equality)
{
    fp8_e4m3 a(1.0f);
    fp8_e4m3 b(1.0f);
    fp8_e4m3 c(2.0f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(TestFp8E4M3, Inequality)
{
    fp8_e4m3 a(1.0f);
    fp8_e4m3 b(2.0f);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TEST_F(TestFp8E4M3, LessThan)
{
    fp8_e4m3 a(1.0f);
    fp8_e4m3 b(2.0f);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TEST_F(TestFp8E4M3, GreaterThan)
{
    fp8_e4m3 a(2.0f);
    fp8_e4m3 b(1.0f);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TEST_F(TestFp8E4M3, LessThanOrEqual)
{
    fp8_e4m3 a(1.0f);
    fp8_e4m3 b(2.0f);
    fp8_e4m3 c(1.0f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(b <= a);
}

TEST_F(TestFp8E4M3, GreaterThanOrEqual)
{
    fp8_e4m3 a(2.0f);
    fp8_e4m3 b(1.0f);
    fp8_e4m3 c(2.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

// ============================================================================
// Special Values Tests
// ============================================================================

TEST_F(TestFp8E4M3, PositiveZero)
{
    fp8_e4m3 zero = fp8_e4m3::from_bits(0x00);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);
    EXPECT_FALSE(signbit(zero));
}

TEST_F(TestFp8E4M3, NegativeZero)
{
    fp8_e4m3 negZero = fp8_e4m3::from_bits(0x80);
    EXPECT_EQ(static_cast<float>(negZero), -0.0f);
    EXPECT_TRUE(signbit(negZero));
}

TEST_F(TestFp8E4M3, NaN)
{
    // E4M3 has no infinity - uses NaN for 0x7F (and 0xFF for negative NaN)
    fp8_e4m3 nan = fp8_e4m3::from_bits(0x7F);
    EXPECT_TRUE(isnan(nan));
}

TEST_F(TestFp8E4M3, NoInfinity)
{
    // E4M3 OCP format has no infinity, large values saturate to max
    fp8_e4m3 nan = fp8_e4m3::from_bits(0x7F);
    EXPECT_FALSE(isinf(nan)); // 0x7F is NaN, not infinity
}

TEST_F(TestFp8E4M3, IsFinite)
{
    EXPECT_TRUE(isfinite(fp8_e4m3(1.0f)));
    EXPECT_TRUE(isfinite(fp8_e4m3(0.0f)));
    EXPECT_FALSE(isfinite(fp8_e4m3::from_bits(0x7F))); // NaN
}

// ============================================================================
// Math Function Tests
// ============================================================================

TEST_F(TestFp8E4M3, Abs)
{
    EXPECT_TRUE(nearEqual(abs(fp8_e4m3(-4.0f)), fp8_e4m3(4.0f)));
    EXPECT_TRUE(nearEqual(abs(fp8_e4m3(4.0f)), fp8_e4m3(4.0f)));
    EXPECT_TRUE(nearEqual(abs(fp8_e4m3(0.0f)), fp8_e4m3(0.0f)));
}

TEST_F(TestFp8E4M3, Fabs)
{
    EXPECT_TRUE(nearEqual(fabs(fp8_e4m3(-4.0f)), fp8_e4m3(4.0f)));
    EXPECT_TRUE(nearEqual(fabs(fp8_e4m3(4.0f)), fp8_e4m3(4.0f)));
}

TEST_F(TestFp8E4M3, Max)
{
    fp8_e4m3 a(1.0f);
    fp8_e4m3 b(2.0f);
    EXPECT_TRUE(nearEqual(max(a, b), b));
    EXPECT_TRUE(nearEqual(max(b, a), b));
}

TEST_F(TestFp8E4M3, MaxWithNaN)
{
    fp8_e4m3 a(1.0f);
    fp8_e4m3 nan = fp8_e4m3::from_bits(0x7F);
    EXPECT_TRUE(nearEqual(max(a, nan), a));
    EXPECT_TRUE(nearEqual(max(nan, a), a));
    EXPECT_TRUE(isnan(max(nan, nan)));
}

TEST_F(TestFp8E4M3, Min)
{
    fp8_e4m3 a(1.0f);
    fp8_e4m3 b(2.0f);
    EXPECT_TRUE(nearEqual(min(a, b), a));
    EXPECT_TRUE(nearEqual(min(b, a), a));
}

TEST_F(TestFp8E4M3, Sqrt)
{
    fp8_e4m3 a(4.0f);
    EXPECT_TRUE(nearEqual(sqrt(a), fp8_e4m3(2.0f)));

    fp8_e4m3 b(16.0f);
    EXPECT_TRUE(nearEqual(sqrt(b), fp8_e4m3(4.0f)));
}

TEST_F(TestFp8E4M3, Exp)
{
    fp8_e4m3 a(0.0f);
    EXPECT_TRUE(nearEqual(exp(a), fp8_e4m3(1.0f)));
}

TEST_F(TestFp8E4M3, Log)
{
    fp8_e4m3 a(1.0f);
    EXPECT_TRUE(nearEqual(log(a), fp8_e4m3(0.0f)));
}

TEST_F(TestFp8E4M3, Tanh)
{
    fp8_e4m3 a(0.0f);
    EXPECT_TRUE(nearEqual(tanh(a), fp8_e4m3(0.0f)));
}

TEST_F(TestFp8E4M3, Floor)
{
    EXPECT_TRUE(nearEqual(floor(fp8_e4m3(2.5f)), fp8_e4m3(2.0f)));
    EXPECT_TRUE(nearEqual(floor(fp8_e4m3(-2.5f)), fp8_e4m3(-3.0f)));
}

TEST_F(TestFp8E4M3, Ceil)
{
    EXPECT_TRUE(nearEqual(ceil(fp8_e4m3(2.5f)), fp8_e4m3(3.0f)));
    EXPECT_TRUE(nearEqual(ceil(fp8_e4m3(-2.5f)), fp8_e4m3(-2.0f)));
}

TEST_F(TestFp8E4M3, Round)
{
    EXPECT_TRUE(nearEqual(round(fp8_e4m3(2.0f)), fp8_e4m3(2.0f)));
    EXPECT_TRUE(nearEqual(round(fp8_e4m3(3.0f)), fp8_e4m3(3.0f)));
}

// ============================================================================
// User-Defined Literal Tests
// ============================================================================

TEST_F(TestFp8E4M3, UserDefinedLiteral)
{
    fp8_e4m3 a = 1.5_fp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.5f));

    fp8_e4m3 b = -2.0_fp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -2.0f));
}

// ============================================================================
// Stream Output Tests
// ============================================================================

TEST_F(TestFp8E4M3, StreamOutput)
{
    fp8_e4m3 a(2.0f);
    std::ostringstream oss;
    oss << a;
    float parsed = std::stof(oss.str());
    EXPECT_TRUE(nearEqual(parsed, 2.0f));
}

// ============================================================================
// numeric_limits Tests
// ============================================================================

TEST_F(TestFp8E4M3, NumericLimitsBasic)
{
    EXPECT_TRUE(std::numeric_limits<fp8_e4m3>::is_specialized);
    EXPECT_TRUE(std::numeric_limits<fp8_e4m3>::is_signed);
    EXPECT_FALSE(std::numeric_limits<fp8_e4m3>::is_integer);
    EXPECT_FALSE(std::numeric_limits<fp8_e4m3>::has_infinity); // E4M3 has no infinity
    EXPECT_TRUE(std::numeric_limits<fp8_e4m3>::has_quiet_NaN);
}

TEST_F(TestFp8E4M3, NumericLimitsNaN)
{
    fp8_e4m3 nan = std::numeric_limits<fp8_e4m3>::quiet_NaN();
    EXPECT_TRUE(isnan(nan));
}

TEST_F(TestFp8E4M3, NumericLimitsMax)
{
    fp8_e4m3 maxVal = std::numeric_limits<fp8_e4m3>::max();
    EXPECT_TRUE(isfinite(maxVal));
    EXPECT_GT(static_cast<float>(maxVal), 0.0f);
    // E4M3 max is 448
    EXPECT_TRUE(nearEqual(static_cast<float>(maxVal), 448.0f, 10.0f));
}

TEST_F(TestFp8E4M3, NumericLimitsMin)
{
    fp8_e4m3 minVal = std::numeric_limits<fp8_e4m3>::min();
    EXPECT_TRUE(isfinite(minVal));
    EXPECT_GT(static_cast<float>(minVal), 0.0f);
}

TEST_F(TestFp8E4M3, NumericLimitsLowest)
{
    fp8_e4m3 lowestVal = std::numeric_limits<fp8_e4m3>::lowest();
    EXPECT_TRUE(isfinite(lowestVal));
    EXPECT_LT(static_cast<float>(lowestVal), 0.0f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST_F(TestFp8E4M3, NamedConstants)
{
    // Access named constants via numeric_limits
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e4m3>::quiet_NaN()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::max()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::min()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::lowest()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::epsilon()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e4m3>::denorm_min()));
}

// ============================================================================
// Saturation Tests (E4M3 specific)
// ============================================================================

TEST_F(TestFp8E4M3, SaturationOnOverflow)
{
    // Values beyond 448 should saturate to max
    fp8_e4m3 large(1000.0f);
    EXPECT_TRUE(isfinite(large));
    EXPECT_TRUE(nearEqual(static_cast<float>(large), 448.0f, 10.0f));
}
