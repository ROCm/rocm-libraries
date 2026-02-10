// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/All.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

using hipdnn_data_sdk::types::fp8_e5m2;
using namespace hipdnn_data_sdk::types;

class TestFp8E5M2 : public ::testing::Test
{
protected:
    // FP8 E5M2 has limited precision (only 2 mantissa bits), use larger tolerance
    static constexpr float K_TOLERANCE = 0.5f; // NOLINT(readability-identifier-naming)

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(fp8_e5m2 a, fp8_e5m2 b, float tol = K_TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

// ============================================================================
// Type Properties Tests
// ============================================================================

TEST_F(TestFp8E5M2, TypeProperties)
{
    EXPECT_EQ(sizeof(fp8_e5m2), 1);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e5m2>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e5m2>);
    EXPECT_TRUE(std::is_default_constructible_v<fp8_e5m2>);
    EXPECT_TRUE(std::is_copy_constructible_v<fp8_e5m2>);
    EXPECT_TRUE(std::is_move_constructible_v<fp8_e5m2>);
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(TestFp8E5M2, ConstructFromFloat)
{
    fp8_e5m2 a(1.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));

    fp8_e5m2 b(0.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 0.0f));

    fp8_e5m2 c(-2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(c), -2.0f));

    // Test a larger value (E5M2 has wider range than E4M3)
    fp8_e5m2 d(1024.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1024.0f, 128.0f));
}

TEST_F(TestFp8E5M2, ConstructFromDouble)
{
    fp8_e5m2 a(1.0);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));

    fp8_e5m2 b(2.0);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.0f));
}

TEST_F(TestFp8E5M2, ConstructFromIntegral)
{
    fp8_e5m2 a(4);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 4.0f));

    fp8_e5m2 b(-8);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -8.0f));

    fp8_e5m2 c(0u);
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 0.0f));

    fp8_e5m2 d(int64_t{16});
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 16.0f));
}

TEST_F(TestFp8E5M2, FromBits)
{
    // 1.0 in E5M2: sign=0, exp=15 (0xF), mantissa=0 -> 0x3C
    fp8_e5m2 one = fp8_e5m2::from_bits(0x3C);
    EXPECT_TRUE(nearEqual(static_cast<float>(one), 1.0f));

    // 0.0 in E5M2
    fp8_e5m2 zero = fp8_e5m2::from_bits(0x00);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);

    // Positive infinity: 0x7C
    fp8_e5m2 inf = fp8_e5m2::from_bits(0x7C);
    EXPECT_TRUE(isinf(inf));

    // NaN: 0x7F
    fp8_e5m2 nan = fp8_e5m2::from_bits(0x7F);
    EXPECT_TRUE(isnan(nan));
}

TEST_F(TestFp8E5M2, CopyConstruct)
{
    fp8_e5m2 a(2.0f);
    fp8_e5m2 b(a);
    EXPECT_EQ(a.data, b.data);
    EXPECT_EQ(static_cast<float>(a), static_cast<float>(b));
}

// ============================================================================
// Conversion Tests
// ============================================================================

TEST_F(TestFp8E5M2, ExplicitConversionToFloat)
{
    fp8_e5m2 a(1.5f);
    auto f = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(f, 1.5f));
}

TEST_F(TestFp8E5M2, ExplicitConversionToDouble)
{
    fp8_e5m2 a(2.0f);
    auto d = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 2.0f));
}

// ============================================================================
// Arithmetic Operator Tests
// ============================================================================

TEST_F(TestFp8E5M2, Addition)
{
    fp8_e5m2 a(1.0f);
    fp8_e5m2 b(2.0f);
    fp8_e5m2 c = a + b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 3.0f));
}

TEST_F(TestFp8E5M2, Subtraction)
{
    fp8_e5m2 a(4.0f);
    fp8_e5m2 b(2.0f);
    fp8_e5m2 c = a - b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 2.0f));
}

TEST_F(TestFp8E5M2, Multiplication)
{
    fp8_e5m2 a(2.0f);
    fp8_e5m2 b(4.0f);
    fp8_e5m2 c = a * b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 8.0f));
}

TEST_F(TestFp8E5M2, Division)
{
    fp8_e5m2 a(8.0f);
    fp8_e5m2 b(2.0f);
    fp8_e5m2 c = a / b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 4.0f));
}

TEST_F(TestFp8E5M2, UnaryNegation)
{
    fp8_e5m2 a(4.0f);
    fp8_e5m2 b = -a;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -4.0f));

    fp8_e5m2 c(-2.0f);
    fp8_e5m2 d = -c;
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 2.0f));
}

TEST_F(TestFp8E5M2, UnaryPlus)
{
    fp8_e5m2 a(4.0f);
    fp8_e5m2 b = +a;
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Compound Assignment Tests
// ============================================================================

TEST_F(TestFp8E5M2, CompoundAddition)
{
    fp8_e5m2 a(1.0f);
    a += fp8_e5m2(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 3.0f));
}

TEST_F(TestFp8E5M2, CompoundSubtraction)
{
    fp8_e5m2 a(4.0f);
    a -= fp8_e5m2(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 2.0f));
}

TEST_F(TestFp8E5M2, CompoundMultiplication)
{
    fp8_e5m2 a(2.0f);
    a *= fp8_e5m2(4.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 8.0f));
}

TEST_F(TestFp8E5M2, CompoundDivision)
{
    fp8_e5m2 a(8.0f);
    a /= fp8_e5m2(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 4.0f));
}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TEST_F(TestFp8E5M2, Equality)
{
    fp8_e5m2 a(1.0f);
    fp8_e5m2 b(1.0f);
    fp8_e5m2 c(2.0f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(TestFp8E5M2, Inequality)
{
    fp8_e5m2 a(1.0f);
    fp8_e5m2 b(2.0f);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TEST_F(TestFp8E5M2, LessThan)
{
    fp8_e5m2 a(1.0f);
    fp8_e5m2 b(2.0f);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TEST_F(TestFp8E5M2, GreaterThan)
{
    fp8_e5m2 a(2.0f);
    fp8_e5m2 b(1.0f);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TEST_F(TestFp8E5M2, LessThanOrEqual)
{
    fp8_e5m2 a(1.0f);
    fp8_e5m2 b(2.0f);
    fp8_e5m2 c(1.0f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(b <= a);
}

TEST_F(TestFp8E5M2, GreaterThanOrEqual)
{
    fp8_e5m2 a(2.0f);
    fp8_e5m2 b(1.0f);
    fp8_e5m2 c(2.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

// ============================================================================
// Special Values Tests
// ============================================================================

TEST_F(TestFp8E5M2, PositiveZero)
{
    fp8_e5m2 zero = fp8_e5m2::from_bits(0x00);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);
    EXPECT_FALSE(signbit(zero));
}

TEST_F(TestFp8E5M2, NegativeZero)
{
    fp8_e5m2 negZero = fp8_e5m2::from_bits(0x80);
    EXPECT_EQ(static_cast<float>(negZero), -0.0f);
    EXPECT_TRUE(signbit(negZero));
}

TEST_F(TestFp8E5M2, PositiveInfinity)
{
    fp8_e5m2 inf = fp8_e5m2::from_bits(0x7C);
    EXPECT_TRUE(isinf(inf));
    EXPECT_FALSE(signbit(inf));
    EXPECT_FALSE(isnan(inf));
}

TEST_F(TestFp8E5M2, NegativeInfinity)
{
    fp8_e5m2 negInf = fp8_e5m2::from_bits(0xFC);
    EXPECT_TRUE(isinf(negInf));
    EXPECT_TRUE(signbit(negInf));
    EXPECT_FALSE(isnan(negInf));
}

TEST_F(TestFp8E5M2, QuietNaN)
{
    fp8_e5m2 nan = fp8_e5m2::from_bits(0x7F);
    EXPECT_TRUE(isnan(nan));
    EXPECT_FALSE(isinf(nan));
}

TEST_F(TestFp8E5M2, SignalingNaN)
{
    fp8_e5m2 snan = fp8_e5m2::from_bits(0x7D);
    EXPECT_TRUE(isnan(snan));
}

TEST_F(TestFp8E5M2, IsFinite)
{
    EXPECT_TRUE(isfinite(fp8_e5m2(1.0f)));
    EXPECT_TRUE(isfinite(fp8_e5m2(0.0f)));
    EXPECT_FALSE(isfinite(fp8_e5m2::from_bits(0x7C))); // inf
    EXPECT_FALSE(isfinite(fp8_e5m2::from_bits(0x7F))); // nan
}

// ============================================================================
// Math Function Tests
// ============================================================================

TEST_F(TestFp8E5M2, Abs)
{
    EXPECT_TRUE(nearEqual(abs(fp8_e5m2(-4.0f)), fp8_e5m2(4.0f)));
    EXPECT_TRUE(nearEqual(abs(fp8_e5m2(4.0f)), fp8_e5m2(4.0f)));
    EXPECT_TRUE(nearEqual(abs(fp8_e5m2(0.0f)), fp8_e5m2(0.0f)));
}

TEST_F(TestFp8E5M2, Fabs)
{
    EXPECT_TRUE(nearEqual(fabs(fp8_e5m2(-4.0f)), fp8_e5m2(4.0f)));
    EXPECT_TRUE(nearEqual(fabs(fp8_e5m2(4.0f)), fp8_e5m2(4.0f)));
}

TEST_F(TestFp8E5M2, Max)
{
    fp8_e5m2 a(1.0f);
    fp8_e5m2 b(2.0f);
    EXPECT_TRUE(nearEqual(max(a, b), b));
    EXPECT_TRUE(nearEqual(max(b, a), b));
}

TEST_F(TestFp8E5M2, MaxWithNaN)
{
    fp8_e5m2 a(1.0f);
    fp8_e5m2 nan = fp8_e5m2::from_bits(0x7F);
    EXPECT_TRUE(nearEqual(max(a, nan), a));
    EXPECT_TRUE(nearEqual(max(nan, a), a));
    EXPECT_TRUE(isnan(max(nan, nan)));
}

TEST_F(TestFp8E5M2, Min)
{
    fp8_e5m2 a(1.0f);
    fp8_e5m2 b(2.0f);
    EXPECT_TRUE(nearEqual(min(a, b), a));
    EXPECT_TRUE(nearEqual(min(b, a), a));
}

TEST_F(TestFp8E5M2, Sqrt)
{
    fp8_e5m2 a(4.0f);
    EXPECT_TRUE(nearEqual(sqrt(a), fp8_e5m2(2.0f)));

    fp8_e5m2 b(16.0f);
    EXPECT_TRUE(nearEqual(sqrt(b), fp8_e5m2(4.0f)));
}

TEST_F(TestFp8E5M2, Exp)
{
    fp8_e5m2 a(0.0f);
    EXPECT_TRUE(nearEqual(exp(a), fp8_e5m2(1.0f)));
}

TEST_F(TestFp8E5M2, Log)
{
    fp8_e5m2 a(1.0f);
    EXPECT_TRUE(nearEqual(log(a), fp8_e5m2(0.0f)));
}

TEST_F(TestFp8E5M2, Tanh)
{
    fp8_e5m2 a(0.0f);
    EXPECT_TRUE(nearEqual(tanh(a), fp8_e5m2(0.0f)));
}

TEST_F(TestFp8E5M2, Floor)
{
    EXPECT_TRUE(nearEqual(floor(fp8_e5m2(2.5f)), fp8_e5m2(2.0f)));
    EXPECT_TRUE(nearEqual(floor(fp8_e5m2(-2.5f)), fp8_e5m2(-3.0f)));
}

TEST_F(TestFp8E5M2, Ceil)
{
    EXPECT_TRUE(nearEqual(ceil(fp8_e5m2(2.5f)), fp8_e5m2(3.0f)));
    EXPECT_TRUE(nearEqual(ceil(fp8_e5m2(-2.5f)), fp8_e5m2(-2.0f)));
}

TEST_F(TestFp8E5M2, Round)
{
    EXPECT_TRUE(nearEqual(round(fp8_e5m2(2.0f)), fp8_e5m2(2.0f)));
    EXPECT_TRUE(nearEqual(round(fp8_e5m2(3.0f)), fp8_e5m2(3.0f)));
}

// ============================================================================
// User-Defined Literal Tests
// ============================================================================

TEST_F(TestFp8E5M2, UserDefinedLiteral)
{
    fp8_e5m2 a = 1.5_bfp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.5f));

    fp8_e5m2 b = -2.0_bfp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -2.0f));
}

// ============================================================================
// Stream Output Tests
// ============================================================================

TEST_F(TestFp8E5M2, StreamOutput)
{
    fp8_e5m2 a(2.0f);
    std::ostringstream oss;
    oss << a;
    float parsed = std::stof(oss.str());
    EXPECT_TRUE(nearEqual(parsed, 2.0f));
}

// ============================================================================
// numeric_limits Tests
// ============================================================================

TEST_F(TestFp8E5M2, NumericLimitsBasic)
{
    EXPECT_TRUE(std::numeric_limits<fp8_e5m2>::is_specialized);
    EXPECT_TRUE(std::numeric_limits<fp8_e5m2>::is_signed);
    EXPECT_FALSE(std::numeric_limits<fp8_e5m2>::is_integer);
    EXPECT_TRUE(std::numeric_limits<fp8_e5m2>::has_infinity); // E5M2 has infinity
    EXPECT_TRUE(std::numeric_limits<fp8_e5m2>::has_quiet_NaN);
}

TEST_F(TestFp8E5M2, NumericLimitsInfinity)
{
    fp8_e5m2 inf = std::numeric_limits<fp8_e5m2>::infinity();
    EXPECT_TRUE(isinf(inf));
    EXPECT_FALSE(signbit(inf));
}

TEST_F(TestFp8E5M2, NumericLimitsNaN)
{
    fp8_e5m2 nan = std::numeric_limits<fp8_e5m2>::quiet_NaN();
    EXPECT_TRUE(isnan(nan));
}

TEST_F(TestFp8E5M2, NumericLimitsMax)
{
    fp8_e5m2 maxVal = std::numeric_limits<fp8_e5m2>::max();
    EXPECT_TRUE(isfinite(maxVal));
    EXPECT_GT(static_cast<float>(maxVal), 0.0f);
    // E5M2 max is 57344
    EXPECT_TRUE(nearEqual(static_cast<float>(maxVal), 57344.0f, 1000.0f));
}

TEST_F(TestFp8E5M2, NumericLimitsMin)
{
    fp8_e5m2 minVal = std::numeric_limits<fp8_e5m2>::min();
    EXPECT_TRUE(isfinite(minVal));
    EXPECT_GT(static_cast<float>(minVal), 0.0f);
}

TEST_F(TestFp8E5M2, NumericLimitsLowest)
{
    fp8_e5m2 lowestVal = std::numeric_limits<fp8_e5m2>::lowest();
    EXPECT_TRUE(isfinite(lowestVal));
    EXPECT_LT(static_cast<float>(lowestVal), 0.0f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST_F(TestFp8E5M2, NamedConstants)
{
    // Access named constants via numeric_limits
    EXPECT_TRUE(isinf(std::numeric_limits<fp8_e5m2>::infinity()));
    EXPECT_FALSE(signbit(std::numeric_limits<fp8_e5m2>::infinity()));
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e5m2>::quiet_NaN()));
    EXPECT_TRUE(isnan(std::numeric_limits<fp8_e5m2>::signaling_NaN()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::max()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::min()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::lowest()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::epsilon()));
    EXPECT_TRUE(isfinite(std::numeric_limits<fp8_e5m2>::denorm_min()));
}
