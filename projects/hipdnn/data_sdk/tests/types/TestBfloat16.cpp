// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/All.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

using hipdnn_data_sdk::types::bfloat16;
using namespace hipdnn_data_sdk::types;

class TestBfloat16 : public ::testing::Test
{
protected:
    static constexpr float K_TOLERANCE = 0.01f; // NOLINT(readability-identifier-naming)

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(bfloat16 a, bfloat16 b, float tol = K_TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

// ============================================================================
// Type Properties Tests
// ============================================================================

TEST_F(TestBfloat16, TypeProperties)
{
    EXPECT_EQ(sizeof(bfloat16), 2);
    EXPECT_TRUE(std::is_trivially_copyable_v<bfloat16>);
    EXPECT_TRUE(std::is_standard_layout_v<bfloat16>);
    EXPECT_TRUE(std::is_default_constructible_v<bfloat16>);
    EXPECT_TRUE(std::is_copy_constructible_v<bfloat16>);
    EXPECT_TRUE(std::is_move_constructible_v<bfloat16>);
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(TestBfloat16, ConstructFromFloat)
{
    bfloat16 a(1.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));

    bfloat16 b(0.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 0.0f));

    bfloat16 c(-3.14159f);
    EXPECT_TRUE(nearEqual(static_cast<float>(c), -3.14159f, 0.02f));

    bfloat16 d(1e10f);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1e10f, 1e8f));
}

TEST_F(TestBfloat16, ConstructFromDouble)
{
    bfloat16 a(1.0);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));

    bfloat16 b(3.14159265358979);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 3.14159f, 0.02f));
}

TEST_F(TestBfloat16, ConstructFromIntegral)
{
    bfloat16 a(42);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 42.0f));

    bfloat16 b(-10);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -10.0f));

    bfloat16 c(0u);
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 0.0f));

    bfloat16 d(int64_t{1000});
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1000.0f));
}

TEST_F(TestBfloat16, FromBits)
{
    // 1.0 in bfloat16: sign=0, exp=127 (0x7F), mantissa=0 -> 0x3F80
    bfloat16 one = bfloat16::from_bits(0x3F80);
    EXPECT_EQ(static_cast<float>(one), 1.0f);

    // -1.0 in bfloat16: sign=1, exp=127 (0x7F), mantissa=0 -> 0xBF80
    bfloat16 negOne = bfloat16::from_bits(0xBF80);
    EXPECT_EQ(static_cast<float>(negOne), -1.0f);

    // 0.0 in bfloat16
    bfloat16 zero = bfloat16::from_bits(0x0000);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);
}

TEST_F(TestBfloat16, CopyConstruct)
{
    bfloat16 a(2.5f);
    bfloat16 b(a);
    EXPECT_EQ(a.data, b.data);
    EXPECT_EQ(static_cast<float>(a), static_cast<float>(b));
}

// ============================================================================
// Conversion Tests
// ============================================================================

TEST_F(TestBfloat16, ExplicitConversionToFloat)
{
    bfloat16 a(1.5f);
    auto f = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(f, 1.5f));
}

TEST_F(TestBfloat16, ExplicitConversionToDouble)
{
    bfloat16 a(2.25f);
    auto d = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 2.25f));
}

// ============================================================================
// Arithmetic Operator Tests
// ============================================================================

TEST_F(TestBfloat16, Addition)
{
    bfloat16 a(1.0f);
    bfloat16 b(2.0f);
    bfloat16 c = a + b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 3.0f));
}

TEST_F(TestBfloat16, Subtraction)
{
    bfloat16 a(5.0f);
    bfloat16 b(3.0f);
    bfloat16 c = a - b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 2.0f));
}

TEST_F(TestBfloat16, Multiplication)
{
    bfloat16 a(3.0f);
    bfloat16 b(4.0f);
    bfloat16 c = a * b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 12.0f));
}

TEST_F(TestBfloat16, Division)
{
    bfloat16 a(10.0f);
    bfloat16 b(2.0f);
    bfloat16 c = a / b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 5.0f));
}

TEST_F(TestBfloat16, UnaryNegation)
{
    bfloat16 a(3.5f);
    bfloat16 b = -a;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -3.5f));

    bfloat16 c(-2.0f);
    bfloat16 d = -c;
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 2.0f));
}

TEST_F(TestBfloat16, UnaryPlus)
{
    bfloat16 a(3.5f);
    bfloat16 b = +a;
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Compound Assignment Tests
// ============================================================================

TEST_F(TestBfloat16, CompoundAddition)
{
    bfloat16 a(1.0f);
    a += bfloat16(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 3.0f));
}

TEST_F(TestBfloat16, CompoundSubtraction)
{
    bfloat16 a(5.0f);
    a -= bfloat16(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 3.0f));
}

TEST_F(TestBfloat16, CompoundMultiplication)
{
    bfloat16 a(3.0f);
    a *= bfloat16(4.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 12.0f));
}

TEST_F(TestBfloat16, CompoundDivision)
{
    bfloat16 a(12.0f);
    a /= bfloat16(4.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 3.0f));
}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TEST_F(TestBfloat16, Equality)
{
    bfloat16 a(1.0f);
    bfloat16 b(1.0f);
    bfloat16 c(2.0f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(TestBfloat16, Inequality)
{
    bfloat16 a(1.0f);
    bfloat16 b(2.0f);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TEST_F(TestBfloat16, LessThan)
{
    bfloat16 a(1.0f);
    bfloat16 b(2.0f);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TEST_F(TestBfloat16, GreaterThan)
{
    bfloat16 a(2.0f);
    bfloat16 b(1.0f);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TEST_F(TestBfloat16, LessThanOrEqual)
{
    bfloat16 a(1.0f);
    bfloat16 b(2.0f);
    bfloat16 c(1.0f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(b <= a);
}

TEST_F(TestBfloat16, GreaterThanOrEqual)
{
    bfloat16 a(2.0f);
    bfloat16 b(1.0f);
    bfloat16 c(2.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

// ============================================================================
// Special Values Tests
// ============================================================================

TEST_F(TestBfloat16, PositiveZero)
{
    bfloat16 zero = bfloat16::from_bits(0x0000);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);
    EXPECT_FALSE(signbit(zero));
}

TEST_F(TestBfloat16, NegativeZero)
{
    bfloat16 negZero = bfloat16::from_bits(0x8000);
    EXPECT_EQ(static_cast<float>(negZero), -0.0f);
    EXPECT_TRUE(signbit(negZero));
}

TEST_F(TestBfloat16, PositiveInfinity)
{
    bfloat16 inf = bfloat16::from_bits(0x7F80);
    EXPECT_TRUE(isinf(inf));
    EXPECT_FALSE(signbit(inf));
    EXPECT_FALSE(isnan(inf));
}

TEST_F(TestBfloat16, NegativeInfinity)
{
    bfloat16 negInf = bfloat16::from_bits(0xFF80);
    EXPECT_TRUE(isinf(negInf));
    EXPECT_TRUE(signbit(negInf));
    EXPECT_FALSE(isnan(negInf));
}

TEST_F(TestBfloat16, QuietNaN)
{
    bfloat16 nan = bfloat16::from_bits(0x7FC0);
    EXPECT_TRUE(isnan(nan));
    EXPECT_FALSE(isinf(nan));
}

TEST_F(TestBfloat16, SignalingNaN)
{
    bfloat16 snan = bfloat16::from_bits(0x7F81);
    EXPECT_TRUE(isnan(snan));
}

TEST_F(TestBfloat16, IsFinite)
{
    EXPECT_TRUE(isfinite(bfloat16(1.0f)));
    EXPECT_TRUE(isfinite(bfloat16(0.0f)));
    EXPECT_FALSE(isfinite(bfloat16::from_bits(0x7F80))); // inf
    EXPECT_FALSE(isfinite(bfloat16::from_bits(0x7FC0))); // nan
}

// ============================================================================
// Math Function Tests
// ============================================================================

TEST_F(TestBfloat16, Abs)
{
    EXPECT_TRUE(nearEqual(abs(bfloat16(-5.0f)), bfloat16(5.0f)));
    EXPECT_TRUE(nearEqual(abs(bfloat16(5.0f)), bfloat16(5.0f)));
    EXPECT_TRUE(nearEqual(abs(bfloat16(0.0f)), bfloat16(0.0f)));
}

TEST_F(TestBfloat16, Fabs)
{
    EXPECT_TRUE(nearEqual(fabs(bfloat16(-5.0f)), bfloat16(5.0f)));
    EXPECT_TRUE(nearEqual(fabs(bfloat16(5.0f)), bfloat16(5.0f)));
}

TEST_F(TestBfloat16, Max)
{
    bfloat16 a(1.0f);
    bfloat16 b(2.0f);
    EXPECT_TRUE(nearEqual(max(a, b), b));
    EXPECT_TRUE(nearEqual(max(b, a), b));
}

TEST_F(TestBfloat16, MaxWithNaN)
{
    bfloat16 a(1.0f);
    bfloat16 nan = bfloat16::from_bits(0x7FC0);
    EXPECT_TRUE(nearEqual(max(a, nan), a));
    EXPECT_TRUE(nearEqual(max(nan, a), a));
    EXPECT_TRUE(isnan(max(nan, nan)));
}

TEST_F(TestBfloat16, Min)
{
    bfloat16 a(1.0f);
    bfloat16 b(2.0f);
    EXPECT_TRUE(nearEqual(min(a, b), a));
    EXPECT_TRUE(nearEqual(min(b, a), a));
}

TEST_F(TestBfloat16, MinWithNaN)
{
    bfloat16 a(1.0f);
    bfloat16 nan = bfloat16::from_bits(0x7FC0);
    EXPECT_TRUE(nearEqual(min(a, nan), a));
    EXPECT_TRUE(nearEqual(min(nan, a), a));
    EXPECT_TRUE(isnan(min(nan, nan)));
}

TEST_F(TestBfloat16, Sqrt)
{
    bfloat16 a(4.0f);
    EXPECT_TRUE(nearEqual(sqrt(a), bfloat16(2.0f)));

    bfloat16 b(9.0f);
    EXPECT_TRUE(nearEqual(sqrt(b), bfloat16(3.0f)));
}

TEST_F(TestBfloat16, Exp)
{
    bfloat16 a(0.0f);
    EXPECT_TRUE(nearEqual(exp(a), bfloat16(1.0f)));

    bfloat16 b(1.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(exp(b)), hipdnn_data_sdk::types::exp(1.0f), 0.1f));
}

TEST_F(TestBfloat16, Log)
{
    bfloat16 a(1.0f);
    EXPECT_TRUE(nearEqual(log(a), bfloat16(0.0f)));

    auto e = bfloat16(hipdnn_data_sdk::types::exp(1.0f));
    EXPECT_TRUE(nearEqual(static_cast<float>(log(e)), 1.0f, 0.1f));
}

TEST_F(TestBfloat16, Pow)
{
    bfloat16 base(2.0f);
    bfloat16 exp(3.0f);
    EXPECT_TRUE(nearEqual(pow(base, exp), bfloat16(8.0f)));
}

TEST_F(TestBfloat16, Tanh)
{
    bfloat16 a(0.0f);
    EXPECT_TRUE(nearEqual(tanh(a), bfloat16(0.0f)));

    bfloat16 b(1.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(tanh(b)), hipdnn_data_sdk::types::tanh(1.0f), 0.1f));
}

TEST_F(TestBfloat16, Floor)
{
    EXPECT_TRUE(nearEqual(floor(bfloat16(2.7f)), bfloat16(2.0f)));
    EXPECT_TRUE(nearEqual(floor(bfloat16(-2.3f)), bfloat16(-3.0f)));
}

TEST_F(TestBfloat16, Ceil)
{
    EXPECT_TRUE(nearEqual(ceil(bfloat16(2.3f)), bfloat16(3.0f)));
    EXPECT_TRUE(nearEqual(ceil(bfloat16(-2.7f)), bfloat16(-2.0f)));
}

TEST_F(TestBfloat16, Round)
{
    EXPECT_TRUE(nearEqual(round(bfloat16(2.3f)), bfloat16(2.0f)));
    EXPECT_TRUE(nearEqual(round(bfloat16(2.7f)), bfloat16(3.0f)));
}

TEST_F(TestBfloat16, Copysign)
{
    bfloat16 a(3.0f);
    bfloat16 b(-1.0f);
    EXPECT_TRUE(nearEqual(copysign(a, b), bfloat16(-3.0f)));
    EXPECT_TRUE(nearEqual(copysign(b, a), bfloat16(1.0f)));
}

TEST_F(TestBfloat16, Sin)
{
    bfloat16 a(0.0f);
    EXPECT_TRUE(nearEqual(sin(a), bfloat16(0.0f)));
}

TEST_F(TestBfloat16, Cos)
{
    bfloat16 a(0.0f);
    EXPECT_TRUE(nearEqual(cos(a), bfloat16(1.0f)));
}

TEST_F(TestBfloat16, Fma)
{
    bfloat16 a(2.0f);
    bfloat16 b(3.0f);
    bfloat16 c(1.0f);
    EXPECT_TRUE(nearEqual(fma(a, b, c), bfloat16(7.0f)));
}

// ============================================================================
// User-Defined Literal Tests
// ============================================================================

TEST_F(TestBfloat16, UserDefinedLiteral)
{
    bfloat16 a = 1.5_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.5f));

    bfloat16 b = -3.14_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -3.14f, 0.02f));
}

// ============================================================================
// Stream Output Tests
// ============================================================================

TEST_F(TestBfloat16, StreamOutput)
{
    bfloat16 a(2.5f);
    std::ostringstream oss;
    oss << a;
    float parsed = std::stof(oss.str());
    EXPECT_TRUE(nearEqual(parsed, 2.5f));
}

// ============================================================================
// numeric_limits Tests
// ============================================================================

TEST_F(TestBfloat16, NumericLimitsBasic)
{
    EXPECT_TRUE(std::numeric_limits<bfloat16>::is_specialized);
    EXPECT_TRUE(std::numeric_limits<bfloat16>::is_signed);
    EXPECT_FALSE(std::numeric_limits<bfloat16>::is_integer);
    EXPECT_TRUE(std::numeric_limits<bfloat16>::has_infinity);
    EXPECT_TRUE(std::numeric_limits<bfloat16>::has_quiet_NaN);
}

TEST_F(TestBfloat16, NumericLimitsInfinity)
{
    bfloat16 inf = std::numeric_limits<bfloat16>::infinity();
    EXPECT_TRUE(isinf(inf));
    EXPECT_FALSE(signbit(inf));
}

TEST_F(TestBfloat16, NumericLimitsNaN)
{
    bfloat16 nan = std::numeric_limits<bfloat16>::quiet_NaN();
    EXPECT_TRUE(isnan(nan));
}

TEST_F(TestBfloat16, NumericLimitsMax)
{
    bfloat16 maxVal = std::numeric_limits<bfloat16>::max();
    EXPECT_TRUE(isfinite(maxVal));
    EXPECT_GT(static_cast<float>(maxVal), 0.0f);
}

TEST_F(TestBfloat16, NumericLimitsMin)
{
    bfloat16 minVal = std::numeric_limits<bfloat16>::min();
    EXPECT_TRUE(isfinite(minVal));
    EXPECT_GT(static_cast<float>(minVal), 0.0f);
}

TEST_F(TestBfloat16, NumericLimitsLowest)
{
    bfloat16 lowestVal = std::numeric_limits<bfloat16>::lowest();
    EXPECT_TRUE(isfinite(lowestVal));
    EXPECT_LT(static_cast<float>(lowestVal), 0.0f);
}
