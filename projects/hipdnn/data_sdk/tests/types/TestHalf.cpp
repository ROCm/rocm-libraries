// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/All.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

using hipdnn_data_sdk::types::half;
using namespace hipdnn_data_sdk::types;

class TestHalf : public ::testing::Test
{
protected:
    static constexpr float K_TOLERANCE = 0.001f; // NOLINT(readability-identifier-naming)

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(half a, half b, float tol = K_TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

// ============================================================================
// Type Properties Tests
// ============================================================================

TEST_F(TestHalf, TypeProperties)
{
    EXPECT_EQ(sizeof(half), 2);
    EXPECT_TRUE(std::is_trivially_copyable_v<half>);
    EXPECT_TRUE(std::is_standard_layout_v<half>);
    EXPECT_TRUE(std::is_default_constructible_v<half>);
    EXPECT_TRUE(std::is_copy_constructible_v<half>);
    EXPECT_TRUE(std::is_move_constructible_v<half>);
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(TestHalf, ConstructFromFloat)
{
    half a(1.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));

    half b(0.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 0.0f));

    half c(-3.14159f);
    EXPECT_TRUE(nearEqual(static_cast<float>(c), -3.14159f, 0.002f));

    half d(1000.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1000.0f, 1.0f));
}

TEST_F(TestHalf, ConstructFromDouble)
{
    half a(1.0);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));

    half b(3.14159265358979);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 3.14159f, 0.002f));
}

TEST_F(TestHalf, ConstructFromIntegral)
{
    half a(42);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 42.0f));

    half b(-10);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -10.0f));

    half c(0u);
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 0.0f));

    half d(int64_t{1000});
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 1000.0f, 1.0f));
}

TEST_F(TestHalf, FromBits)
{
    // 1.0 in half: sign=0, exp=15 (0x0F), mantissa=0 -> 0x3C00
    half one = half::from_bits(0x3C00);
    EXPECT_EQ(static_cast<float>(one), 1.0f);

    // -1.0 in half: sign=1, exp=15 (0x0F), mantissa=0 -> 0xBC00
    half negOne = half::from_bits(0xBC00);
    EXPECT_EQ(static_cast<float>(negOne), -1.0f);

    // 0.0 in half
    half zero = half::from_bits(0x0000);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);
}

TEST_F(TestHalf, CopyConstruct)
{
    half a(2.5f);
    half b(a);
    EXPECT_EQ(a.data, b.data);
    EXPECT_EQ(static_cast<float>(a), static_cast<float>(b));
}

// ============================================================================
// Conversion Tests
// ============================================================================

TEST_F(TestHalf, ExplicitConversionToFloat)
{
    half a(1.5f);
    auto f = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(f, 1.5f));
}

TEST_F(TestHalf, ExplicitConversionToDouble)
{
    half a(2.25f);
    auto d = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 2.25f));
}

// ============================================================================
// Arithmetic Operator Tests
// ============================================================================

TEST_F(TestHalf, Addition)
{
    half a(1.0f);
    half b(2.0f);
    half c = a + b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 3.0f));
}

TEST_F(TestHalf, Subtraction)
{
    half a(5.0f);
    half b(3.0f);
    half c = a - b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 2.0f));
}

TEST_F(TestHalf, Multiplication)
{
    half a(3.0f);
    half b(4.0f);
    half c = a * b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 12.0f));
}

TEST_F(TestHalf, Division)
{
    half a(10.0f);
    half b(2.0f);
    half c = a / b;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 5.0f));
}

TEST_F(TestHalf, UnaryNegation)
{
    half a(3.5f);
    half b = -a;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -3.5f));

    half c(-2.0f);
    half d = -c;
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 2.0f));
}

TEST_F(TestHalf, UnaryPlus)
{
    half a(3.5f);
    half b = +a;
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Compound Assignment Tests
// ============================================================================

TEST_F(TestHalf, CompoundAddition)
{
    half a(1.0f);
    a += half(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 3.0f));
}

TEST_F(TestHalf, CompoundSubtraction)
{
    half a(5.0f);
    a -= half(2.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 3.0f));
}

TEST_F(TestHalf, CompoundMultiplication)
{
    half a(3.0f);
    a *= half(4.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 12.0f));
}

TEST_F(TestHalf, CompoundDivision)
{
    half a(12.0f);
    a /= half(4.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 3.0f));
}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TEST_F(TestHalf, Equality)
{
    half a(1.0f);
    half b(1.0f);
    half c(2.0f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(TestHalf, Inequality)
{
    half a(1.0f);
    half b(2.0f);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TEST_F(TestHalf, LessThan)
{
    half a(1.0f);
    half b(2.0f);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TEST_F(TestHalf, GreaterThan)
{
    half a(2.0f);
    half b(1.0f);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TEST_F(TestHalf, LessThanOrEqual)
{
    half a(1.0f);
    half b(2.0f);
    half c(1.0f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(b <= a);
}

TEST_F(TestHalf, GreaterThanOrEqual)
{
    half a(2.0f);
    half b(1.0f);
    half c(2.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

// ============================================================================
// Special Values Tests
// ============================================================================

TEST_F(TestHalf, PositiveZero)
{
    half zero = half::from_bits(0x0000);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);
    EXPECT_FALSE(signbit(zero));
}

TEST_F(TestHalf, NegativeZero)
{
    half negZero = half::from_bits(0x8000);
    EXPECT_EQ(static_cast<float>(negZero), -0.0f);
    EXPECT_TRUE(signbit(negZero));
}

TEST_F(TestHalf, PositiveInfinity)
{
    half inf = half::from_bits(0x7C00);
    EXPECT_TRUE(isinf(inf));
    EXPECT_FALSE(signbit(inf));
    EXPECT_FALSE(isnan(inf));
}

TEST_F(TestHalf, NegativeInfinity)
{
    half negInf = half::from_bits(0xFC00);
    EXPECT_TRUE(isinf(negInf));
    EXPECT_TRUE(signbit(negInf));
    EXPECT_FALSE(isnan(negInf));
}

TEST_F(TestHalf, QuietNaN)
{
    half nan = half::from_bits(0x7E00);
    EXPECT_TRUE(isnan(nan));
    EXPECT_FALSE(isinf(nan));
}

TEST_F(TestHalf, SignalingNaN)
{
    half snan = half::from_bits(0x7C01);
    EXPECT_TRUE(isnan(snan));
}

TEST_F(TestHalf, IsFinite)
{
    EXPECT_TRUE(isfinite(half(1.0f)));
    EXPECT_TRUE(isfinite(half(0.0f)));
    EXPECT_FALSE(isfinite(half::from_bits(0x7C00))); // inf
    EXPECT_FALSE(isfinite(half::from_bits(0x7E00))); // nan
}

// ============================================================================
// Math Function Tests
// ============================================================================

TEST_F(TestHalf, Abs)
{
    EXPECT_TRUE(nearEqual(abs(half(-5.0f)), half(5.0f)));
    EXPECT_TRUE(nearEqual(abs(half(5.0f)), half(5.0f)));
    EXPECT_TRUE(nearEqual(abs(half(0.0f)), half(0.0f)));
}

TEST_F(TestHalf, Fabs)
{
    EXPECT_TRUE(nearEqual(fabs(half(-5.0f)), half(5.0f)));
    EXPECT_TRUE(nearEqual(fabs(half(5.0f)), half(5.0f)));
}

TEST_F(TestHalf, Max)
{
    half a(1.0f);
    half b(2.0f);
    EXPECT_TRUE(nearEqual(max(a, b), b));
    EXPECT_TRUE(nearEqual(max(b, a), b));
}

TEST_F(TestHalf, MaxWithNaN)
{
    half a(1.0f);
    half nan = half::from_bits(0x7E00);
    EXPECT_TRUE(nearEqual(max(a, nan), a));
    EXPECT_TRUE(nearEqual(max(nan, a), a));
    EXPECT_TRUE(isnan(max(nan, nan)));
}

TEST_F(TestHalf, Min)
{
    half a(1.0f);
    half b(2.0f);
    EXPECT_TRUE(nearEqual(min(a, b), a));
    EXPECT_TRUE(nearEqual(min(b, a), a));
}

TEST_F(TestHalf, MinWithNaN)
{
    half a(1.0f);
    half nan = half::from_bits(0x7E00);
    EXPECT_TRUE(nearEqual(min(a, nan), a));
    EXPECT_TRUE(nearEqual(min(nan, a), a));
    EXPECT_TRUE(isnan(min(nan, nan)));
}

TEST_F(TestHalf, Sqrt)
{
    half a(4.0f);
    EXPECT_TRUE(nearEqual(sqrt(a), half(2.0f)));

    half b(9.0f);
    EXPECT_TRUE(nearEqual(sqrt(b), half(3.0f)));
}

TEST_F(TestHalf, Exp)
{
    half a(0.0f);
    EXPECT_TRUE(nearEqual(exp(a), half(1.0f)));

    half b(1.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(exp(b)), hipdnn_data_sdk::types::exp(1.0f), 0.01f));
}

TEST_F(TestHalf, Log)
{
    half a(1.0f);
    EXPECT_TRUE(nearEqual(log(a), half(0.0f)));

    auto e = half(hipdnn_data_sdk::types::exp(1.0f));
    EXPECT_TRUE(nearEqual(static_cast<float>(log(e)), 1.0f, 0.01f));
}

TEST_F(TestHalf, Pow)
{
    half base(2.0f);
    half exponent(3.0f);
    EXPECT_TRUE(nearEqual(pow(base, exponent), half(8.0f)));
}

TEST_F(TestHalf, Tanh)
{
    half a(0.0f);
    EXPECT_TRUE(nearEqual(tanh(a), half(0.0f)));

    half b(1.0f);
    EXPECT_TRUE(nearEqual(static_cast<float>(tanh(b)), hipdnn_data_sdk::types::tanh(1.0f), 0.01f));
}

TEST_F(TestHalf, Floor)
{
    EXPECT_TRUE(nearEqual(floor(half(2.7f)), half(2.0f)));
    EXPECT_TRUE(nearEqual(floor(half(-2.3f)), half(-3.0f)));
}

TEST_F(TestHalf, Ceil)
{
    EXPECT_TRUE(nearEqual(ceil(half(2.3f)), half(3.0f)));
    EXPECT_TRUE(nearEqual(ceil(half(-2.7f)), half(-2.0f)));
}

TEST_F(TestHalf, Round)
{
    EXPECT_TRUE(nearEqual(round(half(2.3f)), half(2.0f)));
    EXPECT_TRUE(nearEqual(round(half(2.7f)), half(3.0f)));
}

TEST_F(TestHalf, Copysign)
{
    half a(3.0f);
    half b(-1.0f);
    EXPECT_TRUE(nearEqual(copysign(a, b), half(-3.0f)));
    EXPECT_TRUE(nearEqual(copysign(b, a), half(1.0f)));
}

TEST_F(TestHalf, Sin)
{
    half a(0.0f);
    EXPECT_TRUE(nearEqual(sin(a), half(0.0f)));
}

TEST_F(TestHalf, Cos)
{
    half a(0.0f);
    EXPECT_TRUE(nearEqual(cos(a), half(1.0f)));
}

TEST_F(TestHalf, Fma)
{
    half a(2.0f);
    half b(3.0f);
    half c(1.0f);
    EXPECT_TRUE(nearEqual(fma(a, b, c), half(7.0f)));
}

// ============================================================================
// User-Defined Literal Tests
// ============================================================================

TEST_F(TestHalf, UserDefinedLiteral)
{
    half a = 1.5_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.5f));

    half b = -3.14_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), -3.14f, 0.01f));
}

// ============================================================================
// Stream Output Tests
// ============================================================================

TEST_F(TestHalf, StreamOutput)
{
    half a(2.5f);
    std::ostringstream oss;
    oss << a;
    float parsed = std::stof(oss.str());
    EXPECT_TRUE(nearEqual(parsed, 2.5f));
}

// ============================================================================
// numeric_limits Tests
// ============================================================================

TEST_F(TestHalf, NumericLimitsBasic)
{
    EXPECT_TRUE(std::numeric_limits<half>::is_specialized);
    EXPECT_TRUE(std::numeric_limits<half>::is_signed);
    EXPECT_FALSE(std::numeric_limits<half>::is_integer);
    EXPECT_TRUE(std::numeric_limits<half>::has_infinity);
    EXPECT_TRUE(std::numeric_limits<half>::has_quiet_NaN);
}

TEST_F(TestHalf, NumericLimitsInfinity)
{
    half inf = std::numeric_limits<half>::infinity();
    EXPECT_TRUE(isinf(inf));
    EXPECT_FALSE(signbit(inf));
}

TEST_F(TestHalf, NumericLimitsNaN)
{
    half nan = std::numeric_limits<half>::quiet_NaN();
    EXPECT_TRUE(isnan(nan));
}

TEST_F(TestHalf, NumericLimitsMax)
{
    half maxVal = std::numeric_limits<half>::max();
    EXPECT_TRUE(isfinite(maxVal));
    EXPECT_GT(static_cast<float>(maxVal), 0.0f);
    // half max is approximately 65504
    EXPECT_TRUE(nearEqual(static_cast<float>(maxVal), 65504.0f, 10.0f));
}

TEST_F(TestHalf, NumericLimitsMin)
{
    half minVal = std::numeric_limits<half>::min();
    EXPECT_TRUE(isfinite(minVal));
    EXPECT_GT(static_cast<float>(minVal), 0.0f);
}

TEST_F(TestHalf, NumericLimitsLowest)
{
    half lowestVal = std::numeric_limits<half>::lowest();
    EXPECT_TRUE(isfinite(lowestVal));
    EXPECT_LT(static_cast<float>(lowestVal), 0.0f);
}

// ============================================================================
// Named Constants Tests (via std::numeric_limits)
// ============================================================================

TEST_F(TestHalf, NamedConstants)
{
    // Access named constants via numeric_limits
    EXPECT_TRUE(isinf(std::numeric_limits<half>::infinity()));
    EXPECT_FALSE(signbit(std::numeric_limits<half>::infinity()));
    EXPECT_TRUE(isnan(std::numeric_limits<half>::quiet_NaN()));
    EXPECT_TRUE(isnan(std::numeric_limits<half>::signaling_NaN()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::max()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::min()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::lowest()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::epsilon()));
    EXPECT_TRUE(isfinite(std::numeric_limits<half>::denorm_min()));
}
