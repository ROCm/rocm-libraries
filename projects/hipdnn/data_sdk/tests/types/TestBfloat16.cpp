// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

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

TEST_F(TestBfloat16, NanComparisonSemantics)
{
    // IEEE 754 NaN comparison semantics: NaN != NaN, NaN == NaN is false
    bfloat16 nan = std::numeric_limits<bfloat16>::quiet_NaN();
    bfloat16 value(1.0f);

    // NaN is not equal to itself
    EXPECT_FALSE(nan == nan);
    EXPECT_TRUE(nan != nan);

    // NaN is not equal to any value
    EXPECT_FALSE(nan == value);
    EXPECT_TRUE(nan != value);

    // NaN comparisons are always false
    EXPECT_FALSE(nan < value);
    EXPECT_FALSE(nan > value);
    EXPECT_FALSE(nan <= value);
    EXPECT_FALSE(nan >= value);
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
    // bfloat16 max is 0x7F7F = (2 - 2^-7) * 2^127 ≈ 3.3895e+38
    // This is the same exponent range as float32 but with reduced mantissa precision
    auto maxFloat = static_cast<float>(maxVal);
    EXPECT_GT(maxFloat, 3.3e38f);
    EXPECT_LT(maxFloat, std::numeric_limits<float>::max());
    // Verify bit pattern
    EXPECT_EQ(maxVal.data, 0x7F7F);
}

TEST_F(TestBfloat16, NumericLimitsMin)
{
    bfloat16 minVal = std::numeric_limits<bfloat16>::min();
    EXPECT_TRUE(isfinite(minVal));
    // bfloat16 min (smallest positive normal) is 0x0080 = 2^-126 ≈ 1.175e-38
    auto minFloat = static_cast<float>(minVal);
    EXPECT_GT(minFloat, 1.1e-38f);
    EXPECT_LT(minFloat, 1.2e-38f);
    // Verify bit pattern
    EXPECT_EQ(minVal.data, 0x0080);
}

TEST_F(TestBfloat16, NumericLimitsLowest)
{
    bfloat16 lowestVal = std::numeric_limits<bfloat16>::lowest();
    EXPECT_TRUE(isfinite(lowestVal));
    // bfloat16 lowest is -max = 0xFF7F ≈ -3.3895e+38
    auto lowestFloat = static_cast<float>(lowestVal);
    EXPECT_LT(lowestFloat, -3.3e38f);
    EXPECT_GT(lowestFloat, -std::numeric_limits<float>::max());
    // Verify bit pattern
    EXPECT_EQ(lowestVal.data, 0xFF7F);
}

TEST_F(TestBfloat16, NumericLimitsEpsilon)
{
    bfloat16 eps = std::numeric_limits<bfloat16>::epsilon();
    EXPECT_TRUE(isfinite(eps));
    // bfloat16 epsilon is 2^-7 = 0.0078125
    auto epsFloat = static_cast<float>(eps);
    EXPECT_TRUE(nearEqual(epsFloat, 0.0078125f, 0.0001f));
    // Verify bit pattern (2^-7: exp = 127 - 7 = 120 = 0x78, mant = 0 -> 0x3C00)
    EXPECT_EQ(eps.data, 0x3C00);
}

// ============================================================================
// Round-to-Nearest-Even Tests
// ============================================================================

TEST_F(TestBfloat16, RoundToNearestEvenRoundDown)
{
    // Test value that should round down (remainder < 0.5 in the truncated bits)
    // 1.0 in float32: 0x3F800000
    // Adding a small value that's less than half the bfloat16 LSB
    // bfloat16 1.0 = 0x3F80, next value = 0x3F81
    // Half the distance = 0x00008000 in the lower 16 bits
    // Value with 0x00004000 in lower bits should round down
    uint32_t floatBits = 0x3F804000; // 1.0 + small amount (rounds down)
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    bfloat16 bf(f);
    EXPECT_EQ(bf.data, 0x3F80); // Should round down to 1.0
}

TEST_F(TestBfloat16, RoundToNearestEvenRoundUp)
{
    // Test value that should round up (remainder > 0.5 in the truncated bits)
    // Value with 0x0000C000 in lower bits should round up
    uint32_t floatBits = 0x3F80C000; // 1.0 + larger amount (rounds up)
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    bfloat16 bf(f);
    EXPECT_EQ(bf.data, 0x3F81); // Should round up
}

TEST_F(TestBfloat16, RoundToNearestEvenTieToEven)
{
    // Test tie-breaking: exactly 0.5 between two values should round to even
    // Value ending in ...0 with exactly 0x8000 remainder should stay (already even)
    uint32_t floatBitsEven = 0x3F808000; // Tie, result LSB is 0 -> stays at 0x3F80
    float fEven;
    std::memcpy(&fEven, &floatBitsEven, sizeof(float));
    bfloat16 bfEven(fEven);
    EXPECT_EQ(bfEven.data, 0x3F80); // Tie breaks to even (LSB = 0)

    // Value ending in ...1 with exactly 0x8000 remainder should round up
    uint32_t floatBitsOdd = 0x3F818000; // Tie, result LSB would be 1 -> rounds up to even
    float fOdd;
    std::memcpy(&fOdd, &floatBitsOdd, sizeof(float));
    bfloat16 bfOdd(fOdd);
    EXPECT_EQ(bfOdd.data, 0x3F82); // Tie breaks to even (rounds up)
}

TEST_F(TestBfloat16, TruncationVsRNE)
{
    // Test that truncation and RNE produce different results for appropriate inputs
    // Value with 0x8000 remainder where LSB is 1: RNE rounds up, truncation doesn't
    uint32_t floatBits = 0x3F818000; // LSB=1, remainder=0x8000
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));

    // RNE should round up (tie-break to even)
    uint16_t rneResult = hipdnn_data_sdk::types::detail::float_to_bfloat16_bits_rne(f);
    EXPECT_EQ(rneResult, 0x3F82);

    // Truncation should not round
    uint16_t truncResult = hipdnn_data_sdk::types::detail::float_to_bfloat16_bits_truncate(f);
    EXPECT_EQ(truncResult, 0x3F81);

    // Default should match RNE
    uint16_t defaultResult = hipdnn_data_sdk::types::detail::float_to_bfloat16_bits(f);
    EXPECT_EQ(defaultResult, rneResult);
}

// ============================================================================
// bfloat16_truncate Type Tests
// ============================================================================

TEST_F(TestBfloat16, Bfloat16TruncateType)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;
    using hipdnn_data_sdk::types::Bfloat16RoundingMode;

    // Verify type properties
    EXPECT_EQ(sizeof(bfloat16_truncate), 2);
    EXPECT_TRUE(std::is_trivially_copyable_v<bfloat16_truncate>);
    EXPECT_TRUE(std::is_standard_layout_v<bfloat16_truncate>);

    // Verify rounding mode is set correctly
    EXPECT_EQ(bfloat16_truncate::rounding_mode, Bfloat16RoundingMode::Truncate);
    EXPECT_EQ(bfloat16::rounding_mode, Bfloat16RoundingMode::RNE);
}

TEST_F(TestBfloat16, Bfloat16TruncateRounding)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;

    // Value with 0x8000 remainder where LSB is 1: RNE rounds up, truncation doesn't
    uint32_t floatBits = 0x3F818000;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));

    // RNE type should round up
    bfloat16 rneVal(f);
    EXPECT_EQ(rneVal.data, 0x3F82);

    // Truncate type should not round
    bfloat16_truncate truncVal(f);
    EXPECT_EQ(truncVal.data, 0x3F81);
}

TEST_F(TestBfloat16, Bfloat16TruncateInterop)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;

    // Test implicit conversion between rounding modes
    bfloat16 rneVal = bfloat16::from_bits(0x4000); // 2.0
    bfloat16_truncate truncVal = rneVal; // Implicit conversion
    EXPECT_EQ(truncVal.data, rneVal.data);

    // Convert back
    bfloat16 backToRne = truncVal;
    EXPECT_EQ(backToRne.data, rneVal.data);

    // Both should convert to the same float
    EXPECT_EQ(static_cast<float>(rneVal), static_cast<float>(truncVal));
}

TEST_F(TestBfloat16, Bfloat16TruncateMathFunctions)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;

    bfloat16_truncate a(-5.0f);
    bfloat16_truncate b(4.0f);

    // Math functions should work with truncate type
    EXPECT_TRUE(nearEqual(static_cast<float>(abs(a)), 5.0f));
    EXPECT_TRUE(nearEqual(static_cast<float>(sqrt(b)), 2.0f));
    EXPECT_FALSE(isnan(a));
    EXPECT_TRUE(isfinite(a));
}

TEST_F(TestBfloat16, Bfloat16TruncateNumericLimits)
{
    using hipdnn_data_sdk::types::bfloat16_truncate;

    // numeric_limits should work for truncate type too
    EXPECT_TRUE(std::numeric_limits<bfloat16_truncate>::is_specialized);
    EXPECT_EQ(std::numeric_limits<bfloat16_truncate>::max().data, 0x7F7F);
    EXPECT_EQ(std::numeric_limits<bfloat16_truncate>::min().data, 0x0080);
    EXPECT_EQ(std::numeric_limits<bfloat16_truncate>::lowest().data, 0xFF7F);
}
