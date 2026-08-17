// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file TestPortableTypes.cpp
/// @brief Typed tests for portable floating-point types.
///
/// This file contains two test fixtures:
/// - PortableFloatTypes: Common tests for all types
/// - MathFloatTypes: Arithmetic and math function tests for bfloat16 and half only
/// Type-specific tests that cannot be generalized remain in their individual test files.
/// Note: fp8_e8m0 is an unsigned scale format with unique behavior (no zero, no sign bit,
/// only powers of 2) - some tests use if constexpr guards to handle these differences.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

using namespace hipdnn_data_sdk::types;

// ============================================================================
// Type Traits - Provides type-specific constants and tolerances
// ============================================================================

template <typename T>
struct PortableTypeTraits;

template <>
struct PortableTypeTraits<bfloat16>
{
    static constexpr float TOLERANCE = 0.01f;
    static constexpr float LARGE_TOLERANCE = 0.1f;
    static constexpr bool HAS_INFINITY = true;
    static constexpr bool HAS_NAN = true;
    static constexpr uint16_t ONE_BITS = 0x3F80;
    static constexpr uint16_t NEG_ONE_BITS = 0xBF80;
    static constexpr uint16_t ZERO_BITS = 0x0000;
    static constexpr uint16_t NEG_ZERO_BITS = 0x8000;
    static constexpr uint16_t NAN_BITS = 0x7FC0;
    static constexpr uint16_t INF_BITS = 0x7F80;
    static constexpr uint16_t NEG_INF_BITS = 0xFF80;

    static bfloat16 fromBits(uint16_t bits)
    {
        return bfloat16::from_bits(bits);
    }
    static uint16_t toBits(bfloat16 val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<half>
{
    static constexpr float TOLERANCE = 0.001f;
    static constexpr float LARGE_TOLERANCE = 0.01f;
    static constexpr bool HAS_INFINITY = true;
    static constexpr bool HAS_NAN = true;
    static constexpr uint16_t ONE_BITS = 0x3C00;
    static constexpr uint16_t NEG_ONE_BITS = 0xBC00;
    static constexpr uint16_t ZERO_BITS = 0x0000;
    static constexpr uint16_t NEG_ZERO_BITS = 0x8000;
    static constexpr uint16_t NAN_BITS = 0x7E00;
    static constexpr uint16_t INF_BITS = 0x7C00;
    static constexpr uint16_t NEG_INF_BITS = 0xFC00;

    static half fromBits(uint16_t bits)
    {
        return half::from_bits(bits);
    }
    static uint16_t toBits(half val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp4_e2m1>
{
    // Note: fp4_e2m1 has no NaN or infinity
    static constexpr bool HAS_INFINITY = false;
    static constexpr bool HAS_NAN = false;
    static constexpr uint8_t ONE_BITS = 0x02;
    static constexpr uint8_t NEG_ONE_BITS = 0x0A;
    static constexpr uint8_t ZERO_BITS = 0x00;
    static constexpr uint8_t NEG_ZERO_BITS = 0x08;

    static fp4_e2m1 fromBits(uint8_t bits)
    {
        return fp4_e2m1::from_bits(bits);
    }
    static uint8_t toBits(fp4_e2m1 val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp6_e2m3>
{
    // Note: fp6_e2m3 has no NaN or infinity
    static constexpr bool HAS_INFINITY = false;
    static constexpr bool HAS_NAN = false;
    static constexpr uint8_t ONE_BITS = 0x08;
    static constexpr uint8_t NEG_ONE_BITS = 0x28;
    static constexpr uint8_t ZERO_BITS = 0x00;
    static constexpr uint8_t NEG_ZERO_BITS = 0x20;

    static fp6_e2m3 fromBits(uint8_t bits)
    {
        return fp6_e2m3::from_bits(bits);
    }
    static uint8_t toBits(fp6_e2m3 val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp6_e3m2>
{
    // Note: fp6_e3m2 has no NaN or infinity
    static constexpr bool HAS_INFINITY = false;
    static constexpr bool HAS_NAN = false;
    static constexpr uint8_t ONE_BITS = 0x0C;
    static constexpr uint8_t NEG_ONE_BITS = 0x2C;
    static constexpr uint8_t ZERO_BITS = 0x00;
    static constexpr uint8_t NEG_ZERO_BITS = 0x20;

    static fp6_e3m2 fromBits(uint8_t bits)
    {
        return fp6_e3m2::from_bits(bits);
    }
    static uint8_t toBits(fp6_e3m2 val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp8_e4m3>
{
    // Note: fp8_e4m3 has no infinity
    static constexpr bool HAS_INFINITY = false;
    static constexpr bool HAS_NAN = true;
    static constexpr uint8_t ONE_BITS = 0x38;
    static constexpr uint8_t NEG_ONE_BITS = 0xB8;
    static constexpr uint8_t ZERO_BITS = 0x00;
    static constexpr uint8_t NEG_ZERO_BITS = 0x80;
    static constexpr uint8_t NAN_BITS = 0x7F;

    static fp8_e4m3 fromBits(uint8_t bits)
    {
        return fp8_e4m3::from_bits(bits);
    }
    static uint8_t toBits(fp8_e4m3 val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp8_e4m3_fnuz>
{
    // Note: fp8_e4m3_fnuz has no infinity and no negative zero
    static constexpr bool HAS_INFINITY = false;
    static constexpr bool HAS_NAN = true;
    static constexpr uint8_t ONE_BITS = 0x40;
    static constexpr uint8_t NEG_ONE_BITS = 0xC0;
    static constexpr uint8_t ZERO_BITS = 0x00;
    static constexpr uint8_t NAN_BITS = 0x80;

    static fp8_e4m3_fnuz fromBits(uint8_t bits)
    {
        return fp8_e4m3_fnuz::from_bits(bits);
    }
    static uint8_t toBits(fp8_e4m3_fnuz val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp8_e5m2>
{
    static constexpr bool HAS_INFINITY = true;
    static constexpr bool HAS_NAN = true;
    static constexpr uint8_t ONE_BITS = 0x3C;
    static constexpr uint8_t NEG_ONE_BITS = 0xBC;
    static constexpr uint8_t ZERO_BITS = 0x00;
    static constexpr uint8_t NEG_ZERO_BITS = 0x80;
    static constexpr uint8_t NAN_BITS = 0x7F;
    static constexpr uint8_t INF_BITS = 0x7C;
    static constexpr uint8_t NEG_INF_BITS = 0xFC;

    static fp8_e5m2 fromBits(uint8_t bits)
    {
        return fp8_e5m2::from_bits(bits);
    }
    static uint8_t toBits(fp8_e5m2 val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp8_e5m2_fnuz>
{
    // Note: fp8_e5m2_fnuz has no infinity and no negative zero
    static constexpr bool HAS_INFINITY = false;
    static constexpr bool HAS_NAN = true;
    static constexpr uint8_t ONE_BITS = 0x40;
    static constexpr uint8_t NEG_ONE_BITS = 0xC0;
    static constexpr uint8_t ZERO_BITS = 0x00;
    static constexpr uint8_t NAN_BITS = 0x80;

    static fp8_e5m2_fnuz fromBits(uint8_t bits)
    {
        return fp8_e5m2_fnuz::from_bits(bits);
    }
    static uint8_t toBits(fp8_e5m2_fnuz val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp8_e8m0>
{
    // Note: fp8_e8m0 has no zero, negative values, or infinity
    static constexpr bool HAS_INFINITY = false;
    static constexpr bool HAS_NAN = true;
    static constexpr uint8_t ONE_BITS = 0x7F; // 2^0 = 1.0
    static constexpr uint8_t NAN_BITS = 0xFF;

    static fp8_e8m0 fromBits(uint8_t bits)
    {
        return fp8_e8m0::from_bits(bits);
    }
    static uint8_t toBits(fp8_e8m0 val)
    {
        return val.data;
    }
};

// ============================================================================
// Test Fixture (bfloat16, half, fp4_e2m1, fp8_e4m3, fp8_e5m2, fp8_e8m0)
// Common tests for all portable float types
// ============================================================================

template <typename T>
class PortableFloatTypes : public ::testing::Test
{
};

using PortableTypes = ::testing::Types<bfloat16,
                                       half,
                                       fp4_e2m1,
                                       fp6_e2m3,
                                       fp6_e3m2,
                                       fp8_e4m3,
                                       fp8_e4m3_fnuz,
                                       fp8_e5m2,
                                       fp8_e5m2_fnuz,
                                       fp8_e8m0>;
TYPED_TEST_SUITE(PortableFloatTypes, PortableTypes, );

// ============================================================================
// Math Float Types Fixture (bfloat16, half)
// Full arithmetic operations and math functions
// ============================================================================

template <typename T>
class MathFloatTypes : public ::testing::Test
{
protected:
    using Traits = PortableTypeTraits<T>;

    static bool nearEqual(float a, float b, float tol = Traits::TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(T a, T b, float tol = Traits::TOLERANCE)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

using MathTypes = ::testing::Types<bfloat16, half>;
TYPED_TEST_SUITE(MathFloatTypes, MathTypes, );

// ============================================================================
// Type Properties Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, TypeProperties)
{
    using T = TypeParam;

    // Size check - 16-bit types are 2 bytes, 8-bit types are 1 byte
    // fp4_e2m1, fp6_e2m3, fp6_e3m2 skipped - sizeof only reflects storage, not bit width
    if constexpr(std::is_same_v<T, bfloat16> || std::is_same_v<T, half>)
    {
        EXPECT_EQ(sizeof(T), 2);
    }
    else if constexpr(!std::is_same_v<T, fp4_e2m1> && !std::is_same_v<T, fp6_e2m3>
                      && !std::is_same_v<T, fp6_e3m2>)
    {
        EXPECT_EQ(sizeof(T), 1);
    }

    EXPECT_TRUE(std::is_trivially_copyable_v<T>);
    EXPECT_TRUE(std::is_standard_layout_v<T>);
    EXPECT_TRUE(std::is_default_constructible_v<T>);
    EXPECT_TRUE(std::is_copy_constructible_v<T>);
    EXPECT_TRUE(std::is_move_constructible_v<T>);
}

// ============================================================================
// Construction Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, ConstructFromFloat)
{
    using T = TypeParam;

    const T a(1.0f);
    EXPECT_EQ(static_cast<float>(a), 1.0f);

    // fp8_e8m0 zero/negative behavior tested in type-specific tests
    if constexpr(!std::is_same_v<T, fp8_e8m0>)
    {
        const T b(0.0f);
        EXPECT_EQ(static_cast<float>(b), 0.0f);

        const T c(-2.0f);
        EXPECT_EQ(static_cast<float>(c), -2.0f);
    }
}

TYPED_TEST(PortableFloatTypes, ConstructFromDouble)
{
    using T = TypeParam;

    const T a(1.0);
    EXPECT_EQ(static_cast<float>(a), 1.0f);

    const T b(2.0);
    EXPECT_EQ(static_cast<float>(b), 2.0f);
}

TYPED_TEST(PortableFloatTypes, ConstructFromIntegral)
{
    using T = TypeParam;

    const T a(4);
    EXPECT_EQ(static_cast<float>(a), 4.0f);

    // fp8_e8m0 zero/negative behavior tested in type-specific tests
    if constexpr(!std::is_same_v<T, fp8_e8m0>)
    {
        const T b(-6);
        EXPECT_EQ(static_cast<float>(b), -6.0f);

        const T c(0u);
        EXPECT_EQ(static_cast<float>(c), 0.0f);
    }
}

TYPED_TEST(PortableFloatTypes, FromBits)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T one = Traits::fromBits(Traits::ONE_BITS);
    EXPECT_EQ(static_cast<float>(one), 1.0f);

    // fp8_e8m0 has no negative values or zero
    if constexpr(!std::is_same_v<T, fp8_e8m0>)
    {
        const T negOne = Traits::fromBits(Traits::NEG_ONE_BITS);
        EXPECT_EQ(static_cast<float>(negOne), -1.0f);

        const T zero = Traits::fromBits(Traits::ZERO_BITS);
        EXPECT_EQ(static_cast<float>(zero), 0.0f);
    }

    if constexpr(Traits::HAS_NAN)
    {
        const T nan = Traits::fromBits(Traits::NAN_BITS);
        EXPECT_TRUE(isnan(nan));
    }
}

TYPED_TEST(PortableFloatTypes, CopyConstruct)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T a(2.0f);
    const T b(a);
    EXPECT_EQ(Traits::toBits(a), Traits::toBits(b));
    EXPECT_EQ(static_cast<float>(a), static_cast<float>(b));
}

// ============================================================================
// Conversion Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, ExplicitConversionToFloat)
{
    using T = TypeParam;

    // Use 2.0f which is exact for all types including fp8_e8m0 (power of 2)
    const T a(2.0f);
    auto f = static_cast<float>(a);
    EXPECT_EQ(f, 2.0f);

    // 1.5f is not exact for fp8_e8m0 (only powers of 2)
    if constexpr(!std::is_same_v<T, fp8_e8m0>)
    {
        const T b(1.5f);
        auto g = static_cast<float>(b);
        EXPECT_EQ(g, 1.5f);
    }
}

TYPED_TEST(PortableFloatTypes, ExplicitConversionToDouble)
{
    using T = TypeParam;

    const T a(2.0f);
    auto d = static_cast<double>(a);
    EXPECT_EQ(d, 2.0);
}

// ============================================================================
// Arithmetic Operator Tests
// ============================================================================

TYPED_TEST(MathFloatTypes, Addition)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(2.0f);
    const T c = a + b;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 3.0f));
}

TYPED_TEST(MathFloatTypes, Subtraction)
{
    using T = TypeParam;

    const T a(4.0f);
    const T b(2.0f);
    const T c = a - b;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 2.0f));
}

TYPED_TEST(MathFloatTypes, Multiplication)
{
    using T = TypeParam;

    const T a(2.0f);
    const T b(4.0f);
    const T c = a * b;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 8.0f));
}

TYPED_TEST(MathFloatTypes, Division)
{
    using T = TypeParam;

    const T a(8.0f);
    const T b(2.0f);
    const T c = a / b;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 4.0f));
}

TYPED_TEST(PortableFloatTypes, UnaryNegation)
{
    using T = TypeParam;

    // fp8_e8m0 does not have unary negation (unsigned type)
    if constexpr(!std::is_same_v<T, fp8_e8m0>)
    {
        const T a(4.0f);
        const T b = -a;
        EXPECT_EQ(static_cast<float>(b), -4.0f);

        const T c(-2.0f);
        const T d = -c;
        EXPECT_EQ(static_cast<float>(d), 2.0f);
    }
}

TYPED_TEST(PortableFloatTypes, UnaryPlus)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T a(4.0f);
    const T b = +a;
    EXPECT_EQ(Traits::toBits(a), Traits::toBits(b));
}

// ============================================================================
// Compound Assignment Tests
// ============================================================================

TYPED_TEST(MathFloatTypes, CompoundAddition)
{
    using T = TypeParam;

    T a(1.0f);
    a += T(2.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 3.0f));
}

TYPED_TEST(MathFloatTypes, CompoundSubtraction)
{
    using T = TypeParam;

    T a(4.0f);
    a -= T(2.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 2.0f));
}

TYPED_TEST(MathFloatTypes, CompoundMultiplication)
{
    using T = TypeParam;

    T a(2.0f);
    a *= T(4.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 8.0f));
}

TYPED_TEST(MathFloatTypes, CompoundDivision)
{
    using T = TypeParam;

    T a(8.0f);
    a /= T(2.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 4.0f));
}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TYPED_TEST(MathFloatTypes, Equality)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(1.0f);
    const T c(2.0f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TYPED_TEST(MathFloatTypes, Inequality)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(2.0f);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TYPED_TEST(MathFloatTypes, LessThan)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(2.0f);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TYPED_TEST(MathFloatTypes, GreaterThan)
{
    using T = TypeParam;

    const T a(2.0f);
    const T b(1.0f);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TYPED_TEST(MathFloatTypes, LessThanOrEqual)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(2.0f);
    const T c(1.0f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(b <= a);
}

TYPED_TEST(MathFloatTypes, GreaterThanOrEqual)
{
    using T = TypeParam;

    const T a(2.0f);
    const T b(1.0f);
    const T c(2.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

TYPED_TEST(MathFloatTypes, NanComparisonSemantics)
{
    using T = TypeParam;

    // IEEE 754 NaN comparison semantics: NaN != NaN, NaN == NaN is false
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T value(1.0f);

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

// ============================================================================
// Special Values Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, PositiveZero)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    // fp8_e8m0 has no zero representation - tested in type-specific tests
    if constexpr(!std::is_same_v<T, fp8_e8m0>)
    {
        const T zero = Traits::fromBits(Traits::ZERO_BITS);
        EXPECT_EQ(static_cast<float>(zero), 0.0f);
        EXPECT_FALSE(signbit(zero));
    }
}

TYPED_TEST(PortableFloatTypes, NegativeZero)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    // fp8_e8m0 has no negative zero (unsigned).
    // fp8_e4m3_fnuz / fp8_e5m2_fnuz have no negative zero (FNUZ format collapses both zeros to 0x00).
    if constexpr(!std::is_same_v<T, fp8_e8m0> && !std::is_same_v<T, fp8_e4m3_fnuz>
                 && !std::is_same_v<T, fp8_e5m2_fnuz>)
    {
        const T negZero = Traits::fromBits(Traits::NEG_ZERO_BITS);
        EXPECT_EQ(static_cast<float>(negZero), -0.0f);
        EXPECT_TRUE(signbit(negZero));
    }
}

TYPED_TEST(PortableFloatTypes, QuietNaN)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    if constexpr(Traits::HAS_NAN)
    {
        const T nan = Traits::fromBits(Traits::NAN_BITS);
        EXPECT_TRUE(isnan(nan));
    }
}

TYPED_TEST(PortableFloatTypes, IsFinite)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    EXPECT_TRUE(isfinite(T(1.0f)));
    EXPECT_TRUE(isfinite(T(0.0f)));

    if constexpr(Traits::HAS_NAN)
    {
        EXPECT_FALSE(isfinite(Traits::fromBits(Traits::NAN_BITS)));
    }
}

TYPED_TEST(PortableFloatTypes, InfinityHandling)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    if constexpr(Traits::HAS_INFINITY)
    {
        const T inf = Traits::fromBits(Traits::INF_BITS);
        EXPECT_TRUE(isinf(inf));
        EXPECT_FALSE(signbit(inf));
        EXPECT_FALSE(isnan(inf));

        const T negInf = Traits::fromBits(Traits::NEG_INF_BITS);
        EXPECT_TRUE(isinf(negInf));
        EXPECT_TRUE(signbit(negInf));
        EXPECT_FALSE(isnan(negInf));
    }
}

// ============================================================================
// Math Function Tests
// ============================================================================

TYPED_TEST(MathFloatTypes, Abs)
{
    using T = TypeParam;

    EXPECT_EQ(static_cast<float>(abs(T(-4.0f))), 4.0f);
    EXPECT_EQ(static_cast<float>(abs(T(4.0f))), 4.0f);
    EXPECT_EQ(static_cast<float>(abs(T(0.0f))), 0.0f);
}

TYPED_TEST(MathFloatTypes, Fabs)
{
    using T = TypeParam;

    EXPECT_EQ(static_cast<float>(fabs(T(-4.0f))), 4.0f);
    EXPECT_EQ(static_cast<float>(fabs(T(4.0f))), 4.0f);
}

TYPED_TEST(MathFloatTypes, Max)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(2.0f);
    EXPECT_EQ(max(a, b), b);
    EXPECT_EQ(max(b, a), b);
}

TYPED_TEST(MathFloatTypes, MaxNegative)
{
    using T = TypeParam;

    const T a(-1.0f);
    const T b(-2.0f);
    EXPECT_EQ(max(a, b), a);
    EXPECT_EQ(max(b, a), a);
}

TYPED_TEST(MathFloatTypes, MaxMixed)
{
    using T = TypeParam;

    const T neg(-1.0f);
    const T pos(2.0f);
    EXPECT_EQ(max(neg, pos), pos);
    EXPECT_EQ(max(pos, neg), pos);
}

TYPED_TEST(MathFloatTypes, MaxEqual)
{
    using T = TypeParam;

    const T a(2.0f);
    EXPECT_EQ(max(a, a), a);
}

TYPED_TEST(MathFloatTypes, MaxWithInfinity)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T a(1.0f);
    const T inf = Traits::fromBits(Traits::INF_BITS);
    const T negInf = Traits::fromBits(Traits::NEG_INF_BITS);

    EXPECT_EQ(max(a, inf), inf);
    EXPECT_EQ(max(inf, a), inf);
    EXPECT_EQ(max(a, negInf), a);
    EXPECT_EQ(max(negInf, a), a);
    EXPECT_EQ(max(inf, negInf), inf);
    EXPECT_EQ(max(negInf, inf), inf);
}

TYPED_TEST(MathFloatTypes, Fmax)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(2.0f);
    EXPECT_EQ(fmax(a, b), b);
    EXPECT_EQ(fmax(b, a), b);
}

TYPED_TEST(MathFloatTypes, FmaxNegative)
{
    using T = TypeParam;

    const T a(-1.0f);
    const T b(-2.0f);
    EXPECT_EQ(fmax(a, b), a);
    EXPECT_EQ(fmax(b, a), a);
}

TYPED_TEST(MathFloatTypes, FmaxMixed)
{
    using T = TypeParam;

    const T neg(-1.0f);
    const T pos(2.0f);
    EXPECT_EQ(fmax(neg, pos), pos);
    EXPECT_EQ(fmax(pos, neg), pos);
}

TYPED_TEST(MathFloatTypes, FmaxEqual)
{
    using T = TypeParam;

    const T a(2.0f);
    EXPECT_EQ(fmax(a, a), a);
}

TYPED_TEST(MathFloatTypes, FmaxWithInfinity)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T a(1.0f);
    const T inf = Traits::fromBits(Traits::INF_BITS);
    const T negInf = Traits::fromBits(Traits::NEG_INF_BITS);

    EXPECT_EQ(fmax(a, inf), inf);
    EXPECT_EQ(fmax(inf, a), inf);
    EXPECT_EQ(fmax(a, negInf), a);
    EXPECT_EQ(fmax(negInf, a), a);
    EXPECT_EQ(fmax(inf, negInf), inf);
    EXPECT_EQ(fmax(negInf, inf), inf);
}

TYPED_TEST(MathFloatTypes, FmaxWithNaN)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T a(1.0f);
    const T nan = Traits::fromBits(Traits::NAN_BITS);
    EXPECT_EQ(fmax(a, nan), a);
    EXPECT_EQ(fmax(nan, a), a);
    EXPECT_TRUE(isnan(fmax(nan, nan)));
}

TYPED_TEST(MathFloatTypes, Min)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(2.0f);
    EXPECT_EQ(min(a, b), a);
    EXPECT_EQ(min(b, a), a);
}

TYPED_TEST(MathFloatTypes, MinNegative)
{
    using T = TypeParam;

    const T a(-1.0f);
    const T b(-2.0f);
    EXPECT_EQ(min(a, b), b);
    EXPECT_EQ(min(b, a), b);
}

TYPED_TEST(MathFloatTypes, MinMixed)
{
    using T = TypeParam;

    const T neg(-1.0f);
    const T pos(2.0f);
    EXPECT_EQ(min(neg, pos), neg);
    EXPECT_EQ(min(pos, neg), neg);
}

TYPED_TEST(MathFloatTypes, MinEqual)
{
    using T = TypeParam;

    const T a(2.0f);
    EXPECT_EQ(min(a, a), a);
}

TYPED_TEST(MathFloatTypes, MinWithInfinity)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T a(1.0f);
    const T inf = Traits::fromBits(Traits::INF_BITS);
    const T negInf = Traits::fromBits(Traits::NEG_INF_BITS);

    EXPECT_EQ(min(a, inf), a);
    EXPECT_EQ(min(inf, a), a);
    EXPECT_EQ(min(a, negInf), negInf);
    EXPECT_EQ(min(negInf, a), negInf);
    EXPECT_EQ(min(inf, negInf), negInf);
    EXPECT_EQ(min(negInf, inf), negInf);
}

TYPED_TEST(MathFloatTypes, Fmin)
{
    using T = TypeParam;

    const T a(1.0f);
    const T b(2.0f);
    EXPECT_EQ(fmin(a, b), a);
    EXPECT_EQ(fmin(b, a), a);
}

TYPED_TEST(MathFloatTypes, FminNegative)
{
    using T = TypeParam;

    const T a(-1.0f);
    const T b(-2.0f);
    EXPECT_EQ(fmin(a, b), b);
    EXPECT_EQ(fmin(b, a), b);
}

TYPED_TEST(MathFloatTypes, FminMixed)
{
    using T = TypeParam;

    const T neg(-1.0f);
    const T pos(2.0f);
    EXPECT_EQ(fmin(neg, pos), neg);
    EXPECT_EQ(fmin(pos, neg), neg);
}

TYPED_TEST(MathFloatTypes, FminEqual)
{
    using T = TypeParam;

    const T a(2.0f);
    EXPECT_EQ(fmin(a, a), a);
}

TYPED_TEST(MathFloatTypes, FminWithInfinity)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T a(1.0f);
    const T inf = Traits::fromBits(Traits::INF_BITS);
    const T negInf = Traits::fromBits(Traits::NEG_INF_BITS);

    EXPECT_EQ(fmin(a, inf), a);
    EXPECT_EQ(fmin(inf, a), a);
    EXPECT_EQ(fmin(a, negInf), negInf);
    EXPECT_EQ(fmin(negInf, a), negInf);
    EXPECT_EQ(fmin(inf, negInf), negInf);
    EXPECT_EQ(fmin(negInf, inf), negInf);
}

TYPED_TEST(MathFloatTypes, FminWithNaN)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T a(1.0f);
    const T nan = Traits::fromBits(Traits::NAN_BITS);
    EXPECT_EQ(fmin(a, nan), a);
    EXPECT_EQ(fmin(nan, a), a);
    EXPECT_TRUE(isnan(fmin(nan, nan)));
}

// Zero Sign Tests (+0/-0)
// Documents current implementation behavior. The C++ standard does not require
// std::fmin/std::fmax to be sensitive to the sign of zero, and std::min/std::max
// have no special +0/-0 handling specified.
// NOTE: We compare bit patterns (not values) because +0 == -0 in IEEE 754 comparison.
TYPED_TEST(MathFloatTypes, MaxZeroSigns)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T posZero = Traits::fromBits(Traits::ZERO_BITS);
    const T negZero = Traits::fromBits(Traits::NEG_ZERO_BITS);

    // max uses `a < b ? b : a`, so returns first arg when equal
    EXPECT_EQ(Traits::toBits(max(posZero, negZero)), Traits::ZERO_BITS);
    EXPECT_EQ(Traits::toBits(max(negZero, posZero)), Traits::NEG_ZERO_BITS);
}

TYPED_TEST(MathFloatTypes, MinZeroSigns)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T posZero = Traits::fromBits(Traits::ZERO_BITS);
    const T negZero = Traits::fromBits(Traits::NEG_ZERO_BITS);

    // min uses `b < a ? b : a`, so returns first arg when equal
    EXPECT_EQ(Traits::toBits(min(posZero, negZero)), Traits::ZERO_BITS);
    EXPECT_EQ(Traits::toBits(min(negZero, posZero)), Traits::NEG_ZERO_BITS);
}

TYPED_TEST(MathFloatTypes, FmaxZeroSigns)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T posZero = Traits::fromBits(Traits::ZERO_BITS);
    const T negZero = Traits::fromBits(Traits::NEG_ZERO_BITS);

    // fmax uses `a > b ? a : b`, so returns second arg when equal
    EXPECT_EQ(Traits::toBits(fmax(posZero, negZero)), Traits::NEG_ZERO_BITS);
    EXPECT_EQ(Traits::toBits(fmax(negZero, posZero)), Traits::ZERO_BITS);
}

TYPED_TEST(MathFloatTypes, FminZeroSigns)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    const T posZero = Traits::fromBits(Traits::ZERO_BITS);
    const T negZero = Traits::fromBits(Traits::NEG_ZERO_BITS);

    // fmin uses `a < b ? a : b`, so returns second arg when equal
    EXPECT_EQ(Traits::toBits(fmin(posZero, negZero)), Traits::NEG_ZERO_BITS);
    EXPECT_EQ(Traits::toBits(fmin(negZero, posZero)), Traits::ZERO_BITS);
}

TYPED_TEST(MathFloatTypes, Sqrt)
{
    using T = TypeParam;

    const T a(4.0f);
    EXPECT_TRUE(this->nearEqual(sqrt(a), T(2.0f)));

    const T b(16.0f);
    EXPECT_TRUE(this->nearEqual(sqrt(b), T(4.0f)));
}

TYPED_TEST(MathFloatTypes, Exp)
{
    using T = TypeParam;

    const T a(0.0f);
    EXPECT_TRUE(this->nearEqual(exp(a), T(1.0f)));
}

TYPED_TEST(MathFloatTypes, Log)
{
    using T = TypeParam;

    const T a(1.0f);
    EXPECT_TRUE(this->nearEqual(log(a), T(0.0f)));
}

TYPED_TEST(MathFloatTypes, Tanh)
{
    using T = TypeParam;

    const T a(0.0f);
    EXPECT_TRUE(this->nearEqual(tanh(a), T(0.0f)));
}

TYPED_TEST(MathFloatTypes, Floor)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    EXPECT_TRUE(this->nearEqual(floor(T(2.5f)), T(2.0f), Traits::LARGE_TOLERANCE));
    EXPECT_TRUE(this->nearEqual(floor(T(-2.5f)), T(-3.0f), Traits::LARGE_TOLERANCE));
}

TYPED_TEST(MathFloatTypes, Ceil)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    EXPECT_TRUE(this->nearEqual(ceil(T(2.5f)), T(3.0f), Traits::LARGE_TOLERANCE));
    EXPECT_TRUE(this->nearEqual(ceil(T(-2.5f)), T(-2.0f), Traits::LARGE_TOLERANCE));
}

TYPED_TEST(MathFloatTypes, Round)
{
    using T = TypeParam;

    EXPECT_TRUE(this->nearEqual(round(T(2.0f)), T(2.0f)));
    EXPECT_TRUE(this->nearEqual(round(T(3.0f)), T(3.0f)));
}

// ============================================================================
// Stream Output Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, StreamOutput)
{
    using T = TypeParam;

    const T a(2.0f);
    std::ostringstream oss;
    oss << a;
    const float parsed = std::stof(oss.str());
    EXPECT_EQ(parsed, 2.0f);
}

// ============================================================================
// numeric_limits Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, NumericLimitsBasic)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    EXPECT_TRUE(std::numeric_limits<T>::is_specialized);
    EXPECT_EQ(std::numeric_limits<T>::is_signed, (!std::is_same_v<T, fp8_e8m0>));
    EXPECT_FALSE(std::numeric_limits<T>::is_integer);
    EXPECT_EQ(std::numeric_limits<T>::has_infinity, Traits::HAS_INFINITY);
    EXPECT_EQ(std::numeric_limits<T>::has_quiet_NaN, Traits::HAS_NAN);
}

TYPED_TEST(PortableFloatTypes, NumericLimitsNaN)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    if constexpr(Traits::HAS_NAN)
    {
        const T nan = std::numeric_limits<T>::quiet_NaN();
        EXPECT_TRUE(isnan(nan));
    }
}

TYPED_TEST(PortableFloatTypes, NumericLimitsMax)
{
    using T = TypeParam;

    const T maxVal = std::numeric_limits<T>::max();
    EXPECT_TRUE(isfinite(maxVal));
}

TYPED_TEST(PortableFloatTypes, NumericLimitsMin)
{
    using T = TypeParam;

    const T minVal = std::numeric_limits<T>::min();
    EXPECT_TRUE(isfinite(minVal));
    EXPECT_GT(static_cast<float>(minVal), 0.0f);
}

TYPED_TEST(PortableFloatTypes, NumericLimitsLowest)
{
    using T = TypeParam;

    const T lowestVal = std::numeric_limits<T>::lowest();
    EXPECT_TRUE(isfinite(lowestVal));

    if constexpr(!std::is_same_v<T, fp8_e8m0>)
    {
        EXPECT_LT(static_cast<float>(lowestVal), 0.0f);
    }
}

TYPED_TEST(PortableFloatTypes, NumericLimitsEpsilon)
{
    using T = TypeParam;

    const T eps = std::numeric_limits<T>::epsilon();
    EXPECT_TRUE(isfinite(eps));
    EXPECT_GT(static_cast<float>(eps), 0.0f);
}

TYPED_TEST(PortableFloatTypes, NumericLimitsInfinity)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    if constexpr(Traits::HAS_INFINITY)
    {
        const T inf = std::numeric_limits<T>::infinity();
        EXPECT_TRUE(isinf(inf));
        EXPECT_FALSE(signbit(inf));
    }
}
