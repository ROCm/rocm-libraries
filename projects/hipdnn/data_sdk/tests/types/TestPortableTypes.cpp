// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file PortableFloatTypes.cpp
/// @brief Consolidated typed tests for all portable floating-point types.
///
/// This file uses TYPED_TEST to run common test cases across all portable types:
/// bfloat16, half, fp8_e4m3, and fp8_e5m2. Type-specific tests that cannot be
/// generalized remain in their individual test files.

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

// NOLINTBEGIN(readability-identifier-naming) - traits use snake_case by convention
template <>
struct PortableTypeTraits<bfloat16>
{
    static constexpr float tolerance = 0.01f;
    static constexpr float large_tolerance = 0.1f;
    static constexpr bool has_infinity = true;
    static constexpr uint16_t one_bits = 0x3F80;
    static constexpr uint16_t neg_one_bits = 0xBF80;
    static constexpr uint16_t zero_bits = 0x0000;
    static constexpr uint16_t neg_zero_bits = 0x8000;
    static constexpr uint16_t nan_bits = 0x7FC0;
    static constexpr uint16_t inf_bits = 0x7F80;
    static constexpr uint16_t neg_inf_bits = 0xFF80;
    using bits_type = uint16_t;

    static bfloat16 from_bits(uint16_t bits)
    {
        return bfloat16::from_bits(bits);
    }
    static uint16_t to_bits(bfloat16 val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<half>
{
    static constexpr float tolerance = 0.001f;
    static constexpr float large_tolerance = 0.01f;
    static constexpr bool has_infinity = true;
    static constexpr uint16_t one_bits = 0x3C00;
    static constexpr uint16_t neg_one_bits = 0xBC00;
    static constexpr uint16_t zero_bits = 0x0000;
    static constexpr uint16_t neg_zero_bits = 0x8000;
    static constexpr uint16_t nan_bits = 0x7E00;
    static constexpr uint16_t inf_bits = 0x7C00;
    static constexpr uint16_t neg_inf_bits = 0xFC00;
    using bits_type = uint16_t;

    static half from_bits(uint16_t bits)
    {
        return half::from_bits(bits);
    }
    static uint16_t to_bits(half val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp8_e4m3>
{
    static constexpr float tolerance = 0.2f;
    static constexpr float large_tolerance = 0.5f;
    static constexpr bool has_infinity = false;
    static constexpr uint8_t one_bits = 0x38;
    static constexpr uint8_t neg_one_bits = 0xB8;
    static constexpr uint8_t zero_bits = 0x00;
    static constexpr uint8_t neg_zero_bits = 0x80;
    static constexpr uint8_t nan_bits = 0x7F;
    static constexpr uint8_t inf_bits = 0x7F; // No infinity, uses NaN
    static constexpr uint8_t neg_inf_bits = 0xFF;
    using bits_type = uint8_t;

    static fp8_e4m3 from_bits(uint8_t bits)
    {
        return fp8_e4m3::from_bits(bits);
    }
    static uint8_t to_bits(fp8_e4m3 val)
    {
        return val.data;
    }
};

template <>
struct PortableTypeTraits<fp8_e5m2>
{
    static constexpr float tolerance = 0.5f;
    static constexpr float large_tolerance = 1.0f;
    static constexpr bool has_infinity = true;
    static constexpr uint8_t one_bits = 0x3C;
    static constexpr uint8_t neg_one_bits = 0xBC;
    static constexpr uint8_t zero_bits = 0x00;
    static constexpr uint8_t neg_zero_bits = 0x80;
    static constexpr uint8_t nan_bits = 0x7F;
    static constexpr uint8_t inf_bits = 0x7C;
    static constexpr uint8_t neg_inf_bits = 0xFC;
    using bits_type = uint8_t;

    static fp8_e5m2 from_bits(uint8_t bits)
    {
        return fp8_e5m2::from_bits(bits);
    }
    static uint8_t to_bits(fp8_e5m2 val)
    {
        return val.data;
    }
};
// NOLINTEND(readability-identifier-naming)

// ============================================================================
// Test Fixture
// ============================================================================

template <typename T>
class PortableFloatTypes : public ::testing::Test
{
protected:
    using Traits = PortableTypeTraits<T>;

    static bool nearEqual(float a, float b, float tol = Traits::tolerance)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }

    static bool nearEqual(T a, T b, float tol = Traits::tolerance)
    {
        return nearEqual(static_cast<float>(a), static_cast<float>(b), tol);
    }
};

using PortableTypes = ::testing::Types<bfloat16, half, fp8_e4m3, fp8_e5m2>;
TYPED_TEST_SUITE(PortableFloatTypes, PortableTypes, );

// ============================================================================
// Type Properties Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, TypeProperties)
{
    using T = TypeParam;

    // Size check - 16-bit types are 2 bytes, 8-bit types are 1 byte
    if constexpr(std::is_same_v<T, bfloat16> || std::is_same_v<T, half>)
    {
        EXPECT_EQ(sizeof(T), 2);
    }
    else
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
    using Traits = PortableTypeTraits<T>;

    T a(1.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 1.0f));

    T b(0.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(b), 0.0f));

    T c(-2.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), -2.0f, Traits::large_tolerance));
}

TYPED_TEST(PortableFloatTypes, ConstructFromDouble)
{
    using T = TypeParam;

    T a(1.0);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 1.0f));

    T b(2.0);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(b), 2.0f));
}

TYPED_TEST(PortableFloatTypes, ConstructFromIntegral)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    T a(4);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 4.0f, Traits::large_tolerance));

    T b(-8);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(b), -8.0f, Traits::large_tolerance));

    T c(0u);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 0.0f));
}

TYPED_TEST(PortableFloatTypes, FromBits)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    T one = Traits::from_bits(Traits::one_bits);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(one), 1.0f));

    T zero = Traits::from_bits(Traits::zero_bits);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);

    T nan = Traits::from_bits(Traits::nan_bits);
    EXPECT_TRUE(isnan(nan));
}

TYPED_TEST(PortableFloatTypes, CopyConstruct)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    T a(2.0f);
    T b(a);
    EXPECT_EQ(Traits::to_bits(a), Traits::to_bits(b));
    EXPECT_EQ(static_cast<float>(a), static_cast<float>(b));
}

// ============================================================================
// Conversion Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, ExplicitConversionToFloat)
{
    using T = TypeParam;

    T a(1.5f);
    auto f = static_cast<float>(a);
    EXPECT_TRUE(this->nearEqual(f, 1.5f));
}

TYPED_TEST(PortableFloatTypes, ExplicitConversionToDouble)
{
    using T = TypeParam;

    T a(2.0f);
    auto d = static_cast<double>(a);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(d), 2.0f));
}

// ============================================================================
// Arithmetic Operator Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, Addition)
{
    using T = TypeParam;

    T a(1.0f);
    T b(2.0f);
    T c = a + b;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 3.0f));
}

TYPED_TEST(PortableFloatTypes, Subtraction)
{
    using T = TypeParam;

    T a(4.0f);
    T b(2.0f);
    T c = a - b;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 2.0f));
}

TYPED_TEST(PortableFloatTypes, Multiplication)
{
    using T = TypeParam;

    T a(2.0f);
    T b(4.0f);
    T c = a * b;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 8.0f));
}

TYPED_TEST(PortableFloatTypes, Division)
{
    using T = TypeParam;

    T a(8.0f);
    T b(2.0f);
    T c = a / b;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(c), 4.0f));
}

TYPED_TEST(PortableFloatTypes, UnaryNegation)
{
    using T = TypeParam;

    T a(4.0f);
    T b = -a;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(b), -4.0f));

    T c(-2.0f);
    T d = -c;
    EXPECT_TRUE(this->nearEqual(static_cast<float>(d), 2.0f));
}

TYPED_TEST(PortableFloatTypes, UnaryPlus)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    T a(4.0f);
    T b = +a;
    EXPECT_EQ(Traits::to_bits(a), Traits::to_bits(b));
}

// ============================================================================
// Compound Assignment Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, CompoundAddition)
{
    using T = TypeParam;

    T a(1.0f);
    a += T(2.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 3.0f));
}

TYPED_TEST(PortableFloatTypes, CompoundSubtraction)
{
    using T = TypeParam;

    T a(4.0f);
    a -= T(2.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 2.0f));
}

TYPED_TEST(PortableFloatTypes, CompoundMultiplication)
{
    using T = TypeParam;

    T a(2.0f);
    a *= T(4.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 8.0f));
}

TYPED_TEST(PortableFloatTypes, CompoundDivision)
{
    using T = TypeParam;

    T a(8.0f);
    a /= T(2.0f);
    EXPECT_TRUE(this->nearEqual(static_cast<float>(a), 4.0f));
}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, Equality)
{
    using T = TypeParam;

    T a(1.0f);
    T b(1.0f);
    T c(2.0f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TYPED_TEST(PortableFloatTypes, Inequality)
{
    using T = TypeParam;

    T a(1.0f);
    T b(2.0f);
    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != a);
}

TYPED_TEST(PortableFloatTypes, LessThan)
{
    using T = TypeParam;

    T a(1.0f);
    T b(2.0f);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TYPED_TEST(PortableFloatTypes, GreaterThan)
{
    using T = TypeParam;

    T a(2.0f);
    T b(1.0f);
    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TYPED_TEST(PortableFloatTypes, LessThanOrEqual)
{
    using T = TypeParam;

    T a(1.0f);
    T b(2.0f);
    T c(1.0f);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(b <= a);
}

TYPED_TEST(PortableFloatTypes, GreaterThanOrEqual)
{
    using T = TypeParam;

    T a(2.0f);
    T b(1.0f);
    T c(2.0f);
    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

TYPED_TEST(PortableFloatTypes, NanComparisonSemantics)
{
    using T = TypeParam;

    // IEEE 754 NaN comparison semantics: NaN != NaN, NaN == NaN is false
    T nan = std::numeric_limits<T>::quiet_NaN();
    T value(1.0f);

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

    T zero = Traits::from_bits(Traits::zero_bits);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);
    EXPECT_FALSE(signbit(zero));
}

TYPED_TEST(PortableFloatTypes, NegativeZero)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    T negZero = Traits::from_bits(Traits::neg_zero_bits);
    EXPECT_EQ(static_cast<float>(negZero), -0.0f);
    EXPECT_TRUE(signbit(negZero));
}

TYPED_TEST(PortableFloatTypes, QuietNaN)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    T nan = Traits::from_bits(Traits::nan_bits);
    EXPECT_TRUE(isnan(nan));
}

TYPED_TEST(PortableFloatTypes, IsFinite)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    EXPECT_TRUE(isfinite(T(1.0f)));
    EXPECT_TRUE(isfinite(T(0.0f)));
    EXPECT_FALSE(isfinite(Traits::from_bits(Traits::nan_bits)));
}

TYPED_TEST(PortableFloatTypes, InfinityHandling)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    if constexpr(Traits::has_infinity)
    {
        T inf = Traits::from_bits(Traits::inf_bits);
        EXPECT_TRUE(isinf(inf));
        EXPECT_FALSE(signbit(inf));
        EXPECT_FALSE(isnan(inf));

        T negInf = Traits::from_bits(Traits::neg_inf_bits);
        EXPECT_TRUE(isinf(negInf));
        EXPECT_TRUE(signbit(negInf));
        EXPECT_FALSE(isnan(negInf));
    }
    else
    {
        // For types without infinity (fp8_e4m3), verify inf bits give NaN
        T val = Traits::from_bits(Traits::inf_bits);
        EXPECT_FALSE(isinf(val));
    }
}

// ============================================================================
// Math Function Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, Abs)
{
    using T = TypeParam;

    EXPECT_TRUE(this->nearEqual(abs(T(-4.0f)), T(4.0f)));
    EXPECT_TRUE(this->nearEqual(abs(T(4.0f)), T(4.0f)));
    EXPECT_TRUE(this->nearEqual(abs(T(0.0f)), T(0.0f)));
}

TYPED_TEST(PortableFloatTypes, Fabs)
{
    using T = TypeParam;

    EXPECT_TRUE(this->nearEqual(fabs(T(-4.0f)), T(4.0f)));
    EXPECT_TRUE(this->nearEqual(fabs(T(4.0f)), T(4.0f)));
}

TYPED_TEST(PortableFloatTypes, Max)
{
    using T = TypeParam;

    T a(1.0f);
    T b(2.0f);
    EXPECT_TRUE(this->nearEqual(max(a, b), b));
    EXPECT_TRUE(this->nearEqual(max(b, a), b));
}

TYPED_TEST(PortableFloatTypes, MaxWithNaN)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    T a(1.0f);
    T nan = Traits::from_bits(Traits::nan_bits);
    EXPECT_TRUE(this->nearEqual(max(a, nan), a));
    EXPECT_TRUE(this->nearEqual(max(nan, a), a));
    EXPECT_TRUE(isnan(max(nan, nan)));
}

TYPED_TEST(PortableFloatTypes, Min)
{
    using T = TypeParam;

    T a(1.0f);
    T b(2.0f);
    EXPECT_TRUE(this->nearEqual(min(a, b), a));
    EXPECT_TRUE(this->nearEqual(min(b, a), a));
}

TYPED_TEST(PortableFloatTypes, MinWithNaN)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    T a(1.0f);
    T nan = Traits::from_bits(Traits::nan_bits);
    EXPECT_TRUE(this->nearEqual(min(a, nan), a));
    EXPECT_TRUE(this->nearEqual(min(nan, a), a));
    EXPECT_TRUE(isnan(min(nan, nan)));
}

TYPED_TEST(PortableFloatTypes, Sqrt)
{
    using T = TypeParam;

    T a(4.0f);
    EXPECT_TRUE(this->nearEqual(sqrt(a), T(2.0f)));

    T b(16.0f);
    EXPECT_TRUE(this->nearEqual(sqrt(b), T(4.0f)));
}

TYPED_TEST(PortableFloatTypes, Exp)
{
    using T = TypeParam;

    T a(0.0f);
    EXPECT_TRUE(this->nearEqual(exp(a), T(1.0f)));
}

TYPED_TEST(PortableFloatTypes, Log)
{
    using T = TypeParam;

    T a(1.0f);
    EXPECT_TRUE(this->nearEqual(log(a), T(0.0f)));
}

TYPED_TEST(PortableFloatTypes, Tanh)
{
    using T = TypeParam;

    T a(0.0f);
    EXPECT_TRUE(this->nearEqual(tanh(a), T(0.0f)));
}

TYPED_TEST(PortableFloatTypes, Floor)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    EXPECT_TRUE(this->nearEqual(floor(T(2.5f)), T(2.0f), Traits::large_tolerance));
    EXPECT_TRUE(this->nearEqual(floor(T(-2.5f)), T(-3.0f), Traits::large_tolerance));
}

TYPED_TEST(PortableFloatTypes, Ceil)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    EXPECT_TRUE(this->nearEqual(ceil(T(2.5f)), T(3.0f), Traits::large_tolerance));
    EXPECT_TRUE(this->nearEqual(ceil(T(-2.5f)), T(-2.0f), Traits::large_tolerance));
}

TYPED_TEST(PortableFloatTypes, Round)
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

    T a(2.0f);
    std::ostringstream oss;
    oss << a;
    float parsed = std::stof(oss.str());
    EXPECT_TRUE(this->nearEqual(parsed, 2.0f));
}

// ============================================================================
// numeric_limits Tests
// ============================================================================

TYPED_TEST(PortableFloatTypes, NumericLimitsBasic)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    EXPECT_TRUE(std::numeric_limits<T>::is_specialized);
    EXPECT_TRUE(std::numeric_limits<T>::is_signed);
    EXPECT_FALSE(std::numeric_limits<T>::is_integer);
    EXPECT_EQ(std::numeric_limits<T>::has_infinity, Traits::has_infinity);
    EXPECT_TRUE(std::numeric_limits<T>::has_quiet_NaN);
}

TYPED_TEST(PortableFloatTypes, NumericLimitsNaN)
{
    using T = TypeParam;

    T nan = std::numeric_limits<T>::quiet_NaN();
    EXPECT_TRUE(isnan(nan));
}

TYPED_TEST(PortableFloatTypes, NumericLimitsMax)
{
    using T = TypeParam;

    T maxVal = std::numeric_limits<T>::max();
    EXPECT_TRUE(isfinite(maxVal));
}

TYPED_TEST(PortableFloatTypes, NumericLimitsMin)
{
    using T = TypeParam;

    T minVal = std::numeric_limits<T>::min();
    EXPECT_TRUE(isfinite(minVal));
    EXPECT_GT(static_cast<float>(minVal), 0.0f);
}

TYPED_TEST(PortableFloatTypes, NumericLimitsLowest)
{
    using T = TypeParam;

    T lowestVal = std::numeric_limits<T>::lowest();
    EXPECT_TRUE(isfinite(lowestVal));
    EXPECT_LT(static_cast<float>(lowestVal), 0.0f);
}

TYPED_TEST(PortableFloatTypes, NumericLimitsEpsilon)
{
    using T = TypeParam;

    T eps = std::numeric_limits<T>::epsilon();
    EXPECT_TRUE(isfinite(eps));
    EXPECT_GT(static_cast<float>(eps), 0.0f);
}

TYPED_TEST(PortableFloatTypes, NumericLimitsInfinity)
{
    using T = TypeParam;
    using Traits = PortableTypeTraits<T>;

    if constexpr(Traits::has_infinity)
    {
        T inf = std::numeric_limits<T>::infinity();
        EXPECT_TRUE(isinf(inf));
        EXPECT_FALSE(signbit(inf));
    }
}
