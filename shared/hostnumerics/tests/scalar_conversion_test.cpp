// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/scalar.hpp>
#include <type_traits>

namespace {
using namespace roc::hostnumerics;

TEST(ScalarConversionTutorial, Rounding) {
    // Integer conversion is cast<T>(f_T(g(x))). Both policies are explicit:
    // g rounds TowardZero or NearestEven; f_T applies Identity, Saturate, or
    // ModuloWrap. Identity leaves g(x) unchanged, so it must already fit in T.

    // TowardZero drops the fractional part; negative values move toward zero too.
    EXPECT_EQ(
        convertScalar<int32_t>(3.75, {IntegerRounding::TowardZero, IntegerOverflow::Identity}), 3);
    EXPECT_EQ(
        convertScalar<int32_t>(-3.75, {IntegerRounding::TowardZero, IntegerOverflow::Identity}),
        -3);

    EXPECT_EQ(
        convertScalar<int32_t>(2.75, {IntegerRounding::NearestEven, IntegerOverflow::Identity}), 3);
    EXPECT_EQ(
        convertScalar<int32_t>(-2.75, {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
        -3);
    // Halfway values choose the even integer, including on the negative side.
    EXPECT_EQ(
        convertScalar<int32_t>(2.5, {IntegerRounding::NearestEven, IntegerOverflow::Identity}), 2);
    EXPECT_EQ(
        convertScalar<int32_t>(3.5, {IntegerRounding::NearestEven, IntegerOverflow::Identity}), 4);
    EXPECT_EQ(
        convertScalar<int32_t>(-2.5, {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
        -2);
    EXPECT_EQ(
        convertScalar<int32_t>(-3.5, {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
        -4);
}

TEST(ScalarConversionTutorial, Overflow) {
    // int8_t spans [-128, 127]. Saturate clamps to that range; ModuloWrap keeps
    // the low eight bits and interprets a signed result as two's complement.
    EXPECT_EQ(
        convertScalar<int8_t>(128.75, {IntegerRounding::TowardZero, IntegerOverflow::Saturate}),
        127);
    EXPECT_EQ(
        convertScalar<int8_t>(128.75, {IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap}),
        -128);
    EXPECT_EQ(
        convertScalar<uint8_t>(-1.75, {IntegerRounding::TowardZero, IntegerOverflow::Saturate}), 0);
    EXPECT_EQ(
        convertScalar<uint8_t>(-1.75, {IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap}),
        255);

    // Rounding happens first: 127.5 truncates to 127 but rounds to 128 with
    // NearestEven. The latter needs Saturate or ModuloWrap to satisfy the contract.
    EXPECT_EQ(
        convertScalar<int8_t>(127.5, {IntegerRounding::TowardZero, IntegerOverflow::Identity}),
        127);
    EXPECT_EQ(
        convertScalar<int8_t>(127.5, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
        127);
    EXPECT_EQ(
        convertScalar<int8_t>(127.5, {IntegerRounding::NearestEven, IntegerOverflow::ModuloWrap}),
        -128);

    // A negative fraction can still use Identity for an unsigned destination:
    // TowardZero produces zero before representability matters.
    EXPECT_EQ(
        convertScalar<uint8_t>(-0.75, {IntegerRounding::TowardZero, IntegerOverflow::Identity}), 0);
}

TEST(ScalarConversionTutorial, FloatingAndBooleanDestinations) {
    // These policies control integer destinations. Floating-point destinations
    // follow native casts and can lose precision: binary32 cannot represent 2^24 + 1.
    EXPECT_EQ(
        convertScalar<double>(2.75f, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
        2.75);
    EXPECT_EQ(convertScalar<float>(int32_t{16'777'217},
                                   {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
              16'777'216.0f);

    // Boolean conversion tests whether the source is nonzero, even when integer
    // rounding of the same value would produce zero. Negative zero is still zero.
    EXPECT_EQ(
        convertScalar<int32_t>(0.25, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}), 0);
    EXPECT_TRUE(
        convertScalar<bool>(0.25, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}));
    EXPECT_FALSE(
        convertScalar<bool>(-0.0, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}));
}

TEST(ScalarConversionTutorial, NonFiniteValues) {
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Infinity can convert to a non-Boolean integer only with Saturate.
    EXPECT_EQ(
        convertScalar<int8_t>(infinity, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
        127);
    EXPECT_EQ(
        convertScalar<int8_t>(-infinity, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
        -128);
    EXPECT_EQ(convertScalar<uint8_t>(-infinity,
                                     {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
              0);

    // NaN is outside the domain for non-Boolean integer destinations, even with
    // Saturate. Floating-point conversion preserves NaN; Boolean conversion sees nonzero.
    EXPECT_TRUE(std::isnan(
        convertScalar<float>(nan, {IntegerRounding::NearestEven, IntegerOverflow::Saturate})));
    EXPECT_TRUE(
        convertScalar<bool>(nan, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}));
}

TEST(ScalarConversionTutorial, ComplexValues) {
    // Real-to-complex supplies a zero imaginary part. Complex-to-complex converts
    // both components; integer rounding policies do not round floating components.
    EXPECT_EQ(convertScalar<std::complex<double>>(
                  1.25f, {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
              std::complex<double>(1.25, 0.0));
    EXPECT_EQ(convertScalar<std::complex<float>>(
                  std::complex<double>{1.25, -2.5},
                  {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
              std::complex<float>(1.25f, -2.5f));

    // Complex-to-real requires a zero imaginary part, then converts the real part.
    // A nonzero imaginary part violates an unchecked caller precondition.
    EXPECT_EQ(convertScalar<int32_t>(std::complex<double>{3.5, 0.0},
                                     {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
              4);
}

constexpr ScalarConversionOptions identity{IntegerRounding::TowardZero, IntegerOverflow::Identity};
constexpr ScalarConversionOptions saturate{IntegerRounding::TowardZero, IntegerOverflow::Saturate};
constexpr ScalarConversionOptions wrap{IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};

template <typename Source>
concept ConvertsWithoutPolicies = requires(Source source) { convertScalar<int32_t>(source); };

TEST(ScalarConversion, ExplicitOptionsTruncateAndWrap) {
    EXPECT_EQ(convertScalar<int32_t>(3.75, wrap), 3);
    EXPECT_EQ(convertScalar<int32_t>(-3.75, wrap), -3);
    EXPECT_EQ(convertScalar<int8_t>(128.75, wrap), -128);
    EXPECT_EQ(convertScalar<uint8_t>(-1.75, wrap), 255);
    EXPECT_EQ(convertScalar<uint8_t>(uint16_t{256}, wrap), 0);
}

template <typename T>
class IntegerConversion : public ::testing::Test {};
using Integers =
    ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(IntegerConversion, Integers);

TYPED_TEST(IntegerConversion, PreservesLimitsAndAppliesSignednessPolicies) {
    using T = TypeParam;
    constexpr T minimum = std::numeric_limits<T>::min();
    constexpr T maximum = std::numeric_limits<T>::max();
    EXPECT_EQ(convertScalar<T>(minimum, identity), minimum);
    EXPECT_EQ(convertScalar<T>(maximum, identity), maximum);
    if constexpr (std::is_signed_v<T>) {
        EXPECT_EQ(convertScalar<T>(std::numeric_limits<uint64_t>::max(), saturate), maximum);
        EXPECT_EQ(convertScalar<T>(std::numeric_limits<uint64_t>::max(), wrap), -1);
    } else {
        EXPECT_EQ(convertScalar<T>(int64_t{-1}, saturate), 0);
        EXPECT_EQ(convertScalar<T>(int64_t{-1}, wrap), maximum);
    }
}

TEST(ScalarConversion, PreservesNarrowIntegerLimits) {
    EXPECT_EQ(convertScalar<int8_t>(int16_t{-128}, identity), -128);
    EXPECT_EQ(convertScalar<int8_t>(int16_t{127}, identity), 127);

    EXPECT_EQ(convertScalar<uint8_t>(uint16_t{255}, identity), 255);
    EXPECT_EQ(convertScalar<int8_t>(-128.75, identity), -128);
    EXPECT_EQ(convertScalar<int8_t>(127.75, identity), 127);
    EXPECT_EQ(convertScalar<uint8_t>(-0.75, identity), 0);
}

template <typename T>
class FloatingIntegerConversion : public ::testing::Test {};
using FloatingTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(FloatingIntegerConversion, FloatingTypes);

TYPED_TEST(FloatingIntegerConversion, ConvertsWideBounds) {
    using T = TypeParam;
    const T twoTo63 = std::ldexp(T{1}, 63);
    const T twoTo64 = std::ldexp(T{1}, 64);
    EXPECT_EQ(convertScalar<int64_t>(-twoTo63, identity), std::numeric_limits<int64_t>::min());

    // The previous representable value is 2^64 minus one floating-point ULP.
    constexpr int precision =
        std::numeric_limits<T>::digits < 64 ? std::numeric_limits<T>::digits : 64;
    const auto gap = uint64_t{1} << (64 - precision);
    EXPECT_EQ(convertScalar<uint64_t>(std::nextafter(twoTo64, T{0}), identity),
              std::numeric_limits<uint64_t>::max() - (gap - 1));
    EXPECT_EQ(convertScalar<int64_t>(twoTo63, wrap), std::numeric_limits<int64_t>::min());
    EXPECT_EQ(convertScalar<uint64_t>(twoTo64, wrap), 0);
    EXPECT_EQ(convertScalar<uint64_t>(T{-1}, wrap), std::numeric_limits<uint64_t>::max());
}

TEST(ScalarConversion, SaturatesOrWrapsAfterRounding) {
    const ScalarConversionOptions nearestSaturate{IntegerRounding::NearestEven,
                                                  IntegerOverflow::Saturate};
    const ScalarConversionOptions nearestWrap{IntegerRounding::NearestEven,
                                              IntegerOverflow::ModuloWrap};
    EXPECT_EQ(convertScalar<int8_t>(int16_t{-129}, saturate), -128);
    EXPECT_EQ(convertScalar<int8_t>(int16_t{128}, saturate), 127);
    EXPECT_EQ(convertScalar<uint8_t>(int16_t{-1}, saturate), 0);
    EXPECT_EQ(convertScalar<uint8_t>(uint16_t{256}, saturate), 255);
    EXPECT_EQ(convertScalar<int8_t>(-128.75, nearestSaturate), -128);
    EXPECT_EQ(convertScalar<int8_t>(127.5, nearestSaturate), 127);
    EXPECT_EQ(convertScalar<int8_t>(127.5, nearestWrap), -128);
    EXPECT_EQ(convertScalar<int8_t>(int16_t{-129}, wrap), 127);
    EXPECT_EQ(convertScalar<uint8_t>(258.6, nearestWrap), 3);
    const double twoTo64 = std::ldexp(1.0, 64);
    EXPECT_EQ(convertScalar<uint64_t>(
                  std::nextafter(twoTo64, std::numeric_limits<double>::infinity()), wrap),
              4096);
    EXPECT_EQ(convertScalar<uint64_t>(std::numeric_limits<int64_t>::min(), wrap),
              uint64_t{1} << 63);
    EXPECT_EQ(convertScalar<int8_t>(std::numeric_limits<double>::max(), wrap), 0);
    EXPECT_EQ(convertScalar<int8_t>(-std::numeric_limits<double>::max(), wrap), 0);
}

TEST(ScalarConversion, SaturatesInfinity) {
    const double infinity = std::numeric_limits<double>::infinity();
    EXPECT_EQ(convertScalar<int8_t>(infinity, saturate), 127);
    EXPECT_EQ(convertScalar<int8_t>(-infinity, saturate), -128);
    EXPECT_EQ(convertScalar<uint64_t>(infinity, saturate), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(convertScalar<uint8_t>(-infinity, saturate), 0);
}

TEST(ScalarConversion, ConvertsComplexComponents) {
    const ScalarConversionOptions nearest{IntegerRounding::NearestEven, IntegerOverflow::Identity};
    EXPECT_EQ(convertScalar<int32_t>(std::complex<double>{3.5, 0.0}, nearest), 4);
    EXPECT_EQ(convertScalar<double>(std::complex<double>{-2.25, -0.0}, wrap), -2.25);
    EXPECT_EQ(convertScalar<std::complex<double>>(1.25f, wrap), std::complex<double>(1.25, 0.0));
    EXPECT_EQ(convertScalar<std::complex<float>>(std::complex<double>{1.25, -2.5}, wrap),
              std::complex<float>(1.25f, -2.5f));
}

TEST(ScalarConversion, ConvertsBooleansAndFloatingValues) {
    EXPECT_EQ(convertScalar<int32_t>(true, identity), 1);
    EXPECT_EQ(convertScalar<uint8_t>(false, wrap), 0);
    EXPECT_FALSE(convertScalar<bool>(-0.0, wrap));
    EXPECT_TRUE(convertScalar<bool>(-0.25, wrap));
    EXPECT_EQ(convertScalar<double>(int64_t{-123}, wrap), -123.0);
    EXPECT_EQ(convertScalar<float>(2.25, wrap), 2.25f);
    EXPECT_TRUE(std::isnan(convertScalar<double>(std::numeric_limits<float>::quiet_NaN(), wrap)));
    EXPECT_TRUE(std::isinf(convertScalar<float>(std::numeric_limits<double>::infinity(), wrap)));
}

template <typename T>
class CharacterConversion : public ::testing::Test {};
using Characters = ::testing::Types<char, wchar_t, char8_t, char16_t, char32_t>;
TYPED_TEST_SUITE(CharacterConversion, Characters);

TYPED_TEST(CharacterConversion, ConvertsNumericValuesUnderEveryPolicy) {
    using T = TypeParam;
    for (const auto options : {identity, saturate, wrap}) {
        EXPECT_EQ(convertScalar<int64_t>(T{42}, options), 42);
        EXPECT_EQ(convertScalar<uint64_t>(std::numeric_limits<T>::max(), options),
                  static_cast<uint64_t>(std::numeric_limits<T>::max()));
    }
    EXPECT_EQ(convertScalar<int64_t>(std::numeric_limits<T>::min(), saturate),
              static_cast<int64_t>(std::numeric_limits<T>::min()));
    EXPECT_EQ(convertScalar<uint64_t>(std::numeric_limits<T>::min(), saturate), 0);
}

TEST(ScalarConversion, NarrowsCharacterValues) {
    EXPECT_EQ(convertScalar<uint8_t>(char16_t{256}, saturate), 255);
    EXPECT_EQ(convertScalar<uint8_t>(char16_t{256}, wrap), 0);
}

TEST(ScalarConversion, RequiresExplicitPolicies) {
    static_assert(!std::is_default_constructible_v<ScalarConversionOptions>);
    static_assert(!std::is_constructible_v<ScalarConversionOptions, IntegerRounding>);
    static_assert(!ConvertsWithoutPolicies<double>);
    static_assert(std::is_same_v<decltype(convertScalar<int8_t>(128, wrap)), int8_t>);
    EXPECT_EQ(convertScalar<int8_t>(128, wrap), -128);
    EXPECT_EQ(convertScalar<int8_t>(2.5, identity), 2);
    EXPECT_EQ(convertScalar<std::complex<double>>(1.5, wrap), std::complex<double>(1.5, 0.0));
}

TYPED_TEST(IntegerConversion, PoliciesMatchIndependentIntegerExpectations) {
    using T = TypeParam;
    const auto check = [](auto value, int rounded, const ScalarConversionOptions& options) {
        const bool fits = std::in_range<T>(rounded);
        if (options.integerOverflow == IntegerOverflow::Identity && !fits) return;
        T expected = static_cast<T>(rounded);
        if (options.integerOverflow == IntegerOverflow::Saturate && !fits)
            expected = std::cmp_less(rounded, std::numeric_limits<T>::min())
                           ? std::numeric_limits<T>::min()
                           : std::numeric_limits<T>::max();
        EXPECT_EQ(convertScalar<T>(value, options), expected);
    };
    for (int input = -300; input <= 300; ++input) {
        for (const auto options : {identity, saturate, wrap}) {
            check(input, input, options);
            for (const double offset : {-0.75, -0.5, 0.0, 0.5, 0.75}) {
                const double value = input + offset;
                const int lower = static_cast<int>(std::floor(value));
                const double fraction = value - lower;
                const int nearest = lower + (fraction > 0.5 || (fraction == 0.5 && lower % 2 != 0));
                check(value, static_cast<int>(value), options);
                check(value, nearest, {IntegerRounding::NearestEven, options.integerOverflow});
            }
        }
    }
}

}  // namespace
