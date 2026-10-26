// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/scalar.hpp>
#include <type_traits>

namespace {
using namespace roc::hostnumerics;

TEST(ScalarConversionTutorial, Rounding) {
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
    // Halfway values choose the even integer.
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
    // Saturation and wrapping differ outside int8_t's [-128, 127] range.
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

    // Rounding can move an in-range input past the integer limit.
    EXPECT_EQ(
        convertScalar<int8_t>(127.5, {IntegerRounding::TowardZero, IntegerOverflow::Identity}),
        127);
    EXPECT_EQ(
        convertScalar<int8_t>(127.5, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
        127);
    EXPECT_EQ(
        convertScalar<int8_t>(127.5, {IntegerRounding::NearestEven, IntegerOverflow::ModuloWrap}),
        -128);

    // Identity applies to the rounded value; a negative fraction can become unsigned zero.
    EXPECT_EQ(
        convertScalar<uint8_t>(-0.75, {IntegerRounding::TowardZero, IntegerOverflow::Identity}), 0);
}

TEST(ScalarConversionTutorial, FloatingAndBooleanDestinations) {
    // Floating destinations ignore integer policies; binary32 cannot represent 2^24 + 1.
    EXPECT_EQ(
        convertScalar<double>(2.75f, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
        2.75);
    EXPECT_EQ(convertScalar<float>(int32_t{16'777'217},
                                   {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
              16'777'216.0f);

    // Boolean conversion tests for nonzero before any integer rounding.
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

    EXPECT_EQ(
        convertScalar<uint64_t>(infinity, {IntegerRounding::TowardZero, IntegerOverflow::Saturate}),
        std::numeric_limits<uint64_t>::max());

    // NaN is outside the integer domain. Floating conversion preserves it; bool sees nonzero.
    EXPECT_TRUE(std::isnan(
        convertScalar<float>(nan, {IntegerRounding::NearestEven, IntegerOverflow::Saturate})));
    EXPECT_TRUE(
        convertScalar<bool>(nan, {IntegerRounding::NearestEven, IntegerOverflow::Saturate}));
}

TEST(ScalarConversionTutorial, ComplexValues) {
    // Complex destinations convert each component; real sources supply a zero imaginary part.
    EXPECT_EQ(convertScalar<std::complex<double>>(
                  1.25f, {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
              std::complex<double>(1.25, 0.0));
    EXPECT_EQ(convertScalar<std::complex<float>>(
                  std::complex<double>{1.25, -2.5},
                  {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
              std::complex<float>(1.25f, -2.5f));

    // Complex-to-real requires a zero imaginary part.
    EXPECT_EQ(convertScalar<int32_t>(std::complex<double>{3.5, 0.0},
                                     {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
              4);
    EXPECT_EQ(convertScalar<double>(std::complex<double>{-2.25, -0.0},
                                    {IntegerRounding::NearestEven, IntegerOverflow::Identity}),
              -2.25);
}

constexpr ScalarConversionOptions identity{IntegerRounding::TowardZero, IntegerOverflow::Identity};
constexpr ScalarConversionOptions saturate{IntegerRounding::TowardZero, IntegerOverflow::Saturate};
constexpr ScalarConversionOptions wrap{IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};

template <typename Source>
concept ConvertsWithoutPolicies = requires(Source source) { convertScalar<int32_t>(source); };

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

TEST(ScalarConversion, SaturatesUnsignedSourcesAndWrapsWideValues) {
    EXPECT_EQ(convertScalar<uint8_t>(uint16_t{256}, saturate), 255);
    const double twoTo64 = std::ldexp(1.0, 64);
    EXPECT_EQ(convertScalar<uint64_t>(
                  std::nextafter(twoTo64, std::numeric_limits<double>::infinity()), wrap),
              4096);
    EXPECT_EQ(convertScalar<uint64_t>(std::numeric_limits<int64_t>::min(), wrap),
              uint64_t{1} << 63);
    EXPECT_EQ(convertScalar<int8_t>(std::numeric_limits<double>::max(), wrap), 0);
    EXPECT_EQ(convertScalar<int8_t>(-std::numeric_limits<double>::max(), wrap), 0);
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

class RoundingModeRestore {
   public:
    RoundingModeRestore() : mode(std::fegetround()) {}
    ~RoundingModeRestore() {
        std::fesetround(mode);
    }

   private:
    int mode;
};

TEST(ScalarRounding, NearestEvenAndTowardZeroIgnoreHostRoundingMode) {
    const RoundingModeRestore restore;
    struct Case {
        double value;
        int32_t nearest;
        int32_t truncated;
    };
    const std::array<Case, 12> cases{{
        {0.5, 0, 0},
        {1.5, 2, 1},
        {2.5, 2, 2},
        {3.5, 4, 3},
        {-0.5, 0, 0},
        {-1.5, -2, -1},
        {-2.5, -2, -2},
        {-3.5, -4, -3},
        {2.25, 2, 2},
        {2.75, 3, 2},
        {-2.25, -2, -2},
        {-2.75, -3, -2},
    }};
    const ScalarConversionOptions nearest{IntegerRounding::NearestEven, IntegerOverflow::Identity};
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        for (const auto& test : cases) {
            // A runtime load prevents constant folding under the default rounding mode.
            volatile double value = test.value;
            EXPECT_EQ(convertScalar<int32_t>(value, nearest), test.nearest);
            EXPECT_EQ(convertScalar<int32_t>(value, identity), test.truncated);
        }
        volatile double positiveTie = 127.5;
        volatile double negativeTie = -128.5;
        EXPECT_EQ(convertScalar<int8_t>(negativeTie, nearest), -128);
        EXPECT_EQ(convertScalar<int8_t>(positiveTie,
                                        {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
                  127);
        EXPECT_EQ(convertScalar<int8_t>(
                      positiveTie, {IntegerRounding::NearestEven, IntegerOverflow::ModuloWrap}),
                  -128);
    }
}

}  // namespace
