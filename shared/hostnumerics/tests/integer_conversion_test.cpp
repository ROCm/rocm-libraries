// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/integer_conversion.hpp>
#include <type_traits>
#include <utility>

namespace {
using namespace roc::hostnumerics;
using enum IntegerRounding;
using enum IntegerOverflow;

TEST(IntegerConversionTutorial, Rounding) {
    EXPECT_EQ(convertToInteger<int32_t>(3.75, TowardZero, Identity), 3);
    EXPECT_EQ(convertToInteger<int32_t>(-3.75, TowardZero, Identity), -3);

    EXPECT_EQ(convertToInteger<int32_t>(2.75, NearestEven, Identity), 3);
    EXPECT_EQ(convertToInteger<int32_t>(-2.75, NearestEven, Identity), -3);
    // Halfway values choose the even integer.
    EXPECT_EQ(convertToInteger<int32_t>(2.5, NearestEven, Identity), 2);
    EXPECT_EQ(convertToInteger<int32_t>(3.5, NearestEven, Identity), 4);
    EXPECT_EQ(convertToInteger<int32_t>(-2.5, NearestEven, Identity), -2);
    EXPECT_EQ(convertToInteger<int32_t>(-3.5, NearestEven, Identity), -4);
}

TEST(IntegerConversionTutorial, Overflow) {
    // Saturation and wrapping differ outside int8_t's [-128, 127] range.
    EXPECT_EQ(convertToInteger<int8_t>(128.75, TowardZero, Saturate), 127);
    EXPECT_EQ(convertToInteger<int8_t>(128.75, TowardZero, ModuloWrap), -128);
    EXPECT_EQ(convertToInteger<uint8_t>(-1.75, TowardZero, Saturate), 0);
    EXPECT_EQ(convertToInteger<uint8_t>(-1.75, TowardZero, ModuloWrap), 255);
    EXPECT_EQ(convertToInteger<int8_t>(int32_t{300}, TowardZero, Saturate), 127);

    // Rounding can move an in-range input past the integer limit.
    EXPECT_EQ(convertToInteger<int8_t>(127.5, TowardZero, Identity), 127);
    EXPECT_EQ(convertToInteger<int8_t>(127.5, NearestEven, Saturate), 127);
    EXPECT_EQ(convertToInteger<int8_t>(127.5, NearestEven, ModuloWrap), -128);

    // Identity applies to the rounded value; a negative fraction can become unsigned zero.
    EXPECT_EQ(convertToInteger<uint8_t>(-0.75, TowardZero, Identity), 0);
}

TEST(IntegerConversionTutorial, Infinity) {
    // Infinity can convert only with Saturate. NaN is outside the domain.
    const double infinity = std::numeric_limits<double>::infinity();
    EXPECT_EQ(convertToInteger<int8_t>(infinity, NearestEven, Saturate), 127);
    EXPECT_EQ(convertToInteger<int8_t>(-infinity, NearestEven, Saturate), -128);
    EXPECT_EQ(convertToInteger<uint8_t>(-infinity, NearestEven, Saturate), 0);
    EXPECT_EQ(convertToInteger<uint64_t>(infinity, TowardZero, Saturate),
              std::numeric_limits<uint64_t>::max());
}

template <typename Target, typename Source>
concept Converts =
    requires(Source source) { convertToInteger<Target>(source, TowardZero, Identity); };

// The examples in the convertToInteger header comment.
TEST(IntegerConversion, HeaderExamples) {
    EXPECT_EQ(convertToInteger<int8_t>(double{2.5}, NearestEven, Identity), 2);
    EXPECT_EQ(convertToInteger<int8_t>(double{3.5}, NearestEven, Identity), 4);
    EXPECT_EQ(convertToInteger<int8_t>(double{-3.75}, TowardZero, Identity), -3);
    EXPECT_EQ(convertToInteger<int8_t>(int32_t{300}, TowardZero, Saturate), 127);
    EXPECT_EQ(convertToInteger<int8_t>(int32_t{300}, TowardZero, ModuloWrap), 44);
    EXPECT_EQ(convertToInteger<uint8_t>(int32_t{-1}, TowardZero, Saturate), 0);
    EXPECT_EQ(convertToInteger<uint8_t>(int32_t{-1}, TowardZero, ModuloWrap), 255);
    EXPECT_EQ(convertToInteger<int8_t>(double{1e300}, TowardZero, Saturate), 127);
    EXPECT_EQ(convertToInteger<int64_t>(bool{true}, TowardZero, Identity), 1);
    EXPECT_EQ(convertToInteger<size_t>(uint64_t{42}, TowardZero, Identity), 42);
    EXPECT_EQ(convertToInteger<char>(double{65.0}, TowardZero, Identity), 'A');

    static_assert(!Converts<bool, double>);
    static_assert(!Converts<float, int32_t>);
    static_assert(!Converts<char, std::complex<double>>);
#ifdef __SIZEOF_INT128__
    static_assert(!Converts<__int128, int64_t>);
    static_assert(!Converts<int64_t, __int128>);
#endif
}

// Includes the out-of-domain examples in the convertToInteger header comment.
TEST(IntegerConversionDeathTest, AssertsInputRequirements) {
#ifndef NDEBUG
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Identity requires the rounded value to fit.
    EXPECT_DEATH(convertToInteger<int8_t>(int32_t{300}, TowardZero, Identity), "");
    EXPECT_DEATH(convertToInteger<uint8_t>(int32_t{-1}, TowardZero, Identity), "");
    EXPECT_DEATH(convertToInteger<int8_t>(127.5, NearestEven, Identity), "");
    EXPECT_DEATH(convertToInteger<uint8_t>(-1.0, TowardZero, Identity), "");
    EXPECT_DEATH(convertToInteger<int8_t>(infinity, TowardZero, Identity), "");

    // ModuloWrap requires a finite input.
    EXPECT_DEATH(convertToInteger<int8_t>(infinity, TowardZero, ModuloWrap), "");
    EXPECT_DEATH(convertToInteger<int8_t>(-infinity, TowardZero, ModuloWrap), "");

    // NaN is outside the domain of every policy.
    for (const auto overflow : {Identity, Saturate, ModuloWrap})
        EXPECT_DEATH(convertToInteger<int8_t>(nan, NearestEven, overflow), "");
#else
    GTEST_SKIP() << "Input requirements are checked only when assertions are enabled.";
#endif
}

template <typename T>
class IntegerTargets : public ::testing::Test {};
using Integers =
    ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(IntegerTargets, Integers);

TYPED_TEST(IntegerTargets, PreservesLimitsAndAppliesSignednessPolicies) {
    using T = TypeParam;
    constexpr T minimum = std::numeric_limits<T>::min();
    constexpr T maximum = std::numeric_limits<T>::max();
    EXPECT_EQ(convertToInteger<T>(minimum, TowardZero, Identity), minimum);
    EXPECT_EQ(convertToInteger<T>(maximum, TowardZero, Identity), maximum);
    if constexpr (std::is_signed_v<T>) {
        EXPECT_EQ(convertToInteger<T>(std::numeric_limits<uint64_t>::max(), TowardZero, Saturate),
                  maximum);
        EXPECT_EQ(convertToInteger<T>(std::numeric_limits<uint64_t>::max(), TowardZero, ModuloWrap),
                  -1);
    } else {
        EXPECT_EQ(convertToInteger<T>(int64_t{-1}, TowardZero, Saturate), 0);
        EXPECT_EQ(convertToInteger<T>(int64_t{-1}, TowardZero, ModuloWrap), maximum);
    }
}

TYPED_TEST(IntegerTargets, PoliciesMatchIndependentIntegerExpectations) {
    using T = TypeParam;
    const auto check = [](auto value, int rounded, IntegerRounding rounding,
                          IntegerOverflow overflow) {
        const bool fits = std::in_range<T>(rounded);
        if (overflow == Identity && !fits) return;
        T expected = static_cast<T>(rounded);
        if (overflow == Saturate && !fits)
            expected = std::cmp_less(rounded, std::numeric_limits<T>::min())
                           ? std::numeric_limits<T>::min()
                           : std::numeric_limits<T>::max();
        EXPECT_EQ(convertToInteger<T>(value, rounding, overflow), expected);
    };
    for (int input = -300; input <= 300; ++input) {
        for (const auto overflow : {Identity, Saturate, ModuloWrap}) {
            check(input, input, TowardZero, overflow);
            for (const double offset : {-0.75, -0.5, 0.0, 0.5, 0.75}) {
                const double value = input + offset;
                const int lower = static_cast<int>(std::floor(value));
                const double fraction = value - lower;
                const int nearest = lower + (fraction > 0.5 || (fraction == 0.5 && lower % 2 != 0));
                check(value, static_cast<int>(value), TowardZero, overflow);
                check(value, nearest, NearestEven, overflow);
            }
        }
    }
}

template <typename T>
class FloatingSources : public ::testing::Test {};
using FloatingTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(FloatingSources, FloatingTypes);

TYPED_TEST(FloatingSources, ConvertsWideBounds) {
    using T = TypeParam;
    const T twoTo63 = std::ldexp(T{1}, 63);
    const T twoTo64 = std::ldexp(T{1}, 64);
    EXPECT_EQ(convertToInteger<int64_t>(-twoTo63, TowardZero, Identity),
              std::numeric_limits<int64_t>::min());

    // The previous representable value is 2^64 minus one floating-point ULP.
    constexpr int precision =
        std::numeric_limits<T>::digits < 64 ? std::numeric_limits<T>::digits : 64;
    const auto gap = uint64_t{1} << (64 - precision);
    EXPECT_EQ(convertToInteger<uint64_t>(std::nextafter(twoTo64, T{0}), TowardZero, Identity),
              std::numeric_limits<uint64_t>::max() - (gap - 1));
    EXPECT_EQ(convertToInteger<int64_t>(twoTo63, TowardZero, ModuloWrap),
              std::numeric_limits<int64_t>::min());
    EXPECT_EQ(convertToInteger<uint64_t>(twoTo64, TowardZero, ModuloWrap), 0);
    EXPECT_EQ(convertToInteger<uint64_t>(T{-1}, TowardZero, ModuloWrap),
              std::numeric_limits<uint64_t>::max());
}

TEST(IntegerConversion, SaturatesUnsignedSourcesAndWrapsWideValues) {
    EXPECT_EQ(convertToInteger<uint8_t>(uint16_t{256}, TowardZero, Saturate), 255);
    const double twoTo64 = std::ldexp(1.0, 64);
    EXPECT_EQ(
        convertToInteger<uint64_t>(std::nextafter(twoTo64, std::numeric_limits<double>::infinity()),
                                   TowardZero, ModuloWrap),
        4096);
    EXPECT_EQ(
        convertToInteger<uint64_t>(std::numeric_limits<int64_t>::min(), TowardZero, ModuloWrap),
        uint64_t{1} << 63);
    EXPECT_EQ(convertToInteger<int8_t>(std::numeric_limits<double>::max(), TowardZero, ModuloWrap),
              0);
    EXPECT_EQ(convertToInteger<int8_t>(-std::numeric_limits<double>::max(), TowardZero, ModuloWrap),
              0);
}

template <typename T>
class CharacterSources : public ::testing::Test {};
using Characters = ::testing::Types<char, wchar_t, char8_t, char16_t, char32_t>;
TYPED_TEST_SUITE(CharacterSources, Characters);

TYPED_TEST(CharacterSources, ConvertsNumericValuesUnderEveryPolicy) {
    using T = TypeParam;
    for (const auto overflow : {Identity, Saturate, ModuloWrap}) {
        EXPECT_EQ(convertToInteger<int64_t>(T{42}, TowardZero, overflow), 42);
        EXPECT_EQ(convertToInteger<uint64_t>(std::numeric_limits<T>::max(), TowardZero, overflow),
                  static_cast<uint64_t>(std::numeric_limits<T>::max()));
    }
    EXPECT_EQ(convertToInteger<int64_t>(std::numeric_limits<T>::min(), TowardZero, Saturate),
              static_cast<int64_t>(std::numeric_limits<T>::min()));
    EXPECT_EQ(convertToInteger<uint64_t>(std::numeric_limits<T>::min(), TowardZero, Saturate), 0);
}

TEST(IntegerConversion, NarrowsCharacterValues) {
    EXPECT_EQ(convertToInteger<uint8_t>(char16_t{256}, TowardZero, Saturate), 255);
    EXPECT_EQ(convertToInteger<uint8_t>(char16_t{256}, TowardZero, ModuloWrap), 0);
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

TEST(IntegerConversion, RoundingIgnoresHostRoundingMode) {
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
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        for (const auto& test : cases) {
            // A runtime load prevents constant folding under the default rounding mode.
            volatile double value = test.value;
            EXPECT_EQ(convertToInteger<int32_t>(value, NearestEven, Identity), test.nearest);
            EXPECT_EQ(convertToInteger<int32_t>(value, TowardZero, Identity), test.truncated);
        }
        volatile double positiveTie = 127.5;
        volatile double negativeTie = -128.5;
        EXPECT_EQ(convertToInteger<int8_t>(negativeTie, NearestEven, Identity), -128);
        EXPECT_EQ(convertToInteger<int8_t>(positiveTie, NearestEven, Saturate), 127);
        EXPECT_EQ(convertToInteger<int8_t>(positiveTie, NearestEven, ModuloWrap), -128);
    }
}

}  // namespace
