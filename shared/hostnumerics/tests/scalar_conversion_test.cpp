// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/scalar.hpp>
#include <stdexcept>

namespace {
using namespace roc::hostnumerics;

constexpr ScalarConversionOptions reject;
constexpr ScalarConversionOptions saturate{IntegerRounding::TowardZero, IntegerOverflow::Saturate};
constexpr ScalarConversionOptions wrap{IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};

TEST(ScalarConversion, OmittedOptionsTruncateAndWrapWhileExplicitOptionsReject) {
    EXPECT_EQ(convertScalar<int32_t>(3.75), 3);
    EXPECT_EQ(convertScalar<int32_t>(-3.75), -3);
    EXPECT_EQ(convertScalar<int8_t>(128.75), -128);
    EXPECT_EQ(convertScalar<uint8_t>(-1.75), 255);
    EXPECT_EQ(convertScalar<uint8_t>(uint16_t{256}), 0);
    EXPECT_THROW((void)convertScalar<int8_t>(128.75, {}), std::runtime_error);
    EXPECT_THROW((void)convertScalar<uint8_t>(-1.75, {}), std::runtime_error);
}

template <typename T>
class IntegerConversion : public ::testing::Test {};
using Integers =
    ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(IntegerConversion, Integers);

TYPED_TEST(IntegerConversion, PreservesLimitsAndRejectsSignednessOverflow) {
    using T = TypeParam;
    constexpr T minimum = std::numeric_limits<T>::min();
    constexpr T maximum = std::numeric_limits<T>::max();
    EXPECT_EQ(convertScalar<T>(minimum, reject), minimum);
    EXPECT_EQ(convertScalar<T>(maximum, reject), maximum);
    if constexpr (std::is_signed_v<T>) {
        EXPECT_THROW((void)convertScalar<T>(std::numeric_limits<uint64_t>::max(), reject),
                     std::runtime_error);
        EXPECT_EQ(convertScalar<T>(std::numeric_limits<uint64_t>::max(), saturate), maximum);
        EXPECT_EQ(convertScalar<T>(std::numeric_limits<uint64_t>::max(), wrap), -1);
    } else {
        EXPECT_THROW((void)convertScalar<T>(int64_t{-1}, reject), std::runtime_error);
        EXPECT_EQ(convertScalar<T>(int64_t{-1}, saturate), 0);
        EXPECT_EQ(convertScalar<T>(int64_t{-1}, wrap), maximum);
    }
}

TEST(ScalarConversion, RejectsValuesImmediatelyOutsideNarrowIntegerLimits) {
    EXPECT_EQ(convertScalar<int8_t>(int16_t{-128}, reject), -128);
    EXPECT_EQ(convertScalar<int8_t>(int16_t{127}, reject), 127);
    EXPECT_THROW((void)convertScalar<int8_t>(int16_t{-129}, reject), std::runtime_error);
    EXPECT_THROW((void)convertScalar<int8_t>(int16_t{128}, reject), std::runtime_error);
    EXPECT_EQ(convertScalar<uint8_t>(uint16_t{255}, reject), 255);
    EXPECT_THROW((void)convertScalar<uint8_t>(uint16_t{256}, reject), std::runtime_error);
}

template <typename T>
class FloatingIntegerConversion : public ::testing::Test {};
using FloatingTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(FloatingIntegerConversion, FloatingTypes);

TYPED_TEST(FloatingIntegerConversion, ChecksWideBoundsBeforeCasting) {
    using T = TypeParam;
    const T twoTo63 = std::ldexp(T{1}, 63);
    const T twoTo64 = std::ldexp(T{1}, 64);
    const T infinity = std::numeric_limits<T>::infinity();
    EXPECT_EQ(convertScalar<int64_t>(-twoTo63, reject), std::numeric_limits<int64_t>::min());
    EXPECT_THROW((void)convertScalar<int64_t>(twoTo63, reject), std::runtime_error);
    EXPECT_THROW((void)convertScalar<int64_t>(std::nextafter(-twoTo63, -infinity), reject),
                 std::runtime_error);
    EXPECT_THROW((void)convertScalar<uint64_t>(twoTo64, reject), std::runtime_error);
    // The previous representable value is 2^64 minus one floating-point ULP.
    constexpr int precision =
        std::numeric_limits<T>::digits < 64 ? std::numeric_limits<T>::digits : 64;
    const auto gap = uint64_t{1} << (64 - precision);
    EXPECT_EQ(convertScalar<uint64_t>(std::nextafter(twoTo64, T{0}), reject),
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

TEST(ScalarConversion, RejectsNaNAndOnlySaturatesInfinity) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (const auto overflow :
         {IntegerOverflow::Reject, IntegerOverflow::Saturate, IntegerOverflow::ModuloWrap}) {
        const ScalarConversionOptions options{IntegerRounding::TowardZero, overflow};
        EXPECT_THROW((void)convertScalar<int32_t>(nan, options), std::runtime_error);
    }
    for (const auto options : {reject, wrap}) {
        EXPECT_THROW((void)convertScalar<int32_t>(infinity, options), std::runtime_error);
        EXPECT_THROW((void)convertScalar<int32_t>(-infinity, options), std::runtime_error);
    }
    EXPECT_EQ(convertScalar<int8_t>(infinity, saturate), 127);
    EXPECT_EQ(convertScalar<int8_t>(-infinity, saturate), -128);
    EXPECT_EQ(convertScalar<uint64_t>(infinity, saturate), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(convertScalar<uint8_t>(-infinity, saturate), 0);
    EXPECT_THROW((void)convertScalar<int8_t>(nan), std::runtime_error);
    EXPECT_THROW((void)convertScalar<int8_t>(infinity), std::runtime_error);
}

TEST(ScalarConversion, ConvertsComplexComponentsAndRejectsNonrealInputs) {
    const ScalarConversionOptions nearest{IntegerRounding::NearestEven, IntegerOverflow::Reject};
    EXPECT_EQ(convertScalar<int32_t>(std::complex<double>{3.5, 0.0}, nearest), 4);
    EXPECT_EQ(convertScalar<double>(std::complex<double>{-2.25, -0.0}), -2.25);
    EXPECT_EQ(convertScalar<std::complex<double>>(1.25f), std::complex<double>(1.25, 0.0));
    EXPECT_EQ(convertScalar<std::complex<float>>(std::complex<double>{1.25, -2.5}),
              std::complex<float>(1.25f, -2.5f));
    EXPECT_THROW((void)convertScalar<int32_t>(std::complex<double>{3.5, 1.0}, nearest),
                 std::runtime_error);
    EXPECT_THROW((void)convertScalar<double>(std::complex<double>{3.5, 1.0}), std::runtime_error);
    EXPECT_THROW((void)convertScalar<bool>(std::complex<double>{0.0, 1.0}), std::runtime_error);
    EXPECT_THROW((void)convertScalar<double>(
                     std::complex<double>{1.0, std::numeric_limits<double>::quiet_NaN()}),
                 std::runtime_error);
}

TEST(ScalarConversion, ConvertsBooleansEnumsAndFloatingValues) {
    enum class SignedValue : int16_t { Negative = -129, Positive = 128 };
    EXPECT_EQ(convertScalar<int32_t>(true, reject), 1);
    EXPECT_EQ(convertScalar<uint8_t>(false), 0);
    EXPECT_EQ(convertScalar<int8_t>(SignedValue::Positive), -128);
    EXPECT_EQ(convertScalar<int8_t>(SignedValue::Negative, saturate), -128);
    EXPECT_FALSE(convertScalar<bool>(-0.0));
    EXPECT_TRUE(convertScalar<bool>(-0.25));
    EXPECT_EQ(convertScalar<double>(int64_t{-123}), -123.0);
    EXPECT_EQ(convertScalar<float>(2.25), 2.25f);
    EXPECT_TRUE(std::isnan(convertScalar<double>(std::numeric_limits<float>::quiet_NaN())));
    EXPECT_TRUE(std::isinf(convertScalar<float>(std::numeric_limits<double>::infinity())));
}

TEST(ScalarConversion, ReturnsTypedFailuresWithoutFormattingDiagnostics) {
    using detail::ConversionError;
    struct Case {
        double value;
        ScalarConversionOptions options;
        ConversionError error;
    };
    const Case cases[]{
        {300.0, reject, ConversionError::RoundedOutOfRange},
        {std::numeric_limits<double>::quiet_NaN(), reject, ConversionError::NaN},
        {std::numeric_limits<double>::infinity(), reject, ConversionError::Infinity},
        {2.0, {IntegerRounding::Count, IntegerOverflow::Reject}, ConversionError::InvalidRounding},
        {300.0,
         {IntegerRounding::TowardZero, IntegerOverflow::Count},
         ConversionError::InvalidOverflow},
    };
    for (const auto& test : cases) {
        const auto result = detail::getConvertToInteger<int8_t>(test.value, test.options);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, test.error);
        EXPECT_EQ(result.error().destinationBits, 8u);
        EXPECT_TRUE(result.error().signedDestination);
        EXPECT_FALSE(detail::message(result.error()).empty());
    }
    const auto integral = detail::getConvertToInteger<uint8_t>(-1, reject);
    ASSERT_FALSE(integral.has_value());
    EXPECT_EQ(integral.error().code, ConversionError::IntegerOutOfRange);
    EXPECT_EQ(integral.error().destinationBits, 8u);
    EXPECT_FALSE(integral.error().signedDestination);
    const auto nonreal =
        detail::getConvertToInteger<int64_t>(std::complex<double>{2.0, 1.0}, reject);
    ASSERT_FALSE(nonreal.has_value());
    EXPECT_EQ(nonreal.error().code, ConversionError::Nonreal);
    EXPECT_EQ(nonreal.error().destinationBits, 64u);
    const auto invalid = detail::getConvertToInteger<uint8_t>(
        -1, {IntegerRounding::TowardZero, static_cast<IntegerOverflow>(255)});
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, ConversionError::InvalidOverflow);
}
}  // namespace
