// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "_scalar_codec_test_support.hpp"

namespace scalar_codec_test {
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
}

template <typename T>
class CharacterConversion : public ::testing::Test {};
using Characters = ::testing::Types<char, wchar_t, char8_t, char16_t, char32_t>;
TYPED_TEST_SUITE(CharacterConversion, Characters);

TYPED_TEST(CharacterConversion, ConvertsNumericValuesUnderEveryPolicy) {
    using T = TypeParam;
    const ScalarConversionOptions saturate{IntegerRounding::TowardZero, IntegerOverflow::Saturate};
    for (const auto overflow :
         {IntegerOverflow::Identity, IntegerOverflow::Saturate, IntegerOverflow::ModuloWrap}) {
        const ScalarConversionOptions options{IntegerRounding::TowardZero, overflow};
        EXPECT_EQ(convertScalar<int64_t>(T{42}, options), 42);
        EXPECT_EQ(convertScalar<uint64_t>(std::numeric_limits<T>::max(), options),
                  static_cast<uint64_t>(std::numeric_limits<T>::max()));
    }
    EXPECT_EQ(convertScalar<int64_t>(std::numeric_limits<T>::min(), saturate),
              static_cast<int64_t>(std::numeric_limits<T>::min()));
    EXPECT_EQ(convertScalar<uint64_t>(std::numeric_limits<T>::min(), saturate), 0);
}

TEST(ScalarConversion, NarrowsCharacterValues) {
    const ScalarConversionOptions saturate{IntegerRounding::TowardZero, IntegerOverflow::Saturate};
    const ScalarConversionOptions wrap{IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};
    EXPECT_EQ(convertScalar<uint8_t>(char16_t{256}, saturate), 255);
    EXPECT_EQ(convertScalar<uint8_t>(char16_t{256}, wrap), 0);
}

template <typename Source>
concept ConvertsWithoutPolicies = requires(Source source) { convertScalar<int32_t>(source); };

void testIntegerConversionPrimitives() {
    static_assert(!std::is_default_constructible_v<ScalarConversionOptions>);
    static_assert(!std::is_constructible_v<ScalarConversionOptions, IntegerRounding>);
    static_assert(!ConvertsWithoutPolicies<double>);
    const ScalarConversionOptions nearestBFloat16{
        IntegerRounding::TowardZero, IntegerOverflow::Identity, BFloat16Rounding::NearestEven};

    constexpr float bfloat16RoundingBoundary = 1.005859375f;
    Tensor nearestEven(ScalarType::BFloat16, Shape{1});
    nearestEven.storeFrom({0}, bfloat16RoundingBoundary, nearestBFloat16);
    require(tensorRaw(nearestEven) == 0x3f81,
            "Explicit BFloat16 conversion did not round to nearest even.");

    const ScalarConversionOptions truncateBFloat16{
        IntegerRounding::TowardZero, IntegerOverflow::Identity, BFloat16Rounding::Truncate};
    Tensor truncated(ScalarType::BFloat16, Shape{1});
    truncated.storeFrom({0}, bfloat16RoundingBoundary, truncateBFloat16);
    require(tensorRaw(truncated) == 0x3f80,
            "Explicit BFloat16 truncation did not discard the low bits.");

    const ScalarConversionOptions identityTowardZero{
        IntegerRounding::TowardZero,
        IntegerOverflow::Identity,
    };
    const ScalarConversionOptions identityNearestEven{
        IntegerRounding::NearestEven,
        IntegerOverflow::Identity,
    };
    const ScalarConversionOptions saturateNearestEven{
        IntegerRounding::NearestEven,
        IntegerOverflow::Saturate,
    };
    const ScalarConversionOptions wrapNearestEven{
        IntegerRounding::NearestEven,
        IntegerOverflow::ModuloWrap,
    };

    require(convertScalar<int32_t>(3.75, identityTowardZero) == 3 &&
                convertScalar<int32_t>(-3.75, identityTowardZero) == -3,
            "Toward-zero integer rounding mismatch.");
    require(convertScalar<int8_t>(
                128.0, {IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap}) == -128,
            "Explicit modulo overflow conversion mismatch.");

    const Tensor saturatedInt4 = Tensor::scalar(ScalarType::Int4, 9, saturateNearestEven);
    const Tensor wrappedInt4 = Tensor::scalar(ScalarType::Int4, 9, wrapNearestEven);
    require(saturatedInt4.item<int>() == 7 && wrappedInt4.item<int>() == -7,
            "Explicit-policy rank-zero integer construction mismatch.");
    const Tensor explicitlyTruncatedBFloat16 =
        Tensor::scalar(ScalarType::BFloat16, bfloat16RoundingBoundary, truncateBFloat16);
    require(tensorRaw(explicitlyTruncatedBFloat16) == 0x3f80,
            "Explicit-policy rank-zero BFloat16 construction did not truncate.");

    struct TieCase {
        double value;
        int32_t expected;
    };
    const std::array<TieCase, 6> ties{{
        {0.5, 0},
        {1.5, 2},
        {2.5, 2},
        {-0.5, 0},
        {-1.5, -2},
        {-2.5, -2},
    }};
    for (const auto& tie : ties)
        require(convertScalar<int32_t>(tie.value, identityNearestEven) == tie.expected,
                "Nearest-even tie rounding mismatch.");

    require(convertScalar<int8_t>(int16_t{-128}, identityTowardZero) == -128 &&
                convertScalar<int8_t>(int16_t{127}, identityTowardZero) == 127,
            "Signed integer boundary conversion mismatch.");

    require(convertScalar<uint8_t>(uint16_t{0}, identityTowardZero) == 0 &&
                convertScalar<uint8_t>(uint16_t{255}, identityTowardZero) == 255,
            "Unsigned integer boundary conversion mismatch.");

    require(convertScalar<int64_t>(std::numeric_limits<int64_t>::min(), identityTowardZero) ==
                    std::numeric_limits<int64_t>::min() &&
                convertScalar<int64_t>(std::numeric_limits<int64_t>::max(), identityTowardZero) ==
                    std::numeric_limits<int64_t>::max() &&
                convertScalar<uint64_t>(std::numeric_limits<uint64_t>::max(), identityTowardZero) ==
                    std::numeric_limits<uint64_t>::max(),
            "Wide integer boundary conversion mismatch.");

    const double twoTo63 = std::ldexp(1.0, 63);
    const double twoTo64 = std::ldexp(1.0, 64);
    const double belowTwoTo64 = std::nextafter(twoTo64, 0.0);
    require(
        convertScalar<int64_t>(-twoTo63, identityTowardZero) == std::numeric_limits<int64_t>::min(),
        "Float-to-Int64 minimum conversion mismatch.");
    require(convertScalar<uint64_t>(belowTwoTo64, identityTowardZero) ==
                std::numeric_limits<uint64_t>::max() - uint64_t{2047},
            "Float-to-UInt64 maximum representable value mismatch.");

    require(convertScalar<int8_t>(int16_t{-129}, saturateNearestEven) == -128 &&
                convertScalar<int8_t>(int16_t{128}, saturateNearestEven) == 127 &&
                convertScalar<uint8_t>(int16_t{-1}, saturateNearestEven) == 0 &&
                convertScalar<uint8_t>(uint16_t{256}, saturateNearestEven) == 255,
            "Saturating integer conversion mismatch.");
    require(convertScalar<int8_t>(std::numeric_limits<double>::infinity(), saturateNearestEven) ==
                    127 &&
                convertScalar<int8_t>(-std::numeric_limits<double>::infinity(),
                                      saturateNearestEven) == -128 &&
                convertScalar<uint8_t>(-std::numeric_limits<double>::infinity(),
                                       saturateNearestEven) == 0,
            "Saturating infinity conversion mismatch.");

    require(convertScalar<int8_t>(int16_t{128}, wrapNearestEven) == -128 &&
                convertScalar<int8_t>(int16_t{129}, wrapNearestEven) == -127 &&
                convertScalar<int8_t>(int16_t{-129}, wrapNearestEven) == 127 &&
                convertScalar<uint8_t>(int16_t{-1}, wrapNearestEven) == 255 &&
                convertScalar<uint8_t>(uint16_t{256}, wrapNearestEven) == 0 &&
                convertScalar<uint8_t>(258.6, wrapNearestEven) == 3,
            "Modulo-wrap integer conversion mismatch.");
    require(convertScalar<int64_t>(std::numeric_limits<uint64_t>::max(), wrapNearestEven) == -1 &&
                convertScalar<uint64_t>(std::numeric_limits<int64_t>::min(), wrapNearestEven) ==
                    (uint64_t{1} << 63) &&
                convertScalar<uint64_t>(-1.0, wrapNearestEven) ==
                    std::numeric_limits<uint64_t>::max() &&
                convertScalar<int64_t>(twoTo63, wrapNearestEven) ==
                    std::numeric_limits<int64_t>::min() &&
                convertScalar<uint64_t>(
                    std::nextafter(twoTo64, std::numeric_limits<double>::infinity()),
                    wrapNearestEven) == 4096,
            "Wide modulo-wrap conversion mismatch.");

    require(
        convertScalar<int32_t>(std::complex<double>{3.5, 0.0}, identityNearestEven) == 4 &&
            convertScalar<double>(std::complex<double>{-2.25, -0.0}, identityTowardZero) == -2.25,
        "Zero-imaginary complex-to-real conversion mismatch.");
}

}  // namespace scalar_codec_test
