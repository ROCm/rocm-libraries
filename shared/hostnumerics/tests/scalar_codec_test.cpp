// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <roc/hostnumerics/scalar_storage.hpp>
#include <roc/hostnumerics/tensor.hpp>
#include <span>

#include "_scalar_codec_test_support.hpp"

TEST(ScalarCodecs, ScalarTypeInfoContract) {
    scalar_codec_test::testScalarTypeInfoContract();
}

TEST(ScalarCodecs, IntegerConversionPrimitives) {
    scalar_codec_test::testIntegerConversionPrimitives();
}

TEST(ScalarCodecsDeathTest, IntegerConversionAssertsInputRequirements) {
#ifndef NDEBUG
    using namespace roc::hostnumerics;
    using enum IntegerRounding;
    using enum IntegerOverflow;
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Identity requires the rounded value to fit.
    EXPECT_DEATH(convertToInteger<int8_t>(int32_t{300}, TowardZero, Identity), "");
    EXPECT_DEATH(convertToInteger<int8_t>(127.5, NearestEven, Identity), "");
    // ModuloWrap requires a finite input.
    EXPECT_DEATH(convertToInteger<int8_t>(infinity, TowardZero, ModuloWrap), "");
    // NaN is outside the domain of every policy.
    for (const auto overflow : {Identity, Saturate, ModuloWrap})
        EXPECT_DEATH(convertToInteger<int8_t>(nan, NearestEven, overflow), "");
#else
    GTEST_SKIP() << "Input requirements are checked only when assertions are enabled.";
#endif
}

TEST(ScalarCodecs, NearestEvenIgnoresHostRoundingMode) {
    scalar_codec_test::testNearestEvenIgnoresHostRoundingMode();
}

TEST(ScalarCodecs, IntegerCodecPolicies) {
    scalar_codec_test::testIntegerCodecPolicies();
}

TEST(ScalarCodecs, ScalarEncodings) {
    scalar_codec_test::testScalarEncodings();
}

TEST(ScalarCodecs, DirectDoubleAndExplicitFloatIntermediate) {
    using namespace roc::hostnumerics;
    ScalarConversionOptions options{
        IntegerRounding::NearestEven,    IntegerOverflow::Saturate,
        FloatingRounding::NearestEven,   FloatingSourcePrecision::Direct,
        FloatingOverflow::FormatDefault, E8M0Underflow::Nearest};
    const auto encode = [&](ScalarType type, double value) {
        Tensor result(type, Shape{1});
        result.storeFrom({0}, value, options);
        return result.loadAs<double>({0});
    };
    EXPECT_EQ(encode(ScalarType::BFloat16, 1.0 + 0x1p-8 + 0x1p-30), 1.0 + 0x1p-7);
    options.floatingSourcePrecision = FloatingSourcePrecision::Float32;
    EXPECT_EQ(encode(ScalarType::BFloat16, 1.0 + 0x1p-8 + 0x1p-30), 1.0);
    options.floatingSourcePrecision = FloatingSourcePrecision::Direct;
    EXPECT_EQ(encode(ScalarType::Float16, 1.0 + 0x1p-11 + 0x1p-30), 1.0 + 0x1p-10);
    EXPECT_EQ(encode(ScalarType::Float8E4M3, 1.0 + 0x1p-4 + 0x1p-30), 1.125);
    EXPECT_EQ(encode(ScalarType::E8M0, 1.5 - 0x1p-30), 1.0);
}

// Scalar storage tutorials: explicit policies and byte-level results.
TEST(ScalarStorageTutorial, PackFp4AndFp6) {
    using namespace roc::hostnumerics;
    const ScalarConversionOptions options{
        IntegerRounding::NearestEven,    IntegerOverflow::Saturate,
        FloatingRounding::NearestEven,   FloatingSourcePrecision::Float32,
        FloatingOverflow::FormatDefault, E8M0Underflow::Nearest};
    std::array<std::byte, 1> fp4{};
    detail::encodeScalar(ScalarType::Float4E2M1, fp4, 0, 0.5, options);
    detail::encodeScalar(ScalarType::Float4E2M1, fp4, 1, 3.0, options);
    EXPECT_EQ(fp4[0], std::byte{0x51});
    EXPECT_EQ(detail::decodeScalar<float>(ScalarType::Float4E2M1, fp4, 1, options), 3.0f);

    // FP6 element 1 occupies bits 6-11, so it changes parts of two bytes.
    std::array<std::byte, 3> fp6{std::byte{0xff}, std::byte{0xff}, std::byte{0xff}};
    detail::encodeScalar(ScalarType::Float6E3M2, fp6, 1, 1.0, options);
    EXPECT_EQ(fp6[0], std::byte{0x3f});
    EXPECT_EQ(fp6[1], std::byte{0xf3});
    EXPECT_EQ(fp6[2], std::byte{0xff});
    EXPECT_EQ(detail::decodeScalar<float>(ScalarType::Float6E3M2, fp6, 1, options), 1.0f);
    // Element 3 ends at the last storage bit and is also valid.
    detail::encodeScalar(ScalarType::Float6E3M2, fp6, 3, 1.0, options);
    EXPECT_EQ(detail::decodeScalar<float>(ScalarType::Float6E3M2, fp6, 3, options), 1.0f);
}

TEST(ScalarStorageTutorial, EncodeSpecialValues) {
    using namespace roc::hostnumerics;
    ScalarConversionOptions options{
        IntegerRounding::NearestEven,    IntegerOverflow::Saturate,
        FloatingRounding::NearestEven,   FloatingSourcePrecision::Float32,
        FloatingOverflow::FormatDefault, E8M0Underflow::Nearest};
    std::array<std::byte, 1> storage{};
    const double infinity = std::numeric_limits<double>::infinity();
    detail::encodeScalar(ScalarType::Float8E5M2, storage, 0, infinity, options);
    EXPECT_EQ(storage[0], std::byte{0x7c});
    EXPECT_EQ(detail::decodeScalar<double>(ScalarType::Float8E5M2, storage, 0, options), infinity);
    detail::encodeScalar(ScalarType::Float8E5M2, storage, 0,
                         std::numeric_limits<double>::quiet_NaN(), options);
    EXPECT_EQ(storage[0], std::byte{0x7f});
    EXPECT_TRUE(
        std::isnan(detail::decodeScalar<double>(ScalarType::Float8E5M2, storage, 0, options)));
    // Finite saturation clamps infinity but leaves NaN as NaN.
    options.floatingOverflow = FloatingOverflow::SaturateFinite;
    detail::encodeScalar(ScalarType::Float8E5M2, storage, 0, infinity, options);
    EXPECT_EQ(storage[0], std::byte{0x7b});
    EXPECT_EQ(detail::decodeScalar<double>(ScalarType::Float8E5M2, storage, 0, options), 57344.0);
}

TEST(ScalarCodecsDeathTest, AssertsStoragePreconditions) {
#ifndef NDEBUG
    using namespace roc::hostnumerics;
    const ScalarConversionOptions options{
        IntegerRounding::NearestEven,    IntegerOverflow::Saturate,
        FloatingRounding::NearestEven,   FloatingSourcePrecision::Float32,
        FloatingOverflow::FormatDefault, E8M0Underflow::Nearest};
    std::array<std::byte, 3> storage{};
    EXPECT_DEATH(detail::encodeScalar(ScalarType::Float16, storage, 1, 1.0, options), "byteOffset");
    EXPECT_DEATH(detail::encodeScalar(ScalarType::Float6E3M2, storage, 4, 1.0, options),
                 "storage.size");
    EXPECT_DEATH(detail::decodeScalar<float>(ScalarType::Float6E3M2, storage, -1, options),
                 "logicalOffset >= 0");
    EXPECT_DEATH(
        detail::decodeScalar<float>(ScalarType::Float32, std::span<const std::byte>{}, 0, options),
        "byteOffset");
    EXPECT_DEATH(detail::encodeScalar(ScalarType::Count, storage, 0, 1.0, options),
                 "isConcreteScalarType");
#else
    GTEST_SKIP() << "Input requirements are checked only when assertions are enabled.";
#endif
}
