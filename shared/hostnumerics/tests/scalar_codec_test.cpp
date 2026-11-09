// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <roc/hostnumerics/scalar_storage.hpp>
#include <roc/hostnumerics/tensor.hpp>
#include <span>
#include <type_traits>

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

TEST(ScalarCodecs, NativeValuesRoundTrip) {
    using namespace roc::hostnumerics;
    const ScalarConversionOptions options{
        IntegerRounding::TowardZero,     IntegerOverflow::Identity,
        FloatingRounding::NearestEven,   FloatingSourcePrecision::Float32,
        FloatingOverflow::FormatDefault, E8M0Underflow::Nearest};
    const auto check = [&]<NativeScalar T, size_t N>(const std::array<T, N>& values) {
        SCOPED_TRACE(scalarTypeName(nativeScalarType<T>));
        std::array<std::byte, sizeof(T) * N> storage{};
        for (size_t i = 0; i < N; ++i)
            detail::encodeScalar(nativeScalarType<T>, storage, i, values[i], options);
        for (size_t i = 0; i < N; ++i) {
            const T decoded = detail::decodeScalar<T>(nativeScalarType<T>, storage, i, options);
            EXPECT_EQ(decoded, values[i]) << "element " << i;
            if constexpr (std::is_floating_point_v<T>)
                EXPECT_EQ(std::signbit(decoded), std::signbit(values[i]));
        }
    };
    check(std::array{false, true});
    const auto integers = [&]<typename T>() {
        // In particular, the 64-bit endpoints must not pass through double.
        check(std::array{std::numeric_limits<T>::lowest(), T{0}, T{1},
                         std::numeric_limits<T>::max()});
    };
    integers.template operator()<int8_t>();
    integers.template operator()<int16_t>();
    integers.template operator()<int32_t>();
    integers.template operator()<int64_t>();
    integers.template operator()<uint8_t>();
    integers.template operator()<uint16_t>();
    integers.template operator()<uint32_t>();
    integers.template operator()<uint64_t>();
    const auto floating = [&]<typename T>() {
        // Native storage preserves subnormals, signed zero and the full mantissa.
        check(std::array{std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max(),
                         std::numeric_limits<T>::min(), std::numeric_limits<T>::denorm_min(), -T{0},
                         std::nextafter(T{1}, T{2}), std::numeric_limits<T>::infinity()});
    };
    floating.template operator()<float>();
    floating.template operator()<double>();
    check(std::array{std::complex<float>{0.5f, -2.0f},
                     std::complex<float>{std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::denorm_min()}});
    check(std::array{
        std::complex<double>{0.5, -2.0},
        std::complex<double>{std::numeric_limits<double>::max(), std::nextafter(1.0, 2.0)}});
}

TEST(ScalarCodecs, KnownFloatingEncodings) {
    using namespace roc::hostnumerics;
    // Fixed code/value pairs check numerical meaning independently of round trips.
    struct Example {
        ScalarType type;
        uint16_t code;
        float value;
    };
    using enum ScalarType;
    const Example examples[]{
        {Float4E2M1, 0x03, 1.5f},     {Float6E2M3, 0x0c, 1.5f}, {Float6E3M2, 0x0e, 1.5f},
        {Float8E4M3, 0x3c, 1.5f},     {Float8E5M2, 0x3e, 1.5f}, {Float8E4M3Fnuz, 0x44, 1.5f},
        {Float8E5M2Fnuz, 0x42, 1.5f}, {E4M3, 0x3c, 1.5f},       {E5M3, 0x7c, 1.5f},
        {E8M0, 0x80, 2.0f},           {E8M0Zero, 0x80, 2.0f},   {Float16, 0x3e00, 1.5f},
        {BFloat16, 0x3fc0, 1.5f},
    };
    const ScalarConversionOptions options{
        IntegerRounding::NearestEven,    IntegerOverflow::Saturate,
        FloatingRounding::NearestEven,   FloatingSourcePrecision::Float32,
        FloatingOverflow::FormatDefault, E8M0Underflow::Nearest};
    for (const auto& example : examples) {
        SCOPED_TRACE(scalarTypeName(example.type));
        std::array<std::byte, 2> expected{};
        if (scalarTypeInfo(example.type).storageBits == 16)
            std::memcpy(expected.data(), &example.code, sizeof(example.code));
        else
            expected[0] = static_cast<std::byte>(example.code);
        EXPECT_EQ(detail::decodeScalar<float>(example.type, expected, 0, options), example.value);
        std::array<std::byte, 2> actual{};
        detail::encodeScalar(example.type, actual, 0, example.value, options);
        EXPECT_EQ(actual, expected);
    }
}
