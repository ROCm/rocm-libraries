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
#include <utility>
#include <vector>

#include "_scalar_codec_test_support.hpp"

namespace {
// Storage holding one element with the given code: one byte, or two bytes in
// host byte order for 16-bit types.
std::array<std::byte, 2> codeStorage(roc::hostnumerics::ScalarType type, uint16_t code) {
    std::array<std::byte, 2> storage{};
    if (roc::hostnumerics::scalarTypeInfo(type).storageBits == 16)
        std::memcpy(storage.data(), &code, sizeof(code));
    else
        storage[0] = static_cast<std::byte>(code);
    return storage;
}

constexpr roc::hostnumerics::ScalarConversionOptions knownEncodingOptions{
    roc::hostnumerics::IntegerRounding::NearestEven,
    roc::hostnumerics::IntegerOverflow::Saturate,
    roc::hostnumerics::FloatingRounding::NearestEven,
    roc::hostnumerics::FloatingSourcePrecision::Float32,
    roc::hostnumerics::FloatingOverflow::FormatDefault,
    roc::hostnumerics::E8M0Underflow::Nearest};
}  // namespace

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
    // Fixed code/value pairs check numerical meaning independently of round
    // trips, which cannot detect a wrong code shared by the encoder and decoder.
    // Each format lists a normal value, its largest finite value, its smallest
    // positive value and any infinities.
    struct Example {
        ScalarType type;
        uint16_t code;
        float value;
    };
    using enum ScalarType;
    const float infinity = std::numeric_limits<float>::infinity();
    const Example examples[]{
        {Float4E2M1, 0x03, 1.5f},
        {Float4E2M1, 0x07, 6.0f},
        {Float4E2M1, 0x01, 0.5f},
        {Float6E2M3, 0x0c, 1.5f},
        {Float6E2M3, 0x1f, 7.5f},
        {Float6E2M3, 0x01, 0x1p-3f},
        {Float6E3M2, 0x0e, 1.5f},
        {Float6E3M2, 0x1f, 28.0f},
        {Float6E3M2, 0x01, 0x1p-4f},
        {Float8E4M3, 0x3c, 1.5f},
        {Float8E4M3, 0x7e, 448.0f},
        {Float8E4M3, 0x01, 0x1p-9f},
        {Float8E5M2, 0x3e, 1.5f},
        {Float8E5M2, 0x7b, 57344.0f},
        {Float8E5M2, 0x01, 0x1p-16f},
        {Float8E5M2, 0x7c, infinity},
        {Float8E5M2, 0xfc, -infinity},
        {Float8E4M3Fnuz, 0x44, 1.5f},
        {Float8E4M3Fnuz, 0x7f, 240.0f},
        {Float8E4M3Fnuz, 0x01, 0x1p-10f},
        {Float8E5M2Fnuz, 0x42, 1.5f},
        {Float8E5M2Fnuz, 0x7f, 57344.0f},
        {Float8E5M2Fnuz, 0x01, 0x1p-17f},
        {E4M3, 0x3c, 1.5f},
        {E4M3, 0x7e, 448.0f},
        {E4M3, 0x01, 0x1p-9f},
        {E5M3, 0x7c, 1.5f},
        {E5M3, 0xfe, 114688.0f},
        {E5M3, 0x01, 0x1p-17f},
        {E8M0, 0x80, 2.0f},
        {E8M0, 0xfe, 0x1p127f},
        {E8M0, 0x00, 0x1p-127f},
        {E8M0Zero, 0x80, 2.0f},
        {E8M0Zero, 0xfe, 0x1p127f},
        {E8M0Zero, 0x01, 0x1p-126f},
        {Float16, 0x3e00, 1.5f},
        {Float16, 0x7bff, 65504.0f},
        {Float16, 0x0001, 0x1p-24f},
        {Float16, 0x7c00, infinity},
        {BFloat16, 0x3fc0, 1.5f},
        {BFloat16, 0x7f7f, 0x1.fep127f},
        {BFloat16, 0x0001, 0x1p-133f},
        {BFloat16, 0x7f80, infinity},
    };
    const ScalarConversionOptions& options = knownEncodingOptions;
    for (const auto& example : examples) {
        SCOPED_TRACE(::testing::Message()
                     << scalarTypeName(example.type) << " code " << example.code);
        const auto expected = codeStorage(example.type, example.code);
        EXPECT_EQ(detail::decodeScalar<float>(example.type, expected, 0, options), example.value);
        std::array<std::byte, 2> actual{};
        detail::encodeScalar(example.type, actual, 0, example.value, options);
        EXPECT_EQ(actual, expected);
    }
}

TEST(ScalarCodecs, KnownNanAndNegativeZeroEncodings) {
    using namespace roc::hostnumerics;
    using enum ScalarType;
    const ScalarConversionOptions& options = knownEncodingOptions;

    // Every listed code decodes to NaN, and encoding NaN produces the first.
    struct NanCodes {
        ScalarType type;
        std::vector<uint16_t> codes;
    };
    const NanCodes nanCodes[]{
        {Float8E4M3, {0x7f, 0xff}},
        {Float8E5M2, {0x7f, 0x7d, 0x7e, 0xfd, 0xff}},
        {Float8E4M3Fnuz, {0x80}},
        {Float8E5M2Fnuz, {0x80}},
        {E4M3, {0x7f}},
        {E5M3, {0xff}},
        {E8M0, {0xff}},
        {E8M0Zero, {0xff}},
        {Float16, {0x7e00, 0x7c01, 0xfe00}},
        {BFloat16, {0x7fc0, 0x7f81, 0xffc0}},
    };
    for (const auto& [type, codes] : nanCodes) {
        SCOPED_TRACE(scalarTypeName(type));
        for (const uint16_t code : codes)
            EXPECT_TRUE(
                std::isnan(detail::decodeScalar<float>(type, codeStorage(type, code), 0, options)))
                << "code " << code;
        std::array<std::byte, 2> actual{};
        detail::encodeScalar(type, actual, 0, std::numeric_limits<float>::quiet_NaN(), options);
        EXPECT_EQ(actual, codeStorage(type, codes.front()));
    }

    // Negative zero keeps its sign where the format has one. FNUZ formats use
    // the negative-zero code for NaN, so negative zero encodes as zero.
    const std::pair<ScalarType, uint16_t> negativeZeros[]{
        {Float4E2M1, 0x08},     {Float6E2M3, 0x20}, {Float6E3M2, 0x20},
        {Float8E4M3, 0x80},     {Float8E5M2, 0x80}, {Float8E4M3Fnuz, 0x00},
        {Float8E5M2Fnuz, 0x00}, {Float16, 0x8000},  {BFloat16, 0x8000},
    };
    for (const auto& [type, code] : negativeZeros) {
        SCOPED_TRACE(scalarTypeName(type));
        std::array<std::byte, 2> actual{};
        detail::encodeScalar(type, actual, 0, -0.0f, options);
        EXPECT_EQ(actual, codeStorage(type, code));
    }
}
