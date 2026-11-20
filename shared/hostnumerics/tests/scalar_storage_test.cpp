// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <roc/hostnumerics/scalar_storage.hpp>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace roc::hostnumerics;
using enum IntegerRounding;
using enum IntegerOverflow;

int byteAt(std::span<const std::byte> storage, size_t index) {
    return std::to_integer<int>(storage[index]);
}

// Storage holding one element with the given code: one byte, or two bytes in
// host byte order for 16-bit types.
std::array<std::byte, 2> codeStorage(ScalarType type, uint16_t code) {
    std::array<std::byte, 2> storage{};
    if (scalarTypeInfo(type).storageBits == 16)
        std::memcpy(storage.data(), &code, sizeof(code));
    else
        storage[0] = static_cast<std::byte>(code);
    return storage;
}

// Scalar storage tutorials. Run these examples with
// --gtest_filter=ScalarStorageTutorial.*.

TEST(ScalarStorageTutorial, EncodeFp8Values) {
    // Integer rounding and overflow are required, although only integer
    // destinations use them. Floating encodings use the defaulted choices.
    const ScalarConversionOptions options{NearestEven, Saturate};
    std::array<std::byte, 4> storage{};

    // Float8E4M3 has 3 mantissa bits, so its neighbors of 1.0 are 0.9375 and 1.125.
    // 1.0625 is halfway between 1.0 and 1.125 and rounds to the even code, 1.0.
    encodeScalar(ScalarType::Float8E4M3, storage, 0, 1.0625, options);
    encodeScalar(ScalarType::Float8E4M3, storage, 1, 1.1, options);
    // E4M3 has no infinity, so large values clamp to its largest value, 448.
    encodeScalar(ScalarType::Float8E4M3, storage, 2, 1000.0, options);
    // A code's sign bit is its top bit.
    encodeScalar(ScalarType::Float8E4M3, storage, 3, -2.0, options);
    EXPECT_EQ(byteAt(storage, 0), 0x38);
    EXPECT_EQ(byteAt(storage, 1), 0x39);
    EXPECT_EQ(byteAt(storage, 2), 0x7e);
    EXPECT_EQ(byteAt(storage, 3), 0xc0);

    // Decoding to float or double is exact.
    EXPECT_EQ(decodeScalar<double>(ScalarType::Float8E4M3, storage, 0, options), 1.0);
    EXPECT_EQ(decodeScalar<double>(ScalarType::Float8E4M3, storage, 1, options), 1.125);
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float8E4M3, storage, 2, options), 448.0f);
    // An integer destination rounds the decoded value with the integer options.
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Float8E4M3, storage, 1, options), 1);
    EXPECT_EQ(decodeScalar<int8_t>(ScalarType::Float8E4M3, storage, 2, options), 127);
}

TEST(ScalarStorageTutorial, PackFp4Values) {
    // FP4 E2M1 elements take four bits, so byte k holds elements 2k (low four
    // bits) and 2k + 1 (high four bits). Its values are 0, 0.5, 1, 1.5, 2, 3,
    // 4 and 6 with either sign.
    const ScalarConversionOptions options{NearestEven, Saturate};
    std::vector<std::byte> storage(2);
    const std::array<double, 4> values{0.5, 3.0, -1.5, 5.0};
    for (size_t index = 0; index < values.size(); ++index)
        encodeScalar(ScalarType::Float4E2M1, storage, index, values[index], options);
    // 5 is halfway between 4 (code 6) and 6 (code 7) and rounds to the even code.
    EXPECT_EQ(byteAt(storage, 0), 0x51);
    EXPECT_EQ(byteAt(storage, 1), 0x6b);

    // Writing an element leaves the other element in its byte unchanged.
    encodeScalar(ScalarType::Float4E2M1, storage, 0, 6.0, options);
    EXPECT_EQ(byteAt(storage, 0), 0x57);
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float4E2M1, storage, 1, options), 3.0f);
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float4E2M1, storage, 3, options), 4.0f);
}

// The examples in the encodeScalar and decodeScalar header comment.
TEST(ScalarStorage, HeaderExamples) {
    const ScalarConversionOptions options{NearestEven, Saturate};
    std::array<std::byte, 2> storage{};
    encodeScalar(ScalarType::Float8E4M3, storage, 1, 1.0625, options);
    EXPECT_EQ(byteAt(storage, 1), 0x38);
    encodeScalar(ScalarType::Float8E4M3, storage, 0, 1000.0, options);
    EXPECT_EQ(byteAt(storage, 0), 0x7e);
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Float8E4M3, storage, 0, options), 448);
    encodeScalar(ScalarType::Int4, storage, 2, 9, options);
    EXPECT_EQ(byteAt(storage, 1), 0x37);

    std::array<std::byte, 1> fp4{};
    encodeScalar(ScalarType::Float4E2M1, fp4, 0, 0.5, options);
    encodeScalar(ScalarType::Float4E2M1, fp4, 1, 3.0, options);
    EXPECT_EQ(byteAt(fp4, 0), 0x51);

    // The FP6 example is checked in PackedElementsSpanBytes below.
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    encodeScalar(ScalarType::Float8E5M2, storage, 0, infinity, options);
    EXPECT_EQ(byteAt(storage, 0), 0x7c);
    EXPECT_EQ(decodeScalar<double>(ScalarType::Float8E5M2, storage, 0, options), infinity);
    encodeScalar(ScalarType::Float8E5M2, storage, 0, nan, options);
    EXPECT_EQ(byteAt(storage, 0), 0x7f);
    EXPECT_TRUE(std::isnan(decodeScalar<double>(ScalarType::Float8E5M2, storage, 0, options)));
    ScalarConversionOptions saturate = options;
    saturate.floatingOverflow = FloatingOverflow::SaturateFinite;
    encodeScalar(ScalarType::Float8E5M2, storage, 0, infinity, saturate);
    EXPECT_EQ(byteAt(storage, 0), 0x7b);
    EXPECT_EQ(decodeScalar<double>(ScalarType::Float8E5M2, storage, 0, options), 57344.0);
}

TEST(ScalarStorageDeathTest, AssertsValidTypeAndStorageBounds) {
#ifndef NDEBUG
    const ScalarConversionOptions options{TowardZero, Identity};
    std::array<std::byte, 3> storage{std::byte{0xa5}, std::byte{0xa5}, std::byte{0xa5}};

    // Three bytes hold one Float16 element and four FP6 elements.
    EXPECT_DEATH(encodeScalar(ScalarType::Float16, storage, 1, 1.0f, options), "index < elements");
    EXPECT_DEATH(encodeScalar(ScalarType::Float6E2M3, storage, 4, 1.0f, options),
                 "index < elements");
    EXPECT_DEATH(decodeScalar<float>(ScalarType::Float6E2M3, storage, SIZE_MAX, options),
                 "index < elements");
    EXPECT_DEATH(encodeScalar(ScalarType::Float32, std::span<std::byte>{}, 0, 1.0f, options),
                 "index < elements");
    EXPECT_DEATH(encodeScalar(ScalarType::Count, storage, 0, 1.0f, options),
                 "isConcreteScalarType");
    EXPECT_DEATH(decodeScalar<float>(ScalarType::Count, storage, 0, options),
                 "isConcreteScalarType");
#else
    GTEST_SKIP() << "Input requirements are checked only when assertions are enabled.";
#endif
}

TEST(ScalarStorage, PackedElementsSpanBytes) {
    const ScalarConversionOptions options{TowardZero, Identity};
    std::array<std::byte, 3> storage{std::byte{0xff}, std::byte{0xff}, std::byte{0xff}};
    // FP6 E3M2 element 1 occupies bits 6-11: the top two bits of byte 0 and the
    // low four bits of byte 1. The code for 1.0 is 0b001100.
    encodeScalar(ScalarType::Float6E3M2, storage, 1, 1.0f, options);
    EXPECT_EQ(byteAt(storage, 0), 0x3f);
    EXPECT_EQ(byteAt(storage, 1), 0xf3);
    EXPECT_EQ(byteAt(storage, 2), 0xff);
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float6E3M2, storage, 1, options), 1.0f);
    // All-ones FP6 E3M2 codes are -28, the most negative value.
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float6E3M2, storage, 0, options), -28.0f);
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float6E3M2, storage, 3, options), -28.0f);
    // Element 3 ends at the last bit of storage.
    encodeScalar(ScalarType::Float6E3M2, storage, 3, 1.0f, options);
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float6E3M2, storage, 3, options), 1.0f);
}

TEST(ScalarStorage, IntegerDestinationsUseIntegerOptions) {
    std::array<std::byte, 2> storage{};
    // Int4 holds [-8, 7].
    encodeScalar(ScalarType::Int4, storage, 0, 9, {TowardZero, Saturate});
    encodeScalar(ScalarType::Int4, storage, 1, 9, {TowardZero, ModuloWrap});
    encodeScalar(ScalarType::Int4, storage, 2, -2.5, {NearestEven, Identity});
    encodeScalar(ScalarType::Int4, storage, 3, -2.5, {TowardZero, Identity});
    const ScalarConversionOptions exact{TowardZero, Identity};
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Int4, storage, 0, exact), 7);
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Int4, storage, 1, exact), -7);
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Int4, storage, 2, exact), -2);
    EXPECT_EQ(decodeScalar<double>(ScalarType::Int4, storage, 3, exact), -2.0);

    encodeScalar(ScalarType::Int8, storage, 0, 128.5, {NearestEven, Saturate});
    encodeScalar(ScalarType::UInt8, storage, 1, 300, {TowardZero, ModuloWrap});
    EXPECT_EQ(decodeScalar<int64_t>(ScalarType::Int8, storage, 0, exact), 127);
    EXPECT_EQ(decodeScalar<uint64_t>(ScalarType::UInt8, storage, 1, exact), 44u);
    // Decoding to a narrower integer applies the options too.
    EXPECT_EQ(decodeScalar<int8_t>(ScalarType::UInt8, storage, 1, exact), 44);
    encodeScalar(ScalarType::UInt8, storage, 1, 200, exact);
    EXPECT_EQ(decodeScalar<int8_t>(ScalarType::UInt8, storage, 1, {TowardZero, Saturate}), 127);
    EXPECT_EQ(decodeScalar<int8_t>(ScalarType::UInt8, storage, 1, {TowardZero, ModuloWrap}), -56);

    // A floating code rounds to an integer destination.
    encodeScalar(ScalarType::Float16, storage, 0, 2.5f, exact);
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Float16, storage, 0, {NearestEven, Identity}), 2);
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Float16, storage, 0, {TowardZero, Identity}), 2);
    encodeScalar(ScalarType::Float16, storage, 0, -3.5f, exact);
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Float16, storage, 0, {NearestEven, Identity}), -4);
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Float16, storage, 0, {TowardZero, Identity}), -3);
}

TEST(ScalarStorage, BooleanAndComplexValues) {
    const ScalarConversionOptions options{NearestEven, Identity};
    std::array<std::byte, 16> storage{};
    // Boolean storage tests for nonzero and encodes true as 1.
    encodeScalar(ScalarType::Boolean, storage, 0, 0.25, options);
    EXPECT_EQ(byteAt(storage, 0), 1);
    storage[1] = std::byte{2};
    EXPECT_TRUE(decodeScalar<bool>(ScalarType::Boolean, storage, 1, options));
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::Boolean, storage, 1, options), 1);

    // Real values gain a zero imaginary part, and complex values convert by component.
    encodeScalar(ScalarType::ComplexFloat32, storage, 1, 1.5, options);
    EXPECT_EQ(decodeScalar<std::complex<double>>(ScalarType::ComplexFloat32, storage, 1, options),
              std::complex<double>(1.5, 0.0));
    encodeScalar(ScalarType::ComplexFloat64, storage, 0, std::complex<float>{0.5f, -2.0f}, options);
    EXPECT_EQ(decodeScalar<std::complex<float>>(ScalarType::ComplexFloat64, storage, 0, options),
              std::complex<float>(0.5f, -2.0f));

    // A complex value with a zero imaginary part converts to a real destination.
    encodeScalar(ScalarType::Float8E5M2, storage, 0, std::complex<double>{1.5, 0.0}, options);
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float8E5M2, storage, 0, options), 1.5f);
    encodeScalar(ScalarType::ComplexFloat64, storage, 0, std::complex<double>{3.5, 0.0}, options);
    EXPECT_EQ(decodeScalar<int32_t>(ScalarType::ComplexFloat64, storage, 0, options), 4);
}

TEST(ScalarStorage, NativeValuesRoundTrip) {
    const ScalarConversionOptions options{TowardZero, Identity};
    const auto check = [&]<NativeScalar T, size_t N>(const std::array<T, N>& values) {
        SCOPED_TRACE(scalarTypeName(nativeScalarType<T>));
        std::array<std::byte, sizeof(T) * N> storage{};
        for (size_t i = 0; i < N; ++i)
            encodeScalar(nativeScalarType<T>, storage, i, values[i], options);
        for (size_t i = 0; i < N; ++i) {
            const T decoded = decodeScalar<T>(nativeScalarType<T>, storage, i, options);
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

TEST(ScalarStorageDeathTest, AssertsRealConversionOfComplexValues) {
#ifndef NDEBUG
    std::array<std::byte, 1> storage{};
    const ScalarConversionOptions options{NearestEven, Identity};
    EXPECT_DEATH(
        encodeScalar(ScalarType::Float8E4M3, storage, 0, std::complex<double>{1.0, 1.0}, options),
        "");
    EXPECT_DEATH(encodeScalar(ScalarType::Int4, storage, 0, std::complex<float>{1, 1}, options),
                 "");
#else
    GTEST_SKIP() << "Input requirements are checked only when assertions are enabled.";
#endif
}

TEST(ScalarStorage, FloatingEncodingOptions) {
    std::array<std::byte, 2> storage{};
    const auto encode16 = [&](ScalarType type, auto value, const ScalarConversionOptions& options) {
        encodeScalar(type, storage, 0, value, options);
        return byteAt(storage, 0) | byteAt(storage, 1) << 8;
    };
    const auto encode8 = [&](ScalarType type, auto value, const ScalarConversionOptions& options) {
        encodeScalar(type, storage, 0, value, options);
        return byteAt(storage, 0);
    };
    using enum ScalarType;
    const ScalarConversionOptions defaults{NearestEven, Saturate};
    ScalarConversionOptions direct = defaults;
    direct.floatingSourcePrecision = FloatingSourcePrecision::Direct;
    ScalarConversionOptions saturate = defaults;
    saturate.floatingOverflow = FloatingOverflow::SaturateFinite;
    ScalarConversionOptions truncate = defaults;
    truncate.bfloat16Rounding = FloatingRounding::TowardZero;
    ScalarConversionOptions flush = defaults;
    flush.e8m0Underflow = E8M0Underflow::FlushBelowFloat32Normal;
    const double infinity = std::numeric_limits<double>::infinity();

    // The value rounds to float before BF16 by default, or rounds once directly.
    const double aboveTie = 1.0 + 0x1p-8 + 0x1p-30;
    EXPECT_EQ(encode16(BFloat16, aboveTie, defaults), 0x3f80);
    EXPECT_EQ(encode16(BFloat16, aboveTie, direct), 0x3f81);
    EXPECT_EQ(encode16(Float16, 1.0 + 0x1p-11 + 0x1p-30, direct), 0x3c01);
    EXPECT_EQ(encode8(Float8E4M3, 1.0 + 0x1p-4 + 0x1p-30, direct), 0x39);

    // FP16 overflows to infinity by default; FP8 E5M2 keeps infinity but clamps finite overflow.
    EXPECT_EQ(encode16(Float16, 65520.0, defaults), 0x7c00);
    EXPECT_EQ(encode16(Float16, 65520.0, saturate), 0x7bff);
    EXPECT_EQ(encode16(BFloat16, -infinity, saturate), 0xff7f);
    EXPECT_EQ(encode8(Float8E5M2, 1.0e10, defaults), 0x7b);
    EXPECT_EQ(encode8(Float8E5M2, infinity, defaults), 0x7c);
    EXPECT_EQ(encode8(Float8E5M2, infinity, saturate), 0x7b);

    // BF16 truncation rounds toward zero and encodes NaN as 0xffc1.
    EXPECT_EQ(encode16(BFloat16, 1.005859375f, defaults), 0x3f81);
    EXPECT_EQ(encode16(BFloat16, 1.005859375f, truncate), 0x3f80);
    EXPECT_EQ(encode16(BFloat16, -1.005859375f, truncate), 0xbf80);
    EXPECT_EQ(encode16(BFloat16, std::numeric_limits<float>::quiet_NaN(), truncate), 0xffc1);

    // E8M0 rounds to a power of two, choosing the even code at a tie. Values
    // below 2^-126 can instead flush to the smallest code, 2^-127.
    EXPECT_EQ(encode8(E8M0, 1.5f, defaults), 128);
    EXPECT_EQ(encode8(E8M0, 0x1.8p-127f, defaults), 0);
    EXPECT_EQ(encode8(E8M0, 0x1.cp-127f, defaults), 1);
    EXPECT_EQ(encode8(E8M0, 0x1.cp-127f, flush), 0);
    EXPECT_EQ(encode8(E8M0, 0.0f, defaults), 0);
    // E8M0Zero keeps the exponent field of |value|, truncating the mantissa.
    EXPECT_EQ(encode8(E8M0Zero, 1.75f, defaults), 127);
    EXPECT_EQ(encode8(E8M0Zero, -2.0f, defaults), 128);
    EXPECT_EQ(encode8(E8M0Zero, 0.0f, defaults), 0);
    EXPECT_EQ(encode8(E8M0Zero, infinity, defaults), 0xff);

    // Integer sources take the same floating path.
    EXPECT_EQ(encode16(BFloat16, int32_t{16'777'217}, defaults), 0x4b80);
}

TEST(ScalarStorage, UnsignedScalesEncodeNegativeZeroAsZero) {
    const ScalarConversionOptions options{NearestEven, Saturate};
    std::array<std::byte, 1> storage{std::byte{0x5a}};
    for (const ScalarType type : {ScalarType::E4M3, ScalarType::E5M3, ScalarType::E8M0}) {
        encodeScalar(type, storage, 0, -0.0f, options);
        EXPECT_EQ(byteAt(storage, 0), 0);
    }
}

TEST(ScalarStorage, DirectFloatingPrecisionPreservesWideIntegerRounding) {
    ScalarConversionOptions direct{NearestEven, Saturate};
    direct.floatingSourcePrecision = FloatingSourcePrecision::Direct;
    std::array<std::byte, 2> storage{};
    const auto bfloatCode = [&](auto value, const ScalarConversionOptions& options) {
        encodeScalar(ScalarType::BFloat16, storage, 0, value, options);
        uint16_t code;
        std::memcpy(&code, storage.data(), sizeof(code));
        return code;
    };

    // Adjacent BF16 values in [2^60, 2^61) are 2^53 apart. Double loses the
    // unit on either side of these midpoints; direct encoding must keep it.
    const int64_t evenTie = (int64_t{1} << 60) + (int64_t{1} << 52);
    const int64_t oddTie = evenTie + (int64_t{1} << 53);
    EXPECT_EQ(bfloatCode(evenTie - 1, direct), 0x5d80);
    EXPECT_EQ(bfloatCode(evenTie, direct), 0x5d80);
    EXPECT_EQ(bfloatCode(evenTie + 1, direct), 0x5d81);
    EXPECT_EQ(bfloatCode(oddTie - 1, direct), 0x5d81);
    EXPECT_EQ(bfloatCode(oddTie, direct), 0x5d82);
    EXPECT_EQ(bfloatCode(-evenTie - 1, direct), 0xdd81);
    EXPECT_EQ(bfloatCode(-oddTie + 1, direct), 0xdd81);
    const uint64_t unsignedTie = (uint64_t{1} << 63) + (uint64_t{1} << 55);
    EXPECT_EQ(bfloatCode(unsignedTie + 1, direct), 0x5f01);
    EXPECT_EQ(bfloatCode(UINT64_MAX, direct), 0x5f80);
    EXPECT_EQ(bfloatCode(INT64_MIN, direct), 0xdf00);
    EXPECT_EQ(bfloatCode(evenTie + 1, {NearestEven, Saturate}), 0x5d80);

    ScalarConversionOptions truncate = direct;
    truncate.bfloat16Rounding = FloatingRounding::TowardZero;
    const int64_t belowPowerOfTwo = (int64_t{1} << 60) - 1;
    EXPECT_EQ(bfloatCode(belowPowerOfTwo, truncate), 0x5d7f);
    EXPECT_EQ(bfloatCode(-belowPowerOfTwo, truncate), 0xdd7f);

    // E8M0's tie chooses the even exponent code, while E8M0Zero truncates
    // to the lower power of two. Neither may lose a unit before that choice.
    encodeScalar(ScalarType::E8M0, storage, 0, (int64_t{3} << 59) - 1, direct);
    EXPECT_EQ(byteAt(storage, 0), 187);
    encodeScalar(ScalarType::E8M0Zero, storage, 0, belowPowerOfTwo, direct);
    EXPECT_EQ(byteAt(storage, 0), 186);
}

TEST(ScalarStorageDeathTest, AssertsNonnegativeUnsignedScales) {
#ifndef NDEBUG
    const ScalarConversionOptions options{NearestEven, Saturate};
    std::array<std::byte, 1> storage{};
    for (const ScalarType type : {ScalarType::E4M3, ScalarType::E5M3, ScalarType::E8M0}) {
        EXPECT_DEATH(encodeScalar(type, storage, 0, -1.0f, options), "< 0");
        // This negative double rounds to negative zero under Float32 precision.
        EXPECT_DEATH(encodeScalar(type, storage, 0, -0x1p-200, options), "< 0");
        EXPECT_DEATH(
            encodeScalar(type, storage, 0, -std::numeric_limits<float>::infinity(), options),
            "< 0");
    }
#else
    GTEST_SKIP() << "Input requirements are checked only when assertions are enabled.";
#endif
}

TEST(ScalarStorage, KnownFloatingEncodings) {
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
    const ScalarConversionOptions options{NearestEven, Saturate};
    for (const auto& example : examples) {
        SCOPED_TRACE(::testing::Message()
                     << scalarTypeName(example.type) << " code " << example.code);
        const auto expected = codeStorage(example.type, example.code);
        EXPECT_EQ(decodeScalar<float>(example.type, expected, 0, options), example.value);
        std::array<std::byte, 2> actual{};
        encodeScalar(example.type, actual, 0, example.value, options);
        EXPECT_EQ(actual, expected);
    }
}

TEST(ScalarStorage, KnownNanAndNegativeZeroEncodings) {
    using enum ScalarType;
    const ScalarConversionOptions options{NearestEven, Saturate};

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
            EXPECT_TRUE(std::isnan(decodeScalar<float>(type, codeStorage(type, code), 0, options)))
                << "code " << code;
        std::array<std::byte, 2> actual{};
        encodeScalar(type, actual, 0, std::numeric_limits<float>::quiet_NaN(), options);
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
        encodeScalar(type, actual, 0, -0.0f, options);
        EXPECT_EQ(actual, codeStorage(type, code));
    }
}

TEST(ScalarStorage, EveryFloatingCodeRoundTrips) {
    const ScalarConversionOptions defaults{NearestEven, Saturate};
    ScalarConversionOptions direct = defaults;
    direct.floatingSourcePrecision = FloatingSourcePrecision::Direct;
    for (const ScalarType type :
         {ScalarType::Float4E2M1, ScalarType::Float6E2M3, ScalarType::Float6E3M2,
          ScalarType::Float8E4M3, ScalarType::Float8E5M2, ScalarType::Float8E4M3Fnuz,
          ScalarType::Float8E5M2Fnuz, ScalarType::E4M3, ScalarType::E5M3, ScalarType::E8M0,
          ScalarType::E8M0Zero, ScalarType::Float16, ScalarType::BFloat16}) {
        SCOPED_TRACE(scalarTypeName(type));
        const uint32_t bits = scalarTypeInfo(type).storageBits;
        // E4M3 ignores the top bit of its byte.
        const uint32_t valueMask = type == ScalarType::E4M3 ? 0x7f : (1u << bits) - 1;
        std::array<std::byte, 2> storage{};
        size_t failures = 0;
        for (uint32_t code = 0; code < (1u << bits) && failures < 5; ++code) {
            storage[0] = static_cast<std::byte>(code & 0xff);
            storage[1] = static_cast<std::byte>(code >> 8);
            const float value = decodeScalar<float>(type, storage, 0, defaults);
            for (const ScalarConversionOptions& options : {defaults, direct}) {
                storage = {};
                encodeScalar(type, storage, 0, value, options);
                const uint32_t encoded = static_cast<uint32_t>(byteAt(storage, 0)) |
                                         static_cast<uint32_t>(byteAt(storage, 1)) << 8;
                const bool same = std::isnan(value)
                                      ? std::isnan(decodeScalar<float>(type, storage, 0, defaults))
                                      : encoded == (code & valueMask);
                if (!same) ++failures;
                EXPECT_TRUE(same) << "code " << code << " decodes to " << value
                                  << " and encodes to " << encoded;
            }
        }
    }
}

// The default BF16 options use a faster rounding path than other options.
TEST(ScalarStorage, DefaultBFloat16RoundingMatchesDirectRounding) {
    const ScalarConversionOptions defaults{NearestEven, Saturate};
    ScalarConversionOptions direct = defaults;
    direct.floatingSourcePrecision = FloatingSourcePrecision::Direct;
    std::array<std::byte, 2> fast{};
    std::array<std::byte, 2> general{};
    size_t failures = 0;
    for (uint32_t code = 0; code < 0xffff && failures < 5; ++code) {
        // Each code and the values around its midpoint with the next code.
        const float lower = std::bit_cast<float>(code << 16);
        const float upper = std::bit_cast<float>((code + 1) << 16);
        if (std::isnan(lower) || std::isnan(upper)) continue;
        const float midpoint = std::isinf(upper)
                                   ? std::bit_cast<float>((code << 16) | 0x8000)
                                   : static_cast<float>((static_cast<double>(lower) + upper) / 2);
        for (const float value :
             {lower, std::nextafter(midpoint, lower), midpoint, std::nextafter(midpoint, upper)}) {
            encodeScalar(ScalarType::BFloat16, fast, 0, value, defaults);
            encodeScalar(ScalarType::BFloat16, general, 0, value, direct);
            if (fast != general) ++failures;
            EXPECT_EQ(fast, general) << "value " << value;
        }
    }
}
}  // namespace
