// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/scalar_storage.hpp>
#include <stdexcept>
#include <vector>

namespace {
using namespace roc::hostnumerics;
using enum IntegerRounding;
using enum IntegerOverflow;

int byteAt(std::span<const std::byte> storage, size_t index) {
    return std::to_integer<int>(storage[index]);
}

// Scalar storage tutorials. Run these examples with
// --gtest_filter=ScalarStorageTutorial.*.

TEST(ScalarStorageTutorial, EncodeFp8Values) {
    // Integer rounding and overflow are required, although only integer
    // destinations use them. Floating encodings use the defaulted choices.
    const ScalarConversionOptions options{NearestEven, Saturate};
    std::array<std::byte, 4> storage{};

    // Float8E4M3 has 3 fraction bits, so its neighbors of 1.0 are 0.875 and 1.125.
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

    // The storage holds whole elements only: two bytes hold four FP4 elements.
    EXPECT_THROW(decodeScalar<float>(ScalarType::Float4E2M1, storage, 4, options),
                 std::runtime_error);
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
}

TEST(ScalarStorage, RejectsElementsOutsideStorage) {
    const ScalarConversionOptions options{TowardZero, Identity};
    std::array<std::byte, 3> storage{std::byte{0xa5}, std::byte{0xa5}, std::byte{0xa5}};
    const std::array<std::byte, 3> original = storage;

    // Three bytes hold one Float16 element and four FP6 elements.
    EXPECT_THROW(encodeScalar(ScalarType::Float16, storage, 1, 1.0f, options), std::runtime_error);
    EXPECT_THROW(encodeScalar(ScalarType::Float6E2M3, storage, 4, 1.0f, options),
                 std::runtime_error);
    EXPECT_THROW(decodeScalar<float>(ScalarType::Float6E2M3, storage, SIZE_MAX, options),
                 std::runtime_error);
    EXPECT_THROW(encodeScalar(ScalarType::Float32, std::span<std::byte>{}, 0, 1.0f, options),
                 std::runtime_error);
    EXPECT_THROW(encodeScalar(ScalarType::Count, storage, 0, 1.0f, options), std::runtime_error);
    EXPECT_EQ(storage, original);

    encodeScalar(ScalarType::Float6E2M3, storage, 3, 1.0f, options);
    EXPECT_EQ(decodeScalar<float>(ScalarType::Float6E2M3, storage, 3, options), 1.0f);
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
    truncate.bfloat16Rounding = BFloat16Rounding::Truncate;
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
    // E8M0Zero keeps the exponent field of |value|, truncating the fraction.
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

TEST(ScalarStorageDeathTest, AssertsNonnegativeUnsignedScales) {
#ifndef NDEBUG
    const ScalarConversionOptions options{NearestEven, Saturate};
    std::array<std::byte, 1> storage{};
    for (const ScalarType type : {ScalarType::E4M3, ScalarType::E5M3, ScalarType::E8M0}) {
        EXPECT_DEATH(encodeScalar(type, storage, 0, -1.0f, options), "");
        EXPECT_DEATH(
            encodeScalar(type, storage, 0, -std::numeric_limits<float>::infinity(), options), "");
    }
#else
    GTEST_SKIP() << "Input requirements are checked only when assertions are enabled.";
#endif
}

// Covers lanes without the Python tests, which check every code against an independent decoder.
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
