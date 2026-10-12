// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace roc::hostnumerics {
enum class ScalarCategory : uint8_t {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
    Complex,
    Count,
};

// Dense table index for every supported scalar encoding. Count is one-past-last
// and exists only to size the metadata table; it is not a scalar encoding.
enum class ScalarType : uint16_t {
    Boolean,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float4E2M1,
    Float6E2M3,
    Float6E3M2,
    Float8E4M3,
    Float8E5M2,
    // FNUZ means finite, NaN, unsigned zero: no infinities or negative zero.
    // The former negative-zero byte (0x80) encodes NaN; exponent biases are 8 and 16.
    Float8E4M3Fnuz,
    Float8E5M2Fnuz,
    // Unsigned float: four exponent bits and three fraction bits in an eight-bit slot.
    // The top bit is ignored when decoding and zero when encoding; 0x7f is NaN.
    // Float8E4M3 instead uses the top bit as a sign bit.
    E4M3,
    // Unsigned float: five exponent bits and three fraction bits; 0xff is NaN.
    E5M3,
    // Unsigned powers of two: byte b represents 2^(b-127), except 0xff is NaN.
    E8M0,
    // E8M0 with byte 0x00 representing zero instead of 2^-127.
    E8M0Zero,
    Float16,
    BFloat16,
    Float32,
    Float64,
    ComplexFloat32,
    ComplexFloat64,
    Count,
};

struct ScalarTypeInfo {
    std::string_view name;
    ScalarCategory category;
    // Bits per element in tensor storage: 4 for Int4, 6 for Float6, 8 for Boolean
    // and E4M3, and 64 for ComplexFloat32 (two 32-bit components).
    uint16_t storageBits;
    // Exponent and fraction widths are per real component; zero for integers and Boolean.
    uint8_t exponentBits;
    // Explicitly stored fraction bits per real component, excluding any implicit leading bit.
    uint8_t mantissaBits;
    // Bias subtracted from an encoded exponent; zero when exponentBits is zero.
    int16_t exponentBias;
    // Whether each real component can represent negative values.
    bool isSigned;
    // Whether the encoding has at least one representation for NaN.
    bool supportsNaN;
    // Whether the encoding has representations for positive and negative infinity.
    bool supportsInfinity;

    bool isPacked() const {
        return storageBits % 8 != 0;
    }
};

inline constexpr size_t scalarTypeCount = static_cast<size_t>(ScalarType::Count);

// Reject Count and unnamed values produced by casts from integers.
inline constexpr bool isConcreteScalarType(ScalarType type) {
    return static_cast<size_t>(type) < scalarTypeCount;
}

namespace detail {
constexpr ScalarTypeInfo getSignedInteger(std::string_view name, uint16_t bits) {
    return {name, ScalarCategory::SignedInteger, bits, 0, 0, 0, true, false, false};
}

constexpr ScalarTypeInfo getUnsignedInteger(std::string_view name, uint16_t bits) {
    return {name, ScalarCategory::UnsignedInteger, bits, 0, 0, 0, false, false, false};
}

struct FloatingPointOptions {
    bool isSigned = true;
    bool supportsNaN = true;
    bool supportsInfinity = true;
};

constexpr ScalarTypeInfo getFloatingPoint(std::string_view name, uint16_t bits,
                                          uint8_t exponentBits, uint8_t mantissaBits,
                                          int16_t exponentBias, FloatingPointOptions options = {}) {
    return {name,
            ScalarCategory::FloatingPoint,
            bits,
            exponentBits,
            mantissaBits,
            exponentBias,
            options.isSigned,
            options.supportsNaN,
            options.supportsInfinity};
}

constexpr ScalarTypeInfo getComplex(std::string_view name, ScalarTypeInfo component) {
    component.name = name;
    component.category = ScalarCategory::Complex;
    component.storageBits *= 2;
    return component;
}
}  // namespace detail

inline constexpr std::array<ScalarTypeInfo, scalarTypeCount> scalarTypeInfos{{
    {"bool", ScalarCategory::Boolean, 8, 0, 0, 0, false, false, false},
    detail::getSignedInteger("i4", 4),
    detail::getSignedInteger("i8", 8),
    detail::getSignedInteger("i16", 16),
    detail::getSignedInteger("i32", 32),
    detail::getSignedInteger("i64", 64),
    detail::getUnsignedInteger("u8", 8),
    detail::getUnsignedInteger("u16", 16),
    detail::getUnsignedInteger("u32", 32),
    detail::getUnsignedInteger("u64", 64),
    detail::getFloatingPoint("f4e2m1", 4, 2, 1, 1,
                             {.supportsNaN = false, .supportsInfinity = false}),
    detail::getFloatingPoint("f6e2m3", 6, 2, 3, 1,
                             {.supportsNaN = false, .supportsInfinity = false}),
    detail::getFloatingPoint("f6e3m2", 6, 3, 2, 3,
                             {.supportsNaN = false, .supportsInfinity = false}),
    detail::getFloatingPoint("f8e4m3", 8, 4, 3, 7, {.supportsInfinity = false}),
    detail::getFloatingPoint("f8e5m2", 8, 5, 2, 15),
    detail::getFloatingPoint("f8e4m3fnuz", 8, 4, 3, 8, {.supportsInfinity = false}),
    detail::getFloatingPoint("f8e5m2fnuz", 8, 5, 2, 16, {.supportsInfinity = false}),
    detail::getFloatingPoint("e4m3", 8, 4, 3, 7, {.isSigned = false, .supportsInfinity = false}),
    detail::getFloatingPoint("e5m3", 8, 5, 3, 15, {.isSigned = false, .supportsInfinity = false}),
    detail::getFloatingPoint("e8m0", 8, 8, 0, 127, {.isSigned = false, .supportsInfinity = false}),
    detail::getFloatingPoint("e8m0_zero", 8, 8, 0, 127,
                             {.isSigned = false, .supportsInfinity = false}),
    detail::getFloatingPoint("f16", 16, 5, 10, 15),
    detail::getFloatingPoint("bf16", 16, 8, 7, 127),
    detail::getFloatingPoint("f32", 32, 8, 23, 127),
    detail::getFloatingPoint("f64", 64, 11, 52, 1023),
    detail::getComplex("c64", detail::getFloatingPoint("f32", 32, 8, 23, 127)),
    detail::getComplex("c128", detail::getFloatingPoint("f64", 64, 11, 52, 1023)),
}};

// The caller must supply a concrete scalar type, including in release builds.
inline constexpr const ScalarTypeInfo& scalarTypeInfo(ScalarType type) {
    assert(isConcreteScalarType(type));
    return scalarTypeInfos[static_cast<size_t>(type)];
}

inline constexpr std::string_view scalarTypeName(ScalarType type) {
    return scalarTypeInfo(type).name;
}

}  // namespace roc::hostnumerics
