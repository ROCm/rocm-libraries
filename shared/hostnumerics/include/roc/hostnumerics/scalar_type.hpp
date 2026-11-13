// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

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

namespace detail {
template <typename T>
struct NativeScalarType;

template <>
struct NativeScalarType<bool> {
    static constexpr ScalarType value = ScalarType::Boolean;
};
template <>
struct NativeScalarType<uint8_t> {
    static constexpr ScalarType value = ScalarType::UInt8;
};
template <>
struct NativeScalarType<int8_t> {
    static constexpr ScalarType value = ScalarType::Int8;
};
template <>
struct NativeScalarType<uint16_t> {
    static constexpr ScalarType value = ScalarType::UInt16;
};
template <>
struct NativeScalarType<int16_t> {
    static constexpr ScalarType value = ScalarType::Int16;
};
template <>
struct NativeScalarType<uint32_t> {
    static constexpr ScalarType value = ScalarType::UInt32;
};
template <>
struct NativeScalarType<int32_t> {
    static constexpr ScalarType value = ScalarType::Int32;
};
template <>
struct NativeScalarType<uint64_t> {
    static constexpr ScalarType value = ScalarType::UInt64;
};
template <>
struct NativeScalarType<int64_t> {
    static constexpr ScalarType value = ScalarType::Int64;
};
template <>
struct NativeScalarType<float> {
    static constexpr ScalarType value = ScalarType::Float32;
};
template <>
struct NativeScalarType<double> {
    static constexpr ScalarType value = ScalarType::Float64;
};
template <>
struct NativeScalarType<std::complex<float>> {
    static constexpr ScalarType value = ScalarType::ComplexFloat32;
};
template <>
struct NativeScalarType<std::complex<double>> {
    static constexpr ScalarType value = ScalarType::ComplexFloat64;
};
}  // namespace detail

// Tests whether a C++ type has a native mapping, ignoring cv/ref qualifiers.
// Integer mappings use the fixed-width aliases (int8_t through uint64_t), whose
// underlying types vary by platform. Use int64_t rather than long or long long
// for portable callers. An integer type is mapped only if it is the same type
// as one of those aliases; plain char is not mapped.
// C++23's optional <stdfloat> types, including float16_t and bfloat16_t, are not mapped.
template <typename T>
concept NativeScalar = requires { detail::NativeScalarType<std::remove_cvref_t<T>>::value; };

// Unsupported types fail the NativeScalar constraint at compile time.
template <NativeScalar T>
inline constexpr ScalarType nativeScalarType =
    detail::NativeScalarType<std::remove_cvref_t<T>>::value;

namespace detail {
// The C++ type that holds one encoded value of Type. Signed integers, float,
// double and complex values use their native type. Every other whole-byte
// encoding, including bool, half, bfloat16 and the 8-bit floating-point and
// scale formats, uses an unsigned integer of the same width. If the storage
// width is not divisible by 8 bits, ScalarTag::Storage is void: elements must
// be accessed as packed bits rather than individual C++ objects.
template <ScalarType Type>
constexpr auto scalarStorageIdentity() {
    constexpr uint16_t bits = scalarTypeInfo(Type).storageBits;
    if constexpr (bits % 8 != 0) {
        return std::type_identity<void>{};
    } else if constexpr (Type == ScalarType::Int8) {
        return std::type_identity<int8_t>{};
    } else if constexpr (Type == ScalarType::Int16) {
        return std::type_identity<int16_t>{};
    } else if constexpr (Type == ScalarType::Int32) {
        return std::type_identity<int32_t>{};
    } else if constexpr (Type == ScalarType::Int64) {
        return std::type_identity<int64_t>{};
    } else if constexpr (Type == ScalarType::Float32) {
        return std::type_identity<float>{};
    } else if constexpr (Type == ScalarType::Float64) {
        return std::type_identity<double>{};
    } else if constexpr (Type == ScalarType::ComplexFloat32) {
        return std::type_identity<std::complex<float>>{};
    } else if constexpr (Type == ScalarType::ComplexFloat64) {
        return std::type_identity<std::complex<double>>{};
    } else if constexpr (bits == 8) {
        return std::type_identity<uint8_t>{};
    } else if constexpr (bits == 16) {
        return std::type_identity<uint16_t>{};
    } else if constexpr (bits == 32) {
        return std::type_identity<uint32_t>{};
    } else {
        static_assert(bits == 64);
        return std::type_identity<uint64_t>{};
    }
}

template <ScalarType TypeValue>
struct ScalarTag {
    static_assert(isConcreteScalarType(TypeValue));
    static constexpr ScalarType type = TypeValue;
    static constexpr uint16_t storageBits = scalarTypeInfo(TypeValue).storageBits;
    using Storage = typename decltype(scalarStorageIdentity<TypeValue>())::type;
};
}  // namespace detail

// Dispatches a type-templated operation on a runtime ScalarType by calling
// visitor.template operator()<Tag>(args...). Tag supplies type, storageBits,
// and Storage (void when the storage width is not divisible by 8 bits).
// The visitor is instantiated for every tag and must return the same type for each.
// Visitors and arguments are forwarded; value, reference, and void returns are preserved.
// Invalid types throw std::runtime_error before invoking the visitor.
template <typename Visitor, typename... Args>
decltype(auto) visitScalarType(ScalarType type, Visitor&& visitor, Args&&... args) {
    auto dispatch = [&]<ScalarType TypeValue>() -> decltype(auto) {
        return std::forward<Visitor>(visitor).template operator()<detail::ScalarTag<TypeValue>>(
            std::forward<Args>(args)...);
    };
    switch (type) {
        case ScalarType::Boolean:
            return dispatch.template operator()<ScalarType::Boolean>();
        case ScalarType::Int4:
            return dispatch.template operator()<ScalarType::Int4>();
        case ScalarType::Int8:
            return dispatch.template operator()<ScalarType::Int8>();
        case ScalarType::Int16:
            return dispatch.template operator()<ScalarType::Int16>();
        case ScalarType::Int32:
            return dispatch.template operator()<ScalarType::Int32>();
        case ScalarType::Int64:
            return dispatch.template operator()<ScalarType::Int64>();
        case ScalarType::UInt8:
            return dispatch.template operator()<ScalarType::UInt8>();
        case ScalarType::UInt16:
            return dispatch.template operator()<ScalarType::UInt16>();
        case ScalarType::UInt32:
            return dispatch.template operator()<ScalarType::UInt32>();
        case ScalarType::UInt64:
            return dispatch.template operator()<ScalarType::UInt64>();
        case ScalarType::Float4E2M1:
            return dispatch.template operator()<ScalarType::Float4E2M1>();
        case ScalarType::Float6E2M3:
            return dispatch.template operator()<ScalarType::Float6E2M3>();
        case ScalarType::Float6E3M2:
            return dispatch.template operator()<ScalarType::Float6E3M2>();
        case ScalarType::Float8E4M3:
            return dispatch.template operator()<ScalarType::Float8E4M3>();
        case ScalarType::Float8E5M2:
            return dispatch.template operator()<ScalarType::Float8E5M2>();
        case ScalarType::Float8E4M3Fnuz:
            return dispatch.template operator()<ScalarType::Float8E4M3Fnuz>();
        case ScalarType::Float8E5M2Fnuz:
            return dispatch.template operator()<ScalarType::Float8E5M2Fnuz>();
        case ScalarType::E4M3:
            return dispatch.template operator()<ScalarType::E4M3>();
        case ScalarType::E5M3:
            return dispatch.template operator()<ScalarType::E5M3>();
        case ScalarType::E8M0:
            return dispatch.template operator()<ScalarType::E8M0>();
        case ScalarType::E8M0Zero:
            return dispatch.template operator()<ScalarType::E8M0Zero>();
        case ScalarType::Float16:
            return dispatch.template operator()<ScalarType::Float16>();
        case ScalarType::BFloat16:
            return dispatch.template operator()<ScalarType::BFloat16>();
        case ScalarType::Float32:
            return dispatch.template operator()<ScalarType::Float32>();
        case ScalarType::Float64:
            return dispatch.template operator()<ScalarType::Float64>();
        case ScalarType::ComplexFloat32:
            return dispatch.template operator()<ScalarType::ComplexFloat32>();
        case ScalarType::ComplexFloat64:
            return dispatch.template operator()<ScalarType::ComplexFloat64>();
        case ScalarType::Count:
            break;
    }
    throw std::runtime_error("Invalid ScalarType: " + std::to_string(static_cast<unsigned>(type)) +
                             ".");
}
}  // namespace roc::hostnumerics
