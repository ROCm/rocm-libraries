// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <roc/hostnumerics/scalar.hpp>
#include <roc/hostnumerics/scalar_small_float.hpp>
#include <roc/hostnumerics/scalar_type.hpp>
#include <span>
#include <type_traits>
#include <utility>

namespace roc::hostnumerics::detail {
// encodeScalar and decodeScalar support every concrete ScalarType: Boolean,
// signed and unsigned integers, floating types, scales and complex types.
// ScalarType::Count and unnamed enum values are invalid.
//
// Storage helpers require a concrete ScalarType, a nonnegative element offset,
// and storage containing the whole element or bit range. Bit-offset arithmetic
// must fit uint64_t. Builds without NDEBUG assert these preconditions. Tensor
// callers establish storage bounds from their layouts before entering this layer.
// Packed elements use consecutive low-to-high bits and can straddle bytes;
// byte-addressable elements use the host's byte order.
//
// Decoding an encoded value returns the value after the selected conversions,
// not necessarily the original input: FP4 maps 5.0 to 4.0 with the default
// floating options, and Int4 maps 9 to 7 with IntegerOverflow::Saturate.
// A finite value round-trips unchanged when it is exactly representable in the
// storage type, any selected intermediate and Target, and no option changes it
// (for example, E8M0 underflow flushing). NaNs must be checked with isnan;
// encoding may canonicalize their payloads, and NaN is not equal to itself.
//
// With options specifying NearestEven integer rounding, Saturate integer overflow
// and the default floating choices (NearestEven, Float32, FormatDefault, Nearest):
//   encodeScalar(Float4E2M1, storage, 0, 0.5, options), followed by element 1 = 3.0,
//     turns one zero byte into 0x51 (low code 1, high code 5).
//   encodeScalar(Float6E3M2, storage, 1, 1.0, options)
//     turns {0xff, 0xff, 0xff} into {0x3f, 0xf3, 0xff}: code 0b001100 occupies
//     bits 6-11, changing parts of two bytes and preserving all other bits.
//   encodeScalar(Float8E5M2, storage, 0, infinity, options) writes 0x7c;
//     a quiet NaN writes 0x7f. decodeScalar<double> returns infinity/NaN respectively.
//     SaturateFinite instead encodes infinity as 0x7b (57344).

// Storage decoding truncates and wraps when returning a native C++ value.
inline constexpr ScalarConversionOptions implicitNativeConversionOptions() {
    return {IntegerRounding::TowardZero,     IntegerOverflow::ModuloWrap,
            FloatingRounding::NearestEven,   FloatingSourcePrecision::Float32,
            FloatingOverflow::FormatDefault, E8M0Underflow::Nearest};
}

// No-options storage conversion follows the native policy for byte-addressable
// integers. Packed Int4 saturates because it has no native C++ cast whose
// behavior can define the conversion.
inline constexpr ScalarConversionOptions implicitStorageConversionOptions(ScalarType destination) {
    const IntegerOverflow overflow =
        destination == ScalarType::Int4 ? IntegerOverflow::Saturate : IntegerOverflow::ModuloWrap;
    return {IntegerRounding::TowardZero,     overflow,
            FloatingRounding::NearestEven,   FloatingSourcePrecision::Float32,
            FloatingOverflow::FormatDefault, E8M0Underflow::Nearest};
}

inline uint32_t readPackedBits(std::span<const std::byte> storage, uint64_t bitOffset,
                               uint32_t bitCount) {
    assert(bitCount > 0 && bitCount <= 32);
    assert(bitOffset <= std::numeric_limits<uint64_t>::max() - (bitCount - 1));
    assert((bitOffset + bitCount - 1) / 8 < storage.size());

    uint32_t result = 0;
    for (uint32_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t absoluteBit = bitOffset + bit;
        const size_t byteIndex = static_cast<size_t>(absoluteBit / 8);
        const uint32_t sourceBit = static_cast<uint32_t>(absoluteBit % 8);
        const uint32_t value = (std::to_integer<uint8_t>(storage[byteIndex]) >> sourceBit) & 1U;
        result |= value << bit;
    }
    return result;
}

inline void writePackedBits(std::span<std::byte> storage, uint64_t bitOffset, uint32_t bitCount,
                            uint32_t value) {
    assert(bitCount > 0 && bitCount <= 32);
    assert(bitOffset <= std::numeric_limits<uint64_t>::max() - (bitCount - 1));
    assert((bitOffset + bitCount - 1) / 8 < storage.size());

    for (uint32_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t absoluteBit = bitOffset + bit;
        const size_t byteIndex = static_cast<size_t>(absoluteBit / 8);
        const uint8_t mask = static_cast<uint8_t>(1U << (absoluteBit % 8));
        uint8_t byte = std::to_integer<uint8_t>(storage[byteIndex]);
        if ((value >> bit) & 1U)
            byte |= mask;
        else
            byte &= static_cast<uint8_t>(~mask);
        storage[byteIndex] = static_cast<std::byte>(byte);
    }
}

template <typename T>
T readNative(std::span<const std::byte> storage, size_t byteOffset) {
    assert(byteOffset <= storage.size() && sizeof(T) <= storage.size() - byteOffset);
    T value;
    std::memcpy(&value, storage.data() + byteOffset, sizeof(T));
    return value;
}

template <typename T>
void writeNative(std::span<std::byte> storage, size_t byteOffset, const T& value) {
    assert(byteOffset <= storage.size() && sizeof(T) <= storage.size() - byteOffset);
    std::memcpy(storage.data() + byteOffset, &value, sizeof(T));
}

inline uint64_t bitOffset(ScalarType type, ptrdiff_t logicalOffset) {
    assert(isConcreteScalarType(type));
    assert(logicalOffset >= 0);
    const uint64_t bits = scalarTypeInfo(type).storageBits;
    const uint64_t offset = static_cast<uint64_t>(logicalOffset);
    assert(offset <= std::numeric_limits<uint64_t>::max() / bits);
    return offset * bits;
}

inline void copyBitRange(std::span<const std::byte> source, uint64_t sourceBitOffset,
                         std::span<std::byte> destination, uint64_t destinationBitOffset,
                         uint16_t bitCount) {
    assert(bitCount <= 128);
    assert(bitCount == 0 ||
           (sourceBitOffset <= std::numeric_limits<uint64_t>::max() - (bitCount - 1) &&
            (sourceBitOffset + bitCount - 1) / 8 < source.size()));
    assert(bitCount == 0 ||
           (destinationBitOffset <= std::numeric_limits<uint64_t>::max() - (bitCount - 1) &&
            (destinationBitOffset + bitCount - 1) / 8 < destination.size()));
    std::array<bool, 128> bits{};
    for (uint16_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t sourcePosition = sourceBitOffset + bit;
        const uint8_t sourceByte =
            std::to_integer<uint8_t>(source[static_cast<size_t>(sourcePosition / 8)]);
        bits[bit] = ((sourceByte >> (sourcePosition % 8)) & 1U) != 0;
    }
    for (uint16_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t destinationPosition = destinationBitOffset + bit;
        std::byte& destinationByte = destination[static_cast<size_t>(destinationPosition / 8)];
        const uint8_t mask = static_cast<uint8_t>(1U << (destinationPosition % 8));
        uint8_t value = std::to_integer<uint8_t>(destinationByte);
        value = bits[bit] ? static_cast<uint8_t>(value | mask)
                          : static_cast<uint8_t>(value & static_cast<uint8_t>(~mask));
        destinationByte = static_cast<std::byte>(value);
    }
}

inline int64_t signExtend(uint32_t value, uint32_t bits) {
    const uint32_t sign = 1U << (bits - 1U);
    return static_cast<int32_t>((value ^ sign) - sign);
}

template <typename Tag, typename Target>
    requires(!std::is_void_v<typename Tag::Storage>)
Target decodeScalarValueKnown(typename Tag::Storage encoded,
                              const ScalarConversionOptions& options) {
    constexpr ScalarType Type = Tag::type;
    static_assert(isConcreteScalarType(Type));
    if constexpr (Type == ScalarType::Boolean)
        return convertScalar<Target>(encoded != 0, options.integerConversion());
    else if constexpr (Type == ScalarType::Float16)
        return convertScalar<Target>(decodeFloat16(encoded), options.integerConversion());
    else if constexpr (Type == ScalarType::BFloat16)
        return convertScalar<Target>(decodeBFloat16(encoded), options.integerConversion());
    else if constexpr (IsSmallFloatTypeV<Type>)
        return convertScalar<Target>(decodeSmallFloatKnown<Type>(encoded),
                                     options.integerConversion());
    else if constexpr (Type == ScalarType::E8M0)
        return convertScalar<Target>(decodeE8M0(encoded), options.integerConversion());
    else if constexpr (Type == ScalarType::E8M0Zero)
        return convertScalar<Target>(decodeE8M0Zero(encoded), options.integerConversion());
    else
        return convertScalar<Target>(encoded, options.integerConversion());
}

template <typename Tag, typename Target>
Target decodeScalarKnown(std::span<const std::byte> storage, ptrdiff_t logicalOffset,
                         const ScalarConversionOptions& options) {
    constexpr ScalarType Type = Tag::type;
    static_assert(isConcreteScalarType(Type));
    const uint64_t offsetBits = bitOffset(Type, logicalOffset);
    const size_t offsetBytes = static_cast<size_t>(offsetBits / 8);

    if constexpr (!std::is_void_v<typename Tag::Storage>)
        return decodeScalarValueKnown<Tag, Target>(
            readNative<typename Tag::Storage>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int4)
        return convertScalar<Target>(signExtend(readPackedBits(storage, offsetBits, 4), 4),
                                     options.integerConversion());
    else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                       Type == ScalarType::Float6E3M2) {
        constexpr uint16_t storageBits = scalarTypeInfo(Type).storageBits;
        const uint32_t raw = readPackedBits(storage, offsetBits, storageBits);
        return convertScalar<Target>(decodeSmallFloatKnown<Type>(raw), options.integerConversion());
    } else
        static_assert(AlwaysFalseV<Tag>, "Unhandled ScalarType decoding.");
}

template <typename Tag, typename Target>
Target decodeScalarKnown(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalarKnown<Tag, Target>(storage, logicalOffset,
                                          implicitNativeConversionOptions());
}

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset,
                    const ScalarConversionOptions& options) {
    assert(isConcreteScalarType(type));
    return visitScalarType(type, [&]<typename Tag>() {
        return decodeScalarKnown<Tag, Target>(storage, logicalOffset, options);
    });
}

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalar<Target>(type, storage, logicalOffset, implicitNativeConversionOptions());
}

template <typename Source>
double floatingEncodingSource(Source source, const ScalarConversionOptions& options) {
    return options.floatingSourcePrecision == FloatingSourcePrecision::Direct
               ? convertScalar<double>(source, options.integerConversion())
               : static_cast<double>(convertScalar<float>(source, options.integerConversion()));
}

template <typename Tag, typename Source>
auto encodeScalarValueKnown(Source source, const ScalarConversionOptions& options) {
    constexpr ScalarType Type = Tag::type;
    static_assert(isConcreteScalarType(Type));

    if constexpr (Type == ScalarType::Boolean)
        return convertScalar<bool>(source, options.integerConversion()) ? uint8_t{1} : uint8_t{0};
    else if constexpr (Type == ScalarType::Float16) {
        if (options.floatingSourcePrecision == FloatingSourcePrecision::Direct ||
            options.floatingOverflow == FloatingOverflow::SaturateFinite)
            return encodeFloat16(floatingEncodingSource(source, options), options.floatingOverflow);
        return encodeFloat16(convertScalar<float>(source, options.integerConversion()));
    } else if constexpr (Type == ScalarType::BFloat16) {
        if (options.floatingSourcePrecision == FloatingSourcePrecision::Direct ||
            options.floatingOverflow == FloatingOverflow::SaturateFinite)
            return encodeBFloat16(floatingEncodingSource(source, options), options.bfloat16Rounding,
                                  options.floatingOverflow);
        return options.bfloat16Rounding == FloatingRounding::TowardZero
                   ? encodeBFloat16Truncated(
                         convertScalar<float>(source, options.integerConversion()))
                   : encodeBFloat16(convertScalar<float>(source, options.integerConversion()));
    } else if constexpr (Type == ScalarType::Int4) {
        // Complex sources must have a zero imaginary part.
        if constexpr (RuntimeIsComplexV<Source>)
            return encodeScalarValueKnown<Tag>(source.real(), options);
        else
            return static_cast<uint32_t>(convertToIntegerBits(
                source, 4, true, options.integerRounding, options.integerOverflow));
    } else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                         Type == ScalarType::Float6E3M2 || Type == ScalarType::Float8E4M3 ||
                         Type == ScalarType::Float8E5M2 || Type == ScalarType::Float8E4M3Fnuz ||
                         Type == ScalarType::Float8E5M2Fnuz || Type == ScalarType::E5M3 ||
                         Type == ScalarType::E4M3) {
        const uint32_t raw =
            options.floatingSourcePrecision == FloatingSourcePrecision::Direct ||
                    options.floatingOverflow == FloatingOverflow::SaturateFinite
                ? encodeSmallFloatKnown<Type>(floatingEncodingSource(source, options),
                                              options.floatingOverflow)
                : encodeSmallFloatKnown<Type>(
                      convertScalar<float>(source, options.integerConversion()));
        if constexpr (scalarTypeInfo(Type).storageBits == 8)
            return static_cast<uint8_t>(raw);
        else
            return raw;
    } else if constexpr (Type == ScalarType::E8M0) {
        if (options.floatingSourcePrecision == FloatingSourcePrecision::Direct ||
            options.e8m0Underflow == E8M0Underflow::FlushBelowFloat32Normal)
            return encodeE8M0(floatingEncodingSource(source, options), options.e8m0Underflow);
        return encodeE8M0(convertScalar<float>(source, options.integerConversion()));
    } else if constexpr (Type == ScalarType::E8M0Zero) {
        if (options.floatingSourcePrecision == FloatingSourcePrecision::Direct)
            return encodeE8M0Zero(floatingEncodingSource(source, options));
        return encodeE8M0Zero(convertScalar<float>(source, options.integerConversion()));
    } else if constexpr (!std::is_void_v<typename Tag::Storage>)
        return convertScalar<typename Tag::Storage>(source, options.integerConversion());
    else
        static_assert(AlwaysFalseV<Tag>, "Unhandled ScalarType encoding.");
}

template <typename Tag, typename Source>
auto encodeScalarValueKnown(Source source) {
    return encodeScalarValueKnown<Tag>(std::move(source),
                                       implicitStorageConversionOptions(Tag::type));
}

template <typename Tag, typename Source>
void encodeScalarKnown(std::span<std::byte> storage, ptrdiff_t logicalOffset, Source source,
                       const ScalarConversionOptions& options) {
    constexpr ScalarType Type = Tag::type;
    constexpr uint16_t storageBits = scalarTypeInfo(Type).storageBits;
    const uint64_t offsetBits = bitOffset(Type, logicalOffset);
    auto encoded = encodeScalarValueKnown<Tag>(std::move(source), options);
    if constexpr (storageBits % 8 == 0)
        writeNative(storage, static_cast<size_t>(offsetBits / 8), encoded);
    else
        writePackedBits(storage, offsetBits, storageBits, static_cast<uint32_t>(encoded));
}

template <typename Tag, typename Source>
void encodeScalarKnown(std::span<std::byte> storage, ptrdiff_t logicalOffset, Source source) {
    encodeScalarKnown<Tag>(storage, logicalOffset, std::move(source),
                           implicitStorageConversionOptions(Tag::type));
}

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source, const ScalarConversionOptions& options) {
    assert(isConcreteScalarType(type));
    visitScalarType(type, [&]<typename Tag>() {
        encodeScalarKnown<Tag>(storage, logicalOffset, std::move(source), options);
    });
}

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source) {
    encodeScalar(type, storage, logicalOffset, std::move(source),
                 implicitStorageConversionOptions(type));
}
}  // namespace roc::hostnumerics::detail
