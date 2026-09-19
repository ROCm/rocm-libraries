// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <roc/host_numerics/scalar_binary_float.hpp>
#include <roc/host_numerics/scalar_conversion.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace roc::host_numerics::detail {
inline uint32_t readPackedBits(std::span<const std::byte> storage, uint64_t bitOffset,
                               uint32_t bitCount) {
    if (bitCount == 0 || bitCount > 32)
        throw std::invalid_argument("Packed scalar bit count must be in [1, 32].");

    uint32_t result = 0;
    for (uint32_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t absoluteBit = bitOffset + bit;
        const size_t byteIndex = static_cast<size_t>(absoluteBit / 8);
        if (byteIndex >= storage.size())
            throw std::out_of_range("Packed scalar read exceeds tensor storage.");
        const uint32_t sourceBit = static_cast<uint32_t>(absoluteBit % 8);
        const uint32_t value = (std::to_integer<uint8_t>(storage[byteIndex]) >> sourceBit) & 1U;
        result |= value << bit;
    }
    return result;
}

inline void writePackedBits(std::span<std::byte> storage, uint64_t bitOffset, uint32_t bitCount,
                            uint32_t value) {
    if (bitCount == 0 || bitCount > 32)
        throw std::invalid_argument("Packed scalar bit count must be in [1, 32].");

    for (uint32_t bit = 0; bit < bitCount; ++bit) {
        const uint64_t absoluteBit = bitOffset + bit;
        const size_t byteIndex = static_cast<size_t>(absoluteBit / 8);
        if (byteIndex >= storage.size())
            throw std::out_of_range("Packed scalar write exceeds tensor storage.");
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
    if (byteOffset > storage.size() || sizeof(T) > storage.size() - byteOffset)
        throw std::out_of_range("Native scalar read exceeds tensor storage.");
    T value;
    std::memcpy(&value, storage.data() + byteOffset, sizeof(T));
    return value;
}

template <typename T>
void writeNative(std::span<std::byte> storage, size_t byteOffset, const T& value) {
    if (byteOffset > storage.size() || sizeof(T) > storage.size() - byteOffset)
        throw std::out_of_range("Native scalar write exceeds tensor storage.");
    std::memcpy(storage.data() + byteOffset, &value, sizeof(T));
}

inline uint64_t bitOffset(ScalarType type, ptrdiff_t logicalOffset) {
    if (logicalOffset < 0) throw std::out_of_range("Tensor logical offset is negative.");
    const uint64_t bits = scalarTypeInfo(type).storageBits;
    const uint64_t offset = static_cast<uint64_t>(logicalOffset);
    if (offset > std::numeric_limits<uint64_t>::max() / bits)
        throw std::overflow_error("Tensor bit offset overflow.");
    return offset * bits;
}

inline void copyBitRange(std::span<const std::byte> source, uint64_t sourceBitOffset,
                         std::span<std::byte> destination, uint64_t destinationBitOffset,
                         uint16_t bitCount) {
    if (bitCount > 128) throw std::invalid_argument("Tensor scalar storage exceeds 128 bits.");
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
        return convertScalarValue<Target>(encoded != 0, options);
    else if constexpr (Type == ScalarType::Float16)
        return convertScalarValue<Target>(decodeFloat16(encoded), options);
    else if constexpr (Type == ScalarType::BFloat16)
        return convertScalarValue<Target>(decodeBFloat16(encoded), options);
    else if constexpr (IsBinaryFloatTypeV<Type>)
        return convertScalarValue<Target>(decodeBinaryFloatKnown<Type>(encoded), options);
    else if constexpr (Type == ScalarType::E8M0)
        return convertScalarValue<Target>(decodeE8M0(encoded), options);
    else if constexpr (Type == ScalarType::E8M0Zero)
        return convertScalarValue<Target>(decodeE8M0Zero(encoded), options);
    else
        return convertScalarValue<Target>(encoded, options);
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
        return convertScalarValue<Target>(signExtend(readPackedBits(storage, offsetBits, 4), 4),
                                          options);
    else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                       Type == ScalarType::Float6E3M2) {
        constexpr uint16_t storageBits = scalarTypeInfo(Type).storageBits;
        const uint32_t raw = readPackedBits(storage, offsetBits, storageBits);
        return convertScalarValue<Target>(decodeBinaryFloatKnown<Type>(raw), options);
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
    return visitScalarType(type, [&]<typename Tag>() {
        return decodeScalarKnown<Tag, Target>(storage, logicalOffset, options);
    });
}

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalar<Target>(type, storage, logicalOffset, implicitNativeConversionOptions());
}

template <typename Tag, typename Source>
auto encodeScalarValueKnown(Source source, const ScalarConversionOptions& options) {
    constexpr ScalarType Type = Tag::type;
    static_assert(isConcreteScalarType(Type));

    if constexpr (Type == ScalarType::Boolean)
        return convertScalarValue<bool>(source, options) ? uint8_t{1} : uint8_t{0};
    else if constexpr (Type == ScalarType::Float16)
        return encodeFloat16(convertScalarValue<float>(source, options));
    else if constexpr (Type == ScalarType::BFloat16)
        return options.bfloat16Rounding == BFloat16Rounding::Truncate
                   ? encodeBFloat16Truncated(convertScalarValue<float>(source, options))
                   : encodeBFloat16(convertScalarValue<float>(source, options));
    else if constexpr (Type == ScalarType::Int4)
        return static_cast<uint32_t>(convertToIntegerBits(source, 4, true, options));
    else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                       Type == ScalarType::Float6E3M2 || Type == ScalarType::Float8E4M3 ||
                       Type == ScalarType::Float8E5M2 || Type == ScalarType::Float8E4M3Fnuz ||
                       Type == ScalarType::Float8E5M2Fnuz || Type == ScalarType::E5M3 ||
                       Type == ScalarType::E4M3) {
        const uint32_t raw =
            encodeBinaryFloatKnown<Type>(convertScalarValue<float>(source, options));
        if constexpr (scalarTypeInfo(Type).storageBits == 8)
            return static_cast<uint8_t>(raw);
        else
            return raw;
    } else if constexpr (Type == ScalarType::E8M0)
        return encodeE8M0(convertScalarValue<float>(source, options));
    else if constexpr (Type == ScalarType::E8M0Zero)
        return encodeE8M0Zero(convertScalarValue<float>(source, options));
    else if constexpr (!std::is_void_v<typename Tag::Storage>)
        return convertScalarValue<typename Tag::Storage>(source, options);
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
}  // namespace roc::host_numerics::detail
