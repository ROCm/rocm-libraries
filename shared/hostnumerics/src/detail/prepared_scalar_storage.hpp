// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/scalar_storage.hpp>

namespace roc::hostnumerics::detail {
// Internal codecs for offsets in a Tensor whose constructor has validated the
// full layout and backing storage, or a fixed scalar scratch buffer. These omit
// structural checks; numerical conversions follow convertScalar's caller preconditions.
inline void preparedCopyBitRange(std::span<const std::byte> source, uint64_t sourceBitOffset,
                                 std::span<std::byte> destination, uint64_t destinationBitOffset,
                                 uint16_t bitCount) {
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

inline uint64_t preparedBitOffset(ScalarType type, ptrdiff_t offset) noexcept {
    return static_cast<uint64_t>(offset) * scalarTypeInfo(type).storageBits;
}
inline uint32_t preparedReadPackedBits(std::span<const std::byte> storage, uint64_t offset,
                                       uint32_t count) noexcept {
    uint32_t result = 0;
    for (uint32_t bit = 0; bit < count; ++bit) {
        const uint64_t position = offset + bit;
        result |= ((std::to_integer<uint8_t>(storage[static_cast<size_t>(position / 8)]) >>
                    (position % 8)) &
                   1U)
                  << bit;
    }
    return result;
}
inline void preparedWritePackedBits(std::span<std::byte> storage, uint64_t offset, uint32_t count,
                                    uint32_t value) noexcept {
    for (uint32_t bit = 0; bit < count; ++bit) {
        const uint64_t position = offset + bit;
        auto& destination = storage[static_cast<size_t>(position / 8)];
        const auto mask = static_cast<uint8_t>(1U << (position % 8));
        const auto byte = std::to_integer<uint8_t>(destination);
        destination = static_cast<std::byte>((value >> bit) & 1U ? byte | mask : byte & ~mask);
    }
}

template <typename Tag, typename Target>
Target preparedDecodeScalarKnown(std::span<const std::byte> storage, ptrdiff_t offset,
                                 const ScalarConversionOptions& options) {
    constexpr auto type = Tag::type;
    const uint64_t bits = static_cast<uint64_t>(offset) * scalarTypeInfo(type).storageBits;
    if constexpr (!std::is_void_v<typename Tag::Storage>) {
        typename Tag::Storage encoded;
        std::memcpy(&encoded, storage.data() + static_cast<size_t>(offset) * sizeof(encoded),
                    sizeof(encoded));
        return decodeScalarValueKnown<Tag, Target>(encoded, options);
    } else if constexpr (type == ScalarType::Int4) {
        return convertScalarValue<Target>(signExtend(preparedReadPackedBits(storage, bits, 4), 4),
                                          options);
    } else {
        return convertScalarValue<Target>(decodeBinaryFloatKnown<type>(preparedReadPackedBits(
                                              storage, bits, scalarTypeInfo(type).storageBits)),
                                          options);
    }
}
template <typename Tag, typename Target>
Target preparedDecodeScalarKnown(std::span<const std::byte> storage, ptrdiff_t offset) {
    return preparedDecodeScalarKnown<Tag, Target>(storage, offset,
                                                  implicitNativeConversionOptions());
}
template <typename Tag, typename Source>
void preparedEncodeScalarKnown(std::span<std::byte> storage, ptrdiff_t offset, Source value,
                               const ScalarConversionOptions& options) {
    constexpr auto count = scalarTypeInfo(Tag::type).storageBits;
    const uint64_t bits = static_cast<uint64_t>(offset) * count;
    const auto encoded = encodeScalarValueKnown<Tag>(value, options);
    if constexpr (count % 8 == 0)
        std::memcpy(storage.data() + static_cast<size_t>(bits / 8), &encoded, sizeof(encoded));
    else
        preparedWritePackedBits(storage, bits, count, static_cast<uint32_t>(encoded));
}
template <typename Tag, typename Source>
void preparedEncodeScalarKnown(std::span<std::byte> storage, ptrdiff_t offset, Source value) {
    preparedEncodeScalarKnown<Tag>(storage, offset, value,
                                   implicitStorageConversionOptions(Tag::type));
}
}  // namespace roc::hostnumerics::detail
