// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <roc/hostnumerics/detail/binary_float.hpp>
#include <roc/hostnumerics/integer_conversion.hpp>
#include <roc/hostnumerics/scalar_type.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace roc::hostnumerics {
enum class BFloat16Rounding : uint8_t {
    NearestEven,
    // Round toward zero. NaN encodes as 0xffc1, matching TensileLite and rocRoller.
    Truncate,
};

// The precision of a value just before it is rounded to FP16, BF16, a narrower
// floating type or E8M0.
enum class FloatingSourcePrecision : uint8_t {
    // Round to float first, so a double source can round twice:
    // 1 + 2^-8 + 2^-30 becomes 1 + 2^-8 as a float, and that tie becomes 1.0 in BF16.
    // Doubles beyond the float range become infinite.
    Float32,
    // Round once from the source value: 1 + 2^-8 + 2^-30 becomes 1 + 2^-7 in BF16.
    Direct,
};

// Encoding of values beyond the largest finite FP16, BF16 or narrower floating value.
enum class FloatingOverflow : uint8_t {
    // FP16 and BF16 round overflow to infinity, as IEEE 754 does. Narrower types
    // clamp finite values to the largest finite value of the same sign. They
    // encode an infinite value as infinity if the type has one (of the narrower
    // types, only Float8E5M2) and otherwise as the largest finite value.
    FormatDefault,
    // Clamp finite and infinite values to the largest finite value of the same sign.
    SaturateFinite,
};

// Encoding of E8M0 values below 2^-126, the smallest normal float.
enum class E8M0Underflow : uint8_t {
    // Round to the nearest power of two, as for larger values. Values below
    // the smallest code's value, 2^-127, including zero, encode as that code.
    Nearest,
    // Encode every value below 2^-126 as the smallest code, 2^-127.
    FlushBelowFloat32Normal,
};

// Choices for converting a value to or from an encoded scalar. Integer rounding
// and overflow apply whenever the destination is an integer, including packed
// Int4. The remaining choices apply only when encoding FP16, BF16, narrower
// floating types and E8M0 scales.
struct ScalarConversionOptions {
    constexpr ScalarConversionOptions(
        IntegerRounding rounding, IntegerOverflow overflow,
        BFloat16Rounding bfloatRounding = BFloat16Rounding::NearestEven,
        FloatingSourcePrecision precision = FloatingSourcePrecision::Float32,
        FloatingOverflow floatingRange = FloatingOverflow::FormatDefault,
        E8M0Underflow scaleUnderflow = E8M0Underflow::Nearest) noexcept
        : integerRounding(rounding),
          integerOverflow(overflow),
          bfloat16Rounding(bfloatRounding),
          floatingSourcePrecision(precision),
          floatingOverflow(floatingRange),
          e8m0Underflow(scaleUnderflow) {}

    IntegerRounding integerRounding;
    IntegerOverflow integerOverflow;
    BFloat16Rounding bfloat16Rounding;
    FloatingSourcePrecision floatingSourcePrecision;
    FloatingOverflow floatingOverflow;
    E8M0Underflow e8m0Underflow;
};

namespace detail {
template <NativeScalar T>
inline constexpr bool isComplexScalar =
    scalarTypeInfo(nativeScalarType<T>).category == ScalarCategory::Complex;

// Converts between native scalar types: integer destinations use the integer
// options, Boolean destinations test for nonzero, floating destinations use a
// static_cast, and complex values convert by component.
template <NativeScalar Target, NativeScalar Source>
Target convertNativeScalar(Source value, const ScalarConversionOptions& options) noexcept {
    if constexpr (isComplexScalar<Target>) {
        using Component = typename Target::value_type;
        if constexpr (isComplexScalar<Source>)
            return {convertNativeScalar<Component>(value.real(), options),
                    convertNativeScalar<Component>(value.imag(), options)};
        else
            return {convertNativeScalar<Component>(value, options), Component{0}};
    } else if constexpr (isComplexScalar<Source>) {
        assert(value.imag() == 0);
        return convertNativeScalar<Target>(value.real(), options);
    } else if constexpr (std::is_same_v<Target, bool>) {
        return value != Source{0};
    } else if constexpr (std::is_integral_v<Target>) {
        return convertToInteger<Target>(value, options.integerRounding, options.integerOverflow);
    } else {
        return static_cast<Target>(value);
    }
}

// The C++ type of one encoded element; packed elements use the low bits of a byte.
template <typename Tag>
using ScalarCode =
    std::conditional_t<std::is_void_v<typename Tag::Storage>, uint8_t, typename Tag::Storage>;

template <typename Tag, NativeScalar Source>
ScalarCode<Tag> encodeScalarCode(Source value, const ScalarConversionOptions& options) {
    constexpr ScalarType type = Tag::type;
    using Code = ScalarCode<Tag>;
    if constexpr (type == ScalarType::Boolean) {
        return Code{convertNativeScalar<bool>(value, options)};
    } else if constexpr (type == ScalarType::Int4) {
        if constexpr (isComplexScalar<Source>) {
            assert(value.imag() == 0);
            return encodeScalarCode<Tag>(value.real(), options);
        } else {
            return static_cast<Code>(convertToIntegerBits(value, 4, true, options.integerRounding,
                                                          options.integerOverflow));
        }
    } else if constexpr (isBinaryFloat(type) || type == ScalarType::E8M0 ||
                         type == ScalarType::E8M0Zero) {
        if constexpr (type == ScalarType::BFloat16) {
            // A faster equivalent of the general rounding below for the default BF16
            // options: adding just under half of the discarded range, plus the kept
            // low bit, rounds to nearest even and carries into the exponent as needed.
            if (options.floatingSourcePrecision == FloatingSourcePrecision::Float32 &&
                options.bfloat16Rounding == BFloat16Rounding::NearestEven &&
                options.floatingOverflow == FloatingOverflow::FormatDefault) {
                const float source = convertNativeScalar<float>(value, options);
                if (!std::isnan(source)) {
                    const auto bits = std::bit_cast<uint32_t>(source);
                    return static_cast<Code>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
                }
            }
        }
        const double source = options.floatingSourcePrecision == FloatingSourcePrecision::Direct
                                  ? convertNativeScalar<double>(value, options)
                                  : static_cast<double>(convertNativeScalar<float>(value, options));
        if constexpr (type == ScalarType::E8M0) {
            return encodeE8M0(source,
                              options.e8m0Underflow == E8M0Underflow::FlushBelowFloat32Normal);
        } else if constexpr (type == ScalarType::E8M0Zero) {
            return encodeE8M0Zero(source);
        } else {
            const bool truncate = type == ScalarType::BFloat16 &&
                                  options.bfloat16Rounding == BFloat16Rounding::Truncate;
            if constexpr (type == ScalarType::BFloat16)
                if (truncate && std::isnan(source)) return Code{0xffc1};
            return static_cast<Code>(encodeBinaryFloat<type>(
                source, options.floatingOverflow == FloatingOverflow::SaturateFinite, truncate));
        }
    } else {
        return convertNativeScalar<Code>(value, options);
    }
}

template <typename Tag, NativeScalar Target>
Target decodeScalarCode(ScalarCode<Tag> code, const ScalarConversionOptions& options) {
    constexpr ScalarType type = Tag::type;
    if constexpr (type == ScalarType::Boolean)
        return convertNativeScalar<Target>(code != 0, options);
    else if constexpr (type == ScalarType::Int4)
        return convertNativeScalar<Target>(static_cast<int8_t>((code ^ 8) - 8), options);
    else if constexpr (isBinaryFloat(type))
        return convertNativeScalar<Target>(decodeBinaryFloat<type>(code), options);
    else if constexpr (type == ScalarType::E8M0)
        return convertNativeScalar<Target>(decodeE8M0(code), options);
    else if constexpr (type == ScalarType::E8M0Zero)
        return convertNativeScalar<Target>(decodeE8M0Zero(code), options);
    else
        return convertNativeScalar<Target>(code, options);
}

// Returns the first bit of element index after checking that the whole element
// lies within storageBytes.
inline uint64_t elementBitOffset(ScalarType type, size_t storageBytes, size_t index) {
    const uint64_t bits = scalarTypeInfo(type).storageBits;
    const uint64_t bytes = storageBytes;
    const uint64_t elements = bytes / bits * 8 + bytes % bits * 8 / bits;
    if (index >= elements)
        throw std::runtime_error("Scalar element " + std::to_string(index) +
                                 " is outside storage of " + std::to_string(storageBytes) +
                                 " bytes.");
    return index * bits;
}

// Packed elements have at most 8 bits, so each lies within two adjacent bytes.
inline uint8_t readPackedCode(std::span<const std::byte> storage, uint64_t bitOffset,
                              uint32_t bits) {
    const auto byte = static_cast<size_t>(bitOffset / 8);
    const auto shift = static_cast<uint32_t>(bitOffset % 8);
    uint32_t window = std::to_integer<uint32_t>(storage[byte]);
    if (shift + bits > 8) window |= std::to_integer<uint32_t>(storage[byte + 1]) << 8;
    return static_cast<uint8_t>((window >> shift) & ((uint32_t{1} << bits) - 1));
}

inline void writePackedCode(std::span<std::byte> storage, uint64_t bitOffset, uint32_t bits,
                            uint8_t code) {
    const auto byte = static_cast<size_t>(bitOffset / 8);
    const auto shift = static_cast<uint32_t>(bitOffset % 8);
    const uint32_t mask = ((uint32_t{1} << bits) - 1) << shift;
    const uint32_t value = (static_cast<uint32_t>(code) << shift) & mask;
    const auto update = [](std::byte& target, uint32_t targetMask, uint32_t targetValue) {
        target =
            static_cast<std::byte>((std::to_integer<uint32_t>(target) & ~targetMask) | targetValue);
    };
    update(storage[byte], mask & 0xff, value & 0xff);
    if (shift + bits > 8) update(storage[byte + 1], mask >> 8, value >> 8);
}
}  // namespace detail

// Storage holds a sequence of elements encoded as one ScalarType. Element i
// occupies storage bits [i * storageBits, (i + 1) * storageBits), where bit b
// is bit b % 8 of byte b / 8, counting from the least significant bit. A
// packed element (Int4, FP4 or FP6) can span two bytes; wider elements are
// whole bytes in the host's byte order.
//
// encodeScalar converts value to type and writes element index, leaving every
// other bit unchanged. decodeScalar reads element index and converts it to
// Target. Both throw std::runtime_error, without changing storage, if type is
// invalid or the element does not lie entirely within storage.
//
// Integer destinations, including Int4, use convertToInteger with the options'
// integer rounding and overflow. A Boolean destination tests for nonzero,
// float and double destinations use a static_cast, and complex values convert
// by component; a real value gains a zero imaginary part. Encoding FP16, BF16,
// narrower floating types and E8M0 scales uses the floating options. Every
// floating type decodes exactly to double, and every one except Float64 to float.
//
// The caller meets the input requirements of the chosen IntegerOverflow, keeps
// float and double values within the destination's range, converts a complex
// value to a real destination only when its imaginary part is zero, and
// encodes only values that are not negative as the unsigned scales E4M3, E5M3
// and E8M0. Builds without NDEBUG assert each requirement except the float and
// double range.
//
// Examples with options{IntegerRounding::NearestEven, IntegerOverflow::Saturate}
// and std::array<std::byte, 2> storage{}:
//   encodeScalar(ScalarType::Float8E4M3, storage, 1, 1.0625, options)
//     writes 0x38 (1.0, the even neighbor of the tie) to storage[1].
//   encodeScalar(ScalarType::Float8E4M3, storage, 0, 1000.0, options)
//     writes 0x7e (448, the largest finite value) to storage[0].
//   decodeScalar<int32_t>(ScalarType::Float8E4M3, storage, 0, options) == 448
//   encodeScalar(ScalarType::Int4, storage, 2, 9, options)
//     writes 7 to the low four bits of storage[1], which becomes 0x37.
template <NativeScalar Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, size_t index, Source value,
                  const ScalarConversionOptions& options) {
    visitScalarType(type, [&]<typename Tag>() {
        const uint64_t bitOffset = detail::elementBitOffset(Tag::type, storage.size(), index);
        const detail::ScalarCode<Tag> code = detail::encodeScalarCode<Tag>(value, options);
        if constexpr (std::is_void_v<typename Tag::Storage>)
            detail::writePackedCode(storage, bitOffset, Tag::storageBits, code);
        else
            std::memcpy(storage.data() + bitOffset / 8, &code, sizeof(code));
    });
}

template <NativeScalar Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, size_t index,
                    const ScalarConversionOptions& options) {
    return visitScalarType(type, [&]<typename Tag>() {
        const uint64_t bitOffset = detail::elementBitOffset(Tag::type, storage.size(), index);
        detail::ScalarCode<Tag> code;
        if constexpr (std::is_void_v<typename Tag::Storage>)
            code = detail::readPackedCode(storage, bitOffset, Tag::storageBits);
        else
            std::memcpy(&code, storage.data() + bitOffset / 8, sizeof(code));
        return detail::decodeScalarCode<Tag, Target>(code, options);
    });
}
}  // namespace roc::hostnumerics
