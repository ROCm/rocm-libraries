// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace roc::host_validation {
enum class ScalarCategory : uint8_t {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
    Complex,
    Scale,
};

enum class ScalarType : uint16_t {
    Boolean,
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float16,
    BFloat16,
    Float32,
    Float64,
    ComplexFloat32,
    ComplexFloat64,
    Float8E4M3,
    Float8E5M2,
    Float8E4M3Fnuz,
    Float8E5M2Fnuz,
    Float6E2M3,
    Float6E3M2,
    Float4E2M1,
    Int4,
    Int12,
    E8M0,
    E5M3,
    Count,
};

struct ScalarTypeInfo {
    std::string_view name;
    ScalarCategory category;
    uint16_t storageBits;
    uint8_t exponentBits;
    uint8_t mantissaBits;
    int16_t exponentBias;
    bool supportsNaN;
    bool supportsInfinity;

    bool isPacked() const {
        return storageBits % 8 != 0;
    }
};

inline constexpr std::array<ScalarTypeInfo, static_cast<size_t>(ScalarType::Count)> scalarTypeInfos{
    {
        {"bool", ScalarCategory::Boolean, 8, 0, 0, 0, false, false},
        {"u8", ScalarCategory::UnsignedInteger, 8, 0, 0, 0, false, false},
        {"i8", ScalarCategory::SignedInteger, 8, 0, 0, 0, false, false},
        {"u16", ScalarCategory::UnsignedInteger, 16, 0, 0, 0, false, false},
        {"i16", ScalarCategory::SignedInteger, 16, 0, 0, 0, false, false},
        {"u32", ScalarCategory::UnsignedInteger, 32, 0, 0, 0, false, false},
        {"i32", ScalarCategory::SignedInteger, 32, 0, 0, 0, false, false},
        {"u64", ScalarCategory::UnsignedInteger, 64, 0, 0, 0, false, false},
        {"i64", ScalarCategory::SignedInteger, 64, 0, 0, 0, false, false},
        {"f16", ScalarCategory::FloatingPoint, 16, 5, 10, 15, true, true},
        {"bf16", ScalarCategory::FloatingPoint, 16, 8, 7, 127, true, true},
        {"f32", ScalarCategory::FloatingPoint, 32, 8, 23, 127, true, true},
        {"f64", ScalarCategory::FloatingPoint, 64, 11, 52, 1023, true, true},
        {"c64", ScalarCategory::Complex, 64, 8, 23, 127, true, true},
        {"c128", ScalarCategory::Complex, 128, 11, 52, 1023, true, true},
        {"f8e4m3", ScalarCategory::FloatingPoint, 8, 4, 3, 7, true, false},
        {"f8e5m2", ScalarCategory::FloatingPoint, 8, 5, 2, 15, true, true},
        {"f8e4m3fnuz", ScalarCategory::FloatingPoint, 8, 4, 3, 8, true, false},
        {"f8e5m2fnuz", ScalarCategory::FloatingPoint, 8, 5, 2, 16, true, false},
        {"f6e2m3", ScalarCategory::FloatingPoint, 6, 2, 3, 1, false, false},
        {"f6e3m2", ScalarCategory::FloatingPoint, 6, 3, 2, 3, false, false},
        {"f4e2m1", ScalarCategory::FloatingPoint, 4, 2, 1, 1, false, false},
        {"i4", ScalarCategory::SignedInteger, 4, 0, 0, 0, false, false},
        {"i12", ScalarCategory::SignedInteger, 12, 0, 0, 0, false, false},
        {"e8m0", ScalarCategory::Scale, 8, 8, 0, 127, true, false},
        {"e5m3", ScalarCategory::Scale, 8, 5, 3, 15, true, false},
    }};

inline constexpr const ScalarTypeInfo& scalarTypeInfo(ScalarType type) {
    const size_t index = static_cast<size_t>(type);
    if (index >= scalarTypeInfos.size()) throw std::invalid_argument("Invalid ScalarType.");
    return scalarTypeInfos[index];
}

inline constexpr std::string_view scalarTypeName(ScalarType type) {
    return scalarTypeInfo(type).name;
}

template <typename T>
struct NativeScalarType;

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

template <typename T>
inline constexpr ScalarType nativeScalarType = NativeScalarType<std::remove_cv_t<T>>::value;

namespace detail {
template <ScalarType TypeValue, typename StorageType>
struct ScalarTag {
    using Storage = StorageType;
    static constexpr ScalarType type = TypeValue;
    static constexpr uint16_t storageBits = scalarTypeInfo(TypeValue).storageBits;
};

using BooleanTag = ScalarTag<ScalarType::Boolean, uint8_t>;
using UInt8Tag = ScalarTag<ScalarType::UInt8, uint8_t>;
using Int8Tag = ScalarTag<ScalarType::Int8, int8_t>;
using UInt16Tag = ScalarTag<ScalarType::UInt16, uint16_t>;
using Int16Tag = ScalarTag<ScalarType::Int16, int16_t>;
using UInt32Tag = ScalarTag<ScalarType::UInt32, uint32_t>;
using Int32Tag = ScalarTag<ScalarType::Int32, int32_t>;
using UInt64Tag = ScalarTag<ScalarType::UInt64, uint64_t>;
using Int64Tag = ScalarTag<ScalarType::Int64, int64_t>;
using Float16Tag = ScalarTag<ScalarType::Float16, uint16_t>;
using BFloat16Tag = ScalarTag<ScalarType::BFloat16, uint16_t>;
using Float32Tag = ScalarTag<ScalarType::Float32, float>;
using Float64Tag = ScalarTag<ScalarType::Float64, double>;
using ComplexFloat32Tag = ScalarTag<ScalarType::ComplexFloat32, std::complex<float>>;
using ComplexFloat64Tag = ScalarTag<ScalarType::ComplexFloat64, std::complex<double>>;
using Float8E4M3Tag = ScalarTag<ScalarType::Float8E4M3, uint8_t>;
using Float8E5M2Tag = ScalarTag<ScalarType::Float8E5M2, uint8_t>;
using Float8E4M3FnuzTag = ScalarTag<ScalarType::Float8E4M3Fnuz, uint8_t>;
using Float8E5M2FnuzTag = ScalarTag<ScalarType::Float8E5M2Fnuz, uint8_t>;
using Float6E2M3Tag = ScalarTag<ScalarType::Float6E2M3, void>;
using Float6E3M2Tag = ScalarTag<ScalarType::Float6E3M2, void>;
using Float4E2M1Tag = ScalarTag<ScalarType::Float4E2M1, void>;
using Int4Tag = ScalarTag<ScalarType::Int4, void>;
using Int12Tag = ScalarTag<ScalarType::Int12, void>;
using E8M0Tag = ScalarTag<ScalarType::E8M0, uint8_t>;
using E5M3Tag = ScalarTag<ScalarType::E5M3, uint8_t>;
}  // namespace detail

template <typename Visitor, typename... Args>
decltype(auto) visitScalarType(ScalarType type, Visitor&& visitor, Args&&... args) {
    switch (type) {
        case ScalarType::Boolean:
            return std::forward<Visitor>(visitor).template operator()<detail::BooleanTag>(
                std::forward<Args>(args)...);
        case ScalarType::UInt8:
            return std::forward<Visitor>(visitor).template operator()<detail::UInt8Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int8:
            return std::forward<Visitor>(visitor).template operator()<detail::Int8Tag>(
                std::forward<Args>(args)...);
        case ScalarType::UInt16:
            return std::forward<Visitor>(visitor).template operator()<detail::UInt16Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int16:
            return std::forward<Visitor>(visitor).template operator()<detail::Int16Tag>(
                std::forward<Args>(args)...);
        case ScalarType::UInt32:
            return std::forward<Visitor>(visitor).template operator()<detail::UInt32Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int32:
            return std::forward<Visitor>(visitor).template operator()<detail::Int32Tag>(
                std::forward<Args>(args)...);
        case ScalarType::UInt64:
            return std::forward<Visitor>(visitor).template operator()<detail::UInt64Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int64:
            return std::forward<Visitor>(visitor).template operator()<detail::Int64Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float16:
            return std::forward<Visitor>(visitor).template operator()<detail::Float16Tag>(
                std::forward<Args>(args)...);
        case ScalarType::BFloat16:
            return std::forward<Visitor>(visitor).template operator()<detail::BFloat16Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float32:
            return std::forward<Visitor>(visitor).template operator()<detail::Float32Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float64:
            return std::forward<Visitor>(visitor).template operator()<detail::Float64Tag>(
                std::forward<Args>(args)...);
        case ScalarType::ComplexFloat32:
            return std::forward<Visitor>(visitor).template operator()<detail::ComplexFloat32Tag>(
                std::forward<Args>(args)...);
        case ScalarType::ComplexFloat64:
            return std::forward<Visitor>(visitor).template operator()<detail::ComplexFloat64Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float8E4M3:
            return std::forward<Visitor>(visitor).template operator()<detail::Float8E4M3Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float8E5M2:
            return std::forward<Visitor>(visitor).template operator()<detail::Float8E5M2Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float8E4M3Fnuz:
            return std::forward<Visitor>(visitor).template operator()<detail::Float8E4M3FnuzTag>(
                std::forward<Args>(args)...);
        case ScalarType::Float8E5M2Fnuz:
            return std::forward<Visitor>(visitor).template operator()<detail::Float8E5M2FnuzTag>(
                std::forward<Args>(args)...);
        case ScalarType::Float6E2M3:
            return std::forward<Visitor>(visitor).template operator()<detail::Float6E2M3Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float6E3M2:
            return std::forward<Visitor>(visitor).template operator()<detail::Float6E3M2Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Float4E2M1:
            return std::forward<Visitor>(visitor).template operator()<detail::Float4E2M1Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int4:
            return std::forward<Visitor>(visitor).template operator()<detail::Int4Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Int12:
            return std::forward<Visitor>(visitor).template operator()<detail::Int12Tag>(
                std::forward<Args>(args)...);
        case ScalarType::E8M0:
            return std::forward<Visitor>(visitor).template operator()<detail::E8M0Tag>(
                std::forward<Args>(args)...);
        case ScalarType::E5M3:
            return std::forward<Visitor>(visitor).template operator()<detail::E5M3Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Count:
            break;
    }
    throw std::invalid_argument("Invalid ScalarType.");
}

class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions) : m_dimensions(std::move(dimensions)) {}

    Shape(std::initializer_list<size_t> dimensions) : m_dimensions(dimensions) {}

    explicit Shape(std::span<const size_t> dimensions)
        : m_dimensions(dimensions.begin(), dimensions.end()) {}

    size_t rank() const {
        return m_dimensions.size();
    }

    bool empty() const {
        return elementCount() == 0;
    }

    size_t operator[](size_t dimension) const {
        return m_dimensions.at(dimension);
    }

    std::span<const size_t> dimensions() const {
        return m_dimensions;
    }

    size_t elementCount() const {
        size_t count = 1;
        for (const size_t dimension : m_dimensions) {
            if (dimension == 0) return 0;
            if (count > std::numeric_limits<size_t>::max() / dimension)
                throw std::overflow_error("Tensor shape element count overflow.");
            count *= dimension;
        }
        return count;
    }

    friend bool operator==(const Shape&, const Shape&) = default;

   private:
    std::vector<size_t> m_dimensions;
};

class Layout {
   public:
    Layout() = default;

    Layout(Shape shape, std::vector<ptrdiff_t> strides, ptrdiff_t offset = 0)
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset) {
        if (m_shape.rank() != m_strides.size())
            throw std::invalid_argument("Tensor layout rank and stride count differ.");
    }

    static Layout contiguous(const Shape& shape) {
        std::vector<ptrdiff_t> strides(shape.rank(), 1);
        ptrdiff_t stride = 1;
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            strides[index] = stride;
            const size_t extent = shape[index];
            if (extent > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
                throw std::overflow_error("Tensor extent exceeds ptrdiff_t.");
            const ptrdiff_t signedExtent = static_cast<ptrdiff_t>(extent);
            if (signedExtent != 0 && stride > std::numeric_limits<ptrdiff_t>::max() / signedExtent)
                throw std::overflow_error("Tensor contiguous stride overflow.");
            stride *= signedExtent;
        }
        return Layout(shape, std::move(strides));
    }

    const Shape& shape() const {
        return m_shape;
    }

    std::span<const ptrdiff_t> strides() const {
        return m_strides;
    }

    ptrdiff_t offset() const {
        return m_offset;
    }

    ptrdiff_t elementOffset(std::span<const size_t> indices) const {
        if (indices.size() != m_shape.rank())
            throw std::invalid_argument("Tensor index rank does not match layout rank.");

        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= m_shape[dimension])
                throw std::out_of_range("Tensor index exceeds shape.");
            result += static_cast<ptrdiff_t>(indices[dimension]) * m_strides[dimension];
        }
        return result;
    }

    friend bool operator==(const Layout&, const Layout&) = default;

   private:
    Shape m_shape;
    std::vector<ptrdiff_t> m_strides;
    ptrdiff_t m_offset = 0;
};

namespace detail {
template <typename T>
struct RuntimeIsComplex : std::false_type {};

template <typename T>
struct RuntimeIsComplex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool RuntimeIsComplexV = RuntimeIsComplex<T>::value;

template <typename T>
auto realComponent(const T& value) {
    if constexpr (RuntimeIsComplexV<T>)
        return value.real();
    else
        return value;
}

inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout) {
    if (layout.shape().elementCount() == 0) return {0, -1};

    ptrdiff_t lower = layout.offset();
    ptrdiff_t upper = layout.offset();
    for (size_t dimension = 0; dimension < layout.shape().rank(); ++dimension) {
        const ptrdiff_t delta =
            static_cast<ptrdiff_t>(layout.shape()[dimension] - 1) * layout.strides()[dimension];
        if (delta < 0)
            lower += delta;
        else
            upper += delta;
    }
    return {lower, upper};
}

inline size_t storageBytesForLayout(ScalarType type, const Layout& layout) {
    const auto [lower, upper] = elementBounds(layout);
    if (upper < lower) return 0;
    if (lower < 0) throw std::invalid_argument("Tensor layout addresses before the storage base.");

    const uint64_t bits = scalarTypeInfo(type).storageBits;
    const uint64_t elementCount = static_cast<uint64_t>(upper) + 1;
    if (elementCount > std::numeric_limits<uint64_t>::max() / bits)
        throw std::overflow_error("Tensor storage size overflow.");
    const uint64_t totalBits = elementCount * bits;
    const uint64_t bytes = (totalBits + 7) / 8;
    if (bytes > std::numeric_limits<size_t>::max())
        throw std::overflow_error("Tensor storage byte count overflow.");
    return static_cast<size_t>(bytes);
}

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

inline float decodeFloat16(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits >> 15);
    const uint32_t exponent = (bits >> 10) & 0x1fU;
    const uint32_t mantissa = bits & 0x3ffU;

    float value;
    if (exponent == 0) {
        value = mantissa == 0 ? 0.0f : std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 0x1fU) {
        value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                           static_cast<int>(exponent) - 15);
    }
    return sign ? -value : value;
}

inline uint16_t encodeFloat16(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t exponent = (bits >> 23) & 0xffU;
    uint32_t mantissa = bits & 0x7fffffU;

    if (exponent == 0xffU) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00U);
        return static_cast<uint16_t>(sign | 0x7e00U);
    }

    int32_t halfExponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (halfExponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00U);
    if (halfExponent <= 0) {
        if (halfExponent < -10) return sign;
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - halfExponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t roundedMantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (roundedMantissa & 1U))) {
        ++roundedMantissa;
        if (roundedMantissa == 0x400U) {
            roundedMantissa = 0;
            ++halfExponent;
            if (halfExponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExponent) << 10) |
                                 roundedMantissa);
}

inline float decodeBFloat16(uint16_t bits) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

inline uint16_t encodeBFloat16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t leastSignificantBit = (bits >> 16) & 1U;
    bits += 0x7fffU + leastSignificantBit;
    return static_cast<uint16_t>(bits >> 16);
}

struct BinaryFloatFormat {
    uint8_t exponentBits;
    uint8_t mantissaBits;
    int16_t exponentBias;
    uint8_t totalBits;
    bool hasSign;
    bool hasSignedZero;
    bool hasInfinity;
    uint32_t maximumPositiveFiniteRaw;
    uint32_t positiveInfinityRaw;
    uint32_t canonicalNaNRaw;
};

inline BinaryFloatFormat binaryFloatFormat(ScalarType type) {
    switch (type) {
        case ScalarType::Float4E2M1:
            return {2, 1, 1, 4, true, true, false, 0x7, 0, 0};
        case ScalarType::Float6E2M3:
            return {2, 3, 1, 6, true, true, false, 0x1f, 0, 0};
        case ScalarType::Float6E3M2:
            return {3, 2, 3, 6, true, true, false, 0x1f, 0, 0};
        case ScalarType::Float8E4M3:
            return {4, 3, 7, 8, true, true, false, 0x7e, 0, 0x7f};
        case ScalarType::Float8E5M2:
            return {5, 2, 15, 8, true, true, true, 0x7b, 0x7c, 0x7f};
        case ScalarType::Float8E4M3Fnuz:
            return {4, 3, 8, 8, true, false, false, 0x7f, 0, 0x80};
        case ScalarType::Float8E5M2Fnuz:
            return {5, 2, 16, 8, true, false, false, 0x7f, 0, 0x80};
        case ScalarType::E5M3:
            return {5, 3, 15, 8, false, false, false, 0xfe, 0, 0xff};
        default:
            throw std::invalid_argument(
                "ScalarType is not a supported binary floating-point format.");
    }
}

inline bool isBinaryFloatNaN(ScalarType type, uint32_t raw) {
    switch (type) {
        case ScalarType::Float8E4M3:
            return (raw & 0x7fU) == 0x7fU;
        case ScalarType::Float8E5M2:
            return (raw & 0x7fU) > 0x7cU;
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
            return raw == 0x80U;
        case ScalarType::E5M3:
            return raw == 0xffU;
        default:
            return false;
    }
}

inline bool isBinaryFloatInfinity(ScalarType type, uint32_t raw) {
    return type == ScalarType::Float8E5M2 && (raw & 0x7fU) == 0x7cU;
}

inline float decodeFiniteBinaryFloatMagnitude(uint32_t raw, const BinaryFloatFormat& format) {
    const uint32_t exponentMask = (1U << format.exponentBits) - 1U;
    const uint32_t mantissaMask = (1U << format.mantissaBits) - 1U;
    const uint32_t exponent = (raw >> format.mantissaBits) & exponentMask;
    const uint32_t mantissa = raw & mantissaMask;
    const float mantissaScale = 1.0f / static_cast<float>(1U << format.mantissaBits);

    if (exponent == 0)
        return std::ldexp(static_cast<float>(mantissa) * mantissaScale, 1 - format.exponentBias);
    return std::ldexp(1.0f + static_cast<float>(mantissa) * mantissaScale,
                      static_cast<int>(exponent) - format.exponentBias);
}

inline float decodeBinaryFloat(ScalarType type, uint32_t raw) {
    const auto format = binaryFloatFormat(type);
    if (isBinaryFloatNaN(type, raw)) return std::numeric_limits<float>::quiet_NaN();

    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const bool negative = format.hasSign && (raw & signMask) != 0;
    if (isBinaryFloatInfinity(type, raw))
        return negative ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();

    const uint32_t magnitude = format.hasSign ? raw & (signMask - 1U) : raw;
    const float value = decodeFiniteBinaryFloatMagnitude(magnitude, format);
    return negative ? -value : value;
}

inline uint32_t nearestPositiveBinaryFloatRaw(float value, const BinaryFloatFormat& format) {
    uint32_t lower = 0;
    uint32_t upper = format.maximumPositiveFiniteRaw;
    while (lower + 1 < upper) {
        const uint32_t middle = lower + (upper - lower) / 2;
        if (decodeFiniteBinaryFloatMagnitude(middle, format) < value)
            lower = middle;
        else
            upper = middle;
    }

    const double lowerValue = static_cast<double>(decodeFiniteBinaryFloatMagnitude(lower, format));
    const double upperValue = static_cast<double>(decodeFiniteBinaryFloatMagnitude(upper, format));
    const double lowerDistance = static_cast<double>(value) - lowerValue;
    const double upperDistance = upperValue - static_cast<double>(value);
    if (lowerDistance < upperDistance) return lower;
    if (upperDistance < lowerDistance) return upper;
    return (lower & 1U) == 0 ? lower : upper;
}

inline uint32_t encodeBinaryFloat(ScalarType type, float value) {
    const auto format = binaryFloatFormat(type);
    if (std::isnan(value)) {
        if (!scalarTypeInfo(type).supportsNaN) {
            const uint32_t signMask =
                format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
            const uint32_t sign = std::signbit(value) ? signMask : 0U;
            return sign | format.maximumPositiveFiniteRaw;
        }
        const bool preserveSign = type == ScalarType::Float8E4M3 || type == ScalarType::Float8E5M2;
        const uint32_t sign = preserveSign && std::signbit(value) ? 0x80U : 0U;
        return sign | format.canonicalNaNRaw;
    }

    if (!format.hasSign && value != 0.0f && std::signbit(value))
        throw std::domain_error("Unsigned scale formats cannot encode negative values.");

    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const uint32_t sign = std::signbit(value) ? signMask : 0U;
    value = std::fabs(value);

    if (std::isinf(value)) {
        if (format.hasInfinity) return sign | format.positiveInfinityRaw;
        return sign | format.maximumPositiveFiniteRaw;
    }
    if (value == 0.0f) return format.hasSignedZero ? sign : 0U;

    const float maximumValue =
        decodeFiniteBinaryFloatMagnitude(format.maximumPositiveFiniteRaw, format);
    if (value >= maximumValue) return sign | format.maximumPositiveFiniteRaw;

    const uint32_t magnitude = nearestPositiveBinaryFloatRaw(value, format);
    if (magnitude == 0 && !format.hasSignedZero) return 0;
    return sign | magnitude;
}

inline float decodeE8M0(uint8_t raw) {
    if (raw == 0xffU) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(raw) - 127);
}

inline uint8_t encodeE8M0(float value) {
    if (std::isnan(value)) return 0xffU;
    if (value != 0.0f && std::signbit(value))
        throw std::domain_error("E8M0 cannot encode negative values.");
    if (value <= std::ldexp(1.0f, -127)) return 0;
    if (std::isinf(value) || value >= std::ldexp(1.0f, 127)) return 0xfeU;

    uint32_t lower = 0;
    uint32_t upper = 0xfeU;
    while (lower + 1 < upper) {
        const uint32_t middle = lower + (upper - lower) / 2;
        if (decodeE8M0(static_cast<uint8_t>(middle)) < value)
            lower = middle;
        else
            upper = middle;
    }

    const double lowerDistance =
        static_cast<double>(value) - decodeE8M0(static_cast<uint8_t>(lower));
    const double upperDistance =
        decodeE8M0(static_cast<uint8_t>(upper)) - static_cast<double>(value);
    if (lowerDistance < upperDistance) return static_cast<uint8_t>(lower);
    if (upperDistance < lowerDistance) return static_cast<uint8_t>(upper);
    return static_cast<uint8_t>((lower & 1U) == 0 ? lower : upper);
}

inline uint64_t bitOffset(ScalarType type, ptrdiff_t logicalOffset) {
    if (logicalOffset < 0) throw std::out_of_range("Tensor logical offset is negative.");
    const uint64_t bits = scalarTypeInfo(type).storageBits;
    const uint64_t offset = static_cast<uint64_t>(logicalOffset);
    if (offset > std::numeric_limits<uint64_t>::max() / bits)
        throw std::overflow_error("Tensor bit offset overflow.");
    return offset * bits;
}

inline int64_t signExtend(uint32_t value, uint32_t bits) {
    const uint32_t sign = 1U << (bits - 1U);
    return static_cast<int32_t>((value ^ sign) - sign);
}

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    const uint64_t offsetBits = bitOffset(type, logicalOffset);
    const size_t offsetBytes = static_cast<size_t>(offsetBits / 8);

    switch (type) {
        case ScalarType::Boolean:
            return static_cast<Target>(readNative<uint8_t>(storage, offsetBytes) != 0);
        case ScalarType::UInt8:
            return static_cast<Target>(readNative<uint8_t>(storage, offsetBytes));
        case ScalarType::Int8:
            return static_cast<Target>(readNative<int8_t>(storage, offsetBytes));
        case ScalarType::UInt16:
            return static_cast<Target>(readNative<uint16_t>(storage, offsetBytes));
        case ScalarType::Int16:
            return static_cast<Target>(readNative<int16_t>(storage, offsetBytes));
        case ScalarType::UInt32:
            return static_cast<Target>(readNative<uint32_t>(storage, offsetBytes));
        case ScalarType::Int32:
            return static_cast<Target>(readNative<int32_t>(storage, offsetBytes));
        case ScalarType::UInt64:
            return static_cast<Target>(readNative<uint64_t>(storage, offsetBytes));
        case ScalarType::Int64:
            return static_cast<Target>(readNative<int64_t>(storage, offsetBytes));
        case ScalarType::Float16:
            return static_cast<Target>(decodeFloat16(readNative<uint16_t>(storage, offsetBytes)));
        case ScalarType::BFloat16:
            return static_cast<Target>(decodeBFloat16(readNative<uint16_t>(storage, offsetBytes)));
        case ScalarType::Float32:
            return static_cast<Target>(readNative<float>(storage, offsetBytes));
        case ScalarType::Float64:
            return static_cast<Target>(readNative<double>(storage, offsetBytes));
        case ScalarType::ComplexFloat32: {
            if constexpr (!RuntimeIsComplexV<Target>)
                throw std::invalid_argument("Complex tensor value requires a complex target.");
            else {
                const auto value = readNative<std::complex<float>>(storage, offsetBytes);
                return Target(value.real(), value.imag());
            }
        }
        case ScalarType::ComplexFloat64: {
            if constexpr (!RuntimeIsComplexV<Target>)
                throw std::invalid_argument("Complex tensor value requires a complex target.");
            else {
                const auto value = readNative<std::complex<double>>(storage, offsetBytes);
                return Target(value.real(), value.imag());
            }
        }
        case ScalarType::Int4:
            return static_cast<Target>(signExtend(readPackedBits(storage, offsetBits, 4), 4));
        case ScalarType::Int12:
            return static_cast<Target>(signExtend(readPackedBits(storage, offsetBits, 12), 12));
        case ScalarType::Float4E2M1:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::E5M3:
            return static_cast<Target>(decodeBinaryFloat(
                type, readPackedBits(storage, offsetBits, scalarTypeInfo(type).storageBits)));
        case ScalarType::E8M0: {
            const uint8_t value = readNative<uint8_t>(storage, offsetBytes);
            return static_cast<Target>(decodeE8M0(value));
        }
        case ScalarType::Count:
            break;
    }
    throw std::invalid_argument("Invalid ScalarType.");
}

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source) {
    const uint64_t offsetBits = bitOffset(type, logicalOffset);
    const size_t offsetBytes = static_cast<size_t>(offsetBits / 8);
    const auto scalar = realComponent(source);

    switch (type) {
        case ScalarType::Boolean:
            writeNative<uint8_t>(storage, offsetBytes, scalar != 0 ? 1U : 0U);
            return;
        case ScalarType::UInt8:
            writeNative<uint8_t>(storage, offsetBytes, static_cast<uint8_t>(scalar));
            return;
        case ScalarType::Int8:
            writeNative<int8_t>(storage, offsetBytes, static_cast<int8_t>(scalar));
            return;
        case ScalarType::UInt16:
            writeNative<uint16_t>(storage, offsetBytes, static_cast<uint16_t>(scalar));
            return;
        case ScalarType::Int16:
            writeNative<int16_t>(storage, offsetBytes, static_cast<int16_t>(scalar));
            return;
        case ScalarType::UInt32:
            writeNative<uint32_t>(storage, offsetBytes, static_cast<uint32_t>(scalar));
            return;
        case ScalarType::Int32:
            writeNative<int32_t>(storage, offsetBytes, static_cast<int32_t>(scalar));
            return;
        case ScalarType::UInt64:
            writeNative<uint64_t>(storage, offsetBytes, static_cast<uint64_t>(scalar));
            return;
        case ScalarType::Int64:
            writeNative<int64_t>(storage, offsetBytes, static_cast<int64_t>(scalar));
            return;
        case ScalarType::Float16:
            writeNative<uint16_t>(storage, offsetBytes, encodeFloat16(static_cast<float>(scalar)));
            return;
        case ScalarType::BFloat16:
            writeNative<uint16_t>(storage, offsetBytes, encodeBFloat16(static_cast<float>(scalar)));
            return;
        case ScalarType::Float32:
            writeNative<float>(storage, offsetBytes, static_cast<float>(scalar));
            return;
        case ScalarType::Float64:
            writeNative<double>(storage, offsetBytes, static_cast<double>(scalar));
            return;
        case ScalarType::ComplexFloat32: {
            if constexpr (RuntimeIsComplexV<Source>) {
                writeNative<std::complex<float>>(
                    storage, offsetBytes,
                    std::complex<float>(static_cast<float>(source.real()),
                                        static_cast<float>(source.imag())));
            } else {
                writeNative<std::complex<float>>(
                    storage, offsetBytes, std::complex<float>(static_cast<float>(scalar), 0.0f));
            }
            return;
        }
        case ScalarType::ComplexFloat64: {
            if constexpr (RuntimeIsComplexV<Source>) {
                writeNative<std::complex<double>>(
                    storage, offsetBytes,
                    std::complex<double>(static_cast<double>(source.real()),
                                         static_cast<double>(source.imag())));
            } else {
                writeNative<std::complex<double>>(
                    storage, offsetBytes, std::complex<double>(static_cast<double>(scalar), 0.0));
            }
            return;
        }
        case ScalarType::Int4: {
            const int64_t value =
                std::max<int64_t>(-8, std::min<int64_t>(7, static_cast<int64_t>(scalar)));
            writePackedBits(storage, offsetBits, 4, static_cast<uint32_t>(value) & 0xfU);
            return;
        }
        case ScalarType::Int12: {
            const int64_t value =
                std::max<int64_t>(-2048, std::min<int64_t>(2047, static_cast<int64_t>(scalar)));
            writePackedBits(storage, offsetBits, 12, static_cast<uint32_t>(value) & 0xfffU);
            return;
        }
        case ScalarType::Float4E2M1:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::E5M3:
            writePackedBits(storage, offsetBits, scalarTypeInfo(type).storageBits,
                            encodeBinaryFloat(type, static_cast<float>(scalar)));
            return;
        case ScalarType::E8M0: {
            writeNative<uint8_t>(storage, offsetBytes, encodeE8M0(static_cast<float>(scalar)));
            return;
        }
        case ScalarType::Count:
            break;
    }
    throw std::invalid_argument("Invalid ScalarType.");
}
}  // namespace detail

inline size_t storageBytesForLayout(ScalarType type, const Layout& layout) {
    return detail::storageBytesForLayout(type, layout);
}

class TensorView {
   public:
    TensorView(ScalarType type, Layout layout, std::span<const std::byte> storage)
        : m_type(type), m_layout(std::move(layout)), m_storage(storage) {
        if (m_storage.size() < storageBytesForLayout(m_type, m_layout))
            throw std::invalid_argument("TensorView storage is too small for its layout.");
    }

    template <typename T>
    static TensorView fromNative(Layout layout, std::span<const T> values) {
        return TensorView(nativeScalarType<T>, std::move(layout), std::as_bytes(values));
    }

    ScalarType type() const {
        return m_type;
    }

    const Shape& shape() const {
        return m_layout.shape();
    }

    const Layout& layout() const {
        return m_layout;
    }

    std::span<const std::byte> storage() const {
        return m_storage;
    }

    template <typename Target>
    Target loadAs(std::span<const size_t> indices) const {
        return detail::decodeScalar<Target>(m_type, m_storage, m_layout.elementOffset(indices));
    }

    template <typename Target>
    Target loadAs(std::initializer_list<size_t> indices) const {
        return loadAs<Target>(std::span<const size_t>(indices.begin(), indices.size()));
    }

   private:
    ScalarType m_type;
    Layout m_layout;
    std::span<const std::byte> m_storage;
};

class MutableTensorView {
   public:
    MutableTensorView(ScalarType type, Layout layout, std::span<std::byte> storage)
        : m_type(type), m_layout(std::move(layout)), m_storage(storage) {
        if (m_storage.size() < storageBytesForLayout(m_type, m_layout))
            throw std::invalid_argument("MutableTensorView storage is too small for its layout.");
    }

    template <typename T>
    static MutableTensorView fromNative(Layout layout, std::span<T> values) {
        return MutableTensorView(nativeScalarType<T>, std::move(layout),
                                 std::as_writable_bytes(values));
    }

    ScalarType type() const {
        return m_type;
    }

    const Shape& shape() const {
        return m_layout.shape();
    }

    const Layout& layout() const {
        return m_layout;
    }

    std::span<std::byte> storage() const {
        return m_storage;
    }

    template <typename Target>
    Target loadAs(std::span<const size_t> indices) const {
        return detail::decodeScalar<Target>(m_type, m_storage, m_layout.elementOffset(indices));
    }

    template <typename Target>
    Target loadAs(std::initializer_list<size_t> indices) const {
        return loadAs<Target>(std::span<const size_t>(indices.begin(), indices.size()));
    }

    template <typename Source>
    void storeFrom(std::span<const size_t> indices, Source value) const {
        detail::encodeScalar(m_type, m_storage, m_layout.elementOffset(indices), value);
    }

    template <typename Source>
    void storeFrom(std::initializer_list<size_t> indices, Source value) const {
        storeFrom(std::span<const size_t>(indices.begin(), indices.size()), value);
    }

    TensorView asConst() const {
        return TensorView(m_type, m_layout, m_storage);
    }

   private:
    ScalarType m_type;
    Layout m_layout;
    std::span<std::byte> m_storage;
};

class Tensor {
   public:
    Tensor(ScalarType type, Shape shape) : Tensor(type, Layout::contiguous(shape)) {}

    Tensor(ScalarType type, Layout layout)
        : m_type(type),
          m_layout(std::move(layout)),
          m_storage(storageBytesForLayout(m_type, m_layout)) {}

    Tensor(ScalarType type, Layout layout, std::vector<std::byte> storage)
        : m_type(type), m_layout(std::move(layout)), m_storage(std::move(storage)) {
        if (m_storage.size() < storageBytesForLayout(m_type, m_layout))
            throw std::invalid_argument("Tensor storage is too small for its layout.");
    }

    static Tensor fromStorage(ScalarType type, Layout layout, std::vector<std::byte> storage) {
        return Tensor(type, std::move(layout), std::move(storage));
    }

    template <typename Source>
    static Tensor fromValues(ScalarType type, Shape shape, std::span<const Source> values) {
        if (values.size() != shape.elementCount())
            throw std::invalid_argument("Tensor value count does not match shape.");
        Tensor result(type, shape);
        for (size_t index = 0; index < values.size(); ++index)
            detail::encodeScalar(type, result.m_storage, static_cast<ptrdiff_t>(index),
                                 values[index]);
        return result;
    }

    template <typename Source>
    static Tensor fromNativeValues(Shape shape, std::span<const Source> values) {
        return fromValues(nativeScalarType<Source>, std::move(shape), values);
    }

    ScalarType type() const {
        return m_type;
    }

    const Shape& shape() const {
        return m_layout.shape();
    }

    const Layout& layout() const {
        return m_layout;
    }

    size_t size() const {
        return shape().elementCount();
    }

    std::span<const std::byte> storage() const {
        return m_storage;
    }

    std::span<std::byte> mutableStorage() {
        return m_storage;
    }

    TensorView view() const {
        return TensorView(m_type, m_layout, m_storage);
    }

    MutableTensorView mutableView() {
        return MutableTensorView(m_type, m_layout, m_storage);
    }

   private:
    ScalarType m_type;
    Layout m_layout;
    std::vector<std::byte> m_storage;
};
}  // namespace roc::host_validation
