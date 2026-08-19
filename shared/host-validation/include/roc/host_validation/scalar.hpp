// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace roc::host_validation {
// Scalar types define the runtime numeric vocabulary shared by every host-validation component.
// This header owns type identity, metadata, native C++ mappings, runtime dispatch, conversion
// policy, and the owning Scalar value. Encoding details live in detail/scalar_codec.hpp because
// Scalar and Tensor templates require their definitions; callers address scalar.hpp as the API.
enum class ScalarCategory : uint8_t {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
    Complex,
    Scale,
};

// Dense table index for every supported scalar encoding. Count is one-past-last
// and exists only to size and exhaustively validate metadata/dispatch tables.
// APIs reject Count as a scalar value; absence uses std::optional<ScalarType>.
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
    // Int12 is intentionally not tied to a current hardware type. It proves that scalar storage,
    // packing, and conversion remain general for elements that cross byte boundaries and are not
    // power-of-two widths. Keep its scalar and tensor codec coverage.
    Int12,
    E8M0,
    E5M3,
    E4M3,
    Count,
};

enum class IntegerRounding : uint8_t {
    TowardZero,
    NearestEven,
};

enum class IntegerOverflow : uint8_t {
    Reject,
    Saturate,
    // Reduce modulo 2^N, then interpret the N-bit result as two's complement when signed.
    ModuloWrap,
};

struct ScalarConversionOptions {
    // Explicit-options conversions reject overflow by default. Legacy overloads keep their
    // destination-specific behavior for source compatibility.
    IntegerRounding integerRounding = IntegerRounding::TowardZero;
    IntegerOverflow integerOverflow = IntegerOverflow::Reject;
};

struct ScalarTypeInfo {
    // Short human-readable spelling used in diagnostics and bindings.
    std::string_view name;
    // Broad conversion and comparison behavior of the scalar.
    ScalarCategory category;
    // Encoded bits occupied by one logical scalar, including both complex components and any
    // packed or reserved payload bits.
    uint16_t storageBits;
    // Encoded exponent bits per real component; zero for non-floating categories.
    uint8_t exponentBits;
    // Explicitly stored fraction bits per real component, excluding any implicit leading bit.
    uint8_t mantissaBits;
    // Bias subtracted from an encoded exponent; zero when exponentBits is zero.
    int16_t exponentBias;
    // Whether the encoding has at least one representation for NaN.
    bool supportsNaN;
    // Whether the encoding has representations for positive and negative infinity.
    bool supportsInfinity;

    bool isPacked() const {
        return storageBits % 8 != 0;
    }
};

inline constexpr size_t scalarTypeCount = static_cast<size_t>(ScalarType::Count);

inline constexpr bool isConcreteScalarType(ScalarType type) {
    return static_cast<size_t>(type) < scalarTypeCount;
}

// Columns follow ScalarTypeInfo's fields: name, category, storage bits, exponent bits,
// mantissa bits, exponent bias, NaN support, and infinity support.
inline constexpr std::array<ScalarTypeInfo, scalarTypeCount> scalarTypeInfos{{
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
    {"e4m3", ScalarCategory::Scale, 8, 4, 3, 7, true, false},
}};

inline constexpr const ScalarTypeInfo& scalarTypeInfo(ScalarType type) {
    if (!isConcreteScalarType(type)) throw std::invalid_argument("Invalid ScalarType.");
    return scalarTypeInfos[static_cast<size_t>(type)];
}

inline constexpr std::string_view scalarTypeName(ScalarType type) {
    return scalarTypeInfo(type).name;
}

inline constexpr size_t scalarElementGroupSize(ScalarType type) {
    const size_t bits = scalarTypeInfo(type).storageBits;
    return 8 / std::gcd(bits, size_t{8});
}

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
using E4M3Tag = ScalarTag<ScalarType::E4M3, uint8_t>;
}  // namespace detail

// Runtime dispatch maps every concrete ScalarType to a compile-time tag carrying its enum value,
// encoded width, and native storage type when one C++ object represents one scalar.
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
        case ScalarType::E4M3:
            return std::forward<Visitor>(visitor).template operator()<detail::E4M3Tag>(
                std::forward<Args>(args)...);
        case ScalarType::Count:
            break;
    }
    throw std::invalid_argument("Invalid ScalarType.");
}

namespace detail {
template <typename Target, typename Source>
Target convertScalarValue(Source source, const ScalarConversionOptions& options);

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset);

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset,
                    const ScalarConversionOptions& options);

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source);

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source, const ScalarConversionOptions& options);
}  // namespace detail

// Convert one native numeric value without encoding it. Complex-to-real conversion rejects a
// nonzero imaginary component. Integer conversion applies the supplied rounding and overflow
// policy after complex validation.
template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options = {}) {
    return detail::convertScalarValue<Target>(std::move(source), options);
}

// One runtime-typed encoded value stored inline. Scalar has ordinary deep value
// semantics, no shape or strides, and never aliases external storage.
class Scalar {
   public:
    static constexpr size_t maximumStorageBytes = 16;

    template <typename Source>
        requires(!std::is_same_v<std::remove_cvref_t<Source>, Scalar> &&
                 requires { nativeScalarType<std::remove_cvref_t<Source>>; })
    Scalar(Source value) : m_type(nativeScalarType<std::remove_cvref_t<Source>>) {
        detail::encodeScalar(m_type, m_storage, 0, std::move(value));
    }

    template <typename Source>
    static Scalar from(Source value) {
        return Scalar(std::move(value));
    }

    static Scalar fromStorage(ScalarType type, std::span<const std::byte> storage) {
        Scalar result(type);
        if (storage.size() != result.storageSize())
            throw std::invalid_argument("Scalar storage size does not match its type.");
        std::copy(storage.begin(), storage.end(), result.m_storage.begin());
        const uint16_t remainder = scalarTypeInfo(type).storageBits % 8;
        if (remainder != 0) {
            const uint8_t mask = static_cast<uint8_t>((1U << remainder) - 1U);
            const size_t finalByte = result.storageSize() - 1;
            result.m_storage[finalByte] = static_cast<std::byte>(
                std::to_integer<uint8_t>(result.m_storage[finalByte]) & mask);
        }
        return result;
    }

    static Scalar zero(ScalarType type) {
        Scalar result(type);
        detail::encodeScalar(type, result.m_storage, 0, int64_t{0});
        return result;
    }

    static Scalar one(ScalarType type) {
        Scalar result(type);
        detail::encodeScalar(type, result.m_storage, 0, int64_t{1});
        return result;
    }

    ScalarType type() const {
        return m_type;
    }

    std::span<const std::byte> storage() const {
        return std::span<const std::byte>(m_storage).first(storageSize());
    }

    template <typename Target>
    Target as() const {
        return detail::decodeScalar<Target>(m_type, m_storage, 0);
    }

    template <typename Target>
    Target as(const ScalarConversionOptions& options) const {
        return detail::decodeScalar<Target>(m_type, m_storage, 0, options);
    }

    friend bool operator==(const Scalar& left, const Scalar& right) {
        return left.m_type == right.m_type &&
               std::equal(left.storage().begin(), left.storage().end(), right.storage().begin(),
                          right.storage().end());
    }

   private:
    explicit Scalar(ScalarType type) : m_type(type) {
        if (!isConcreteScalarType(type))
            throw std::invalid_argument("Scalar requires a concrete scalar type.");
        if (storageSize() > maximumStorageBytes)
            throw std::invalid_argument("Scalar type exceeds inline scalar storage.");
    }

    size_t storageSize() const {
        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        return bits / 8 + static_cast<size_t>(bits % 8 != 0);
    }

    ScalarType m_type;
    std::array<std::byte, maximumStorageBytes> m_storage{};
};
}  // namespace roc::host_validation

// Scalar and Tensor are header-only. Install this implementation dependency with the public
// headers, but keep it outside the public namespace and include surface.
#include <roc/host_validation/detail/scalar_codec.hpp>
