// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
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

// ScalarType is the component's dense table-index enum. Count is its one-past-last,
// non-concrete sentinel: it sizes metadata and dispatch tables and remains the compatibility
// "unspecified type" value for existing callers. APIs requiring a concrete type reject it.
// Other component enums intentionally omit Count; optional semantic state should be represented
// separately instead of extending this sentinel pattern.
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

// Integer policies apply after validating any complex source and before writing destination bits.
template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options = {});

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
        return extent(dimension);
    }

    size_t extent(size_t dimension) const {
        return m_dimensions.at(dimension);
    }

    std::span<const size_t> dimensions() const {
        return m_dimensions;
    }

    size_t elementCount() const {
        return elementCount(0, rank());
    }

    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        if (firstDimension > onePastLastDimension || onePastLastDimension > rank())
            throw std::out_of_range("Tensor shape dimension range is invalid.");

        size_t count = 1;
        for (size_t dimension = firstDimension; dimension < onePastLastDimension; ++dimension)
            count = checkedElementProduct(count, extent(dimension));
        return count;
    }

    size_t elementCountExcluding(size_t excludedDimension) const {
        if (excludedDimension >= rank())
            throw std::out_of_range("Excluded tensor shape dimension is invalid.");

        size_t count = 1;
        for (size_t dimension = 0; dimension < rank(); ++dimension) {
            if (dimension != excludedDimension)
                count = checkedElementProduct(count, extent(dimension));
        }
        return count;
    }

    friend bool operator==(const Shape&, const Shape&) = default;

   private:
    static size_t checkedElementProduct(size_t count, size_t extent) {
        if (extent == 0) return 0;
        if (count > std::numeric_limits<size_t>::max() / extent)
            throw std::overflow_error("Tensor shape element count overflow.");
        return count * extent;
    }

    std::vector<size_t> m_dimensions;
};

class Layout;

namespace detail {
inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout);
}

class Layout {
   public:
    Layout() = default;

    Layout(Shape shape, std::vector<ptrdiff_t> strides, ptrdiff_t offset = 0)
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset) {
        if (rank() != m_strides.size())
            throw std::invalid_argument("Tensor layout rank and stride count differ.");
    }

    // Compatibility spelling for contiguousLastDimensionFastest(). For a matrix this is
    // row-major/C-order: the last dimension has unit stride.
    static Layout contiguous(const Shape& shape) {
        return contiguousLastDimensionFastest(shape);
    }

    static Layout contiguousLastDimensionFastest(const Shape& shape) {
        std::vector<ptrdiff_t> strides(shape.rank(), 1);
        ptrdiff_t stride = 1;
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            strides[index] = stride;
            stride = checkedContiguousStride(stride, shape.extent(index));
        }
        return Layout(shape, std::move(strides));
    }

    // For a matrix this is column-major/Fortran-order: the first dimension has unit stride.
    static Layout contiguousFirstDimensionFastest(const Shape& shape) {
        std::vector<ptrdiff_t> strides(shape.rank(), 1);
        ptrdiff_t stride = 1;
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            strides[dimension] = stride;
            stride = checkedContiguousStride(stride, shape.extent(dimension));
        }
        return Layout(shape, std::move(strides));
    }

    const Shape& shape() const {
        return m_shape;
    }

    size_t rank() const {
        return m_shape.rank();
    }

    size_t extent(size_t dimension) const {
        return m_shape.extent(dimension);
    }

    std::span<const size_t> dimensions() const {
        return m_shape.dimensions();
    }

    size_t elementCount() const {
        return m_shape.elementCount();
    }

    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        return m_shape.elementCount(firstDimension, onePastLastDimension);
    }

    size_t elementCountExcluding(size_t excludedDimension) const {
        return m_shape.elementCountExcluding(excludedDimension);
    }

    std::span<const ptrdiff_t> strides() const {
        return m_strides;
    }

    ptrdiff_t stride(size_t dimension) const {
        return m_strides.at(dimension);
    }

    ptrdiff_t offset() const {
        return m_offset;
    }

    ptrdiff_t elementOffset(std::span<const size_t> indices) const {
        if (indices.size() != rank())
            throw std::invalid_argument("Tensor index rank does not match layout rank.");

        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= extent(dimension))
                throw std::out_of_range("Tensor index exceeds shape.");
            const ptrdiff_t delta = checkedMultiply(indices[dimension], stride(dimension));
            result = checkedAdd(result, delta);
        }
        return result;
    }

    friend bool operator==(const Layout&, const Layout&) = default;

   private:
    static ptrdiff_t checkedContiguousStride(ptrdiff_t stride, size_t extent) {
        if (extent > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("Tensor extent exceeds ptrdiff_t.");
        const ptrdiff_t signedExtent = static_cast<ptrdiff_t>(extent);
        if (signedExtent != 0 && stride > std::numeric_limits<ptrdiff_t>::max() / signedExtent)
            throw std::overflow_error("Tensor contiguous stride overflow.");
        return stride * signedExtent;
    }

    static ptrdiff_t checkedMultiply(size_t value, ptrdiff_t factor) {
        if (value == 0 || factor == 0) return 0;

        const bool negative = factor < 0;
        const uintmax_t factorMagnitude =
            negative ? static_cast<uintmax_t>(-(factor + 1)) + 1 : static_cast<uintmax_t>(factor);
        const uintmax_t limit =
            negative ? static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) + 1
                     : static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max());
        if (static_cast<uintmax_t>(value) > limit / factorMagnitude)
            throw std::overflow_error("Tensor layout offset multiplication overflow.");

        const uintmax_t magnitude = static_cast<uintmax_t>(value) * factorMagnitude;
        if (!negative) return static_cast<ptrdiff_t>(magnitude);
        if (magnitude == limit) return std::numeric_limits<ptrdiff_t>::min();
        return -static_cast<ptrdiff_t>(magnitude);
    }

    static ptrdiff_t checkedAdd(ptrdiff_t left, ptrdiff_t right) {
        if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
            throw std::overflow_error("Tensor layout offset addition overflow.");
        return left + right;
    }

    std::pair<ptrdiff_t, ptrdiff_t> checkedElementBounds() const {
        for (size_t dimension = 0; dimension < rank(); ++dimension) {
            if (extent(dimension) == 0) return {0, -1};
        }

        ptrdiff_t lower = m_offset;
        ptrdiff_t upper = m_offset;
        for (size_t dimension = 0; dimension < rank(); ++dimension) {
            const ptrdiff_t delta = checkedMultiply(extent(dimension) - 1, stride(dimension));
            if (delta < 0)
                lower = checkedAdd(lower, delta);
            else
                upper = checkedAdd(upper, delta);
        }
        return {lower, upper};
    }

    friend std::pair<ptrdiff_t, ptrdiff_t> detail::elementBounds(const Layout& layout);

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

template <typename>
inline constexpr bool AlwaysFalseV = false;

inline constexpr uint64_t integerMask(uint32_t bits) {
    return bits == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << bits) - 1;
}

inline constexpr uint64_t signedIntegerMinimumRaw(uint32_t bits) {
    return uint64_t{1} << (bits - 1);
}

inline constexpr uint64_t integerMaximumRaw(uint32_t bits, bool signedDestination) {
    return signedDestination ? signedIntegerMinimumRaw(bits) - 1 : integerMask(bits);
}

inline constexpr ScalarConversionOptions legacyNativeScalarConversionOptions() {
    return {IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap};
}

inline constexpr ScalarConversionOptions legacyScalarConversionOptions(ScalarType destination) {
    const IntegerOverflow overflow =
        destination == ScalarType::Int4 || destination == ScalarType::Int12
            ? IntegerOverflow::Saturate
            : IntegerOverflow::ModuloWrap;
    return {IntegerRounding::TowardZero, overflow};
}

template <typename Floating>
Floating roundForIntegerConversion(Floating value, IntegerRounding rounding) {
    static_assert(std::is_floating_point_v<Floating>);
    switch (rounding) {
        case IntegerRounding::TowardZero:
            return std::trunc(value);
        case IntegerRounding::NearestEven: {
            Floating integral = 0;
            const Floating fraction = std::modf(value, &integral);
            const Floating magnitude = std::fabs(fraction);
            if (magnitude < Floating{0.5}) return integral;

            const Floating direction = std::signbit(fraction) ? Floating{-1} : Floating{1};
            if (magnitude > Floating{0.5}) return integral + direction;

            const Floating parity = std::fmod(std::fabs(integral), Floating{2});
            return parity == Floating{0} ? integral : integral + direction;
        }
    }
    throw std::invalid_argument("Invalid integer rounding policy.");
}

template <typename Integral>
bool integralFitsIntegerDestination(Integral value, uint32_t bits, bool signedDestination) {
    static_assert(std::is_integral_v<Integral>);
    if (signedDestination) {
        if (bits == 64) return std::in_range<int64_t>(value);
        const int64_t minimum = -(int64_t{1} << (bits - 1));
        const int64_t maximum = (int64_t{1} << (bits - 1)) - 1;
        return !std::cmp_less(value, minimum) && !std::cmp_greater(value, maximum);
    }

    if (std::cmp_less(value, 0)) return false;
    if (bits == 64) return std::in_range<uint64_t>(value);
    return !std::cmp_greater(value, integerMask(bits));
}

template <typename Integral>
bool integralIsBelowIntegerDestination(Integral value, uint32_t bits, bool signedDestination) {
    static_assert(std::is_integral_v<Integral>);
    if (!signedDestination) return std::cmp_less(value, 0);
    if (bits == 64) return std::cmp_less(value, std::numeric_limits<int64_t>::min());
    const int64_t minimum = -(int64_t{1} << (bits - 1));
    return std::cmp_less(value, minimum);
}

template <typename Integral>
uint64_t convertIntegralToIntegerBits(Integral value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) {
    static_assert(std::is_integral_v<Integral>);
    const uint64_t mask = integerMask(bits);
    if (integralFitsIntegerDestination(value, bits, signedDestination)) {
        if (signedDestination) return static_cast<uint64_t>(static_cast<int64_t>(value)) & mask;
        return static_cast<uint64_t>(value) & mask;
    }

    switch (options.integerOverflow) {
        case IntegerOverflow::Reject:
            throw std::overflow_error("Integer value is outside the destination range.");
        case IntegerOverflow::Saturate:
            return integralIsBelowIntegerDestination(value, bits, signedDestination)
                       ? (signedDestination ? signedIntegerMinimumRaw(bits) : 0)
                       : integerMaximumRaw(bits, signedDestination);
        case IntegerOverflow::ModuloWrap:
            return static_cast<uint64_t>(value) & mask;
    }
    throw std::invalid_argument("Invalid integer overflow policy.");
}

template <typename Floating>
uint64_t convertFloatingToIntegerBits(Floating value, uint32_t bits, bool signedDestination,
                                      const ScalarConversionOptions& options) {
    static_assert(std::is_floating_point_v<Floating>);
    if (std::isnan(value))
        throw std::domain_error("NaN cannot be converted to an integer destination.");

    if (std::isinf(value)) {
        if (options.integerOverflow == IntegerOverflow::Saturate) {
            if (std::signbit(value)) return signedDestination ? signedIntegerMinimumRaw(bits) : 0;
            return integerMaximumRaw(bits, signedDestination);
        }
        throw std::overflow_error("Infinity cannot be converted to an integer destination.");
    }

    const Floating rounded = roundForIntegerConversion(value, options.integerRounding);
    const int rangeExponent =
        signedDestination ? static_cast<int>(bits) - 1 : static_cast<int>(bits);
    const Floating upperExclusive = std::ldexp(Floating{1}, rangeExponent);
    const Floating lowerInclusive = signedDestination ? -upperExclusive : Floating{0};
    if (rounded >= lowerInclusive && rounded < upperExclusive) {
        if (signedDestination)
            return static_cast<uint64_t>(static_cast<int64_t>(rounded)) & integerMask(bits);
        return static_cast<uint64_t>(rounded) & integerMask(bits);
    }

    switch (options.integerOverflow) {
        case IntegerOverflow::Reject:
            throw std::overflow_error("Rounded value is outside the destination integer range.");
        case IntegerOverflow::Saturate:
            return rounded < lowerInclusive
                       ? (signedDestination ? signedIntegerMinimumRaw(bits) : 0)
                       : integerMaximumRaw(bits, signedDestination);
        case IntegerOverflow::ModuloWrap: {
            const Floating modulus = std::ldexp(Floating{1}, static_cast<int>(bits));
            const Floating remainder = std::fmod(std::fabs(rounded), modulus);
            uint64_t raw = static_cast<uint64_t>(remainder);
            if (std::signbit(rounded)) raw = uint64_t{0} - raw;
            return raw & integerMask(bits);
        }
    }
    throw std::invalid_argument("Invalid integer overflow policy.");
}

template <typename Source>
uint64_t convertToIntegerBits(Source source, uint32_t bits, bool signedDestination,
                              const ScalarConversionOptions& options) {
    using Value = std::remove_cvref_t<Source>;
    if constexpr (RuntimeIsComplexV<Value>) {
        if (source.imag() != typename Value::value_type{0})
            throw std::domain_error(
                "A complex value with a nonzero imaginary component cannot be converted to real.");
        return convertToIntegerBits(source.real(), bits, signedDestination, options);
    } else if constexpr (std::is_same_v<Value, bool>) {
        return convertIntegralToIntegerBits(static_cast<uint8_t>(source), bits, signedDestination,
                                            options);
    } else if constexpr (std::is_enum_v<Value>) {
        return convertToIntegerBits(static_cast<std::underlying_type_t<Value>>(source), bits,
                                    signedDestination, options);
    } else if constexpr (std::is_integral_v<Value>) {
        return convertIntegralToIntegerBits(source, bits, signedDestination, options);
    } else if constexpr (std::is_floating_point_v<Value>) {
        return convertFloatingToIntegerBits(source, bits, signedDestination, options);
    } else {
        static_assert(AlwaysFalseV<Value>, "Scalar integer conversion requires a numeric source.");
    }
}

template <typename Target>
Target integerFromBits(uint64_t raw) {
    using Value = std::remove_cv_t<Target>;
    static_assert(std::is_integral_v<Value> && !std::is_same_v<Value, bool>);
    using Unsigned = std::make_unsigned_t<Value>;
    constexpr uint32_t bits = std::numeric_limits<Unsigned>::digits;
    static_assert(bits > 0 && bits <= 64);
    raw &= integerMask(bits);

    if constexpr (std::is_unsigned_v<Value>) {
        return static_cast<Value>(raw);
    } else {
        const uint64_t sign = signedIntegerMinimumRaw(bits);
        if ((raw & sign) == 0) return static_cast<Value>(raw);

        const uint64_t magnitude = (uint64_t{0} - raw) & integerMask(bits);
        if (magnitude == sign) return std::numeric_limits<Value>::min();
        return static_cast<Value>(-static_cast<int64_t>(magnitude));
    }
}

template <typename Target, typename Source>
Target convertToInteger(Source source, const ScalarConversionOptions& options) {
    using Value = std::remove_cv_t<Target>;
    static_assert(std::is_integral_v<Value> && !std::is_same_v<Value, bool>);
    using Unsigned = std::make_unsigned_t<Value>;
    constexpr uint32_t bits = std::numeric_limits<Unsigned>::digits;
    static_assert(bits > 0 && bits <= 64);
    const uint64_t raw =
        convertToIntegerBits(std::move(source), bits, std::is_signed_v<Value>, options);
    return integerFromBits<Value>(raw);
}

template <typename Target, typename Source>
Target convertScalarValue(Source source, const ScalarConversionOptions& options) {
    using Result = std::remove_cv_t<Target>;
    using Value = std::remove_cvref_t<Source>;
    static_assert(!std::is_reference_v<Target>);

    if constexpr (RuntimeIsComplexV<Result>) {
        using Component = typename Result::value_type;
        if constexpr (RuntimeIsComplexV<Value>)
            return Result(convertScalarValue<Component>(source.real(), options),
                          convertScalarValue<Component>(source.imag(), options));
        else
            return Result(convertScalarValue<Component>(source, options), Component{0});
    } else if constexpr (RuntimeIsComplexV<Value>) {
        if (source.imag() != typename Value::value_type{0})
            throw std::domain_error(
                "A complex value with a nonzero imaginary component cannot be converted to real.");
        return convertScalarValue<Result>(source.real(), options);
    } else if constexpr (std::is_same_v<Result, bool>) {
        return source != Value{0};
    } else if constexpr (std::is_integral_v<Result>) {
        return convertToInteger<Result>(std::move(source), options);
    } else {
        return static_cast<Result>(source);
    }
}

inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout) {
    return layout.checkedElementBounds();
}

template <typename Function>
void forEachIndex(const Shape& shape, Function&& function) {
    const size_t count = shape.elementCount();
    std::vector<size_t> indices(shape.rank(), 0);
    for (size_t linearIndex = 0; linearIndex < count; ++linearIndex) {
        function(std::span<const size_t>(indices), linearIndex);
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (++indices[index] < shape.extent(index)) break;
            indices[index] = 0;
        }
    }
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
    const uint64_t bytes = totalBits / 8 + static_cast<uint64_t>(totalBits % 8 != 0);
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

template <ScalarType Type>
inline constexpr bool IsBinaryFloatTypeV =
    Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
    Type == ScalarType::Float6E3M2 || Type == ScalarType::Float8E4M3 ||
    Type == ScalarType::Float8E5M2 || Type == ScalarType::Float8E4M3Fnuz ||
    Type == ScalarType::Float8E5M2Fnuz || Type == ScalarType::E5M3 || Type == ScalarType::E4M3;

template <ScalarType Type>
inline constexpr BinaryFloatFormat binaryFloatFormatKnown() {
    static_assert(IsBinaryFloatTypeV<Type>);
    if constexpr (Type == ScalarType::Float4E2M1)
        return {2, 1, 1, 4, true, true, false, 0x7, 0, 0};
    else if constexpr (Type == ScalarType::Float6E2M3)
        return {2, 3, 1, 6, true, true, false, 0x1f, 0, 0};
    else if constexpr (Type == ScalarType::Float6E3M2)
        return {3, 2, 3, 6, true, true, false, 0x1f, 0, 0};
    else if constexpr (Type == ScalarType::Float8E4M3)
        return {4, 3, 7, 8, true, true, false, 0x7e, 0, 0x7f};
    else if constexpr (Type == ScalarType::Float8E5M2)
        return {5, 2, 15, 8, true, true, true, 0x7b, 0x7c, 0x7f};
    else if constexpr (Type == ScalarType::Float8E4M3Fnuz)
        return {4, 3, 8, 8, true, false, false, 0x7f, 0, 0x80};
    else if constexpr (Type == ScalarType::Float8E5M2Fnuz)
        return {5, 2, 16, 8, true, false, false, 0x7f, 0, 0x80};
    else if constexpr (Type == ScalarType::E5M3)
        return {5, 3, 15, 8, false, false, false, 0xfe, 0, 0xff};
    else
        return {4, 3, 7, 7, false, false, false, 0x7e, 0, 0x7f};
}

inline BinaryFloatFormat binaryFloatFormat(ScalarType type) {
    return visitScalarType(type, []<typename Tag>() -> BinaryFloatFormat {
        if constexpr (IsBinaryFloatTypeV<Tag::type>)
            return binaryFloatFormatKnown<Tag::type>();
        else
            throw std::invalid_argument(
                "ScalarType is not a supported binary floating-point format.");
    });
}

template <ScalarType Type>
inline constexpr bool isBinaryFloatNaNKnown(uint32_t raw) {
    static_assert(IsBinaryFloatTypeV<Type>);
    if constexpr (Type == ScalarType::Float8E4M3 || Type == ScalarType::E4M3)
        return (raw & 0x7fU) == 0x7fU;
    else if constexpr (Type == ScalarType::Float8E5M2)
        return (raw & 0x7fU) > 0x7cU;
    else if constexpr (Type == ScalarType::Float8E4M3Fnuz || Type == ScalarType::Float8E5M2Fnuz)
        return raw == 0x80U;
    else if constexpr (Type == ScalarType::E5M3)
        return raw == 0xffU;
    else
        return false;
}

inline bool isBinaryFloatNaN(ScalarType type, uint32_t raw) {
    return visitScalarType(type, [raw]<typename Tag>() {
        if constexpr (IsBinaryFloatTypeV<Tag::type>)
            return isBinaryFloatNaNKnown<Tag::type>(raw);
        else
            return false;
    });
}

inline bool isBinaryFloatInfinity(ScalarType type, uint32_t raw) {
    const BinaryFloatFormat format = binaryFloatFormat(type);
    if (!format.hasInfinity) return false;
    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const uint32_t payloadMask = (1U << format.totalBits) - 1U;
    const uint32_t magnitude = format.hasSign ? raw & (signMask - 1U) : raw & payloadMask;
    return magnitude == format.positiveInfinityRaw;
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

template <ScalarType Type>
std::vector<float> makePositiveFiniteBinaryFloatValues() {
    constexpr BinaryFloatFormat format = binaryFloatFormatKnown<Type>();
    std::vector<float> values(format.maximumPositiveFiniteRaw + 1U);
    for (uint32_t raw = 0; raw <= format.maximumPositiveFiniteRaw; ++raw)
        values[raw] = decodeFiniteBinaryFloatMagnitude(raw, format);
    return values;
}

inline const std::vector<float>& positiveFiniteBinaryFloatValues(ScalarType type) {
    return visitScalarType(type, []<typename Tag>() -> const std::vector<float>& {
        if constexpr (IsBinaryFloatTypeV<Tag::type>) {
            static const auto values = makePositiveFiniteBinaryFloatValues<Tag::type>();
            return values;
        } else {
            throw std::invalid_argument(
                "ScalarType is not a supported binary floating-point format.");
        }
    });
}

inline float decodeBinaryFloat(ScalarType type, uint32_t raw) {
    const auto format = binaryFloatFormat(type);
    if (isBinaryFloatNaN(type, raw)) return std::numeric_limits<float>::quiet_NaN();

    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const bool negative = format.hasSign && (raw & signMask) != 0;
    if (isBinaryFloatInfinity(type, raw))
        return negative ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();

    const uint32_t payloadMask = (1U << format.totalBits) - 1U;
    const uint32_t magnitude = format.hasSign ? raw & (signMask - 1U) : raw & payloadMask;
    const float value = decodeFiniteBinaryFloatMagnitude(magnitude, format);
    return negative ? -value : value;
}

inline uint32_t nearestPositiveBinaryFloatRaw(ScalarType type, float value,
                                              const BinaryFloatFormat& format) {
    const std::vector<float>& values = positiveFiniteBinaryFloatValues(type);
    const auto upperIterator = std::lower_bound(values.begin(), values.end(), value);
    if (upperIterator == values.begin()) return 0;
    if (upperIterator == values.end()) return format.maximumPositiveFiniteRaw;

    const uint32_t upper = static_cast<uint32_t>(std::distance(values.begin(), upperIterator));
    const uint32_t lower = upper - 1U;
    const double lowerValue = static_cast<double>(values[lower]);
    const double upperValue = static_cast<double>(values[upper]);
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
            const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
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

    const uint32_t magnitude = nearestPositiveBinaryFloatRaw(type, value, format);
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

template <ScalarType Type, typename Target>
Target decodeScalarKnown(std::span<const std::byte> storage, ptrdiff_t logicalOffset,
                         const ScalarConversionOptions& options) {
    static_assert(isConcreteScalarType(Type));
    const uint64_t offsetBits = bitOffset(Type, logicalOffset);
    const size_t offsetBytes = static_cast<size_t>(offsetBits / 8);

    if constexpr (Type == ScalarType::Boolean)
        return convertScalarValue<Target>(readNative<uint8_t>(storage, offsetBytes) != 0, options);
    else if constexpr (Type == ScalarType::UInt8)
        return convertScalarValue<Target>(readNative<uint8_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int8)
        return convertScalarValue<Target>(readNative<int8_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::UInt16)
        return convertScalarValue<Target>(readNative<uint16_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int16)
        return convertScalarValue<Target>(readNative<int16_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::UInt32)
        return convertScalarValue<Target>(readNative<uint32_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int32)
        return convertScalarValue<Target>(readNative<int32_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::UInt64)
        return convertScalarValue<Target>(readNative<uint64_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Int64)
        return convertScalarValue<Target>(readNative<int64_t>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Float16)
        return convertScalarValue<Target>(decodeFloat16(readNative<uint16_t>(storage, offsetBytes)),
                                          options);
    else if constexpr (Type == ScalarType::BFloat16)
        return convertScalarValue<Target>(
            decodeBFloat16(readNative<uint16_t>(storage, offsetBytes)), options);
    else if constexpr (Type == ScalarType::Float32)
        return convertScalarValue<Target>(readNative<float>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::Float64)
        return convertScalarValue<Target>(readNative<double>(storage, offsetBytes), options);
    else if constexpr (Type == ScalarType::ComplexFloat32)
        return convertScalarValue<Target>(readNative<std::complex<float>>(storage, offsetBytes),
                                          options);
    else if constexpr (Type == ScalarType::ComplexFloat64)
        return convertScalarValue<Target>(readNative<std::complex<double>>(storage, offsetBytes),
                                          options);
    else if constexpr (Type == ScalarType::Int4)
        return convertScalarValue<Target>(signExtend(readPackedBits(storage, offsetBits, 4), 4),
                                          options);
    else if constexpr (Type == ScalarType::Int12)
        return convertScalarValue<Target>(signExtend(readPackedBits(storage, offsetBits, 12), 12),
                                          options);
    else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                       Type == ScalarType::Float6E3M2 || Type == ScalarType::Float8E4M3 ||
                       Type == ScalarType::Float8E5M2 || Type == ScalarType::Float8E4M3Fnuz ||
                       Type == ScalarType::Float8E5M2Fnuz || Type == ScalarType::E5M3 ||
                       Type == ScalarType::E4M3)
        return convertScalarValue<Target>(
            decodeBinaryFloat(
                Type, readPackedBits(storage, offsetBits, scalarTypeInfo(Type).storageBits)),
            options);
    else if constexpr (Type == ScalarType::E8M0)
        return convertScalarValue<Target>(decodeE8M0(readNative<uint8_t>(storage, offsetBytes)),
                                          options);
}

template <ScalarType Type, typename Target>
Target decodeScalarKnown(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalarKnown<Type, Target>(storage, logicalOffset,
                                           legacyNativeScalarConversionOptions());
}

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset,
                    const ScalarConversionOptions& options) {
    return visitScalarType(type, [&]<typename Tag>() {
        return decodeScalarKnown<Tag::type, Target>(storage, logicalOffset, options);
    });
}

template <typename Target>
Target decodeScalar(ScalarType type, std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalar<Target>(type, storage, logicalOffset,
                                legacyNativeScalarConversionOptions());
}

template <ScalarType Type, typename Source>
void encodeScalarKnown(std::span<std::byte> storage, ptrdiff_t logicalOffset, Source source,
                       const ScalarConversionOptions& options) {
    static_assert(isConcreteScalarType(Type));
    const uint64_t offsetBits = bitOffset(Type, logicalOffset);
    const size_t offsetBytes = static_cast<size_t>(offsetBits / 8);

    if constexpr (Type == ScalarType::Boolean)
        writeNative<uint8_t>(storage, offsetBytes,
                             convertScalarValue<bool>(source, options) ? uint8_t{1} : uint8_t{0});
    else if constexpr (Type == ScalarType::UInt8)
        writeNative<uint8_t>(storage, offsetBytes, convertScalarValue<uint8_t>(source, options));
    else if constexpr (Type == ScalarType::Int8)
        writeNative<int8_t>(storage, offsetBytes, convertScalarValue<int8_t>(source, options));
    else if constexpr (Type == ScalarType::UInt16)
        writeNative<uint16_t>(storage, offsetBytes, convertScalarValue<uint16_t>(source, options));
    else if constexpr (Type == ScalarType::Int16)
        writeNative<int16_t>(storage, offsetBytes, convertScalarValue<int16_t>(source, options));
    else if constexpr (Type == ScalarType::UInt32)
        writeNative<uint32_t>(storage, offsetBytes, convertScalarValue<uint32_t>(source, options));
    else if constexpr (Type == ScalarType::Int32)
        writeNative<int32_t>(storage, offsetBytes, convertScalarValue<int32_t>(source, options));
    else if constexpr (Type == ScalarType::UInt64)
        writeNative<uint64_t>(storage, offsetBytes, convertScalarValue<uint64_t>(source, options));
    else if constexpr (Type == ScalarType::Int64)
        writeNative<int64_t>(storage, offsetBytes, convertScalarValue<int64_t>(source, options));
    else if constexpr (Type == ScalarType::Float16)
        writeNative<uint16_t>(storage, offsetBytes,
                              encodeFloat16(convertScalarValue<float>(source, options)));
    else if constexpr (Type == ScalarType::BFloat16)
        writeNative<uint16_t>(storage, offsetBytes,
                              encodeBFloat16(convertScalarValue<float>(source, options)));
    else if constexpr (Type == ScalarType::Float32)
        writeNative<float>(storage, offsetBytes, convertScalarValue<float>(source, options));
    else if constexpr (Type == ScalarType::Float64)
        writeNative<double>(storage, offsetBytes, convertScalarValue<double>(source, options));
    else if constexpr (Type == ScalarType::ComplexFloat32)
        writeNative<std::complex<float>>(storage, offsetBytes,
                                         convertScalarValue<std::complex<float>>(source, options));
    else if constexpr (Type == ScalarType::ComplexFloat64)
        writeNative<std::complex<double>>(
            storage, offsetBytes, convertScalarValue<std::complex<double>>(source, options));
    else if constexpr (Type == ScalarType::Int4)
        writePackedBits(storage, offsetBits, 4,
                        static_cast<uint32_t>(convertToIntegerBits(source, 4, true, options)));
    else if constexpr (Type == ScalarType::Int12)
        writePackedBits(storage, offsetBits, 12,
                        static_cast<uint32_t>(convertToIntegerBits(source, 12, true, options)));
    else if constexpr (Type == ScalarType::Float4E2M1 || Type == ScalarType::Float6E2M3 ||
                       Type == ScalarType::Float6E3M2 || Type == ScalarType::Float8E4M3 ||
                       Type == ScalarType::Float8E5M2 || Type == ScalarType::Float8E4M3Fnuz ||
                       Type == ScalarType::Float8E5M2Fnuz || Type == ScalarType::E5M3 ||
                       Type == ScalarType::E4M3)
        writePackedBits(storage, offsetBits, scalarTypeInfo(Type).storageBits,
                        encodeBinaryFloat(Type, convertScalarValue<float>(source, options)));
    else if constexpr (Type == ScalarType::E8M0)
        writeNative<uint8_t>(storage, offsetBytes,
                             encodeE8M0(convertScalarValue<float>(source, options)));
}

template <ScalarType Type, typename Source>
void encodeScalarKnown(std::span<std::byte> storage, ptrdiff_t logicalOffset, Source source) {
    encodeScalarKnown<Type>(storage, logicalOffset, std::move(source),
                            legacyScalarConversionOptions(Type));
}

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source, const ScalarConversionOptions& options) {
    visitScalarType(type, [&]<typename Tag>() {
        encodeScalarKnown<Tag::type>(storage, logicalOffset, std::move(source), options);
    });
}

template <typename Source>
void encodeScalar(ScalarType type, std::span<std::byte> storage, ptrdiff_t logicalOffset,
                  Source source) {
    encodeScalar(type, storage, logicalOffset, std::move(source),
                 legacyScalarConversionOptions(type));
}
}  // namespace detail

template <typename Target, typename Source>
Target convertScalar(Source source, const ScalarConversionOptions& options) {
    return detail::convertScalarValue<Target>(std::move(source), options);
}

inline size_t storageBytesForLayout(ScalarType type, const Layout& layout) {
    return detail::storageBytesForLayout(type, layout);
}

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

class TensorStorage {
   public:
    TensorStorage() = default;

    TensorStorage(std::shared_ptr<void> owner, std::span<std::byte> bytes)
        : m_owner(std::move(owner)), m_bytes(bytes) {
        if (!m_owner && !m_bytes.empty())
            throw std::invalid_argument("Nonempty TensorStorage requires an owner.");
    }

    static TensorStorage allocate(size_t bytes) {
        auto owner = std::make_shared<std::vector<std::byte>>(bytes);
        return TensorStorage(owner, std::span<std::byte>(*owner));
    }

    std::span<const std::byte> bytes() const {
        return m_bytes;
    }

    std::span<std::byte> mutableBytes() const {
        return m_bytes;
    }

    size_t size() const {
        return m_bytes.size();
    }

   private:
    std::shared_ptr<void> m_owner;
    std::span<std::byte> m_bytes;
};

using TensorStorageAllocator = std::function<TensorStorage(size_t)>;

class Tensor {
   public:
    Tensor(ScalarType type, Shape shape) : Tensor(type, Layout::contiguous(shape)) {}

    Tensor(ScalarType type, Layout layout)
        : Tensor(type, std::move(layout), TensorStorage::allocate) {}

    Tensor(ScalarType type, Shape shape, const TensorStorageAllocator& allocator)
        : Tensor(type, Layout::contiguous(shape), allocator) {}

    Tensor(ScalarType type, Layout layout, const TensorStorageAllocator& allocator)
        : m_type(type), m_layout(std::move(layout)) {
        if (!allocator) throw std::invalid_argument("Tensor storage allocator is empty.");
        m_storage = allocator(::roc::host_validation::storageBytesForLayout(m_type, m_layout));
        validateStorage();
    }

    Tensor(ScalarType type, Layout layout, std::vector<std::byte> storage)
        : Tensor(type, std::move(layout), storageFromVector(std::move(storage))) {}

    Tensor(ScalarType type, Layout layout, std::span<const std::byte> storage)
        : Tensor(type, std::move(layout), std::vector<std::byte>(storage.begin(), storage.end())) {}

    Tensor(ScalarType type, Layout layout, std::span<std::byte> storage)
        : Tensor(type, std::move(layout), std::span<const std::byte>(storage)) {}

    static Tensor fromStorage(ScalarType type, Layout layout, TensorStorage storage) {
        return Tensor(type, std::move(layout), std::move(storage));
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
            detail::encodeScalar(type, result.storage(), static_cast<ptrdiff_t>(index),
                                 values[index]);
        return result;
    }

    template <typename Source>
    static Tensor fromValues(ScalarType type, Shape shape, std::span<const Source> values,
                             const ScalarConversionOptions& options) {
        if (values.size() != shape.elementCount())
            throw std::invalid_argument("Tensor value count does not match shape.");
        Tensor result(type, shape);
        for (size_t index = 0; index < values.size(); ++index)
            detail::encodeScalar(type, result.storage(), static_cast<ptrdiff_t>(index),
                                 values[index], options);
        return result;
    }

    template <typename Source>
    static Tensor fromNativeValues(Shape shape, std::span<const Source> values) {
        return fromValues(nativeScalarType<Source>, std::move(shape), values);
    }

    template <typename Source>
    static Tensor fromNative(Layout layout, std::span<const Source> values) {
        constexpr ScalarType type = nativeScalarType<Source>;
        static_assert(scalarTypeInfo(type).storageBits == sizeof(Source) * 8,
                      "Native Tensor storage requires one scalar per C++ object.");
        const std::span<const std::byte> bytes = std::as_bytes(values);
        const size_t required = ::roc::host_validation::storageBytesForLayout(type, layout);
        if (bytes.size() < required)
            throw std::invalid_argument("Native Tensor storage is too small for its layout.");
        return Tensor(type, std::move(layout), bytes.first(required));
    }

    template <typename Source>
    static Tensor fromNative(Layout layout, std::span<Source> values) {
        return fromNative(std::move(layout), std::span<const Source>(values));
    }

    template <typename Source>
    static Tensor fromNative(std::span<const Source> values) {
        return fromNative(Layout::contiguous(Shape{values.size()}), values);
    }

    template <typename Source>
    static Tensor fromNative(std::span<Source> values) {
        return fromNative(std::span<const Source>(values));
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
        return m_layout.elementCount();
    }

    std::span<std::byte> storage() const {
        return m_storage.mutableBytes();
    }

    template <typename Target>
    Target loadAs(std::span<const size_t> indices) const {
        return detail::decodeScalar<Target>(m_type, storage(), m_layout.elementOffset(indices));
    }

    template <typename Target>
    Target loadAs(std::span<const size_t> indices, const ScalarConversionOptions& options) const {
        return detail::decodeScalar<Target>(m_type, storage(), m_layout.elementOffset(indices),
                                            options);
    }

    template <typename Target>
    Target loadAs(std::initializer_list<size_t> indices) const {
        return loadAs<Target>(std::span<const size_t>(indices.begin(), indices.size()));
    }

    template <typename Target>
    Target loadAs(std::initializer_list<size_t> indices,
                  const ScalarConversionOptions& options) const {
        return loadAs<Target>(std::span<const size_t>(indices.begin(), indices.size()), options);
    }

    template <typename Source>
    void storeFrom(std::span<const size_t> indices, Source value) const {
        detail::encodeScalar(m_type, storage(), m_layout.elementOffset(indices), value);
    }

    template <typename Source>
    void storeFrom(std::span<const size_t> indices, Source value,
                   const ScalarConversionOptions& options) const {
        detail::encodeScalar(m_type, storage(), m_layout.elementOffset(indices), std::move(value),
                             options);
    }

    template <typename Source>
    void storeFrom(std::initializer_list<size_t> indices, Source value) const {
        storeFrom(std::span<const size_t>(indices.begin(), indices.size()), value);
    }

    template <typename Source>
    void storeFrom(std::initializer_list<size_t> indices, Source value,
                   const ScalarConversionOptions& options) const {
        storeFrom(std::span<const size_t>(indices.begin(), indices.size()), std::move(value),
                  options);
    }

    Tensor alias(Layout layout) const {
        return Tensor(m_type, std::move(layout), m_storage);
    }

    Tensor clone() const {
        return clone(TensorStorage::allocate);
    }

    Tensor clone(const TensorStorageAllocator& allocator) const {
        Tensor result(m_type, m_layout, allocator);
        const size_t required = ::roc::host_validation::storageBytesForLayout(m_type, m_layout);
        std::ranges::copy(storage().first(required), result.storage().begin());
        return result;
    }

    void copyFrom(const Tensor& source) const {
        if (m_type != source.m_type)
            throw std::invalid_argument("Tensor copy requires matching scalar types.");
        if (shape() != source.shape())
            throw std::invalid_argument("Tensor copy requires matching shapes.");
        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        detail::forEachIndex(shape(), [&](std::span<const size_t> indices, size_t) {
            detail::copyBitRange(
                source.storage(), detail::bitOffset(m_type, source.layout().elementOffset(indices)),
                storage(), detail::bitOffset(m_type, layout().elementOffset(indices)), bits);
        });
    }

    void copyFrom(const Tensor& source, std::span<const size_t> linearIndices) const {
        if (m_type != source.m_type)
            throw std::invalid_argument("Tensor copy requires matching scalar types.");
        if (shape() != source.shape())
            throw std::invalid_argument("Tensor copy requires matching shapes.");

        const size_t count = size();
        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        std::vector<size_t> indices(shape().rank(), 0);
        for (const size_t linearIndex : linearIndices) {
            if (linearIndex >= count)
                throw std::out_of_range("Tensor copy index exceeds the logical element count.");

            size_t remaining = linearIndex;
            for (size_t dimension = shape().rank(); dimension > 0; --dimension) {
                const size_t index = dimension - 1;
                indices[index] = remaining % shape().extent(index);
                remaining /= shape().extent(index);
            }
            detail::copyBitRange(
                source.storage(), detail::bitOffset(m_type, source.layout().elementOffset(indices)),
                storage(), detail::bitOffset(m_type, layout().elementOffset(indices)), bits);
        }
    }

    Tensor to(ScalarType type) const;

    Tensor to(ScalarType type, const ScalarConversionOptions& options) const;

   private:
    Tensor(ScalarType type, Layout layout, TensorStorage storage)
        : m_type(type), m_layout(std::move(layout)), m_storage(std::move(storage)) {
        validateStorage();
    }

    static TensorStorage storageFromVector(std::vector<std::byte> storage) {
        auto owner = std::make_shared<std::vector<std::byte>>(std::move(storage));
        return TensorStorage(owner, std::span<std::byte>(*owner));
    }

    void validateStorage() const {
        if (m_storage.size() < ::roc::host_validation::storageBytesForLayout(m_type, m_layout))
            throw std::invalid_argument("Tensor storage is too small for its layout.");
    }

    ScalarType m_type;
    Layout m_layout;
    TensorStorage m_storage;
};

inline Tensor Tensor::to(ScalarType destinationType) const {
    return to(destinationType, detail::legacyScalarConversionOptions(destinationType));
}

inline Tensor Tensor::to(ScalarType destinationType, const ScalarConversionOptions& options) const {
    const ScalarType sourceType = type();
    const Layout& sourceLayout = layout();
    const std::span<const std::byte> sourceStorage = storage();
    const size_t requiredStorage =
        ::roc::host_validation::storageBytesForLayout(sourceType, sourceLayout);
    if (destinationType == sourceType)
        return Tensor::fromStorage(
            destinationType, sourceLayout,
            std::vector<std::byte>(sourceStorage.begin(), sourceStorage.begin() + requiredStorage));

    Tensor result(destinationType, sourceLayout);
    const Tensor destination = result;
    visitScalarType(sourceType, [&]<typename SourceTag>() {
        visitScalarType(destinationType, [&]<typename DestinationTag>() {
            detail::forEachIndex(sourceLayout.shape(), [&](std::span<const size_t> indices,
                                                           size_t) {
                const ptrdiff_t sourceOffset = sourceLayout.elementOffset(indices);
                const ptrdiff_t destinationOffset = destination.layout().elementOffset(indices);
                constexpr ScalarCategory sourceCategory = scalarTypeInfo(SourceTag::type).category;
                if constexpr (sourceCategory == ScalarCategory::Boolean ||
                              sourceCategory == ScalarCategory::UnsignedInteger) {
                    const uint64_t value = detail::decodeScalarKnown<SourceTag::type, uint64_t>(
                        sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.storage(), destinationOffset, value, options);
                } else if constexpr (sourceCategory == ScalarCategory::SignedInteger) {
                    const int64_t value = detail::decodeScalarKnown<SourceTag::type, int64_t>(
                        sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.storage(), destinationOffset, value, options);
                } else if constexpr (sourceCategory == ScalarCategory::Complex) {
                    const std::complex<double> value =
                        detail::decodeScalarKnown<SourceTag::type, std::complex<double>>(
                            sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.storage(), destinationOffset, value, options);
                } else {
                    const double value = detail::decodeScalarKnown<SourceTag::type, double>(
                        sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.storage(), destinationOffset, value, options);
                }
            });
        });
    });
    return result;
}
}  // namespace roc::host_validation
