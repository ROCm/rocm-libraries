// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Shared runtime arithmetic helpers for compiled reference operations.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_validation/operation_types.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace roc::host_validation {
namespace detail {
template <typename T>
struct IsComplex : std::false_type {};

template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
T wrappingAdd(T left, T right) {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result = static_cast<Unsigned>(left) + static_cast<Unsigned>(right);
        return std::bit_cast<T>(result);
    } else {
        return left + right;
    }
}

template <typename T>
T wrappingMultiply(T left, T right) {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result = static_cast<Unsigned>(left) * static_cast<Unsigned>(right);
        return std::bit_cast<T>(result);
    } else {
        return left * right;
    }
}

template <typename T>
T wrappingNegate(T value) {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        return std::bit_cast<T>(Unsigned{0} - static_cast<Unsigned>(value));
    } else {
        return -value;
    }
}

template <typename T>
T conjugateIfNeeded(const T& value, bool conjugate) {
    if constexpr (IsComplex<T>::value)
        return conjugate ? std::conj(value) : value;
    else
        return value;
}

template <typename Accumulator>
Accumulator applyActivation(Activation activation, Accumulator value, Accumulator parameter0,
                            Accumulator parameter1) {
    if constexpr (IsComplex<Accumulator>::value) {
        if (activation != Activation::None)
            throw std::invalid_argument(
                "Complex reference arithmetic does not support activation.");
        return value;
    } else {
        using Transcendental =
            std::conditional_t<std::is_same_v<Accumulator, double>, double, float>;
        switch (activation) {
            case Activation::None:
                return value;
            case Activation::Absolute:
                return value >= Accumulator(0) ? value : wrappingNegate(value);
            case Activation::ClippedRelu:
                return value > parameter0 ? std::min(value, parameter1)
                                          : std::min(Accumulator(0), parameter1);
            case Activation::Relu:
                return std::max(Accumulator(0), value);
            case Activation::Gelu: {
                constexpr Transcendental coefficient0 =
                    static_cast<Transcendental>(0.7978845608028654);
                constexpr Transcendental coefficient1 = static_cast<Transcendental>(0.044715);
                const Transcendental x = static_cast<Transcendental>(value);
                return static_cast<Accumulator>(
                    Transcendental(0.5) * x *
                    (Transcendental(1) +
                     std::tanh(coefficient0 * x * (Transcendental(1) + coefficient1 * x * x))));
            }
            case Activation::GeluDerivative: {
                constexpr Transcendental coefficient0 = static_cast<Transcendental>(0.0535161);
                constexpr Transcendental coefficient1 = static_cast<Transcendental>(0.398942);
                constexpr Transcendental coefficient2 = static_cast<Transcendental>(0.0356774);
                constexpr Transcendental coefficient3 = static_cast<Transcendental>(0.797885);
                const Transcendental x = static_cast<Transcendental>(value);
                const Transcendental cube = x * x * x;
                const Transcendental first = coefficient0 * cube + coefficient1 * x;
                const Transcendental second = coefficient2 * cube + coefficient3 * x;
                const Transcendental derivative =
                    Transcendental(0.5) * std::tanh(second) +
                    first *
                        (Transcendental(4) / std::pow(std::exp(-second) + std::exp(second), 2)) +
                    Transcendental(0.5);
                return static_cast<Accumulator>(derivative);
            }
            case Activation::GeluScaling:
                return applyActivation(Activation::Gelu, value, parameter0, parameter1) *
                       parameter0;
            case Activation::LeakyRelu:
                return value > Accumulator(0) ? value : wrappingMultiply(value, parameter0);
            case Activation::ReluDerivative:
                return value > Accumulator(0) ? Accumulator(1) : Accumulator(0);
            case Activation::Sigmoid: {
                const Transcendental x = static_cast<Transcendental>(value);
                return static_cast<Accumulator>(Transcendental(1) /
                                                (Transcendental(1) + std::exp(-x)));
            }
            case Activation::Tanh:
                return static_cast<Accumulator>(
                    std::tanh(static_cast<Transcendental>(value * parameter0)) *
                    static_cast<Transcendental>(parameter1));
            case Activation::Silu: {
                const Transcendental x = static_cast<Transcendental>(value);
                return static_cast<Accumulator>(x / (Transcendental(1) + std::exp(-x)));
            }
            case Activation::Swish: {
                const Transcendental x = static_cast<Transcendental>(value);
                const Transcendental beta = static_cast<Transcendental>(parameter0);
                return static_cast<Accumulator>(x / (Transcendental(1) + std::exp(-beta * x)));
            }
            case Activation::Clamp:
                return std::max(parameter0, std::min(value, parameter1));
        }
    }

    throw std::invalid_argument("Unsupported reference activation.");
}

inline bool isComplexScalarType(ScalarType type) {
    return scalarTypeInfo(type).category == ScalarCategory::Complex;
}

inline bool isScaleScalarType(ScalarType type) {
    return scalarTypeInfo(type).category == ScalarCategory::Scale;
}

template <typename Accumulator>
using RuntimeLoadFunction = Accumulator (*)(std::span<const std::byte>, ptrdiff_t);

template <typename Accumulator>
using RuntimeStoreFunction = void (*)(std::span<std::byte>, ptrdiff_t, Accumulator);

template <typename Accumulator, typename Tag>
Accumulator runtimeLoadScalar(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalarKnown<Tag::type, Accumulator>(storage, logicalOffset);
}

template <typename Accumulator, typename Tag>
void runtimeStoreScalar(std::span<std::byte> storage, ptrdiff_t logicalOffset, Accumulator value) {
    encodeScalarKnown<Tag::type>(storage, logicalOffset, value);
}

template <typename Accumulator>
RuntimeLoadFunction<Accumulator> runtimeLoadFunction(ScalarType type) {
    return visitScalarType(type,
                           []<typename Tag>() { return &runtimeLoadScalar<Accumulator, Tag>; });
}

template <typename Accumulator>
RuntimeStoreFunction<Accumulator> runtimeStoreFunction(ScalarType type) {
    return visitScalarType(type,
                           []<typename Tag>() { return &runtimeStoreScalar<Accumulator, Tag>; });
}

template <typename Accumulator>
class RuntimeMatrixReader {
   public:
    explicit RuntimeMatrixReader(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_offset(view.layout().offset()),
          m_rowStride(view.layout().strides()[0]),
          m_columnStride(view.layout().strides()[1]),
          m_load(runtimeLoadFunction<Accumulator>(view.type())) {}

    Accumulator operator()(size_t row, size_t column) const {
        return m_load(m_storage, m_offset + static_cast<ptrdiff_t>(row) * m_rowStride +
                                     static_cast<ptrdiff_t>(column) * m_columnStride);
    }

   private:
    std::span<const std::byte> m_storage;
    ptrdiff_t m_offset;
    ptrdiff_t m_rowStride;
    ptrdiff_t m_columnStride;
    RuntimeLoadFunction<Accumulator> m_load;
};

template <typename Accumulator>
class RuntimeMatrixWriter {
   public:
    explicit RuntimeMatrixWriter(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_offset(view.layout().offset()),
          m_rowStride(view.layout().strides()[0]),
          m_columnStride(view.layout().strides()[1]),
          m_store(runtimeStoreFunction<Accumulator>(view.type())) {}

    void store(size_t row, size_t column, Accumulator value) const {
        m_store(m_storage,
                m_offset + static_cast<ptrdiff_t>(row) * m_rowStride +
                    static_cast<ptrdiff_t>(column) * m_columnStride,
                value);
    }

   private:
    std::span<std::byte> m_storage;
    ptrdiff_t m_offset;
    ptrdiff_t m_rowStride;
    ptrdiff_t m_columnStride;
    RuntimeStoreFunction<Accumulator> m_store;
};

template <typename Accumulator>
class RuntimeOutputConverter {
   public:
    RuntimeOutputConverter(ScalarType outputType, OutputConversion conversion)
        : m_outputType(outputType),
          m_conversion(conversion),
          m_load(runtimeLoadFunction<Accumulator>(outputType)),
          m_store(runtimeStoreFunction<Accumulator>(outputType)) {
        if (m_conversion == OutputConversion::SaturatingInt8 && m_outputType != ScalarType::Int8)
            throw std::invalid_argument(
                "Saturating output conversion requires an Int8 output tensor.");
    }

    Accumulator operator()(Accumulator value) const {
        if (m_conversion == OutputConversion::Default) {
            std::array<std::byte, 16> storage{};
            m_store(storage, 0, value);
            return m_load(storage, 0);
        }

        if constexpr (IsComplex<Accumulator>::value) {
            throw std::invalid_argument(
                "Saturating output conversion does not accept complex values.");
        } else if constexpr (std::is_integral_v<Accumulator>) {
            const Accumulator clamped =
                std::clamp(value, static_cast<Accumulator>(-128), static_cast<Accumulator>(127));
            return static_cast<Accumulator>(static_cast<int8_t>(clamped));
        } else {
            const long double rounded = std::nearbyint(static_cast<long double>(value));
            const long double clamped =
                std::clamp(rounded, static_cast<long double>(-128), static_cast<long double>(127));
            return static_cast<Accumulator>(static_cast<int8_t>(clamped));
        }
    }

   private:
    ScalarType m_outputType;
    OutputConversion m_conversion;
    RuntimeLoadFunction<Accumulator> m_load;
    RuntimeStoreFunction<Accumulator> m_store;
};

template <typename Accumulator>
class RuntimeMatrixOutputWriter {
   public:
    RuntimeMatrixOutputWriter(const Tensor& output, OutputConversion conversion)
        : m_defaultWriter(output),
          m_converter(output.type(), conversion),
          m_conversion(conversion) {}

    void store(size_t row, size_t column, Accumulator value) const {
        if (m_conversion == OutputConversion::Default)
            m_defaultWriter.store(row, column, value);
        else
            m_defaultWriter.store(row, column, m_converter(value));
    }

   private:
    RuntimeMatrixWriter<Accumulator> m_defaultWriter;
    RuntimeOutputConverter<Accumulator> m_converter;
    OutputConversion m_conversion;
};

template <typename Accumulator>
class RuntimeVectorReader {
   public:
    explicit RuntimeVectorReader(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_offset(view.layout().offset()),
          m_stride(view.layout().strides()[0]),
          m_load(runtimeLoadFunction<Accumulator>(view.type())) {}

    Accumulator operator[](size_t index) const {
        return m_load(m_storage, m_offset + static_cast<ptrdiff_t>(index) * m_stride);
    }

   private:
    std::span<const std::byte> m_storage;
    ptrdiff_t m_offset;
    ptrdiff_t m_stride;
    RuntimeLoadFunction<Accumulator> m_load;
};

template <typename Accumulator>
class RuntimeTensorReader {
   public:
    explicit RuntimeTensorReader(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_layout(view.layout()),
          m_load(runtimeLoadFunction<Accumulator>(view.type())) {}

    Accumulator operator()(std::span<const size_t> indices) const {
        return m_load(m_storage, m_layout.elementOffset(indices));
    }

   private:
    std::span<const std::byte> m_storage;
    Layout m_layout;
    RuntimeLoadFunction<Accumulator> m_load;
};

template <typename Accumulator>
class RuntimeTensorWriter {
   public:
    explicit RuntimeTensorWriter(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_layout(view.layout()),
          m_store(runtimeStoreFunction<Accumulator>(view.type())) {}

    void store(std::span<const size_t> indices, Accumulator value) const {
        m_store(m_storage, m_layout.elementOffset(indices), value);
    }

   private:
    std::span<std::byte> m_storage;
    Layout m_layout;
    RuntimeStoreFunction<Accumulator> m_store;
};

template <typename Accumulator>
class RuntimeQuantizer {
   public:
    RuntimeQuantizer() = default;

    explicit RuntimeQuantizer(std::optional<ScalarType> type) {
        if (!type) return;
        m_load = runtimeLoadFunction<Accumulator>(*type);
        m_store = runtimeStoreFunction<Accumulator>(*type);
    }

    Accumulator operator()(Accumulator value) const {
        if (m_load == nullptr) return value;
        std::array<std::byte, 16> storage{};
        m_store(storage, 0, value);
        return m_load(storage, 0);
    }

   private:
    RuntimeLoadFunction<Accumulator> m_load = nullptr;
    RuntimeStoreFunction<Accumulator> m_store = nullptr;
};

inline float quantizeXFloat32(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits &= 0xffffe000U;
    return std::bit_cast<float>(bits);
}

template <typename Accumulator>
using RuntimeMathFunction = Accumulator (*)(Accumulator);

template <typename Accumulator>
Accumulator identityMath(Accumulator value) {
    return value;
}

inline float xfloat32Math(float value) {
    return quantizeXFloat32(value);
}

template <typename Accumulator>
RuntimeMathFunction<Accumulator> runtimeMathFunction(MathMode mode) {
    if (mode == MathMode::Default) return &identityMath<Accumulator>;
    if constexpr (std::is_same_v<Accumulator, float>) {
        if (mode == MathMode::XFloat32) return &xfloat32Math;
    }
    throw std::invalid_argument("XFloat32 math mode requires a Float32 accumulator.");
}

template <typename Accumulator>
Accumulator checkedRuntimeScalar(auto value, const char* name) {
    using Source = decltype(value);
    if constexpr (IsComplex<Accumulator>::value) {
        using Real = typename Accumulator::value_type;
        if constexpr (IsComplex<Source>::value)
            return Accumulator(static_cast<Real>(value.real()), static_cast<Real>(value.imag()));
        else
            return Accumulator(static_cast<Real>(value), Real(0));
    } else {
        if constexpr (IsComplex<Source>::value) {
            if (value.imag() != 0)
                throw std::invalid_argument(std::string("Real reference accumulator has complex ") +
                                            name + ".");
            return checkedRuntimeScalar<Accumulator>(value.real(), name);
        } else {
            if constexpr (std::is_integral_v<Accumulator>) {
                if constexpr (std::is_floating_point_v<Source>) {
                    const long double converted = static_cast<long double>(value);
                    if (!std::isfinite(converted) || std::trunc(converted) != converted ||
                        converted <
                            static_cast<long double>(std::numeric_limits<Accumulator>::lowest()) ||
                        converted >
                            static_cast<long double>(std::numeric_limits<Accumulator>::max()))
                        throw std::invalid_argument(
                            std::string("Integer reference accumulator has invalid ") + name + ".");
                } else if constexpr (std::is_same_v<Source, bool>) {
                    return static_cast<Accumulator>(value);
                } else if constexpr (std::is_integral_v<Source>) {
                    if (!std::in_range<Accumulator>(value))
                        throw std::invalid_argument(
                            std::string("Integer reference accumulator has invalid ") + name + ".");
                }
            }
            return static_cast<Accumulator>(value);
        }
    }
}

template <typename Accumulator>
Accumulator runtimeScalar(std::complex<double> value, const char* name) {
    return checkedRuntimeScalar<Accumulator>(value, name);
}

template <typename Accumulator>
Accumulator runtimeScalar(const Scalar& value, const char* name) {
    switch (scalarTypeInfo(value.type()).category) {
        case ScalarCategory::Boolean:
            return checkedRuntimeScalar<Accumulator>(value.as<bool>(), name);
        case ScalarCategory::SignedInteger:
            return checkedRuntimeScalar<Accumulator>(value.as<int64_t>(), name);
        case ScalarCategory::UnsignedInteger:
            return checkedRuntimeScalar<Accumulator>(value.as<uint64_t>(), name);
        case ScalarCategory::FloatingPoint:
        case ScalarCategory::Scale:
            return checkedRuntimeScalar<Accumulator>(value.as<double>(), name);
        case ScalarCategory::Complex:
            return checkedRuntimeScalar<Accumulator>(value.as<std::complex<double>>(), name);
    }
    throw std::invalid_argument("Invalid runtime scalar type.");
}

inline void requireRank(const Shape& shape, size_t rank, std::string_view operation,
                        const char* name) {
    if (shape.rank() != rank)
        throw std::invalid_argument(std::string(operation) + " " + name + " must have rank " +
                                    std::to_string(rank) + ".");
}

inline void validateRuntimeVector(const Tensor& view, size_t expected, std::string_view operation,
                                  const char* name) {
    requireRank(view.shape(), 1, operation, name);
    if (view.shape()[0] != expected)
        throw std::invalid_argument(std::string(operation) + " " + name + " length mismatch.");
}

inline size_t axisExtent(MatrixAxis axis, size_t rows, size_t columns) {
    return axis == MatrixAxis::Row ? rows : columns;
}
}  // namespace detail
}  // namespace roc::host_validation
