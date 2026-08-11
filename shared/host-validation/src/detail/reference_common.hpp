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
                constexpr float coefficient0 = 0.7978845608028654f;
                constexpr float coefficient1 = 0.044715f;
                const float x = static_cast<float>(value);
                return static_cast<Accumulator>(
                    0.5f * x *
                    (1.0f + std::tanh(coefficient0 * x * (1.0f + coefficient1 * x * x))));
            }
            case Activation::GeluDerivative: {
                constexpr float coefficient0 = 0.0535161f;
                constexpr float coefficient1 = 0.398942f;
                constexpr float coefficient2 = 0.0356774f;
                constexpr float coefficient3 = 0.797885f;
                const float x = static_cast<float>(value);
                const float cube = x * x * x;
                const float first = coefficient0 * cube + coefficient1 * x;
                const float second = coefficient2 * cube + coefficient3 * x;
                const float derivative =
                    0.5f * std::tanh(second) +
                    first * (4.0f / std::pow(std::exp(-second) + std::exp(second), 2)) + 0.5f;
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
                const float x = static_cast<float>(value);
                return static_cast<Accumulator>(1.0f / (1.0f + std::exp(-x)));
            }
            case Activation::Tanh:
                return static_cast<Accumulator>(std::tanh(static_cast<float>(value * parameter0)) *
                                                static_cast<float>(parameter1));
            case Activation::Silu: {
                const float x = static_cast<float>(value);
                return static_cast<Accumulator>(x / (1.0f + std::exp(-x)));
            }
            case Activation::Swish: {
                const float x = static_cast<float>(value);
                const float beta = static_cast<float>(parameter0);
                return static_cast<Accumulator>(x / (1.0f + std::exp(-beta * x)));
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
    explicit RuntimeMatrixReader(TensorView view)
        : m_storage(view.storage()),
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
    explicit RuntimeMatrixWriter(MutableTensorView view)
        : m_storage(view.storage()),
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
class RuntimeMatrixOutputWriter {
   public:
    RuntimeMatrixOutputWriter(MutableTensorView output, OutputConversion conversion)
        : m_output(std::move(output)), m_defaultWriter(m_output), m_conversion(conversion) {
        if (m_conversion == OutputConversion::SaturatingInt8 && m_output.type() != ScalarType::Int8)
            throw std::invalid_argument(
                "Saturating output conversion requires an Int8 output tensor.");
    }

    void store(size_t row, size_t column, Accumulator value) const {
        if (m_conversion == OutputConversion::Default) {
            m_defaultWriter.store(row, column, value);
            return;
        }

        if constexpr (IsComplex<Accumulator>::value) {
            throw std::invalid_argument(
                "Saturating output conversion does not accept complex values.");
        } else if constexpr (std::is_integral_v<Accumulator>) {
            const Accumulator clamped =
                std::clamp(value, static_cast<Accumulator>(-128), static_cast<Accumulator>(127));
            m_output.storeFrom({row, column}, static_cast<int8_t>(clamped));
        } else {
            const long double rounded = std::nearbyint(static_cast<long double>(value));
            const long double clamped =
                std::clamp(rounded, static_cast<long double>(-128), static_cast<long double>(127));
            m_output.storeFrom({row, column}, static_cast<int8_t>(clamped));
        }
    }

   private:
    MutableTensorView m_output;
    RuntimeMatrixWriter<Accumulator> m_defaultWriter;
    OutputConversion m_conversion;
};

template <typename Accumulator>
class RuntimeVectorReader {
   public:
    explicit RuntimeVectorReader(TensorView view)
        : m_storage(view.storage()),
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
    explicit RuntimeTensorReader(TensorView view)
        : m_storage(view.storage()),
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
    explicit RuntimeTensorWriter(MutableTensorView view)
        : m_storage(view.storage()),
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
Accumulator runtimeScalar(std::complex<double> value, const char* name) {
    if constexpr (IsComplex<Accumulator>::value) {
        return Accumulator(static_cast<typename Accumulator::value_type>(value.real()),
                           static_cast<typename Accumulator::value_type>(value.imag()));
    } else {
        if (value.imag() != 0.0)
            throw std::invalid_argument(std::string("Real reference accumulator has complex ") +
                                        name + ".");
        if constexpr (std::is_integral_v<Accumulator>) {
            const double real = value.real();
            if (!std::isfinite(real) || std::trunc(real) != real ||
                real < static_cast<double>(std::numeric_limits<Accumulator>::lowest()) ||
                real > static_cast<double>(std::numeric_limits<Accumulator>::max()))
                throw std::invalid_argument(
                    std::string("Integer reference accumulator has invalid ") + name + ".");
            return static_cast<Accumulator>(real);
        } else {
            return static_cast<Accumulator>(value.real());
        }
    }
}

inline void requireRank(const Shape& shape, size_t rank, std::string_view operation,
                        const char* name) {
    if (shape.rank() != rank)
        throw std::invalid_argument(std::string(operation) + " " + name + " must have rank " +
                                    std::to_string(rank) + ".");
}

inline void validateRuntimeVector(const TensorView& view, size_t expected,
                                  std::string_view operation, const char* name) {
    requireRank(view.shape(), 1, operation, name);
    if (view.shape()[0] != expected)
        throw std::invalid_argument(std::string(operation) + " " + name + " length mismatch.");
}

inline size_t axisExtent(MatrixAxis axis, size_t rows, size_t columns) {
    return axis == MatrixAxis::Row ? rows : columns;
}
}  // namespace detail
}  // namespace roc::host_validation
