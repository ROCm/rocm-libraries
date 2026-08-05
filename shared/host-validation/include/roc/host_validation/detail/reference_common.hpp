// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace roc::host_validation {
enum class Activation {
    None,
    Relu,
    Gelu,
    Silu,
    Clamp,
};

enum class MatrixAxis {
    Row,
    Column,
};

struct VectorBinding {
    TensorView values;
    MatrixAxis axis = MatrixAxis::Row;
};

namespace detail {
template <typename T>
struct IsComplex : std::false_type {};

template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

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
            case Activation::Silu: {
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
Accumulator runtimeScalar(std::complex<double> value, const char* name) {
    if constexpr (IsComplex<Accumulator>::value) {
        return Accumulator(static_cast<typename Accumulator::value_type>(value.real()),
                           static_cast<typename Accumulator::value_type>(value.imag()));
    } else {
        if (value.imag() != 0.0)
            throw std::invalid_argument(std::string("Real reference accumulator has complex ") +
                                        name + ".");
        return static_cast<Accumulator>(value.real());
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
