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
#include <limits>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

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

enum class MathMode {
    Default,
    XFloat32,
};

struct VectorBinding {
    TensorView values;
    MatrixAxis axis = MatrixAxis::Row;
};

enum class OutputSelectionKind {
    All,
    Strided,
    Explicit,
};

class OutputSelection {
   public:
    static OutputSelection all() {
        return {};
    }

    static OutputSelection strided(size_t first, size_t stride) {
        if (stride == 0) throw std::invalid_argument("Output selection stride must be nonzero.");
        OutputSelection result;
        result.m_kind = OutputSelectionKind::Strided;
        result.m_first = first;
        result.m_stride = stride;
        return result;
    }

    static OutputSelection explicitIndices(std::vector<size_t> indices) {
        OutputSelection result;
        result.m_kind = OutputSelectionKind::Explicit;
        result.m_indices = std::move(indices);
        return result;
    }

    static OutputSelection primeStride(size_t logicalElements, size_t allocatedElements,
                                       size_t requestedElements) {
        if (requestedElements == 0 || requestedElements >= logicalElements) return all();
        const size_t candidate = std::max<size_t>(1, allocatedElements / requestedElements);
        return strided(0, nextPrime(candidate));
    }

    OutputSelectionKind kind() const {
        return m_kind;
    }

    bool selectsAll() const {
        return m_kind == OutputSelectionKind::All;
    }

    std::vector<size_t> indices(size_t logicalElements) const {
        switch (m_kind) {
            case OutputSelectionKind::All: {
                std::vector<size_t> result(logicalElements);
                for (size_t index = 0; index < logicalElements; ++index) result[index] = index;
                return result;
            }
            case OutputSelectionKind::Strided: {
                std::vector<size_t> result;
                if (m_first >= logicalElements) return result;
                const size_t count = 1 + (logicalElements - 1 - m_first) / m_stride;
                result.reserve(count);
                for (size_t index = m_first; index < logicalElements;) {
                    result.push_back(index);
                    if (index > std::numeric_limits<size_t>::max() - m_stride) break;
                    index += m_stride;
                }
                return result;
            }
            case OutputSelectionKind::Explicit:
                for (size_t index : m_indices) {
                    if (index >= logicalElements)
                        throw std::out_of_range(
                            "Explicit output selection index exceeds output shape.");
                }
                return m_indices;
        }
        throw std::invalid_argument("Invalid output selection kind.");
    }

    size_t selectedCount(size_t logicalElements) const {
        if (m_kind == OutputSelectionKind::All) return logicalElements;
        return indices(logicalElements).size();
    }

   private:
    static bool isPrime(size_t value) {
        if (value < 2) return false;
        if (value % 2 == 0) return value == 2;
        for (size_t divisor = 3; divisor <= value / divisor; divisor += 2) {
            if (value % divisor == 0) return false;
        }
        return true;
    }

    static size_t nextPrime(size_t value) {
        if (value <= 2) return 2;
        size_t candidate = value % 2 == 0 ? value + 1 : value;
        while (!isPrime(candidate)) {
            if (candidate > std::numeric_limits<size_t>::max() - 2)
                throw std::overflow_error("Prime-stride output selection overflow.");
            candidate += 2;
        }
        return candidate;
    }

    OutputSelectionKind m_kind = OutputSelectionKind::All;
    size_t m_first = 0;
    size_t m_stride = 1;
    std::vector<size_t> m_indices;
};

namespace detail {
template <typename T>
struct IsComplex : std::false_type {};

template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

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
