// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <roc/host_numerics/tensor.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_numerics {
// Operation-specific option types describe numerical policy independently of
// tensor ownership. Owning entry points return newly allocated tensors;
// corresponding ...Into entry points write caller-owned destinations.

// Shared pointwise activation selector for GEMM and standalone epilogue
// descriptors. Consumers provide any activation parameters separately.
enum class Activation {
    None,
    Absolute,
    ClippedRelu,
    Relu,
    Gelu,
    GeluDerivative,
    GeluScaling,
    LeakyRelu,
    ReluDerivative,
    Sigmoid,
    Tanh,
    Silu,
    Swish,
    Clamp,
};

// Selects which matrix coordinate indexes a broadcast vector.
enum class MatrixAxis {
    Row,     // values[row]
    Column,  // values[column]
};

// Selects operand arithmetic applied after input quantization and before
// multiplication.
enum class MathMode {
    Default,
    XFloat32,  // Truncate Float32 operands to the xfloat32 representation.
};

// Selects the final conversion performed before writing an output tensor.
enum class OutputConversion {
    Default,
    SaturatingInt8,  // Round and clamp to [-128, 127].
};

// Associates a rank-one tensor with its row- or column-broadcasting rule.
struct VectorBinding {
    VectorBinding(Tensor tensor, MatrixAxis selectedAxis = MatrixAxis::Row)
        : values(std::move(tensor)), axis(selectedAxis) {}

    Tensor values;
    MatrixAxis axis = MatrixAxis::Row;
};

// Storage form used by OutputSelection.
enum class OutputSelectionKind {
    All,
    Strided,
    Explicit,
};

namespace detail {
// Tests divisors only through floor(sqrt(value)). The quotient comparison is
// equivalent to divisor * divisor <= value and cannot overflow size_t.
inline bool isPrimeStride(size_t value) {
    if (value < 2) return false;
    if (value % 2 == 0) return value == 2;
    if (value % 3 == 0) return value == 3;

    for (size_t divisor = 5; divisor <= value / divisor; divisor += 6) {
        if (value % divisor == 0 || value % (divisor + 2) == 0) return false;
    }
    return true;
}

inline size_t nextPrimeStride(size_t value) {
    if (value <= 2) return 2;
    size_t candidate = value % 2 == 0 ? value + 1 : value;
    while (!isPrimeStride(candidate)) {
        if (candidate > std::numeric_limits<size_t>::max() - 2)
            throw std::overflow_error("Prime-stride output selection overflow.");
        candidate += 2;
    }
    return candidate;
}
}  // namespace detail

// Describes which logical tensor elements an operation or comparison should
// visit. Linear indices use indexOrder. Explicit indices are normalized when
// constructed and validated against the tensor's logical element count when
// consumed.
class OutputSelection {
   public:
    static OutputSelection all(IndexOrder indexOrder = IndexOrder::LastDimensionFastest) {
        OutputSelection result;
        result.m_indexOrder = indexOrder;
        return result;
    }

    static OutputSelection strided(size_t first, size_t stride,
                                   size_t maxElements = std::numeric_limits<size_t>::max(),
                                   IndexOrder indexOrder = IndexOrder::LastDimensionFastest) {
        if (stride == 0) throw std::invalid_argument("Output selection stride must be nonzero.");
        OutputSelection result;
        result.m_kind = OutputSelectionKind::Strided;
        result.m_first = first;
        result.m_stride = stride;
        result.m_maxElements = maxElements;
        result.m_indexOrder = indexOrder;
        return result;
    }

    static OutputSelection explicitIndices(
        std::vector<size_t> indices, IndexOrder indexOrder = IndexOrder::LastDimensionFastest) {
        OutputSelection result;
        result.m_kind = OutputSelectionKind::Explicit;
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        result.m_indices = std::move(indices);
        result.m_indexOrder = indexOrder;
        return result;
    }

    // Chooses the smallest prime stride at least allocatedElements /
    // requestedElements. The resulting sample size is approximate.
    static OutputSelection primeStride(size_t logicalElements, size_t allocatedElements,
                                       size_t requestedElements,
                                       IndexOrder indexOrder = IndexOrder::LastDimensionFastest) {
        if (requestedElements == 0 || requestedElements >= logicalElements) return all(indexOrder);
        const size_t candidate = std::max<size_t>(1, allocatedElements / requestedElements);
        return strided(0, detail::nextPrimeStride(candidate), std::numeric_limits<size_t>::max(),
                       indexOrder);
    }

    OutputSelectionKind kind() const {
        return m_kind;
    }

    bool selectsAll() const {
        return m_kind == OutputSelectionKind::All;
    }

    size_t first() const {
        return m_first;
    }

    size_t stride() const {
        return m_stride;
    }

    size_t maxElements() const {
        return m_maxElements;
    }

    IndexOrder indexOrder() const {
        return m_indexOrder;
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
                if (m_first >= logicalElements || m_maxElements == 0) return result;
                const size_t available = 1 + (logicalElements - 1 - m_first) / m_stride;
                const size_t count = std::min(available, m_maxElements);
                result.reserve(count);
                for (size_t index = m_first; index < logicalElements && result.size() < count;) {
                    result.push_back(index);
                    if (index > std::numeric_limits<size_t>::max() - m_stride) break;
                    index += m_stride;
                }
                return result;
            }
            case OutputSelectionKind::Explicit:
                validateExplicitIndices(logicalElements);
                return m_indices;
        }
        throw std::invalid_argument("Invalid output selection kind.");
    }

    size_t selectedCount(size_t logicalElements) const {
        switch (m_kind) {
            case OutputSelectionKind::All:
                return logicalElements;
            case OutputSelectionKind::Strided:
                if (m_first >= logicalElements) return 0;
                return std::min(1 + (logicalElements - 1 - m_first) / m_stride, m_maxElements);
            case OutputSelectionKind::Explicit:
                validateExplicitIndices(logicalElements);
                return m_indices.size();
        }
        throw std::invalid_argument("Invalid output selection kind.");
    }

   private:
    void validateExplicitIndices(size_t logicalElements) const {
        // explicitIndices keeps this vector sorted, so its last element is the
        // only one needed to validate the complete selection.
        if (!m_indices.empty() && m_indices.back() >= logicalElements)
            throw std::out_of_range("Explicit output selection index exceeds output shape.");
    }

    OutputSelectionKind m_kind = OutputSelectionKind::All;
    size_t m_first = 0;
    size_t m_stride = 1;
    size_t m_maxElements = std::numeric_limits<size_t>::max();
    IndexOrder m_indexOrder = IndexOrder::LastDimensionFastest;
    std::vector<size_t> m_indices;
};
}  // namespace roc::host_numerics
