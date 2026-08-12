// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <roc/host_validation/tensor.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_validation {
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

enum class MatrixAxis {
    Row,
    Column,
};

enum class MathMode {
    Default,
    XFloat32,
};

enum class OutputConversion {
    Default,
    SaturatingInt8,
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
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
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
}  // namespace roc::host_validation
