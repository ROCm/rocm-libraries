// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace roc::hostnumerics {
// Operation-specific option types describe numerical policy independently of
// tensor ownership. Owning entry points return newly allocated tensors;
// corresponding ...Into entry points write caller-owned destinations.

// Stable activation identifiers used by product adapters and compatibility
// bindings. Tensor-centric C++ callers use the typed ActivationFunction below.
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
    Count,
};

struct IdentityActivation {};
struct AbsoluteActivation {};
struct ReluActivation {};
struct GeluActivation {};
struct GeluDerivativeActivation {};
struct ReluDerivativeActivation {};
struct SigmoidActivation {};
struct SiluActivation {};

struct ClippedReluActivation {
    double lower = 0.0;
    double upper = 0.0;
};

struct GeluScalingActivation {
    double scale = 1.0;
};

struct LeakyReluActivation {
    double negativeSlope = 0.0;
};

struct TanhActivation {
    double inputScale = 1.0;
    double outputScale = 1.0;
};

struct SwishActivation {
    double beta = 1.0;
};

struct ClampActivation {
    double minimum = 0.0;
    double maximum = 0.0;
};

using ActivationFunction =
    std::variant<IdentityActivation, AbsoluteActivation, ClippedReluActivation, ReluActivation,
                 GeluActivation, GeluDerivativeActivation, GeluScalingActivation,
                 LeakyReluActivation, ReluDerivativeActivation, SigmoidActivation, TanhActivation,
                 SiluActivation, SwishActivation, ClampActivation>;

// Selects operand arithmetic applied after input quantization and before
// multiplication.
enum class MathMode {
    Default,
    XFloat32,  // Truncate Float32 operands to the xfloat32 representation.
    Count,
};

// Selects the final conversion performed before writing an output tensor.
enum class OutputConversion {
    Default,
    SaturatingInt8,  // Round and clamp to [-128, 127].
    Count,
};

// Storage form used by OutputSelection.
enum class OutputSelectionKind {
    All,
    Strided,
    Explicit,
    Count,
};

class OutputSelection;

namespace detail {
enum class OutputSelectionError { InvalidKind, IndexOutOfRange };
inline constexpr std::string_view message(OutputSelectionError error) noexcept {
    return error == OutputSelectionError::InvalidKind
               ? "Invalid output selection kind."
               : "Explicit output selection index exceeds output shape.";
}
inline std::optional<OutputSelectionError> outputSelectionError(const OutputSelection& selection,
                                                                size_t logicalElements) noexcept;

// Tests divisors only through floor(sqrt(value)). The quotient comparison is
// equivalent to divisor * divisor <= value and cannot overflow size_t.
inline bool isPrime(size_t value) {
    if (value < 2) return false;
    if (value % 2 == 0) return value == 2;
    if (value % 3 == 0) return value == 3;

    for (size_t divisor = 5; divisor <= value / divisor; divisor += 6) {
        if (value % divisor == 0 || value % (divisor + 2) == 0) return false;
    }
    return true;
}

inline std::optional<size_t> getNextPrimeStride(size_t value) {
    if (value <= 2) return 2;
    size_t candidate = value % 2 == 0 ? value + 1 : value;
    while (!isPrime(candidate)) {
        if (candidate > std::numeric_limits<size_t>::max() - 2) return std::nullopt;
        candidate += 2;
    }
    return candidate;
}

inline size_t nextPrimeStride(size_t value) {
    const auto stride = getNextPrimeStride(value);
    if (!stride) throw std::runtime_error("Prime-stride output selection overflow.");
    return *stride;
}
}  // namespace detail

// Describes which logical tensor elements an operation or comparison should
// visit. Linear indices use indexOrder. Explicit indices are normalized when
// constructed and validated against the tensor's logical element count when
// consumed. Orders list axes from slowest to fastest; empty selects NumPy/C
// identity order. Factories copy the supplied span, so selections own their axes.
class OutputSelection {
   public:
    static OutputSelection all(std::span<const size_t> indexOrder = {}) {
        OutputSelection result;
        result.m_indexOrder.assign(indexOrder.begin(), indexOrder.end());
        return result;
    }

    static OutputSelection strided(size_t first, size_t stride,
                                   size_t maxElements = std::numeric_limits<size_t>::max(),
                                   std::span<const size_t> indexOrder = {}) {
        if (stride == 0) throw std::runtime_error("Output selection stride must be nonzero.");
        OutputSelection result;
        result.m_kind = OutputSelectionKind::Strided;
        result.m_first = first;
        result.m_stride = stride;
        result.m_maxElements = maxElements;
        result.m_indexOrder.assign(indexOrder.begin(), indexOrder.end());
        return result;
    }

    static OutputSelection explicitIndices(std::vector<size_t> indices,
                                           std::span<const size_t> indexOrder = {}) {
        OutputSelection result;
        result.m_kind = OutputSelectionKind::Explicit;
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        result.m_indices = std::move(indices);
        result.m_indexOrder.assign(indexOrder.begin(), indexOrder.end());
        return result;
    }

    // Chooses the smallest prime stride at least allocatedElements /
    // requestedElements. The resulting sample size is approximate.
    static OutputSelection primeStride(size_t logicalElements, size_t allocatedElements,
                                       size_t requestedElements,
                                       std::span<const size_t> indexOrder = {}) {
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

    // Borrows this selection's order; valid while it is alive and unmodified.
    std::span<const size_t> indexOrder() const {
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
            case OutputSelectionKind::Count:
                break;
        }
        throw std::runtime_error("Invalid output selection kind.");
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
            case OutputSelectionKind::Count:
                break;
        }
        throw std::runtime_error("Invalid output selection kind.");
    }

   private:
    friend std::optional<detail::OutputSelectionError> detail::outputSelectionError(
        const OutputSelection&, size_t) noexcept;

    void validateExplicitIndices(size_t logicalElements) const {
        if (auto error = detail::outputSelectionError(*this, logicalElements))
            throw std::runtime_error(std::string(detail::message(*error)));
    }

    OutputSelectionKind m_kind = OutputSelectionKind::All;
    size_t m_first = 0;
    size_t m_stride = 1;
    size_t m_maxElements = std::numeric_limits<size_t>::max();
    std::vector<size_t> m_indexOrder;
    std::vector<size_t> m_indices;
};
namespace detail {
// Support queries and adapters consume the same check as the throwing accessors.
inline std::optional<OutputSelectionError> outputSelectionError(const OutputSelection& selection,
                                                                size_t logicalElements) noexcept {
    if (selection.m_kind == OutputSelectionKind::All ||
        selection.m_kind == OutputSelectionKind::Strided)
        return std::nullopt;
    if (selection.m_kind != OutputSelectionKind::Explicit) return OutputSelectionError::InvalidKind;
    if (!selection.m_indices.empty() && selection.m_indices.back() >= logicalElements)
        return OutputSelectionError::IndexOutOfRange;
    return std::nullopt;
}
}  // namespace detail
}  // namespace roc::hostnumerics
