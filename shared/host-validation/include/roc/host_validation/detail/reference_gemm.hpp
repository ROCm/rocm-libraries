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
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <stdexcept>
#include <string>
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

enum class GemmBackend {
    Automatic,
    Canonical,
    Tiled,
    Blas,
};

struct VectorBinding {
    TensorView values;
    MatrixAxis axis = MatrixAxis::Row;
};

struct BlockScaleBinding {
    TensorView values;
    size_t blockSize;
};

struct GemmOperand {
    explicit GemmOperand(TensorView tensor) : values(std::move(tensor)) {}

    TensorView values;
    std::optional<ScalarType> computeType;
    std::optional<BlockScaleBinding> blockScale;
    bool conjugate = false;
};

struct GemmEpilogue {
    std::complex<double> alpha = {1.0, 0.0};
    std::complex<double> beta = {0.0, 0.0};
    std::optional<VectorBinding> bias;
    std::optional<VectorBinding> scaleAlpha;
    std::optional<TensorView> scaleA;
    std::optional<TensorView> scaleB;
    Activation activation = Activation::None;
    double activationParameter0 = 0.0;
    double activationParameter1 = 0.0;
};

struct GemmProblem {
    GemmProblem(GemmOperand aOperand, GemmOperand bOperand, TensorView cTensor,
                MutableTensorView dTensor, ScalarType accumulator)
        : a(std::move(aOperand)),
          b(std::move(bOperand)),
          c(std::move(cTensor)),
          d(std::move(dTensor)),
          accumulatorType(accumulator) {}

    GemmOperand a;
    GemmOperand b;
    TensorView c;
    MutableTensorView d;
    ScalarType accumulatorType;
    MathMode mathMode = MathMode::Default;
    GemmEpilogue epilogue;
};

struct GemmSupportInfo {
    bool supported = false;
    std::string reason;

    explicit operator bool() const {
        return supported;
    }
};

struct GemmRunInfo {
    GemmBackend backendUsed = GemmBackend::Canonical;
    std::optional<std::string> fallbackReason;
    size_t outputElementsComputed = 0;
};

class GemmBackendImplementation {
   public:
    virtual ~GemmBackendImplementation() = default;

    virtual GemmBackend backend() const = 0;
    virtual GemmSupportInfo querySupport(const GemmProblem&) const = 0;
    virtual GemmRunInfo run(const GemmProblem&) const = 0;
};

struct GemmRunOptions {
    GemmBackend backend = GemmBackend::Automatic;
    bool requireRequestedBackend = false;
    const GemmBackendImplementation* backendImplementation = nullptr;
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
            throw std::invalid_argument("Complex reference GEMM does not support activation.");
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

    throw std::invalid_argument("Unsupported reference GEMM activation.");
}

inline bool isComplexScalarType(ScalarType type) {
    return scalarTypeInfo(type).category == ScalarCategory::Complex;
}

inline bool isScaleScalarType(ScalarType type) {
    return scalarTypeInfo(type).category == ScalarCategory::Scale;
}

inline bool isRuntimeGemmAccumulator(ScalarType type) {
    switch (type) {
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            return true;
        default:
            return false;
    }
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
    if ((bits & 0x7f800000U) == 0x7f800000U) return value;
    const uint32_t retainedLeastSignificantBit = (bits >> 13) & 1U;
    bits += 0x0fffU + retainedLeastSignificantBit;
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
            throw std::invalid_argument(
                std::string("Reference GEMM real accumulator has complex ") + name + ".");
        return static_cast<Accumulator>(value.real());
    }
}

inline void requireRank(const Shape& shape, size_t rank, const char* name) {
    if (shape.rank() != rank)
        throw std::invalid_argument(std::string("Reference GEMM ") + name + " must have rank " +
                                    std::to_string(rank) + ".");
}

inline void validateRuntimeVector(const TensorView& view, size_t expected, const char* name) {
    requireRank(view.shape(), 1, name);
    if (view.shape()[0] != expected)
        throw std::invalid_argument(std::string("Reference GEMM ") + name + " length mismatch.");
}

inline size_t axisExtent(MatrixAxis axis, size_t rows, size_t columns) {
    return axis == MatrixAxis::Row ? rows : columns;
}

inline void validateRuntimeGemm(const GemmProblem& problem) {
    requireRank(problem.a.values.shape(), 2, "A");
    requireRank(problem.b.values.shape(), 2, "B");
    requireRank(problem.c.shape(), 2, "C");
    requireRank(problem.d.shape(), 2, "D");

    const size_t m = problem.a.values.shape()[0];
    const size_t k = problem.a.values.shape()[1];
    const size_t n = problem.b.values.shape()[1];
    if (problem.b.values.shape()[0] != k)
        throw std::invalid_argument("Reference GEMM K dimension mismatch.");
    if (problem.c.shape() != Shape{m, n})
        throw std::invalid_argument("Reference GEMM C shape mismatch.");
    if (problem.d.shape() != Shape{m, n})
        throw std::invalid_argument("Reference GEMM D shape mismatch.");
    if (!isRuntimeGemmAccumulator(problem.accumulatorType))
        throw std::invalid_argument(
            "Runtime reference GEMM currently supports F32, F64, C64, and "
            "C128 accumulators.");

    const bool complexAccumulator = isComplexScalarType(problem.accumulatorType);
    auto validateOperandType = [&](ScalarType type, const char* name) {
        if (type == ScalarType::Count || type == ScalarType::Boolean || isScaleScalarType(type))
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " has an unsupported scalar type.");
        if (!complexAccumulator && isComplexScalarType(type))
            throw std::invalid_argument(
                std::string("Reference GEMM real accumulator cannot consume complex ") + name +
                ".");
    };
    validateOperandType(problem.a.values.type(), "A");
    validateOperandType(problem.b.values.type(), "B");
    validateOperandType(problem.c.type(), "C");
    validateOperandType(problem.d.type(), "D");
    if (complexAccumulator != isComplexScalarType(problem.d.type()))
        throw std::invalid_argument("Reference GEMM complex accumulator/output mismatch.");

    auto validateComputeType = [&](const GemmOperand& operand, const char* name) {
        if (!operand.computeType) return;
        validateOperandType(*operand.computeType, name);
        if (isComplexScalarType(operand.values.type()) &&
            !isComplexScalarType(*operand.computeType))
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " compute-input type has incompatible complexity.");
    };
    validateComputeType(problem.a, "A");
    validateComputeType(problem.b, "B");

    if (problem.mathMode == MathMode::XFloat32 && problem.accumulatorType != ScalarType::Float32)
        throw std::invalid_argument("XFloat32 math mode requires a Float32 accumulator.");
    if (complexAccumulator && problem.epilogue.activation != Activation::None)
        throw std::invalid_argument("Complex reference GEMM does not support activation.");
    if (!complexAccumulator &&
        (problem.epilogue.alpha.imag() != 0.0 || problem.epilogue.beta.imag() != 0.0))
        throw std::invalid_argument("Reference GEMM real accumulator has complex alpha or beta.");

    auto validateEpilogueVector = [&](const TensorView& values, size_t expected, const char* name) {
        validateRuntimeVector(values, expected, name);
        if (!complexAccumulator && isComplexScalarType(values.type()))
            throw std::invalid_argument(
                std::string("Reference GEMM real accumulator cannot consume complex ") + name +
                ".");
    };
    if (problem.epilogue.bias) {
        const auto& binding = *problem.epilogue.bias;
        validateEpilogueVector(binding.values, axisExtent(binding.axis, m, n), "bias");
    }
    if (problem.epilogue.scaleAlpha) {
        const auto& binding = *problem.epilogue.scaleAlpha;
        validateEpilogueVector(binding.values, axisExtent(binding.axis, m, n), "scale-alpha");
    }
    if (problem.epilogue.scaleA) validateEpilogueVector(*problem.epilogue.scaleA, m, "scale-A");
    if (problem.epilogue.scaleB) validateEpilogueVector(*problem.epilogue.scaleB, n, "scale-B");

    const bool hasBlockScaleA = problem.a.blockScale.has_value();
    const bool hasBlockScaleB = problem.b.blockScale.has_value();
    if (hasBlockScaleA != hasBlockScaleB)
        throw std::invalid_argument(
            "Reference GEMM requires block scales for both operands or neither.");

    auto validateBlockScale = [&](const BlockScaleBinding& binding, size_t freeExtent,
                                  const char* name) {
        if (binding.blockSize == 0)
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block size must be nonzero.");
        requireRank(binding.values.shape(), 2, name);
        const size_t blockCount = k / binding.blockSize + (k % binding.blockSize != 0 ? 1 : 0);
        if (binding.values.shape()[0] != freeExtent || binding.values.shape()[1] < blockCount)
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block-scale shape mismatch.");
        if (isComplexScalarType(binding.values.type()))
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block scales must be real.");
    };
    if (hasBlockScaleA) {
        validateBlockScale(*problem.a.blockScale, m, "A");
        validateBlockScale(*problem.b.blockScale, n, "B");
        if (complexAccumulator)
            throw std::invalid_argument("Complex reference GEMM does not support block scaling.");
    }
}

template <typename Accumulator>
GemmRunInfo referenceRuntimeCanonical(const GemmProblem& problem) {
    const RuntimeMatrixReader<Accumulator> a(problem.a.values);
    const RuntimeMatrixReader<Accumulator> b(problem.b.values);
    const RuntimeMatrixReader<Accumulator> c(problem.c);
    const RuntimeMatrixWriter<Accumulator> d(problem.d);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.a.computeType);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.b.computeType);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);

    std::optional<RuntimeVectorReader<Accumulator>> bias;
    std::optional<RuntimeVectorReader<Accumulator>> scaleAlpha;
    std::optional<RuntimeVectorReader<Accumulator>> scaleA;
    std::optional<RuntimeVectorReader<Accumulator>> scaleB;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleA;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleB;
    if (problem.epilogue.bias) bias.emplace(problem.epilogue.bias->values);
    if (problem.epilogue.scaleAlpha) scaleAlpha.emplace(problem.epilogue.scaleAlpha->values);
    if (problem.epilogue.scaleA) scaleA.emplace(*problem.epilogue.scaleA);
    if (problem.epilogue.scaleB) scaleB.emplace(*problem.epilogue.scaleB);
    if (problem.a.blockScale) {
        blockScaleA.emplace(problem.a.blockScale->values);
        blockScaleB.emplace(problem.b.blockScale->values);
    }

    const size_t m = problem.a.values.shape()[0];
    const size_t k = problem.a.values.shape()[1];
    const size_t n = problem.b.values.shape()[1];
    const Accumulator alpha = runtimeScalar<Accumulator>(problem.epilogue.alpha, "alpha");
    const Accumulator beta = runtimeScalar<Accumulator>(problem.epilogue.beta, "beta");
    const Accumulator activationParameter0 =
        static_cast<Accumulator>(problem.epilogue.activationParameter0);
    const Accumulator activationParameter1 =
        static_cast<Accumulator>(problem.epilogue.activationParameter1);

    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            Accumulator sum = Accumulator(0);

            if (blockScaleA) {
                const size_t blockSizeA = problem.a.blockScale->blockSize;
                const size_t blockSizeB = problem.b.blockScale->blockSize;
                size_t blockBase = 0;
                while (blockBase < k) {
                    const size_t remainingA = blockSizeA - blockBase % blockSizeA;
                    const size_t remainingB = blockSizeB - blockBase % blockSizeB;
                    const size_t blockLength = std::min({k - blockBase, remainingA, remainingB});
                    const size_t blockEnd = blockBase + blockLength;
                    Accumulator blockSum = Accumulator(0);
                    for (size_t reduction = blockBase; reduction < blockEnd; ++reduction) {
                        Accumulator aValue =
                            conjugateIfNeeded(a(row, reduction), problem.a.conjugate);
                        Accumulator bValue =
                            conjugateIfNeeded(b(reduction, column), problem.b.conjugate);
                        aValue = operandMath(quantizeA(aValue));
                        bValue = operandMath(quantizeB(bValue));
                        blockSum += aValue * bValue;
                    }

                    const Accumulator scale = (*blockScaleA)(row, blockBase / blockSizeA) *
                                              (*blockScaleB)(column, blockBase / blockSizeB);
                    sum += blockSum * scale;
                    blockBase = blockEnd;
                }
            } else {
                for (size_t reduction = 0; reduction < k; ++reduction) {
                    Accumulator aValue = conjugateIfNeeded(a(row, reduction), problem.a.conjugate);
                    Accumulator bValue =
                        conjugateIfNeeded(b(reduction, column), problem.b.conjugate);
                    aValue = operandMath(quantizeA(aValue));
                    bValue = operandMath(quantizeB(bValue));
                    sum += aValue * bValue;
                }
            }

            Accumulator effectiveAlpha = alpha;
            if (scaleA) effectiveAlpha *= (*scaleA)[row];
            if (scaleB) effectiveAlpha *= (*scaleB)[column];
            if (scaleAlpha) {
                const MatrixAxis axis = problem.epilogue.scaleAlpha->axis;
                effectiveAlpha *= (*scaleAlpha)[axis == MatrixAxis::Row ? row : column];
            }

            Accumulator result = effectiveAlpha * sum + beta * c(row, column);
            if (bias) {
                const MatrixAxis axis = problem.epilogue.bias->axis;
                result += (*bias)[axis == MatrixAxis::Row ? row : column];
            }
            result = applyActivation(problem.epilogue.activation, result, activationParameter0,
                                     activationParameter1);
            d.store(row, column, result);
        }
    }

    return {
        .backendUsed = GemmBackend::Canonical,
        .fallbackReason = std::nullopt,
        .outputElementsComputed = problem.d.shape().elementCount(),
    };
}
}  // namespace detail

inline GemmSupportInfo queryGemmSupport(const GemmProblem& problem, GemmBackend backend,
                                        const GemmBackendImplementation* implementation = nullptr) {
    try {
        detail::validateRuntimeGemm(problem);
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }

    switch (backend) {
        case GemmBackend::Automatic:
        case GemmBackend::Canonical:
            return {.supported = true, .reason = {}};
        case GemmBackend::Tiled:
        case GemmBackend::Blas:
            if (implementation == nullptr)
                return {
                    .supported = false,
                    .reason =
                        "No implementation was supplied for the requested "
                        "runtime GEMM backend.",
                };
            if (implementation->backend() != backend)
                return {
                    .supported = false,
                    .reason =
                        "The supplied runtime GEMM implementation does not "
                        "match the requested backend.",
                };
            return implementation->querySupport(problem);
    }
    return {.supported = false, .reason = "Invalid reference GEMM backend."};
}

inline GemmRunInfo referenceGemm(const GemmProblem& problem, const GemmRunOptions& options = {}) {
    GemmBackend backend = options.backend;
    std::optional<std::string> fallbackReason;
    if (backend == GemmBackend::Automatic) {
        if (options.backendImplementation != nullptr) {
            const GemmBackend implementationBackend = options.backendImplementation->backend();
            const GemmSupportInfo implementationSupport =
                queryGemmSupport(problem, implementationBackend, options.backendImplementation);
            if (implementationSupport) return options.backendImplementation->run(problem);
            fallbackReason = implementationSupport.reason;
        }
        backend = GemmBackend::Canonical;
    }

    const GemmSupportInfo requestedSupport =
        queryGemmSupport(problem, backend, options.backendImplementation);
    if (!requestedSupport) {
        if (options.requireRequestedBackend) throw std::invalid_argument(requestedSupport.reason);
        if (backend == GemmBackend::Canonical) throw std::invalid_argument(requestedSupport.reason);
        fallbackReason = requestedSupport.reason;
        backend = GemmBackend::Canonical;
    } else if (backend != GemmBackend::Canonical) {
        return options.backendImplementation->run(problem);
    }

    const GemmSupportInfo canonicalSupport = queryGemmSupport(problem, GemmBackend::Canonical);
    if (!canonicalSupport) throw std::invalid_argument(canonicalSupport.reason);

    GemmRunInfo result;
    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            result = detail::referenceRuntimeCanonical<float>(problem);
            break;
        case ScalarType::Float64:
            result = detail::referenceRuntimeCanonical<double>(problem);
            break;
        case ScalarType::ComplexFloat32:
            result = detail::referenceRuntimeCanonical<std::complex<float>>(problem);
            break;
        case ScalarType::ComplexFloat64:
            result = detail::referenceRuntimeCanonical<std::complex<double>>(problem);
            break;
        default:
            throw std::invalid_argument("Unsupported runtime reference GEMM accumulator type.");
    }
    result.fallbackReason = std::move(fallbackReason);
    return result;
}

inline std::vector<GemmRunInfo> referenceGroupedGemm(std::span<const GemmProblem> problems,
                                                     const GemmRunOptions& options = {}) {
    std::vector<GemmRunInfo> results;
    results.reserve(problems.size());
    for (const GemmProblem& problem : problems) results.push_back(referenceGemm(problem, options));
    return results;
}
}  // namespace roc::host_validation
