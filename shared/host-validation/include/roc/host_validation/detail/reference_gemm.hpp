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
#include <roc/host_validation/detail/reference_common.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace roc::host_validation {
enum class GemmBackend {
    Automatic,
    Canonical,
    Tiled,
    Blas,
};

enum class GemmOutputConversion {
    Default,
    SaturatingInt8,
};

enum class AccumulationRounding {
    TypeDefault,
    FullPrecision,
    AfterProductAndSum,
};

struct BlockScaleBinding {
    TensorView values;
    size_t blockSize;
};

struct GemmOperand {
    explicit GemmOperand(TensorView tensor) : values(std::move(tensor)) {}

    TensorView values;
    std::optional<ScalarType> computeType;
    std::vector<VectorBinding> preQuantizationScales;
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
    std::complex<double> outputScale = {1.0, 0.0};
    GemmOutputConversion outputConversion = GemmOutputConversion::Default;
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
    AccumulationRounding accumulationRounding = AccumulationRounding::TypeDefault;
    MathMode mathMode = MathMode::Default;
    GemmEpilogue epilogue;
    OutputSelection outputSelection = OutputSelection::all();
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
inline bool isRuntimeGemmAccumulator(ScalarType type) {
    switch (type) {
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Int32:
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            return true;
        default:
            return false;
    }
}

inline void validateRuntimeGemm(const GemmProblem& problem) {
    requireRank(problem.a.values.shape(), 2, "Reference GEMM", "A");
    requireRank(problem.b.values.shape(), 2, "Reference GEMM", "B");
    requireRank(problem.c.shape(), 2, "Reference GEMM", "C");
    requireRank(problem.d.shape(), 2, "Reference GEMM", "D");

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
            "Runtime reference GEMM currently supports F16, BF16, F32, F64, I32, C64, and "
            "C128 accumulators.");
    if (problem.accumulationRounding == AccumulationRounding::AfterProductAndSum &&
        problem.accumulatorType != ScalarType::Float16 &&
        problem.accumulatorType != ScalarType::BFloat16)
        throw std::invalid_argument(
            "Product-and-sum accumulator rounding currently requires an F16 or BF16 "
            "accumulator type.");

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
    auto validatePreQuantizationScales = [&](const GemmOperand& operand, const char* name) {
        for (const VectorBinding& binding : operand.preQuantizationScales) {
            requireRank(binding.values.shape(), 1, "Reference GEMM", name);
            const size_t expected =
                axisExtent(binding.axis, operand.values.shape()[0], operand.values.shape()[1]);
            if (binding.values.shape()[0] != 1 && binding.values.shape()[0] != expected)
                throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                            " length mismatch.");
            if (!complexAccumulator && isComplexScalarType(binding.values.type()))
                throw std::invalid_argument(
                    std::string("Reference GEMM real accumulator cannot consume complex ") + name +
                    ".");
        }
    };
    validatePreQuantizationScales(problem.a, "A pre-quantization scale");
    validatePreQuantizationScales(problem.b, "B pre-quantization scale");

    if (problem.mathMode == MathMode::XFloat32 && problem.accumulatorType != ScalarType::Float32)
        throw std::invalid_argument("XFloat32 math mode requires a Float32 accumulator.");
    if (complexAccumulator && problem.epilogue.activation != Activation::None)
        throw std::invalid_argument("Complex reference GEMM does not support activation.");
    if (!complexAccumulator &&
        (problem.epilogue.alpha.imag() != 0.0 || problem.epilogue.beta.imag() != 0.0 ||
         problem.epilogue.outputScale.imag() != 0.0))
        throw std::invalid_argument(
            "Reference GEMM real accumulator has a complex scalar.");
    if (problem.epilogue.outputConversion == GemmOutputConversion::SaturatingInt8 &&
        problem.d.type() != ScalarType::Int8)
        throw std::invalid_argument(
            "Reference GEMM saturating output conversion currently requires Int8 output.");

    auto validateEpilogueVector = [&](const TensorView& values, size_t expected, const char* name) {
        validateRuntimeVector(values, expected, "Reference GEMM", name);
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
        requireRank(binding.values.shape(), 2, "Reference GEMM", name);
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
    (void)problem.outputSelection.selectedCount(problem.d.shape().elementCount());
}

template <typename Accumulator>
class RuntimeGemmOutputWriter {
   public:
    RuntimeGemmOutputWriter(MutableTensorView output, GemmOutputConversion conversion)
        : m_output(std::move(output)), m_defaultWriter(m_output), m_conversion(conversion) {}

    void store(size_t row, size_t column, Accumulator value) const {
        if (m_conversion == GemmOutputConversion::Default) {
            m_defaultWriter.store(row, column, value);
            return;
        }

        if constexpr (IsComplex<Accumulator>::value) {
            throw std::invalid_argument(
                "Saturating GEMM output conversion does not accept complex values.");
        } else {
            const long double rounded = std::nearbyint(static_cast<long double>(value));
            const long double clamped =
                std::clamp(rounded, static_cast<long double>(-128),
                           static_cast<long double>(127));
            m_output.storeFrom(
                {row, column}, static_cast<int8_t>(clamped));
        }
    }

   private:
    MutableTensorView m_output;
    RuntimeMatrixWriter<Accumulator> m_defaultWriter;
    GemmOutputConversion m_conversion;
};

template <typename Accumulator>
GemmRunInfo referenceRuntimeCanonical(const GemmProblem& problem) {
    const RuntimeMatrixReader<Accumulator> a(problem.a.values);
    const RuntimeMatrixReader<Accumulator> b(problem.b.values);
    const RuntimeMatrixReader<Accumulator> c(problem.c);
    const RuntimeGemmOutputWriter<Accumulator> d(
        problem.d, problem.epilogue.outputConversion);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.a.computeType);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.b.computeType);
    const bool typeRoundsAfterEachStep =
        problem.accumulatorType == ScalarType::Float16 ||
        problem.accumulatorType == ScalarType::BFloat16;
    const bool roundAfterEachStep =
        problem.accumulationRounding == AccumulationRounding::AfterProductAndSum ||
        (problem.accumulationRounding == AccumulationRounding::TypeDefault &&
         typeRoundsAfterEachStep);
    const RuntimeQuantizer<Accumulator> quantizeAccumulator(
        roundAfterEachStep ? std::optional<ScalarType>(problem.accumulatorType) : std::nullopt);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);
    auto multiplyAccumulator = [&](Accumulator left, Accumulator right) {
        return quantizeAccumulator(left * right);
    };
    auto addAccumulator = [&](Accumulator left, Accumulator right) {
        return quantizeAccumulator(left + right);
    };

    std::optional<RuntimeVectorReader<Accumulator>> bias;
    std::vector<RuntimeVectorReader<Accumulator>> preScalesA;
    std::vector<RuntimeVectorReader<Accumulator>> preScalesB;
    std::optional<RuntimeVectorReader<Accumulator>> scaleAlpha;
    std::optional<RuntimeVectorReader<Accumulator>> scaleA;
    std::optional<RuntimeVectorReader<Accumulator>> scaleB;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleA;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleB;
    if (problem.epilogue.bias) bias.emplace(problem.epilogue.bias->values);
    preScalesA.reserve(problem.a.preQuantizationScales.size());
    for (const VectorBinding& binding : problem.a.preQuantizationScales)
        preScalesA.emplace_back(binding.values);
    preScalesB.reserve(problem.b.preQuantizationScales.size());
    for (const VectorBinding& binding : problem.b.preQuantizationScales)
        preScalesB.emplace_back(binding.values);
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
    const Accumulator alpha =
        quantizeAccumulator(runtimeScalar<Accumulator>(problem.epilogue.alpha, "alpha"));
    const Accumulator beta =
        quantizeAccumulator(runtimeScalar<Accumulator>(problem.epilogue.beta, "beta"));
    const Accumulator outputScale =
        runtimeScalar<Accumulator>(problem.epilogue.outputScale, "output scale");
    const Accumulator activationParameter0 =
        quantizeAccumulator(static_cast<Accumulator>(problem.epilogue.activationParameter0));
    const Accumulator activationParameter1 =
        quantizeAccumulator(static_cast<Accumulator>(problem.epilogue.activationParameter1));

    auto computeOutput = [&](size_t row, size_t column) {
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
                    Accumulator aValue = conjugateIfNeeded(a(row, reduction), problem.a.conjugate);
                    Accumulator bValue =
                        conjugateIfNeeded(b(reduction, column), problem.b.conjugate);
                    for (size_t scaleIndex = 0; scaleIndex < preScalesA.size(); ++scaleIndex) {
                        const auto& binding = problem.a.preQuantizationScales[scaleIndex];
                        const size_t index =
                            binding.values.shape()[0] == 1
                                ? 0
                                : (binding.axis == MatrixAxis::Row ? row : reduction);
                        aValue *= preScalesA[scaleIndex][index];
                    }
                    for (size_t scaleIndex = 0; scaleIndex < preScalesB.size(); ++scaleIndex) {
                        const auto& binding = problem.b.preQuantizationScales[scaleIndex];
                        const size_t index =
                            binding.values.shape()[0] == 1
                                ? 0
                                : (binding.axis == MatrixAxis::Row ? reduction : column);
                        bValue *= preScalesB[scaleIndex][index];
                    }
                    aValue = operandMath(quantizeA(aValue));
                    bValue = operandMath(quantizeB(bValue));
                    blockSum = addAccumulator(blockSum, multiplyAccumulator(aValue, bValue));
                }

                const Accumulator scale =
                    multiplyAccumulator((*blockScaleA)(row, blockBase / blockSizeA),
                                        (*blockScaleB)(column, blockBase / blockSizeB));
                sum = addAccumulator(sum, multiplyAccumulator(blockSum, scale));
                blockBase = blockEnd;
            }
        } else {
            for (size_t reduction = 0; reduction < k; ++reduction) {
                Accumulator aValue = conjugateIfNeeded(a(row, reduction), problem.a.conjugate);
                Accumulator bValue = conjugateIfNeeded(b(reduction, column), problem.b.conjugate);
                for (size_t scaleIndex = 0; scaleIndex < preScalesA.size(); ++scaleIndex) {
                    const auto& binding = problem.a.preQuantizationScales[scaleIndex];
                    const size_t index = binding.values.shape()[0] == 1
                                             ? 0
                                             : (binding.axis == MatrixAxis::Row ? row : reduction);
                    aValue *= preScalesA[scaleIndex][index];
                }
                for (size_t scaleIndex = 0; scaleIndex < preScalesB.size(); ++scaleIndex) {
                    const auto& binding = problem.b.preQuantizationScales[scaleIndex];
                    const size_t index =
                        binding.values.shape()[0] == 1
                            ? 0
                            : (binding.axis == MatrixAxis::Row ? reduction : column);
                    bValue *= preScalesB[scaleIndex][index];
                }
                aValue = operandMath(quantizeA(aValue));
                bValue = operandMath(quantizeB(bValue));
                sum = addAccumulator(sum, multiplyAccumulator(aValue, bValue));
            }
        }

        Accumulator effectiveAlpha = alpha;
        if (scaleA) effectiveAlpha = multiplyAccumulator(effectiveAlpha, (*scaleA)[row]);
        if (scaleB) effectiveAlpha = multiplyAccumulator(effectiveAlpha, (*scaleB)[column]);
        if (scaleAlpha) {
            const MatrixAxis axis = problem.epilogue.scaleAlpha->axis;
            effectiveAlpha = multiplyAccumulator(
                effectiveAlpha, (*scaleAlpha)[axis == MatrixAxis::Row ? row : column]);
        }

        Accumulator result = addAccumulator(multiplyAccumulator(effectiveAlpha, sum),
                                            multiplyAccumulator(beta, c(row, column)));
        if (bias) {
            const MatrixAxis axis = problem.epilogue.bias->axis;
            result = addAccumulator(result, (*bias)[axis == MatrixAxis::Row ? row : column]);
        }
        result = quantizeAccumulator(applyActivation(problem.epilogue.activation, result,
                                                     activationParameter0, activationParameter1));
        result *= outputScale;
        d.store(row, column, result);
    };

    const size_t logicalElements = problem.d.shape().elementCount();
    size_t computedElements = 0;
    if (problem.outputSelection.selectsAll()) {
        for (size_t row = 0; row < m; ++row) {
            for (size_t column = 0; column < n; ++column) {
                computeOutput(row, column);
                ++computedElements;
            }
        }
    } else {
        const auto selected = problem.outputSelection.indices(logicalElements);
        for (size_t logicalIndex : selected) {
            const size_t row = logicalIndex / n;
            const size_t column = logicalIndex % n;
            computeOutput(row, column);
        }
        computedElements = selected.size();
    }

    return {
        .backendUsed = GemmBackend::Canonical,
        .fallbackReason = std::nullopt,
        .outputElementsComputed = computedElements,
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
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            result = detail::referenceRuntimeCanonical<float>(problem);
            break;
        case ScalarType::Float64:
            result = detail::referenceRuntimeCanonical<double>(problem);
            break;
        case ScalarType::Int32:
            result = detail::referenceRuntimeCanonical<int32_t>(problem);
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
