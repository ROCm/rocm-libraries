// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_numerics/gemm.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gemm_invocation.hpp"
#include "reference_common.hpp"
#include "threading.hpp"

namespace roc::host_numerics {
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

template <typename Accumulator>
void validateRuntimeGemmScalars(const GemmSpecification& problem) {
    (void)runtimeScalar<Accumulator>(problem.alpha, "alpha");
    (void)runtimeScalar<Accumulator>(problem.beta, "beta");
    (void)runtimeScalar<Accumulator>(problem.scaleC, "C scale");
    (void)runtimeScalar<Accumulator>(problem.outputScale, "output scale");
    (void)runtimeScalar<Accumulator>(problem.activationParameter0, "activation parameter 0");
    (void)runtimeScalar<Accumulator>(problem.activationParameter1, "activation parameter 1");
}

inline void validateRuntimeGemmProblem(const GemmSpecification& problem) {
    requireRank(problem.a.shape(), 2, "Reference GEMM", "A");
    requireRank(problem.b.shape(), 2, "Reference GEMM", "B");
    requireRank(problem.c.shape(), 2, "Reference GEMM", "C");

    const size_t m = problem.a.shape()[0];
    const size_t k = problem.a.shape()[1];
    const size_t n = problem.b.shape()[1];
    if (problem.b.shape()[0] != k)
        throw std::invalid_argument("Reference GEMM K dimension mismatch.");
    if (problem.c.shape() != Shape{m, n})
        throw std::invalid_argument("Reference GEMM C shape mismatch.");
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
    validateOperandType(problem.a.type(), "A");
    validateOperandType(problem.b.type(), "B");
    validateOperandType(problem.c.type(), "C");
    validateOperandType(problem.outputType, "D");
    if (complexAccumulator != isComplexScalarType(problem.outputType))
        throw std::invalid_argument("Reference GEMM complex accumulator/output mismatch.");

    auto validateComputeType = [&](const Tensor& operand,
                                   const std::optional<ScalarType>& computeType, const char* name) {
        if (!computeType) return;
        validateOperandType(*computeType, name);
        if (isComplexScalarType(operand.type()) && !isComplexScalarType(*computeType))
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " compute-input type has incompatible complexity.");
    };
    validateComputeType(problem.a, problem.computeTypeA, "A");
    validateComputeType(problem.b, problem.computeTypeB, "B");
    auto validatePreQuantizationScales = [&](const Tensor& operand,
                                             const std::vector<Tensor>& scales, const char* name) {
        for (const Tensor& scale : scales) {
            try {
                (void)scale.broadcastTo(operand.shape());
            } catch (const std::invalid_argument&) {
                throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                            " is not broadcast-compatible with its operand.");
            }
            if (!complexAccumulator && isComplexScalarType(scale.type()))
                throw std::invalid_argument(
                    std::string("Reference GEMM real accumulator cannot consume complex ") + name +
                    ".");
        }
    };
    validatePreQuantizationScales(problem.a, problem.preQuantizationScalesA,
                                  "A pre-quantization scale");
    validatePreQuantizationScales(problem.b, problem.preQuantizationScalesB,
                                  "B pre-quantization scale");

    if (problem.mathMode == MathMode::XFloat32 && problem.accumulatorType != ScalarType::Float32)
        throw std::invalid_argument("XFloat32 math mode requires a Float32 accumulator.");
    if (complexAccumulator && problem.activation != Activation::None)
        throw std::invalid_argument("Complex reference GEMM does not support activation.");
    if (problem.accumulatorType == ScalarType::Int32) {
        switch (problem.activation) {
            case Activation::None:
            case Activation::Absolute:
            case Activation::ClippedRelu:
            case Activation::Relu:
            case Activation::LeakyRelu:
            case Activation::ReluDerivative:
            case Activation::Clamp:
                break;
            default:
                throw std::invalid_argument(
                    "Int32 reference GEMM does not support floating-point activation.");
        }
    }
    switch (problem.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            validateRuntimeGemmScalars<float>(problem);
            break;
        case ScalarType::Float64:
            validateRuntimeGemmScalars<double>(problem);
            break;
        case ScalarType::Int32:
            validateRuntimeGemmScalars<int32_t>(problem);
            break;
        case ScalarType::ComplexFloat32:
            validateRuntimeGemmScalars<std::complex<float>>(problem);
            break;
        case ScalarType::ComplexFloat64:
            validateRuntimeGemmScalars<std::complex<double>>(problem);
            break;
        default:
            throw std::invalid_argument("Unsupported runtime reference GEMM accumulator type.");
    }
    if (problem.outputConversion == OutputConversion::SaturatingInt8 &&
        problem.outputType != ScalarType::Int8)
        throw std::invalid_argument(
            "Reference GEMM saturating output conversion currently requires Int8 output.");

    const Shape outputShape{m, n};
    auto validateEpilogueTensor = [&](const Tensor& values, const char* name) {
        try {
            (void)values.broadcastTo(outputShape);
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " is not broadcast-compatible with D.");
        }
        if (!complexAccumulator && isComplexScalarType(values.type()))
            throw std::invalid_argument(
                std::string("Reference GEMM real accumulator cannot consume complex ") + name +
                ".");
    };
    if (problem.bias) validateEpilogueTensor(*problem.bias, "bias");
    if (problem.scaleAlpha) validateEpilogueTensor(*problem.scaleAlpha, "scale-alpha");
    if (problem.scaleA) validateEpilogueTensor(*problem.scaleA, "scale-A");
    if (problem.scaleB) validateEpilogueTensor(*problem.scaleB, "scale-B");

    auto validateBlockScale = [&](const std::optional<Tensor>& scale, size_t blockSize,
                                  size_t freeExtent, const char* name) {
        if (!scale) {
            if (blockSize != 0)
                throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                            " block size requires a scale tensor.");
            return;
        }
        if (blockSize == 0)
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block size must be nonzero.");
        requireRank(scale->shape(), 2, "Reference GEMM", name);
        const size_t blockCount = k / blockSize + (k % blockSize != 0 ? 1 : 0);
        if (scale->shape()[0] != freeExtent || scale->shape()[1] < blockCount)
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block-scale shape mismatch.");
        if (isComplexScalarType(scale->type()))
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block scales must be real.");
    };
    validateBlockScale(problem.blockScaleA, problem.blockSizeA, m, "A");
    validateBlockScale(problem.blockScaleB, problem.blockSizeB, n, "B");
    if (problem.blockScaleA) {
        if (complexAccumulator)
            throw std::invalid_argument("Complex reference GEMM does not support block scaling.");
    }
    if (problem.blockScaleB) {
        if (complexAccumulator)
            throw std::invalid_argument("Complex reference GEMM does not support block scaling.");
    }
}

inline bool canParallelizeGemmOutput(const GemmInvocation& problem) {
    return hasProvablyIndependentElements(problem.d);
}

inline bool hasSameStorageTypeAndLayout(const Tensor& left, const Tensor& right) {
    const auto leftStorage = left.rawEncodedBackingStorage();
    const auto rightStorage = right.rawEncodedBackingStorage();
    return left.type() == right.type() && left.layout() == right.layout() &&
           leftStorage.data() == rightStorage.data() && leftStorage.size() == rightStorage.size();
}

inline void validateGemmOutputAliasing(const GemmInvocation& problem) {
    if (!hasProvablyDistinctElementOffsets(problem.d.layout()))
        throw std::invalid_argument(
            "Reference GEMM requires distinct logical destination elements.");

    if (storageOverlaps(problem.d, problem.a) || storageOverlaps(problem.d, problem.b))
        throw std::invalid_argument("Reference GEMM destination must not overlap A or B.");

    if (storageOverlaps(problem.d, problem.c) && !hasSameStorageTypeAndLayout(problem.d, problem.c))
        throw std::invalid_argument(
            "Reference GEMM permits C and D to overlap only as the same tensor layout.");

    for (const Tensor& scale : problem.preQuantizationScalesA)
        if (storageOverlaps(problem.d, scale))
            throw std::invalid_argument(
                "Reference GEMM destination must not overlap an A pre-quantization scale.");
    for (const Tensor& scale : problem.preQuantizationScalesB)
        if (storageOverlaps(problem.d, scale))
            throw std::invalid_argument(
                "Reference GEMM destination must not overlap a B pre-quantization scale.");
    if (problem.blockScaleA && storageOverlaps(problem.d, *problem.blockScaleA))
        throw std::invalid_argument(
            "Reference GEMM destination must not overlap the A block scale.");
    if (problem.blockScaleB && storageOverlaps(problem.d, *problem.blockScaleB))
        throw std::invalid_argument(
            "Reference GEMM destination must not overlap the B block scale.");
    if (problem.bias && storageOverlaps(problem.d, *problem.bias))
        throw std::invalid_argument("Reference GEMM destination must not overlap the bias.");
    if (problem.scaleAlpha && storageOverlaps(problem.d, *problem.scaleAlpha))
        throw std::invalid_argument("Reference GEMM destination must not overlap the alpha scale.");
    if (problem.scaleA && storageOverlaps(problem.d, *problem.scaleA))
        throw std::invalid_argument("Reference GEMM destination must not overlap scale A.");
    if (problem.scaleB && storageOverlaps(problem.d, *problem.scaleB))
        throw std::invalid_argument("Reference GEMM destination must not overlap scale B.");
}

inline void validateRuntimeGemm(const GemmInvocation& problem) {
    validateRuntimeGemmProblem(problem);
    requireRank(problem.d.shape(), 2, "Reference GEMM", "D");

    const Shape expectedShape{problem.a.shape()[0], problem.b.shape()[1]};
    if (problem.d.shape() != expectedShape)
        throw std::invalid_argument("Reference GEMM D shape mismatch.");
    if (problem.d.type() != problem.outputType)
        throw std::invalid_argument(
            "Reference GEMM destination type does not match the problem output type.");
    (void)problem.outputSelection.selectedCount(problem.d.shape().elementCount());
    validateGemmOutputAliasing(problem);
}

template <typename Accumulator>
class RuntimeGemmFinalizer {
   public:
    explicit RuntimeGemmFinalizer(
        const GemmSpecification& problem,
        RuntimeQuantizer<Accumulator> quantizeAccumulator = RuntimeQuantizer<Accumulator>())
        : m_problem(problem),
          m_c(problem.c),
          m_quantizeAccumulator(std::move(quantizeAccumulator)),
          m_alpha(m_quantizeAccumulator(runtimeScalar<Accumulator>(problem.alpha, "alpha"))),
          m_beta(m_quantizeAccumulator(runtimeScalar<Accumulator>(problem.beta, "beta"))),
          m_scaleC(m_quantizeAccumulator(runtimeScalar<Accumulator>(problem.scaleC, "C scale"))),
          m_outputScale(runtimeScalar<Accumulator>(problem.outputScale, "output scale")),
          m_activationParameter0(m_quantizeAccumulator(
              runtimeScalar<Accumulator>(problem.activationParameter0, "activation parameter 0"))),
          m_activationParameter1(m_quantizeAccumulator(
              runtimeScalar<Accumulator>(problem.activationParameter1, "activation parameter 1"))),
          m_alphaIsZero(m_alpha == Accumulator(0)),
          m_betaIsZero(m_beta == Accumulator(0)) {
        if (problem.bias) m_bias.emplace(problem.bias->broadcastTo(problem.c.shape()));
        if (problem.scaleAlpha)
            m_scaleAlpha.emplace(problem.scaleAlpha->broadcastTo(problem.c.shape()));
        if (problem.scaleA) m_scaleA.emplace(problem.scaleA->broadcastTo(problem.c.shape()));
        if (problem.scaleB) m_scaleB.emplace(problem.scaleB->broadcastTo(problem.c.shape()));
    }

    bool alphaIsZero() const {
        return m_alphaIsZero;
    }

    Accumulator multiply(Accumulator left, Accumulator right) const {
        return m_quantizeAccumulator(wrappingMultiply(left, right));
    }

    Accumulator add(Accumulator left, Accumulator right) const {
        return m_quantizeAccumulator(wrappingAdd(left, right));
    }

    // Finalize a backend-produced raw accumulator.
    Accumulator finalize(size_t row, size_t column, Accumulator accumulation) const {
        Accumulator effectiveAlpha = m_alpha;
        if (!m_alphaIsZero) {
            if (m_scaleA) effectiveAlpha = multiply(effectiveAlpha, (*m_scaleA)(row, column));
            if (m_scaleB) effectiveAlpha = multiply(effectiveAlpha, (*m_scaleB)(row, column));
            if (m_scaleAlpha)
                effectiveAlpha = multiply(effectiveAlpha, (*m_scaleAlpha)(row, column));
        }

        Accumulator result = multiply(effectiveAlpha, accumulation);
        if (!m_betaIsZero)
            result = add(result, multiply(multiply(m_beta, m_scaleC), m_c(row, column)));
        return finalizeCombined(row, column, result);
    }

    // Finalize a value whose alpha/beta combination was already performed by
    // the backend, preserving that backend's established floating-point order.
    Accumulator finalizeCombined(size_t row, size_t column, Accumulator result) const {
        if (m_bias) result = add(result, (*m_bias)(row, column));
        result = m_quantizeAccumulator(applyActivation(
            m_problem.activation, result, m_activationParameter0, m_activationParameter1));
        result = multiply(result, m_outputScale);
        return result;
    }

   private:
    const GemmSpecification& m_problem;
    RuntimeMatrixReader<Accumulator> m_c;
    RuntimeQuantizer<Accumulator> m_quantizeAccumulator;
    std::optional<RuntimeMatrixReader<Accumulator>> m_bias;
    std::optional<RuntimeMatrixReader<Accumulator>> m_scaleAlpha;
    std::optional<RuntimeMatrixReader<Accumulator>> m_scaleA;
    std::optional<RuntimeMatrixReader<Accumulator>> m_scaleB;
    Accumulator m_alpha;
    Accumulator m_beta;
    Accumulator m_scaleC;
    Accumulator m_outputScale;
    Accumulator m_activationParameter0;
    Accumulator m_activationParameter1;
    bool m_alphaIsZero;
    bool m_betaIsZero;
};

template <typename Accumulator>
GemmExecutionInfo runPointwiseGemmTyped(const GemmInvocation& problem,
                                        Tensor* selectedOutput = nullptr) {
    const RuntimeMatrixReader<Accumulator> a(problem.a);
    const RuntimeMatrixReader<Accumulator> b(problem.b);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.computeTypeA);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.computeTypeB);
    const bool typeRoundsAfterEachStep = problem.accumulatorType == ScalarType::Float16 ||
                                         problem.accumulatorType == ScalarType::BFloat16;
    const bool roundAfterEachStep =
        problem.accumulationRounding == AccumulationRounding::AfterProductAndSum ||
        (problem.accumulationRounding == AccumulationRounding::TypeDefault &&
         typeRoundsAfterEachStep);
    const RuntimeQuantizer<Accumulator> quantizeAccumulator(
        roundAfterEachStep ? std::optional<ScalarType>(problem.accumulatorType) : std::nullopt);
    const RuntimeGemmFinalizer<Accumulator> finalizer(problem, quantizeAccumulator);
    const RuntimeMatrixOutputWriter<Accumulator> output(problem.d, problem.outputConversion);
    std::optional<RuntimeMatrixOutputWriter<Accumulator>> selectedOutputWriter;
    if (selectedOutput != nullptr)
        selectedOutputWriter.emplace(*selectedOutput, problem.outputConversion);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);

    std::vector<RuntimeMatrixReader<Accumulator>> preScalesA;
    std::vector<RuntimeMatrixReader<Accumulator>> preScalesB;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleA;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleB;
    preScalesA.reserve(problem.preQuantizationScalesA.size());
    for (const Tensor& scale : problem.preQuantizationScalesA)
        preScalesA.emplace_back(scale.broadcastTo(problem.a.shape()));
    preScalesB.reserve(problem.preQuantizationScalesB.size());
    for (const Tensor& scale : problem.preQuantizationScalesB)
        preScalesB.emplace_back(scale.broadcastTo(problem.b.shape()));
    if (problem.blockScaleA) blockScaleA.emplace(*problem.blockScaleA);
    if (problem.blockScaleB) blockScaleB.emplace(*problem.blockScaleB);

    const size_t m = problem.a.shape()[0];
    const size_t k = problem.a.shape()[1];
    const size_t n = problem.b.shape()[1];

    auto computeOutput = [&](size_t row, size_t column, size_t selectedIndex) {
        Accumulator sum = Accumulator(0);

        if (!finalizer.alphaIsZero() && (blockScaleA || blockScaleB)) {
            size_t blockBase = 0;
            while (blockBase < k) {
                const size_t remainingA = blockScaleA
                                              ? problem.blockSizeA - blockBase % problem.blockSizeA
                                              : k - blockBase;
                const size_t remainingB = blockScaleB
                                              ? problem.blockSizeB - blockBase % problem.blockSizeB
                                              : k - blockBase;
                const size_t blockLength = std::min({k - blockBase, remainingA, remainingB});
                const size_t blockEnd = blockBase + blockLength;
                Accumulator blockSum = Accumulator(0);
                for (size_t reduction = blockBase; reduction < blockEnd; ++reduction) {
                    Accumulator aValue = conjugateIfNeeded(a(row, reduction), problem.conjugateA);
                    Accumulator bValue =
                        conjugateIfNeeded(b(reduction, column), problem.conjugateB);
                    for (const auto& scale : preScalesA)
                        aValue = finalizer.multiply(aValue, scale(row, reduction));
                    for (const auto& scale : preScalesB)
                        bValue = finalizer.multiply(bValue, scale(reduction, column));
                    aValue = operandMath(quantizeA(aValue));
                    bValue = operandMath(quantizeB(bValue));
                    blockSum = finalizer.add(blockSum, finalizer.multiply(aValue, bValue));
                }

                Accumulator scale = Accumulator(1);
                if (blockScaleA)
                    scale = finalizer.multiply(scale,
                                               (*blockScaleA)(row, blockBase / problem.blockSizeA));
                if (blockScaleB)
                    scale = finalizer.multiply(
                        scale, (*blockScaleB)(column, blockBase / problem.blockSizeB));
                sum = finalizer.add(sum, finalizer.multiply(blockSum, scale));
                blockBase = blockEnd;
            }
        } else if (!finalizer.alphaIsZero()) {
            for (size_t reduction = 0; reduction < k; ++reduction) {
                Accumulator aValue = conjugateIfNeeded(a(row, reduction), problem.conjugateA);
                Accumulator bValue = conjugateIfNeeded(b(reduction, column), problem.conjugateB);
                for (const auto& scale : preScalesA)
                    aValue = finalizer.multiply(aValue, scale(row, reduction));
                for (const auto& scale : preScalesB)
                    bValue = finalizer.multiply(bValue, scale(reduction, column));
                aValue = operandMath(quantizeA(aValue));
                bValue = operandMath(quantizeB(bValue));
                sum = finalizer.add(sum, finalizer.multiply(aValue, bValue));
            }
        }

        const Accumulator value = finalizer.finalize(row, column, sum);
        if (selectedOutputWriter)
            selectedOutputWriter->store(0, selectedIndex, value);
        else
            output.store(row, column, value);
    };

    const size_t logicalElements = problem.d.shape().elementCount();
    size_t outputElementsWritten = 0;
    const bool parallelOutput = canParallelizeGemmOutput(problem);
    const size_t reductionWork = finalizer.alphaIsZero() ? 0 : k;
    if (problem.outputSelection.selectsAll()) {
        outputElementsWritten = logicalElements;
        forEachParallelIndex(logicalElements, saturatedProduct(logicalElements, reductionWork),
                             parallelOutput, 500'000, [&](size_t logicalIndex) {
                                 computeOutput(logicalIndex / n, logicalIndex % n, logicalIndex);
                             });
    } else {
        const auto selected = problem.outputSelection.indices(logicalElements);
        const auto& outputShape = problem.d.shape();
        outputElementsWritten = selected.size();
        forEachParallelIndex(selected.size(), saturatedProduct(selected.size(), reductionWork),
                             parallelOutput, 500'000, [&](size_t selectionIndex) {
                                 const size_t logicalIndex = selected[selectionIndex];
                                 const auto coordinates = outputShape.coordinates(
                                     logicalIndex, problem.outputSelection.indexOrder());
                                 computeOutput(coordinates[0], coordinates[1], selectionIndex);
                             });
    }

    return {
        .backendUsed = GemmBackend::Pointwise,
        .fallbackReason = std::nullopt,
        .outputElementsWritten = outputElementsWritten,
        .outputElementsCovered = outputElementsWritten,
    };
}

inline GemmExecutionInfo runPointwiseGemm(const GemmInvocation& problem) {
    switch (problem.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            return runPointwiseGemmTyped<float>(problem);
        case ScalarType::Float64:
            return runPointwiseGemmTyped<double>(problem);
        case ScalarType::Int32:
            return runPointwiseGemmTyped<int32_t>(problem);
        case ScalarType::ComplexFloat32:
            return runPointwiseGemmTyped<std::complex<float>>(problem);
        case ScalarType::ComplexFloat64:
            return runPointwiseGemmTyped<std::complex<double>>(problem);
        default:
            throw std::invalid_argument("Unsupported runtime reference GEMM accumulator type.");
    }
}

inline GemmExecutionInfo runPointwiseGemmToSelectedOutput(const GemmInvocation& problem,
                                                          Tensor& selectedOutput) {
    if (problem.outputSelection.selectsAll())
        throw std::invalid_argument("Streaming pointwise GEMM requires a partial selection.");
    const size_t selectedCount =
        problem.outputSelection.selectedCount(problem.d.shape().elementCount());
    if (selectedOutput.type() != problem.outputType ||
        selectedOutput.shape() != Shape{1, selectedCount})
        throw std::invalid_argument("Streaming pointwise GEMM output shape or type mismatch.");

    switch (problem.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            return runPointwiseGemmTyped<float>(problem, &selectedOutput);
        case ScalarType::Float64:
            return runPointwiseGemmTyped<double>(problem, &selectedOutput);
        case ScalarType::Int32:
            return runPointwiseGemmTyped<int32_t>(problem, &selectedOutput);
        case ScalarType::ComplexFloat32:
            return runPointwiseGemmTyped<std::complex<float>>(problem, &selectedOutput);
        case ScalarType::ComplexFloat64:
            return runPointwiseGemmTyped<std::complex<double>>(problem, &selectedOutput);
        default:
            throw std::invalid_argument("Unsupported runtime reference GEMM accumulator type.");
    }
}
}  // namespace detail
}  // namespace roc::host_numerics
