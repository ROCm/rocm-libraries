// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/gemm.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "reference_common.hpp"
#include "threading.hpp"

namespace roc::host_validation {
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
void validateRuntimeGemmScalars(const GemmProblem& problem) {
    (void)runtimeScalar<Accumulator>(problem.epilogue.alpha, "alpha");
    (void)runtimeScalar<Accumulator>(problem.epilogue.beta, "beta");
    (void)runtimeScalar<Accumulator>(problem.epilogue.outputScale, "output scale");
    (void)runtimeScalar<Accumulator>(problem.epilogue.activationParameter0,
                                     "activation parameter 0");
    (void)runtimeScalar<Accumulator>(problem.epilogue.activationParameter1,
                                     "activation parameter 1");
}

inline void validateRuntimeGemmProblem(const GemmProblem& problem) {
    requireRank(problem.a.values.shape(), 2, "Reference GEMM", "A");
    requireRank(problem.b.values.shape(), 2, "Reference GEMM", "B");
    requireRank(problem.c.shape(), 2, "Reference GEMM", "C");

    const size_t m = problem.a.values.shape()[0];
    const size_t k = problem.a.values.shape()[1];
    const size_t n = problem.b.values.shape()[1];
    if (problem.b.values.shape()[0] != k)
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
    validateOperandType(problem.a.values.type(), "A");
    validateOperandType(problem.b.values.type(), "B");
    validateOperandType(problem.c.type(), "C");
    validateOperandType(problem.outputType, "D");
    if (complexAccumulator != isComplexScalarType(problem.outputType))
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
    if (problem.accumulatorType == ScalarType::Int32) {
        switch (problem.epilogue.activation) {
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
    if (problem.epilogue.outputConversion == OutputConversion::SaturatingInt8 &&
        problem.outputType != ScalarType::Int8)
        throw std::invalid_argument(
            "Reference GEMM saturating output conversion currently requires Int8 output.");

    auto validateEpilogueVector = [&](const Tensor& values, size_t expected, const char* name) {
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
}

inline bool canParallelizeGemmOutput(const GemmRequest& problem) {
    if (!hasProvablyIndependentElements(problem.d)) return false;
    if (storageOverlaps(problem.d, problem.a.values) ||
        storageOverlaps(problem.d, problem.b.values) || storageOverlaps(problem.d, problem.c))
        return false;
    for (const VectorBinding& binding : problem.a.preQuantizationScales)
        if (storageOverlaps(problem.d, binding.values)) return false;
    for (const VectorBinding& binding : problem.b.preQuantizationScales)
        if (storageOverlaps(problem.d, binding.values)) return false;
    if (problem.a.blockScale && storageOverlaps(problem.d, problem.a.blockScale->values))
        return false;
    if (problem.b.blockScale && storageOverlaps(problem.d, problem.b.blockScale->values))
        return false;
    if (problem.epilogue.bias && storageOverlaps(problem.d, problem.epilogue.bias->values))
        return false;
    if (problem.epilogue.scaleAlpha &&
        storageOverlaps(problem.d, problem.epilogue.scaleAlpha->values))
        return false;
    if (problem.epilogue.scaleA && storageOverlaps(problem.d, *problem.epilogue.scaleA))
        return false;
    if (problem.epilogue.scaleB && storageOverlaps(problem.d, *problem.epilogue.scaleB))
        return false;
    return true;
}

inline void validateRuntimeGemm(const GemmRequest& problem) {
    validateRuntimeGemmProblem(problem);
    requireRank(problem.d.shape(), 2, "Reference GEMM", "D");

    const Shape expectedShape{problem.a.values.shape()[0], problem.b.values.shape()[1]};
    if (problem.d.shape() != expectedShape)
        throw std::invalid_argument("Reference GEMM D shape mismatch.");
    if (problem.d.type() != problem.outputType)
        throw std::invalid_argument(
            "Reference GEMM destination type does not match the problem output type.");
    (void)problem.outputSelection.selectedCount(problem.d.shape().elementCount());
}

template <typename Accumulator>
class RuntimeGemmFinalizer {
   public:
    explicit RuntimeGemmFinalizer(
        const GemmProblem& problem,
        RuntimeQuantizer<Accumulator> quantizeAccumulator = RuntimeQuantizer<Accumulator>())
        : m_problem(problem),
          m_c(problem.c),
          m_quantizeAccumulator(std::move(quantizeAccumulator)),
          m_alpha(
              m_quantizeAccumulator(runtimeScalar<Accumulator>(problem.epilogue.alpha, "alpha"))),
          m_beta(m_quantizeAccumulator(runtimeScalar<Accumulator>(problem.epilogue.beta, "beta"))),
          m_outputScale(runtimeScalar<Accumulator>(problem.epilogue.outputScale, "output scale")),
          m_activationParameter0(m_quantizeAccumulator(runtimeScalar<Accumulator>(
              problem.epilogue.activationParameter0, "activation parameter 0"))),
          m_activationParameter1(m_quantizeAccumulator(runtimeScalar<Accumulator>(
              problem.epilogue.activationParameter1, "activation parameter 1"))),
          m_alphaIsZero(m_alpha == Accumulator(0)),
          m_betaIsZero(m_beta == Accumulator(0)) {
        if (problem.epilogue.bias) m_bias.emplace(problem.epilogue.bias->values);
        if (problem.epilogue.scaleAlpha) m_scaleAlpha.emplace(problem.epilogue.scaleAlpha->values);
        if (problem.epilogue.scaleA) m_scaleA.emplace(*problem.epilogue.scaleA);
        if (problem.epilogue.scaleB) m_scaleB.emplace(*problem.epilogue.scaleB);
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
            if (m_scaleA) effectiveAlpha = multiply(effectiveAlpha, (*m_scaleA)[row]);
            if (m_scaleB) effectiveAlpha = multiply(effectiveAlpha, (*m_scaleB)[column]);
            if (m_scaleAlpha) {
                const MatrixAxis axis = m_problem.epilogue.scaleAlpha->axis;
                effectiveAlpha = multiply(effectiveAlpha,
                                          (*m_scaleAlpha)[axis == MatrixAxis::Row ? row : column]);
            }
        }

        Accumulator result = multiply(effectiveAlpha, accumulation);
        if (!m_betaIsZero) result = add(result, multiply(m_beta, m_c(row, column)));
        return finalizeCombined(row, column, result);
    }

    // Finalize a value whose alpha/beta combination was already performed by
    // the backend, preserving that backend's established floating-point order.
    Accumulator finalizeCombined(size_t row, size_t column, Accumulator result) const {
        if (m_bias) {
            const MatrixAxis axis = m_problem.epilogue.bias->axis;
            result = add(result, (*m_bias)[axis == MatrixAxis::Row ? row : column]);
        }
        result = m_quantizeAccumulator(applyActivation(
            m_problem.epilogue.activation, result, m_activationParameter0, m_activationParameter1));
        result = multiply(result, m_outputScale);
        return result;
    }

   private:
    const GemmProblem& m_problem;
    RuntimeMatrixReader<Accumulator> m_c;
    RuntimeQuantizer<Accumulator> m_quantizeAccumulator;
    std::optional<RuntimeVectorReader<Accumulator>> m_bias;
    std::optional<RuntimeVectorReader<Accumulator>> m_scaleAlpha;
    std::optional<RuntimeVectorReader<Accumulator>> m_scaleA;
    std::optional<RuntimeVectorReader<Accumulator>> m_scaleB;
    Accumulator m_alpha;
    Accumulator m_beta;
    Accumulator m_outputScale;
    Accumulator m_activationParameter0;
    Accumulator m_activationParameter1;
    bool m_alphaIsZero;
    bool m_betaIsZero;
};

template <typename Accumulator>
GemmRunInfo runPointwiseGemmTyped(const GemmRequest& problem) {
    const RuntimeMatrixReader<Accumulator> a(problem.a.values);
    const RuntimeMatrixReader<Accumulator> b(problem.b.values);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.a.computeType);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.b.computeType);
    const bool typeRoundsAfterEachStep = problem.accumulatorType == ScalarType::Float16 ||
                                         problem.accumulatorType == ScalarType::BFloat16;
    const bool roundAfterEachStep =
        problem.accumulationRounding == AccumulationRounding::AfterProductAndSum ||
        (problem.accumulationRounding == AccumulationRounding::TypeDefault &&
         typeRoundsAfterEachStep);
    const RuntimeQuantizer<Accumulator> quantizeAccumulator(
        roundAfterEachStep ? std::optional<ScalarType>(problem.accumulatorType) : std::nullopt);
    const RuntimeGemmFinalizer<Accumulator> finalizer(problem, quantizeAccumulator);
    const RuntimeMatrixOutputWriter<Accumulator> output(problem.d,
                                                        problem.epilogue.outputConversion);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);

    std::vector<RuntimeVectorReader<Accumulator>> preScalesA;
    std::vector<RuntimeVectorReader<Accumulator>> preScalesB;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleA;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleB;
    preScalesA.reserve(problem.a.preQuantizationScales.size());
    for (const VectorBinding& binding : problem.a.preQuantizationScales)
        preScalesA.emplace_back(binding.values);
    preScalesB.reserve(problem.b.preQuantizationScales.size());
    for (const VectorBinding& binding : problem.b.preQuantizationScales)
        preScalesB.emplace_back(binding.values);
    if (problem.a.blockScale) {
        blockScaleA.emplace(problem.a.blockScale->values);
        blockScaleB.emplace(problem.b.blockScale->values);
    }

    const size_t m = problem.a.values.shape()[0];
    const size_t k = problem.a.values.shape()[1];
    const size_t n = problem.b.values.shape()[1];

    auto computeOutput = [&](size_t row, size_t column) {
        Accumulator sum = Accumulator(0);

        if (!finalizer.alphaIsZero() && blockScaleA) {
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
                        aValue = finalizer.multiply(aValue, preScalesA[scaleIndex][index]);
                    }
                    for (size_t scaleIndex = 0; scaleIndex < preScalesB.size(); ++scaleIndex) {
                        const auto& binding = problem.b.preQuantizationScales[scaleIndex];
                        const size_t index =
                            binding.values.shape()[0] == 1
                                ? 0
                                : (binding.axis == MatrixAxis::Row ? reduction : column);
                        bValue = finalizer.multiply(bValue, preScalesB[scaleIndex][index]);
                    }
                    aValue = operandMath(quantizeA(aValue));
                    bValue = operandMath(quantizeB(bValue));
                    blockSum = finalizer.add(blockSum, finalizer.multiply(aValue, bValue));
                }

                const Accumulator scale =
                    finalizer.multiply((*blockScaleA)(row, blockBase / blockSizeA),
                                       (*blockScaleB)(column, blockBase / blockSizeB));
                sum = finalizer.add(sum, finalizer.multiply(blockSum, scale));
                blockBase = blockEnd;
            }
        } else if (!finalizer.alphaIsZero()) {
            for (size_t reduction = 0; reduction < k; ++reduction) {
                Accumulator aValue = conjugateIfNeeded(a(row, reduction), problem.a.conjugate);
                Accumulator bValue = conjugateIfNeeded(b(reduction, column), problem.b.conjugate);
                for (size_t scaleIndex = 0; scaleIndex < preScalesA.size(); ++scaleIndex) {
                    const auto& binding = problem.a.preQuantizationScales[scaleIndex];
                    const size_t index = binding.values.shape()[0] == 1
                                             ? 0
                                             : (binding.axis == MatrixAxis::Row ? row : reduction);
                    aValue = finalizer.multiply(aValue, preScalesA[scaleIndex][index]);
                }
                for (size_t scaleIndex = 0; scaleIndex < preScalesB.size(); ++scaleIndex) {
                    const auto& binding = problem.b.preQuantizationScales[scaleIndex];
                    const size_t index =
                        binding.values.shape()[0] == 1
                            ? 0
                            : (binding.axis == MatrixAxis::Row ? reduction : column);
                    bValue = finalizer.multiply(bValue, preScalesB[scaleIndex][index]);
                }
                aValue = operandMath(quantizeA(aValue));
                bValue = operandMath(quantizeB(bValue));
                sum = finalizer.add(sum, finalizer.multiply(aValue, bValue));
            }
        }

        output.store(row, column, finalizer.finalize(row, column, sum));
    };

    const size_t logicalElements = problem.d.shape().elementCount();
    size_t outputElementsWritten = 0;
    const bool parallelOutput = canParallelizeGemmOutput(problem);
    const size_t reductionWork = finalizer.alphaIsZero() ? 0 : k;
    if (problem.outputSelection.selectsAll()) {
        outputElementsWritten = logicalElements;
        forEachParallelIndex(logicalElements, saturatedProduct(logicalElements, reductionWork),
                             parallelOutput, 500'000, [&](size_t logicalIndex) {
                                 computeOutput(logicalIndex / n, logicalIndex % n);
                             });
    } else {
        const auto selected = problem.outputSelection.indices(logicalElements);
        outputElementsWritten = selected.size();
        forEachParallelIndex(selected.size(), saturatedProduct(selected.size(), reductionWork),
                             parallelOutput, 500'000, [&](size_t selectionIndex) {
                                 const size_t logicalIndex = selected[selectionIndex];
                                 computeOutput(logicalIndex / n, logicalIndex % n);
                             });
    }

    return {
        .backendUsed = GemmBackend::Pointwise,
        .fallbackReason = std::nullopt,
        .outputElementsWritten = outputElementsWritten,
        .outputElementsCovered = outputElementsWritten,
    };
}

inline GemmRunInfo runPointwiseGemm(const GemmRequest& problem) {
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
}  // namespace detail
}  // namespace roc::host_validation
