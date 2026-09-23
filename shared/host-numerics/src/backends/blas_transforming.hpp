// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cblas.h>

#include <complex>
#include <cstddef>
#include <limits>
#include <optional>
#include <roc/host_numerics/backends/blas.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "blas_direct.hpp"
#include "detail/reference_gemm.hpp"
#include "detail/tensor_conversion.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics {
namespace {
Layout columnMajorLayout(const Shape& shape) {
    return Layout(shape, {1, static_cast<ptrdiff_t>(shape[0])});
}

size_t saturatedSum(size_t left, size_t right) {
    if (right > std::numeric_limits<size_t>::max() - left)
        return std::numeric_limits<size_t>::max();
    return left + right;
}

void validateTransforming(const GemmInvocation& problem) {
    detail::validateRuntimeGemm(problem);

    switch (problem.accumulatorType) {
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            break;
        default:
            throw std::invalid_argument(
                "Transforming BLAS backend supports F32, F64, C64, and C128 accumulation.");
    }
    if (problem.blockScaleA || problem.blockScaleB)
        throw std::invalid_argument(
            "Transforming BLAS backend cannot preserve block-scale reduction boundaries.");
    if (!problem.outputSelection.selectsAll())
        throw std::invalid_argument(
            "Transforming BLAS backend requires complete output selection.");

    const size_t m = problem.a.shape()[0];
    const size_t n = problem.b.shape()[1];
    const size_t k = problem.a.shape()[1];
    if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Transforming BLAS backend dimensions exceed int.");
}

template <typename Accumulator>
Tensor materializeOperand(const Tensor& operand, const std::optional<ScalarType>& computeType,
                          const std::vector<Tensor>& preQuantizationScales, bool conjugate,
                          MathMode mathMode) {
    using namespace detail;
    Tensor output = Tensor::allocateUninitialized(nativeScalarType<Accumulator>,
                                                  columnMajorLayout(operand.shape()));
    if (!requiresInputQuantization(operand.type(), computeType, nativeScalarType<Accumulator>) &&
        preQuantizationScales.empty() && !conjugate && mathMode == MathMode::Default) {
        const TensorConversion conversion(operand, output,
                                          implicitStorageConversionOptions(output.type()));
        forEachParallelIndex(conversion.blockCount(), conversion.elementCount(),
                             conversion.canParallelize(), 500'000,
                             [&](size_t block) { conversion.copyBlock(block); });
        return output;
    }
    const RuntimeMatrixReader<Accumulator> input(operand);
    const RuntimeMatrixWriter<Accumulator> writer(output);
    const RuntimeQuantizer<Accumulator> quantize(computeType);
    const RuntimeMathFunction<Accumulator> operandMath = runtimeMathFunction<Accumulator>(mathMode);
    std::vector<RuntimeMatrixReader<Accumulator>> scaleReaders;
    scaleReaders.reserve(preQuantizationScales.size());
    for (const Tensor& scale : preQuantizationScales)
        scaleReaders.emplace_back(scale.broadcastTo(operand.shape()));
    const size_t rows = operand.shape()[0];
    const size_t columns = operand.shape()[1];
    const size_t elementCount = detail::saturatedProduct(rows, columns);
    detail::forEachParallelIndex(
        elementCount, elementCount, true, 500'000, [&](size_t linearIndex) {
            const size_t column = linearIndex / rows;
            const size_t row = linearIndex % rows;
            Accumulator value = conjugateIfNeeded(input(row, column), conjugate);
            for (const auto& scale : scaleReaders) value *= scale(row, column);
            value = operandMath(quantize(value));
            writer.store(row, column, value);
        });
    return output;
}

template <typename Accumulator>
bool canUseBlasOperandWithoutMaterialization(const Tensor& operand,
                                             const std::optional<ScalarType>& computeType,
                                             const std::vector<Tensor>& preQuantizationScales,
                                             bool conjugate, MathMode mathMode, const char* name) {
    const bool requiresValueTransform =
        operand.type() != nativeScalarType<Accumulator> ||
        detail::requiresInputQuantization(operand.type(), computeType,
                                          nativeScalarType<Accumulator>) ||
        !preQuantizationScales.empty() || mathMode != MathMode::Default;
    if (requiresValueTransform) return false;
    try {
        (void)toBlasLayout(operand, conjugate, name);
        (void)typedData<Accumulator>(operand, name);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

bool canUseBlasOperandWithoutMaterialization(const Tensor& operand,
                                             const std::optional<ScalarType>& computeType,
                                             const std::vector<Tensor>& preQuantizationScales,
                                             bool conjugate, ScalarType accumulatorType,
                                             MathMode mathMode, const char* name) {
    switch (accumulatorType) {
        case ScalarType::Float32:
            return canUseBlasOperandWithoutMaterialization<float>(
                operand, computeType, preQuantizationScales, conjugate, mathMode, name);
        case ScalarType::Float64:
            return canUseBlasOperandWithoutMaterialization<double>(
                operand, computeType, preQuantizationScales, conjugate, mathMode, name);
        case ScalarType::ComplexFloat32:
            return canUseBlasOperandWithoutMaterialization<std::complex<float>>(
                operand, computeType, preQuantizationScales, conjugate, mathMode, name);
        case ScalarType::ComplexFloat64:
            return canUseBlasOperandWithoutMaterialization<std::complex<double>>(
                operand, computeType, preQuantizationScales, conjugate, mathMode, name);
        default:
            return false;
    }
}

template <typename Accumulator>
std::pair<Tensor, bool> prepareBlasOperand(const Tensor& operand,
                                           const std::optional<ScalarType>& computeType,
                                           const std::vector<Tensor>& preQuantizationScales,
                                           bool conjugate, MathMode mathMode, const char* name) {
    if (canUseBlasOperandWithoutMaterialization<Accumulator>(
            operand, computeType, preQuantizationScales, conjugate, mathMode, name))
        return {operand, conjugate};
    return {materializeOperand<Accumulator>(operand, computeType, preQuantizationScales, conjugate,
                                            mathMode),
            false};
}

template <typename Accumulator>
GemmExecutionInfo runTransforming(const GemmInvocation& problem) {
    using namespace detail;
    static const BlasGemmBackend blas;
    if (problem.d.elementCount() == 0)
        return {
            .backendUsed = GemmBackend::Blas,
            .fallbackReason = std::nullopt,
            .outputElementsWritten = 0,
            .outputElementsCovered = 0,
        };
    if (problem.a.shape()[1] != 0 && blas.querySupport(problem)) return blas.run(problem);

    const bool hasReductionElements = problem.a.shape()[1] != 0;
    Tensor stagedOutput =
        hasReductionElements
            ? Tensor::allocateUninitialized(nativeScalarType<Accumulator>,
                                            columnMajorLayout(problem.d.shape()))
            : Tensor(nativeScalarType<Accumulator>, columnMajorLayout(problem.d.shape()));
    if (hasReductionElements) {
        auto [stagedA, conjugateA] = prepareBlasOperand<Accumulator>(
            problem.a, problem.computeTypeA, problem.preQuantizationScalesA, problem.conjugateA,
            problem.mathMode, "A");
        auto [stagedB, conjugateB] = prepareBlasOperand<Accumulator>(
            problem.b, problem.computeTypeB, problem.preQuantizationScalesB, problem.conjugateB,
            problem.mathMode, "B");
        MatmulOptions stagedOptions(nativeScalarType<Accumulator>);
        stagedOptions.conjugateA = conjugateA;
        stagedOptions.conjugateB = conjugateB;
        GemmInvocation stagedProblem(std::move(stagedA), std::move(stagedB), stagedOutput,
                                     stagedOptions);

        blas.run(stagedProblem);
    }

    const TensorConversion outputConversion(stagedOutput, problem.d,
                                            implicitStorageConversionOptions(problem.d.type()));
    detail::forEachParallelIndex(outputConversion.blockCount(), outputConversion.elementCount(),
                                 outputConversion.canParallelize(), 500'000,
                                 [&](size_t block) { outputConversion.copyBlock(block); });

    return {
        .backendUsed = GemmBackend::Blas,
        .fallbackReason = std::nullopt,
        .outputElementsWritten = problem.d.shape().elementCount(),
        .outputElementsCovered = problem.d.shape().elementCount(),
    };
}

GemmSupportInfo queryTransformingBlasGemmSupport(const GemmInvocation& problem) {
    try {
        validateTransforming(problem);
        if (problem.d.elementCount() == 0 || problem.a.shape()[1] == 0)
            return {
                .supported = true,
                .reason = {},
                .preferredForAutomaticExecution = false,
            };
        static const BlasGemmBackend directBlas;
        if (directBlas.querySupport(problem)) return {.supported = true, .reason = {}};

        const size_t rows = problem.a.shape()[0];
        const size_t columns = problem.b.shape()[1];
        const size_t reductions = problem.a.shape()[1];
        const size_t multiplyAdds =
            detail::saturatedProduct(detail::saturatedProduct(rows, columns), reductions);
        const size_t stagedAElements =
            canUseBlasOperandWithoutMaterialization(
                problem.a, problem.computeTypeA, problem.preQuantizationScalesA, problem.conjugateA,
                problem.accumulatorType, problem.mathMode, "A")
                ? 0
                : detail::saturatedProduct(rows, reductions);
        const size_t stagedBElements =
            canUseBlasOperandWithoutMaterialization(
                problem.b, problem.computeTypeB, problem.preQuantizationScalesB, problem.conjugateB,
                problem.accumulatorType, problem.mathMode, "B")
                ? 0
                : detail::saturatedProduct(reductions, columns);
        const size_t stagedOperandElements = saturatedSum(stagedAElements, stagedBElements);
        const size_t stagedAndFinalOutputElements =
            detail::saturatedProduct(detail::saturatedProduct(rows, columns), size_t{2});
        const size_t stagedElements =
            saturatedSum(stagedOperandElements, stagedAndFinalOutputElements);

        constexpr size_t minimumMultiplyAdds = 1'000'000;
        constexpr size_t minimumArithmeticIntensity = 8;
        return {
            .supported = true,
            .reason = {},
            .preferredForAutomaticExecution =
                multiplyAdds >= minimumMultiplyAdds &&
                multiplyAdds >=
                    detail::saturatedProduct(stagedElements, minimumArithmeticIntensity),
        };
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }
}

GemmExecutionInfo runTransformingBlasGemm(const GemmInvocation& problem) {
    const GemmSupportInfo support = queryTransformingBlasGemmSupport(problem);
    if (!support) throw std::invalid_argument(support.reason);

    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            return runTransforming<float>(problem);
        case ScalarType::Float64:
            return runTransforming<double>(problem);
        case ScalarType::ComplexFloat32:
            return runTransforming<std::complex<float>>(problem);
        case ScalarType::ComplexFloat64:
            return runTransforming<std::complex<double>>(problem);
        default:
            throw std::invalid_argument(
                "Transforming BLAS backend accumulator type is unsupported.");
    }
}
}  // namespace
}  // namespace roc::host_numerics
