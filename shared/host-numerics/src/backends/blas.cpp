// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cblas.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_numerics/backends/blas.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/reference_gemm.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics {
namespace {
using detail::GemmSupportInfo;

// Executes the subset of dense GEMM requests that CBLAS can consume directly.
// A, B, C, D, and the accumulator must already use one matching BLAS scalar
// type, and their views must satisfy the direct-layout and aliasing restrictions.
class BlasGemmBackend final {
   public:
    GemmSupportInfo querySupport(const GemmInvocation& request) const;
    GemmExecutionInfo run(const GemmInvocation& request) const;
};

struct BlasMatrixLayout {
    CBLAS_TRANSPOSE transpose;
    int leadingDimension;
};

BlasMatrixLayout toBlasLayout(const Tensor& view, bool conjugate, const char* name) {
    const ptrdiff_t rowStride = view.layout().strides()[0];
    const ptrdiff_t columnStride = view.layout().strides()[1];
    const size_t rows = view.shape()[0];
    const size_t columns = view.shape()[1];

    if (rowStride == 1) {
        if (conjugate)
            throw std::invalid_argument(
                std::string("BLAS backend cannot conjugate non-transposed ") + name + ".");
        if (columnStride < static_cast<ptrdiff_t>(std::max<size_t>(1, rows)))
            throw std::invalid_argument(std::string("BLAS backend ") + name +
                                        " column stride is smaller than its stored row count.");
        if (columnStride > std::numeric_limits<int>::max())
            throw std::invalid_argument(std::string("BLAS backend ") + name +
                                        " leading dimension exceeds int.");
        return {CblasNoTrans, static_cast<int>(columnStride)};
    }

    if (columnStride == 1) {
        if (rowStride < static_cast<ptrdiff_t>(std::max<size_t>(1, columns)))
            throw std::invalid_argument(std::string("BLAS backend ") + name +
                                        " row stride is smaller than its stored row count.");
        if (rowStride > std::numeric_limits<int>::max())
            throw std::invalid_argument(std::string("BLAS backend ") + name +
                                        " leading dimension exceeds int.");
        return {
            conjugate ? CblasConjTrans : CblasTrans,
            static_cast<int>(rowStride),
        };
    }

    throw std::invalid_argument(std::string("BLAS backend ") + name +
                                " must have a unit row or column stride.");
}

template <typename View>
const std::byte* adjustedStorage(const View& view) {
    const uint16_t bits = scalarTypeInfo(view.type()).storageBits;
    if (bits % 8 != 0)
        throw std::invalid_argument("BLAS backend requires byte-addressable scalar storage.");
    const ptrdiff_t logicalOffset = view.layout().offset();
    if (logicalOffset < 0)
        throw std::invalid_argument("BLAS backend received a negative storage offset.");
    return view.rawEncodedBackingStorage().data() + static_cast<size_t>(logicalOffset) * (bits / 8);
}

template <typename T, typename View>
const T* typedData(const View& view, const char* name) {
    const std::byte* data = adjustedStorage(view);
    if (reinterpret_cast<uintptr_t>(data) % alignof(T) != 0)
        throw std::invalid_argument(std::string("BLAS backend ") + name +
                                    " storage is not aligned.");
    return reinterpret_cast<const T*>(data);
}

template <typename T>
T* typedMutableData(const Tensor& view, const char* name) {
    const std::byte* data = adjustedStorage(view);
    if (reinterpret_cast<uintptr_t>(data) % alignof(T) != 0)
        throw std::invalid_argument(std::string("BLAS backend ") + name +
                                    " storage is not aligned.");
    return reinterpret_cast<T*>(const_cast<std::byte*>(data));
}

void validateCommon(const GemmInvocation& problem) {
    detail::validateRuntimeGemm(problem);

    if (problem.a.type() != problem.accumulatorType ||
        problem.b.type() != problem.accumulatorType || problem.d.type() != problem.accumulatorType)
        throw std::invalid_argument(
            "BLAS backend requires A, B, output, and accumulator types to match.");
    if (problem.computeTypeA || problem.computeTypeB)
        throw std::invalid_argument("BLAS backend does not support compute-input quantization.");
    if (!problem.preQuantizationScalesA.empty() || !problem.preQuantizationScalesB.empty())
        throw std::invalid_argument("BLAS backend does not support pre-quantization scaling.");
    if (problem.blockScaleA || problem.blockScaleB)
        throw std::invalid_argument("BLAS backend does not support block scaling.");
    if (problem.mathMode != MathMode::Default)
        throw std::invalid_argument("BLAS backend supports only default operand math.");
    if (!problem.outputSelection.selectsAll())
        throw std::invalid_argument("BLAS backend requires complete output selection.");
    if (problem.d.layout().strides()[0] != 1 ||
        problem.d.layout().strides()[1] <
            static_cast<ptrdiff_t>(std::max<size_t>(1, problem.d.shape()[0])))
        throw std::invalid_argument("BLAS backend requires column-major output storage.");

    const size_t m = problem.a.shape()[0];
    const size_t n = problem.b.shape()[1];
    const size_t k = problem.a.shape()[1];
    if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        problem.d.layout().strides()[1] > std::numeric_limits<int>::max())
        throw std::invalid_argument("BLAS backend dimensions exceed int.");

    (void)toBlasLayout(problem.a, problem.conjugateA, "A");
    (void)toBlasLayout(problem.b, problem.conjugateB, "B");
}

template <typename T>
GemmExecutionInfo runReal(const GemmInvocation& problem) {
    const auto aLayout = toBlasLayout(problem.a, problem.conjugateA, "A");
    const auto bLayout = toBlasLayout(problem.b, problem.conjugateB, "B");
    const int m = static_cast<int>(problem.a.shape()[0]);
    const int n = static_cast<int>(problem.b.shape()[1]);
    const int k = static_cast<int>(problem.a.shape()[1]);
    const int ldc = static_cast<int>(problem.d.layout().strides()[1]);
    const T alpha = T(1);
    const T beta = T(0);
    const T* a = typedData<T>(problem.a, "A");
    const T* b = typedData<T>(problem.b, "B");
    T* d = typedMutableData<T>(problem.d, "D");

    if constexpr (std::is_same_v<T, float>)
        cblas_sgemm(CblasColMajor, aLayout.transpose, bLayout.transpose, m, n, k, alpha, a,
                    aLayout.leadingDimension, b, bLayout.leadingDimension, beta, d, ldc);
    else
        cblas_dgemm(CblasColMajor, aLayout.transpose, bLayout.transpose, m, n, k, alpha, a,
                    aLayout.leadingDimension, b, bLayout.leadingDimension, beta, d, ldc);

    return {
        .backendUsed = GemmBackend::Blas,
        .fallbackReason = std::nullopt,
        .outputElementsWritten = problem.d.shape().elementCount(),
        .outputElementsCovered = problem.d.shape().elementCount(),
    };
}

template <typename T>
GemmExecutionInfo runComplex(const GemmInvocation& problem) {
    const auto aLayout = toBlasLayout(problem.a, problem.conjugateA, "A");
    const auto bLayout = toBlasLayout(problem.b, problem.conjugateB, "B");
    const int m = static_cast<int>(problem.a.shape()[0]);
    const int n = static_cast<int>(problem.b.shape()[1]);
    const int k = static_cast<int>(problem.a.shape()[1]);
    const int ldc = static_cast<int>(problem.d.layout().strides()[1]);
    const T alpha = T(1);
    const T beta = T(0);
    const T* a = typedData<T>(problem.a, "A");
    const T* b = typedData<T>(problem.b, "B");
    T* d = typedMutableData<T>(problem.d, "D");

    if constexpr (std::is_same_v<T, std::complex<float>>)
        cblas_cgemm(CblasColMajor, aLayout.transpose, bLayout.transpose, m, n, k, &alpha, a,
                    aLayout.leadingDimension, b, bLayout.leadingDimension, &beta, d, ldc);
    else
        cblas_zgemm(CblasColMajor, aLayout.transpose, bLayout.transpose, m, n, k, &alpha, a,
                    aLayout.leadingDimension, b, bLayout.leadingDimension, &beta, d, ldc);

    return {
        .backendUsed = GemmBackend::Blas,
        .fallbackReason = std::nullopt,
        .outputElementsWritten = problem.d.shape().elementCount(),
        .outputElementsCovered = problem.d.shape().elementCount(),
    };
}

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
    Tensor output(nativeScalarType<Accumulator>, columnMajorLayout(operand.shape()));
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
    const bool requiresValueTransform = operand.type() != nativeScalarType<Accumulator> ||
                                        computeType || !preQuantizationScales.empty() ||
                                        mathMode != MathMode::Default;
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

    Tensor stagedOutput(nativeScalarType<Accumulator>, columnMajorLayout(problem.d.shape()));
    if (problem.a.shape()[1] != 0) {
        auto [stagedA, conjugateA] = prepareBlasOperand<Accumulator>(
            problem.a, problem.computeTypeA, problem.preQuantizationScalesA, problem.conjugateA,
            problem.mathMode, "A");
        auto [stagedB, conjugateB] = prepareBlasOperand<Accumulator>(
            problem.b, problem.computeTypeB, problem.preQuantizationScalesB, problem.conjugateB,
            problem.mathMode, "B");
        MatmulOptions stagedOptions(nativeScalarType<Accumulator>);
        stagedOptions.conjugateA = conjugateA;
        stagedOptions.conjugateB = conjugateB;
        GemmInvocation stagedProblem(
            std::move(stagedA), std::move(stagedB), stagedOutput, stagedOptions);

        blas.run(stagedProblem);
    }

    const RuntimeMatrixReader<Accumulator> stagedOutputReader(stagedOutput);
    const RuntimeMatrixWriter<Accumulator> output(problem.d);
    const size_t rows = problem.d.shape()[0];
    const size_t outputElementCount = problem.d.shape().elementCount();
    detail::forEachParallelIndex(
        outputElementCount, outputElementCount, detail::canParallelizeGemmOutput(problem), 500'000,
        [&](size_t linearIndex) {
            const size_t column = linearIndex / rows;
            const size_t row = linearIndex % rows;
            output.store(row, column, stagedOutputReader(row, column));
        });

    return {
        .backendUsed = GemmBackend::Blas,
        .fallbackReason = std::nullopt,
        .outputElementsWritten = problem.d.shape().elementCount(),
        .outputElementsCovered = problem.d.shape().elementCount(),
    };
}

GemmSupportInfo BlasGemmBackend::querySupport(const GemmInvocation& problem) const {
    try {
        validateCommon(problem);
        switch (problem.accumulatorType) {
            case ScalarType::Float32:
                (void)typedData<float>(problem.a, "A");
                (void)typedData<float>(problem.b, "B");
                (void)typedMutableData<float>(problem.d, "D");
                break;
            case ScalarType::Float64:
                (void)typedData<double>(problem.a, "A");
                (void)typedData<double>(problem.b, "B");
                (void)typedMutableData<double>(problem.d, "D");
                break;
            case ScalarType::ComplexFloat32:
                (void)typedData<std::complex<float>>(problem.a, "A");
                (void)typedData<std::complex<float>>(problem.b, "B");
                (void)typedMutableData<std::complex<float>>(problem.d, "D");
                break;
            case ScalarType::ComplexFloat64:
                (void)typedData<std::complex<double>>(problem.a, "A");
                (void)typedData<std::complex<double>>(problem.b, "B");
                (void)typedMutableData<std::complex<double>>(problem.d, "D");
                break;
            default:
                throw std::invalid_argument("BLAS backend accumulator type is unsupported.");
        }
        return {.supported = true, .reason = {}};
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }
}

GemmExecutionInfo BlasGemmBackend::run(const GemmInvocation& problem) const {
    const GemmSupportInfo support = querySupport(problem);
    if (!support) throw std::invalid_argument(support.reason);

    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            return runReal<float>(problem);
        case ScalarType::Float64:
            return runReal<double>(problem);
        case ScalarType::ComplexFloat32:
            return runComplex<std::complex<float>>(problem);
        case ScalarType::ComplexFloat64:
            return runComplex<std::complex<double>>(problem);
        default:
            throw std::invalid_argument("BLAS backend accumulator type is unsupported.");
    }
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

void matmulIntoWithBlasBackend(const Tensor& a, const Tensor& b, Tensor output,
                               const MatmulOptions& options, OutputSelection selection) {
    (void)detail::executeBlasGemm(
        GemmInvocation(a, b, std::move(output), options, std::move(selection)),
        GemmBackend::Automatic);
}

Tensor matmulWithBlasBackend(const Tensor& a, const Tensor& b, ScalarType outputType,
                             const MatmulOptions& options, std::optional<Layout> outputLayout) {
    detail::requireRank(a.shape(), 2, "matmul", "A");
    detail::requireRank(b.shape(), 2, "matmul", "B");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul reduction dimension mismatch.");
    const Shape outputShape{a.shape()[0], b.shape()[1]};
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (layout.shape() != outputShape)
        throw std::invalid_argument("matmul output layout shape mismatch.");
    Tensor output(outputType, layout);
    matmulIntoWithBlasBackend(a, b, output, options);
    return output;
}

GemmSupportInfo detail::queryBlasGemmSupport(const GemmInvocation& problem, GemmBackend backend) {
    if (backend == GemmBackend::Blas) return queryTransformingBlasGemmSupport(problem);
    return detail::queryGemmSupport(problem, backend);
}

detail::GemmExecutionInfo detail::executeBlasGemm(const GemmInvocation& problem,
                                                  GemmBackend backend) {
    if (backend == GemmBackend::Blas) {
        const GemmSupportInfo support = queryTransformingBlasGemmSupport(problem);
        if (!support) throw std::invalid_argument(support.reason);
        return runTransformingBlasGemm(problem);
    }
    if (backend != GemmBackend::Automatic) return detail::executeGemm(problem, backend);

    const GemmSupportInfo blasSupport = queryTransformingBlasGemmSupport(problem);
    if (blasSupport && blasSupport.preferredForAutomaticExecution)
        return runTransformingBlasGemm(problem);

    GemmExecutionInfo runInfo = detail::executeGemm(problem, GemmBackend::Automatic);
    if (!blasSupport) runInfo.fallbackReason = blasSupport.reason;
    return runInfo;
}

}  // namespace roc::host_numerics
