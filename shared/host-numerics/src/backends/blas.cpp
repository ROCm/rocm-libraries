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

#include "detail/reference_gemm.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics {
namespace {
using detail::GemmOperand;

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
    const GemmSupportInfo pointwise = detail::queryGemmSupport(problem, GemmBackend::Pointwise);
    if (!pointwise) throw std::invalid_argument(pointwise.reason);

    if (problem.a.values.type() != problem.accumulatorType ||
        problem.b.values.type() != problem.accumulatorType ||
        problem.c.type() != problem.accumulatorType || problem.d.type() != problem.accumulatorType)
        throw std::invalid_argument(
            "BLAS backend requires A, B, C, D, and accumulator types to match.");
    if (problem.a.computeType || problem.b.computeType)
        throw std::invalid_argument("BLAS backend does not support compute-input quantization.");
    if (!problem.a.preQuantizationScales.empty() || !problem.b.preQuantizationScales.empty())
        throw std::invalid_argument("BLAS backend does not support pre-quantization scaling.");
    if (problem.a.blockScale || problem.b.blockScale)
        throw std::invalid_argument("BLAS backend does not support block scaling.");
    if (problem.mathMode != MathMode::Default)
        throw std::invalid_argument("BLAS backend supports only default operand math.");
    if (problem.epilogue.bias || problem.epilogue.scaleAlpha || problem.epilogue.scaleA ||
        problem.epilogue.scaleB || problem.epilogue.activation != Activation::None ||
        detail::runtimeScalar<std::complex<double>>(problem.epilogue.scaleC, "C scale") !=
            std::complex<double>(1.0, 0.0) ||
        detail::runtimeScalar<std::complex<double>>(problem.epilogue.outputScale, "output scale") !=
            std::complex<double>(1.0, 0.0) ||
        problem.epilogue.outputConversion != OutputConversion::Default)
        throw std::invalid_argument("BLAS backend does not support a fused epilogue.");
    if (!problem.outputSelection.selectsAll())
        throw std::invalid_argument("BLAS backend requires complete output selection.");
    if (problem.c.layout() != problem.d.layout() ||
        adjustedStorage(problem.c) != adjustedStorage(problem.d))
        throw std::invalid_argument("BLAS backend currently requires C and D to alias.");
    if (problem.d.layout().strides()[0] != 1 ||
        problem.d.layout().strides()[1] <
            static_cast<ptrdiff_t>(std::max<size_t>(1, problem.d.shape()[0])))
        throw std::invalid_argument("BLAS backend requires column-major C/D storage.");

    const size_t m = problem.a.values.shape()[0];
    const size_t n = problem.b.values.shape()[1];
    const size_t k = problem.a.values.shape()[1];
    if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        problem.d.layout().strides()[1] > std::numeric_limits<int>::max())
        throw std::invalid_argument("BLAS backend dimensions exceed int.");

    (void)toBlasLayout(problem.a.values, problem.a.conjugate, "A");
    (void)toBlasLayout(problem.b.values, problem.b.conjugate, "B");
}

template <typename T>
GemmExecutionInfo runReal(const GemmInvocation& problem) {
    const auto aLayout = toBlasLayout(problem.a.values, problem.a.conjugate, "A");
    const auto bLayout = toBlasLayout(problem.b.values, problem.b.conjugate, "B");
    const int m = static_cast<int>(problem.a.values.shape()[0]);
    const int n = static_cast<int>(problem.b.values.shape()[1]);
    const int k = static_cast<int>(problem.a.values.shape()[1]);
    const int ldc = static_cast<int>(problem.d.layout().strides()[1]);
    const T alpha = detail::runtimeScalar<T>(problem.epilogue.alpha, "alpha");
    const T beta = detail::runtimeScalar<T>(problem.epilogue.beta, "beta");
    const T* a = typedData<T>(problem.a.values, "A");
    const T* b = typedData<T>(problem.b.values, "B");
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
    const auto aLayout = toBlasLayout(problem.a.values, problem.a.conjugate, "A");
    const auto bLayout = toBlasLayout(problem.b.values, problem.b.conjugate, "B");
    const int m = static_cast<int>(problem.a.values.shape()[0]);
    const int n = static_cast<int>(problem.b.values.shape()[1]);
    const int k = static_cast<int>(problem.a.values.shape()[1]);
    const int ldc = static_cast<int>(problem.d.layout().strides()[1]);
    const T alpha = detail::runtimeScalar<T>(problem.epilogue.alpha, "alpha");
    const T beta = detail::runtimeScalar<T>(problem.epilogue.beta, "beta");
    const T* a = typedData<T>(problem.a.values, "A");
    const T* b = typedData<T>(problem.b.values, "B");
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
    const GemmSupportInfo pointwise = detail::queryGemmSupport(problem, GemmBackend::Pointwise);
    if (!pointwise) throw std::invalid_argument(pointwise.reason);

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
    const auto hasNonScalarScale = [](const std::optional<Tensor>& scale) {
        return scale && scale->elementCount() != 1;
    };
    if (problem.epilogue.bias || problem.epilogue.scaleAlpha ||
        hasNonScalarScale(problem.epilogue.scaleA) || hasNonScalarScale(problem.epilogue.scaleB) ||
        problem.epilogue.activation != Activation::None)
        throw std::invalid_argument(
            "Transforming BLAS backend supports operand transforms, scalar A/B scales, and "
            "output conversion, but not the general GEMM epilogue.");
    if (problem.a.blockScale || problem.b.blockScale)
        throw std::invalid_argument(
            "Transforming BLAS backend cannot preserve block-scale reduction boundaries.");
    if (!problem.outputSelection.selectsAll())
        throw std::invalid_argument(
            "Transforming BLAS backend requires complete output selection.");

    const size_t m = problem.a.values.shape()[0];
    const size_t n = problem.b.values.shape()[1];
    const size_t k = problem.a.values.shape()[1];
    if (m == 0 || n == 0 || k == 0)
        throw std::invalid_argument("Transforming BLAS backend requires nonzero M, N, and K.");
    if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Transforming BLAS backend dimensions exceed int.");
}

template <typename Accumulator>
Tensor materializeOperand(const GemmOperand& operand, MathMode mathMode) {
    using namespace detail;
    Tensor output(nativeScalarType<Accumulator>, columnMajorLayout(operand.values.shape()));
    const RuntimeMatrixReader<Accumulator> input(operand.values);
    const RuntimeMatrixWriter<Accumulator> writer(output);
    const RuntimeQuantizer<Accumulator> quantize(operand.computeType);
    const RuntimeMathFunction<Accumulator> operandMath = runtimeMathFunction<Accumulator>(mathMode);
    std::vector<RuntimeMatrixReader<Accumulator>> scaleReaders;
    scaleReaders.reserve(operand.preQuantizationScales.size());
    for (const Tensor& scale : operand.preQuantizationScales)
        scaleReaders.emplace_back(scale.broadcastTo(operand.values.shape()));
    const size_t rows = operand.values.shape()[0];
    const size_t columns = operand.values.shape()[1];
    const size_t elementCount = detail::saturatedProduct(rows, columns);
    detail::forEachParallelIndex(
        elementCount, elementCount, true, 500'000, [&](size_t linearIndex) {
            const size_t column = linearIndex / rows;
            const size_t row = linearIndex % rows;
            Accumulator value = conjugateIfNeeded(input(row, column), operand.conjugate);
            for (const auto& scale : scaleReaders) value *= scale(row, column);
            value = operandMath(quantize(value));
            writer.store(row, column, value);
        });
    return output;
}

template <typename Accumulator>
bool canUseBlasOperandWithoutMaterialization(const GemmOperand& operand, MathMode mathMode,
                                             const char* name) {
    const bool requiresValueTransform =
        operand.values.type() != nativeScalarType<Accumulator> || operand.computeType ||
        !operand.preQuantizationScales.empty() || mathMode != MathMode::Default;
    if (requiresValueTransform) return false;
    try {
        (void)toBlasLayout(operand.values, operand.conjugate, name);
        (void)typedData<Accumulator>(operand.values, name);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

bool canUseBlasOperandWithoutMaterialization(const GemmOperand& operand, ScalarType accumulatorType,
                                             MathMode mathMode, const char* name) {
    switch (accumulatorType) {
        case ScalarType::Float32:
            return canUseBlasOperandWithoutMaterialization<float>(operand, mathMode, name);
        case ScalarType::Float64:
            return canUseBlasOperandWithoutMaterialization<double>(operand, mathMode, name);
        case ScalarType::ComplexFloat32:
            return canUseBlasOperandWithoutMaterialization<std::complex<float>>(operand, mathMode,
                                                                                name);
        case ScalarType::ComplexFloat64:
            return canUseBlasOperandWithoutMaterialization<std::complex<double>>(operand, mathMode,
                                                                                 name);
        default:
            return false;
    }
}

template <typename Accumulator>
GemmOperand prepareBlasOperand(const GemmOperand& operand, MathMode mathMode, const char* name) {
    if (canUseBlasOperandWithoutMaterialization<Accumulator>(operand, mathMode, name)) {
        GemmOperand direct(operand.values);
        direct.conjugate = operand.conjugate;
        return direct;
    }
    return GemmOperand(materializeOperand<Accumulator>(operand, mathMode));
}

template <typename Accumulator>
GemmExecutionInfo runTransforming(const GemmInvocation& problem) {
    using namespace detail;
    static const BlasGemmBackend blas;
    if (blas.querySupport(problem)) return blas.run(problem);

    Tensor stagedOutput(nativeScalarType<Accumulator>, columnMajorLayout(problem.d.shape()));
    const RuntimeGemmFinalizer<Accumulator> finalizer(problem);

    if (!finalizer.alphaIsZero()) {
        GemmOperand stagedA = prepareBlasOperand<Accumulator>(problem.a, problem.mathMode, "A");
        GemmOperand stagedB = prepareBlasOperand<Accumulator>(problem.b, problem.mathMode, "B");
        GemmInvocation stagedProblem(std::move(stagedA), std::move(stagedB), stagedOutput,
                                     stagedOutput, nativeScalarType<Accumulator>);

        blas.run(stagedProblem);
    }

    const RuntimeMatrixReader<Accumulator> stagedOutputReader(stagedOutput);
    const RuntimeMatrixOutputWriter<Accumulator> output(problem.d,
                                                        problem.epilogue.outputConversion);
    const size_t rows = problem.d.shape()[0];
    const size_t outputElementCount = problem.d.shape().elementCount();
    detail::forEachParallelIndex(
        outputElementCount, outputElementCount, detail::canParallelizeGemmOutput(problem), 500'000,
        [&](size_t linearIndex) {
            const size_t column = linearIndex / rows;
            const size_t row = linearIndex % rows;
            output.store(row, column,
                         finalizer.finalize(row, column, stagedOutputReader(row, column)));
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
                (void)typedData<float>(problem.a.values, "A");
                (void)typedData<float>(problem.b.values, "B");
                (void)typedMutableData<float>(problem.d, "D");
                break;
            case ScalarType::Float64:
                (void)typedData<double>(problem.a.values, "A");
                (void)typedData<double>(problem.b.values, "B");
                (void)typedMutableData<double>(problem.d, "D");
                break;
            case ScalarType::ComplexFloat32:
                (void)typedData<std::complex<float>>(problem.a.values, "A");
                (void)typedData<std::complex<float>>(problem.b.values, "B");
                (void)typedMutableData<std::complex<float>>(problem.d, "D");
                break;
            case ScalarType::ComplexFloat64:
                (void)typedData<std::complex<double>>(problem.a.values, "A");
                (void)typedData<std::complex<double>>(problem.b.values, "B");
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
        static const BlasGemmBackend directBlas;
        if (directBlas.querySupport(problem)) return {.supported = true, .reason = {}};

        const size_t rows = problem.a.values.shape()[0];
        const size_t columns = problem.b.values.shape()[1];
        const size_t reductions = problem.a.values.shape()[1];
        const size_t multiplyAdds =
            detail::saturatedProduct(detail::saturatedProduct(rows, columns), reductions);
        const size_t stagedAElements =
            canUseBlasOperandWithoutMaterialization(problem.a, problem.accumulatorType,
                                                    problem.mathMode, "A")
                ? 0
                : detail::saturatedProduct(rows, reductions);
        const size_t stagedBElements =
            canUseBlasOperandWithoutMaterialization(problem.b, problem.accumulatorType,
                                                    problem.mathMode, "B")
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

GemmSupportInfo queryGemmSupportWithBlasBackend(const Tensor& a, const Tensor& b, const Tensor& c,
                                                const Tensor& d, const GemmOptions& options,
                                                GemmBackend backend) {
    return detail::queryBlasGemmSupport(GemmInvocation(a, b, c, d, options), backend);
}

GemmBackend referenceGemmIntoWithBlasBackend(Tensor a, Tensor b, Tensor c, Tensor d,
                                             const GemmOptions& options, GemmBackend backend) {
    return detail::executeBlasGemm(
               GemmInvocation(std::move(a), std::move(b), std::move(c), std::move(d), options),
               backend)
        .backendUsed;
}

Tensor referenceGemmWithBlasBackend(Tensor a, Tensor b, Tensor c, ScalarType outputType,
                                    const GemmOptions& options, std::optional<Layout> outputLayout,
                                    GemmBackend backend) {
    const GemmSpecification problem(std::move(a), std::move(b), std::move(c), outputType, options);
    const Shape outputShape{problem.a.values.shape()[0], problem.b.values.shape()[1]};
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    Tensor destination(outputType, layout);
    (void)detail::executeBlasGemm(GemmInvocation(problem, destination, options.outputSelection),
                                  backend);
    return destination;
}
}  // namespace roc::host_numerics
