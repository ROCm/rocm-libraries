// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cblas.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_validation/backends/blas.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace roc::host_validation {
namespace {
struct BlasMatrixLayout {
    CBLAS_TRANSPOSE transpose;
    int leadingDimension;
};

BlasMatrixLayout toBlasLayout(const TensorView& view, bool conjugate, const char* name) {
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
    return view.storage().data() + static_cast<size_t>(logicalOffset) * (bits / 8);
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
T* typedMutableData(const MutableTensorView& view, const char* name) {
    const std::byte* data = adjustedStorage(view);
    if (reinterpret_cast<uintptr_t>(data) % alignof(T) != 0)
        throw std::invalid_argument(std::string("BLAS backend ") + name +
                                    " storage is not aligned.");
    return reinterpret_cast<T*>(const_cast<std::byte*>(data));
}

void validateCommon(const GemmProblem& problem) {
    const GemmSupportInfo canonical = queryGemmSupport(problem, GemmBackend::Canonical);
    if (!canonical) throw std::invalid_argument(canonical.reason);

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
        problem.epilogue.outputScale != std::complex<double>(1.0, 0.0) ||
        problem.epilogue.outputConversion != GemmOutputConversion::Default)
        throw std::invalid_argument("BLAS backend does not support a fused epilogue.");
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
GemmRunInfo runReal(const GemmProblem& problem) {
    const auto aLayout = toBlasLayout(problem.a.values, problem.a.conjugate, "A");
    const auto bLayout = toBlasLayout(problem.b.values, problem.b.conjugate, "B");
    const int m = static_cast<int>(problem.a.values.shape()[0]);
    const int n = static_cast<int>(problem.b.values.shape()[1]);
    const int k = static_cast<int>(problem.a.values.shape()[1]);
    const int ldc = static_cast<int>(problem.d.layout().strides()[1]);
    const T alpha = static_cast<T>(problem.epilogue.alpha.real());
    const T beta = static_cast<T>(problem.epilogue.beta.real());
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
        .outputElementsComputed = problem.d.shape().elementCount(),
    };
}

template <typename T>
GemmRunInfo runComplex(const GemmProblem& problem) {
    const auto aLayout = toBlasLayout(problem.a.values, problem.a.conjugate, "A");
    const auto bLayout = toBlasLayout(problem.b.values, problem.b.conjugate, "B");
    const int m = static_cast<int>(problem.a.values.shape()[0]);
    const int n = static_cast<int>(problem.b.values.shape()[1]);
    const int k = static_cast<int>(problem.a.values.shape()[1]);
    const int ldc = static_cast<int>(problem.d.layout().strides()[1]);
    const T alpha(static_cast<typename T::value_type>(problem.epilogue.alpha.real()),
                  static_cast<typename T::value_type>(problem.epilogue.alpha.imag()));
    const T beta(static_cast<typename T::value_type>(problem.epilogue.beta.real()),
                 static_cast<typename T::value_type>(problem.epilogue.beta.imag()));
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
        .outputElementsComputed = problem.d.shape().elementCount(),
    };
}

Layout columnMajorLayout(const Shape& shape) {
    return Layout(shape, {1, static_cast<ptrdiff_t>(shape[0])});
}

void validateTransforming(const GemmProblem& problem) {
    const GemmSupportInfo canonical = queryGemmSupport(problem, GemmBackend::Canonical);
    if (!canonical) throw std::invalid_argument(canonical.reason);

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
    if (problem.a.blockScale || problem.b.blockScale)
        throw std::invalid_argument("Transforming BLAS backend does not support block scaling.");
    if (problem.epilogue.bias || problem.epilogue.scaleAlpha || problem.epilogue.scaleA ||
        problem.epilogue.scaleB || problem.epilogue.activation != Activation::None)
        throw std::invalid_argument(
            "Transforming BLAS backend supports operand transforms and output conversion, "
            "but not the general GEMM epilogue.");
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
    const RuntimeMatrixWriter<Accumulator> writer(output.mutableView());
    const RuntimeQuantizer<Accumulator> quantize(operand.computeType);
    const RuntimeMathFunction<Accumulator> operandMath = runtimeMathFunction<Accumulator>(mathMode);
    std::vector<RuntimeVectorReader<Accumulator>> scaleReaders;
    scaleReaders.reserve(operand.preQuantizationScales.size());
    for (const VectorBinding& binding : operand.preQuantizationScales)
        scaleReaders.emplace_back(binding.values);

    const size_t rows = operand.values.shape()[0];
    const size_t columns = operand.values.shape()[1];
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            Accumulator value = conjugateIfNeeded(input(row, column), operand.conjugate);
            for (size_t scaleIndex = 0; scaleIndex < scaleReaders.size(); ++scaleIndex) {
                const VectorBinding& binding = operand.preQuantizationScales[scaleIndex];
                const size_t index = binding.values.shape()[0] == 1
                                         ? 0
                                         : (binding.axis == MatrixAxis::Row ? row : column);
                value *= scaleReaders[scaleIndex][index];
            }
            writer.store(row, column, operandMath(quantize(value)));
        }
    }
    return output;
}

template <typename Accumulator>
Tensor materializeMatrix(TensorView input) {
    using namespace detail;
    Tensor output(nativeScalarType<Accumulator>, columnMajorLayout(input.shape()));
    const RuntimeMatrixReader<Accumulator> reader(input);
    const RuntimeMatrixWriter<Accumulator> writer(output.mutableView());
    for (size_t row = 0; row < input.shape()[0]; ++row)
        for (size_t column = 0; column < input.shape()[1]; ++column)
            writer.store(row, column, reader(row, column));
    return output;
}

template <typename Accumulator>
GemmRunInfo runTransforming(const GemmProblem& problem) {
    using namespace detail;
    Tensor stagedA = materializeOperand<Accumulator>(problem.a, problem.mathMode);
    Tensor stagedB = materializeOperand<Accumulator>(problem.b, problem.mathMode);
    Tensor stagedC = materializeMatrix<Accumulator>(problem.c);

    GemmProblem stagedProblem(GemmOperand(stagedA.view()), GemmOperand(stagedB.view()),
                              stagedC.view(), stagedC.mutableView(), nativeScalarType<Accumulator>);
    stagedProblem.epilogue.alpha = problem.epilogue.alpha;
    stagedProblem.epilogue.beta = problem.epilogue.beta;

    static const BlasGemmBackend blas;
    referenceGemm(stagedProblem, {
                                     .backend = GemmBackend::Blas,
                                     .requireRequestedBackend = true,
                                     .backendImplementation = &blas,
                                 });

    const RuntimeMatrixReader<Accumulator> stagedOutput(stagedC.view());
    const RuntimeGemmOutputWriter<Accumulator> output(problem.d, problem.epilogue.outputConversion);
    const Accumulator outputScale =
        runtimeScalar<Accumulator>(problem.epilogue.outputScale, "output scale");
    for (size_t row = 0; row < problem.d.shape()[0]; ++row)
        for (size_t column = 0; column < problem.d.shape()[1]; ++column)
            output.store(row, column, stagedOutput(row, column) * outputScale);

    return {
        .backendUsed = GemmBackend::Blas,
        .fallbackReason = std::nullopt,
        .outputElementsComputed = problem.d.shape().elementCount(),
    };
}
}  // namespace

GemmBackend BlasGemmBackend::backend() const {
    return GemmBackend::Blas;
}

GemmSupportInfo BlasGemmBackend::querySupport(const GemmProblem& problem) const {
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

GemmRunInfo BlasGemmBackend::run(const GemmProblem& problem) const {
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

GemmBackend TransformingBlasGemmBackend::backend() const {
    return GemmBackend::Blas;
}

GemmSupportInfo TransformingBlasGemmBackend::querySupport(const GemmProblem& problem) const {
    try {
        validateTransforming(problem);
        return {.supported = true, .reason = {}};
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }
}

GemmRunInfo TransformingBlasGemmBackend::run(const GemmProblem& problem) const {
    const GemmSupportInfo support = querySupport(problem);
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
}  // namespace roc::host_validation
