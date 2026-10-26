// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cblas.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/hostnumerics/backends/blas.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/reference_gemm.hpp"
#include "detail/threading.hpp"

namespace roc::hostnumerics {
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

detail::Expected<BlasMatrixLayout, std::string> checkBlasLayout(const Tensor& view, bool conjugate,
                                                                const char* name) {
    const ptrdiff_t rowStride = view.layout().strides()[0];
    const ptrdiff_t columnStride = view.layout().strides()[1];
    const size_t rows = view.shape()[0];
    const size_t columns = view.shape()[1];

    if (rowStride == 1) {
        if (conjugate) {
            return detail::Unexpected(std::string("BLAS backend cannot conjugate non-transposed ") +
                                      name + ".");
        }
        if (columnStride < static_cast<ptrdiff_t>(std::max<size_t>(1, rows))) {
            return detail::Unexpected(std::string("BLAS backend ") + name +
                                      " column stride is smaller than its stored row count.");
        }
        if (columnStride > std::numeric_limits<int>::max()) {
            return detail::Unexpected(std::string("BLAS backend ") + name +
                                      " leading dimension exceeds int.");
        }
        return BlasMatrixLayout{CblasNoTrans, static_cast<int>(columnStride)};
    }

    if (columnStride == 1) {
        if (rowStride < static_cast<ptrdiff_t>(std::max<size_t>(1, columns))) {
            return detail::Unexpected(std::string("BLAS backend ") + name +
                                      " row stride is smaller than its stored row count.");
        }
        if (rowStride > std::numeric_limits<int>::max()) {
            return detail::Unexpected(std::string("BLAS backend ") + name +
                                      " leading dimension exceeds int.");
        }
        return BlasMatrixLayout{
            conjugate ? CblasConjTrans : CblasTrans,
            static_cast<int>(rowStride),
        };
    }

    return detail::Unexpected(std::string("BLAS backend ") + name +
                              " must have a unit row or column stride.");
}

BlasMatrixLayout toBlasLayout(const Tensor& view, bool conjugate, const char* name) {
    const auto result = checkBlasLayout(view, conjugate, name);
    if (!result) throw std::runtime_error(result.error());
    return *result;
}

template <typename View>
const std::byte* adjustedStorage(const View& view) {
    // Called only after checkTypedData has established byte-addressability and offset.
    const auto* base = view.rawEncodedBackingStorage().data();
    const size_t offset =
        static_cast<size_t>(view.layout().offset()) * (scalarTypeInfo(view.type()).storageBits / 8);
    return offset == 0 ? base : base + offset;
}

template <typename T, typename View>
GemmSupportInfo checkTypedData(const View& view, const char* name) {
    if (scalarTypeInfo(view.type()).storageBits % 8 != 0)
        return {.reason = "BLAS backend requires byte-addressable scalar storage."};
    if (view.layout().offset() < 0)
        return {.reason = "BLAS backend received a negative storage offset."};
    const auto storage = view.rawEncodedBackingStorage();
    const size_t elementBytes = scalarTypeInfo(view.type()).storageBits / 8;
    const size_t offset = static_cast<size_t>(view.layout().offset());
    if (offset > storage.size() / elementBytes)
        return {.reason = "BLAS backend storage offset exceeds its backing storage."};
    const uintptr_t address = reinterpret_cast<uintptr_t>(storage.data()) + offset * elementBytes;
    if (address % alignof(T) != 0)
        return {.reason = std::string("BLAS backend ") + name + " storage is not aligned."};
    return {.supported = true};
}

template <typename T, typename View>
const T* typedData(const View& view, const char* name) {
    const auto result = checkTypedData<T>(view, name);
    if (!result) throw std::runtime_error(result.reason);
    return reinterpret_cast<const T*>(adjustedStorage(view));
}

template <typename T>
T* typedMutableData(const Tensor& view, const char* name) {
    return const_cast<T*>(typedData<T>(view, name));
}
GemmSupportInfo checkCommon(const GemmInvocation& problem) {
    if (auto result = detail::checkRuntimeGemm(problem); !result) return result;

    if (problem.a.type() != problem.accumulatorType ||
        problem.b.type() != problem.accumulatorType || problem.d.type() != problem.accumulatorType)
        return {.reason = "BLAS backend requires A, B, output, and accumulator types to match."};
    if (detail::requiresInputQuantization(problem.a.type(), problem.computeTypeA,
                                          problem.accumulatorType) ||
        detail::requiresInputQuantization(problem.b.type(), problem.computeTypeB,
                                          problem.accumulatorType))
        return {.reason = "BLAS backend does not support compute-input quantization."};
    if (!problem.preQuantizationScalesA.empty() || !problem.preQuantizationScalesB.empty())
        return {.reason = "BLAS backend does not support pre-quantization scaling."};
    if (problem.blockScaleA || problem.blockScaleB)
        return {.reason = "BLAS backend does not support block scaling."};
    if (problem.mathMode != MathMode::Default)
        return {.reason = "BLAS backend supports only default operand math."};
    if (!(problem.indicesCount == problem.d.elementCount()))
        return {.reason = "BLAS backend requires complete output selection."};
    const size_t m = problem.a.shape()[0];
    const size_t n = problem.b.shape()[1];
    const size_t k = problem.a.shape()[1];
    if (m > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        problem.d.layout().strides()[1] > std::numeric_limits<int>::max())
        return {.reason = "BLAS backend dimensions exceed int."};

    if (problem.d.layout().strides()[0] != 1 ||
        problem.d.layout().strides()[1] <
            static_cast<ptrdiff_t>(std::max<size_t>(1, problem.d.shape()[0])))
        return {.reason = "BLAS backend requires column-major output storage."};

    if (auto result = checkBlasLayout(problem.a, problem.conjugateA, "A"); !result)
        return {.reason = std::move(result).error()};
    if (auto result = checkBlasLayout(problem.b, problem.conjugateB, "B"); !result)
        return {.reason = std::move(result).error()};
    return {.supported = true};
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

GemmSupportInfo BlasGemmBackend::querySupport(const GemmInvocation& problem) const {
    if (auto result = checkCommon(problem); !result) return result;
    auto checkStorage = [&]<typename T>() -> GemmSupportInfo {
        if (auto result = checkTypedData<T>(problem.a, "A"); !result) return result;
        if (auto result = checkTypedData<T>(problem.b, "B"); !result) return result;
        return checkTypedData<T>(problem.d, "D");
    };
    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            return checkStorage.template operator()<float>();
        case ScalarType::Float64:
            return checkStorage.template operator()<double>();
        case ScalarType::ComplexFloat32:
            return checkStorage.template operator()<std::complex<float>>();
        case ScalarType::ComplexFloat64:
            return checkStorage.template operator()<std::complex<double>>();
        default:
            return {.reason = "BLAS backend accumulator type is unsupported."};
    }
}

GemmExecutionInfo BlasGemmBackend::run(const GemmInvocation& problem) const {
    const GemmSupportInfo support = querySupport(problem);
    if (!support) throw std::runtime_error(support.reason);

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
            throw std::runtime_error("BLAS backend accumulator type is unsupported.");
    }
}
}  // namespace
}  // namespace roc::hostnumerics
