/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

// Product-private hipBLASLt adapter.

// hipBLASLt adapter over host-validation-owned initialization.

#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_ostream.hpp"
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/host_validation/HipblasltDataInitialization.hpp>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

enum class ABC_dims
{
    A,
    B,
    C
};

void set_host_side_fill_kernel_state(bool enable);
bool host_side_fill_kernel();

// When enabled, the hpl and trig_float device initializers emit positive-only
// values. Used for ULP validation: signed zero-mean inputs make the reference
// dot products cancel toward zero, where the magnitude-independent accumulation
// noise floor inflates the per-element ULP error even for a correct kernel.
void set_ulp_positive_init_state(bool enable);
bool ulp_positive_init();

void hipblaslt_init_device(ABC_dims                 ABC_dims,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           void*                    A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           hipDataType              type,
                           size_t                   stride,
                           size_t                   batch_count,
                           int                      norm_dist_one_special_type = -1);

namespace hipblaslt::host_validation::detail
{
    enum class RuntimeInitialization : uint8_t
    {
        General      = 1U << 0,
        Random       = 1U << 1,
        Small        = 1U << 2,
        LowPrecision = 1U << 3,
    };

    inline constexpr uint8_t runtimeInitializationCapabilities(ScalarType type)
    {
        constexpr uint8_t general      = static_cast<uint8_t>(RuntimeInitialization::General);
        constexpr uint8_t random       = static_cast<uint8_t>(RuntimeInitialization::Random);
        constexpr uint8_t small        = static_cast<uint8_t>(RuntimeInitialization::Small);
        constexpr uint8_t lowPrecision = static_cast<uint8_t>(RuntimeInitialization::LowPrecision);

        switch(type)
        {
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::Float16:
        case ScalarType::Int32:
            return general | random | small | lowPrecision;
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            return general | random | small;
        case ScalarType::BFloat16:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::Int8:
            return general | random | lowPrecision;
        case ScalarType::E8M0:
            return random;
        default:
            return 0;
        }
    }

    inline constexpr bool supportsRuntimeInitialization(ScalarType            type,
                                                        RuntimeInitialization required)
    {
        return (runtimeInitializationCapabilities(type) & static_cast<uint8_t>(required)) != 0;
    }

    inline void reportUnsupportedRuntimeInitialization(std::string_view functionName,
                                                       const std::optional<ScalarType>& type,
                                                       bool identifyPackedType)
    {
        if(identifyPackedType && type)
        {
            switch(*type)
            {
            case ScalarType::Float6E2M3:
                hipblaslt_cerr << functionName << " not supports FP6" << std::endl;
                return;
            case ScalarType::Float6E3M2:
                hipblaslt_cerr << functionName << " not supports BF6" << std::endl;
                return;
            case ScalarType::Float4E2M1:
                hipblaslt_cerr << functionName << " not supports FP4" << std::endl;
                return;
            default:
                break;
            }
        }
        hipblaslt_cerr << "Error type in " << functionName << std::endl;
    }

    template <typename OptionsFactory>
    inline void initializeRuntimeTensor(void*                 data,
                                        hipDataType           runtimeType,
                                        Layout                layout,
                                        RuntimeInitialization required,
                                        std::string_view      functionName,
                                        bool                  identifyPackedType,
                                        OptionsFactory&&      optionsFactory)
    {
        const std::optional<ScalarType> type = tryScalarType(runtimeType);
        if(!type || !supportsRuntimeInitialization(*type, required))
        {
            reportUnsupportedRuntimeInitialization(functionName, type, identifyPackedType);
            return;
        }

        GenerationOptions options = std::forward<OptionsFactory>(optionsFactory)(*type);
        initializeTensor(data, *type, std::move(layout), options);
    }

    inline Layout matrixBatchLayout(
        size_t rows, size_t columns, size_t leadingDimension, size_t batchStride, size_t batchCount)
    {
        return Layout(
            Shape{rows, columns, batchCount},
            {1, static_cast<ptrdiff_t>(leadingDimension), static_cast<ptrdiff_t>(batchStride)});
    }

    inline Layout contiguousRangeLayout(size_t startOffset, size_t endOffset)
    {
        return Layout(Shape{endOffset - startOffset}, {1}, static_cast<ptrdiff_t>(startOffset));
    }
} // namespace hipblaslt::host_validation::detail

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize matrices with random values
template <typename T>
inline void
    hipblaslt_init(T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::legacyRandomOptions(
        hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

// Initialize matrices with random values
template <typename T>
inline void hipblaslt_init_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::randomIntegerOptions(
        hipblaslt::host_validation::scalarType<T>(), true, false, false);
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

// Initialize matrices with random values
inline void hipblaslt_init(void*       A,
                           size_t      M,
                           size_t      N,
                           size_t      lda,
                           hipDataType type,
                           size_t      stride      = 0,
                           size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::Random,
        "hipblaslt_init",
        true,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::legacyRandomOptions(scalar);
        });
}

inline void hipblaslt_init_small(void*       A,
                                 size_t      M,
                                 size_t      N,
                                 size_t      lda,
                                 hipDataType type,
                                 size_t      stride      = 0,
                                 size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::Small,
        "hipblaslt_init_small",
        false,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::randomIntegerOptions(scalar, true, false, false);
        });
}

template <typename T>
inline void hipblaslt_init_sin(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::sineOptions(hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_sin(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_sin",
        true,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::sineOptions(scalar);
        });
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alternating
// Checkerboard ± so first element of each row and column alternates; keeps
// reduction sums from growing too large (helps 16bit with 5-bit exponent).
template <typename T>
inline void hipblaslt_init_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::randomIntegerOptions(
        hipblaslt::host_validation::scalarType<T>(), false, true, false);
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_alternating_sign(void*       A,
                                            size_t      M,
                                            size_t      N,
                                            size_t      lda,
                                            hipDataType type,
                                            size_t      stride      = 0,
                                            size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_alternating_sign",
        true,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::randomIntegerOptions(scalar, false, true, false);
        });
}

// Initialize matrix so adjacent entries have alternating sign.
template <typename T>
inline void hipblaslt_init_hpl_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::hplOptions(
        hipblaslt::host_validation::scalarType<T>(), false, true, false);
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_hpl_alternating_sign(void*       A,
                                                size_t      M,
                                                size_t      N,
                                                size_t      lda,
                                                hipDataType type,
                                                size_t      stride      = 0,
                                                size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_hpl_alternating_sign",
        true,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::hplOptions(scalar, false, true, false);
        });
}

template <typename T>
inline void hipblaslt_init_cos(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    roc::host_validation::GenerationOptions options;
    options.real.pattern = roc::host_validation::GenerationPattern::Cosine;
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_cos(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_cos",
        true,
        [](roc::host_validation::ScalarType) {
            roc::host_validation::GenerationOptions options;
            options.real.pattern = roc::host_validation::GenerationPattern::Cosine;
            return options;
        });
}

// Initialize vector with HPL-like random values
template <typename T>
inline void hipblaslt_init_hpl(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::hplOptions(
        hipblaslt::host_validation::scalarType<T>(), false, false, false);
    hipblaslt::host_validation::initializeMatrixBatches(
        A.data(), M, N, lda, stride, batch_count, options);
}

template <typename T>
inline void hipblaslt_init_hpl(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::hplOptions(
        hipblaslt::host_validation::scalarType<T>(), false, false, false);
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_hpl(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_hpl",
        true,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::hplOptions(scalar, false, false, false);
        });
}

// Initialize vector with uniform random values in [-6, 6]
template <typename T>
inline void hipblaslt_init_low_precision(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::lowPrecisionOptions(
        hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeMatrixBatches(
        A.data(), M, N, lda, stride, batch_count, options);
}

template <typename T>
inline void hipblaslt_init_low_precision(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::lowPrecisionOptions(
        hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_low_precision(void*       A,
                                         size_t      M,
                                         size_t      N,
                                         size_t      lda,
                                         hipDataType type,
                                         size_t      stride      = 0,
                                         size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::LowPrecision,
        "hipblaslt_init_low_precision",
        false,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::lowPrecisionOptions(scalar);
        });
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
inline void hipblaslt_init_nan(T* A, size_t N)
{
    const auto options
        = hipblaslt::host_validation::nanOptions(hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeTensor(
        A, roc::host_validation::Layout::contiguous(roc::host_validation::Shape{N}), options);
}

template <typename T>
inline void hipblaslt_init_nan(T* A, size_t start_offset, size_t end_offset)
{
    hipblaslt_init_nan(A + start_offset, end_offset - start_offset);
}

inline void hipblaslt_init_nan(void* A, size_t N, hipDataType type)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        roc::host_validation::Layout::contiguous(roc::host_validation::Shape{N}),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_nan",
        true,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::nanOptions(scalar);
        });
}

inline void hipblaslt_init_nan(void* A, size_t start_offset, size_t end_offset, hipDataType type)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::contiguousRangeLayout(start_offset, end_offset),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_nan",
        true,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::nanOptions(scalar);
        });
}

template <typename T>
inline void hipblaslt_init_nan(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::nanOptions(hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_nan(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_nan",
        false,
        [](roc::host_validation::ScalarType scalar) {
            return hipblaslt::host_validation::nanOptions(scalar);
        });
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with zero */

template <typename T>
inline void hipblaslt_init_zero(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    roc::host_validation::GenerationOptions options;
    hipblaslt::host_validation::initializeMatrixBatches(
        A.data(), M, N, lda, stride, batch_count, options);
}

template <typename T>
inline void hipblaslt_init_zero(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    roc::host_validation::GenerationOptions options;
    hipblaslt::host_validation::initializeMatrixBatches(A, M, N, lda, stride, batch_count, options);
}

template <typename T>
inline void hipblaslt_init_zero(T* A, size_t start_offset, size_t end_offset)
{
    roc::host_validation::GenerationOptions options;
    hipblaslt::host_validation::initializeTensor(
        A + start_offset,
        roc::host_validation::Layout::contiguous(
            roc::host_validation::Shape{end_offset - start_offset}),
        options);
}

inline void hipblaslt_init_zero(void*       A,
                                size_t      M,
                                size_t      N,
                                size_t      lda,
                                hipDataType type,
                                size_t      stride      = 0,
                                size_t      batch_count = 1)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_zero",
        false,
        [](roc::host_validation::ScalarType) { return roc::host_validation::GenerationOptions{}; });
}

inline void hipblaslt_init_zero(void* A, size_t start_offset, size_t end_offset, hipDataType type)
{
    hipblaslt::host_validation::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_validation::detail::contiguousRangeLayout(start_offset, end_offset),
        hipblaslt::host_validation::detail::RuntimeInitialization::General,
        "hipblaslt_init_zero",
        false,
        [](roc::host_validation::ScalarType) { return roc::host_validation::GenerationOptions{}; });
}
