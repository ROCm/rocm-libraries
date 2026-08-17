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
#include <hipblaslt/host_validation/HipblasltDataInitialization.hpp>
#include <cinttypes>
#include <complex>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <omp.h>
#include <type_traits>
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
                           int norm_dist_one_special_type = -1);

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize matrices with random values
template <typename T>
inline void
    hipblaslt_init(T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    roc::host_validation::GenerationOptions options;
    if constexpr(std::is_same_v<T, hipblaslt_e8>)
    {
        options.real.pattern    = roc::host_validation::GenerationPattern::UniformRawInteger;
        options.real.parameter0 = 1;
        options.real.parameter1 = 10;
    }
    else
        options = hipblaslt::host_validation::randomIntegerOptions(
            hipblaslt::host_validation::scalarType<T>(),
            false,
            false,
            false);
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
}

// Initialize matrices with random values
template <typename T>
inline void hipblaslt_init_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::randomIntegerOptions(
            hipblaslt::host_validation::scalarType<T>(),
            true,
            false,
            false);
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
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
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init<hip_bfloat16>(static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_UE8M0:
        hipblaslt_init<hipblaslt_e8>(
            static_cast<hipblaslt_e8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3:
        hipblaslt_cerr << "hipblaslt_init not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2:
        hipblaslt_cerr << "hipblaslt_init not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1:
        hipblaslt_cerr << "hipblaslt_init not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init" << std::endl;
        break;
    }
}

inline void hipblaslt_init_small(void*       A,
                                 size_t      M,
                                 size_t      N,
                                 size_t      lda,
                                 hipDataType type,
                                 size_t      stride      = 0,
                                 size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_small<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_small<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init_small<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_small<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_small<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_small<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_small" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_sin(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    roc::host_validation::GenerationOptions options;
    options.real.pattern = roc::host_validation::GenerationPattern::Sine;
    if constexpr(std::is_same_v<T, std::complex<float>>
                 || std::is_same_v<T, std::complex<double>>)
        options.imaginary.pattern = roc::host_validation::GenerationPattern::Cosine;
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_sin(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_sin<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_sin<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init_sin<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_sin<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_sin<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_sin<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_sin<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_sin<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_sin<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_sin<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_sin<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_sin<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3:
        hipblaslt_cerr << "hipblaslt_init_sin not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2:
        hipblaslt_cerr << "hipblaslt_init_sin not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1:
        hipblaslt_cerr << "hipblaslt_init_sin not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_sin" << std::endl;
        break;
    }
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alternating
// Checkerboard ± so first element of each row and column alternates; keeps
// reduction sums from growing too large (helps 16bit with 5-bit exponent).
template <typename T>
inline void hipblaslt_init_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::randomIntegerOptions(
            hipblaslt::host_validation::scalarType<T>(),
            false,
            true,
            false);
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_alternating_sign(void*       A,
                                            size_t      M,
                                            size_t      N,
                                            size_t      lda,
                                            hipDataType type,
                                            size_t      stride      = 0,
                                            size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_alternating_sign<float>(
            static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_alternating_sign<double>(
            static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init_alternating_sign<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_alternating_sign<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_alternating_sign<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_alternating_sign<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_alternating_sign<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_alternating_sign<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_alternating_sign<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_alternating_sign<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_alternating_sign<int32_t>(
            static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_alternating_sign<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3:
        hipblaslt_cerr << "hipblaslt_init_alternating_sign not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2:
        hipblaslt_cerr << "hipblaslt_init_alternating_sign not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1:
        hipblaslt_cerr << "hipblaslt_init_alternating_sign not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_alternating_sign" << std::endl;
        break;
    }
}

// Initialize matrix so adjacent entries have alternating sign.
template <typename T>
inline void hipblaslt_init_hpl_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::hplOptions(
            hipblaslt::host_validation::scalarType<T>(),
            false,
            true,
            false);
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_hpl_alternating_sign(void*       A,
                                                size_t      M,
                                                size_t      N,
                                                size_t      lda,
                                                hipDataType type,
                                                size_t      stride      = 0,
                                                size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_hpl_alternating_sign<float>(
            static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_hpl_alternating_sign<double>(
            static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init_hpl_alternating_sign<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_hpl_alternating_sign<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_hpl_alternating_sign<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_hpl_alternating_sign<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_hpl_alternating_sign<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_hpl_alternating_sign<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_hpl_alternating_sign<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_hpl_alternating_sign<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_hpl_alternating_sign<int32_t>(
            static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_hpl_alternating_sign<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3:
        hipblaslt_cerr << "hipblaslt_init_hpl_alternating_sign not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2:
        hipblaslt_cerr << "hipblaslt_init_hpl_alternating_sign not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1:
        hipblaslt_cerr << "hipblaslt_init_hpl_alternating_sign not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_hpl_alternating_sign" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_cos(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    roc::host_validation::GenerationOptions options;
    options.real.pattern = roc::host_validation::GenerationPattern::Cosine;
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_cos(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_cos<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_cos<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init_cos<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_cos<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_cos<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_cos<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_cos<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_cos<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_cos<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_cos<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_cos<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_cos<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3:
        hipblaslt_cerr << "hipblaslt_init_cos not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2:
        hipblaslt_cerr << "hipblaslt_init_cos not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1:
        hipblaslt_cerr << "hipblaslt_init_cos not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_cos" << std::endl;
        break;
    }
}

// Initialize vector with HPL-like random values
template <typename T>
inline void hipblaslt_init_hpl(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::hplOptions(
            hipblaslt::host_validation::scalarType<T>(),
            false,
            false,
            false);
    hipblaslt::host_validation::initializeMatrixBatches(
        A.data(), M, N, lda, stride, batch_count, options);
}

template <typename T>
inline void hipblaslt_init_hpl(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::hplOptions(
            hipblaslt::host_validation::scalarType<T>(),
            false,
            false,
            false);
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_hpl(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_hpl<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_hpl<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init_hpl<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_hpl<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_hpl<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_hpl<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_hpl<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_hpl<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_hpl<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_hpl<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_hpl<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_hpl<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_6F_E2M3:
        hipblaslt_cerr << "hipblaslt_init_hpl not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2:
        hipblaslt_cerr << "hipblaslt_init_hpl not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1:
        hipblaslt_cerr << "hipblaslt_init_hpl not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_hpl" << std::endl;
        break;
    }
}

// Initialize vector with uniform random values in [-6, 6]
template <typename T>
inline void hipblaslt_init_low_precision(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::lowPrecisionOptions(
            hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeMatrixBatches(
        A.data(), M, N, lda, stride, batch_count, options);
}

template <typename T>
inline void hipblaslt_init_low_precision(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options
        = hipblaslt::host_validation::lowPrecisionOptions(
            hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_low_precision(void*       A,
                                         size_t      M,
                                         size_t      N,
                                         size_t      lda,
                                         hipDataType type,
                                         size_t      stride      = 0,
                                         size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_low_precision<float>(
            static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_low_precision<double>(
            static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_low_precision<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_low_precision<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_low_precision<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_low_precision<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_low_precision<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_low_precision<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_low_precision<int32_t>(
            static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_low_precision<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_low_precision" << std::endl;
        break;
    }
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
inline void hipblaslt_init_nan(T* A, size_t N)
{
    const auto options = hipblaslt::host_validation::nanOptions(
        hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeTensor(
        A, roc::host_validation::Layout::contiguous(
               roc::host_validation::Shape{N}),
        options);
}

template <typename T>
inline void hipblaslt_init_nan(T* A, size_t start_offset, size_t end_offset)
{
    hipblaslt_init_nan(A + start_offset, end_offset - start_offset);
}

inline void hipblaslt_init_nan(void* A, size_t N, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_nan<float>(static_cast<float*>(A), N);
        break;
    case HIP_R_64F:
        hipblaslt_init_nan<double>(static_cast<double*>(A), N);
        break;
    case HIP_C_32F:
        hipblaslt_init_nan<std::complex<float>>(static_cast<std::complex<float>*>(A), N);
        break;
    case HIP_C_64F:
        hipblaslt_init_nan<std::complex<double>>(static_cast<std::complex<double>*>(A), N);
        break;
    case HIP_R_16F:
        hipblaslt_init_nan<hipblasLtHalf>(static_cast<hipblasLtHalf*>(A), N);
        break;
    case HIP_R_16BF:
        hipblaslt_init_nan<hip_bfloat16>(static_cast<hip_bfloat16*>(A), N);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_nan<hipblaslt_f8_fnuz>(static_cast<hipblaslt_f8_fnuz*>(A), N);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_nan<hipblaslt_bf8_fnuz>(static_cast<hipblaslt_bf8_fnuz*>(A), N);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_nan<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), N);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_nan<hipblaslt_bf8>(static_cast<hipblaslt_bf8*>(A), N);
        break;
    case HIP_R_32I:
        hipblaslt_init_nan<int32_t>(static_cast<int32_t*>(A), N);
        break;
    case HIP_R_8I:
        hipblaslt_init_nan<hipblasLtInt8>(static_cast<hipblasLtInt8*>(A), N);
        break;
    case HIP_R_6F_E2M3:
        hipblaslt_cerr << "hipblaslt_init_nan not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2:
        hipblaslt_cerr << "hipblaslt_init_nan not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1:
        hipblaslt_cerr << "hipblaslt_init_nan not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_nan" << std::endl;
        break;
    }
}

inline void hipblaslt_init_nan(void* A, size_t start_offset, size_t end_offset, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_nan<float>(static_cast<float*>(A), start_offset, end_offset);
        break;
    case HIP_R_64F:
        hipblaslt_init_nan<double>(static_cast<double*>(A), start_offset, end_offset);
        break;
    case HIP_C_32F:
        hipblaslt_init_nan<std::complex<float>>(
            static_cast<std::complex<float>*>(A), start_offset, end_offset);
        break;
    case HIP_C_64F:
        hipblaslt_init_nan<std::complex<double>>(
            static_cast<std::complex<double>*>(A), start_offset, end_offset);
        break;
    case HIP_R_16F:
        hipblaslt_init_nan<hipblasLtHalf>(static_cast<hipblasLtHalf*>(A), start_offset, end_offset);
        break;
    case HIP_R_16BF:
        hipblaslt_init_nan<hip_bfloat16>(static_cast<hip_bfloat16*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_nan<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_nan<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_nan<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_nan<hipblaslt_bf8>(static_cast<hipblaslt_bf8*>(A), start_offset, end_offset);
        break;
    case HIP_R_32I:
        hipblaslt_init_nan<int32_t>(static_cast<int32_t*>(A), start_offset, end_offset);
        break;
    case HIP_R_8I:
        hipblaslt_init_nan<hipblasLtInt8>(static_cast<hipblasLtInt8*>(A), start_offset, end_offset);
        break;
    case HIP_R_6F_E2M3:
        hipblaslt_cerr << "hipblaslt_init_nan not supports FP6" << std::endl;
        break;
    case HIP_R_6F_E3M2:
        hipblaslt_cerr << "hipblaslt_init_nan not supports BF6" << std::endl;
        break;
    case HIP_R_4F_E2M1:
        hipblaslt_cerr << "hipblaslt_init_nan not supports FP4" << std::endl;
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_nan" << std::endl;
        break;
    }
}

template <typename T>
inline void hipblaslt_init_nan(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto options = hipblaslt::host_validation::nanOptions(
        hipblaslt::host_validation::scalarType<T>());
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
}

inline void hipblaslt_init_nan(void*       A,
                               size_t      M,
                               size_t      N,
                               size_t      lda,
                               hipDataType type,
                               size_t      stride      = 0,
                               size_t      batch_count = 1)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_nan<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_nan<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init_nan<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_nan<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_nan<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_nan<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_nan<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_nan<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_nan<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_nan<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_nan<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_nan<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_nan" << std::endl;
        break;
    }
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
    hipblaslt::host_validation::initializeMatrixBatches(
        A, M, N, lda, stride, batch_count, options);
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
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_zero<float>(static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_zero<double>(static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_32F:
        hipblaslt_init_zero<std::complex<float>>(
            static_cast<std::complex<float>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_zero<std::complex<double>>(
            static_cast<std::complex<double>*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_zero<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_zero<hip_bfloat16>(
            static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_zero<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_zero<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_zero<hipblaslt_f8>(
            static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_zero<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_32I:
        hipblaslt_init_zero<int32_t>(static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_zero<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_zero" << std::endl;
        break;
    }
}

inline void hipblaslt_init_zero(void* A, size_t start_offset, size_t end_offset, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_zero<float>(static_cast<float*>(A), start_offset, end_offset);
        break;
    case HIP_R_64F:
        hipblaslt_init_zero<double>(static_cast<double*>(A), start_offset, end_offset);
        break;
    case HIP_C_32F:
        hipblaslt_init_zero<std::complex<float>>(
            static_cast<std::complex<float>*>(A), start_offset, end_offset);
        break;
    case HIP_C_64F:
        hipblaslt_init_zero<std::complex<double>>(
            static_cast<std::complex<double>*>(A), start_offset, end_offset);
        break;
    case HIP_R_16F:
        hipblaslt_init_zero<hipblasLtHalf>(
            static_cast<hipblasLtHalf*>(A), start_offset, end_offset);
        break;
    case HIP_R_16BF:
        hipblaslt_init_zero<hip_bfloat16>(static_cast<hip_bfloat16*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_zero<hipblaslt_f8_fnuz>(
            static_cast<hipblaslt_f8_fnuz*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_zero<hipblaslt_bf8_fnuz>(
            static_cast<hipblaslt_bf8_fnuz*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E4M3:
        hipblaslt_init_zero<hipblaslt_f8>(static_cast<hipblaslt_f8*>(A), start_offset, end_offset);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_zero<hipblaslt_bf8>(
            static_cast<hipblaslt_bf8*>(A), start_offset, end_offset);
        break;
    case HIP_R_32I:
        hipblaslt_init_zero<int32_t>(static_cast<int32_t*>(A), start_offset, end_offset);
        break;
    case HIP_R_8I:
        hipblaslt_init_zero<hipblasLtInt8>(
            static_cast<hipblasLtInt8*>(A), start_offset, end_offset);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_zero" << std::endl;
        break;
    }
}
