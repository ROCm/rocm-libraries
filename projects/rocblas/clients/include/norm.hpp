/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "cblas_interface.hpp"
#include "client_utility.hpp"
#include "norm.hpp"
#include "rocblas.h"
#include "rocblas_vector.hpp"
#include <cstdio>
#include <limits>
#include <memory>

/* =====================================================================
        Norm check: norm(A-B)/norm(A), evaluate relative error
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Norm check
 */

/* ========================================Norm Check* ==================================================== */

template <typename T>
void m_axpy_64(int64_t N, T* alpha, T* x, int64_t incx, T* y, int64_t incy)
{
    int64_t x_offset = incx >= 0 ? 0 : incx * (1 - N);
    int64_t y_offset = incy >= 0 ? 0 : incy * (1 - N);
    for(int64_t i = 0; i < N; i++)
    {
        y[y_offset + i * incy] = (*alpha) * x[x_offset + i * incx] + y[y_offset + i * incy];
    }
}

/* ============== Norm Check for General Matrix ============= */
/*! \brief compare the norm error of two matrices hCPU & hGPU */

// Real
template <
    typename T,
    std::enable_if_t<!rocblas_is_complex<
                         T> && !(std::is_same<T, rocblas_f8>{} || std::is_same<T, rocblas_bf8>{}),
                     int> = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    host_vector<double> work(std::max(int64_t(1), M));
    int64_t             incx  = 1;
    double              alpha = -1.0;

    size_t size = M * size_t(N); // copying data so lda is M

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        int64_t src_col = i * int64_t(lda);
        int64_t dst_col = i * int64_t(M);
        for(int64_t j = 0; j < M; j++)
        {
            hCPU_double[size_t(dst_col + j)] = double(hCPU[src_col + j]);
            hGPU_double[size_t(dst_col + j)] = double(hGPU[src_col + j]);
        }
    }

    double cpu_norm = ref_lapack_xlange(norm_type, M, N, hCPU_double.data(), M, work.data());
    m_axpy_64(size, &alpha, hCPU_double.data(), incx, hGPU_double.data(), incx);
    double error
        = ref_lapack_xlange(norm_type, M, N, hGPU_double.data(), M, work.data()) / cpu_norm;
    return error;
}

// For F8 , we convert the results to float to double first
template <
    typename T,
    std::enable_if_t<(std::is_same<T, rocblas_f8>{} || std::is_same<T, rocblas_bf8>{}), int> = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    size_t size = M * size_t(N); // copying data so lda is M

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        int64_t src_col = i * int64_t(lda);
        int64_t dst_col = i * int64_t(M);
        for(int64_t j = 0; j < M; j++)
        {
            hCPU_double[size_t(dst_col + j)] = double(float(hCPU[src_col + j]));
            hGPU_double[size_t(dst_col + j)] = double(float(hGPU[src_col + j]));
        }
    }

    host_vector<double> work(std::max(int64_t(1), M));
    int64_t             incx  = 1;
    double              alpha = -1.0;

    double cpu_norm = ref_lapack_xlange(norm_type, M, N, hCPU_double.data(), M, work.data());
    m_axpy_64(size, &alpha, hCPU_double.data(), incx, hGPU_double.data(), incx);
    double error
        = ref_lapack_xlange(norm_type, M, N, hGPU_double.data(), M, work.data()) / cpu_norm;
    return error;
}

// Complex
template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    host_vector<double> work(std::max(int64_t(1), M));
    int64_t             incx  = 1;
    T                   alpha = -1.0;
    int64_t             size  = N * (int64_t)lda;

    double cpu_norm = ref_lapack_xlange(norm_type, M, N, hCPU, lda, work.data());
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
    double error = ref_lapack_xlange(norm_type, M, N, hGPU, lda, work.data()) / cpu_norm;

    return error;
}

// For BF16 and half, we convert the results to double first
template <typename T,
          typename VEC,
          std::enable_if_t<std::is_same_v<T, rocblas_half> || std::is_same_v<T, rocblas_bfloat16>,
                           int> = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, VEC&& hCPU, T* hGPU)
{
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = hCPU[idx];
            hGPU_double[idx] = hGPU[idx];
        }
    }

    return norm_check_general<double>(norm_type, M, N, lda, hCPU_double, hGPU_double);
}

/* ============== Norm Check for strided_batched case ============= */
template <typename T, template <typename> class VEC, typename T_hpa>
double norm_check_general(char           norm_type,
                          int64_t        M,
                          int64_t        N,
                          int64_t        lda,
                          rocblas_stride stride_a,
                          VEC<T_hpa>&    hCPU,
                          T*             hGPU,
                          int64_t        batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(size_t i = 0; i < batch_count; i++)
    {
        auto index = i * stride_a;

        auto error = norm_check_general(norm_type, M, N, lda, (T_hpa*)hCPU + index, hGPU + index);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T, typename U>
double norm_check_general(char norm_type, T& hCPU, U& hGPU)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix
    int64_t M                = hCPU.m();
    int64_t N                = hCPU.n();
    size_t  lda              = hCPU.lda();
    int64_t batch_count      = hCPU.batch_count();
    double  cumulative_error = 0.0;

    for(int64_t b = 0; b < batch_count; b++)
    {
        auto* CPU   = hCPU[b];
        auto* GPU   = hGPU[b];
        auto  error = norm_check_general(norm_type, M, N, lda, CPU, GPU);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

/* ============== Norm Check for batched case ============= */

template <typename T, typename T_hpa>
double norm_check_general(char                      norm_type,
                          int64_t                   M,
                          int64_t                   N,
                          int64_t                   lda,
                          host_batch_vector<T_hpa>& hCPU,
                          host_batch_vector<T>&     hGPU,
                          int64_t                   batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(int64_t i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T>
double norm_check_general(
    char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU[], T* hGPU[], int64_t batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(int64_t i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

/* ============== Norm Check for Symmetric Matrix ============= */
/*! \brief compare the norm error of two Hermitian/symmetric matrices hCPU & hGPU */
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0, bool HERM = false>
double norm_check_symmetric(char norm_type, char uplo, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<double> work(std::max(int64_t(1), N));
    int64_t             incx  = 1;
    double              alpha = -1.0;
    size_t              size  = N * (size_t)lda;

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    double cpu_norm
        = ref_lapack_xlansy<HERM>(norm_type, uplo, N, hCPU_double.data(), lda, work.data());
    m_axpy_64(size, &alpha, hCPU_double.data(), incx, hGPU_double.data(), incx);
    double error = ref_lapack_xlansy<HERM>(norm_type, uplo, N, hGPU_double.data(), lda, work.data())
                   / cpu_norm;

    return error;
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0, bool HERM = false>
double norm_check_symmetric(char norm_type, char uplo, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly
    host_vector<double> work(std::max(int64_t(1), N));
    int64_t             incx  = 1;
    T                   alpha = -1.0;
    size_t              size  = (size_t)lda * N;

    double cpu_norm = ref_lapack_xlansy<HERM>(norm_type, uplo, N, hCPU, lda, work.data());
    m_axpy_64(size, &alpha, hCPU, incx, hGPU, incx);
    double error = ref_lapack_xlansy<HERM>(norm_type, uplo, N, hGPU, lda, work.data()) / cpu_norm;

    return error;
}

template <>
inline double norm_check_symmetric(
    char norm_type, char uplo, int64_t N, int64_t lda, rocblas_half* hCPU, rocblas_half* hGPU)
{
    size_t              size = N * (size_t)lda;
    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < N; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = hCPU[idx];
            hGPU_double[idx] = hGPU[idx];
        }
    }

    return norm_check_symmetric(norm_type, uplo, N, lda, hCPU_double.data(), hGPU_double.data());
}

template <typename T, bool HERM = false>
double norm_check_symmetric(
    char norm_type, char uplo, int64_t N, int64_t lda, T* hCPU[], T* hGPU[], int64_t batch_count)
{
    double cumulative_error = 0.0;

    for(int64_t b = 0; b < batch_count; b++)
    {
        auto error = norm_check_symmetric<T, HERM>(norm_type, uplo, N, lda, hCPU[b], hGPU[b]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T, bool HERM = false>
double norm_check_symmetric(char           norm_type,
                            char           uplo,
                            int64_t        N,
                            int64_t        lda,
                            rocblas_stride stridea,
                            T*             hCPU,
                            T*             hGPU,
                            int64_t        batch_count)
{
    double cumulative_error = 0.0;

    for(int64_t b = 0; b < batch_count; b++)
    {
        auto error = norm_check_symmetric<T, HERM>(
            norm_type, uplo, N, lda, hCPU + b * stridea, hGPU + b * stridea);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

template <typename T>
double matrix_norm_1(int64_t M, int64_t N, int64_t lda, T* hA_gold, T* hA)
{
    double max_err_scal = 0.0;
    double max_err      = 0.0;
    double err          = 0.0;
    double err_scal     = 0.0;
    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx = j + i * (size_t)lda;
            err += rocblas_abs((hA_gold[idx] - hA[idx]));
            err_scal += rocblas_abs(hA_gold[idx]);
        }
        max_err_scal = max_err_scal > err_scal ? max_err_scal : err_scal;
        max_err      = max_err > err ? max_err : err;
    }

    return max_err / max_err_scal;
}

// overload with different leading dimensions
template <typename T>
double matrix_norm_1(int64_t M, int64_t N, T* hA_gold, int64_t lda_gold, T* hA, int64_t lda)
{
    double max_err_scal = 0.0;
    double max_err      = 0.0;
    double err          = 0.0;
    double err_scal     = 0.0;
    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idxAg = j + i * (size_t)lda_gold;
            size_t idxA  = j + i * (size_t)lda;
            err += rocblas_abs((hA_gold[idxAg] - hA[idxA]));
            err_scal += rocblas_abs(hA_gold[idxAg]);
        }
        max_err_scal = max_err_scal > err_scal ? max_err_scal : err_scal;
        max_err      = max_err > err ? max_err : err;
    }

    return max_err / max_err_scal;
}

template <typename T>
double vector_norm_1(int64_t M, int64_t incx, T* hx_gold, T* hx)
{
    double  max_err_scal = 0.0;
    double  max_err      = 0.0;
    int64_t x_offset     = incx >= 0 ? 0 : int64_t(incx) * (1 - M);
    for(int64_t i = 0; i < M; i++)
    {
        size_t idx = x_offset + i * (int64_t)incx;
        max_err += rocblas_abs((hx_gold[idx] - hx[idx]));
        max_err_scal += rocblas_abs(hx_gold[idx]);
    }

    return max_err / max_err_scal;
}

template <typename T>
double vector_norm_1(int64_t                     M,
                     int64_t                     incx,
                     const host_batch_vector<T>& hx_gold,
                     const host_batch_vector<T>& hx)
{
    double max_err = 0.0;

    for(size_t b = 0; b < hx_gold.batch_count(); b++)
    {
        max_err = std::max(max_err, vector_norm_1(M, incx, hx_gold[b], hx[b]));
    }

    return max_err;
}

template <typename T>
double vector_norm_1(int64_t                             M,
                     int64_t                             incx,
                     const host_strided_batch_vector<T>& hx_gold,
                     const host_strided_batch_vector<T>& hx)
{
    double max_err = 0.0;

    for(size_t b = 0; b < hx_gold.batch_count(); b++)
    {
        max_err = std::max(
            max_err, vector_norm_1(M, incx, hx_gold + b * hx_gold.stride(), hx + b * hx.stride()));
    }

    return max_err;
}
