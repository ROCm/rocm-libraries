/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "cblas.h"
#include "hipsparselt_vector.hpp"
#include "hipsparselt_fp8.hpp"
#include "norm.hpp"
#include "utility.hpp"
#include <cstdio>
#include <hipsparselt/hipsparselt.h>
#include <limits>
#include <memory>

/* =====================================================================
        Norm check: norm(A-B)/norm(A), evaluate relative error
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Norm check
 */

/* ========================================Norm Check
 * ==================================================== */

/* LAPACK fortran library functionality */
extern "C" {
float  slange_(char* norm_type, int* m, int* n, float* A, int* lda, float* work);
double dlange_(char* norm_type, int* m, int* n, double* A, int* lda, double* work);

float  slansy_(char* norm_type, char* uplo, int* n, float* A, int* lda, float* work);
double dlansy_(char* norm_type, char* uplo, int* n, double* A, int* lda, double* work);

void saxpy_(int* n, float* alpha, float* x, int* incx, float* y, int* incy);
void daxpy_(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
}

/*! \brief  Overloading: norm check for general Matrix: half/float/doubel/complex */
inline float xlange(char* norm_type, int* m, int* n, float* A, int* lda, float* work)
{
    return slange_(norm_type, m, n, A, lda, work);
}

inline double xlange(char* norm_type, int* m, int* n, double* A, int* lda, double* work)
{
    return dlange_(norm_type, m, n, A, lda, work);
}

inline float xlanhe(char* norm_type, char* uplo, int* n, float* A, int* lda, float* work)
{
    return slansy_(norm_type, uplo, n, A, lda, work);
}

inline double xlanhe(char* norm_type, char* uplo, int* n, double* A, int* lda, double* work)
{
    return dlansy_(norm_type, uplo, n, A, lda, work);
}

inline void xaxpy(int* n, float* alpha, float* x, int* incx, float* y, int* incy)
{
    return saxpy_(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int* n, double* alpha, double* x, int* incx, double* y, int* incy)
{
    return daxpy_(n, alpha, x, incx, y, incy);
}

template <typename T>
void m_axpy(size_t* N, T* alpha, T* x, int* incx, T* y, int* incy)
{
    for(size_t i = 0; i < *N; i++)
    {
        y[i * (*incy)] = (*alpha) * x[i * (*incx)] + y[i * (*incy)];
    }
}

/* ============== Norm Check for General Matrix ============= */
/*! \brief compare the norm error of two matrices hCPU & hGPU */

// Real
template <
    typename T,
    std::enable_if_t<!(std::is_same<T, hipsparselt_fp8_e4m3>{} || std::is_same<T, hipsparselt_fp8_e5m2>{}
                       ),
                     int>
    = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    if(M * N == 0)
        return 0;
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    size_t size = N * (size_t)lda;

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(hCPU[idx]);
            hGPU_double[idx] = double(hGPU[idx]);
        }
    }

    double work[1];
    int    incx  = 1;
    double alpha = -1.0;
    int    m     = static_cast<int>(M);
    int    n     = static_cast<int>(N);
    int    l     = static_cast<int>(lda);

    double cpu_norm = xlange(&norm_type, &m, &n, hCPU_double.data(), &l, work);
    m_axpy(&size, &alpha, hCPU_double.data(), &incx, hGPU_double.data(), &incx);
    double gpu_norm = xlange(&norm_type, &m, &n, hGPU_double.data(), &l, work);
    if(gpu_norm == 0.0 && cpu_norm == 0.0)
        return 0;
    double error = gpu_norm / cpu_norm;
    return error;
}


template <
    typename T,
    std::enable_if_t<(std::is_same<T, hipsparselt_fp8_e4m3>{} || std::is_same<T, hipsparselt_fp8_e5m2>{}), int>
    = 0>
double norm_check_general(char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU, T* hGPU)
{
    if(M * N == 0)
        return 0;
    size_t size = N * (size_t)lda;

    host_vector<double> hCPU_double(size);
    host_vector<double> hGPU_double(size);

    for(int64_t i = 0; i < N; i++)
    {
        for(int64_t j = 0; j < M; j++)
        {
            size_t idx       = j + i * (size_t)lda;
            hCPU_double[idx] = double(float(hCPU[idx]));
            hGPU_double[idx] = double(float(hGPU[idx]));
        }
    }

    double work[1];
    int    incx  = 1;
    double alpha = -1.0;
    int    m     = static_cast<int>(M);
    int    n     = static_cast<int>(N);
    int    l     = static_cast<int>(lda);

    double cpu_norm = xlange(&norm_type, &m, &n, hCPU_double.data(), &l, work);
    m_axpy(&size, &alpha, hCPU_double.data(), &incx, hGPU_double.data(), &incx);
    double gpu_norm = xlange(&norm_type, &m, &n, hGPU_double.data(), &l, work);
    if(gpu_norm == 0.0 && cpu_norm == 0.0)
        return 0;
    double error = gpu_norm / cpu_norm;

    return error;
}

/* ============== Norm Check for strided_batched case ============= */
template <typename T, template <typename> class VEC, typename T_hpa>
double norm_check_general(char        norm_type,
                          int64_t     M,
                          int64_t     N,
                          int64_t     lda,
                          int64_t     stride_a,
                          VEC<T_hpa>& hCPU,
                          T*          hGPU,
                          int64_t     batch_count)
{
    if(M * N == 0)
        return 0;
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

/* ============== Norm Check for strided_batched case ============= */
template <typename T, typename T_hpa>
double norm_check_general(char    norm_type,
                          int64_t M,
                          int64_t N,
                          int64_t lda,
                          int64_t stride_a,
                          T_hpa*  hCPU,
                          T*      hGPU,
                          int64_t batch_count)
{
    if(M * N == 0)
        return 0;
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

        auto error = norm_check_general(norm_type, M, N, lda, hCPU + index, hGPU + index);

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
template <typename T>
double norm_check_general(
    char norm_type, int64_t M, int64_t N, int64_t lda, T* hCPU[], T* hGPU[], int64_t batch_count)
{
    if(M * N == 0)
        return 0;
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

double norm_check_general(
    char norm_type, int64_t M, int64_t N, int64_t lda, void* hCPU, void* hGPU, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return norm_check_general<float>(
            norm_type, M, N, lda, static_cast<float*>(hCPU), static_cast<float*>(hGPU));
    case HIP_R_64F:
        return norm_check_general<double>(
            norm_type, M, N, lda, static_cast<double*>(hCPU), static_cast<double*>(hGPU));
    case HIP_R_16F:
        return norm_check_general<__half>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 static_cast<__half*>(hCPU),
                                                 static_cast<__half*>(hGPU));
    case HIP_R_16BF:
        return norm_check_general<hip_bfloat16>(norm_type,
                                                M,
                                                N,
                                                lda,
                                                static_cast<hip_bfloat16*>(hCPU),
                                                static_cast<hip_bfloat16*>(hGPU));
    case HIP_R_8F_E4M3:
        return norm_check_general<hipsparselt_fp8_e4m3>(norm_type,
                                                M,
                                                N,
                                                lda,
                                                static_cast<hipsparselt_fp8_e4m3*>(hCPU),
                                                static_cast<hipsparselt_fp8_e4m3*>(hGPU));
    case HIP_R_8F_E5M2:
        return norm_check_general<hipsparselt_fp8_e5m2>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 static_cast<hipsparselt_fp8_e5m2*>(hCPU),
                                                 static_cast<hipsparselt_fp8_e5m2*>(hGPU));
    case HIP_R_32I:
        return norm_check_general<int32_t>(
            norm_type, M, N, lda, static_cast<int32_t*>(hCPU), static_cast<int32_t*>(hGPU));
    case HIP_R_8I:
        return norm_check_general<int8_t>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 static_cast<int8_t*>(hCPU),
                                                 static_cast<int8_t*>(hGPU));
    default:
        hipsparselt_cerr << "Error type in norm_check_general" << std::endl;
        return 0;
    }
}

double norm_check_general(char        norm_type,
                          int64_t     M,
                          int64_t     N,
                          int64_t     lda,
                          int64_t     stride_a,
                          void*       hCPU,
                          void*       hGPU,
                          int64_t     batch_count,
                          hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return norm_check_general<float>(norm_type,
                                         M,
                                         N,
                                         lda,
                                         stride_a,
                                         static_cast<float*>(hCPU),
                                         static_cast<float*>(hGPU),
                                         batch_count);
    case HIP_R_16F:
        return norm_check_general<__half>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 stride_a,
                                                 static_cast<__half*>(hCPU),
                                                 static_cast<__half*>(hGPU),
                                                 batch_count);
    case HIP_R_16BF:
        return norm_check_general<hip_bfloat16>(norm_type,
                                                M,
                                                N,
                                                lda,
                                                stride_a,
                                                static_cast<hip_bfloat16*>(hCPU),
                                                static_cast<hip_bfloat16*>(hGPU),
                                                batch_count);
        return norm_check_general<hipsparselt_fp8_e4m3>(norm_type,
                                                M,
                                                N,
                                                lda,
                                                stride_a,
                                                static_cast<hipsparselt_fp8_e4m3*>(hCPU),
                                                static_cast<hipsparselt_fp8_e4m3*>(hGPU),
                                                batch_count);
    case HIP_R_8F_E5M2:
        return norm_check_general<hipsparselt_fp8_e5m2>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 stride_a,
                                                 static_cast<hipsparselt_fp8_e5m2*>(hCPU),
                                                 static_cast<hipsparselt_fp8_e5m2*>(hGPU),
                                                 batch_count);
    case HIP_R_32I:
        return norm_check_general<int32_t>(norm_type,
                                           M,
                                           N,
                                           lda,
                                           stride_a,
                                           static_cast<int32_t*>(hCPU),
                                           static_cast<int32_t*>(hGPU),
                                           batch_count);
    case HIP_R_8I:
        return norm_check_general<int8_t>(norm_type,
                                                 M,
                                                 N,
                                                 lda,
                                                 stride_a,
                                                 static_cast<int8_t*>(hCPU),
                                                 static_cast<int8_t*>(hGPU),
                                                 batch_count);
    default:
        hipsparselt_cerr << "Error type in norm_check_general" << std::endl;
        return 0;
    }
}

template <typename T>
bool norm_check(double norm_error)
{
    hipsparselt_cout << norm_error << std::endl;
    if(std::is_same<T, float>{})
        return norm_error < 0.00001;
    if(std::is_same<T, double>{})
        return norm_error < 0.000000000001;
    if(std::is_same<T, __half>{})
        return norm_error < 0.01;
    if(std::is_same<T, hip_bfloat16>{})
        return norm_error < 0.1;
    if(std::is_same<T, int8_t>{})
        return norm_error < 0.01;
    if(std::is_same<T, int32_t>{})
        return norm_error < 0.0001;
    if(std::is_same<T, hipsparselt_fp8_e4m3>{})
        return norm_error < 0.125;
    if(std::is_same<T, hipsparselt_fp8_e5m2>{})
        return norm_error < 0.25;
    return false;
}

bool norm_check(double norm_error, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return norm_error < 0.00001;
    case HIP_R_64F:
        return norm_error < 0.000000000001;
    case HIP_R_16F:
        return norm_error < 0.01;
    case HIP_R_16BF:
        return norm_error < 0.1;
    case HIP_R_8F_E4M3:
        return norm_error < 0.125;
    case HIP_R_8F_E5M2:
        return norm_error < 0.25;
    case HIP_R_32I:
        return norm_error < 0.0001;
    case HIP_R_8I:
        return norm_error < 0.01;
    default:
        return false;
    }
}
