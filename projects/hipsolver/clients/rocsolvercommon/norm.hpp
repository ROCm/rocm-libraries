/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************ */

#pragma once

#include <vector>

#include "../include/complex.hpp"
#include "hipsolver.h"

using rocblas_float_complex  = hipsolverComplex;
using rocblas_double_complex = hipsolverDoubleComplex;

/* LAPACK fortran library functionality */

extern "C" {
float  slange_(const char*  norm_type,
               const int*   m,
               const int*   n,
               const float* A,
               const int*   lda,
               float*       rwork);
double dlange_(const char*   norm_type,
               const int*    m,
               const int*    n,
               const double* A,
               const int*    lda,
               double*       rwork);
float  clange_(const char*                  norm_type,
               const int*                   m,
               const int*                   n,
               const rocblas_float_complex* A,
               const int*                   lda,
               float*                       rwork);
double zlange_(const char*                   norm_type,
               const int*                    m,
               const int*                    n,
               const rocblas_double_complex* A,
               const int*                    lda,
               double*                       rwork);

void daxpy_(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
void zaxpy_(int*                    n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* x,
            int*                    incx,
            rocblas_double_complex* y,
            int*                    incy);
}

inline float xlange(
    const char* norm_type, const int* m, const int* n, const float* A, const int* lda, float* rwork)
{
    return slange_(norm_type, m, n, A, lda, rwork);
}

inline double xlange(const char*   norm_type,
                     const int*    m,
                     const int*    n,
                     const double* A,
                     const int*    lda,
                     double*       rwork)
{
    return dlange_(norm_type, m, n, A, lda, rwork);
}

inline float xlange(const char*                  norm_type,
                    const int*                   m,
                    const int*                   n,
                    const rocblas_float_complex* A,
                    const int*                   lda,
                    float*                       rwork)
{
    return clange_(norm_type, m, n, A, lda, rwork);
}

inline double xlange(const char*                   norm_type,
                     const int*                    m,
                     const int*                    n,
                     const rocblas_double_complex* A,
                     const int*                    lda,
                     double*                       rwork)
{
    return zlange_(norm_type, m, n, A, lda, rwork);
}

inline void xaxpy(int* n, double* alpha, double* x, int* incx, double* y, int* incy)
{
    return daxpy_(n, alpha, x, incx, y, incy);
}

inline void xaxpy(int*                    n,
                  rocblas_double_complex* alpha,
                  rocblas_double_complex* x,
                  int*                    incx,
                  rocblas_double_complex* y,
                  int*                    incy)
{
    return zaxpy_(n, alpha, x, incx, y, incy);
}

/* Norm of error functions */
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
double norm_error(char        norm_type,
                  rocblas_int M,
                  rocblas_int N,
                  rocblas_int lda_gold,
                  T*          gold,
                  T*          comp,
                  rocblas_int lda_comp = 0)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or
    // Frobenius norm one norm is max column sum infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    rocblas_int lda = M;
    lda_comp        = lda_comp > 0 ? lda_comp : lda_gold;

    std::vector<double> gold_double(N * lda);
    std::vector<double> comp_double(N * lda);

    for(rocblas_int i = 0; i < M; i++)
    {
        for(rocblas_int j = 0; j < N; j++)
        {
            gold_double[i + j * lda] = double(gold[i + j * lda_gold]);
            comp_double[i + j * lda] = double(comp[i + j * lda_comp]);
        }
    }

    std::vector<double> work(M);
    rocblas_int         incx  = 1;
    double              alpha = -1.0;
    rocblas_int         size  = lda * N;

    double gold_norm = xlange(&norm_type, &M, &N, gold_double.data(), &lda, work.data());
    xaxpy(&size, &alpha, gold_double.data(), &incx, comp_double.data(), &incx);
    double error = xlange(&norm_type, &M, &N, comp_double.data(), &lda, work.data());
    if(gold_norm > 0)
        error /= gold_norm;

    return error;
}

template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
double norm_error(char        norm_type,
                  rocblas_int M,
                  rocblas_int N,
                  rocblas_int lda_gold,
                  T*          gold,
                  T*          comp,
                  rocblas_int lda_comp = 0)
{
    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or
    // Frobenius norm one norm is max column sum infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    rocblas_int lda = M;
    lda_comp        = lda_comp > 0 ? lda_comp : lda_gold;

    std::vector<rocblas_double_complex> gold_double(N * lda);
    std::vector<rocblas_double_complex> comp_double(N * lda);

    for(rocblas_int i = 0; i < M; i++)
    {
        for(rocblas_int j = 0; j < N; j++)
        {
            gold_double[i + j * lda] = rocblas_double_complex(std::real(gold[i + j * lda_gold]),
                                                              std::imag(gold[i + j * lda_gold]));
            comp_double[i + j * lda] = rocblas_double_complex(std::real(comp[i + j * lda_comp]),
                                                              std::imag(comp[i + j * lda_comp]));
        }
    }

    std::vector<double>    work(M);
    rocblas_int            incx  = 1;
    rocblas_double_complex alpha = -1.0;
    rocblas_int            size  = lda * N;

    double gold_norm = xlange(&norm_type, &M, &N, gold_double.data(), &lda, work.data());
    xaxpy(&size, &alpha, gold_double.data(), &incx, comp_double.data(), &incx);
    double error = xlange(&norm_type, &M, &N, comp_double.data(), &lda, work.data());
    if(gold_norm > 0)
        error /= gold_norm;

    return error;
}

template <typename T>
double norm_error_upperTr(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* gold, T* comp)
{
    for(rocblas_int i = 0; i < M; ++i)
    {
        for(rocblas_int j = 0; j < N; ++j)
        {
            if(i > j)
            {
                gold[i + j * lda] = T(0);
                comp[i + j * lda] = T(0);
            }
        }
    }
    return norm_error(norm_type, M, N, lda, gold, comp);
}

template <typename T>
double norm_error_lowerTr(
    char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* gold, T* comp)
{
    for(rocblas_int i = 0; i < M; ++i)
    {
        for(rocblas_int j = 0; j < N; ++j)
        {
            if(i < j)
            {
                gold[i + j * lda] = T(0);
                comp[i + j * lda] = T(0);
            }
        }
    }
    return norm_error(norm_type, M, N, lda, gold, comp);
}

template <typename T, typename S = decltype(std::real(T{}))>
S snorm(char norm_type, rocblas_int m, rocblas_int n, T* A, rocblas_int lda)
{
    return xlange(&norm_type, &m, &n, A, &lda, (S*)nullptr);
}
