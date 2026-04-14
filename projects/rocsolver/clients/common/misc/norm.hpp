/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <type_traits>
#include <vector>

#include <rocblas/rocblas.h>

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"

/* Norm of error functions */

template <typename T>
double norm_error(char norm_type,
                  rocblas_int M,
                  rocblas_int N,
                  rocblas_int lda_gold,
                  T* gold,
                  T* comp,
                  rocblas_int lda_comp = 0)
{
    using DoublePrecisionType
        = std::conditional_t<rocblas_is_complex<T>, rocblas_double_complex, double>;

    // norm type can be 'O', 'I', 'F', 'o', 'i', 'f' for one, infinity or
    // Frobenius norm one norm is max column sum infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries

    rocblas_int lda = M;
    if(lda_comp <= 0)
        lda_comp = lda_gold;

    std::vector<DoublePrecisionType> gold_double(N * lda);
    std::vector<DoublePrecisionType> comp_double(N * lda);

    for(rocblas_int j = 0; j < N; j++)
    {
        for(rocblas_int i = 0; i < M; i++)
        {
            gold_double[i + j * lda] = DoublePrecisionType(gold[i + j * lda_gold]);
            comp_double[i + j * lda] = DoublePrecisionType(comp[i + j * lda_comp]);
        }
    }

    std::vector<double> work(M);
    rocblas_int incx = 1;
    DoublePrecisionType alpha = -1.0;
    rocblas_int size = lda * N;

    double gold_norm = cpu_lange(norm_type, M, N, gold_double.data(), lda, work.data());
    cpu_axpy(size, alpha, gold_double.data(), incx, comp_double.data(), incx);
    double error = cpu_lange(norm_type, M, N, comp_double.data(), lda, work.data());
    if(gold_norm > 0)
        error /= gold_norm;

    return error;
}

template <typename T>
double norm_error_upperTr(char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* gold, T* comp)
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
double norm_error_lowerTr(char norm_type, rocblas_int M, rocblas_int N, rocblas_int lda, T* gold, T* comp)
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
    return cpu_lange(norm_type, m, n, A, lda, (S*)nullptr);
}
