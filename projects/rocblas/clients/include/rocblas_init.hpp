/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../library/src/include/rocblas_ostream.hpp"
#include "host_strided_batch_vector.hpp"
#include "host_vector.hpp"
#include "rocblas.h"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include <cinttypes>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>

//!
//! @brief enum to check for NaN initialization of the Input vector/matrix
//!
typedef enum rocblas_check_nan_init_
{
    // Alpha sets NaN
    rocblas_client_alpha_sets_nan,

    // Beta sets NaN
    rocblas_client_beta_sets_nan,

    //  Never set NaN
    rocblas_client_never_set_nan

} rocblas_check_nan_init;

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alternating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.

template <typename T>
void rocblas_init_matrix_alternating_sign(rocblas_check_matrix_type matrix_type,
                                          const char                uplo,
                                          T                         rand_gen(),
                                          host_vector<T>&           A,
                                          size_t                    M,
                                          size_t                    N,
                                          size_t                    lda,
                                          rocblas_stride            stride      = 0,
                                          int64_t                   batch_count = 1)
{
    if(matrix_type == rocblas_client_general_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value                  = rand_gen();
                    A[i + j * lda + b * stride] = (i ^ j) & 1 ? T(value) : T(negate(value));
                }
    }
    else if(matrix_type == rocblas_client_triangular_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : 0) : (j <= i ? rand_gen() : 0);
                    A[i + j * lda + b * stride] = (i ^ j) & 1 ? T(value) : T(negate(value));
                }
    }
}

template <typename U, typename T>
void rocblas_init_matrix_alternating_sign(rocblas_check_matrix_type matrix_type,
                                          const char                uplo,
                                          T                         rand_gen(),
                                          U&                        hA)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

        if(matrix_type == rocblas_client_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value     = rand_gen();
                    A[i + j * lda] = (i ^ j) & 1 ? T(value) : T(negate(value));
                }
        }
        else if(matrix_type == rocblas_client_triangular_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda] = (i ^ j) & 1 ? T(value) : T(negate(value));
                }
        }
    }
}

// Initialize vector so adjacent entries have alternating sign.
template <typename T>
void rocblas_init_vector_alternating_sign(T rand_gen(), T* x, int64_t N, int64_t incx)
{
    if(incx < 0)
        x -= (N - 1) * incx;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int64_t j = 0; j < N; ++j)
    {
        auto value  = rand_gen();
        x[j * incx] = j & 1 ? T(value) : T(negate(value));
    }
}

/* ============================================================================================ */
/*! \brief  matrix initialization: */
// Initialize matrix according to the matrix_types

template <typename T>
void rocblas_init_matrix(rocblas_check_matrix_type matrix_type,
                         const char                uplo,
                         T                         rand_gen(),
                         host_vector<T>&           A,
                         size_t                    M,
                         size_t                    N,
                         size_t                    lda,
                         rocblas_stride            stride      = 0,
                         int64_t                   batch_count = 1)
{
    if(matrix_type == rocblas_client_general_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda + b * stride] = rand_gen();
    }
    else if(matrix_type == rocblas_client_hermitian_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[b * stride + j + i * lda] = std::real(value);
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = conjugate(value);
                    }
                }
    }
    else if(matrix_type == rocblas_client_symmetric_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[b * stride + j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = value;
                    }
                }
    }
    else if(matrix_type == rocblas_client_triangular_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda + b * stride] = value;
                }
    }

    /*An n x n triangle matrix with random entries has a condition number that grows exponentially with n ("Condition numbers of random triangular matrices" D. Viswanath and L.N.Trefethen).
    Here we use a triangle matrix with random values that is strictly row and column diagonal dominant.
    This matrix should have a lower condition number. An alternative is to calculate the Cholesky factor of an SPD matrix with random values and make it diagonal dominant.
    This approach is not used because it is slow.*/

    else if(matrix_type == rocblas_client_diagonally_dominant_triangular_matrix)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value
                    = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                A[i + j * lda] = value;
            }

        const T multiplier = T(
            1.01); // Multiplying factor to slightly increase the base value of (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) dominant diagonal element. If tests fail and it seems that there are numerical stability problems, try increasing multiplier, it should decrease the condition number of the matrix and thereby avoid numerical stability issues.

        if(uplo == 'U') // rocblas_fill_upper
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; i++)
            {
                T abs_sum_off_diagonal_row
                    = T(0); //store absolute sum of entire row of the particular diagonal element
                T abs_sum_off_diagonal_col
                    = T(0); //store absolute sum of entire column of the particular diagonal element

                for(size_t j = i + 1; j < N; j++)
                    abs_sum_off_diagonal_row += rocblas_abs(A[i + j * lda]);
                for(size_t j = 0; j < i; j++)
                    abs_sum_off_diagonal_col += rocblas_abs(A[j + i * lda]);

                A[i + i * lda]
                    = (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) == T(0)
                          ? T(1)
                          : T((abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) * multiplier);
            }
        }
        else // rocblas_fill_lower
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; j++)
            {
                T abs_sum_off_diagonal_row
                    = T(0); //store absolute sum of entire row of the particular diagonal element
                T abs_sum_off_diagonal_col
                    = T(0); //store absolute sum of entire column of the particular diagonal element

                for(size_t i = j + 1; i < N; i++)
                    abs_sum_off_diagonal_col += rocblas_abs(A[i + j * lda]);

                for(size_t i = 0; i < j; i++)
                    abs_sum_off_diagonal_row += rocblas_abs(A[j + i * lda]);

                A[j + j * lda]
                    = (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) == T(0)
                          ? T(1)
                          : T((abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) * multiplier);
            }
        }
    }
}

template <typename U, typename T>
void rocblas_init_matrix(rocblas_check_matrix_type matrix_type,
                         const char                uplo,
                         T                         rand_gen(),
                         U&                        hA)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto*   A   = hA[batch_index];
        int64_t M   = hA.m();
        int64_t N   = hA.n();
        int64_t lda = hA.lda();
        if(matrix_type == rocblas_client_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                    A[i + j * lda] = rand_gen();
        }
        else if(matrix_type == rocblas_client_hermitian_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[j + i * lda] = std::real(value);
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = conjugate(value);
                    }
                }
        }
        else if(matrix_type == rocblas_client_symmetric_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = value;
                    }
                }
        }
        else if(matrix_type == rocblas_client_triangular_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda] = value;
                }
        }
        else if(matrix_type == rocblas_client_diagonally_dominant_triangular_matrix)
        {
            /*An n x n triangle matrix with random entries has a condition number that grows exponentially with n ("Condition numbers of random triangular matrices" D. Viswanath and L.N.Trefethen).
            Here we use a triangle matrix with random values that is strictly row and column diagonal dominant.
            This matrix should have a lower condition number. An alternative is to calculate the Cholesky factor of an SPD matrix with random values and make it diagonal dominant.
            This approach is not used because it is slow.*/

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda] = value;
                }

            const T multiplier = T(
                1.01); // Multiplying factor to slightly increase the base value of (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) dominant diagonal element. If tests fail and it seems that there are numerical stability problems, try increasing multiplier, it should decrease the condition number of the matrix and thereby avoid numerical stability issues.

            if(uplo == 'U') // rocblas_fill_upper
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(size_t i = 0; i < N; i++)
                {
                    T abs_sum_off_diagonal_row = T(
                        0); //store absolute sum of entire row of the particular diagonal element
                    T abs_sum_off_diagonal_col = T(
                        0); //store absolute sum of entire column of the particular diagonal element

                    for(size_t j = i + 1; j < N; j++)
                        abs_sum_off_diagonal_row += rocblas_abs(A[i + j * lda]);
                    for(size_t j = 0; j < i; j++)
                        abs_sum_off_diagonal_col += rocblas_abs(A[j + i * lda]);

                    A[i + i * lda] = (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) == T(0)
                                         ? T(1)
                                         : T((abs_sum_off_diagonal_row + abs_sum_off_diagonal_col)
                                             * multiplier);
                }
            }
            else // rocblas_fill_lower
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(size_t j = 0; j < N; j++)
                {
                    T abs_sum_off_diagonal_row = T(
                        0); //store absolute sum of entire row of the particular diagonal element
                    T abs_sum_off_diagonal_col = T(
                        0); //store absolute sum of entire column of the particular diagonal element

                    for(size_t i = j + 1; i < N; i++)
                        abs_sum_off_diagonal_col += rocblas_abs(A[i + j * lda]);

                    for(size_t i = 0; i < j; i++)
                        abs_sum_off_diagonal_row += rocblas_abs(A[j + i * lda]);

                    A[j + j * lda] = (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) == T(0)
                                         ? T(1)
                                         : T((abs_sum_off_diagonal_row + abs_sum_off_diagonal_col)
                                             * multiplier);
                }
            }
        }
    }
}

/*! \brief  vector initialization: */
// Initialize vectors with rand_int/hpl/NaN values

template <typename T>
void rocblas_init_vector(T rand_gen(), T* x, int64_t N, int64_t incx)
{
    if(incx < 0)
        x -= (N - 1) * incx;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int64_t j = 0; j < N; ++j)
        x[j * incx] = rand_gen();
}

/* ============================================================================================ */
/*! \brief  Trigonometric matrix initialization: */
// Initialize matrix with rand_int/hpl/NaN values

template <typename T>
void rocblas_init_matrix_trig(rocblas_check_matrix_type matrix_type,
                              const char                uplo,
                              host_vector<T>&           A,
                              size_t                    M,
                              size_t                    N,
                              size_t                    lda,
                              rocblas_stride            stride      = 0,
                              int64_t                   batch_count = 1,
                              bool                      seedReset   = false)
{
    if(matrix_type == rocblas_client_general_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda + b * stride]
                        = T(seedReset ? cos(i + j * M + b * M * N) : sin(i + j * M + b * M * N));
    }
    else if(matrix_type == rocblas_client_hermitian_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value
                        = T(seedReset ? cos(i + j * N + b * M * N) : sin(i + j * N + b * M * N));

                    if(i == j)
                        A[b * stride + j + i * lda] = std::real(value);
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = conjugate(value);
                    }
                }
    }
    else if(matrix_type == rocblas_client_symmetric_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value
                        = T(seedReset ? cos(i + j * N + b * M * N) : sin(i + j * N + b * M * N));
                    if(i == j)
                        A[b * stride + j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = value;
                    }
                }
    }
    else if(matrix_type == rocblas_client_triangular_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value = uplo == 'U' ? (j >= i ? T(seedReset ? cos(i + j * M + b * M * N)
                                                                     : sin(i + j * M + b * M * N))
                                                       : T(0))
                                             : (j <= i ? T(seedReset ? cos(i + j * M + b * M * N)
                                                                     : sin(i + j * M + b * M * N))
                                                       : T(0));
                    A[i + j * lda + b * stride] = value;
                }
    }
}

template <typename T, typename U>
void rocblas_init_matrix_trig(rocblas_check_matrix_type matrix_type,
                              const char                uplo,
                              U&                        hA,
                              bool                      seedReset = false)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

        if(matrix_type == rocblas_client_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda] = T(seedReset ? cos(i + j * M) : sin(i + j * M));
        }
        else if(matrix_type == rocblas_client_hermitian_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(seedReset ? cos(i + j * N) : sin(i + j * N));

                    if(i == j)
                        A[j + i * lda] = std::real(value);
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = conjugate(value);
                    }
                }
        }
        else if(matrix_type == rocblas_client_symmetric_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(seedReset ? cos(i + j * N) : sin(i + j * N));
                    if(i == j)
                        A[j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = value;
                    }
                }
        }
        else if(matrix_type == rocblas_client_triangular_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U'
                              ? (j >= i ? T(seedReset ? cos(i + j * M) : sin(i + j * M)) : T(0))
                              : (j <= i ? T(seedReset ? cos(i + j * M) : sin(i + j * M)) : T(0));
                    A[i + j * lda] = value;
                }
        }
    }
}

/*! \brief  Trigonometric vector initialization: */
// Initialize vector with rand_int/hpl/NaN values

template <typename T>
void rocblas_init_vector_trig(T* x, int64_t N, rocblas_stride incx, bool seedReset = false)
{
    if(incx < 0)
        x -= (N - 1) * incx;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int64_t j = 0; j < N; ++j)
        x[j * incx] = T(seedReset ? cos(j * N) : sin(j * N));
}

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize matrices with random values
template <typename T>
void rocblas_init(T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t b_idx = i_batch * stride;
        for(size_t j = 0; j < N; ++j)
        {
            size_t col_idx = b_idx + j * lda;
            if(M > 4)
                random_run_generator<T>(A + col_idx, M);
            else
            {
                for(size_t i = 0; i < M; ++i)
                    A[col_idx + i] = random_generator<T>();
            }
        }
    }
}

// Initialize matrices with random values
template <typename T>
inline void rocblas_init(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init(A.data(), M, N, lda, stride, batch_count);
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocblas_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocblas_nan_rng());
}

template <typename T>
void rocblas_init_nan_range(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocblas_nan_rng());
}

template <typename T>
void rocblas_init_nan_tri(
    bool upper, T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                T val                             = upper ? (j >= i ? T(rocblas_nan_rng()) : 0)
                                                          : (j <= i ? T(rocblas_nan_rng()) : 0);
                A[i + j * lda + i_batch * stride] = val;
            }
}

template <typename T>
void rocblas_init_nan(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_nan_rng());
}

template <typename T>
void rocblas_init_nan(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_nan(A.data(), M, N, lda, stride, batch_count);
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with Inf where appropriate */

template <typename T>
void rocblas_init_inf(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocblas_inf_rng());
}

template <typename T>
void rocblas_init_inf_range(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocblas_inf_rng());
}

template <typename T>
void rocblas_init_inf(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_inf_rng());
}

template <typename T>
void rocblas_init_inf(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    rocblas_init_inf(A.data(), M, N, lda, stride, batch_count);
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with zero */

template <typename T>
void rocblas_init_zero(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_zero_rng());
}

template <typename T>
void rocblas_init_zero_range(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocblas_zero_rng());
}

/* ============================================================================================ */
/*! \brief  Initialize an array with denorm values*/

template <typename T>
void rocblas_init_denorm(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_denorm_rng());
}

template <typename T>
void rocblas_init_denorm_range(T* A, size_t start_offset, size_t end_offset)
{
    for(size_t i = start_offset; i < end_offset; ++i)
        A[i] = T(rocblas_denorm_rng());
}

template <typename T, typename U>
void rocblas_init_identity(U& hA)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                if(i == j)
                    A[i + j * lda] = T(1);
                else
                    A[i + j * lda] = T(0);
    }
}

template <typename T, typename U>
void rocblas_init_matrix_zero(U& hA)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda] = T(0);
    }
}

template <typename T, typename U>
void rocblas_init_non_rep_bf16_vals(U& hA)
{
    const rocblas_half ieee_half_vals[4] = {2028, 2034, 2036, 2038};
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda] = T(ieee_half_vals[(i + j * lda) % 4]);
    }
}

template <typename T, typename U>
void rocblas_init_alt_impl_big(U& hA)
{
    const rocblas_half ieee_half_large(65280.0);
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda] = T(ieee_half_large);
    }
}

template <typename T>
void rocblas_init_alt_impl_one(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    T value(1.0f);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : negate(value);
            }
}

template <typename T>
void rocblas_init_impl_one(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    T value(1.0f);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                A[i + j * lda + i_batch * stride] = value;
            }
}

template <typename T>
void rocblas_init_alt_impl_big(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocblas_half ieee_half_large(65280.0);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_large);
}

// Initialize vector with random values
template <typename T>
inline void rocblas_init_alt_impl_big(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const rocblas_half ieee_half_large(65280.0);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_large);
}

template <typename T, typename U>
void rocblas_init_alt_impl_small(U& hA)
{
    //using a rocblas_half subnormal value
    const rocblas_half ieee_half_small(0.0000607967376708984375);
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda] = T(ieee_half_small);
    }
}

template <typename T>
void rocblas_init_alt_impl_small(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    //using a rocblas_half sunormal value
    const rocblas_half ieee_half_small(0.0000607967376708984375);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_small);
}

template <typename T>
void rocblas_init_alt_impl_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    //using a rocblas_half sunormal value
    const rocblas_half ieee_half_small(0.0000607967376708984375);
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(ieee_half_small);
}

/* ============================================================================================ */
/*! \brief  matrix matrix initialization: copies from A into same position in B */
template <typename T>
void rocblas_copy_matrix(const T* A,
                         T*       B,
                         size_t   M,
                         size_t   N,
                         size_t   lda,
                         size_t   ldb,
                         size_t   stridea     = 0,
                         size_t   strideb     = 0,
                         size_t   batch_count = 1)
{

    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
        size_t stride_offset_a = i_batch * stridea;
        size_t stride_offset_b = i_batch * strideb;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset_a = stride_offset_a + j * lda;
            size_t offset_b = stride_offset_b + j * ldb;
            memcpy(B + offset_b, A + offset_a, M * sizeof(T));
        }
    }
}

/* ============================================================================================ */
/*! \brief  matrix matrix initialization: copies from A into same position in B */
template <typename T>
void rocblas_copy_matrix(
    const T* const* A, T** B, size_t M, size_t N, size_t lda, size_t ldb, size_t batch_count = 1)
{

    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset_a = j * lda;
            size_t offset_b = j * ldb;
            memcpy(B[i_batch] + offset_b, A[i_batch] + offset_a, M * sizeof(T));
        }
    }
}
