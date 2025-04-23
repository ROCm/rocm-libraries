/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cinttypes>
#include <iostream>
#include <vector>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <rocblas/rocblas.h>

#include "rocblas_math.hpp"
#include "rocblas_random.hpp"

/* ============================================================================================
 */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same
// value

// Initialize vector with random values
template <typename T>
void rocblas_init(std::vector<T>& A,
                  size_t M,
                  size_t N,
                  size_t lda,
                  size_t stride = 0,
                  size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator<T>();
}

// Initialize vector with random values
template <typename T>
inline void
    rocblas_init(T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator<T>();
}

template <typename T>
void rocblas_init_sin(std::vector<T>& A,
                      size_t M,
                      size_t N,
                      size_t lda,
                      size_t stride = 0,
                      size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = sin(i + j * lda + i_batch * stride);
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
void rocblas_init_alternating_sign(std::vector<T>& A,
                                   size_t M,
                                   size_t N,
                                   size_t lda,
                                   size_t stride = 0,
                                   size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value = random_generator<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : negate(value);
            }
}

template <typename T>
void rocblas_init_alternating_sign(T* A,
                                   size_t M,
                                   size_t N,
                                   size_t lda,
                                   size_t stride = 0,
                                   size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value = random_generator<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : negate(value);
            }
}

template <typename T>
void rocblas_init_cos(std::vector<T>& A,
                      size_t M,
                      size_t N,
                      size_t lda,
                      size_t stride = 0,
                      size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = cos(i + j * lda + i_batch * stride);
}

/*! \brief  symmetric matrix initialization: */
// for real matrix only
template <typename T>
void rocblas_init_symmetric(std::vector<T>& A, size_t N, size_t lda)
{
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j <= i; ++j)
        {
            auto value = random_generator<T>();
            // Warning: It's undefined behavior to assign to the
            // same array element twice in same sequence point (i==j)
            A[j + i * lda] = value;
            A[i + j * lda] = value;
        }
}

/*! \brief  symmetric matrix initialization: */
template <typename T>
void rocblas_init_symmetric(T* A, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t b = 0; b < batch_count; ++b)
    {
        for(size_t i = 0; i < N; ++i)
            for(size_t j = 0; j <= i; ++j)
            {
                auto value = random_generator<T>();
                // Warning: It's undefined behavior to assign to the
                // same array element twice in same sequence point (i==j)
                A[b * stride + j + i * lda] = value;
                A[b * stride + i + j * lda] = value;
            }
    }
}

/*! \brief  symmetric matrix clear: */
template <typename T>
void rocblas_clear_symmetric(rocblas_fill uplo,
                             T* A,
                             size_t N,
                             size_t lda,
                             size_t stride = 0,
                             size_t batch_count = 1)
{
    for(size_t b = 0; b < batch_count; ++b)
    {
        for(size_t i = 0; i < N; ++i)
            for(size_t j = i + 1; j < N; ++j)
            {
                if(uplo == rocblas_fill_upper)
                    A[b * stride + j + i * lda] = 0; // clear lower
                else
                    A[b * stride + i + j * lda] = 0; // clear upper
            }
    }
}

/*! \brief  hermitian matrix initialization: */
// for complex matrix only, the real/imag part would be initialized with the
// same value except the diagonal elment must be real
template <typename T>
void rocblas_init_hermitian(std::vector<T>& A, size_t N, size_t lda)
{
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j <= i; ++j)
        {
            auto value = random_generator<T>();
            A[j + i * lda] = value;
            value.y = (i == j) ? 0 : negate(value.y);
            A[i + j * lda] = value;
        }
}

// Initialize vector with HPL-like random values
template <typename T>
void rocblas_init_hpl(std::vector<T>& A,
                      size_t M,
                      size_t N,
                      size_t lda,
                      size_t stride = 0,
                      size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

/* ============================================================================================
 */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
void rocblas_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocblas_nan_rng());
}

template <typename T>
void rocblas_init_nan(std::vector<T>& A,
                      size_t M,
                      size_t N,
                      size_t lda,
                      size_t stride = 0,
                      size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_nan_rng());
}

/* ============================================================================================
 */
/*! \brief  Packs strided_batched matricies into groups of 4 in N */

template <typename T>
void rocblas_packInt8(std::vector<T>& A, size_t M, size_t N, size_t batch_count, size_t lda, size_t stride_a)
{
    if(N % 4 != 0)
        fmt::print(stderr, "ERROR: dimension must be a multiple of 4 in order to pack\n");

    std::vector<T> temp(A);
    for(size_t count = 0; count < batch_count; count++)
        for(size_t colBase = 0; colBase < N; colBase += 4)
            for(size_t row = 0; row < lda; row++)
                for(size_t colOffset = 0; colOffset < 4; colOffset++)
                    A[(colBase * lda + 4 * row) + colOffset + (stride_a * count)]
                        = temp[(colBase + colOffset) * lda + row + (stride_a * count)];
}

/* ============================================================================================
 */
/*! \brief  Packs matricies into groups of 4 in N */
template <typename T>
void rocblas_packInt8(std::vector<T>& A, size_t M, size_t N, size_t lda)
{
    /* Assumes original matrix provided in column major order, where N is a
 multiple of 4

      ---------- N ----------
 |  | 00 05 10 15 20 25 30 35      |00 05 10 15|20 25 30 35|
 |  | 01 06 11 16 21 26 31 36      |01 06 11 16|21 26 31 36|
 l  M 02 07 12 17 22 27 32 37  --> |02 07 12 17|22 27 32 37|
 d  | 03 08 13 18 23 28 33 38      |03 08 13 18|23 28 33 38|
 a  | 04 09 14 19 24 29 34 39      |04 09 14 19|24 29 34 39|
 |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|
 |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|

   Input :  00 01 02 03 04 ** ** 05   ...  38 39 ** **
   Output:  00 05 10 15 01 06 11 16   ...  ** ** ** **

 */

    //  call general code with batch_count = 1 and stride_a = 0
    rocblas_packInt8(A, M, N, 1, lda, 0);
}

/* ============================================================================================
 */
/*! \brief  matrix matrix initialization: copies from A into same position in B
 */
template <typename T>
void rocblas_copy_matrix(const T* A,
                         T* B,
                         size_t M,
                         size_t N,
                         size_t lda,
                         size_t ldb,
                         size_t stridea = 0,
                         size_t strideb = 0,
                         size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                B[i + j * ldb + i_batch * strideb] = A[i + j * lda + i_batch * stridea];
}

/** rocsolver_diagonal_mode enum is used to define the type of diagonal when
    initializing random matrices:
    random: diagonal elements could be anything
    nonzero: diagonal elements could be any nonzero value
    unit: diagonal elements are all equal 1 **/
typedef enum rocsolver_diagonal_mode_
{
    rocsolver_diagonal_mode_random,
    rocsolver_diagonal_mode_nonzero,
    rocsolver_diagonal_mode_unit
} rocsolver_diagonal_mode;

/** RANDOM_SPARSE_MATRIX generates a sparse matrix with random non-zero positions/values.
    The matrix could be lower/upper triangular depending on the value of fill, and the
    elements in the diagonal could be all 1, non-zero or random depending on the value of diag.
    (TODO: The number and positions of the non-zeros are chosen randomly per
    row depending on the matrix shape, and directly in csr format. It might be good
    to explore other methods such as generating a random permutation of nnz positions in [0, n*n)
    and mapping it to csr format, for example.) **/
template <typename T>
void random_sparse_matrix(rocblas_int n,
                          rocblas_int nnzA,
                          rocblas_int* ptrA,
                          rocblas_int* indA,
                          T* valA,
                          rocblas_fill fill,
                          rocsolver_diagonal_mode diag)
{
    // auxiliary variables
    rocblas_int nnz, nn, pp, p, x, m, op, in;
    std::vector<rocblas_int> ops(n);
    std::vector<rocblas_int> z(n);
    bool randomdiag = (diag == rocsolver_diagonal_mode_random);
    bool unitdiag = (diag == rocsolver_diagonal_mode_unit);
    bool fullmatrix = (fill == rocblas_fill_full);
    bool lowertriang = (fill == rocblas_fill_lower);
    bool uppertriang = (fill == rocblas_fill_upper);

    // seed random generator
    rocblas_seedrand();

    ///////////////////////
    /// Generates ptrA ////
    ///////////////////////

    // initialize ptrA
    nnz = (!randomdiag) ? n : 0;
    for(rocblas_int j = 0; j <= n; ++j)
        ptrA[j] = (!randomdiag) ? j : 0;

    while(nnz < nnzA)
    {
        // for each row in matrix
        for(rocblas_int i = 0; i < n; ++i)
        {
            // set the number of non-zeros
            // op is the max number of non-zeros in a row
            op = (fullmatrix) ? n : (lowertriang) ? i + 1 : n - i;
            if(nnz < nnzA)
            {
                nn = random_generator<rocblas_int>(0, op + ptrA[i] - ptrA[i + 1]);
                if(nn > 0)
                    nn = 1; // take only one non-zero at a time to ensure good distribution
                nnz += nn;
            }
            else
                nn = 0;

            // update ptrA
            for(rocblas_int j = i + 1; j <= n; ++j)
                ptrA[j] += nn;
        }
    }

    /////////////////////////////
    // Generates indA and valA //
    /////////////////////////////

    // random non-zero values
    for(rocblas_int i = 0; i < nnzA; ++i)
        valA[i] = random_generator<T>(1, 10);

    // for each row in matrix
    for(rocblas_int i = 0; i < n; ++i)
    {
        nn = ptrA[i + 1] - ptrA[i];

        // if there are non-zero entries, then:
        if(nn > 0)
        {
            // determine possible positions of non-zeros
            if(fullmatrix)
            {
                // full matrix
                for(rocblas_int j = 0; j < n; ++j)
                    ops[j] = j;
            }
            else if(lowertriang)
            {
                // lower triangular
                for(rocblas_int j = 0; j <= i; ++j)
                    ops[j] = j;
            }
            else
            {
                // upper triangular
                for(rocblas_int j = i; j < n; ++j)
                    ops[j - i] = j;
            }

            // if no random diag, then make sure the position in diagonal is non-zero
            in = 0;
            if(!randomdiag)
            {
                z[0] = i;
                op = uppertriang ? 0 : i;
                ops[op] = -1;
                in = 1;
            }

            // choose the other non-zero positions
            // op is the max number of non-zeros in a row
            op = (fullmatrix) ? n : (lowertriang) ? i + 1 : n - i;
            for(rocblas_int j = in; j < nn; ++j)
            {
                pp = random_generator<rocblas_int>(0, op - 1);
                p = pp;
                x = ops[p];
                while(x < 0)
                {
                    pp++;
                    p = pp % op;
                    x = ops[p];
                }
                ops[p] = -1;
                z[j] = x;
            }

            // order non-zero positions in increasing order
            for(rocblas_int j = 0; j < nn - 1; ++j)
            {
                m = j;
                p = z[j];
                for(rocblas_int k = j + 1; k < nn; ++k)
                {
                    if(z[k] < p)
                    {
                        m = k;
                        p = z[k];
                    }
                }
                if(m != j)
                {
                    z[m] = z[j];
                    z[j] = p;
                }
            }

            // update indA and valA if necessary
            for(rocblas_int j = 0; j < nn; ++j)
            {
                indA[ptrA[i] + j] = z[j];
                if(unitdiag && z[j] == i)
                    valA[ptrA[i] + j] = 1;
            }
        }
    }
}

/** CPU_SUMLU Computes the bunddle matrix T = L - I + U given lower and upper
    factors L and U. **/
template <typename T>
void cpu_sumlu(const rocblas_int n,
               rocblas_int* ptrL,
               rocblas_int* indL,
               T* valL,
               rocblas_int* ptrU,
               rocblas_int* indU,
               T* valU,
               rocblas_int* ptrT,
               rocblas_int* indT,
               T* valT)
{
    // generate ptrT
    for(rocblas_int i = 0; i <= n; ++i)
        ptrT[i] = ptrL[i] + ptrU[i] - i;

    // generate indT and valT
    rocblas_int p = 0;
    rocblas_int nzL, nzU, iL, iU;
    for(rocblas_int i = 0; i < n; ++i)
    {
        iL = ptrL[i];
        iU = ptrU[i];
        nzL = ptrL[i + 1] - iL - 1;
        nzU = ptrU[i + 1] - iU;

        // insert lower part - I
        for(rocblas_int j = 0; j < nzL; ++j)
        {
            indT[p] = indL[iL + j];
            valT[p] = valL[iL + j];
            p++;
        }

        // insert upper part
        for(rocblas_int j = 0; j < nzU; ++j)
        {
            indT[p] = indU[iU + j];
            valT[p] = valU[iU + j];
            p++;
        }
    }
}
