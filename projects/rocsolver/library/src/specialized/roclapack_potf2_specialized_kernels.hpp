/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocsolver_run_specialized_kernels.hpp"
#include <algorithm>
#include <cmath>

ROCSOLVER_BEGIN_NAMESPACE

/**
 * indexing for packed storage
 * for upper triangular
 *
 * ---------------------------
 * 0 1 3
 *   2 4
 *     5
 * ---------------------------
 *
 **/

template <typename I>
__device__ static I idx_upper(I i, I j, I n)
{
    assert((0 <= i) && (i <= (n - 1)));
    assert((0 <= j) && (j <= (n - 1)));
    assert(i <= j);

    return (i + (j * (j + 1)) / 2);
}

/**
 * indexing for packed storage
 * for lower triangular
 *
 * ---------------------------
 * 0
 * 1      n
 * *      (n+1)
 * *
 * (n-1)  ...        n*(n+1)/2
 * ---------------------------
 **/
template <typename I>
__device__ static I idx_lower(I i, I j, I n)
{
    assert((0 <= i) && (i <= (n - 1)));
    assert((0 <= j) && (j <= (n - 1)));
    assert(i >= j);

    return ((i - j) + (j * (2 * n + 1 - j)) / 2);
}

/**
 * ------------------------------------------------------
 * Perform Cholesky factorization for small n by n matrix.
 * The function executes in a single thread block.
 * ------------------------------------------------------
**/
template <typename T, typename I, typename INFO>
__device__ static void potf2_simple(bool const is_upper, I const n, T* const A, INFO* const info)
{
    auto const lda = n;
    bool const is_lower = (!is_upper);

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;
    assert(hipBlockDim_z == 1);

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const j0_start = tid;
    auto const j0_inc = nthreads;

    if(is_lower)
    {
        // ---------------------------------------------------
        // [  l11     ]  * [ l11'   vl21' ]  =  [ a11       ]
        // [ vl21  L22]    [        L22' ]     [ va21, A22 ]
        //
        //
        //   assume l11 is scalar 1x1 matrix
        //
        //   (1) l11 * l11' = a11 =>  l11 = sqrt( abs(a11) ), scalar computation
        //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
        //   (3) L22 * L22' + vl21 * vl21' = A22
        //
        //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
        //   (3b) L22 * L22' = A22,   cholesky factorization, tail recursion
        // ---------------------------------------------------

        for(I kcol = 0; kcol < n; kcol++)
        {
            auto kk = idx_lower(kcol, kcol, lda);
            auto const akk = std::real(A[kk]);
            bool const isok = (akk > 0) && (std::isfinite(akk));
            if(!isok)
            {
                if(tid == 0)
                {
                    A[kk] = akk;
                    // Fortran 1-based index
                    if(*info == 0)
                        *info = kcol + 1;
                }
                break;
            }

            auto const lkk = std::sqrt(akk);
            if(tid == 0)
            {
                A[kk] = lkk;
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
            // ------------------------------------------------------------

            auto const conj_lkk = conj(lkk);
            for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
            {
                auto const j0k = idx_lower(j0, kcol, lda);

                A[j0k] = (A[j0k] / conj_lkk);
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
            //
            //   note: update lower triangular part
            // ------------------------------------------------------------

            for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
            {
                auto const vj = A[idx_lower(j, kcol, lda)];
                for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
                {
                    bool const lower_part = (i >= j);
                    if(lower_part)
                    {
                        auto const vi = A[idx_lower(i, kcol, lda)];
                        auto const ij = idx_lower(i, j, lda);

                        A[ij] = A[ij] - vi * conj(vj);
                    }
                }
            }

            __syncthreads();

        } // end for kcol
    }
    else
    {
        // --------------------------------------------------
        // [u11'        ] * [u11    vU12 ] = [ a11     vA12 ]
        // [vU12'   U22']   [       U22  ]   [ vA12'   A22  ]
        //
        // (1) u11' * u11 = a11 =?  u11 = sqrt( abs( a11 ) )
        // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
        //     or vU12 = vA12/u11'
        // (3) vU12' * vU12 + U22'*U22 = A22
        //
        // (3a) A22 = A22 - vU12' * vU12
        // (3b) U22' * U22 = A22,  cholesky factorization, tail recursion
        // --------------------------------------------------

        for(I kcol = 0; kcol < n; kcol++)
        {
            auto const kk = idx_upper(kcol, kcol, lda);
            auto const akk = std::real(A[kk]);
            bool const isok = (akk > 0) && (std::isfinite(akk));
            if(!isok)
            {
                if(tid == 0)
                {
                    A[kk] = akk;
                    // Fortran 1-based index
                    if(*info == 0)
                        *info = kcol + 1;
                }

                break;
            }

            auto const ukk = std::sqrt(akk);
            if(tid == 0)
            {
                A[kk] = ukk;
            }
            __syncthreads();

            // ----------------------------------------------
            // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
            // ----------------------------------------------
            for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
            {
                auto const kj0 = idx_upper(kcol, j0, lda);

                A[kj0] = A[kj0] / ukk;
            }

            __syncthreads();

            // -----------------------------
            // (3a) A22 = A22 - vU12' * vU12
            //
            // note: update upper triangular part
            // -----------------------------
            for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
            {
                auto const vj = A[idx_upper(kcol, j, lda)];
                for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
                {
                    bool const upper_part = (i <= j);
                    if(upper_part)
                    {
                        auto const vi = A[idx_upper(kcol, i, lda)];
                        auto const ij = idx_upper(i, j, lda);

                        A[ij] = A[ij] - conj(vi) * vj;
                    }
                }
            }

            __syncthreads();

        } // end for kcol
    }
}

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void potf2_kernel_small(const bool is_upper,
                                         const I n,
                                         U AA,
                                         const rocblas_stride shiftA,
                                         const I lda,
                                         const rocblas_stride strideA,
                                         INFO* const info)
{
    bool const is_lower = (!is_upper);

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;
    assert(hipBlockDim_z == 1);

    // --------------------------------
    // note hipGridDim_z == batch_count
    // --------------------------------
    auto const bid = hipBlockIdx_z;
    assert(AA != nullptr);
    assert(info != nullptr);

    T* const A = load_ptr_batch(AA, bid, shiftA, strideA);
    INFO* const info_bid = info + bid;

    assert(A != nullptr);

    // -----------------------------------------
    // assume n by n matrix will fit in LDS cache
    // -----------------------------------------
    extern __shared__ rocblas_int lsmem[];
    T* Ash = reinterpret_cast<T*>(lsmem);

    // --------------------------------------------------------
    // factoring Lower triangular matrix may be slightly faster
    // due to simpler index calculation down a column
    // --------------------------------------------------------
    bool const use_compute_lower = true;

    // ------------------------------------
    // copy n by n packed matrix into shared memory
    // ------------------------------------
    __syncthreads();

    if(is_lower)
    {
        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = j + i_start; i < n; i += i_inc)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const ij_packed = idx_lower(i, j, n);

                Ash[ij_packed] = A[ij];
            }
        }
    }
    else
    {
        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = i_start; i <= j; i += i_inc)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const ij_packed = (use_compute_lower) ? idx_lower(j, i, n) : idx_upper(i, j, n);

                auto const aij = A[ij];
                Ash[ij_packed] = (use_compute_lower) ? conj(aij) : aij;
            }
        }
    }

    __syncthreads();

    bool const is_up = (use_compute_lower) ? false : is_upper;
    potf2_simple<T>(is_up, n, Ash, info_bid);

    __syncthreads();

    // -------------------------------------
    // copy n by n packed matrix into global memory
    // -------------------------------------
    if(is_lower)
    {
        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = j + i_start; i < n; i += i_inc)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const ij_packed = idx_lower(i, j, n);

                A[ij] = Ash[ij_packed];
            }
        }
    }
    else
    {
        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = i_start; i <= j; i += i_inc)
            {
                auto const ij = i + j * static_cast<int64_t>(lda);
                auto const ij_packed = (use_compute_lower) ? idx_lower(j, i, n) : idx_upper(i, j, n);

                auto const aij_packed = Ash[ij_packed];
                A[ij] = (use_compute_lower) ? conj(aij_packed) : aij_packed;
            }
        }
    }

    __syncthreads();
}

/*************************************************************
    Launchers of specilized kernels
*************************************************************/

template <typename T, typename I, typename INFO, typename U>
rocblas_status potf2_run_small(rocblas_handle handle,
                               const rocblas_fill uplo,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               INFO* info,
                               const I batch_count)
{
    ROCSOLVER_ENTER("potf2_kernel_small", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    size_t lmemsize = sizeof(T) * (n * (n + 1)) / 2;

    bool const is_upper = (uplo == rocblas_fill_upper);
    ROCSOLVER_LAUNCH_KERNEL((potf2_kernel_small<T, I, INFO, U>), dim3(1, 1, batch_count),
                            dim3(BS2, BS2, 1), lmemsize, stream, is_upper, n, A, shiftA, lda,
                            strideA, info);

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_POTF2_SMALL(T, I, INFO, U)                                                       \
    template rocblas_status potf2_run_small<T, I, INFO, U>(                                          \
        rocblas_handle handle, const rocblas_fill uplo, const I n, U A, const rocblas_stride shiftA, \
        const I lda, const rocblas_stride strideA, INFO* info, const I batch_count)

ROCSOLVER_END_NAMESPACE
