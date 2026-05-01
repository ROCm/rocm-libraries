/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#define USE_SYTRS2
#ifdef USE_SYTRS2
#include "roclapack_sytrs2.hpp"
#endif

ROCSOLVER_BEGIN_NAMESPACE

#ifdef USE_SYTRS2
// ----------------------------------------
// simple heuristic to switch to use SYTRS2
// when nrhs is sufficiently large
// ----------------------------------------
template <typename T, typename I>
static inline bool use_sytrs2(I const n, I const nrhs, [[maybe_unused]] I const batch_count)
{
    return (nrhs >= ((n / 2) + (n % 2)));
}
#endif

template <typename T, typename I>
__device__ static T reduce_sum_shfl_wsize(I const wsize, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    if(wsize == 64)
    {
        val += shift_left(val, 32); // offset = 32
        val += shift_left(val, 16); // offset = 16
        val += shift_left(val, 8); // offset = 8
        val += shift_left(val, 4); // offset = 4
        val += shift_left(val, 2); // offset = 2
        val += shift_left(val, 1); // offset = 1
    }
    else if(wsize == 32)
    {
        val += shift_left(val, 16); // offset = 16
        val += shift_left(val, 8); // offset = 8
        val += shift_left(val, 4); // offset = 4
        val += shift_left(val, 2); // offset = 2
        val += shift_left(val, 1); // offset = 1
    }
    else
    {
        for(auto offset = wsize / 2; offset > 0; offset /= 2)
        {
            val += shift_left(val, offset);
        }
    }
    return val; // note: only thread 0 will return full sum
}

/** thread-block size for calling the sytrs kernel.
    (MAX_THDS sizes must be one of 128, 256, 512, or 1024) **/
#ifndef SYTRS_MAX_THDS
#define SYTRS_MAX_THDS 256
#endif

// ------------------------------------------------
// NOTE: intended for execution within single block
//
// launch as dim3( nbx, 1, batch_count), dim3( SYTRS_MAX_THDS,1,1)
//
// nrhs_arg  columns of matrix B are  spread across nbx thread blocks
// ------------------------------------------------

template <typename T, typename I, typename Istride, typename UA, typename UB>
ROCSOLVER_KERNEL void __launch_bounds__(SYTRS_MAX_THDS) sytrs_kernel(bool const use_upper,
                                                                     I const n,
                                                                     I const nrhs_arg,
                                                                     UA AA,
                                                                     Istride const shiftA,
                                                                     I const lda,
                                                                     Istride strideA,
                                                                     I* const ipivA,
                                                                     Istride const strideP,
                                                                     UB BB,
                                                                     Istride const shiftB,
                                                                     I const ldb_arg,
                                                                     Istride const strideB,
                                                                     I const batch_count,
                                                                     size_t const lds_size)
{
    // select batch instance
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const tx = threadIdx.x;
    I const ty = threadIdx.y;
    I const nx = blockDim.x;
    I const ny = blockDim.y;

    I const ij_start = tx + ty * nx;
    I const ij_inc = nx * ny;

    I const i_start = tx;
    I const i_inc = nx;

    I const j_start = ty;
    I const j_inc = ny;

    // ------------------------
    // Fortran 1-based indexing
    // ------------------------
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + ((j - 1) * static_cast<int64_t>(ld))); };

    // ------------------
    // C 0-based indexing
    // ------------------
    auto idx2D = [](auto i, auto j, auto ld) { return (i + (j * static_cast<int64_t>(ld))); };

    auto ceildiv = [](auto const n, auto const b) { return ((n <= 0) ? 0 : ((n - 1) / b + 1)); };

    I nrhs = nrhs_arg;
    Istride offsetB = 0;

    {
        // ----------------------------------------------
        // each thread block handles about nb_rhs columns
        // ----------------------------------------------
        I const nbx = gridDim.x;
        I const ibx = blockIdx.x;

        I const nb_rhs = ceildiv(nrhs_arg, nbx);
        I const rhs_start = ibx * nb_rhs;
        I const rhs_end = std::min(nrhs_arg, rhs_start + nb_rhs);

        offsetB = idx2D(0, rhs_start, ldb_arg);
        nrhs = rhs_end - rhs_start;
    }

    if((nrhs == 0) || (n == 0) || (batch_count == 0))
    {
        return;
    }

    auto swap = [](T& x, T& y) {
        auto const temp = x;
        x = y;
        y = temp;
    };

    T const one = 1;

    // -------------------------------
    // Compute rank-1 update
    //
    // C <- alpha * x * y^t + C
    // where
    // C is m by n
    // -------------------------------
    auto sytrs_geru = [=](I const m, I const n, T const alpha,

                          T* const x, I const incx,

                          T* const y, I const incy,

                          T* const C, I const ldc) {
        I ii_start = i_start;
        I ii_inc = i_inc;
        I jj_start = j_start;
        I jj_inc = j_inc;

        {
            // -----------------------------
            // optimization for special case
            // -----------------------------
            bool const use_m_loop = (m >= ij_inc) || (n == 1);
            bool const use_n_loop = (m == 1);

            if(use_m_loop)
            {
                jj_start = 0;
                jj_inc = 1;
                ii_start = ij_start;
                ii_inc = ij_inc;
            }
            else if(use_n_loop)
            {
                ii_start = 0;
                ii_inc = 1;
                jj_start = ij_start;
                jj_inc = ij_inc;
            }
        }

        __syncthreads();

        for(I j = 0 + jj_start; j < n; j += jj_inc)
        {
            T const yj = (incy == 1) ? y[j] : y[j * static_cast<int64_t>(incy)];

            for(I i = 0 + ii_start; i < m; i += ii_inc)
            {
                T const xi = (incx == 1) ? x[i] : x[i * static_cast<int64_t>(incx)];

                C[idx2D(i, j, ldc)] += xi * (yj * alpha);
            }
        }

        __syncthreads();
    };

    // ----------------
    // swap two vectors
    // ----------------
    auto sytrs_swap = [=](I const n, T* const x, I const incx, T* const y, I const incy) {
        __syncthreads();

        for(auto ij = 0 + ij_start; ij < n; ij += ij_inc)
        {
            auto const ix = (incx == 1) ? ij : ij * static_cast<int64_t>(incx);
            auto const iy = (incy == 1) ? ij : ij * static_cast<int64_t>(incy);

            swap(x[ix], y[iy]);
        }

        __syncthreads();
    };

    // -------------
    // scale a vector
    // -------------
    auto sytrs_scal = [=](I const n, T const alpha, T* const x, I const incx) {
        {
            T const one = 1;

            bool const has_work_to_do = (n >= 1) && (alpha != one);
            if(!has_work_to_do)
            {
                return;
            }
        }

        __syncthreads();

        for(auto ij = 0 + ij_start; ij < n; ij += ij_inc)
        {
            auto const ix = (incx == 1) ? ij : ij * static_cast<int64_t>(incx);

            x[ix] *= alpha;
        }

        __syncthreads();
    };

    // ---------------------------------------
    // compute matrix-vector multiply
    //         y <- alpha * A   * x + beta * y
    // or      y <- alpha * A^T * x + beta * y
    // or      y <- alpha * A^H * x + beta * y
    //
    // A is m by n matrix
    // ---------------------------------------
    auto sytrs_gemv = [=](char const trans, I const m, I const n,

                          T const alpha,

                          T* const A, I const lda,

                          T* const x, I const incx,

                          T const beta,

                          T* const y, I const incy) {
        bool const is_conj_trans = (trans == 'C') || (trans == 'c');
        bool const is_only_trans = (trans == 'T') || (trans == 't');
        bool const is_no_trans = (!is_conj_trans) && (!is_only_trans);

        I const lenx = (is_no_trans) ? n : m;
        I const leny = (is_no_trans) ? m : n;

        T const zero = 0;
        T const one = 1;

        // --------------
        // scale vector y
        // --------------
        if(beta != one)
        {
            for(I ij = 0 + ij_start; ij < leny; ij += ij_inc)
            {
                auto const iy = (incy == 1) ? ij : ij * static_cast<int64_t>(incy);
                y[iy] = (beta == zero) ? zero : (beta * y[iy]);
            }

            __syncthreads();
        }

        for(I j = 0 + j_start; j < leny; j += j_inc)
        {
            T ysum = zero;

            // ------------------------------
            // note execution in the same warp
            // in the for (i) loop
            // ------------------------------
            for(I i = 0 + i_start; i < lenx; i += i_inc)
            {
                T const xi = (incx == 1) ? x[i] : x[i * static_cast<int64_t>(incx)];
                T const aval = (is_no_trans) ? A[idx2D(j, i, lda)]
                    : (is_only_trans)        ? A[idx2D(i, j, lda)]
                                             : conj(A[idx2D(i, j, lda)]);

                ysum += aval * xi;
            }

            // -------------
            // sum reduction
            // -------------
            {
                auto const wsize = i_inc;
                __syncwarp();
                ysum = reduce_sum_shfl_wsize(wsize, ysum);
                __syncwarp();
            }

            if(i_start == 0)
            {
                auto const jy = (incy == 1) ? j : j * static_cast<int64_t>(incy);
                y[jy] += alpha * ysum;
            }
        }

        __syncthreads();
    };

    extern __shared__ std::byte ldsmem[];
    std::byte* pfree = &(ldsmem[0]);

    size_t const size_B_lds = sizeof(T) * n * nrhs;
    bool const use_B_lds = (size_B_lds <= lds_size);

    T* const B_lds = (use_B_lds) ? reinterpret_cast<T*>(pfree) : nullptr;

    if(use_B_lds)
    {
        pfree += size_B_lds;
    }

    // --------------------------------
    // Main loop over batch index "bid"
    // --------------------------------
    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // get array pointers
        T* const A_bid = load_ptr_batch<T>(AA, bid, shiftA, strideA);
        I* const ipiv_bid = ipivA + (bid * strideP);

        T* const B_bid = offsetB + load_ptr_batch<T>(BB, bid, shiftB, strideB);

        // --------------------
        // try to hold B in LDS
        // --------------------

        T* const B_p = (use_B_lds) ? B_lds : B_bid;
        I const ldb = (use_B_lds) ? n : ldb_arg;

        if(use_B_lds)
        {
            // ---------------
            // load B into LDS
            // ---------------

            I ii_start = i_start;
            I ii_inc = i_inc;
            I jj_start = j_start;
            I jj_inc = j_inc;

            if(nrhs == 1)
            {
                jj_start = 0;
                jj_inc = 1;

                ii_start = ij_start;
                ii_inc = ij_inc;
            }

            __syncthreads();

            for(I j = 0 + jj_start; j < nrhs; j += jj_inc)
            {
                for(I i = 0 + ii_start; i < n; i += ii_inc)
                {
                    auto const ij = i + j * ldb;
                    B_lds[ij] = B_bid[idx2D(i, j, ldb_arg)];
                }
            }

            __syncthreads();
        }

        auto A = [=](auto i, auto j) -> T& { return (A_bid[idx2F(i, j, lda)]); };

        auto B = [=](auto i, auto j) -> T& { return (B_p[idx2F(i, j, ldb)]); };

        auto ipiv = [=](auto i) -> I& { return (ipiv_bid[(i - 1)]); };

        // -----------------------------------
        // When each column is consistently accessed and modified by the
        // *same* thread, then there is consistency in the memory order
        // and the syncthread() is not needed.
        //
        // However, when nrhs is small, then most of the threads may be idle.
        // In that case, we'll need a different distribution of work to threads
        // and we'll need to call syncthread()
        // -----------------------------------

        // --------------------------------------------
        // A simple heuristic to determine when nrhs is sufficiently large
        // to use the algorithm for reducing number of syncthreads()
        // --------------------------------------------
        bool use_reduce_sync = (nrhs >= warpSize);

        //   -------------------------------------------------------------------------------------
        //   Matrix-vector multiply to update a row of matrix B
        //
        //   B( krow, j) += alpha * transpose( B( kstart:(kstart+m-1), j  ) ) * x( 1:m )
        //
        //   where j=1:nrhs
        //   -------------------------------------------------------------------------------------
        auto sytrs_gemv_reduce_sync = [=](I const m, I const kstart, T const* const x, I const incx,
                                          I const krow) {
            T const alpha = -one;

            for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
            {
                T dsum = 0;
                for(I i = 1; i <= m; i++)
                {
                    auto const ix = (incx == 1) ? (i - 1) : (i - 1) * static_cast<int64_t>(incx);
                    T const xi = x[ix];
                    dsum += B((kstart - 1) + i, j) * xi;
                }

                B(krow, j) += alpha * dsum;
            }
        };

        // ------------------------------------------------------------
        // rank-1 update to m by n sub-matrix of B
        //
        // B(krow:(krow+m-1), 1:nrhs)  += alpha * x(1:m) * y(1:nrhs)^t
        // ------------------------------------------------------------
        auto sytrs_geru_reduce_sync = [=](

                                          I const m,

                                          T const* const x, I const incx,

                                          I const row_y,

                                          I const krow) {
            T const alpha = -one;
            for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
            {
                auto const yj = B(row_y, j);

                for(I i = 1; i <= m; i++)
                {
                    auto const ix = (incx == 1) ? (i - 1) : (i - 1) * static_cast<int64_t>(incx);
                    auto const xi = x[ix];

                    B((krow - 1) + i, j) += xi * (yj * alpha);
                }
            }
        };
        // -----------------------------------------
        // swap row B(k,1:nrhs) and row B(kp,1:nrhs)
        // -----------------------------------------
        auto sytrs_swap_rows = [=](I const k, I const kp) {
            for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
            {
                swap(B(k, j), B(kp, j));
            }
        };

        // -----------------------
        // scale row krow of matrix B
        //
        // B(krow, 1:nrhs) *= alpha
        // -----------------------
        auto sytrs_scal_row = [=](T const alpha, I const krow) {
            for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
            {
                B(krow, j) *= alpha;
            }
        };

        if(use_upper)
        {
            //        Solve A*X = B, where A = U*D*U**T.
            //
            //        First solve U*D*X = B, overwriting B with X.
            //
            //        K is the main loop index,
            //        decreasing from N to 1 in steps of
            //        1 or 2, depending on the size of the diagonal blocks.

            I k = n;
            while(k >= 1)
            {
                if(ipiv(k) > 0)
                {
                    // --------------------------------
                    //           1 x 1 diagonal block
                    //
                    //           Interchange rows K and IPIV(K).
                    // --------------------------------
                    I const kp = ipiv(k);
                    if(kp != k)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_swap_rows(k, kp);
                        }
                        else
                        {
                            sytrs_swap(nrhs, &(B(k, 1)), ldb, &(B(kp, 1)), ldb);
                        }
                    }
                    // --------------------------------
                    //
                    //           Multiply by inv(U(K)),
                    //           where U(K) is the transformation
                    //           stored in column K of A.
                    //
                    // --------------------------------

                    if(use_reduce_sync)
                    {
                        sytrs_geru_reduce_sync(k - 1, &(A(1, k)), 1, k, 1);
                    }
                    else
                    {
                        sytrs_geru(k - 1, nrhs, -one, &(A(1, k)), 1, &(B(k, 1)), ldb, &(B(1, 1)),
                                   ldb);
                    }

                    // --------------------------------
                    //    Multiply by the inverse of the diagonal block.
                    // --------------------------------

                    if(use_reduce_sync)
                    {
                        auto const alpha = one / A(k, k);
                        sytrs_scal_row(alpha, k);
                    }
                    else
                    {
                        sytrs_scal(nrhs, one / A(k, k), &(B(k, 1)), ldb);
                    }

                    k = k - 1;
                }
                else
                {
                    // --------------------------------
                    //
                    //           2 x 2 diagonal block
                    //
                    //           Interchange rows K-1 and -IPIV(K).
                    //
                    // --------------------------------
                    I const kp = -ipiv(k);
                    if(kp != (k - 1))
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_swap_rows(k - 1, kp);
                        }
                        else
                        {
                            sytrs_swap(nrhs, &(B(k - 1, 1)), ldb, &(B(kp, 1)), ldb);
                        }
                    }
                    // --------------------------------
                    //
                    //      Multiply by inv(U(K)),
                    //      where U(K) is the transformation
                    //      stored in columns K-1 and K of A.
                    //
                    // --------------------------------

                    if(use_reduce_sync)
                    {
                        sytrs_geru_reduce_sync(k - 2, &(A(1, k)), 1, k, 1);
                    }
                    else
                    {
                        sytrs_geru(k - 2, nrhs, -one, &(A(1, k)), 1, &(B(k, 1)), ldb, &(B(1, 1)),
                                   ldb);
                    }

                    if(use_reduce_sync)
                    {
                        sytrs_geru_reduce_sync(k - 2, &(A(1, k - 1)), 1, k - 1, 1);
                    }
                    else
                    {
                        sytrs_geru(k - 2, nrhs, -one, &(A(1, k - 1)), 1, &(B(k - 1, 1)), ldb,
                                   &(B(1, 1)), ldb);
                    }
                    // --------------------------------
                    //           Multiply by the inverse of the diagonal block.
                    // --------------------------------
                    {
                        auto const akm1k = A(k - 1, k);
                        auto const akm1 = A(k - 1, k - 1) / akm1k;
                        auto const ak = A(k, k) / akm1k;
                        auto const denom = akm1 * ak - one;
                        for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
                        {
                            auto const bkm1 = B(k - 1, j) / akm1k;
                            auto const bk = B(k, j) / akm1k;
                            B(k - 1, j) = (ak * bkm1 - bk) / denom;
                            B(k, j) = (akm1 * bk - bkm1) / denom;
                        }
                    }

                    k = k - 2;
                }
            } // end while

            __syncthreads();

            // --------------------------------
            //        Next solve U**T *X = B, overwriting B with X.
            //
            //        K is the main loop index, increasing from 1 to N in steps of
            //        1 or 2, depending on the size of the diagonal blocks.
            // --------------------------------
            k = 1;
            while(k <= n)
            {
                if(ipiv(k) > 0)
                {
                    // --------------------------------
                    //      1 x 1 diagonal block
                    //
                    //      Multiply by inv(U**T(K)),
                    //      where U(K) is the transformation
                    //      stored in column K of A.
                    // --------------------------------

                    if(use_reduce_sync)
                    {
                        sytrs_gemv_reduce_sync(k - 1, 1, &(A(1, k)), 1, k);
                    }
                    else
                    {
                        sytrs_gemv('t', k - 1, nrhs, -one, &(B(1, 1)), ldb, &(A(1, k)), 1, one,
                                   &(B(k, 1)), ldb);
                    }

                    // --------------------------------
                    //           Interchange rows K and IPIV(K).
                    // --------------------------------
                    I const kp = ipiv(k);
                    if(kp != k)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_swap_rows(k, kp);
                        }
                        else
                        {
                            sytrs_swap(nrhs, &(B(k, 1)), ldb, &(B(kp, 1)), ldb);
                        }
                    }
                    k = k + 1;
                }
                else
                {
                    // --------------------------------
                    //      2 x 2 diagonal block
                    //
                    //      Multiply by inv(U**T(K+1)),
                    //      where U(K+1) is the transformation
                    //      stored in columns K and K+1 of A.
                    // --------------------------------

                    if(use_reduce_sync)
                    {
                        sytrs_gemv_reduce_sync(k - 1, 1, &(A(1, k)), 1, k);
                    }
                    else
                    {
                        sytrs_gemv('t', k - 1, nrhs, -one, &(B(1, 1)), ldb, &(A(1, k)), 1, one,
                                   &(B(k, 1)), ldb);
                    }

                    if(use_reduce_sync)
                    {
                        sytrs_gemv_reduce_sync(k - 1, 1, &(A(1, k + 1)), 1, k + 1);
                    }
                    else
                    {
                        sytrs_gemv('t', k - 1, nrhs, -one, &(B(1, 1)), ldb, &(A(1, k + 1)), 1, one,
                                   &(B(k + 1, 1)), ldb);
                    }
                    // --------------------------------
                    //           Interchange rows K and -IPIV(K).
                    // --------------------------------
                    I const kp = -ipiv(k);
                    if(kp != k)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_swap_rows(k, kp);
                        }
                        else
                        {
                            sytrs_swap(nrhs, &(B(k, 1)), ldb, &(B(kp, 1)), ldb);
                        }
                    }
                    k = k + 2;
                }
            } // end while
        }
        else
        {
            //  -----------------------------------
            //        Solve A*X = B, where A = L*D*L**T.
            //
            //        First solve L*D*X = B, overwriting B with X.
            //
            //        K is the main loop index, increasing from 1 to N in steps of
            //        1 or 2, depending on the size of the diagonal blocks.
            //  -----------------------------------

            I k = 1;
            while(k <= n)
            {
                if(ipiv(k) > 0)
                {
                    //  -----------------------------------
                    //           1 x 1 diagonal block
                    //
                    //           Interchange rows K and IPIV(K).
                    //  -----------------------------------
                    I const kp = ipiv(k);
                    if(kp != k)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_swap_rows(k, kp);
                        }
                        else
                        {
                            sytrs_swap(nrhs, &(B(k, 1)), ldb, &(B(kp, 1)), ldb);
                        }
                    }

                    //  -----------------------------------
                    //           Multiply by inv(L(K)), where L(K) is the transformation
                    //           stored in column K of A.
                    //  -----------------------------------
                    if(k < n)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_geru_reduce_sync(n - k, &(A(k + 1, k)), 1, k, k + 1);
                        }
                        else
                        {
                            sytrs_geru(n - k, nrhs, -one, &(A(k + 1, k)), 1, &(B(k, 1)), ldb,
                                       &(B(k + 1, 1)), ldb);
                        }
                    }

                    //  -----------------------------------
                    //           Multiply by the inverse of the diagonal block.
                    //  -----------------------------------

                    if(use_reduce_sync)
                    {
                        sytrs_scal_row(one / A(k, k), k);
                    }
                    else
                    {
                        sytrs_scal(nrhs, one / A(k, k), &(B(k, 1)), ldb);
                    }

                    k = k + 1;
                }
                else
                {
                    //  -----------------------------------
                    //           2 x 2 diagonal block
                    //
                    //           Interchange rows K+1 and -IPIV(K).
                    //  -----------------------------------
                    I const kp = -ipiv(k);
                    if(kp != (k + 1))
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_swap_rows(k + 1, kp);
                        }
                        else
                        {
                            sytrs_swap(nrhs, &(B(k + 1, 1)), ldb, &(B(kp, 1)), ldb);
                        }
                    }
                    //  -----------------------------------
                    //           Multiply by inv(L(K)), where L(K) is the transformation
                    //           stored in columns K and K+1 of A.
                    //  -----------------------------------
                    if(k < (n - 1))
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_geru_reduce_sync(n - k - 1, &(A(k + 2, k)), 1, k, k + 2);
                        }
                        else
                        {
                            sytrs_geru(n - k - 1, nrhs, -one, &(A(k + 2, k)), 1, &(B(k, 1)), ldb,
                                       &(B(k + 2, 1)), ldb);
                        }

                        if(use_reduce_sync)
                        {
                            sytrs_geru_reduce_sync(n - k - 1, &(A(k + 2, k + 1)), 1, k + 1, k + 2);
                        }
                        else
                        {
                            sytrs_geru(n - k - 1, nrhs, -one, &(A(k + 2, k + 1)), 1, &(B(k + 1, 1)),
                                       ldb, &(B(k + 2, 1)), ldb);
                        }
                    }
                    //  -----------------------------------
                    //           Multiply by the inverse of the diagonal block.
                    //  -----------------------------------
                    {
                        auto const akm1k = A(k + 1, k);
                        auto const akm1 = A(k, k) / akm1k;
                        auto const ak = A(k + 1, k + 1) / akm1k;
                        auto const denom = akm1 * ak - one;
                        for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
                        {
                            auto const bkm1 = B(k, j) / akm1k;
                            auto const bk = B(k + 1, j) / akm1k;
                            B(k, j) = (ak * bkm1 - bk) / denom;
                            B(k + 1, j) = (akm1 * bk - bkm1) / denom;
                        }
                    }
                    k = k + 2;
                }
            } // end while

            __syncthreads();

            //  -----------------------------------
            //        Next solve L**T *X = B, overwriting B with X.
            //
            //        K is the main loop index, decreasing from N to 1 in steps of
            //        1 or 2, depending on the size of the diagonal blocks.
            //  -----------------------------------

            k = n;
            while(k >= 1)
            {
                if(ipiv(k) > 0)
                {
                    //  -----------------------------------
                    //           1 x 1 diagonal block
                    //
                    //           Multiply by inv(L**T(K)), where L(K) is the transformation
                    //           stored in column K of A.
                    //  -----------------------------------
                    if(k < n)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_gemv_reduce_sync(n - k, k + 1, &(A(k + 1, k)), 1, k);
                        }
                        else
                        {
                            sytrs_gemv('t', n - k, nrhs, -one, &(B(k + 1, 1)), ldb, &(A(k + 1, k)),
                                       1, one, &(B(k, 1)), ldb);
                        }
                    }
                    //  -----------------------------------
                    //           Interchange rows K and IPIV(K).
                    //  -----------------------------------
                    I const kp = ipiv(k);
                    if(kp != k)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_swap_rows(k, kp);
                        }
                        else
                        {
                            sytrs_swap(nrhs, &(B(k, 1)), ldb, &(B(kp, 1)), ldb);
                        }
                    }
                    k = k - 1;
                }
                else
                {
                    //  -----------------------------------
                    //           2 x 2 diagonal block
                    //
                    //           Multiply by inv(L**T(K-1)), where L(K-1) is the transformation
                    //           stored in columns K-1 and K of A.
                    //  -----------------------------------
                    if(k < n)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_gemv_reduce_sync(n - k, k + 1, &(A(k + 1, k)), 1, k);
                        }
                        else
                        {
                            sytrs_gemv('t', n - k, nrhs, -one, &(B(k + 1, 1)), ldb, &(A(k + 1, k)),
                                       1, one, &(B(k, 1)), ldb);
                        }

                        if(use_reduce_sync)
                        {
                            sytrs_gemv_reduce_sync(n - k, k + 1, &(A(k + 1, k - 1)), 1, k - 1);
                        }
                        else
                        {
                            sytrs_gemv('t', n - k, nrhs, -one, &(B(k + 1, 1)), ldb,
                                       &(A(k + 1, k - 1)), 1, one, &(B(k - 1, 1)), ldb);
                        }
                    }
                    //  -----------------------------------
                    //           Interchange rows K and -IPIV(K).
                    //  -----------------------------------
                    I const kp = -ipiv(k);
                    if(kp != k)
                    {
                        if(use_reduce_sync)
                        {
                            sytrs_swap_rows(k, kp);
                        }
                        else
                        {
                            sytrs_swap(nrhs, &(B(k, 1)), ldb, &(B(kp, 1)), ldb);
                        }
                    }
                    k = k - 2;
                }
            } // end while
        }

        if(use_B_lds)
        {
            // ---------------------------------
            // write B from LDS to global memory
            // ---------------------------------
            I ii_start = i_start;
            I ii_inc = i_inc;

            I jj_start = j_start;
            I jj_inc = j_inc;

            if(nrhs == 1)
            {
                // --------------------------------
                // use all threads in row dimension
                // --------------------------------
                jj_start = 0;
                jj_inc = 1;

                ii_start = ij_start;
                ii_inc = ij_inc;
            }

            __syncthreads();

            for(I j = 0 + jj_start; j < nrhs; j += jj_inc)
            {
                for(I i = 0 + ii_start; i < n; i += ii_inc)
                {
                    auto const ij = i + j * ldb;
                    B_bid[idx2D(i, j, ldb_arg)] = B_lds[ij];
                }
            }

            __syncthreads();
        }

    } // end for (bid)
}
template <typename UA, typename UB, typename I>
rocblas_status rocsolver_sytrs_argCheck(rocblas_handle handle,
                                        rocblas_fill const uplo,
                                        I const n,
                                        I const nrhs,
                                        I const lda,
                                        I const ldb,
                                        UA A,
                                        UB B,
                                        I* const ipiv,
                                        I const batch_count = 1)
{
    // order is important for unit tests:

    // 0. check handle
    if(!handle)
        return rocblas_status_invalid_handle;

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && batch_count && !A) || (n && batch_count && !ipiv) || (nrhs && n && batch_count && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I>
rocblas_status rocsolver_sytrs_getMemorySize(rocblas_handle handle,
                                             I const n,
                                             I const nrhs,
                                             I const batch_count,
                                             I const lda,
                                             I const ldb,
                                             size_t* const p_size_work)
{
    // ---------------------------
    // request at least 1 byte to avoid possibly getting
    // a nullptr from rocblas_device_malloc()
    // ---------------------------

    size_t size_work = 1;

#ifdef USE_SYTRS2
    if(use_sytrs2<T>(n, nrhs, batch_count))
    {
        size_t size_sytrs2 = 0;
        ROCBLAS_CHECK(rocsolver_sytrs2_getMemorySize<T, I>(handle, n, nrhs, batch_count, lda, ldb,
                                                           &size_sytrs2));

        size_work += size_sytrs2;
    }
#endif

    *p_size_work = size_work;
    return rocblas_status_success;
}

template <typename T, typename I, typename Istride, typename UA, typename UB>
rocblas_status rocsolver_sytrs_template(rocblas_handle handle,
                                        rocblas_fill const uplo,
                                        I const n,
                                        I const nrhs,
                                        UA A,
                                        Istride const shiftA,
                                        I const lda,
                                        Istride const strideA,
                                        I* const ipiv,
                                        Istride const strideP,
                                        UB B,
                                        Istride const shiftB,
                                        I const ldb,
                                        Istride const strideB,
                                        I const batch_count,
                                        void* const work,
                                        size_t const size_work)

{
    // --------------------------------------
    // note no scratch storage needed for now but
    // may require scratch storage in future
    // --------------------------------------

    ROCSOLVER_ENTER("sytrs", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

#ifdef USE_SYTRS2
    if(use_sytrs2<T>(n, nrhs, batch_count))
    {
        return rocsolver_sytrs2_template<T, I>(handle, uplo, n, nrhs, A, shiftA, lda, strideA, ipiv,
                                               strideP, B, shiftB, ldb, strideB, batch_count, work,
                                               size_work);
    }
#endif

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    I const warp_size = props->warpSize;
    I const max_blocks = 64 * 1024 - 3;
    I const nbz = std::max(I(1), std::min(max_blocks, batch_count));

    // -----------------------------------------
    // each thread block handles about NB columns of B
    // -----------------------------------------
    auto ceildiv = [](auto const n, auto const b) { return ((n <= 0) ? 0 : ((n - 1) / b + 1)); };
    I const NB = SYTRS_MAX_THDS;
    I const nbx = ceildiv(nrhs, NB);

    I const lrhs = ceildiv(nrhs, nbx);

    I const nthreads = (lrhs >= SYTRS_MAX_THDS) ? SYTRS_MAX_THDS
        : (lrhs >= SYTRS_MAX_THDS / 2)          ? SYTRS_MAX_THDS / 2
        : (lrhs >= SYTRS_MAX_THDS / 4)          ? SYTRS_MAX_THDS / 4
                                                : warp_size;

    size_t const lds_size_max = props->sharedMemPerBlock;

    I const nx = warp_size;
    I const ny = std::max(I(1), nthreads / nx);

    // ------------------------------
    // check whether B can fit in LDS
    // ------------------------------
    size_t const size_B_lds = sizeof(T) * n * lrhs;
    bool const use_B_lds = (size_B_lds <= lds_size_max);
    I const lds_size = (use_B_lds) ? static_cast<I>(size_B_lds) : I(0);

    bool const use_upper = (uplo == rocblas_fill_upper);
    ROCSOLVER_LAUNCH_KERNEL(sytrs_kernel<T>, dim3(nbx, 1, nbz), dim3(nx, ny, 1), lds_size, stream,
                            use_upper, n, nrhs, A, shiftA, lda, strideA, ipiv, strideP, B, shiftB,
                            ldb, strideB, batch_count, lds_size);

    return rocblas_status_success;
}
ROCSOLVER_END_NAMESPACE
#undef USE_SYTRS2
#undef SYTRS_MAX_THDS
