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

#include "roclapack_syconv.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*

===============================================================
sytrs2 is not intended for inclusion in the public API.
It exists to provide an alternative, sometimes faster, implementation of sytrs.
==============================================================
*/

#ifndef SYTRS1_MAX_THDS
#define SYTRS1_MAX_THDS 256
#endif

// -------------------------------------------------
// Apply the 1x1 or 2x2 diagonal blocks to matrix B
//
// Matrix B is partitioned into column panels and
// each column panel is handled by a single thread block.
// -------------------------------------------------
template <typename T, typename I, typename UA, typename UB, typename Istride>
static __global__
    __launch_bounds__(SYTRS1_MAX_THDS) void apply_diag_block_kernel(bool const is_forward,
                                                                    I const n,
                                                                    I const nrhs_arg,
                                                                    UA A_arg,
                                                                    Istride const shiftA,
                                                                    I const lda,
                                                                    Istride strideA,
                                                                    I* const ipiv_arg,
                                                                    Istride const strideP,
                                                                    T* const E_arg,
                                                                    Istride const strideE,
                                                                    UB B_arg,
                                                                    Istride const shiftB,
                                                                    I const ldb,
                                                                    Istride strideB,
                                                                    I const batch_count)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;

    I const ij_inc = (blockDim.x * blockDim.y) * blockDim.z;
    I const ij_start
        = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    T const one = 1;

    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto ceildiv = [](auto n, auto b) { return ((n <= 0) ? 0 : ((n - 1) / b) + 1); };

    I const nb = ceildiv(nrhs_arg, nbx);

    I const col_start = ibx * nb;
    I const col_end = std::min(nrhs_arg, col_start + nb);

    // ------------------------------------------------------
    // NOTE: nrhs is the local number of columns processed by
    // this thread block
    // ------------------------------------------------------
    I const nrhs = (col_end - col_start);
    if((nrhs == 0) || (n == 0) || (batch_count == 0))
    {
        return;
    }

    Istride const offsetB = idx2D(0, col_start, ldb);

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A_bid = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);

        T* const B_bid = load_ptr_batch<T>(B_arg, bid, shiftB, strideB);
        T* const B_with_offset = B_bid + offsetB;

        T const* const E_bid = E_arg + bid * strideE;

        I const* const ipiv_bid = ipiv_arg + bid * strideP;

        // -------------------------------------------------------
        // Use 1-based indexing to match Fortran/matlab convention
        // -------------------------------------------------------
        auto A = [=](I const i, I const j) -> T { return (A_bid[idx2F(i, j, lda)]); };

        auto B = [=](I const i, I const j) -> T& { return (B_with_offset[idx2F(i, j, ldb)]); };

        auto E = [=](I const i) -> T { return (E_bid[(i - 1)]); };

        auto ipiv = [=](I const i) -> I { return (ipiv_bid[(i - 1)]); };

        // ---------------------------
        // scale row of B
        // B( krow, 1:nrhs) *= alpha
        // ---------------------------
        auto scale_row = [=](I const krow, T const alpha) {
            for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
            {
                B(krow, j) *= alpha;
            }
        };

        if(is_forward)
        {
            I i = 1;
            while(i <= n)
            {
                if(ipiv(i) > 0)
                {
                    auto const alpha = one / A(i, i);
                    I krow{};

                    scale_row(krow = i, alpha);
                }
                else
                {
                    auto const akm1k = E(i);
                    auto const akm1 = A(i, i) / akm1k;
                    auto const ak = A(i + 1, i + 1) / akm1k;
                    auto const denom = akm1 * ak - one;

                    for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
                    {
                        auto const bkm1 = B(i, j) / akm1k;
                        auto const bk = B(i + 1, j) / akm1k;
                        B(i, j) = (ak * bkm1 - bk) / denom;
                        B(i + 1, j) = (akm1 * bk - bkm1) / denom;
                    }

                    i = i + 1;
                }
                i = i + 1;
            } // end while
        }
        else
        {
            // ----------------------
            // apply in reverse order
            // ----------------------
            I i = n;
            while(i >= 1)
            {
                if(ipiv(i) > 0)
                {
                    auto const alpha = one / A(i, i);
                    I krow{};
                    scale_row(krow = i, alpha);
                }
                else if(i > 1)
                {
                    if(ipiv(i - 1) == ipiv(i))
                    {
                        auto const akm1k = E(i);
                        auto const akm1 = A(i - 1, i - 1) / akm1k;
                        auto const ak = A(i, i) / akm1k;
                        auto const denom = akm1 * ak - one;

                        for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
                        {
                            auto const bkm1 = B(i - 1, j) / akm1k;
                            auto const bk = B(i, j) / akm1k;
                            B(i - 1, j) = (ak * bkm1 - bk) / denom;
                            B(i, j) = (akm1 * bk - bkm1) / denom;
                        }

                        i = i - 1;
                    }
                }

                i = i - 1;
            } // end while do
        } // if (is_forward)

        __syncthreads();
    } // end for bid
}

template <typename T, typename I, typename UA, typename UB, typename Istride>
static rocblas_status apply_diag_block(rocblas_handle handle,
                                       bool const is_forward,
                                       I const n,
                                       I const nrhs_arg,
                                       UA A_arg,
                                       Istride const shiftA,
                                       I const lda,
                                       Istride strideA,
                                       I* const ipiv_arg,
                                       Istride const strideP,
                                       T* const E_arg,
                                       Istride const strideE,
                                       UB B_arg,
                                       Istride const shiftB,
                                       I const ldb,
                                       Istride strideB,
                                       I const batch_count)
{
    if((n == 0) || (batch_count == 0) || (nrhs_arg == 0))
    {
        return rocblas_status_success;
    }

    auto ceildiv = [](auto n, auto b) { return ((n <= 0) ? 0 : (n - 1) / b + 1); };

    I const max_blocks = 64 * 1024 - 3;
    I const nbz = std::min(max_blocks, batch_count);

    I const NB = SYTRS1_MAX_THDS;
    I const nbx = std::max(I(1), std::min(max_blocks, ceildiv(nrhs_arg, NB)));
    I const nby = 1;

    I const lrhs = ceildiv(nrhs_arg, nbx);
    I const nx = std::min(lrhs, I(SYTRS1_MAX_THDS));
    I const ny = 1;
    I const nz = 1;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCSOLVER_LAUNCH_KERNEL((apply_diag_block_kernel<T, I, UA, UB, Istride>), dim3(nbx, nby, nbz),
                            dim3(nx, ny, nz), 0, stream, is_forward, n, nrhs_arg, A_arg, shiftA,
                            lda, strideA, ipiv_arg, strideP, E_arg, strideE, B_arg, shiftB, ldb,
                            strideB, batch_count);

    return rocblas_status_success;
}

// -------------------------------------------------
// Apply the row pivoting to matrix B
//
// Matrix B is partitioned into column panels and
// each column panel is handled by a single thread block.
// -------------------------------------------------
template <typename T, typename I, typename UB, typename Istride>
static __global__
    __launch_bounds__(SYTRS1_MAX_THDS) void apply_pivot_upper_kernel(bool const is_forward,
                                                                     I const n,
                                                                     I const nrhs_arg,
                                                                     UB B_arg,
                                                                     Istride const shiftB,
                                                                     I const ldb,
                                                                     Istride const strideB,
                                                                     I* const ipiv_arg,
                                                                     Istride const strideP,
                                                                     I const batch_count)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;

    I const ij_inc = (blockDim.x * blockDim.y) * blockDim.z;
    I const ij_start
        = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    auto ceildiv = [](auto const n, auto const b) { return ((n <= 0) ? 0 : ((n - 1) / b + 1)); };

    I const nb = ceildiv(nrhs_arg, nbx);

    I const jstart = ibx * nb;
    I const jend = std::min(nrhs_arg, jstart + nb);
    I const nrhs = (jend - jstart);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto const offsetB = idx2D(0, jstart, ldb);

    // -------------------------------------------
    // NOTE: each thread block swap nrhs columns locally
    // so check value of nrhs
    // -------------------------------------------
    if((nrhs == 0) || (n == 0) || (batch_count == 0))
    {
        return;
    }

    // Fortran 1-based index value
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto swap = [](T& x, T& y) {
        auto const temp = x;
        x = y;
        y = temp;
    };

    for(auto bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const B_bid = load_ptr_batch<T>(B_arg, bid, shiftB, strideB);
        T* const B_with_offset = B_bid + offsetB;

        I const* const ipiv_bid = ipiv_arg + bid * strideP;

        // ---------------------------------------------------------
        // Use 1-based indexing to be compatible with Fortran/matlab
        // ---------------------------------------------------------

        auto B = [=](auto i, auto j) -> T& { return (B_with_offset[idx2F(i, j, ldb)]); };

        auto ipiv = [=](auto i) { return (ipiv_bid[(i - 1)]); };

        // --------------------------------------------
        // No syncthreads() is needed since each column
        // is consistently swapped by the same thread
        // --------------------------------------------

        auto swap_rows = [=](I const k, I const kp) {
            for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
            {
                swap(B(k, j), B(kp, j));
            }
        };

        if(!is_forward)
        {
            // -----------------------------
            // apply pivots in reverse order
            // -----------------------------

            //         k=n
            //         do while ( k .ge. 1 )
            //          if( ipiv( k ).gt.0 ) then
            // *           1 x 1 diagonal block
            // *           interchange rows k and ipiv(k).
            //             kp = ipiv( k )
            //             if( kp.ne.k )
            //      $         call zswap( nrhs, b( k, 1 ), ldb, b( kp, 1 ), ldb )
            //             k=k-1
            //          else
            // *           2 x 2 diagonal block
            // *           interchange rows k-1 and -ipiv(k).
            //             kp = -ipiv( k )
            //             if( kp.eq.-ipiv( k-1 ) )
            //      $         call zswap( nrhs, b( k-1, 1 ), ldb, b( kp, 1 ), ldb )
            //             k=k-2
            //          end if
            //         end do

            I k = n;
            while(k >= 1)
            {
                bool const use_1x1_block = (ipiv(k) > 0);
                if(use_1x1_block)
                {
                    // ------------------------------------------
                    //             1 x 1 diagonal block
                    //             interchange rows k and ipiv(k).
                    // ------------------------------------------
                    I const kp = ipiv(k);
                    if(kp != k)
                    {
                        swap_rows(k, kp);
                    }
                    k = k - 1;
                }
                else
                {
                    // ----------------------------------------------
                    //             2 x 2 diagonal block
                    //             interchange rows k-1 and -ipiv(k).
                    // ----------------------------------------------
                    I const kp = -ipiv(k);
                    if(kp == (-ipiv(k - 1)))
                    {
                        swap_rows(k - 1, kp);
                    }
                    k = k - 2;
                }
            } // end while
        }
        else
        {
            // -----------------------------
            // apply pivots in forward order
            // -----------------------------

            //        k=1
            //        do while ( k .le. n )
            //         if( ipiv( k ).gt.0 ) then
            //*           1 x 1 diagonal block
            //*           interchange rows k and ipiv(k).
            //            kp = ipiv( k )
            //            if( kp.ne.k )
            //     $         call zswap( nrhs, b( k, 1 ), ldb, b( kp, 1 ), ldb )
            //            k=k+1
            //         else
            //*           2 x 2 diagonal block
            //*           interchange rows k-1 and -ipiv(k).
            //            kp = -ipiv( k )
            //            if( k .lt. n .and. kp.eq.-ipiv( k+1 ) )
            //     $         call zswap( nrhs, b( k, 1 ), ldb, b( kp, 1 ), ldb )
            //            k=k+2
            //         endif
            //        end do
            I k = 1;
            while(k <= n)
            {
                bool const use_1x1_block = (ipiv(k) > 0);
                if(use_1x1_block)
                {
                    //             ------------------------------
                    //             1 x 1 diagonal block
                    //             interchange rows k and ipiv(k).
                    //             ------------------------------
                    I const kp = ipiv(k);
                    if(kp != k)
                    {
                        swap_rows(k, kp);
                    }
                    k = k + 1;
                }
                else
                {
                    //            ---------------------------------
                    //            2 x 2 diagonal block
                    //            interchange rows k-1 and -ipiv(k).
                    //            ---------------------------------
                    I const kp = -ipiv(k);
                    if((k < n) && (kp == (-ipiv(k + 1))))
                    {
                        // call_swap(nrhs, &(B(k, 1)), ldb, &(B(kp, 1)), ldb);
                        swap_rows(k, kp);
                    }
                    k = k + 2;
                }
            } // end while
        }

        __syncthreads();
    } // end for bid
}

template <typename T, typename I, typename UB, typename Istride>
static rocblas_status apply_pivot_upper(rocblas_handle handle,
                                        bool const is_forward,
                                        I const n,
                                        I const nrhs_arg,
                                        UB B,
                                        Istride const shiftB,
                                        I const ldb,
                                        Istride const strideB,
                                        I* const ipiv_arg,
                                        Istride const strideP,
                                        I const batch_count)
{
    if((n == 0) || (nrhs_arg == 0) || (batch_count == 0))
    {
        return rocblas_status_success;
    }

    auto ceildiv = [](auto const n, auto const b) { return (n <= 0) ? 0 : (((n - 1) / b + 1)); };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const max_blocks = 64 * 1024 - 3;

    I const NB = SYTRS1_MAX_THDS;
    I const nbx = std::max(I(1), std::min(max_blocks, ceildiv(nrhs_arg, NB)));
    I const nby = 1;
    I const nbz = std::min(max_blocks, batch_count);

    I const lrhs = ceildiv(nrhs_arg, nbx);
    I const nx = std::min(lrhs, I(SYTRS1_MAX_THDS));
    I const ny = 1;
    I const nz = 1;

    ROCSOLVER_LAUNCH_KERNEL((apply_pivot_upper_kernel<T, I, UB, Istride>), dim3(nbx, nby, nbz),
                            dim3(nx, ny, nz), 0, stream, is_forward, n, nrhs_arg, B, shiftB, ldb,
                            strideB, ipiv_arg, strideP, batch_count);

    return rocblas_status_success;
}

// -------------------------------------------------
// Apply the row pivoting to matrix B
//
// Matrix B is partitioned into column panels and
// each column panel is handled by a single thread block.
// -------------------------------------------------
template <typename T, typename I, typename UB, typename Istride>
static __global__
    __launch_bounds__(SYTRS1_MAX_THDS) void apply_pivot_lower_kernel(bool const is_forward,
                                                                     I const n,
                                                                     I const nrhs_arg,
                                                                     UB B_arg,
                                                                     Istride const shiftB,
                                                                     I const ldb,
                                                                     Istride const strideB,
                                                                     I* const ipiv_arg,
                                                                     Istride const strideP,
                                                                     I const batch_count)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;

    I const ij_inc = (blockDim.x * blockDim.y) * blockDim.z;
    I const ij_start
        = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    auto ceildiv = [](auto const n, auto const b) { return ((n <= 0) ? 0 : ((n - 1) / b) + 1); };

    I const nb = ceildiv(nrhs_arg, nbx);

    I const jstart = ibx * nb;
    I const jend = std::min(nrhs_arg, jstart + nb);
    I const nrhs = (jend - jstart);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto const offsetB = idx2D(0, jstart, ldb);

    // -------------------------------------------
    // NOTE: each thread block swap nrhs columns locally
    // so check value of nrhs
    // -------------------------------------------
    if((nrhs == 0) || (n == 0) || (batch_count == 0))
    {
        return;
    }

    // Fortran 1-based index value
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto swap = [](T& x, T& y) {
        auto const temp = x;
        x = y;
        y = temp;
    };

    for(auto bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const B_bid = load_ptr_batch<T>(B_arg, bid, shiftB, strideB);
        T* const B_with_offset = B_bid + offsetB;

        I const* const ipiv_bid = ipiv_arg + bid * strideP;

        // ---------------------------------------------------------
        // Use 1-based indexing to be compatible with Fortran/matlab
        // ---------------------------------------------------------

        auto B = [=](auto i, auto j) -> T& { return (B_with_offset[idx2F(i, j, ldb)]); };

        auto ipiv = [=](auto i) { return (ipiv_bid[(i - 1)]); };

        // --------------------------------------------
        // swap rows k and kp of matrix B
        //
        // swap(  B(k,1:nrhs),  B(kp, 1:nrhs) )
        // --------------------------------------------
        auto swap_rows = [=](I const k, I const kp) {
            for(I j = 1 + ij_start; j <= nrhs; j += ij_inc)
            {
                swap(B(k, j), B(kp, j));
            }
        };

        if(!is_forward)
        {
            // -----------------------------
            // apply pivots in reverse order
            // -----------------------------

            //         k=n
            //         do while ( k .ge. 1 )
            //          if( ipiv( k ).gt.0 ) then
            // *           1 x 1 diagonal block
            // *           interchange rows k and ipiv(k).
            //             kp = ipiv( k )
            //             if( kp.ne.k )
            //      $         call zswap( nrhs, b( k, 1 ), ldb, b( kp, 1 ), ldb )
            //             k=k-1
            //          else
            // *           2 x 2 diagonal block
            // *           interchange rows k-1 and -ipiv(k).
            //             kp = -ipiv( k )
            //             if( k.gt.1 .and. kp.eq.-ipiv( k-1 ) )
            //      $         call zswap( nrhs, b( k, 1 ), ldb, b( kp, 1 ), ldb )
            //             k=k-2
            //          endif
            //         end do
            I k = n;
            while(k >= 1)
            {
                bool const use_1x1_block = (ipiv(k) > 0);
                if(use_1x1_block)
                {
                    //             ------------------------------
                    //             1 x 1 diagonal block
                    //             interchange rows k and ipiv(k).
                    //             ------------------------------
                    I const kp = ipiv(k);
                    if(kp != k)
                    {
                        swap_rows(k, kp);
                    }
                    k = k - 1;
                }
                else
                {
                    //           ---------------------------------
                    //           2 x 2 diagonal block
                    //           ---------------------------------
                    I const kp = -ipiv(k);
                    if((k > 1) && (kp == (-ipiv(k - 1))))
                    {
                        swap_rows(k, kp);
                    }
                    k = k - 2;
                }
            } // end while
        }
        else
        {
            // -----------------------------
            // apply pivots in forward order
            // -----------------------------

            //         k=1
            //         do while ( k .le. n )
            //          if( ipiv( k ).gt.0 ) then
            // *           1 x 1 diagonal block
            // *           interchange rows k and ipiv(k).
            //             kp = ipiv( k )
            //             if( kp.ne.k )
            //      $         call zswap( nrhs, b( k, 1 ), ldb, b( kp, 1 ), ldb )
            //             k=k+1
            //          else
            // *           2 x 2 diagonal block
            // *           interchange rows k and -ipiv(k+1).
            //             kp = -ipiv( k+1 )
            //             if( kp.eq.-ipiv( k ) )
            //      $         call zswap( nrhs, b( k+1, 1 ), ldb, b( kp, 1 ), ldb )
            //             k=k+2
            //          endif
            //         end do
            I k = 1;
            while(k <= n)
            {
                bool const use_1x1_block = (ipiv(k) > 0);
                if(use_1x1_block)
                {
                    // ------------------------------------------
                    //             1 x 1 diagonal block
                    //             interchange rows k and ipiv(k).
                    // ------------------------------------------
                    I const kp = ipiv(k);
                    if(kp != k)
                    {
                        swap_rows(k, kp);
                    }
                    k = k + 1;
                }
                else
                {
                    // --------------------------------------------
                    //            2 x 2 diagonal block
                    // --------------------------------------------
                    I const kp = -ipiv(k + 1);
                    if(kp == (-ipiv(k)))
                    {
                        // call_swap(nrhs, &(B(k + 1, 1)), ldb, &(B(kp, 1)), ldb);
                        swap_rows(k + 1, kp);
                    }
                    k = k + 2;
                }
            } // end while
        }

        __syncthreads();
    } // end for bid
}

template <typename T, typename I, typename UB, typename Istride>
static rocblas_status apply_pivot_lower(rocblas_handle handle,
                                        bool const is_forward,
                                        I const n,
                                        I const nrhs_arg,
                                        UB B,
                                        Istride const shiftB,
                                        I const ldb,
                                        Istride const strideB,
                                        I* const ipiv_arg,
                                        Istride const strideP,
                                        I const batch_count)
{
    if((n == 0) || (nrhs_arg == 0) || (batch_count == 0))
    {
        return rocblas_status_success;
    }

    auto ceildiv = [](auto const n, auto const b) { return ((n <= 0) ? 0 : ((n - 1) / b) + 1); };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const max_blocks = 64 * 1024 - 3;

    I const NB = SYTRS1_MAX_THDS;
    I const nbx = std::max(I(1), std::min(max_blocks, ceildiv(nrhs_arg, NB)));
    I const nby = 1;
    I const nbz = std::min(max_blocks, batch_count);

    I const lrhs = ceildiv(nrhs_arg, nbx);

    I const nx = std::min(lrhs, I(SYTRS1_MAX_THDS));
    I const ny = 1;
    I const nz = 1;

    ROCSOLVER_LAUNCH_KERNEL((apply_pivot_lower_kernel<T, I, UB, Istride>), dim3(nbx, nby, nbz),
                            dim3(nx, ny, nz), 0, stream, is_forward, n, nrhs_arg, B, shiftB, ldb,
                            strideB, ipiv_arg, strideP, batch_count);

    return rocblas_status_success;
}

//  ----------------------------------------------------------
//  solve linear system A * X = B, where A is symmetric indefinite
//  The matrix A initially contains the factorization by SYTRF
//  but has been modified by SYCONV  to use TRSM
//  ----------------------------------------------------------
template <typename T, typename I, typename UA, typename UB, typename Istride>
static rocblas_status sytrs2_inner_template(rocblas_handle handle,
                                            bool const is_upper,
                                            I const n,
                                            I const nrhs,
                                            UA A_arg,
                                            Istride const shiftA,
                                            I const lda,
                                            Istride const strideA,
                                            I* const ipiv_arg,
                                            Istride const strideP,
                                            T* E_arg,
                                            Istride const strideE,
                                            UB B_arg,
                                            Istride const shiftB,
                                            I const ldb,
                                            Istride const strideB,
                                            I const batch_count,
                                            void* const work,
                                            size_t const size_work)
{
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    T const one = 1;

    std::byte* const pwork = static_cast<std::byte*>(work);
    std::byte* pfree = pwork;

    auto call_trsm
        = [=, &pfree](rocblas_side const side, rocblas_fill const uplo,
                      rocblas_operation const trans, rocblas_diagonal const diag, I const n,
                      I const nrhs, T alpha, UA A_arg, Istride const shiftA, I const lda,
                      Istride const strideA, UB B_arg, Istride const shiftB, I const ldb,
                      Istride const strideB, I const batch_count) -> rocblas_status {
        static_assert(std::is_pointer_v<decltype(A_arg)>, "invalid A type");
        static_assert(std::is_pointer_v<decltype(B_arg)>, "invalid B type");
        static_assert(std::is_pointer_v<decltype(A_arg)> == std::is_pointer_v<decltype(B_arg)>,
                      "mismatch of A,B types");

        bool constexpr BATCHED
            = std::is_pointer_v<std::remove_reference_t<
                  decltype(A_arg[0])>> || std::is_array_v<std::remove_reference_t<decltype(A_arg[0])>>;
        bool constexpr STRIDED = !BATCHED;

        auto const pfree_saved = pfree;

        bool const is_upper = (uplo == rocblas_fill_upper);

        bool optim_mem = true;

        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;

        // -----------------------------------
        // workspace required for calling TRSM
        // -----------------------------------
        {
            rocblas_status const istat = rocblasCall_trsm_mem<BATCHED, T>(
                side, trans, n, nrhs, lda, ldb, batch_count, &size_work1, &size_work2, &size_work3,
                &size_work4);

            bool const is_ok
                = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
            if(!is_ok)
            {
                return rocblas_status_internal_error;
            }
        }

        void* const work1 = static_cast<void*>(pfree);
        pfree += size_work1;
        void* const work2 = static_cast<void*>(pfree);
        pfree += size_work2;
        void* const work3 = static_cast<void*>(pfree);
        pfree += size_work3;
        void* const work4 = static_cast<void*>(pfree);
        pfree += size_work4;

        if(pfree > (pwork + size_work))
            return rocblas_status_memory_error;

        ROCBLAS_CHECK(rocblasCall_trsm<T, I>(handle, side, uplo, trans, diag, n, nrhs, &alpha, A_arg,
                                             shiftA, lda, strideA, B_arg, shiftB, ldb, strideB,
                                             batch_count, optim_mem, work1, work2, work3, work4));

        pfree = pfree_saved;

        return rocblas_status_success;
    }; // end call_trsm

    bool const is_forward = false;
    T const alpha = one;

    if(is_upper)
    {
        // ------------------------------------------
        //         solve A*X = B, where A = U*D*U**t.
        //
        //        P**t * B
        // ------------------------------------------
        ROCBLAS_CHECK(apply_pivot_upper<T, I>(handle, is_forward, n, nrhs, B_arg, shiftB, ldb,
                                              strideB, ipiv_arg, strideP, batch_count));

        // --------------------------------------------------
        //    compute (U \P**t * B) -> B    [ (U \P**t * B) ]
        // --------------------------------------------------
        ROCBLAS_CHECK(call_trsm(rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
                                rocblas_diagonal_unit, n, nrhs, alpha, A_arg, shiftA, lda, strideA,
                                B_arg, shiftB, ldb, strideB, batch_count));

        // ---------------------------------------------
        //    compute D \ B -> B   [ B \ (U \P**t * B) ]
        // ---------------------------------------------
        ROCBLAS_CHECK(apply_diag_block<T, I>(handle, is_forward, n, nrhs, A_arg, shiftA, lda,
                                             strideA, ipiv_arg, strideP, E_arg, strideE, B_arg,
                                             shiftB, ldb, strideB, batch_count));

        // --------------------------------------------------------------
        //      compute (U**t \ B) -> B   [ U**t \ (D \ (U \P**t * B) ) ]
        //      NOTE: need pure transpose, not conjugate transpose
        // --------------------------------------------------------------
        ROCBLAS_CHECK(call_trsm(rocblas_side_left, rocblas_fill_upper, rocblas_operation_transpose,
                                rocblas_diagonal_unit, n, nrhs, alpha, A_arg, shiftA, lda, strideA,
                                B_arg, shiftB, ldb, strideB, batch_count));

        //        --------------------------------------------
        //        P * B  [ P * (U**t \ (D \ (U \P**t * B) )) ]
        //        --------------------------------------------
        ROCBLAS_CHECK(apply_pivot_upper<T, I>(handle, is_forward, n, nrhs, B_arg, shiftB, ldb,
                                              strideB, ipiv_arg, strideP, batch_count));
    }
    else
    {
        // -------------------------------------------
        // *        solve A*X = B, where A = L*D*L**t.
        // *
        // *       P**t * B
        // -------------------------------------------
        ROCBLAS_CHECK(apply_pivot_lower<T, I>(handle, is_forward, n, nrhs, B_arg, shiftB, ldb,
                                              strideB, ipiv_arg, strideP, batch_count));

        // ---------------------------------------------------
        //    compute (L \P**t * B) -> B    [ (L \P**t * B) ]
        // ---------------------------------------------------
        ROCBLAS_CHECK(call_trsm(rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                                rocblas_diagonal_unit, n, nrhs, alpha, A_arg, shiftA, lda, strideA,
                                B_arg, shiftB, ldb, strideB, batch_count));

        // ---------------------------------------------
        //    compute D \ B -> B   [ D \ (L \P**t * B) ]
        // ---------------------------------------------
        ROCBLAS_CHECK(apply_diag_block<T, I>(handle, is_forward, n, nrhs, A_arg, shiftA, lda,
                                             strideA, ipiv_arg, strideP, E_arg, strideE, B_arg,
                                             shiftB, ldb, strideB, batch_count));

        // ------------------------------------------------------------
        //    compute (L**t \ B) -> B   [ L**t \ (D \ (L \P**t * B) ) ]
        //    NOTE: need pure transpose, not conjugate transpose
        // ------------------------------------------------------------
        ROCBLAS_CHECK(call_trsm(rocblas_side_left, rocblas_fill_lower, rocblas_operation_transpose,
                                rocblas_diagonal_unit, n, nrhs, alpha, A_arg, shiftA, lda, strideA,
                                B_arg, shiftB, ldb, strideB, batch_count));

        // ----------------------------------------------------
        //         P * B  [ P * (L**t \ (D \ (L \P**t * B) )) ]
        // ----------------------------------------------------
        ROCBLAS_CHECK(apply_pivot_lower<T, I>(handle, is_forward, n, nrhs, B_arg, shiftB, ldb,
                                              strideB, ipiv_arg, strideP, batch_count));
    }

    return (rocblas_status_success);
}

int constexpr ialign = 128;

template <typename T, typename I>
rocblas_status rocsolver_sytrs2_getMemorySize(rocblas_handle handle,
                                              I const n,
                                              I const nrhs,
                                              I const batch_count,
                                              I const lda,
                                              I const ldb,
                                              size_t* const p_size_work)
{
    *p_size_work = 0;

    // if quick return no workspace needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    size_t size_E = sizeof(T) * n * batch_count;

    // sizes for trsm
    size_t size_trsm = 0;

    {
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        bool const optim_mem = true;

        rocblas_side const side = rocblas_side_left;

        for(I itrans = 0; itrans <= 1; itrans++)
        {
            rocblas_operation const trans
                = (itrans == 0) ? rocblas_operation_none : rocblas_operation_transpose;

            {
                bool constexpr BATCHED = true;
                bool constexpr STRIDED = !BATCHED;

                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                auto const istat = rocblasCall_trsm_mem<BATCHED, T, I>(
                    side, trans, n, nrhs, lda, ldb, batch_count, &lsize_work1, &lsize_work2,
                    &lsize_work3, &lsize_work4);

                bool const is_ok
                    = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
                if(!is_ok)
                {
                    return (istat);
                }

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }

            {
                bool constexpr BATCHED = false;
                bool constexpr STRIDED = !BATCHED;

                size_t lsize_work1 = 0;
                size_t lsize_work2 = 0;
                size_t lsize_work3 = 0;
                size_t lsize_work4 = 0;

                auto const istat = rocblasCall_trsm_mem<BATCHED, T>(
                    side, trans, n, nrhs, lda, ldb, batch_count, &lsize_work1, &lsize_work2,
                    &lsize_work3, &lsize_work4);

                if(istat != rocblas_status_success)
                {
                    return (istat);
                }

                size_work1 = std::max(size_work1, lsize_work1);
                size_work2 = std::max(size_work2, lsize_work2);
                size_work3 = std::max(size_work3, lsize_work3);
                size_work4 = std::max(size_work4, lsize_work4);
            }

        } // end for itrans

        size_trsm = size_work1 + size_work2 + size_work3 + size_work4;
    }

    size_t size_syconv = 0;
    ROCBLAS_CHECK(rocsolver_syconv_getMemorySize<T>(handle, n, batch_count, &size_syconv));

    *p_size_work = size_E + std::max(size_syconv, size_trsm);

    return rocblas_status_success;
}

template <typename UA, typename UB, typename I>
static inline rocblas_status rocsolver_sytrs2_argCheck(rocblas_handle handle,
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
    if(n < 0 || lda < n || ldb < n || nrhs < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && batch_count && !A) || (n && batch_count && !ipiv) || (nrhs && n && batch_count && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename UA, typename UB, typename Istride = rocblas_stride>
static inline rocblas_status rocsolver_sytrs2_template(rocblas_handle handle,
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
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    bool const is_upper = (uplo == rocblas_fill_upper);

    std::byte* const pwork = static_cast<std::byte*>(work);
    std::byte* pfree = pwork;

    size_t const size_E = sizeof(T) * n * batch_count;
    T* const E = reinterpret_cast<T*>(pfree);
    pfree += size_E;
    Istride const strideE = n;

    // ----------------
    // convert matrix A
    // ----------------
    size_t const size_remain = (pwork + size_work) - pfree;
    bool is_convert = true;
    ROCBLAS_CHECK(rocsolver_syconv_template<T, I>(handle, is_upper, is_convert = true, n, A, shiftA,
                                                  lda, strideA, ipiv, strideP, E, strideE,
                                                  batch_count, pfree, size_remain));

    // -------------------------------------------------
    // solve linear system with converted storage format
    // -------------------------------------------------
    rocblas_status istat_sytrs1 = rocblas_status_success;
    {
        auto const pfree_save = pfree;
        size_t size_remain = (pwork + size_work) - pfree;

        // everything must be executed with scalars on the host
        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle, &old_mode);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

        istat_sytrs1 = sytrs2_inner_template<T, I>(handle, is_upper, n, nrhs, A, shiftA, lda,
                                                   strideA, ipiv, strideP, E, strideE, B, shiftB,
                                                   ldb, strideB, batch_count, pfree, size_remain);

        rocblas_set_pointer_mode(handle, old_mode);

        pfree = pfree_save;
    }

    // ---------------
    // NOTE: always revert matrix A
    // even if sytrs1 has an error
    // ---------------

    rocblas_status istat_syconv = rocblas_status_success;
    {
        auto const pfree_save = pfree;

        size_t const size_remain = (pwork + size_work) - pfree;
        bool is_convert = false;
        istat_syconv = rocsolver_syconv_template<T, I>(
            handle, is_upper, is_convert = false, n, A, shiftA, lda, strideA, ipiv, strideP, E,
            strideE, batch_count, (void*)pfree, size_remain);

        pfree = pfree_save;
    }

    if(istat_sytrs1 != rocblas_status_success)
    {
        return (istat_sytrs1);
    }

    if(istat_syconv != rocblas_status_success)
    {
        return (istat_syconv);
    }

    return rocblas_status_success;
}
ROCSOLVER_END_NAMESPACE
#undef SYTRS1_MAX_THDS
