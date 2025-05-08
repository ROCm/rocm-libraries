/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "definitions.hpp"
#include "device_macros.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_gemm.hpp"
#include "rocblas_trmm.hpp"
#include "src64/blas3/rocblas_gemm_64.hpp"

template <typename T>
static const T beta_1 = T(1);

template <typename T>
static const T alpha_0 = T(0);

//-- Innovative Computing Laboratory
//  -- Electrical Engineering and Computer Science Department
//  -- University of Tennessee
//  -- (C) Copyright 2009-2020
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of the University of Tennessee, Knoxville nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors
//  ``as is'' and any express or implied warranties, including, but not
//  limited to, the implied warranties of merchantability and fitness for
//  a particular purpose are disclaimed. In no event shall the copyright
//  holders or contributors be liable for any direct, indirect, incidental,
//  special, exemplary, or consequential damages (including, but not
//  limited to, procurement of substitute goods or services; loss of use,
//  data, or profits; or business interruption) however caused and on any
//  theory of liability, whether in contract, strict liability, or tort
//  (including negligence or otherwise) arising in any way out of the use
//  of this software, even if advised of the possibility of such damage.

rocblas_int rocblas_get_trmm_recursive_nb(rocblas_int n)
{
    if(n > 8192)
        return 8192;
    else if(n <= 1)
        return 1;

    return rocblas_previous_po2(n - 1);
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                             rocblas_int    n,
                                             TScal          alpha_device_host,
                                             rocblas_stride stride_alpha,
                                             TPtr           Aa,
                                             int64_t        lda,
                                             rocblas_stride a_st_or_of,
                                             rocblas_int    batch_count)
{
    ptrdiff_t tx = blockIdx.x * DIM_X + threadIdx.x;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);

        if(alpha == 0)
        {
            for(ptrdiff_t ty = blockIdx.y * DIM_Y + threadIdx.y; tx < m && ty < n;
                ty += DIM_Y * gridDim.y)
            {
                auto* A = load_ptr_batch(Aa, batch, a_st_or_of);

                A[tx + size_t(lda) * ty] = 0;
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename TScal, typename TPtr>
rocblas_status rocblas_set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                              rocblas_int    m,
                                                              rocblas_int    n,
                                                              TScal          alpha,
                                                              rocblas_stride stride_alpha,
                                                              TPtr           A,
                                                              int64_t        lda,
                                                              rocblas_stride a_st_or_of,
                                                              rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();
    int         batches        = handle->getBatchGridDim((int)batch_count);

    static constexpr int GEMV_DIM_X = 16;
    static constexpr int GEMV_DIM_Y = 16;
    rocblas_int          blocksX    = (m - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = std::min(c_YZ_grid_launch_limit, (n - 1) / GEMV_DIM_Y + 1);

    dim3 grid(blocksX, blocksY, batches);
    dim3 threads(GEMV_DIM_X, GEMV_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
        ROCBLAS_LAUNCH_KERNEL(
            (rocblas_set_matrix_zero_if_alpha_zero_kernel<GEMV_DIM_X, GEMV_DIM_Y>),
            grid,
            threads,
            0,
            rocblas_stream,
            m,
            n,
            alpha,
            stride_alpha,
            A,
            lda,
            a_st_or_of,
            batch_count);
    else
        ROCBLAS_LAUNCH_KERNEL(
            (rocblas_set_matrix_zero_if_alpha_zero_kernel<GEMV_DIM_X, GEMV_DIM_Y>),
            grid,
            threads,
            0,
            rocblas_stream,
            m,
            n,
            *alpha,
            stride_alpha,
            A,
            lda,
            a_st_or_of,
            batch_count);
    return rocblas_status_success;
}

// right, transpose_and_conjugate_transpose
template <typename T,
          const int NB,
          const int THR_DIM,
          bool      LEFT,
          bool      UPPER,
          bool      TRANSPOSE,
          bool      CONJ,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_outofplace_kernel(rocblas_diagonal diag,
                               int              m,
                               int              n,
                               TScal            alpha_device_host,
                               rocblas_stride   stride_alpha,
                               TConstPtr*       A_arg,
                               rocblas_stride   offset_a,
                               int64_t          lda,
                               rocblas_stride   stride_a,
                               TConstPtr*       B_arg,
                               rocblas_stride   offset_b,
                               int64_t          ldb,
                               rocblas_stride   stride_b,
                               TPtr*            C_arg,
                               rocblas_stride   offset_c,
                               int64_t          ldc,
                               rocblas_stride   stride_c,
                               rocblas_int      batch_count)
{
    constexpr bool        ITER_UPPER = (UPPER && !TRANSPOSE) || (!UPPER && TRANSPOSE);
    constexpr rocblas_int DIM        = NB / THR_DIM;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto A     = load_ptr_batch(A_arg, batch, offset_a, stride_a);
        auto B     = load_ptr_batch(B_arg, batch, offset_b, stride_b);
        auto C     = load_ptr_batch(C_arg, batch, offset_c, stride_c);

        if(alpha == 0)
            return;

        const rocblas_int k = LEFT ? m : n;

        const rocblas_int tx = threadIdx.x;
        const rocblas_int ty = threadIdx.y;
        const rocblas_int bx = blockIdx.x;

        __shared__ T sA[NB][NB];
        __shared__ T sB[NB][NB];

        T rC[THR_DIM][THR_DIM];

        for(rocblas_int by = blockIdx.y; by < ((n - 1) / NB + 1); by += gridDim.y)
        {
            // A and B offset by blocks and threads
            // For LEFT,  bx is block row of A. For upper triangular, we need to offset the column
            //            as well since the first (lower) portion doesn't need to be accessed.
            // For RIGHT, by is block column of A. For lower triangular, we need to offset the row
            //            as well since the first (upper) portion doesn't need to be accessed.
            //
            rocblas_stride A_col_offset = LEFT ? (ITER_UPPER ? bx * NB + ty : ty)
                                               : (ITER_UPPER ? by * NB + ty : ty + NB * by);
            rocblas_stride A_row_offset = LEFT ? bx * NB + tx : (ITER_UPPER ? tx : by * NB + tx);
            rocblas_stride B_row_offset = LEFT ? (ITER_UPPER ? NB * bx + tx : tx) : bx * NB + tx;
            rocblas_stride B_col_offset = LEFT ? by * NB + ty : (ITER_UPPER ? ty : ty + NB * by);

            const T* dA = A
                          + (TRANSPOSE ? A_col_offset + A_row_offset * lda
                                       : A_row_offset + A_col_offset * lda);
            const T* dB = B + B_row_offset + B_col_offset * ldb;

            // zero out result matrix
            for(rocblas_int i = 0; i < THR_DIM; i++)
            {
                for(rocblas_int j = 0; j < THR_DIM; j++)
                {
                    rC[i][j] = 0.0;
                }
            }

            // full blocks of A. bx is the block row which is equal to
            // the number of full blocks in that block row (for lower triangular)
            const rocblas_int full_blocks = LEFT ? NB * bx : NB * by;

            // Iterate through the blocks. If we iterate up the triangular matrix on the left, we use the inverse of what is calculated above,
            // otherwise we add a BLK for the triangular block.
            rocblas_int block_iter_end = ((ITER_UPPER && LEFT) || (!ITER_UPPER && !LEFT))
                                             ? k - full_blocks
                                             : full_blocks + NB;
            for(rocblas_int blk_iter = 0; blk_iter < block_iter_end; blk_iter += NB)
            {
                // store A in shared memory
                for(rocblas_int i = 0; i < NB; i += DIM)
                {
                    for(rocblas_int j = 0; j < NB; j += DIM)
                    {
                        // Check if the A index is within the bounds of the matrix, is on a diagonal, and is within the triangular section.
                        size_t A_idx = TRANSPOSE ? j * size_t(lda) + i : i * size_t(lda) + j;
                        bool   in_diag
                            = diag == rocblas_diagonal_unit && j + A_row_offset == i + A_col_offset;
                        bool in_size = j + A_row_offset < k && i + A_col_offset < k;
                        bool in_bounds
                            = in_size
                              && (UPPER ? (TRANSPOSE ? (j + A_row_offset >= i + A_col_offset)
                                                     : (j + A_row_offset <= i + A_col_offset))
                                        : (TRANSPOSE ? (j + A_row_offset <= i + A_col_offset)
                                                     : (j + A_row_offset >= i + A_col_offset)));

                        if(in_bounds && !in_diag)
                            sA[i + ty][j + tx] = CONJ ? conj(dA[A_idx]) : dA[A_idx];
                        else if(in_diag)
                            sA[i + ty][j + tx] = 1;
                        else
                            sA[i + ty][j + tx] = 0;
                    }
                }

                // store B in shared memory
                for(rocblas_int i = 0; i < NB; i += DIM)
                {
                    for(rocblas_int j = 0; j < NB; j += DIM)
                    {
                        if(i + B_col_offset < n && j + B_row_offset < m)
                            sB[i + ty][j + tx] = dB[j + i * size_t(ldb)];
                        else
                            sB[i + ty][j + tx] = 0;
                    }
                }

                __syncthreads();

                // multiply C = AB
                for(rocblas_int i = 0; i < NB; i++)
                {
                    for(rocblas_int jn = 0; jn < THR_DIM; jn++)
                    {
                        for(rocblas_int jm = 0; jm < THR_DIM; jm++)
                        {
                            if(LEFT)
                                rC[jn][jm] += sA[i][jm * DIM + tx] * sB[jn * DIM + ty][i];
                            else
                                rC[jn][jm] += sB[i][jm * DIM + tx] * sA[jn * DIM + ty][i];
                        }
                    }
                }

                // Iterate to next block column of A to multiply
                // For transpose, we iterate down the row of memory, effectively
                // iterating across the column of the transposed matrix
                if(LEFT)
                {
                    dA += TRANSPOSE ? NB : NB * size_t(lda);
                    A_col_offset += NB;
                    dB += NB;
                    B_row_offset += NB;
                }
                else
                {
                    dA += !TRANSPOSE ? NB : NB * size_t(lda);
                    A_row_offset += NB;
                    dB += NB * size_t(ldb);
                    B_col_offset += NB;
                }

                __syncthreads();
            }

            // store the C matrix
            for(rocblas_int jn = 0; jn < THR_DIM; jn++)
            {
                rocblas_int c_idxn = by * NB + jn * DIM + ty;
                for(rocblas_int jm = 0; jm < THR_DIM; jm++)
                {
                    rocblas_int c_idxm = bx * NB + jm * DIM + tx;
                    if(c_idxm < m && c_idxn < n)
                    {
                        C[c_idxn * size_t(ldc) + c_idxm] += alpha * rC[jn][jm];
                    }
                }
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename T,
          rocblas_int NB,
          bool        LEFT,
          bool        UPPER,
          bool        TRANSPOSE,
          bool        CONJ,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_trmm_outofplace_dispatch(rocblas_handle   handle,
                                                rocblas_diagonal diag,
                                                rocblas_int      m,
                                                rocblas_int      n,
                                                TScal*           alpha,
                                                rocblas_stride   stride_alpha,
                                                TConstPtr*       dA,
                                                rocblas_stride   offset_a,
                                                int64_t          lda,
                                                rocblas_stride   stride_a,
                                                TConstPtr*       dB,
                                                rocblas_stride   offset_b,
                                                int64_t          ldb,
                                                rocblas_stride   stride_b,
                                                TPtr*            dC,
                                                rocblas_stride   offset_c,
                                                int64_t          lddc,
                                                rocblas_stride   stride_c,
                                                rocblas_int      batch_count)
{
    // grid of  ((m - 1) / blk_m) + 1, ((n - 1) / blk_n) + 1) blocks per batch
    // block of (dim_m, dim_n) threads per block
    // for float, NB = 32, so just using that for now in case limited mem
    constexpr rocblas_int THR_DIM = 2;

    hipStream_t rocblas_stream = handle->get_stream();
    int         batches        = handle->getBatchGridDim((int)batch_count);

    const rocblas_int blkx = (m - 1) / NB + 1;
    const rocblas_int blky = std::min(c_YZ_grid_launch_limit, ((n - 1) / NB + 1));
    dim3              grid(blkx, blky, batches);

    dim3 threads(NB / THR_DIM, NB / THR_DIM);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        ROCBLAS_LAUNCH_KERNEL_GRID(
            grid,
            (rocblas_trmm_outofplace_kernel<T, NB, THR_DIM, LEFT, UPPER, TRANSPOSE, CONJ>),
            grid,
            threads,
            0,
            rocblas_stream,
            diag,
            m,
            n,
            alpha,
            stride_alpha,
            dA,
            offset_a,
            lda,
            stride_a,
            dB,
            offset_b,
            ldb,
            stride_b,
            dC,
            offset_c,
            lddc,
            stride_c,
            batch_count);
    }
    else
    {
        ROCBLAS_LAUNCH_KERNEL_GRID(
            grid,
            (rocblas_trmm_outofplace_kernel<T, NB, THR_DIM, LEFT, UPPER, TRANSPOSE, CONJ>),
            grid,
            threads,
            0,
            rocblas_stream,
            diag,
            m,
            n,
            *alpha,
            stride_alpha,
            dA,
            offset_a,
            lda,
            stride_a,
            dB,
            offset_b,
            ldb,
            stride_b,
            dC,
            offset_c,
            lddc,
            stride_c,
            batch_count);
    }

    return rocblas_status_success;
}

// left, NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_lNx_kernel(rocblas_fill     uplo,
                        rocblas_diagonal diag,
                        int              m,
                        int              n, // m must be <= NB
                        TScal            alpha_device_host,
                        rocblas_stride   stride_alpha,
                        TConstPtr*       A_arg,
                        int64_t          lda,
                        rocblas_stride   a_st_or_of,
                        TConstPtr*       B_arg,
                        int64_t          ldb,
                        rocblas_stride   b_st_or_of,
                        TPtr*            C_arg,
                        int64_t          ldc,
                        rocblas_stride   c_st_or_of,
                        rocblas_int      batch_count)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        T alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        if(alpha != T(0))
        {
            auto* A = load_ptr_batch(A_arg, batch, a_st_or_of);
            auto* B = load_ptr_batch(B_arg, batch, b_st_or_of);
            auto* C = load_ptr_batch(C_arg, batch, c_st_or_of);

            const int nblocks = (n - 1) / NB + 1;
            const int nn      = (bx < nblocks - 1) ? NB : n - (nblocks - 1) * NB;
            B += bx * NB * size_t(ldb);
            C += bx * NB * size_t(ldc);

            __shared__ T sA[NB * NB];
            __shared__ T sB[NB * NB];

            // initialize sA and sB to zero
            sA[ty * NB + tx] = 0;
            sB[ty * NB + tx] = 0;

            // load A and B
            if(ty < m && tx < m)
                sA[ty * NB + tx] = A[ty * size_t(lda) + tx];
            if(ty < nn && tx < m)
                sB[ty * NB + tx] = B[ty * size_t(ldb) + tx];

            // handle diag
            if(diag == rocblas_diagonal_unit)
            {
                if(ty == tx)
                    sA[ty * NB + tx] = 1.0;
            }

            // handle uplo
            if(uplo == rocblas_fill_upper)
            {
                if(tx > ty)
                    sA[ty * NB + tx] = 0.0;
            }
            else
            {
                if(tx < ty)
                    sA[ty * NB + tx] = 0.0;
            }
            __syncthreads();

            T accumulator = 0;
#pragma unroll
            for(int i = 0; i < NB; i++)
                accumulator += sA[i * NB + tx] * sB[ty * NB + i];
            accumulator *= alpha;
            if(ty < nn && tx < m)
                C[ty * size_t(ldc) + tx] = accumulator;
        }
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

// left, Trans|ConjTrans
template <const int NB, bool CONJA, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_lTx_kernel(rocblas_fill     uplo,
                        rocblas_diagonal diag,
                        int              m,
                        int              n, // m must be <= NB
                        TScal            alpha_device_host,
                        rocblas_stride   stride_alpha,
                        TConstPtr*       A_arg,
                        int64_t          lda,
                        rocblas_stride   a_st_or_of,
                        TConstPtr*       B_arg,
                        int64_t          ldb,
                        rocblas_stride   b_st_or_of,
                        TPtr*            C_arg,
                        int64_t          ldc,
                        rocblas_stride   c_st_or_of,
                        rocblas_int      batch_count)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        T alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        if(alpha != T(0))
        {
            auto* A = load_ptr_batch(A_arg, batch, a_st_or_of);
            auto* B = load_ptr_batch(B_arg, batch, b_st_or_of);
            auto* C = load_ptr_batch(C_arg, batch, c_st_or_of);

            const int nblocks = (n - 1) / NB + 1;
            const int nn      = (bx < nblocks - 1) ? NB : n - (nblocks - 1) * NB;
            B += bx * NB * size_t(ldb);
            C += bx * NB * size_t(ldc);

            __shared__ T sA[NB * NB];
            __shared__ T sB[NB * NB];

            // init sA and sB to zero
            sA[ty * NB + tx] = 0.0;
            sB[ty * NB + tx] = 0.0;
            __syncthreads(); // needed because sA will be stored as transposed

            // load A and B
            if(ty < m && tx < m)
            {
                if(CONJA)
                {
                    sA[tx * NB + ty] = conj(A[ty * size_t(lda) + tx]);
                }
                else
                {
                    sA[tx * NB + ty] = A[ty * size_t(lda) + tx];
                }
            }
            if(ty < nn && tx < m)
                sB[ty * NB + tx] = B[ty * size_t(ldb) + tx];

            // handle diag
            if(diag == rocblas_diagonal_unit)
            {
                if(ty == tx)
                    sA[ty * NB + tx] = 1.0;
            }

            // handle uplo
            __syncthreads();
            if(uplo == rocblas_fill_lower)
            {
                if(tx > ty)
                    sA[ty * NB + tx] = 0.0;
            }
            else
            {
                if(tx < ty)
                    sA[ty * NB + tx] = 0.0;
            }
            __syncthreads();

            T accumulator = 0.0;
#pragma unroll
            for(int i = 0; i < NB; i++)
                accumulator += sA[i * NB + tx] * sB[ty * NB + i];
            accumulator *= alpha;

            // write C
            if(ty < nn && tx < m)
                C[ty * size_t(ldc) + tx] = accumulator;
        }
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

// right NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_rNx_kernel(rocblas_fill     uplo,
                        rocblas_diagonal diag,
                        int              m,
                        int              n, // m must be <= NB
                        TScal            alpha_device_host,
                        rocblas_stride   stride_alpha,
                        TConstPtr*       A_arg,
                        int64_t          lda,
                        rocblas_stride   a_st_or_of,
                        TConstPtr*       B_arg,
                        int64_t          ldb,
                        rocblas_stride   b_st_or_of,
                        TPtr*            C_arg,
                        int64_t          ldc,
                        rocblas_stride   c_st_or_of,
                        rocblas_int      batch_count)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        T alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        if(alpha != T(0))
        {
            auto* A = load_ptr_batch(A_arg, batch, a_st_or_of);
            auto* B = load_ptr_batch(B_arg, batch, b_st_or_of);
            auto* C = load_ptr_batch(C_arg, batch, c_st_or_of);

            const int nblocks = (m - 1) / NB + 1;
            const int mm      = (bx < nblocks - 1) ? NB : m - (nblocks - 1) * NB;
            B += bx * NB;
            C += bx * NB;

            __shared__ T sA[NB * NB];
            __shared__ T sB[NB * NB];

            // init sA and sB to zero
            sA[ty * NB + tx] = 0.0;
            sB[ty * NB + tx] = 0.0;

            // load A and B
            if(ty < n && tx < n)
                sA[ty * NB + tx] = A[ty * size_t(lda) + tx];
            if(ty < n && tx < mm)
                sB[ty * NB + tx] = B[ty * size_t(ldb) + tx];

            // handle diag
            if(diag == rocblas_diagonal_unit)
            {
                if(ty == tx)
                    sA[ty * NB + tx] = 1.0;
            }

            // handle uplo
            if(uplo == rocblas_fill_upper)
            {
                if(tx > ty)
                    sA[ty * NB + tx] = 0.0;
            }
            else
            {
                if(tx < ty)
                    sA[ty * NB + tx] = 0.0;
            }
            __syncthreads();

            T accumulator = 0.0;
#pragma unroll
            for(int i = 0; i < NB; i++)
                accumulator += sB[i * NB + tx] * sA[ty * NB + i];
            accumulator *= alpha;
            // write C
            if(ty < n && tx < mm)
                C[ty * size_t(ldc) + tx] = accumulator;
        }
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

// right, transpose_and_conjugate_transpose
template <const int NB, bool CONJA, typename T, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(NB* NB)
rocblas_trmm_rTx_kernel(rocblas_fill     uplo,
                        rocblas_diagonal diag,
                        int              m,
                        int              n, // m must be <= NB
                        TScal            alpha_device_host,
                        rocblas_stride   stride_alpha,
                        TConstPtr*       A_arg,
                        int64_t          lda,
                        rocblas_stride   a_st_or_of,
                        TConstPtr*       B_arg,
                        int64_t          ldb,
                        rocblas_stride   b_st_or_of,
                        TPtr*            C_arg,
                        int64_t          ldc,
                        rocblas_stride   c_st_or_of,
                        rocblas_int      batch_count)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        T alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        if(alpha != T(0))
        {
            auto* A = load_ptr_batch(A_arg, batch, a_st_or_of);
            auto* B = load_ptr_batch(B_arg, batch, b_st_or_of);
            auto* C = load_ptr_batch(C_arg, batch, c_st_or_of);

            const int nblocks = (m - 1) / NB + 1;
            const int mm      = (bx < nblocks - 1) ? NB : m - (nblocks - 1) * NB;
            B += bx * NB;
            C += bx * NB;

            __shared__ T sA[NB * NB];
            __shared__ T sB[NB * NB];

            // init sA and sB to zero
            sA[ty * NB + tx] = 0.0;
            sB[ty * NB + tx] = 0.0;

            // load A and B
            if(ty < n && tx < n)
            {
                if(CONJA)
                {
                    sA[ty * NB + tx] = conj(A[ty * size_t(lda) + tx]);
                }
                else
                {
                    sA[ty * NB + tx] = A[ty * size_t(lda) + tx];
                }
            }
            if(ty < n && tx < mm)
                sB[ty * NB + tx] = B[ty * size_t(ldb) + tx];

            // handle diag
            if(diag == rocblas_diagonal_unit)
            {
                if(ty == tx)
                    sA[ty * NB + tx] = 1.0;
            }

            // handle uplo
            if(uplo == rocblas_fill_upper)
            {
                if(tx > ty)
                    sA[ty * NB + tx] = 0.0;
            }
            else
            {
                if(tx < ty)
                    sA[ty * NB + tx] = 0.0;
            }
            __syncthreads();

            T accumulator = 0.0;
#pragma unroll
            for(int i = 0; i < NB; i++)
                accumulator += sB[i * NB + tx] * sA[i * NB + ty];
            accumulator *= alpha;
            // write C
            if(ty < n && tx < mm)
                C[ty * size_t(ldc) + tx] = accumulator;
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

// clang-format off
// left, NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_trmm_template_lNx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, int64_t lda, rocblas_stride a_st_or_of,
                       TConstPtr*       dB, int64_t ldb, rocblas_stride b_st_or_of,
                       TPtr*            dC, int64_t ldc, rocblas_stride c_st_or_of,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();
    int         batches        = handle->getBatchGridDim((int)batch_count);

    dim3 threads(NB, NB);
    dim3 grid((n  - 1) / NB + 1, 1, batches);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        ROCBLAS_LAUNCH_KERNEL_GRID(grid, (rocblas_trmm_lNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of,
                           dC, ldc, c_st_or_of, batch_count);
    else
        ROCBLAS_LAUNCH_KERNEL_GRID(grid, (rocblas_trmm_lNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of,
                           dC, ldc, c_st_or_of, batch_count);

    return rocblas_status_success;
}

// left, Trans|ConjTrans
template <const int NB, bool CONJ, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status trmm_template_lTx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, int64_t lda, rocblas_stride a_st_or_of,
                       TConstPtr*       dB, int64_t ldb, rocblas_stride b_st_or_of,
                       TPtr*            dC, int64_t ldc, rocblas_stride c_st_or_of,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();
    int         batches        = handle->getBatchGridDim((int)batch_count);

    dim3 threads(NB, NB);
    dim3 grid(((n - 1) / NB) + 1, 1, batches);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        ROCBLAS_LAUNCH_KERNEL_GRID(grid, (rocblas_trmm_lTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of,
                           dC, ldc, c_st_or_of, batch_count);
    else
        ROCBLAS_LAUNCH_KERNEL_GRID(grid, (rocblas_trmm_lTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of,
                           dC, ldc, c_st_or_of, batch_count);

    return rocblas_status_success;
}

// right, NoTrans
template <const int NB, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_trmm_template_rNx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, int64_t lda, rocblas_stride a_st_or_of,
                       TConstPtr*       dB, int64_t ldb, rocblas_stride b_st_or_of,
                       TPtr*            dC, int64_t ldc, rocblas_stride c_st_or_of,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();
    int         batches        = handle->getBatchGridDim((int)batch_count);

    dim3 threads(NB, NB);
    dim3 grid((m - 1) / NB + 1, 1, batches);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        ROCBLAS_LAUNCH_KERNEL_GRID(grid, (rocblas_trmm_rNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of,
                           dC, ldc, c_st_or_of, batch_count);
    else
        ROCBLAS_LAUNCH_KERNEL_GRID(grid, (rocblas_trmm_rNx_kernel<NB, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of,
                           dC, ldc, c_st_or_of, batch_count);

    return rocblas_status_success;
}

// right, Trans|ConjTrans
template <const int NB, bool CONJ, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_trmm_template_rTx(rocblas_handle   handle,
                       rocblas_fill     uplo,
                       rocblas_diagonal diag,
                       rocblas_int      m,
                       rocblas_int      n,
                       TScal*           alpha,
                       rocblas_stride   stride_alpha,
                       TConstPtr*       dA, int64_t lda, rocblas_stride a_st_or_of,
                       TConstPtr*       dB, int64_t ldb, rocblas_stride b_st_or_of,
                       TPtr*            dC, int64_t ldc, rocblas_stride c_st_or_of,
                       rocblas_int      batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();
    int         batches        = handle->getBatchGridDim((int)batch_count);

    dim3 threads(NB, NB);
    dim3 grid((m - 1) / NB + 1, 1, batches);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        ROCBLAS_LAUNCH_KERNEL_GRID(grid, (rocblas_trmm_rTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of,
                           dC, ldc, c_st_or_of, batch_count);
    else
        ROCBLAS_LAUNCH_KERNEL_GRID(grid, (rocblas_trmm_rTx_kernel<NB, CONJ, T>), grid, threads, 0, rocblas_stream,
                           uplo, diag,
                           m, n, *alpha, stride_alpha,
                           dA, lda, a_st_or_of,
                           dB, ldb, b_st_or_of,
                           dC, ldc, c_st_or_of, batch_count);

    return rocblas_status_success;
}

// clang-format on
rocblas_int inline rocblas_trmm_get_shape(rocblas_side      side,
                                          rocblas_fill      uplo,
                                          rocblas_operation trans_a)
{
    rocblas_int shape = -1;
    if(side == rocblas_side_left)
    {
        if(trans_a == rocblas_operation_none)
        {
            if(uplo == rocblas_fill_lower)
                shape = 0;
            else
                shape = 1; // lNL, lNU
        }
        else if(trans_a == rocblas_operation_transpose)
        {
            if(uplo == rocblas_fill_lower)
                shape = 2;
            else
                shape = 3; // lTL, lTU
        }
        else if(trans_a == rocblas_operation_conjugate_transpose)
        {
            if(uplo == rocblas_fill_lower)
                shape = 4;
            else
                shape = 5; // lCL, lCU
        }
    }
    else if(side == rocblas_side_right)
    {
        if(trans_a == rocblas_operation_none)
        {
            if(uplo == rocblas_fill_lower)
                shape = 6;
            else
                shape = 7; // rNL, rNU
        }
        else if(trans_a == rocblas_operation_transpose)
        {
            if(uplo == rocblas_fill_lower)
                shape = 8;
            else
                shape = 9; // rTL, rTU
        }
        else if(trans_a == rocblas_operation_conjugate_transpose)
        {
            if(uplo == rocblas_fill_lower)
                shape = 10;
            else
                shape = 11; // rCL, rCU
        }
    }

    return shape;
}

template <bool BATCHED,
          int  STOPPING_NB,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_trmm_small(rocblas_handle    handle,
                                  rocblas_side      side,
                                  rocblas_fill      uplo,
                                  rocblas_operation trans_a,
                                  rocblas_diagonal  diag,
                                  rocblas_int       m,
                                  rocblas_int       n,
                                  TScal             alpha,
                                  rocblas_stride    stride_alpha,
                                  TConstPtr         dA,
                                  rocblas_stride    offset_a,
                                  int64_t           lda,
                                  rocblas_stride    stride_a,
                                  TConstPtr         dB,
                                  rocblas_stride    offset_b,
                                  int64_t           ldb,
                                  rocblas_stride    stride_b,
                                  TPtr              dC,
                                  rocblas_stride    offset_c,
                                  int64_t           ldc,
                                  rocblas_stride    stride_c,
                                  rocblas_int       batch_count)
{
    TConstPtr      dA_krn;
    TConstPtr      dB_krn;
    TPtr           dC_krn;
    rocblas_stride a_st_or_of;
    rocblas_stride b_st_or_of;
    rocblas_stride c_st_or_of;

    if(BATCHED)
    {
        dA_krn     = dA;
        dB_krn     = dB;
        dC_krn     = dC;
        a_st_or_of = offset_a;
        b_st_or_of = offset_b;
        c_st_or_of = offset_c;
    }
    else
    {
        dA_krn     = dA + offset_a;
        dB_krn     = dB + offset_b;
        dC_krn     = dC + offset_c;
        a_st_or_of = stride_a;
        b_st_or_of = stride_b;
        c_st_or_of = stride_c;
    }

    rocblas_int shape = rocblas_trmm_get_shape(side, uplo, trans_a);

    if(shape == 0 || shape == 1) // lNx, left, NoTrans
        return rocblas_trmm_template_lNx<STOPPING_NB, T>(handle,
                                                         uplo,
                                                         diag,
                                                         m,
                                                         n,
                                                         alpha,
                                                         stride_alpha,
                                                         dA_krn,
                                                         lda,
                                                         a_st_or_of,
                                                         dB_krn,
                                                         ldb,
                                                         b_st_or_of,
                                                         dC_krn,
                                                         ldc,
                                                         c_st_or_of,
                                                         batch_count);
    else if(shape == 2 || shape == 3) // lTx, left, Transpose
        return trmm_template_lTx<STOPPING_NB, false, T>(handle,
                                                        uplo,
                                                        diag,
                                                        m,
                                                        n,
                                                        alpha,
                                                        stride_alpha,
                                                        dA_krn,
                                                        lda,
                                                        a_st_or_of,
                                                        dB_krn,
                                                        ldb,
                                                        b_st_or_of,
                                                        dC_krn,
                                                        ldc,
                                                        c_st_or_of,
                                                        batch_count);
    else if(shape == 4 || shape == 5) // lCx, left, ConjTrans
        return trmm_template_lTx<STOPPING_NB, true, T>(handle,
                                                       uplo,
                                                       diag,
                                                       m,
                                                       n,
                                                       alpha,
                                                       stride_alpha,
                                                       dA_krn,
                                                       lda,
                                                       a_st_or_of,
                                                       dB_krn,
                                                       ldb,
                                                       b_st_or_of,
                                                       dC_krn,
                                                       ldc,
                                                       c_st_or_of,
                                                       batch_count);
    else if(shape == 6 || shape == 7) // rNx, right, NoTrans
        return rocblas_trmm_template_rNx<STOPPING_NB, T>(handle,
                                                         uplo,
                                                         diag,
                                                         m,
                                                         n,
                                                         alpha,
                                                         stride_alpha,
                                                         dA_krn,
                                                         lda,
                                                         a_st_or_of,
                                                         dB_krn,
                                                         ldb,
                                                         b_st_or_of,
                                                         dC_krn,
                                                         ldc,
                                                         c_st_or_of,
                                                         batch_count);
    else if(shape == 8 || shape == 9) // rTx, right, Transpose
        return rocblas_trmm_template_rTx<STOPPING_NB, false, T>(handle,
                                                                uplo,
                                                                diag,
                                                                m,
                                                                n,
                                                                alpha,
                                                                stride_alpha,
                                                                dA_krn,
                                                                lda,
                                                                a_st_or_of,
                                                                dB_krn,
                                                                ldb,
                                                                b_st_or_of,
                                                                dC_krn,
                                                                ldc,
                                                                c_st_or_of,
                                                                batch_count);
    else if(shape == 10 || shape == 11) // rCx, right, ConjTrans
        return rocblas_trmm_template_rTx<STOPPING_NB, true, T>(handle,
                                                               uplo,
                                                               diag,
                                                               m,
                                                               n,
                                                               alpha,
                                                               stride_alpha,
                                                               dA_krn,
                                                               lda,
                                                               a_st_or_of,
                                                               dB_krn,
                                                               ldb,
                                                               b_st_or_of,
                                                               dC_krn,
                                                               ldc,
                                                               c_st_or_of,
                                                               batch_count);
    else
        return rocblas_status_internal_error;
}

template <rocblas_int NB,
          bool        LEFT,
          bool        UPPER,
          bool        TRANS,
          bool        CONJ,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_trmm_outofplace_template(rocblas_handle   handle,
                                                rocblas_diagonal diag,
                                                rocblas_int      m,
                                                rocblas_int      n,
                                                TScal*           alpha,
                                                TConstPtr*       dA,
                                                rocblas_stride   offset_a,
                                                int64_t          lda,
                                                TConstPtr*       dB,
                                                rocblas_stride   offset_b,
                                                int64_t          ldb,
                                                TPtr*            dC,
                                                rocblas_stride   offset_c,
                                                int64_t          ldc)
{
    // using template params to indicate transpose type
    // declaring constexpr here for easier use in gemm calls
    constexpr rocblas_operation transA = CONJ    ? rocblas_operation_conjugate_transpose
                                         : TRANS ? rocblas_operation_transpose
                                                 : rocblas_operation_none;

    // LEFT:  If accessing upper part of triangular matrix A, our blocks for matrix B are offset
    //        depending on the row of A as the left hand side of A is empty. Writing to matrix C
    //        can start without an offset as we aren't skipping any rows of A or columns of B.
    //        If accessing the lower triangle, the offset of B is 0 as matrix A is full on the left side.
    //        Writing to matrix C must be offset as we skip the rows of A that are already completed.
    // RIGHT: This is opposite of the LEFT case as we are worried about the offset to the columns of the triangular
    //        matrix rather than the rows.
    constexpr bool TRI_OFFSET = (LEFT && ((UPPER && !TRANS) || (!UPPER && TRANS)))
                                || (!LEFT && ((UPPER && TRANS) || (!UPPER && !TRANS)));

    // memory access scalars
    // a is scaled by both rows and columns
    // b is scaled by 1 for LEFT case as we are striding
    //   down the rows of b
    // c is scaled by 1 for LEFT case as it follows b
    rocblas_stride a_block_stride = 1 + lda;
    rocblas_stride b_block_stride = LEFT ? 1 : ldb;
    rocblas_stride c_block_stride = LEFT ? 1 : ldc;

    // iterative matrix sizes
    rocblas_int k     = LEFT ? m : n;
    rocblas_int m_sub = LEFT ? NB : m;
    rocblas_int n_sub = LEFT ? n : NB;

    // trmm full blocks on the diagonal
    rocblas_int trmm_batch_count = k / NB;

    RETURN_IF_ROCBLAS_ERROR(
        (rocblas_trmm_outofplace_dispatch<T, 32, LEFT, UPPER, TRANS, CONJ>)(handle,
                                                                            diag,
                                                                            m_sub,
                                                                            n_sub,
                                                                            alpha,
                                                                            0,
                                                                            dA,
                                                                            offset_a,
                                                                            lda,
                                                                            NB * a_block_stride,
                                                                            dB,
                                                                            offset_b,
                                                                            ldb,
                                                                            NB * b_block_stride,
                                                                            dC,
                                                                            offset_c,
                                                                            ldc,
                                                                            NB * c_block_stride,
                                                                            trmm_batch_count));

    rocblas_stride offsetAin, offsetBin, offsetCin;

    // Calculate remainder triangular section, anything past last divisible NB block
    rocblas_int rem = k % NB;
    if(rem)
    {
        // offset is number of full blocks
        int64_t k_norem = k - rem;
        offsetAin       = offset_a + k_norem * a_block_stride;
        offsetBin       = offset_b + k_norem * b_block_stride;
        offsetCin       = offset_c + k_norem * c_block_stride;

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_trmm_outofplace_dispatch<T, 32, LEFT, UPPER, TRANS, CONJ>)(handle,
                                                                                diag,
                                                                                LEFT ? rem : m,
                                                                                LEFT ? n : rem,
                                                                                alpha,
                                                                                0,
                                                                                dA,
                                                                                offsetAin,
                                                                                lda,
                                                                                0,
                                                                                dB,
                                                                                offsetBin,
                                                                                ldb,
                                                                                0,
                                                                                dC,
                                                                                offsetCin,
                                                                                ldc,
                                                                                0,
                                                                                1));
    }

    for(int64_t k_sub = NB; k_sub < k; k_sub *= 2)
    {
        if(LEFT)
            m_sub = k_sub;
        else
            n_sub = k_sub;

        int64_t k_stride         = k_sub * 2;
        int64_t gemm_batch_count = (k - k_sub) / k_stride;
        rem                      = (k - k_sub) % k_stride;
        if(rem >= k_sub)
        {
            rem = 0;
            gemm_batch_count += 1;
        }

        offsetAin                  = offset_a + (UPPER ? k_sub * int64_t(lda) : k_sub);
        offsetBin                  = offset_b + (TRI_OFFSET ? k_sub * b_block_stride : 0);
        offsetCin                  = offset_c + (TRI_OFFSET ? 0 : k_sub * c_block_stride);
        rocblas_stride strideAgemm = k_stride * a_block_stride;
        rocblas_stride strideBgemm = k_stride * b_block_stride;
        rocblas_stride strideCgemm = k_stride * c_block_stride;

        // already zeroed out C, so set beta to 1
        RETURN_IF_ROCBLAS_ERROR(
            rocblas_internal_gemm_64<false>(handle,
                                            LEFT ? transA : rocblas_operation_none,
                                            LEFT ? rocblas_operation_none : transA,
                                            m_sub,
                                            n_sub,
                                            k_sub,
                                            alpha,
                                            LEFT ? dA : dB,
                                            LEFT ? offsetAin : offsetBin,
                                            LEFT ? lda : ldb,
                                            LEFT ? strideAgemm : strideBgemm,
                                            LEFT ? dB : dA,
                                            LEFT ? offsetBin : offsetAin,
                                            LEFT ? ldb : lda,
                                            LEFT ? strideBgemm : strideAgemm,
                                            &beta_1<T>,
                                            dC,
                                            offsetCin,
                                            ldc,
                                            strideCgemm,
                                            gemm_batch_count));

        if(rem)
        {
            rocblas_stride i1 = k_sub + gemm_batch_count * k_stride;
            rocblas_stride i2 = i1 - k_sub;

            rocblas_int m_rem = LEFT ? (TRI_OFFSET ? k_sub : rem) : m;
            rocblas_int n_rem = LEFT ? n : (TRI_OFFSET ? k_sub : rem);
            rocblas_int k_rem = TRI_OFFSET ? rem : k_sub;

            offsetAin = offset_a + (UPPER ? i2 + i1 * int64_t(lda) : i1 + i2 * int64_t(lda));
            offsetBin = offset_b + (TRI_OFFSET ? i1 * b_block_stride : i2 * b_block_stride);
            offsetCin = offset_c + (TRI_OFFSET ? i2 * c_block_stride : i1 * c_block_stride);

            RETURN_IF_ROCBLAS_ERROR(
                rocblas_internal_gemm_64<false>(handle,
                                                LEFT ? transA : rocblas_operation_none,
                                                LEFT ? rocblas_operation_none : transA,
                                                m_rem,
                                                n_rem,
                                                k_rem,
                                                alpha,
                                                LEFT ? dA : dB,
                                                LEFT ? offsetAin : offsetBin,
                                                LEFT ? lda : ldb,
                                                0,
                                                LEFT ? dB : dA,
                                                LEFT ? offsetBin : offsetAin,
                                                LEFT ? ldb : lda,
                                                0,
                                                &beta_1<T>,
                                                dC,
                                                offsetCin,
                                                ldc,
                                                0,
                                                1));
        }
    }

    return rocblas_status_success;
}

// Currently rocblas_internal_trmm_outofplace_template does not support multiple batches
template <typename T>
rocblas_status rocblas_internal_trmm_outofplace_template(rocblas_handle    handle,
                                                         rocblas_side      side,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation trans_a,
                                                         rocblas_diagonal  diag,
                                                         rocblas_int       m,
                                                         rocblas_int       n,
                                                         const T*          alpha,
                                                         const T*          dA,
                                                         rocblas_stride    offset_a,
                                                         int64_t           lda,
                                                         const T*          dB,
                                                         rocblas_stride    offset_b,
                                                         int64_t           ldb,
                                                         T*                dC,
                                                         rocblas_stride    offset_c,
                                                         int64_t           ldc)
{
#define trmm_out_KARGS \
    handle, diag, m, n, alpha, dA, offset_a, lda, dB, offset_b, ldb, dC, offset_c, ldc
    rocblas_int shape = rocblas_trmm_get_shape(side, uplo, trans_a);

    constexpr int STOPPING_NB = ROCBLAS_TRMM_OUTOFPLACE_NB;
    if(shape == 0)
    {
        // left, lower, non-transpose
        // template args:                                     LEFT, UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, true, false, false, false, T>(
            trmm_out_KARGS);
    }
    else if(shape == 1)
    {
        // left, upper, non-transpose
        // template args:                                     LEFT, UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, true, true, false, false, T>(
            trmm_out_KARGS);
    }
    else if(shape == 2)
    {
        // left, lower, transpose
        // template args:                                     LEFT, UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, true, false, true, false, T>(
            trmm_out_KARGS);
    }
    else if(shape == 3)
    {
        // left, upper, transpose
        // template args:                                     LEFT, UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, true, true, true, false, T>(
            trmm_out_KARGS);
    }
    else if(shape == 4)
    {
        // left, lower, conjugate-transpose
        // template args:                                     LEFT, UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, true, false, true, true, T>(
            trmm_out_KARGS);
    }
    else if(shape == 5)
    {
        // left, upper, conjugate-transpose
        // template args:                                     LEFT, UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, true, true, true, true, T>(
            trmm_out_KARGS);
    }
    else if(shape == 6)
    {
        // right, lower, non-transpose
        // template args:                                     LEFT,  UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, false, false, false, false, T>(
            trmm_out_KARGS);
    }
    else if(shape == 7)
    {
        // right, upper, non-transpose
        // template args:                                     LEFT,  UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, false, true, false, false, T>(
            trmm_out_KARGS);
    }
    else if(shape == 8)
    {
        // right, lower, transpose
        // template args:                                     LEFT,  UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, false, false, true, false, T>(
            trmm_out_KARGS);
    }
    else if(shape == 9)
    {
        // right, upper, transpose
        // template args:                                     LEFT,  UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, false, true, true, false, T>(
            trmm_out_KARGS);
    }
    else if(shape == 10)
    {
        // right, lower, conjugate-transpose
        // template args:                                     LEFT,  UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, false, false, true, true, T>(
            trmm_out_KARGS);
    }
    else if(shape == 11)
    {
        // right, upper, conjugate-transpose
        // template args:                                     LEFT,  UPPER, TRANS, CONJ
        return rocblas_trmm_outofplace_template<STOPPING_NB, false, true, true, true, T>(
            trmm_out_KARGS);
    }
    else
    {
        return rocblas_status_not_implemented;
    }
#undef trmm_out_KARGS
}

template <int  STOPPING_NB,
          bool BATCHED,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_trmm_recursive_template(rocblas_handle    handle,
                                                        rocblas_side      side,
                                                        rocblas_fill      uplo,
                                                        rocblas_operation trans_a,
                                                        rocblas_diagonal  diag,
                                                        rocblas_int       m,
                                                        rocblas_int       n,
                                                        TScal*            alpha,
                                                        rocblas_stride    stride_alpha,
                                                        TConstPtr*        dA,
                                                        rocblas_stride    offset_a,
                                                        int64_t           lda,
                                                        rocblas_stride    stride_a,
                                                        TConstPtr*        dB,
                                                        rocblas_stride    offset_b,
                                                        int64_t           ldb,
                                                        rocblas_stride    stride_b,
                                                        TPtr*             dC,
                                                        rocblas_stride    offset_c,
                                                        int64_t           ldc,
                                                        rocblas_stride    stride_c,
                                                        rocblas_int       batch_count)
{

#define CALC_OFFSET_A(i, j) offset_a + i + j* rocblas_stride(lda)
#define CALC_OFFSET_B(i, j) offset_b + i + j* rocblas_stride(ldb)
#define CALC_OFFSET_C(i, j) offset_c + i + j* rocblas_stride(ldc)

    rocblas_int nrow_a = (side == rocblas_side_left ? m : n);

    // stopping condition
    if(nrow_a <= STOPPING_NB)
    {
        return rocblas_trmm_small<BATCHED, STOPPING_NB, T>(handle,
                                                           side,
                                                           uplo,
                                                           trans_a,
                                                           diag,
                                                           m,
                                                           n,
                                                           alpha,
                                                           stride_alpha,
                                                           dA,
                                                           offset_a,
                                                           lda,
                                                           stride_a,
                                                           dB,
                                                           offset_b,
                                                           ldb,
                                                           stride_b,
                                                           dC,
                                                           offset_c,
                                                           ldc,
                                                           stride_c,
                                                           batch_count);
    }

    rocblas_int shape = rocblas_trmm_get_shape(side, uplo, trans_a);

    rocblas_status status = rocblas_status_success;

    if(shape == 0) // lNl    left, NoTrans, Lower
    {
        const int m1 = rocblas_get_trmm_recursive_nb(m);
        const int m2 = m - m1;

        RETURN_IF_ROCBLAS_ERROR((
            rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                              side,
                                                                              uplo,
                                                                              trans_a,
                                                                              diag,
                                                                              m2,
                                                                              n,
                                                                              alpha,
                                                                              stride_alpha,
                                                                              dA,
                                                                              CALC_OFFSET_A(m1, m1),
                                                                              lda,
                                                                              stride_a,
                                                                              dB,
                                                                              CALC_OFFSET_B(m1, 0),
                                                                              ldb,
                                                                              stride_b,
                                                                              dC,
                                                                              CALC_OFFSET_C(m1, 0),
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_64<BATCHED, T>(handle,
                                                                      rocblas_operation_none,
                                                                      rocblas_operation_none,
                                                                      m2,
                                                                      n,
                                                                      m1,
                                                                      alpha,
                                                                      dA,
                                                                      CALC_OFFSET_A(m1, 0),
                                                                      lda,
                                                                      stride_a,
                                                                      dB,
                                                                      CALC_OFFSET_B(0, 0),
                                                                      ldb,
                                                                      stride_b,
                                                                      &beta_1<T>,
                                                                      dC,
                                                                      CALC_OFFSET_C(m1, 0),
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count)));

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m1,
                                                                               n,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               CALC_OFFSET_A(0, 0),
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               CALC_OFFSET_B(0, 0),
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               CALC_OFFSET_C(0, 0),
                                                                               ldc,
                                                                               stride_c,
                                                                               batch_count)));
    }
    else if(shape == 1) // lNU  left, NoTrans, Upper
    {
        const int m2 = rocblas_get_trmm_recursive_nb(m);
        const int m1 = m - m2;

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m1,
                                                                               n,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               CALC_OFFSET_A(0, 0),
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               CALC_OFFSET_B(0, 0),
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               CALC_OFFSET_C(0, 0),
                                                                               ldc,
                                                                               stride_c,
                                                                               batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_64<BATCHED, T>(handle,
                                                                      rocblas_operation_none,
                                                                      rocblas_operation_none,
                                                                      m1,
                                                                      n,
                                                                      m2,
                                                                      alpha,
                                                                      dA,
                                                                      CALC_OFFSET_A(0, m1),
                                                                      lda,
                                                                      stride_a,
                                                                      dB,
                                                                      CALC_OFFSET_B(m1, 0),
                                                                      ldb,
                                                                      stride_b,
                                                                      &beta_1<T>,
                                                                      dC,
                                                                      CALC_OFFSET_C(0, 0),
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count)));

        RETURN_IF_ROCBLAS_ERROR((
            rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                              side,
                                                                              uplo,
                                                                              trans_a,
                                                                              diag,
                                                                              m2,
                                                                              n,
                                                                              alpha,
                                                                              stride_alpha,
                                                                              dA,
                                                                              CALC_OFFSET_A(m1, m1),
                                                                              lda,
                                                                              stride_a,
                                                                              dB,
                                                                              CALC_OFFSET_B(m1, 0),
                                                                              ldb,
                                                                              stride_b,
                                                                              dC,
                                                                              CALC_OFFSET_C(m1, 0),
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count)));
    }
    else if(shape == 2 || shape == 4) // lTL | lCL    left, Trans|ConjTrans, Lower
    {
        const int m2 = rocblas_get_trmm_recursive_nb(m);
        const int m1 = m - m2;

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m1,
                                                                               n,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               CALC_OFFSET_A(0, 0),
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               CALC_OFFSET_B(0, 0),
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               CALC_OFFSET_C(0, 0),
                                                                               ldc,
                                                                               stride_c,
                                                                               batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_64<BATCHED, T>(handle,
                                                                      trans_a,
                                                                      rocblas_operation_none,
                                                                      m1,
                                                                      n,
                                                                      m2,
                                                                      alpha,
                                                                      dA,
                                                                      CALC_OFFSET_A(m1, 0),
                                                                      lda,
                                                                      stride_a,
                                                                      dB,
                                                                      CALC_OFFSET_B(m1, 0),
                                                                      ldb,
                                                                      stride_b,
                                                                      &beta_1<T>,
                                                                      dC,
                                                                      CALC_OFFSET_C(0, 0),
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count)));

        RETURN_IF_ROCBLAS_ERROR((
            rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                              side,
                                                                              uplo,
                                                                              trans_a,
                                                                              diag,
                                                                              m2,
                                                                              n,
                                                                              alpha,
                                                                              stride_alpha,
                                                                              dA,
                                                                              CALC_OFFSET_A(m1, m1),
                                                                              lda,
                                                                              stride_a,
                                                                              dB,
                                                                              CALC_OFFSET_B(m1, 0),
                                                                              ldb,
                                                                              stride_b,
                                                                              dC,
                                                                              CALC_OFFSET_C(m1, 0),
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count)));
    }
    else if(shape == 3 || shape == 5) // lTU | lCU     left, Trans|ConjTrans, Upper
    {
        const int m1 = rocblas_get_trmm_recursive_nb(m);
        const int m2 = m - m1;

        RETURN_IF_ROCBLAS_ERROR((
            rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                              side,
                                                                              uplo,
                                                                              trans_a,
                                                                              diag,
                                                                              m2,
                                                                              n,
                                                                              alpha,
                                                                              stride_alpha,
                                                                              dA,
                                                                              CALC_OFFSET_A(m1, m1),
                                                                              lda,
                                                                              stride_a,
                                                                              dB,
                                                                              CALC_OFFSET_B(m1, 0),
                                                                              ldb,
                                                                              stride_b,
                                                                              dC,
                                                                              CALC_OFFSET_C(m1, 0),
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_64<BATCHED, T>(handle,
                                                                      trans_a,
                                                                      rocblas_operation_none,
                                                                      m2,
                                                                      n,
                                                                      m1,
                                                                      alpha,
                                                                      dA,
                                                                      CALC_OFFSET_A(0, m1),
                                                                      lda,
                                                                      stride_a,
                                                                      dB,
                                                                      CALC_OFFSET_B(0, 0),
                                                                      ldb,
                                                                      stride_b,
                                                                      &beta_1<T>,
                                                                      dC,
                                                                      CALC_OFFSET_C(m1, 0),
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count)));

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m1,
                                                                               n,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               CALC_OFFSET_A(0, 0),
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               CALC_OFFSET_B(0, 0),
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               CALC_OFFSET_C(0, 0),
                                                                               ldc,
                                                                               stride_c,
                                                                               batch_count)));
    }
    else if(shape == 6) // rNL       right, NoTrans, Lower
    {
        const int n2 = rocblas_get_trmm_recursive_nb(n);
        const int n1 = n - n2;

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m,
                                                                               n1,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               CALC_OFFSET_A(0, 0),
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               CALC_OFFSET_B(0, 0),
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               CALC_OFFSET_C(0, 0),
                                                                               ldc,
                                                                               stride_c,
                                                                               batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_64<BATCHED, T>(handle,
                                                                      rocblas_operation_none,
                                                                      trans_a,
                                                                      m,
                                                                      n1,
                                                                      n2,
                                                                      alpha,
                                                                      dB,
                                                                      CALC_OFFSET_B(0, n1),
                                                                      ldb,
                                                                      stride_b,
                                                                      dA,
                                                                      CALC_OFFSET_A(n1, 0),
                                                                      lda,
                                                                      stride_a,
                                                                      &beta_1<T>,
                                                                      dC,
                                                                      CALC_OFFSET_C(0, 0),
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count)));

        RETURN_IF_ROCBLAS_ERROR((
            rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                              side,
                                                                              uplo,
                                                                              trans_a,
                                                                              diag,
                                                                              m,
                                                                              n2,
                                                                              alpha,
                                                                              stride_alpha,
                                                                              dA,
                                                                              CALC_OFFSET_A(n1, n1),
                                                                              lda,
                                                                              stride_a,
                                                                              dB,
                                                                              CALC_OFFSET_B(0, n1),
                                                                              ldb,
                                                                              stride_b,
                                                                              dC,
                                                                              CALC_OFFSET_C(0, n1),
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count)));
    }
    else if(shape == 7) // rNU       right, NoTrans, Upper
    {
        const int n1 = rocblas_get_trmm_recursive_nb(n);
        const int n2 = n - n1;

        RETURN_IF_ROCBLAS_ERROR((
            rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                              side,
                                                                              uplo,
                                                                              trans_a,
                                                                              diag,
                                                                              m,
                                                                              n2,
                                                                              alpha,
                                                                              stride_alpha,
                                                                              dA,
                                                                              CALC_OFFSET_A(n1, n1),
                                                                              lda,
                                                                              stride_a,
                                                                              dB,
                                                                              CALC_OFFSET_B(0, n1),
                                                                              ldb,
                                                                              stride_b,
                                                                              dC,
                                                                              CALC_OFFSET_C(0, n1),
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_64<BATCHED, T>(handle,
                                                                      rocblas_operation_none,
                                                                      trans_a,
                                                                      m,
                                                                      n2,
                                                                      n1,
                                                                      alpha,
                                                                      dB,
                                                                      CALC_OFFSET_B(0, 0),
                                                                      ldb,
                                                                      stride_b,
                                                                      dA,
                                                                      CALC_OFFSET_A(0, n1),
                                                                      lda,
                                                                      stride_a,
                                                                      &beta_1<T>,
                                                                      dC,
                                                                      CALC_OFFSET_C(0, n1),
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count)));

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m,
                                                                               n1,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               CALC_OFFSET_A(0, 0),
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               CALC_OFFSET_B(0, 0),
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               CALC_OFFSET_C(0, 0),
                                                                               ldc,
                                                                               stride_c,
                                                                               batch_count)));
    }
    else if(shape == 8 || shape == 10) // rTL | rCL      right, Trans|ConjTrans, Lower
    {
        const int n1 = rocblas_get_trmm_recursive_nb(n);
        const int n2 = n - n1;

        RETURN_IF_ROCBLAS_ERROR((
            rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                              side,
                                                                              uplo,
                                                                              trans_a,
                                                                              diag,
                                                                              m,
                                                                              n2,
                                                                              alpha,
                                                                              stride_alpha,
                                                                              dA,
                                                                              CALC_OFFSET_A(n1, n1),
                                                                              lda,
                                                                              stride_a,
                                                                              dB,
                                                                              CALC_OFFSET_B(0, n1),
                                                                              ldb,
                                                                              stride_b,
                                                                              dC,
                                                                              CALC_OFFSET_C(0, n1),
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_64<BATCHED, T>(handle,
                                                                      rocblas_operation_none,
                                                                      trans_a,
                                                                      m,
                                                                      n2,
                                                                      n1,
                                                                      alpha,
                                                                      dB,
                                                                      CALC_OFFSET_B(0, 0),
                                                                      ldb,
                                                                      stride_b,
                                                                      dA,
                                                                      CALC_OFFSET_A(n1, 0),
                                                                      lda,
                                                                      stride_a,
                                                                      &beta_1<T>,
                                                                      dC,
                                                                      CALC_OFFSET_C(0, n1),
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count)));

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m,
                                                                               n1,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               CALC_OFFSET_A(0, 0),
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               CALC_OFFSET_B(0, 0),
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               CALC_OFFSET_C(0, 0),
                                                                               ldc,
                                                                               stride_c,
                                                                               batch_count)));
    }
    else if(shape == 9 || shape == 11) // rTU | rCU      right, Trans|ConjTrans, Upper
    {
        const int n2 = rocblas_get_trmm_recursive_nb(n);
        const int n1 = n - n2;

        RETURN_IF_ROCBLAS_ERROR(
            (rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                               side,
                                                                               uplo,
                                                                               trans_a,
                                                                               diag,
                                                                               m,
                                                                               n1,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               dA,
                                                                               CALC_OFFSET_A(0, 0),
                                                                               lda,
                                                                               stride_a,
                                                                               dB,
                                                                               CALC_OFFSET_B(0, 0),
                                                                               ldb,
                                                                               stride_b,
                                                                               dC,
                                                                               CALC_OFFSET_C(0, 0),
                                                                               ldc,
                                                                               stride_c,
                                                                               batch_count)));

        RETURN_IF_ROCBLAS_ERROR((rocblas_internal_gemm_64<BATCHED, T>(handle,
                                                                      rocblas_operation_none,
                                                                      trans_a,
                                                                      m,
                                                                      n1,
                                                                      n2,
                                                                      alpha,
                                                                      dB,
                                                                      CALC_OFFSET_B(0, n1),
                                                                      ldb,
                                                                      stride_b,
                                                                      dA,
                                                                      CALC_OFFSET_A(0, n1),
                                                                      lda,
                                                                      stride_a,
                                                                      &beta_1<T>,
                                                                      dC,
                                                                      CALC_OFFSET_C(0, 0),
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count)));

        RETURN_IF_ROCBLAS_ERROR((
            rocblas_internal_trmm_recursive_template<STOPPING_NB, BATCHED, T>(handle,
                                                                              side,
                                                                              uplo,
                                                                              trans_a,
                                                                              diag,
                                                                              m,
                                                                              n2,
                                                                              alpha,
                                                                              stride_alpha,
                                                                              dA,
                                                                              CALC_OFFSET_A(n1, n1),
                                                                              lda,
                                                                              stride_a,
                                                                              dB,
                                                                              CALC_OFFSET_B(0, n1),
                                                                              ldb,
                                                                              stride_b,
                                                                              dC,
                                                                              CALC_OFFSET_C(0, n1),
                                                                              ldc,
                                                                              stride_c,
                                                                              batch_count)));
    }
    else
    {
        status = rocblas_status_internal_error;
    }
    return status;
}

template <int NB, bool BATCHED, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_trmm_launcher(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans_a,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              TScal*            alpha,
                                              rocblas_stride    stride_alpha,
                                              TConstPtr*        dA,
                                              rocblas_stride    offset_a,
                                              int64_t           lda,
                                              rocblas_stride    stride_a,
                                              TConstPtr*        dB,
                                              rocblas_stride    offset_b,
                                              int64_t           ldb,
                                              rocblas_stride    stride_b,
                                              TPtr*             dC,
                                              rocblas_stride    offset_c,
                                              int64_t           ldc,
                                              rocblas_stride    stride_c,
                                              rocblas_int       batch_count)
{
    // quick return
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    // rocblas_internal_trmm_outofplace_template() only supports single batches
    bool inplace = (dB == dC) || BATCHED || batch_count != 1;

    rocblas_int k = side == rocblas_side_left ? m : n;
    if(!inplace)
    {
        // always !BATCHED here so avoiding reference to uninstantiated code
        if constexpr(!BATCHED)
        {
            // single bathc so applying offset here
            rocblas_set_matrix_zero_if_alpha_zero_template(
                handle, m, n, &alpha_0<T>, 0, dC + offset_c, ldc, 0, 1);
            return rocblas_internal_trmm_outofplace_template<T>(handle,
                                                                side,
                                                                uplo,
                                                                trans_a,
                                                                diag,
                                                                m,
                                                                n,
                                                                alpha,
                                                                dA,
                                                                offset_a,
                                                                lda,
                                                                dB,
                                                                offset_b,
                                                                ldb,
                                                                dC,
                                                                offset_c,
                                                                ldc);
        }
    }
    else
    {
        return rocblas_internal_trmm_recursive_template<NB, BATCHED, T>(handle,
                                                                        side,
                                                                        uplo,
                                                                        trans_a,
                                                                        diag,
                                                                        m,
                                                                        n,
                                                                        alpha,
                                                                        stride_alpha,
                                                                        dA,
                                                                        offset_a,
                                                                        lda,
                                                                        stride_a,
                                                                        dB,
                                                                        offset_b,
                                                                        ldb,
                                                                        stride_b,
                                                                        dC,
                                                                        offset_c,
                                                                        ldc,
                                                                        stride_c,
                                                                        batch_count);
    }

    return rocblas_status_success;
}

#define TRMM_TEMPLATE_PARAMS                                                                       \
    handle, side, uplo, trans_a, diag, m, n, alpha, stride_alpha, dA, offset_a, lda, stride_a, dB, \
        offset_b, ldb, stride_b, dC, offset_c, ldc, stride_c, batch_count

template <typename T>
rocblas_status rocblas_internal_trmm_template(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans_a,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              rocblas_stride    stride_alpha,
                                              const T*          dA,
                                              rocblas_stride    offset_a,
                                              rocblas_int       lda,
                                              rocblas_stride    stride_a,
                                              const T*          dB,
                                              rocblas_stride    offset_b,
                                              rocblas_int       ldb,
                                              rocblas_stride    stride_b,
                                              T*                dC,
                                              rocblas_stride    offset_c,
                                              rocblas_int       ldc,
                                              rocblas_stride    stride_c,
                                              rocblas_int       batch_count)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trmm_launcher<ROCBLAS_SDTRMM_NB, false, T, T const, T const, T>(
            TRMM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trmm_launcher<ROCBLAS_SDTRMM_NB, false, T>(TRMM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trmm_launcher<ROCBLAS_CZTRMM_NB, false, T>(TRMM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trmm_launcher<ROCBLAS_CZTRMM_NB, false, T>(TRMM_TEMPLATE_PARAMS);

    return rocblas_status_internal_error;
}

template <typename T>
rocblas_status rocblas_internal_trmm_batched_template(rocblas_handle    handle,
                                                      rocblas_side      side,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation trans_a,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      rocblas_stride    stride_alpha,
                                                      const T* const*   dA,
                                                      rocblas_stride    offset_a,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const T* const*   dB,
                                                      rocblas_stride    offset_b,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      T* const*         dC,
                                                      rocblas_stride    offset_c,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trmm_launcher<ROCBLAS_SDTRMM_NB, true, T>(TRMM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trmm_launcher<ROCBLAS_SDTRMM_NB, true, T>(TRMM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trmm_launcher<ROCBLAS_CZTRMM_NB, true, T>(TRMM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trmm_launcher<ROCBLAS_CZTRMM_NB, true, T>(TRMM_TEMPLATE_PARAMS);

    return rocblas_status_internal_error;
}

#undef TRMM_TEMPLATE_PARAMS

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_trmm_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_side      side,
                                           rocblas_fill      uplo,
                                           rocblas_operation trans_a,
                                           int64_t           m,
                                           int64_t           n,
                                           TConstPtr*        A,
                                           int64_t           lda,
                                           rocblas_stride    stride_a,
                                           TPtr*             B,
                                           int64_t           ldb,
                                           rocblas_stride    stride_b,
                                           int64_t           batch_count,
                                           const int         check_numerics,
                                           bool              is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        rocblas_int rows = (side == rocblas_side_left ? m : n);
        rocblas_int cols = (side == rocblas_side_left ? m : n);
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans_a,
                                                              uplo,
                                                              rocblas_client_triangular_matrix,
                                                              rows,
                                                              cols,
                                                              A,
                                                              0,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          B,
                                                          0,
                                                          ldb,
                                                          stride_b,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files trmm*.cpp

// clang-format off
#ifdef INST_TRMM_LAUNCHER
#error INST_TRMM_LAUNCHER already defined
#endif

#define INST_TRMM_LAUNCHER(NB_, BATCHED_, T_, TConstPtr_, TPtr_)               \
    template rocblas_status rocblas_internal_trmm_launcher<NB_, BATCHED_, T_>( \
        rocblas_handle    handle,                                              \
        rocblas_side      side,                                                \
        rocblas_fill      uplo,                                                \
        rocblas_operation trans_a,                                             \
        rocblas_diagonal  diag,                                                \
        rocblas_int       m,                                                   \
        rocblas_int       n,                                                   \
        const T_*         alpha,                                               \
        rocblas_stride    stride_alpha,                                        \
        TConstPtr_        dA,                                                  \
        rocblas_stride    offset_a,                                            \
        int64_t           lda,                                                 \
        rocblas_stride    stride_a,                                            \
        TConstPtr_        dB,                                                  \
        rocblas_stride    offset_b,                                            \
        int64_t           ldb,                                                 \
        rocblas_stride    stride_b,                                            \
        TPtr_             dC,                                                  \
        rocblas_stride    offset_c,                                            \
        int64_t           lddc,                                                \
        rocblas_stride    stride_c,                                            \
        rocblas_int       batch_count);

INST_TRMM_LAUNCHER(ROCBLAS_SDTRMM_NB, false, float, const float*, float*)
INST_TRMM_LAUNCHER(ROCBLAS_SDTRMM_NB, false, double, const double*, double*)
INST_TRMM_LAUNCHER(ROCBLAS_CZTRMM_NB,
                   false,
                   rocblas_float_complex,
                   const rocblas_float_complex*,
                   rocblas_float_complex*)
INST_TRMM_LAUNCHER(ROCBLAS_CZTRMM_NB,
                   false,
                   rocblas_double_complex,
                   const rocblas_double_complex*,
                   rocblas_double_complex*)

INST_TRMM_LAUNCHER(ROCBLAS_SDTRMM_NB, true, float, const float* const*, float* const*)
INST_TRMM_LAUNCHER(ROCBLAS_SDTRMM_NB, true, double, const double* const*, double* const*)
INST_TRMM_LAUNCHER(ROCBLAS_CZTRMM_NB,
                   true,
                   rocblas_float_complex,
                   const rocblas_float_complex* const*,
                   rocblas_float_complex* const*)
INST_TRMM_LAUNCHER(ROCBLAS_CZTRMM_NB,
                   true,
                   rocblas_double_complex,
                   const rocblas_double_complex* const*,
                   rocblas_double_complex* const*)

#ifdef INSTANTIATE_TRMM_TEMPLATE
#error INSTANTIATE_TRMM_TEMPLATE already defined
#endif

#define INSTANTIATE_TRMM_TEMPLATE(T_)                                                       \
template rocblas_status rocblas_internal_trmm_template<T_>                                  \
                                  (rocblas_handle    handle,                                \
                                   rocblas_side      side,                                  \
                                   rocblas_fill      uplo,                                  \
                                   rocblas_operation trans_a,                               \
                                   rocblas_diagonal  diag,                                  \
                                   rocblas_int       m,                                     \
                                   rocblas_int       n,                                     \
                                   const T_*         alpha,                                 \
                                   rocblas_stride    stride_alpha,                          \
                                   const T_*         dA,                                    \
                                   rocblas_stride    offset_a,                              \
                                   rocblas_int       lda,                                   \
                                   rocblas_stride    stride_a,                              \
                                   const T_*         dB,                                    \
                                   rocblas_stride    offset_b,                              \
                                   rocblas_int       ldb,                                   \
                                   rocblas_stride    stride_b,                              \
                                   T_*               dC,                                    \
                                   rocblas_stride    offset_c,                              \
                                   rocblas_int       lddc,                                  \
                                   rocblas_stride    stride_c,                              \
                                   rocblas_int       batch_count);

INSTANTIATE_TRMM_TEMPLATE(float)
INSTANTIATE_TRMM_TEMPLATE(double)
INSTANTIATE_TRMM_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRMM_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRMM_TEMPLATE

#ifdef INSTANTIATE_TRMM_BATCHED_TEMPLATE
#error INSTANTIATE_TRMM_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_TRMM_BATCHED_TEMPLATE(T_)                                                       \
template rocblas_status rocblas_internal_trmm_batched_template<T_>                                  \
                                            (rocblas_handle    handle,                              \
                                            rocblas_side      side,                                 \
                                            rocblas_fill      uplo,                                 \
                                            rocblas_operation trans_a,                              \
                                            rocblas_diagonal  diag,                                 \
                                            rocblas_int       m,                                    \
                                            rocblas_int       n,                                    \
                                            const T_*         alpha,                                \
                                            rocblas_stride    stride_alpha,                         \
                                            const T_* const*  dA,                                   \
                                            rocblas_stride    offset_a,                             \
                                            rocblas_int       lda,                                  \
                                            rocblas_stride    stride_a,                             \
                                            const T_* const*  dB,                                   \
                                            rocblas_stride    offset_b,                             \
                                            rocblas_int       ldb,                                  \
                                            rocblas_stride    stride_b,                             \
                                            T_* const*        dC,                                   \
                                            rocblas_stride    offset_c,                             \
                                            rocblas_int       lddc,                                 \
                                            rocblas_stride    stride_c,                             \
                                            rocblas_int       batch_count);

INSTANTIATE_TRMM_BATCHED_TEMPLATE(float)
INSTANTIATE_TRMM_BATCHED_TEMPLATE(double)
INSTANTIATE_TRMM_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRMM_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRMM_BATCHED_TEMPLATE

#ifdef INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE
#error INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE already defined
#endif

#define INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(TScal_, TPtr_)     \
template rocblas_status rocblas_set_matrix_zero_if_alpha_zero_template  \
                        <TScal_, TPtr_>                         \
                        (rocblas_handle handle,                 \
                         rocblas_int    m,                      \
                         rocblas_int    n,                      \
                         TScal_         alpha,                  \
                         rocblas_stride stride_alpha,           \
                         TPtr_          A,                      \
                         int64_t        lda,                    \
                         rocblas_stride a_st_or_of,             \
                         rocblas_int    batch_count);

INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE( float const*,  float* const*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE( float const*,  float*)

INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(double const*, double*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(double const*, double* const*)

INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex* const*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE( rocblas_float_complex const*,  rocblas_float_complex*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex* const*)
INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex*)

#undef INSTANTIATE_SET_MATRIX_ZERO_TEMPLATE


#ifdef INSTANTIATE_TRMM_NUMERICS
#error INSTANTIATE_TRMM_NUMERICS already defined
#endif

#define INSTANTIATE_TRMM_NUMERICS(TConstPtr_, TPtr_)                                     \
template rocblas_status rocblas_trmm_check_numerics                                      \
                                  <TConstPtr_, TPtr_>                                    \
                                  (const char*       function_name,                      \
                                   rocblas_handle    handle,                             \
                                   rocblas_side      side,                               \
                                   rocblas_fill      uplo,                               \
                                   rocblas_operation trans_a,                            \
                                   int64_t           m,                                  \
                                   int64_t           n,                                  \
                                   TConstPtr_*       dA,                                 \
                                   int64_t           lda,                                \
                                   rocblas_stride    stride_a,                           \
                                   TPtr_*            dB,                                 \
                                   int64_t           ldb,                                \
                                   rocblas_stride    stride_b,                           \
                                   int64_t           batch_count,                        \
                                   const int         check_numerics,                     \
                                   bool              is_input);

// instantiate for rocblas_Xtrmm and rocblas_Xtrmm_strided_batched
INSTANTIATE_TRMM_NUMERICS(float const,  float)
INSTANTIATE_TRMM_NUMERICS(double const, double)
INSTANTIATE_TRMM_NUMERICS(rocblas_float_complex const, rocblas_float_complex)
INSTANTIATE_TRMM_NUMERICS(rocblas_double_complex const, rocblas_double_complex)

// instantiate for rocblas_Xtrmm_batched
INSTANTIATE_TRMM_NUMERICS(float const* const,  float* const)
INSTANTIATE_TRMM_NUMERICS(double const* const, double* const)
INSTANTIATE_TRMM_NUMERICS(rocblas_float_complex const* const, rocblas_float_complex* const)
INSTANTIATE_TRMM_NUMERICS(rocblas_double_complex const* const, rocblas_double_complex* const)

// instantiate for rocblas_Xtrmm_outofplace and rocblas_Xtrmm_outofplace_strided_batched
INSTANTIATE_TRMM_NUMERICS(float const,  float const)
INSTANTIATE_TRMM_NUMERICS(double const, double const)
INSTANTIATE_TRMM_NUMERICS(rocblas_float_complex const, rocblas_float_complex const)
INSTANTIATE_TRMM_NUMERICS(rocblas_double_complex const, rocblas_double_complex const)

// instantiate for rocblas_Xtrmm_outofplace_batched
INSTANTIATE_TRMM_NUMERICS(float const* const,  float const* const)
INSTANTIATE_TRMM_NUMERICS(double const* const, double const* const)
INSTANTIATE_TRMM_NUMERICS(rocblas_float_complex const* const, rocblas_float_complex const* const)
INSTANTIATE_TRMM_NUMERICS(rocblas_double_complex const* const, rocblas_double_complex const* const)

#undef INSTANTIATE_TRMM_NUMERICS
// clang-format on
