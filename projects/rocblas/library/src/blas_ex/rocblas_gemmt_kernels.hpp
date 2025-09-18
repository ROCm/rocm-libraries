/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas3/rocblas_gemm.hpp"
#include "definitions.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_blas_ex_threshold.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_gemmt.hpp"
#include "src64/blas3/rocblas_gemm_64.hpp" // int64 API called
#include "utility.hpp"

template <typename API_INT,
          int  DIM_N,
          int  BLK_N,
          int  BLK_K,
          char TRANSA,
          char TRANSB,
          char UPLO,
          bool HERM_A,
          bool HERM_B,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_N* DIM_N)
rocblas_internal_gemmt_kernel(rocblas_int    N,
                              API_INT        K,
                              TScal          alpha_device_host,
                              TConstPtr      dA_array,
                              API_INT        lda,
                              rocblas_stride stride_a,
                              TConstPtr      dB_array,
                              API_INT        ldb,
                              rocblas_stride stride_b,
                              TScal          beta_device_host,
                              TPtr           dC_array,
                              API_INT        ldc,
                              rocblas_stride stride_c,
                              rocblas_int    batch_count)
{
    auto alpha = load_scalar(alpha_device_host);
    auto beta  = load_scalar(beta_device_host);

    if(beta == 1 && (K == 0 || alpha == 0))
        return;

    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_N * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int thxA = idt % BLK_N; // thread's m position for loading A
    int thyA = idt / BLK_N; // thread's n position for loading A
    int thxB = idt % BLK_K; // thread's m position for loading B
    int thyB = idt / BLK_K; // thread's n position for loading B

    uint32_t blz = blockIdx.z; // block's matrix in the batch

#if DEVICE_GRID_YZ_16BIT
    for(; blz < batch_count; blz += c_YZ_grid_launch_limit)
    {
#endif

        auto* dA = load_ptr_batch(dA_array, blz, stride_a);
        auto* dB = load_ptr_batch(dB_array, blz, stride_b);
        auto* dC = load_ptr_batch(dC_array, blz, stride_c);

        __shared__ T sA[BLK_K][BLK_N]; // shared memory for A
        __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
        T            rC[BLK_N / DIM_N][BLK_N / DIM_N]; // registers for C

        int a_i_offset = thxA + BLK_N * blx;
        int a_j_offset = thyA;
        int b_i_offset = thxB;
        int b_j_offset = thyB + BLK_N * bly;

        for(int n = 0; n < BLK_N / DIM_N; ++n)
            for(int m = 0; m < BLK_N / DIM_N; ++m)
                rC[n][m] = 0.0;

        if(alpha != T(0))
        {
            for(int kk = 0; kk < K; kk += BLK_K)
            {
                int i = a_i_offset;
                int j = kk + a_j_offset;
                if(i < N && j < K)
                {
                    if(TRANSA == 'N')
                        sA[thyA][thxA] = dA[i + j * size_t(lda)];
                    if(TRANSA == 'T')
                        sA[thyA][thxA] = dA[i * size_t(lda) + j];
                    if(TRANSA == 'C')
                        sA[thyA][thxA] = conj_if_true<HERM_A>(dA[i * size_t(lda) + j]);
                }
                else
                {
                    sA[thyA][thxA] = 0.0;
                }
                i = kk + b_i_offset;
                j = b_j_offset;
                if(i < K && j < N)
                {
                    if(TRANSB == 'N')
                        sB[thyB][thxB] = dB[i + j * size_t(ldb)];
                    if(TRANSB == 'T')
                        sB[thyB][thxB] = dB[i * size_t(ldb) + j];
                    if(TRANSB == 'C')
                        sB[thyB][thxB] = conj_if_true<HERM_B>(dB[i * size_t(ldb) + j]);
                }
                else
                {
                    sB[thyB][thxB] = 0;
                }

                __syncthreads();

                for(int k = 0; k < BLK_K; ++k)
                    for(int n = 0; n < BLK_N / DIM_N; ++n)
                        for(int m = 0; m < BLK_N / DIM_N; ++m)
                            rC[n][m] += sA[k][m * DIM_N + thx] * sB[n * DIM_N + thy][k];

                __syncthreads();
            }
        }
        for(int n = 0; n < BLK_N / DIM_N; ++n)
        {
            for(int m = 0; m < BLK_N / DIM_N; ++m)
            {
                int coord_dCm = blx * BLK_N + m * DIM_N + thx;
                int coord_dCn = bly * BLK_N + n * DIM_N + thy;
                if((UPLO == 'L' && coord_dCn <= coord_dCm && coord_dCm < N)
                   || (UPLO == 'U' && coord_dCm <= coord_dCn && coord_dCn < N))
                {
                    if(beta == 0)
                        dC[coord_dCn * size_t(ldc) + coord_dCm] = alpha * rC[n][m];
                    else
                        dC[coord_dCn * size_t(ldc) + coord_dCm]
                            = alpha * rC[n][m] + beta * dC[coord_dCn * size_t(ldc) + coord_dCm];
                }
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_gemmt_general_template(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_operation transB,
                                                       rocblas_int       n,
                                                       API_INT           k,
                                                       const TScal*      alpha,
                                                       TConstPtr         dA,
                                                       API_INT           lda,
                                                       rocblas_stride    stride_a,
                                                       TConstPtr         dB,
                                                       API_INT           ldb,
                                                       rocblas_stride    stride_b,
                                                       const TScal*      beta,
                                                       TPtr              dC,
                                                       API_INT           ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count)
{
    hipStream_t stream  = handle->get_stream();
    int         batches = handle->getBatchGridDim((int)batch_count);

    constexpr bool rocblas_is_complex
        = std::is_same_v<TScal,
                         rocblas_float_complex> || std::is_same_v<TScal, rocblas_double_complex>;

    const int dim_n = 16;
    const int blk_n = 32;
    const int blk_k = 8;
    dim3      dimBlock(dim_n, dim_n);
    dim3      dimGrid(((n - 1) / blk_n) + 1, ((n - 1) / blk_n) + 1, batches);

#define ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha_, beta_)                                     \
    dimGrid, dimBlock, 0, stream, n, k, alpha_, dA, lda, stride_a, dB, ldb, stride_b, beta_, dC, \
        ldc, stride_c, batch_count
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        if(uplo == rocblas_fill_upper)
        {
            if(transA == rocblas_operation_none && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'N',
                                                                     'U',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_none && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'T',
                                                                     'U',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_none
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'C',
                                                                     'U',
                                                                     false,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'N',
                                                                     'U',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'T',
                                                                     'U',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'C',
                                                                     'U',
                                                                     false,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'N',
                                                                     'U',
                                                                     rocblas_is_complex,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'T',
                                                                     'U',
                                                                     rocblas_is_complex,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'C',
                                                                     'U',
                                                                     rocblas_is_complex,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
        }
        else
        {
            if(transA == rocblas_operation_none && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'N',
                                                                     'L',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_none && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'T',
                                                                     'L',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_none
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'C',
                                                                     'L',
                                                                     false,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'N',
                                                                     'L',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'T',
                                                                     'L',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'C',
                                                                     'L',
                                                                     false,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'N',
                                                                     'L',
                                                                     rocblas_is_complex,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'T',
                                                                     'L',
                                                                     rocblas_is_complex,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'C',
                                                                     'L',
                                                                     rocblas_is_complex,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(alpha, beta));
        }
    }
    //pointer mode host
    else
    {
        if(uplo == rocblas_fill_upper)
        {
            if(transA == rocblas_operation_none && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'N',
                                                                     'U',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_none && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'T',
                                                                     'U',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_none
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'C',
                                                                     'U',
                                                                     false,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'N',
                                                                     'U',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'T',
                                                                     'U',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'C',
                                                                     'U',
                                                                     false,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'N',
                                                                     'U',
                                                                     rocblas_is_complex,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'T',
                                                                     'U',
                                                                     rocblas_is_complex,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'C',
                                                                     'U',
                                                                     rocblas_is_complex,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
        }
        else
        {
            if(transA == rocblas_operation_none && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'N',
                                                                     'L',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_none && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'T',
                                                                     'L',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_none
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'N',
                                                                     'C',
                                                                     'L',
                                                                     false,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'N',
                                                                     'L',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'T',
                                                                     'L',
                                                                     false,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'T',
                                                                     'C',
                                                                     'L',
                                                                     false,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'N',
                                                                     'L',
                                                                     rocblas_is_complex,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'T',
                                                                     'L',
                                                                     rocblas_is_complex,
                                                                     false,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
            else if(transA == rocblas_operation_conjugate_transpose
                    && transB == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_internal_gemmt_kernel<API_INT,
                                                                     dim_n,
                                                                     blk_n,
                                                                     blk_k,
                                                                     'C',
                                                                     'C',
                                                                     'L',
                                                                     rocblas_is_complex,
                                                                     rocblas_is_complex,
                                                                     TScal>),
                                      ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS(*alpha, *beta));
        }
    }
#undef ROCBLAS_INTERNAL_GEMMT_GENERAL_PARAMS

    return rocblas_status_success;
}

#define OFFSET_A(i1) i1* rocblas_stride(a_s1)
#define OFFSET_B(i1) i1* rocblas_stride(b_s1)
#define OFFSET_C(i1, i2) i1* rocblas_stride(c_s1) + i2* rocblas_stride(c_s2)

template <typename API_INT,
          rocblas_int MIN_NB,
          bool        BATCHED,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_gemmt_non_batch_block_recursive_template(rocblas_handle    handle,
                                                                         rocblas_fill      uplo,
                                                                         rocblas_operation transA,
                                                                         rocblas_operation transB,
                                                                         rocblas_int       n,
                                                                         API_INT           k,
                                                                         const TScal*      alpha,
                                                                         TConstPtr         dA,
                                                                         API_INT           lda,
                                                                         rocblas_stride    stride_a,
                                                                         TConstPtr         dB,
                                                                         API_INT           ldb,
                                                                         rocblas_stride    stride_b,
                                                                         const TScal*      beta,
                                                                         TPtr              dC,
                                                                         API_INT           ldc,
                                                                         rocblas_stride    stride_c,
                                                                         rocblas_int batch_count)
{
    // quick return
    if(!n)
        return rocblas_status_success;

    rocblas_stride a_s1 = rocblas_operation_none == transA ? 1 : lda;
    rocblas_stride b_s1 = rocblas_operation_none == transB ? ldb : 1;
    rocblas_stride c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    // call rocblas_internal_gemmt_general_template with batch_count = n_nb for n_nb diagonal blocks
    // clang-format off
    rocblas_internal_gemmt_general_template<API_INT>(handle, uplo, transA, transB, nb, k, alpha,
                         dA, lda, nb * a_s1,
                         dB, ldb, nb * b_s1, beta,
                         dC, ldc, nb * (c_s1 + c_s2), n_nb);
    // clang-format on

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;
        // call rocblas_internal_gemmt_general_template for one remainder diagonal block of size n_diag
        // clang-format off
        rocblas_internal_gemmt_general_template<API_INT>(handle, uplo, transA, transB, n_diag, k, alpha,
                          dA + i_diag * a_s1, lda, stride_a,
                          dB + i_diag * b_s1, ldb, stride_b, beta,
                          dC + i_diag * (c_s1 + c_s2), ldc, stride_c, batch_count);
        // clang-format on
    }

    // calls to gemm with m == n == nb.
    // Start with nb == MIN_NB, then for each iteration of nb,i_start loop:
    // - nb doubles
    // - the number of gemm calls in the inner loop halves.
    for(nb = MIN_NB, i_start = MIN_NB; i_start < n; i_start += nb, nb *= 2)
    {
        rocblas_int stride = nb * 2;
        n_nb               = (n - i_start) / stride;
        rem                = (n - i_start) % stride;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }

        // call gemm with batch_count = n_nb for n_nb square blocks of size nb x nb
        if(rocblas_fill_lower == uplo)
        {
            // clang-format off
            RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                 handle, transA, transB, nb, nb, k, alpha,
                 dA, OFFSET_A(i_start),    lda, stride * a_s1,
                 dB, OFFSET_B(0),          ldb, stride * b_s1, beta,
                 dC, OFFSET_C(i_start, 0), ldc, stride * (c_s1 + c_s2), n_nb   )));
            // clang-format on
        }
        else
        {
            // clang-format off
            RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                 handle, transA, transB, nb, nb, k, alpha,
                 dA, OFFSET_A(0),          lda, stride * a_s1,
                 dB, OFFSET_B(i_start),    ldb, stride * b_s1, beta,
                 dC, OFFSET_C(0, i_start), ldc, stride * (c_s1 + c_s2), n_nb)));
            // clang-format on
        }

        // call gemm for remainder block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_stride i1 = i_start + n_nb * stride;
            rocblas_stride i2 = i1 - nb;
            rocblas_stride n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, transA, transB, n1, nb, k, alpha,
                     dA, OFFSET_A(i1),     lda, stride_a,
                     dB, OFFSET_B(i2),     ldb, stride_b, beta,
                     dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, transA, transB, nb, n1, k, alpha,
                     dA, OFFSET_A(i2),     lda, stride_a,
                     dB, OFFSET_B(i1),     ldb, stride_b, beta,
                     dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }
    }
    return rocblas_status_success;
}

template <typename API_INT,
          rocblas_int MIN_NB,
          bool        BATCHED,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_gemmt_batched_strided_batched_block_recursive_template(
    rocblas_handle    handle,
    rocblas_fill      uplo,
    rocblas_operation transA,
    rocblas_operation transB,
    rocblas_int       n,
    API_INT           k,
    const TScal*      alpha,
    TConstPtr         dA,
    API_INT           lda,
    rocblas_stride    stride_a,
    TConstPtr         dB,
    API_INT           ldb,
    rocblas_stride    stride_b,
    const TScal*      beta,
    TPtr              dC,
    API_INT           ldc,
    rocblas_stride    stride_c,
    rocblas_int       batch_count)
{
    if(k == 0)
        return rocblas_status_success;

    rocblas_int a_s1 = rocblas_operation_none == transA ? 1 : lda;
    rocblas_int b_s1 = rocblas_operation_none == transB ? ldb : 1;
    rocblas_int c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    // n_nb diagonal blocks of size nb
    for(int i_nb = 0; i_nb < n_nb; i_nb++)
    {
        i_diag = i_nb * nb; // diag block at c[i_diag, i_diag], size is nb

        // clang-format off
        if(BATCHED)
            rocblas_internal_gemmt_general_template<API_INT>(handle, uplo, transA, transB, nb, k, alpha,
                         dA, lda, OFFSET_A(i_diag),
                         dB, ldb, OFFSET_B(i_diag), beta,
                         dC, ldc, OFFSET_C(i_diag, i_diag), batch_count);
        else
            rocblas_internal_gemmt_general_template<API_INT>(handle, uplo, transA, transB, nb, k, alpha,
                         dA + OFFSET_A(i_diag), lda, stride_a,
                         dB + OFFSET_B(i_diag), ldb, stride_b, beta,
                         dC + OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;

        // clang-format off
        if(BATCHED)
            rocblas_internal_gemmt_general_template<API_INT>(handle, uplo, transA, transB, n_diag, k, alpha,
                         dA, lda, OFFSET_A(i_diag),
                         dB, ldb, OFFSET_B(i_diag), beta,
                         dC, ldc, OFFSET_C(i_diag, i_diag), batch_count);
        else
            rocblas_internal_gemmt_general_template<API_INT>(handle, uplo, transA, transB, n_diag, k, alpha,
                         dA + OFFSET_A(i_diag), lda, stride_a,
                         dB + OFFSET_B(i_diag), ldb, stride_b, beta,
                         dC + OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    // calls to gemm with m == n == nb.
    // Start with nb == MIN_NB, and each iteration of the outer loop:
    // - nb doubles
    // - the number of gemm calls in the inner loop halves.
    for(nb = MIN_NB, i_start = MIN_NB; i_start < n; i_start += nb, nb *= 2)
    {
        rocblas_int stride = nb * 2;
        n_nb               = (n - i_start) / stride;
        rem                = (n - i_start) % stride;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }
        // n_nb gemm blocks of size nb x nb
        for(int i = 0; i < n_nb; i++)
        {
            rocblas_int i1 = i_start + (i * stride);
            rocblas_int i2 = i1 - nb;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, transA, transB, nb, nb, k, alpha,
                     dA, OFFSET_A(i1),     lda, stride_a,
                     dB, OFFSET_B(i2),     ldb, stride_b, beta,
                     dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, transA, transB, nb, nb, k, alpha,
                     dA, OFFSET_A(i2),     lda, stride_a,
                     dB, OFFSET_B(i1),     ldb, stride_b, beta,
                     dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }

        // remainder gemm block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_int i1 = i_start + n_nb * stride;
            rocblas_int i2 = i1 - nb;
            rocblas_int n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, transA, transB, n1, nb, k, alpha,
                     dA, OFFSET_A(i1),     lda, stride_a,
                     dB, OFFSET_B(i2),     ldb, stride_b, beta,
                     dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, transA, transB, nb, n1, k, alpha,
                     dA, OFFSET_A(i2),     lda, stride_a,
                     dB, OFFSET_B(i1),     ldb, stride_b, beta,
                     dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on
            }
        }
    }
    return rocblas_status_success;
}

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_gemmt_launcher(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_operation transB,
                                               rocblas_int       n,
                                               API_INT           k,
                                               const TScal*      alpha_in,
                                               TConstPtr         dA,
                                               API_INT           lda,
                                               rocblas_stride    stride_a,
                                               TConstPtr         dB,
                                               API_INT           ldb,
                                               rocblas_stride    stride_b,
                                               const TScal*      beta_in,
                                               TPtr              dC,
                                               API_INT           ldc,
                                               rocblas_stride    stride_c,
                                               rocblas_int       batch_count)
{
    //Identifying the precision to have an appropriate optimization
    static constexpr bool is_float                  = std::is_same_v<TScal, float>;
    static constexpr bool is_double                 = std::is_same_v<TScal, double>;
    static constexpr bool rocblas_is_complex_float  = std::is_same_v<TScal, rocblas_float_complex>;
    static constexpr bool rocblas_is_complex_double = std::is_same_v<TScal, rocblas_double_complex>;

    // GEMM based block recursive algorithm
    if((n >= n_zgemmt_threshold && k >= k_zgemmt_threshold && rocblas_is_complex_double)
       || (n >= n_gemmt_threshold && k >= k_gemmt_threshold
           && (is_float || is_double || rocblas_is_complex_float)))
    {
        // BATCHED is true for _batched and false for _strided_batched and non-batched
        constexpr bool BATCHED
            = std::is_same_v<
                  TConstPtr,
                  const float* const*> || std::is_same_v<TConstPtr, const double* const*> || std::is_same_v<TConstPtr, const rocblas_float_complex* const*> || std::is_same_v<TConstPtr, const rocblas_double_complex* const*>;

        // Copy over alpha and beta
        TScal alpha_h, beta_h;
        RETURN_IF_ROCBLAS_ERROR(rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha_in, beta_in, alpha_h, beta_h, k));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        // Note: alpha and beta always copied over to host by now
        if(*beta_in == 1 && (k == 0 || *alpha_in == 0))
            return rocblas_status_success;

        bool ab_calc_invalid = !alpha_in || (*alpha_in != 0 && (!dA || !dB));
        if(!dC || (k && ab_calc_invalid))
            return rocblas_status_invalid_pointer;

        // upgrade to complex if needed
        // TODO: Graph safety?
        const TScal alpha_val = (TScal)(*alpha_in);
        const TScal beta_val  = (TScal)(*beta_in);

        const TScal* alpha = &alpha_val;
        const TScal* beta  = &beta_val;

#define ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS                                                \
    handle, uplo, transA, transB, n, k, alpha, dA, lda, stride_a, dB, ldb, stride_b, beta, dC, \
        ldc, stride_c, batch_count

        if(!BATCHED && batch_count == 1)
        {
            //using similar number of blocks as that of syr2k
            if constexpr(std::is_same_v<TScal, float>)
                return rocblas_internal_gemmt_non_batch_block_recursive_template<API_INT,
                                                                                 ROCBLAS_SSYR2K_NB,
                                                                                 BATCHED>(
                    ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS);
            else if constexpr(std::is_same_v<TScal, double>)
                return rocblas_internal_gemmt_non_batch_block_recursive_template<API_INT,
                                                                                 ROCBLAS_DSYR2K_NB,
                                                                                 BATCHED>(
                    ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS);
            else if constexpr(std::is_same_v<TScal, rocblas_float_complex>)
                return rocblas_internal_gemmt_non_batch_block_recursive_template<API_INT,
                                                                                 ROCBLAS_CSYR2K_NB,
                                                                                 BATCHED>(
                    ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS);
            else if constexpr(std::is_same_v<TScal, rocblas_double_complex>)
                return rocblas_internal_gemmt_non_batch_block_recursive_template<API_INT,
                                                                                 ROCBLAS_ZSYR2K_NB,
                                                                                 BATCHED>(
                    ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS);
        }
        else
        {
            //using similar number of blocks as that of syr2k
            if constexpr(std::is_same_v<TScal, float>)
                return rocblas_internal_gemmt_batched_strided_batched_block_recursive_template<
                    API_INT,
                    ROCBLAS_SSYR2K_NB,
                    BATCHED>(ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS);
            else if constexpr(std::is_same_v<TScal, double>)
                return rocblas_internal_gemmt_batched_strided_batched_block_recursive_template<
                    API_INT,
                    ROCBLAS_DSYR2K_NB,
                    BATCHED>(ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS);
            else if constexpr(std::is_same_v<TScal, rocblas_float_complex>)
                return rocblas_internal_gemmt_batched_strided_batched_block_recursive_template<
                    API_INT,
                    ROCBLAS_CSYR2K_NB,
                    BATCHED>(ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS);
            else if constexpr(std::is_same_v<TScal, rocblas_double_complex>)
                return rocblas_internal_gemmt_batched_strided_batched_block_recursive_template<
                    API_INT,
                    ROCBLAS_ZSYR2K_NB,
                    BATCHED>(ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS);
#undef ROCBLAS_INTERNAL_GEMMT_RECURSIVE_PARAMS
        }
    }
    else
    {
#define ROCBLAS_INTERNAL_GEMMT_PARAMS                                                            \
    handle, uplo, transA, transB, n, k, alpha_in, dA, lda, stride_a, dB, ldb, stride_b, beta_in, \
        dC, ldc, stride_c, batch_count

        return rocblas_internal_gemmt_general_template<API_INT>(ROCBLAS_INTERNAL_GEMMT_PARAMS);

#undef ROCBLAS_INTERNAL_GEMMT_PARAMS
    }
    return rocblas_status_success;
}
