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

#include "device_macros.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"

namespace
{

    template <int DIM_X, int DIM_Y, typename T, typename U, typename V>
    ROCBLAS_KERNEL_ILF void gemm_ex_scale_device(
        rocblas_int m, rocblas_int n, T beta, U* C, int64_t ldc, V* D, int64_t ldd)
    {
        auto tx = blockIdx.x * DIM_X + threadIdx.x;
        auto ty = blockIdx.y * DIM_Y + threadIdx.y;

        if(tx < m && ty < n)
        {
            D[ty * (ldd) + tx] = beta ? V(beta * C[ty * (ldc) + tx]) : V(0);
        }
    }

    /**
     *  Loads pointers and launches the actual calculation kernel.
     */
    template <int DIM_X, int DIM_Y, typename T, typename U, typename V>
    ROCBLAS_KERNEL(DIM_X* DIM_Y)
    gemm_ex_scale_kernel(rocblas_int    m,
                         rocblas_int    n,
                         T              beta_host_device,
                         U              CP_array,
                         rocblas_stride shift_c,
                         int64_t        ldc,
                         rocblas_stride stride_c,
                         V              DP_array,
                         rocblas_stride shift_d,
                         int64_t        ldd,
                         rocblas_stride stride_d,
                         rocblas_int    batch_count)
    {
        auto beta = load_scalar(beta_host_device);

        uint32_t batch = blockIdx.z;
#if DEVICE_GRID_YZ_16BIT
        for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
        {
#endif

            auto C = cond_load_ptr_batch(beta != 0, CP_array, batch, shift_c, stride_c);
            auto D = load_ptr_batch(DP_array, batch, shift_d, stride_d);
            gemm_ex_scale_device<DIM_X, DIM_Y>(m, n, beta, C, ldc, D, ldd);

#if DEVICE_GRID_YZ_16BIT
        }
#endif
    }

    template <typename TScal, typename TConstPtr, typename TPtr>
    rocblas_status rocblas_gemm_ex_scale_launcher_64(rocblas_handle __restrict__ handle,
                                                     int64_t        m_64,
                                                     int64_t        n_64,
                                                     TScal          beta,
                                                     TConstPtr      C,
                                                     rocblas_stride offset_c,
                                                     int64_t        ldc_64,
                                                     rocblas_stride stride_c,
                                                     TPtr           D,
                                                     rocblas_stride offset_d,
                                                     int64_t        ldd_64,
                                                     rocblas_stride stride_d,
                                                     rocblas_int    batch_count)
    {
        static constexpr int GEMM_DIM_X = 32;
        static constexpr int GEMM_DIM_Y = 32;

        hipStream_t rocblas_stream = handle->get_stream();
        int         batches        = handle->getBatchGridDim((int)batch_count);

        for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
        {
            int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

            int64_t n_shift   = n_base * ldc_64;
            int64_t d_n_shift = n_base * ldd_64;

            int blocksY = (n - 1) / GEMM_DIM_Y + 1;

            for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
            {
                int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                int blocksX = (m - 1) / GEMM_DIM_X + 1;

                dim3 gemm_grid(blocksX, blocksY, batches);
                dim3 gemm_threads(GEMM_DIM_X, GEMM_DIM_Y);

                ROCBLAS_LAUNCH_KERNEL((gemm_ex_scale_kernel<GEMM_DIM_X, GEMM_DIM_Y>),
                                      gemm_grid,
                                      gemm_threads,
                                      0,
                                      rocblas_stream,
                                      m,
                                      n,
                                      beta,
                                      C,
                                      offset_c + n_shift + m_base,
                                      ldc_64,
                                      stride_c,
                                      D,
                                      offset_d + d_n_shift + m_base,
                                      ldd_64,
                                      stride_d,
                                      batch_count);

            } // m
        } // n

        return rocblas_status_success;
    }

    // Special gemm kernel when K == 0 or alpha == 0
    template <int DIM_X, int DIM_Y, typename T, typename U>
    ROCBLAS_KERNEL_ILF void
        rocblas_gemm_scale_device(rocblas_int m, rocblas_int n, T beta, U* C, int64_t ldc)
    {
        auto tx = blockIdx.x * DIM_X + threadIdx.x;
        auto ty = blockIdx.y * DIM_Y + threadIdx.y;

        if(tx < m && ty < n)
        {
            C[ty * (ldc) + tx] = U(beta ? (beta * C[ty * (ldc) + tx]) : T(0));
        }
    }

    /**
  *  Loads pointers and launches the actual calculation kernel.
  */
    template <int DIM_X, int DIM_Y, typename T, typename TPtr>
    ROCBLAS_KERNEL(DIM_X* DIM_Y)
    rocblas_gemm_scale_kernel(rocblas_int    m,
                              rocblas_int    n,
                              T              beta_host_device,
                              TPtr           dC,
                              rocblas_stride shift_c,
                              int64_t        ldc,
                              rocblas_stride stride_c,
                              rocblas_int    batch_count)
    {
        auto     beta  = load_scalar(beta_host_device);
        uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
        for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
        {
#endif
            auto C = load_ptr_batch(dC, batch, shift_c, stride_c);
            rocblas_gemm_scale_device<DIM_X, DIM_Y>(m, n, beta, C, ldc);

#if DEVICE_GRID_YZ_16BIT
        }
#endif
    }

    template <typename TScal, typename TConstPtr>
    rocblas_status rocblas_gemm_scale_launcher_64(rocblas_handle __restrict__ handle,
                                                  int64_t        m_64,
                                                  int64_t        n_64,
                                                  TScal          beta,
                                                  TConstPtr      C,
                                                  rocblas_stride offset_c,
                                                  int64_t        ldc_64,
                                                  rocblas_stride stride_c,
                                                  rocblas_int    batch_count)
    {
        static constexpr int GEMM_DIM_X = 32;
        static constexpr int GEMM_DIM_Y = 32;

        hipStream_t rocblas_stream = handle->get_stream();
        int         batches        = handle->getBatchGridDim((int)batch_count);

        for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
        {
            int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

            int64_t n_shift = n_base * ldc_64;

            int blocksY = (n - 1) / GEMM_DIM_Y + 1;

            for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
            {
                int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                int blocksX = (m - 1) / GEMM_DIM_X + 1;

                dim3 gemm_grid(blocksX, blocksY, batches);
                dim3 gemm_threads(GEMM_DIM_X, GEMM_DIM_Y);

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_scale_kernel<GEMM_DIM_X, GEMM_DIM_Y>),
                                      gemm_grid,
                                      gemm_threads,
                                      0,
                                      rocblas_stream,
                                      m,
                                      n,
                                      beta,
                                      C,
                                      offset_c + n_shift + m_base,
                                      ldc_64,
                                      stride_c,
                                      batch_count);

            } // m
        } // n

        return rocblas_status_success;
    }

    // general alpha, beta, m, n, k
    template <typename Tc,
              int  DIM_M,
              int  DIM_N,
              int  BLK_M,
              int  BLK_N,
              int  BLK_K,
              int  DIM_M_A,
              int  DIM_N_A,
              int  DIM_M_B,
              int  DIM_N_B,
              char TRANS_A,
              char TRANS_B,
              typename TiConstPtr,
              typename ToConstPtr,
              typename ToPtr>
    ROCBLAS_KERNEL(DIM_M* DIM_N)
    rocblas_gemm_batched_general_kernel(int64_t        M,
                                        int64_t        N,
                                        int64_t        K,
                                        const Tc       alpha,
                                        TiConstPtr*    dA_input,
                                        int64_t        lda,
                                        rocblas_stride a_st_or_of,
                                        TiConstPtr*    dB_input,
                                        int64_t        ldb,
                                        rocblas_stride b_st_or_of,
                                        const Tc       beta,
                                        ToConstPtr*    dC_input,
                                        int64_t        ldc,
                                        rocblas_stride c_st_or_of,
                                        ToPtr*         dD_input,
                                        int64_t        ldd,
                                        rocblas_stride d_st_or_of,
                                        rocblas_int    batch_count)
    {
        int     thx = threadIdx.x; // thread's m position in C
        int     thy = threadIdx.y; // thread's n position in C
        int64_t idt = int64_t(DIM_M) * thy + thx; // thread's number
        int     blx = blockIdx.x; // block's m position
        int     bly = blockIdx.y; // block's n position
        int     blz = blockIdx.z; // block's matrix in the batch

#if DEVICE_GRID_YZ_16BIT
        for(; blz < batch_count; blz += c_YZ_grid_launch_limit)
        {
#endif

            auto* dA = load_ptr_batch(dA_input, blz, a_st_or_of);
            auto* dB = load_ptr_batch(dB_input, blz, b_st_or_of);
            auto* dC = load_ptr_batch(dC_input, blz, c_st_or_of);
            auto* dD = load_ptr_batch(dD_input, blz, d_st_or_of);

            auto tmp = *dD;
            using To = decltype(tmp);

            __shared__ Tc sA[BLK_K][BLK_M]; // shared memory for A
            __shared__ Tc sB[BLK_N][BLK_K]; // shared memory for B
            Tc            rD[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for D

            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                    rD[n][m] = 0.0;

            {
                int     thxA = idt % DIM_M_A; // thread's m position for loading A
                int64_t thyA = idt / DIM_M_A; // thread's n position for loading A
                int     thxB = idt % DIM_M_B; // thread's m position for loading B
                int64_t thyB = idt / DIM_M_B; // thread's n position for loading B

                int64_t a_i_offset = thxA + int64_t(BLK_M) * blx;
                int64_t a_j_offset = thyA;
                int64_t b_i_offset = thxB;
                int64_t b_j_offset = thyB + int64_t(BLK_N) * bly;

                int64_t kk = 0;
                for(; kk < K; kk += BLK_K)
                {
                    for(int n = 0; n < BLK_K; n += DIM_N_A)
                    {
                        for(int m = 0; m < BLK_M; m += DIM_M_A)
                        {
                            int64_t i = m + a_i_offset;
                            int64_t j = n + kk + a_j_offset;
                            if(i < M && j < K)
                            {
                                if(TRANS_A == 'N')
                                {
                                    sA[n + thyA][m + thxA] = dA[i + j * size_t(lda)];
                                }
                                else if(TRANS_A == 'T')
                                {
                                    sA[n + thyA][m + thxA] = dA[i * size_t(lda) + j];
                                }
                                else if(TRANS_A == 'C')
                                {
                                    sA[n + thyA][m + thxA] = conj(dA[i * size_t(lda) + j]);
                                }
                            }
                            else
                            {
                                sA[n + thyA][m + thxA] = 0.0;
                            }
                        }
                    }

                    for(int n = 0; n < BLK_N; n += DIM_N_B)
                    {
                        for(int m = 0; m < BLK_K; m += DIM_M_B)
                        {
                            int64_t i = m + kk + b_i_offset;
                            int64_t j = n + b_j_offset;
                            if(i < K && j < N)
                            {
                                if(TRANS_B == 'N')
                                {
                                    sB[n + thyB][m + thxB] = dB[i + j * size_t(ldb)];
                                }
                                else if(TRANS_B == 'T')
                                {
                                    sB[n + thyB][m + thxB] = dB[i * size_t(ldb) + j];
                                }
                                else if(TRANS_B == 'C')
                                {
                                    sB[n + thyB][m + thxB] = conj(dB[i * size_t(ldb) + j]);
                                }
                            }
                            else
                            {
                                sB[n + thyB][m + thxB] = 0;
                            }
                        }
                    }

                    __syncthreads();

                    for(int k = 0; k < BLK_K; ++k)
                        for(int n = 0; n < BLK_N / DIM_N; ++n)
                            for(int m = 0; m < BLK_M / DIM_M; ++m)
                                rD[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

                    __syncthreads();
                }
            }

            int64_t coord_dCn = int64_t(bly) * BLK_N + thy;
            if(beta != Tc(0))
            {
                for(int n = 0; n < BLK_N / DIM_N && coord_dCn < N; ++n, coord_dCn += DIM_N)
                {
                    int64_t nCIdx     = coord_dCn * size_t(ldc);
                    int64_t nDIdx     = coord_dCn * size_t(ldd);
                    int64_t coord_dCm = int64_t(blx) * BLK_M + thx;

#pragma unroll
                    for(int m = 0; m < BLK_M / DIM_M; ++m, coord_dCm += DIM_M)
                    {
                        if(coord_dCm < M)
                            dD[nDIdx + coord_dCm]
                                = To(alpha * rD[n][m] + beta * dC[nCIdx + coord_dCm]);
                    }
                }
            }
            else
            {
                for(int n = 0; n < BLK_N / DIM_N && coord_dCn < N; ++n, coord_dCn += DIM_N)
                {
                    int64_t nDIdx     = coord_dCn * size_t(ldd);
                    int64_t coord_dCm = int64_t(blx) * BLK_M + thx;

#pragma unroll
                    for(int m = 0; m < BLK_M / DIM_M; ++m, coord_dCm += DIM_M)
                    {
                        if(coord_dCm < M)
                            dD[nDIdx + coord_dCm] = To(alpha * rD[n][m]);
                    }
                }
            }

#if DEVICE_GRID_YZ_16BIT
        }
#endif
    }

    // general alpha, beta, restricted m, n, k to multiples of blocks
    template <typename Tc,
              int  DIM_M,
              int  DIM_N,
              int  BLK_M,
              int  BLK_N,
              int  BLK_K,
              int  DIM_M_A,
              int  DIM_N_A,
              int  DIM_M_B,
              int  DIM_N_B,
              char TRANS_A,
              char TRANS_B,
              typename TiConstPtr,
              typename ToConstPtr,
              typename ToPtr>
    ROCBLAS_KERNEL(DIM_M* DIM_N)
    rocblas_gemm_batched_kernel(int64_t        M,
                                int64_t        N,
                                int64_t        K,
                                const Tc       alpha,
                                TiConstPtr*    dA_input,
                                int64_t        lda,
                                rocblas_stride a_st_or_of,
                                TiConstPtr*    dB_input,
                                int64_t        ldb,
                                rocblas_stride b_st_or_of,
                                const Tc       beta,
                                ToConstPtr*    dC_input,
                                int64_t        ldc,
                                rocblas_stride c_st_or_of,
                                ToPtr*         dD_input,
                                int64_t        ldd,
                                rocblas_stride d_st_or_of,
                                rocblas_int    batch_count)
    {
        int     thx = threadIdx.x; // thread's m position in C
        int     thy = threadIdx.y; // thread's n position in C
        int64_t idt = int64_t(DIM_M) * thy + thx; // thread's number
        int     blx = blockIdx.x; // block's m position
        int     bly = blockIdx.y; // block's n position
        int     blz = blockIdx.z; // block's matrix in the batch

#if DEVICE_GRID_YZ_16BIT
        for(; blz < batch_count; blz += c_YZ_grid_launch_limit)
        {
#endif

            auto* dA = load_ptr_batch(dA_input, blz, a_st_or_of);
            auto* dB = load_ptr_batch(dB_input, blz, b_st_or_of);
            auto* dC = load_ptr_batch(dC_input, blz, c_st_or_of);
            auto* dD = load_ptr_batch(dD_input, blz, d_st_or_of);

            auto tmp = *dD;
            using To = decltype(tmp);

            __shared__ Tc sA[BLK_K][BLK_M]; // shared memory for A
            __shared__ Tc sB[BLK_N][BLK_K]; // shared memory for B
            Tc            rD[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for D

            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                    rD[n][m] = 0.0;

            {
                int     thxA = idt % DIM_M_A; // thread's m position for loading A
                int64_t thyA = idt / DIM_M_A; // thread's n position for loading A
                int     thxB = idt % DIM_M_B; // thread's m position for loading B
                int64_t thyB = idt / DIM_M_B; // thread's n position for loading B

                size_t coord_A, coord_B;
                if(TRANS_A == 'N')
                    coord_A = (thxA + int64_t(blx) * BLK_M) + (thyA)*size_t(lda);
                else if(TRANS_A == 'T' || TRANS_A == 'C')
                    coord_A = (thxA + int64_t(blx) * BLK_M) * size_t(lda) + (thyA);

                if(TRANS_B == 'N')
                    coord_B = thxB + (int64_t(bly) * BLK_N + thyB) * size_t(ldb);
                else if(TRANS_B == 'T' || TRANS_B == 'C')
                    coord_B = thxB * size_t(ldb) + (int64_t(bly) * BLK_N + thyB);

                int64_t kk = 0;
                for(; kk < K; kk += BLK_K)
                {
                    for(int n = 0; n < BLK_K; n += DIM_N_A)
                        for(int m = 0; m < BLK_M; m += DIM_M_A)
                            if(TRANS_A == 'N')
                            {
                                sA[n + thyA][m + thxA] = dA[coord_A + m + n * size_t(lda)];
                            }
                            else if(TRANS_A == 'T')
                            {
                                sA[n + thyA][m + thxA] = dA[coord_A + m * size_t(lda) + n];
                            }
                            else if(TRANS_A == 'C')
                            {
                                sA[n + thyA][m + thxA] = conj(dA[coord_A + m * size_t(lda) + n]);
                            }

                    for(int n = 0; n < BLK_N; n += DIM_N_B)
                        for(int m = 0; m < BLK_K; m += DIM_M_B)
                            if(TRANS_B == 'N')
                            {
                                sB[n + thyB][m + thxB] = dB[coord_B + m + n * size_t(ldb)];
                            }
                            else if(TRANS_B == 'T')
                            {
                                sB[n + thyB][m + thxB] = dB[coord_B + m * size_t(ldb) + n];
                            }
                            else if(TRANS_B == 'C')
                            {
                                sB[n + thyB][m + thxB] = conj(dB[coord_B + m * size_t(ldb) + n]);
                            }

                    __syncthreads();

                    for(int k = 0; k < BLK_K; ++k)
                        for(int n = 0; n < BLK_N / DIM_N; ++n)
                            for(int m = 0; m < BLK_M / DIM_M; ++m)
                                rD[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

                    __syncthreads();

                    if(TRANS_A == 'N')
                        coord_A += BLK_K * size_t(lda);
                    else if(TRANS_A == 'T' || TRANS_A == 'C')
                        coord_A += BLK_K;

                    if(TRANS_B == 'N')
                        coord_B += BLK_K;
                    else if(TRANS_B == 'T' || TRANS_B == 'C')
                        coord_B += BLK_K * size_t(ldb);
                }
            }

            int64_t coord_dCn = int64_t(bly) * BLK_N + thy;
            if(beta != Tc(0))
            {
                for(int n = 0; n < BLK_N / DIM_N; ++n, coord_dCn += DIM_N)
                {
                    int64_t nCIdx     = coord_dCn * size_t(ldc);
                    int64_t nDIdx     = coord_dCn * size_t(ldd);
                    int64_t coord_dCm = int64_t(blx) * BLK_M + thx;

#pragma unroll
                    for(int m = 0; m < BLK_M / DIM_M; ++m, coord_dCm += DIM_M)
                    {
                        dD[nDIdx + coord_dCm] = To(alpha * rD[n][m] + beta * dC[nCIdx + coord_dCm]);
                    }
                }
            }
            else
            {
                for(int n = 0; n < BLK_N / DIM_N; ++n, coord_dCn += DIM_N)
                {
                    int64_t nDIdx     = coord_dCn * size_t(ldd);
                    int64_t coord_dCm = int64_t(blx) * BLK_M + thx;

#pragma unroll
                    for(int m = 0; m < BLK_M / DIM_M; ++m, coord_dCm += DIM_M)
                    {
                        dD[nDIdx + coord_dCm] = To(alpha * rD[n][m]);
                    }
                }
            }

#if DEVICE_GRID_YZ_16BIT
        }
#endif
    }

    template <bool BATCHED, typename T, typename TiConstPtr, typename ToConstPtr, typename ToPtr>
    rocblas_status rocblas_gemm_source_solution_64(rocblas_handle __restrict__ handle,
                                                   rocblas_operation trans_a,
                                                   rocblas_operation trans_b,
                                                   int64_t           m,
                                                   int64_t           n,
                                                   int64_t           k,
                                                   const T           alpha,
                                                   TiConstPtr*       dA,
                                                   int64_t           lda,
                                                   rocblas_stride    stride_a,
                                                   rocblas_stride    offset_a,
                                                   TiConstPtr*       dB,
                                                   int64_t           ldb,
                                                   rocblas_stride    stride_b,
                                                   rocblas_stride    offset_b,
                                                   const T           beta,
                                                   ToConstPtr*       dC,
                                                   int64_t           ldc,
                                                   rocblas_stride    stride_c,
                                                   rocblas_stride    offset_c,
                                                   ToPtr*            dD,
                                                   int64_t           ldd,
                                                   rocblas_stride    stride_d,
                                                   rocblas_stride    offset_d,
                                                   rocblas_int       batch_count)
    {
        // gemm has same behavior for alpha == 0 and k == 0. Special code is needed
        // for alpha == 0, no special code is needed for k == 0. It is more efficient
        // setting k = 0 than adding extra code to a kernel to handle alpha == 0
        if(alpha == T(0))
            k = 0;

        if(k == 0)
        {
            return rocblas_gemm_ex_scale_launcher_64(handle,
                                                     m,
                                                     n,
                                                     beta,
                                                     dC,
                                                     offset_c,
                                                     ldc,
                                                     stride_c,
                                                     dD,
                                                     offset_d,
                                                     ldd,
                                                     stride_d,
                                                     batch_count);
        }

        TiConstPtr*    dA_krn;
        TiConstPtr*    dB_krn;
        ToConstPtr*    dC_krn;
        ToPtr*         dD_krn;
        rocblas_stride a_st_or_of;
        rocblas_stride b_st_or_of;
        rocblas_stride c_st_or_of;
        rocblas_stride d_st_or_of;

        if(BATCHED)
        {
            dA_krn     = dA;
            dB_krn     = dB;
            dC_krn     = dC;
            dD_krn     = dD;
            a_st_or_of = offset_a;
            b_st_or_of = offset_b;
            c_st_or_of = offset_c;
            d_st_or_of = offset_d;
        }
        else
        {
            dA_krn     = dA + offset_a;
            dB_krn     = dB + offset_b;
            dC_krn     = dC + offset_c;
            dD_krn     = dD + offset_d;
            a_st_or_of = stride_a;
            b_st_or_of = stride_b;
            c_st_or_of = stride_c;
            d_st_or_of = stride_d;
        }

#define GEMM_SOURCE_PARAM                                                                    \
    dimGrid, dimBlock, 0, stream, m, n, k, dA_krn, lda, a_st_or_of, dB_krn, ldb, b_st_or_of, \
        dC_krn, ldc, c_st_or_of, dD_krn, ldd, d_st_or_of, batch_count

#define GEMM_SOURCE_PARAM_SCALARS                                                       \
    dimGrid, dimBlock, 0, stream, m, n, k, alpha, dA_krn, lda, a_st_or_of, dB_krn, ldb, \
        b_st_or_of, beta, dC_krn, ldc, c_st_or_of, dD_krn, ldd, d_st_or_of, batch_count

        hipStream_t stream  = handle->get_stream();
        int         batches = handle->getBatchGridDim((int)batch_count);

        if((m % 64 == 0) && (n % 64 == 0) && (k % 4 == 0))
        {
            //m is mult of 64, n is mult of 64, k is mult of 4
            const int dim_m = 16;
            const int dim_n = 16;
            const int blk_m = 64;
            const int blk_n = 64;
            const int blk_k = 4;
            dim3      dimBlock(dim_m, dim_n, 1);
            dim3      dimGrid(m / blk_m, n / blk_n, batches);

            // general alpha, beta
            // clang-format off
            if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            // clang-format on
        }
        else if((m % 32 == 0) && (n % 32 == 0) && (k % 8 == 0))
        {
            // m is mult of 32, n is mult of 32, k is mult of 8
            const int dim_m = 16;
            const int dim_n = 16;
            const int blk_m = 32;
            const int blk_n = 32;
            const int blk_k = 8;
            dim3      dimBlock(dim_m, dim_n, 1);
            dim3      dimGrid(m / blk_m, n / blk_n, batches);

            // general alpha, beta
            // clang-format off
            if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            // clang-format on
        }
        else
        {
            const int dim_m = 16;
            const int dim_n = 16;
            const int blk_m = 32;
            const int blk_n = 32;
            const int blk_k = 8;
            dim3      dimBlock(dim_m, dim_n, 1);
            dim3      dimGrid(((m - 1) / blk_m) + 1, ((n - 1) / blk_n) + 1, batches);

            // general m, n, k, alpha, beta
            // clang-format off
            if(rocblas_operation_none == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_none == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_none == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'N'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_conjugate_transpose == trans_a && rocblas_operation_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'C', 'T'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_none == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'N', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            if(rocblas_operation_transpose == trans_a && rocblas_operation_conjugate_transpose == trans_b)
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemm_batched_general_kernel
                <T, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n, 'T', 'C'>), GEMM_SOURCE_PARAM_SCALARS);
            // clang-format on
        }

#undef GEMM_SOURCE_PARAM
#undef GEMM_SOURCE_PARAM_SCALARS
        return rocblas_status_success;
    }
}
