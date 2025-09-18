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

#include "../blas2/rocblas_trsv.hpp"
#include "definitions.hpp"
#include "device_macros.hpp"
#ifdef BUILD_WITH_TENSILE
#include "../blas_ex/rocblas_gemm_ex.hpp"
#endif
#include "int64_helpers.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_gemm.hpp"
#include "rocblas_trsm.hpp"
#include "src64/blas3/rocblas_gemm_64.hpp"
#include "trtri_trsm.hpp"

/** Constants for block size of trsm **/
// clang-format off
#define TRSM_NUMROWS_REAL 12
#define TRSM_NUMCOLS_REAL 16
#define TRSM_INTERVALSROW_REAL                                          \
    40, 56, 80, 112, 144, 176, 208, 240, 288, 352, 480
#define TRSM_INTERVALSCOL_REAL                                          \
    448, 768, 960, 1152, 1408, 1920, 2304, 2816, 3840, 4096, 4736,      \
    4992, 5888, 7680, 9728
#define TRSM_BLKSIZES_REAL                                              \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},    \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 24, 24, 24, 16},    \
    {1,  1,  1,  1,  1,  1,  1,  1,  1, 32, 32, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1,  1,  1, 48, 48, 48, 48, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1,  1, 64, 64, 64, 48, 48, 32, 32, 32, 24, 16},    \
    {1,  1,  1,  1,  1, 80, 80, 80, 56, 56, 40, 40, 40, 32, 32, 32},    \
    {1,  1,  1,  1, 80, 80, 80, 80, 80, 48, 48, 48, 40, 32,  0,  0},    \
    {1,  1,  1, 80, 80, 80, 80, 80, 56, 56, 32, 32, 32, 32,  0,  0},    \
    {1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},    \
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}

#define TRSM_NUMROWS_COMPLEX 10
#define TRSM_NUMCOLS_COMPLEX 12
#define TRSM_INTERVALSROW_COMPLEX                                       \
    40, 56, 80, 112, 144, 208, 240, 288, 480
#define TRSM_INTERVALSCOL_COMPLEX                                       \
    704, 960, 1344, 1920, 2304, 2816, 3200, 3840, 4864, 5888, 7680
#define TRSM_BLKSIZES_COMPLEX                                           \
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},                               \
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 24, 24, 24},                            \
    {1, 1, 1, 1, 1, 1, 1, 1, 32, 32, 32, 32},                           \
    {1, 1, 1, 1, 1, 72, 72, 56, 48, 32, 32, 32},                        \
    {1, 1, 1, 1, 64, 64, 64, 64, 48, 32, 32, 32},                       \
    {1, 1, 1, 80, 80, 80, 64, 64, 48, 32, 32, 32},                      \
    {1, 1, 80, 80, 80, 80, 64, 64, 40, 40, 32, 32},                     \
    {1, 1, 72, 72, 64, 64, 64, 64, 32, 32, 32, 0},                      \
    {1, 80, 80, 80, 80, 80, 64, 64, 48, 40, 32, 0},                     \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

#define TRSM_BATCH_NUMROWS_REAL 11
#define TRSM_BATCH_NUMCOLS_REAL 17
#define TRSM_BATCH_INTERVALSROW_REAL                                        \
    20, 28, 40, 80, 112, 176, 208, 288, 352, 480
#define TRSM_BATCH_INTERVALSCOL_REAL                                        \
    6, 10, 12, 22, 28, 30, 36, 42, 46, 50, 60, 96, 432, 928, 960, 1472
#define TRSM_BATCH_BLKSIZES_REAL                                            \
    { 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1,  1,  1,  1, 16, 16, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1,  1,  1,  1, 16, 16, 16, 16, 16,  0,  0,  0,  0,  0,  0,  0,  0},   \
    { 1, 24, 24, 24, 24, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},   \
    {48, 48, 32, 32, 24, 24, 16, 16, 16, 32, 32, 32, 16, 16, 16, 16, 16},   \
    {64, 64, 32, 32, 24, 24, 16, 16, 16, 32, 32, 32, 24, 24, 24, 24, 24},   \
    {64, 64, 32, 32, 24, 24, 24, 24, 32, 32, 32, 32, 32, 24, 24, 24, 24},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 40, 40, 40, 32, 32, 24, 24, 32, 32},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 48, 48, 40, 32, 32, 32, 32, 32, 32},   \
    {64, 64, 64, 32, 32, 32, 32, 40, 48, 48, 40, 32, 32, 32, 32, 32,  0},   \
    {64, 64, 64, 32, 32, 32, 48, 48, 48, 48, 40, 32, 32, 32,  0,  0,  0}

#define TRSM_BATCH_NUMROWS_COMPLEX 10
#define TRSM_BATCH_NUMCOLS_COMPLEX 16
#define TRSM_BATCH_INTERVALSROW_COMPLEX                                     \
    20, 28, 40, 56, 80, 112, 144, 176, 480
#define TRSM_BATCH_INTERVALSCOL_COMPLEX                                     \
    4, 12, 16, 28, 32, 40, 48, 50, 60, 72, 88, 176, 232, 400, 464
#define TRSM_BATCH_BLKSIZES_COMPLEX                                         \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  8,  1,  1,  1,  1,  1,  1,  1},        \
    {1,  1,  1,  1, 16, 16, 16, 16,  1,  1,  1, 16, 16, 16, 16, 16},        \
    {1,  1,  1, 24, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},        \
    {1,  1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 48, 48, 32},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 64, 64, 64, 64, 64, 32},        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 80, 80, 56, 56, 32, 32},        \
    {1, 64, 32, 32, 32, 64, 48, 32, 32, 32, 32, 32, 32, 32, 32, 32},        \
    {1,  1,  1,  1,  1,  1, 64, 64, 64, 64, 64, 64, 64, 48, 48, 48}
// clang-format on

static constexpr rocblas_int trsm_intervals_row_real_batch[] = {TRSM_BATCH_INTERVALSROW_REAL};
static constexpr rocblas_int trsm_intervals_col_real_batch[] = {TRSM_BATCH_INTERVALSCOL_REAL};
static constexpr rocblas_int trsm_blksizes_real_batch[][TRSM_BATCH_NUMCOLS_REAL]
    = {TRSM_BATCH_BLKSIZES_REAL};
static constexpr rocblas_int trsm_intervals_row_real_nonbatch[] = {TRSM_INTERVALSROW_REAL};
static constexpr rocblas_int trsm_intervals_col_real_nonbatch[] = {TRSM_INTERVALSCOL_REAL};
static constexpr rocblas_int trsm_blksizes_real_nonbatch[][TRSM_NUMCOLS_REAL]
    = {TRSM_BLKSIZES_REAL};

static constexpr rocblas_int trsm_intervals_row_complex_batch[] = {TRSM_BATCH_INTERVALSROW_COMPLEX};
static constexpr rocblas_int trsm_intervals_col_complex_batch[] = {TRSM_BATCH_INTERVALSCOL_COMPLEX};
static constexpr rocblas_int trsm_blksizes_complex_batch[][TRSM_BATCH_NUMCOLS_COMPLEX]
    = {TRSM_BATCH_BLKSIZES_COMPLEX};
static constexpr rocblas_int trsm_intervals_row_complex_nonbatch[] = {TRSM_INTERVALSROW_COMPLEX};
static constexpr rocblas_int trsm_intervals_col_complex_nonbatch[] = {TRSM_INTERVALSCOL_COMPLEX};
static constexpr rocblas_int trsm_blksizes_complex_nonbatch[][TRSM_NUMCOLS_COMPLEX]
    = {TRSM_BLKSIZES_COMPLEX};

template <typename T>
static const T alpha_negative_one = T(-1);
template <typename T>
static const T beta_0 = T(0);
template <typename T>
static const T alpha_1 = T(1);
template <typename T>
static const T beta_1 = T(1);

template <int DIM_X, int DIM_Y, typename T, typename U, typename V>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_copy_matrix_trsm(rocblas_int    rows,
                         rocblas_int    cols,
                         rocblas_int    elem_size,
                         U              a,
                         rocblas_int    lda,
                         rocblas_stride stride_a,
                         V              b,
                         rocblas_int    ldb,
                         rocblas_stride stride_b,
                         rocblas_stride offset_a,
                         rocblas_stride offset_b,
                         rocblas_int    batch_count)
{
    size_t tx = blockIdx.x * DIM_X + threadIdx.x;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const T* xa = load_ptr_batch(a, batch, offset_a, stride_a);
        T*       xb = load_ptr_batch(b, batch, offset_b, stride_b);

        //looping over ty
        for(size_t ty = blockIdx.y * DIM_Y + threadIdx.y; ty < cols && tx < rows;
            ty += DIM_Y * gridDim.y)
            xb[tx + size_t(ldb) * ty] = xa[tx + size_t(lda) * ty];

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/* ===============copy helper============================================= */
template <typename T, typename U, typename V>
rocblas_status rocblas_copy_block_unit(rocblas_handle handle,
                                       rocblas_int    m,
                                       rocblas_int    n,
                                       U              src,
                                       rocblas_int    src_ld,
                                       rocblas_stride src_stride,
                                       V              dst,
                                       rocblas_int    dst_ld,
                                       rocblas_stride dst_stride,
                                       rocblas_int    batch_count,
                                       rocblas_stride offset_src = 0,
                                       rocblas_stride offset_dst = 0)
{
    static constexpr int COPY_DIM_X = 128;
    static constexpr int COPY_DIM_Y = 8;

    int batches = handle->getBatchGridDim((int)batch_count);

    rocblas_int blocks_X = (m - 1) / COPY_DIM_X + 1; // parameters for device kernel

    //blocksY should be less than 2^16 (65536) to avoid overflow as grid y and z dimensions support only 16-bit values on some gfx
    rocblas_int blocks_Y = std::min(c_YZ_grid_launch_limit, (n - 1) / COPY_DIM_Y + 1);
    dim3        grid(blocks_X, blocks_Y, batches);
    dim3        threads(COPY_DIM_X, COPY_DIM_Y);

    ROCBLAS_LAUNCH_KERNEL((rocblas_copy_matrix_trsm<COPY_DIM_X, COPY_DIM_Y, T>),
                          grid,
                          threads,
                          0,
                          handle->get_stream(),
                          m,
                          n,
                          sizeof(T),
                          src,
                          src_ld,
                          src_stride,
                          dst,
                          dst_ld,
                          dst_stride,
                          offset_src,
                          offset_dst,
                          batch_count);

    return rocblas_status_success;
}

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T, typename U>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_set_matrix_trsm(int64_t        rows,
                        int64_t        cols,
                        rocblas_int    elem_size,
                        U              a,
                        int64_t        lda,
                        rocblas_stride stride_a,
                        T              val,
                        rocblas_stride offset_a,
                        int            batch_count)
{
    size_t tx = blockIdx.x * DIM_X + threadIdx.x;
    size_t ty = blockIdx.y * DIM_Y + threadIdx.y;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        T* xa = load_ptr_batch(a, batch, offset_a, stride_a);

        if(tx < rows && ty < cols)
            xa[tx + lda * ty] = T(0.0);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/* ===============set helper============================================= */
template <typename T, typename U>
rocblas_status set_block_unit(rocblas_handle handle,
                              int64_t        m,
                              int64_t        n,
                              U              src,
                              int64_t        src_ld,
                              rocblas_stride src_stride,
                              rocblas_int    batch_count,
                              T              val,
                              rocblas_stride offset_src)
{
    static constexpr int DIM_X = 128;
    static constexpr int DIM_Y = 8;

    int         batches = handle->getBatchGridDim((int)batch_count);
    rocblas_int blocksX = (m - 1) / DIM_X + 1; // parameters for device kernel
    rocblas_int blocksY = (n - 1) / DIM_Y + 1;
    dim3        grid(blocksX, blocksY, batches);
    dim3        threads(DIM_X, DIM_Y);

    ROCBLAS_LAUNCH_KERNEL((rocblas_set_matrix_trsm<DIM_X, DIM_Y, T>),
                          grid,
                          threads,
                          0,
                          handle->get_stream(),
                          m,
                          n,
                          sizeof(T),
                          src,
                          src_ld,
                          src_stride,
                          val,
                          offset_src,
                          batch_count);

    return rocblas_status_success;
}

/* ===============left==================================================== */

template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
rocblas_status rocblas_trsm_left(rocblas_handle    handle,
                                 rocblas_fill      uplo,
                                 rocblas_operation transA,
                                 rocblas_int       m,
                                 rocblas_int       n,
                                 const T*          alpha,
                                 U                 A,
                                 rocblas_stride    offset_Ain,
                                 rocblas_int       lda,
                                 rocblas_stride    stride_A,
                                 V                 B,
                                 rocblas_stride    offset_Bin,
                                 rocblas_int       ldb,
                                 rocblas_stride    stride_B,
                                 rocblas_int       batch_count,
                                 U                 invA,
                                 rocblas_stride    offset_invAin,
                                 rocblas_stride    stride_invA,
                                 V                 X,
                                 rocblas_stride    stride_X)
{
    rocblas_int i, jb;

    // transB is always non-transpose
    static constexpr rocblas_operation transB = rocblas_operation_none;

    if(transA == transB)
    {
        if(uplo == rocblas_fill_lower)
        {
            // left, lower no-transpose
            jb = std::min(BLOCK, m);
            rocblas_internal_gemm<BATCHED>(handle,
                                           transA,
                                           transB,
                                           jb,
                                           n,
                                           jb,
                                           alpha,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           offset_Bin,
                                           ldb,
                                           stride_B,
                                           &beta_0<T>,
                                           X,
                                           rocblas_int(0),
                                           m,
                                           stride_X,
                                           batch_count);

            if(BLOCK < m)
            {
                rocblas_internal_gemm<BATCHED>(handle,
                                               transA,
                                               transB,
                                               m - BLOCK,
                                               n,
                                               BLOCK,
                                               &alpha_negative_one<T>,
                                               A,
                                               BLOCK + offset_Ain,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               rocblas_int(0),
                                               m,
                                               stride_X,
                                               alpha,
                                               B,
                                               BLOCK + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);
                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = std::min(m - i, BLOCK);

                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   jb,
                                                   n,
                                                   jb,
                                                   &alpha_1<T>,
                                                   invA,
                                                   i * size_t(BLOCK) + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   i + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   &beta_0<T>,
                                                   X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK >= m) // this condition is not necessary at all and can be changed
                        // as if (i+BLOCK<m)
                        break;

                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   m - i - BLOCK,
                                                   n,
                                                   BLOCK,
                                                   &alpha_negative_one<T>,
                                                   A,
                                                   i + BLOCK + i * size_t(lda) + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   (U)X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   &beta_1<T>,
                                                   B,
                                                   i + BLOCK + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }

#if 0
            for( i=0; i < m; i += BLOCK ) {
                jb = std::min(m-i, BLOCK);
                T *tmp = (i == 0) ? alpha : one;
                rocblas_internal_gemm<false>(handle, transA, transB, jb, n, jb, tmp, invA(i), BLOCK, stride_invA, B(i,0), ldb, stride_B, &beta_0<T>, X(i,0), ldb, stride_X, batch_count); // strides?
                if(i + BLOCK < m){
                    rocblas_internal_gemm<false>(handle, transA, transB, m-i-BLOCK, n, BLOCK, &alpha_negative_one<T>, A(i+BLOCK,i), lda, stride_A, X(i,0), ldb, stride_X, tmp, B(i+BLOCK,0), ldb, stride_B, batch_count); // strides?
                }
            }

#endif
        }
        else
        {
            // left, upper no-transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i  = m - jb;

            // if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35}
            rocblas_internal_gemm<BATCHED>(handle,
                                           transA,
                                           transB,
                                           jb,
                                           n,
                                           jb,
                                           alpha,
                                           invA,
                                           i * size_t(BLOCK) + offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           i + offset_Bin,
                                           ldb,
                                           stride_B,
                                           &beta_0<T>,
                                           X,
                                           i,
                                           m,
                                           stride_X,
                                           batch_count);

            if(i - BLOCK >= 0)
            {
                rocblas_internal_gemm<BATCHED>(handle,
                                               transA,
                                               transB,
                                               i,
                                               n,
                                               jb,
                                               &alpha_negative_one<T>,
                                               A,
                                               i * size_t(lda) + offset_Ain,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               i,
                                               m,
                                               stride_X,
                                               alpha,
                                               B,
                                               offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    //{32, 35, 32, 32, 35, 35}
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   BLOCK,
                                                   n,
                                                   BLOCK,
                                                   &alpha_1<T>,
                                                   invA,
                                                   i * size_t(BLOCK) + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   i + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   &beta_0<T>,
                                                   X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   i,
                                                   n,
                                                   BLOCK,
                                                   &alpha_negative_one<T>,
                                                   A,
                                                   i * size_t(lda) + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   (U)X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   &beta_1<T>,
                                                   B,
                                                   offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
    }
    else
    { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if(uplo == rocblas_fill_lower)
        {
            // left, lower transpose
            jb = (m % BLOCK == 0) ? BLOCK : (m % BLOCK);
            i  = m - jb;
            rocblas_internal_gemm<BATCHED>(handle,
                                           transA,
                                           transB,
                                           jb,
                                           n,
                                           jb,
                                           alpha,
                                           invA,
                                           i * size_t(BLOCK) + offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           i + offset_Bin,
                                           ldb,
                                           stride_B,
                                           &beta_0<T>,
                                           X,
                                           i,
                                           m,
                                           stride_X,
                                           batch_count);
            if(i - BLOCK >= 0)
            {
                rocblas_internal_gemm<BATCHED>(handle,
                                               transA,
                                               transB,
                                               i,
                                               n,
                                               jb,
                                               &alpha_negative_one<T>,
                                               A,
                                               i + offset_Ain,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               i,
                                               m,
                                               stride_X,
                                               alpha,
                                               B,
                                               offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   BLOCK,
                                                   n,
                                                   BLOCK,
                                                   &alpha_1<T>,
                                                   invA,
                                                   i * size_t(BLOCK) + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   i + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   &beta_0<T>,
                                                   X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   i,
                                                   n,
                                                   BLOCK,
                                                   &alpha_negative_one<T>,
                                                   A,
                                                   i + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   (U)X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   &beta_1<T>,
                                                   B,
                                                   offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
        else
        {
            // left, upper transpose
            jb = std::min(BLOCK, m);
            rocblas_internal_gemm<BATCHED>(handle,
                                           transA,
                                           transB,
                                           jb,
                                           n,
                                           jb,
                                           alpha,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           offset_Bin,
                                           ldb,
                                           stride_B,
                                           &beta_0<T>,
                                           X,
                                           rocblas_int(0),
                                           m,
                                           stride_X,
                                           batch_count);
            if(BLOCK < m)
            {
                rocblas_internal_gemm<BATCHED>(handle,
                                               transA,
                                               transB,
                                               m - BLOCK,
                                               n,
                                               BLOCK,
                                               &alpha_negative_one<T>,
                                               A,
                                               BLOCK * size_t(lda) + offset_Ain,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               rocblas_int(0),
                                               m,
                                               stride_X,
                                               alpha,
                                               B,
                                               BLOCK + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = std::min(m - i, BLOCK);
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   jb,
                                                   n,
                                                   jb,
                                                   &alpha_1<T>,
                                                   invA,
                                                   i * size_t(BLOCK) + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   i + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   &beta_0<T>,
                                                   X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK >= m)
                        break;
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transA,
                                                   transB,
                                                   m - i - BLOCK,
                                                   n,
                                                   BLOCK,
                                                   &alpha_negative_one<T>,
                                                   A,
                                                   i + (i + BLOCK) * size_t(lda) + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   (U)X,
                                                   i,
                                                   m,
                                                   stride_X,
                                                   &beta_1<T>,
                                                   B,
                                                   i + BLOCK + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
    } // transpose

    return rocblas_status_success;
}

/* ===============right==================================================== */

template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
rocblas_status rocblas_trsm_right(rocblas_handle    handle,
                                  rocblas_fill      uplo,
                                  rocblas_operation transA,
                                  rocblas_int       m,
                                  rocblas_int       n,
                                  const T*          alpha,
                                  U                 A,
                                  rocblas_stride    offset_Ain,
                                  rocblas_int       lda,
                                  rocblas_stride    stride_A,
                                  V                 B,
                                  rocblas_stride    offset_Bin,
                                  rocblas_int       ldb,
                                  rocblas_stride    stride_B,
                                  rocblas_int       batch_count,
                                  U                 invA,
                                  rocblas_stride    offset_invAin,
                                  rocblas_stride    stride_invA,
                                  V                 X,
                                  rocblas_stride    stride_X)
{
    rocblas_int i, jb;

    // transB is always non-transpose
    static constexpr rocblas_operation transB = rocblas_operation_none;

    if(transA == transB)
    {
        if(uplo == rocblas_fill_lower)
        {
            // right, lower no-transpose
            jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
            i  = n - jb;
            rocblas_internal_gemm<BATCHED>(handle,
                                           transB,
                                           transA,
                                           m,
                                           jb,
                                           jb,
                                           alpha,
                                           U(B),
                                           i * size_t(ldb) + offset_Bin,
                                           ldb,
                                           stride_B,
                                           invA,
                                           i * size_t(BLOCK) + offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           &beta_0<T>,
                                           X,
                                           i * size_t(m),
                                           m,
                                           stride_X,
                                           batch_count);
            if(i - BLOCK >= 0)
            {
                rocblas_internal_gemm<BATCHED>(handle,
                                               transB,
                                               transA,
                                               m,
                                               i,
                                               jb,
                                               &alpha_negative_one<T>,
                                               (U)X,
                                               i * size_t(m),
                                               m,
                                               stride_X,
                                               A,
                                               i + offset_Ain,
                                               lda,
                                               stride_A,
                                               alpha,
                                               B,
                                               offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   BLOCK,
                                                   BLOCK,
                                                   &alpha_1<T>,
                                                   (U)B,
                                                   i * size_t(ldb) + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   invA,
                                                   i * size_t(BLOCK) + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   &beta_0<T>,
                                                   X,
                                                   i * size_t(m),
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   i,
                                                   BLOCK,
                                                   &alpha_negative_one<T>,
                                                   (U)X,
                                                   i * size_t(m),
                                                   m,
                                                   stride_X,
                                                   A,
                                                   i + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   &beta_1<T>,
                                                   B,
                                                   offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
        else
        {
            // right, upper no-transpose
            jb = std::min(BLOCK, n);
            rocblas_internal_gemm<BATCHED>(handle,
                                           transB,
                                           transA,
                                           m,
                                           jb,
                                           jb,
                                           alpha,
                                           (U)B,
                                           offset_Bin,
                                           ldb,
                                           stride_B,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           &beta_0<T>,
                                           X,
                                           rocblas_int(0),
                                           m,
                                           stride_X,
                                           batch_count);
            if(BLOCK < n)
            {
                rocblas_internal_gemm<BATCHED>(handle,
                                               transB,
                                               transA,
                                               m,
                                               n - BLOCK,
                                               BLOCK,
                                               &alpha_negative_one<T>,
                                               (U)X,
                                               rocblas_int(0),
                                               m,
                                               stride_X,
                                               A,
                                               BLOCK * size_t(lda) + offset_Ain,
                                               lda,
                                               stride_A,
                                               alpha,
                                               B,
                                               BLOCK * size_t(ldb) + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = BLOCK; i < n; i += BLOCK)
                {
                    jb = std::min(BLOCK, n - i);
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   jb,
                                                   jb,
                                                   &alpha_1<T>,
                                                   (U)B,
                                                   i * size_t(ldb) + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   invA,
                                                   i * size_t(BLOCK) + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   &beta_0<T>,
                                                   X,
                                                   i * size_t(m),
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK >= n)
                        break;
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   n - i - BLOCK,
                                                   BLOCK,
                                                   &alpha_negative_one<T>,
                                                   (U)X,
                                                   i * size_t(m),
                                                   m,
                                                   stride_X,
                                                   A,
                                                   i + (i + BLOCK) * size_t(lda) + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   &beta_1<T>,
                                                   B,
                                                   (i + BLOCK) * size_t(ldb) + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
    }
    else
    { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if(uplo == rocblas_fill_lower)
        {
            // right, lower transpose
            jb = std::min(BLOCK, n);
            rocblas_internal_gemm<BATCHED>(handle,
                                           transB,
                                           transA,
                                           m,
                                           jb,
                                           jb,
                                           alpha,
                                           U(B),
                                           offset_Bin,
                                           ldb,
                                           stride_B,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           &beta_0<T>,
                                           X,
                                           rocblas_int(0),
                                           m,
                                           stride_X,
                                           batch_count);
            if(BLOCK < n)
            {
                rocblas_internal_gemm<BATCHED>(handle,
                                               transB,
                                               transA,
                                               m,
                                               n - BLOCK,
                                               BLOCK,
                                               &alpha_negative_one<T>,
                                               U(X),
                                               rocblas_int(0),
                                               m,
                                               stride_X,
                                               A,
                                               BLOCK + offset_Ain,
                                               lda,
                                               stride_A,
                                               alpha,
                                               B,
                                               BLOCK * size_t(ldb) + offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = BLOCK; i < n; i += BLOCK)
                {
                    jb = std::min(BLOCK, n - i);
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   jb,
                                                   jb,
                                                   &alpha_1<T>,
                                                   (U)B,
                                                   i * size_t(ldb) + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   invA,
                                                   i * size_t(BLOCK) + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   &beta_0<T>,
                                                   X,
                                                   i * size_t(m),
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK >= n)
                        break;
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   n - i - BLOCK,
                                                   BLOCK,
                                                   &alpha_negative_one<T>,
                                                   (U)X,
                                                   i * size_t(m),
                                                   m,
                                                   stride_X,
                                                   A,
                                                   BLOCK + i + i * size_t(lda) + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   &beta_1<T>,
                                                   B,
                                                   (i + BLOCK) * size_t(ldb) + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
        else
        {
            // right, upper transpose
            jb = (n % BLOCK == 0) ? BLOCK : (n % BLOCK);
            i  = n - jb;
            rocblas_internal_gemm<BATCHED>(handle,
                                           transB,
                                           transA,
                                           m,
                                           jb,
                                           jb,
                                           alpha,
                                           (U)B,
                                           i * size_t(ldb) + offset_Bin,
                                           ldb,
                                           stride_B,
                                           invA,
                                           i * size_t(BLOCK) + offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           &beta_0<T>,
                                           X,
                                           i * size_t(m),
                                           m,
                                           stride_X,
                                           batch_count);
            if(i - BLOCK >= 0)
            {
                rocblas_internal_gemm<BATCHED>(handle,
                                               transB,
                                               transA,
                                               m,
                                               i,
                                               jb,
                                               &alpha_negative_one<T>,
                                               (U)X,
                                               i * size_t(m),
                                               m,
                                               stride_X,
                                               A,
                                               i * size_t(lda) + offset_Ain,
                                               lda,
                                               stride_A,
                                               alpha,
                                               B,
                                               offset_Bin,
                                               ldb,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = n - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   BLOCK,
                                                   BLOCK,
                                                   &alpha_1<T>,
                                                   (U)B,
                                                   i * size_t(ldb) + offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   invA,
                                                   i * size_t(BLOCK) + offset_invAin,
                                                   BLOCK,
                                                   stride_invA,
                                                   &beta_0<T>,
                                                   X,
                                                   i * size_t(m),
                                                   m,
                                                   stride_X,
                                                   batch_count);
                    if(i - BLOCK < 0)
                        break;
                    rocblas_internal_gemm<BATCHED>(handle,
                                                   transB,
                                                   transA,
                                                   m,
                                                   i,
                                                   BLOCK,
                                                   &alpha_negative_one<T>,
                                                   (U)X,
                                                   i * size_t(m),
                                                   m,
                                                   stride_X,
                                                   A,
                                                   i * size_t(lda) + offset_Ain,
                                                   lda,
                                                   stride_A,
                                                   &beta_1<T>,
                                                   B,
                                                   offset_Bin,
                                                   ldb,
                                                   stride_B,
                                                   batch_count);
                }
            }
        }
    } // tranpsose

    return rocblas_status_success;
}

#ifdef BUILD_WITH_TENSILE
template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
rocblas_status special_trsm_template(rocblas_handle    handle,
                                     rocblas_side      side,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     const T*          alpha,
                                     U                 A,
                                     rocblas_stride    offset_Ain,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_A,
                                     V                 B,
                                     rocblas_stride    offset_Bin,
                                     rocblas_int       ldb,
                                     rocblas_stride    stride_B,
                                     rocblas_int       batch_count,
                                     U                 invA,
                                     rocblas_stride    offset_invAin,
                                     rocblas_stride    stride_invA,
                                     size_t            B_chunk_size,
                                     V                 w_x_temp,
                                     rocblas_stride    stride_X)
{
    bool   parity = (transA == rocblas_operation_none) ^ (uplo == rocblas_fill_upper);
    size_t k      = side == rocblas_side_left ? m : n;
    size_t R      = k / BLOCK;
    size_t bsize  = side == rocblas_side_left ? n : m;
    size_t W      = 1 + (bsize - 1) / B_chunk_size;
    bool   tensile_supports_ldc_ne_ldd = rocblas_internal_tensile_supports_ldc_ne_ldd(handle);

    for(size_t w = 0; w < W; w++)
    {
        size_t width = std::min(bsize - w * B_chunk_size, B_chunk_size);

        if(side == rocblas_side_left)
        {
            for(size_t r = 0; r < R; r++)
            {
                size_t q = R - 1 - r;
                size_t j = parity ? r : q;

                // copy a BLOCK*n piece we are solving at a time
                if(!r || !tensile_supports_ldc_ne_ldd)
                    rocblas_copy_block_unit<T>(handle,
                                               BLOCK,
                                               width,
                                               B,
                                               ldb,
                                               stride_B,
                                               w_x_temp,
                                               BLOCK,
                                               stride_X,
                                               batch_count,
                                               j * BLOCK + w * B_chunk_size * ldb + offset_Bin,
                                               0);

                if(r)
                {
                    rocblas_stride offsetA = 0;
                    rocblas_stride offsetB = parity
                                                 ? w * B_chunk_size * size_t(ldb)
                                                 : w * B_chunk_size * size_t(ldb) + (q + 1) * BLOCK;

                    if(transA == rocblas_operation_none)
                        offsetA = parity ? r * BLOCK : BLOCK * (q * lda + q + lda);
                    else
                        offsetA = parity ? r * BLOCK * lda : BLOCK * (q * lda + q + 1);

                    if(!tensile_supports_ldc_ne_ldd)
                    {
                        rocblas_internal_gemm<BATCHED>(handle,
                                                       transA,
                                                       rocblas_operation_none,
                                                       BLOCK,
                                                       width,
                                                       r * BLOCK,
                                                       &alpha_negative_one<T>,
                                                       A,
                                                       offsetA + offset_Ain,
                                                       lda,
                                                       stride_A,
                                                       (U)B,
                                                       offsetB + offset_Bin,
                                                       ldb,
                                                       stride_B,
                                                       alpha,
                                                       w_x_temp,
                                                       rocblas_int(0),
                                                       BLOCK,
                                                       stride_X,
                                                       batch_count);
                    }
                    else
                    {
                        rocblas_datatype  compute_type   = rocblas_datatype_from_type<T>;
                        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
                        int32_t           solution_index = 0;
                        uint32_t          flags          = 0;

                        rocblas_gemm_ex_template<BATCHED>(handle,
                                                          transA,
                                                          rocblas_operation_none,
                                                          BLOCK,
                                                          width,
                                                          r * BLOCK,
                                                          &alpha_negative_one<T>,
                                                          A,
                                                          compute_type,
                                                          offsetA + offset_Ain,
                                                          lda,
                                                          stride_A,
                                                          B,
                                                          compute_type,
                                                          offsetB + offset_Bin,
                                                          ldb,
                                                          stride_B,
                                                          alpha,
                                                          B,
                                                          compute_type,
                                                          j * BLOCK + w * B_chunk_size * ldb
                                                              + offset_Bin,
                                                          ldb,
                                                          stride_B,
                                                          (void*)w_x_temp,
                                                          compute_type,
                                                          0,
                                                          BLOCK,
                                                          stride_X,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          0);
                    }
                }

                rocblas_internal_gemm<BATCHED>(
                    handle,
                    transA,
                    rocblas_operation_none,
                    BLOCK,
                    width,
                    BLOCK,
                    r ? &alpha_1<T> : alpha,
                    invA,
                    size_t(j * BLOCK * BLOCK + offset_invAin),
                    size_t(BLOCK),
                    stride_invA,
                    (U)w_x_temp,
                    size_t(0),
                    size_t(BLOCK),
                    stride_X,
                    &beta_0<T>,
                    B,
                    size_t(w * B_chunk_size * size_t(ldb) + j * BLOCK + offset_Bin),
                    size_t(ldb),
                    stride_B,
                    batch_count);
            }
        }
        else
        {
            for(size_t r = 0; r < R; r++)
            {
                size_t q = R - 1 - r;
                size_t j = parity ? q : r;

                // copy a m*BLOCK piece we are solving at a time
                if(!r || !tensile_supports_ldc_ne_ldd)
                    rocblas_copy_block_unit<T>(handle,
                                               width,
                                               BLOCK,
                                               B,
                                               ldb,
                                               stride_B,
                                               w_x_temp,
                                               width,
                                               stride_X,
                                               batch_count,
                                               j * BLOCK * size_t(ldb) + w * B_chunk_size
                                                   + offset_Bin,
                                               0);

                if(r)
                {
                    rocblas_stride offsetA = 0;
                    rocblas_stride offsetB = parity
                                                 ? w * B_chunk_size + (q + 1) * BLOCK * size_t(ldb)
                                                 : w * B_chunk_size;
                    if(transA == rocblas_operation_none)
                        offsetA = parity ? BLOCK * (q * lda + q + 1) : r * BLOCK * lda;
                    else
                        offsetA = parity ? BLOCK * (q * lda + q + lda) : r * BLOCK;

                    if(!tensile_supports_ldc_ne_ldd)
                    {
                        rocblas_internal_gemm<BATCHED>(handle,
                                                       rocblas_operation_none,
                                                       transA,
                                                       width,
                                                       BLOCK,
                                                       r * BLOCK,
                                                       &alpha_negative_one<T>,
                                                       (U)B,
                                                       size_t(offsetB + offset_Bin),
                                                       size_t(ldb),
                                                       stride_B,
                                                       A,
                                                       size_t(offsetA + offset_Ain),
                                                       size_t(lda),
                                                       stride_A,
                                                       alpha,
                                                       w_x_temp,
                                                       size_t(0),
                                                       width,
                                                       stride_X,
                                                       batch_count);
                    }
                    else
                    {
                        rocblas_datatype  compute_type   = rocblas_datatype_from_type<T>;
                        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
                        int32_t           solution_index = 0;
                        uint32_t          flags          = 0;

                        rocblas_gemm_ex_template<BATCHED>(handle,
                                                          rocblas_operation_none,
                                                          transA,
                                                          width,
                                                          BLOCK,
                                                          r * BLOCK,
                                                          &alpha_negative_one<T>,
                                                          B,
                                                          compute_type,
                                                          offsetB + offset_Bin,
                                                          ldb,
                                                          stride_B,
                                                          A,
                                                          compute_type,
                                                          offsetA + offset_Ain,
                                                          lda,
                                                          stride_A,
                                                          alpha,
                                                          B,
                                                          compute_type,
                                                          j * BLOCK * size_t(ldb) + w * B_chunk_size
                                                              + offset_Bin,
                                                          ldb,
                                                          stride_B,
                                                          (void*)w_x_temp,
                                                          compute_type,
                                                          0,
                                                          width,
                                                          stride_X,
                                                          batch_count,
                                                          compute_type,
                                                          algo,
                                                          solution_index,
                                                          0);
                    }
                }

                rocblas_internal_gemm<BATCHED>(
                    handle,
                    rocblas_operation_none,
                    transA,
                    width,
                    BLOCK,
                    BLOCK,
                    r ? &alpha_1<T> : alpha,
                    U(w_x_temp),
                    size_t(0),
                    width,
                    stride_X,
                    invA,
                    size_t(j * BLOCK * BLOCK + offset_invAin),
                    size_t(BLOCK),
                    stride_invA,
                    &beta_0<T>,
                    B,
                    size_t(w * B_chunk_size * size_t(ldb) + j * BLOCK * size_t(ldb) + offset_Bin),
                    size_t(ldb),
                    stride_B,
                    batch_count);
            }
        }
    }

    return rocblas_status_success;
}
#endif

inline bool
    trsm_is_skinny(rocblas_side side, rocblas_operation transA, rocblas_int m, rocblas_int n)
{
    // TODO: work on logic here. This is somewhat data-driven right now, but can be taken further.
    // We see optimizations for right-hand-side cases, along with smaller transpose cases, but just going to
    // worry about this specific configuration for now, in a future release we can expand this.
    return (side == rocblas_side_left && transA == rocblas_operation_none && m > n * 4 && m > 8192);
}

static const size_t rocblas_internal_trsm_reg_kernel_mem_limit = [] {
    // 128 MB
    // How much memory to limit usage of regular trsm_left and trsm_right kernels,
    // when trsm_special can also be used, to increase performance
    // i.e. reduce memory usage if possible if trying to allocate more than this amount of memory
    constexpr size_t TRSM_REG_KERNEL_MEM_LIMIT = 128 * 1024 * 1024;
    size_t           mem_limit;
    const char*      env = getenv("ROCBLAS_INTERNAL_TRSM_REG_KERNEL_MEM_LIMIT");
    return env && sscanf(env, "%zu", &mem_limit) == 1 ? mem_limit : TRSM_REG_KERNEL_MEM_LIMIT;
}();

template <rocblas_int BLOCK, bool BATCHED, typename T>
inline bool trsm_use_special_kernel(rocblas_side      side,
                                    rocblas_operation transA,
                                    rocblas_int       m,
                                    rocblas_int       n,
                                    rocblas_int       batch_count,
                                    rocblas_int       supplied_invA_size)
{
#ifndef BUILD_WITH_TENSILE
    return false;
#endif

    // small sizes have their own kernel
    if(m <= 64 || n <= 64)
        return false;

    rocblas_int k = side == rocblas_side_left ? m : n;

    // to use the special kernel, k must be divisible by block.
    // also, for skinny matrices, the regular kernels perform better
    const bool exact_blocks = (k % BLOCK) == 0;
    const bool is_skinny    = trsm_is_skinny(side, transA, m, n);

    // if k is not divisible by BLOCK, we can't use the special kernel
    if(!exact_blocks)
        return false;

    // if the matrix is not "skinny", go ahead with the special kernel
    if(!is_skinny)
        return true;

    // If the matrix IS "skinny", see if we can allocate enough memory for the regular
    // kernel without going over our defined memory limit

    // Calculate needed memory for regular kernel
    size_t invA_temp_bytes = 0;
    size_t x_temp_bytes    = 0;
    if(supplied_invA_size / BLOCK < k)
    {
        invA_temp_bytes = BLOCK * k * sizeof(T) * batch_count;

        // When k < BLOCK, C is unnecessary for trtri
        x_temp_bytes = ((k / BLOCK) * ((BLOCK / 2) * (BLOCK / 2))) * sizeof(T);

        // don't need remainder bytes, because k is divisible by BLOCK here
    }

    x_temp_bytes = std::max(size_t(m) * n * sizeof(T) * batch_count, x_temp_bytes);
    const size_t total_regular_kernel_req_mem
        = x_temp_bytes + invA_temp_bytes + (BATCHED ? 2 * sizeof(T) * batch_count : 0);

    // If the regular kernel as calculated here needs too much memory, use special kernel, otherwise use regular kernel.
    return total_regular_kernel_req_mem > rocblas_internal_trsm_reg_kernel_mem_limit;
}

inline bool rocblas_internal_trsm_use_substitution(rocblas_side side,
                                                   rocblas_int  m,
                                                   rocblas_int  n,
                                                   rocblas_int  batch_count)
{
    // from various rocBLAS profiling, the following bounds
    // have been chosen to decide between substitution method/inversion method
    // These are just empirically found and will not always be the best choice.
    // TODO: These are very conservative numbers and should be updated with future work.
    //       Ideally, we can use rocblas_trsm_blksize to determine whether or not to use
    //       the rocblas_trsm_small_substitution function or not.
    return side == rocblas_side_left
           && ((n <= 32 && batch_count >= 16 && m < 512) || (n > 32 && n <= 128 && m <= 340));
}

inline rocblas_int get_index(const rocblas_int* intervals, rocblas_int max, rocblas_int dim)
{
    rocblas_int i;

    for(i = 0; i < max; ++i)
    {
        if(dim <= intervals[i])
            break;
    }

    return i;
}

/** This function returns the block size for the internal blocked trsm implementation.
 *  The block sizes and logic is taken directly from rocSOLVER.
 */
template <bool BATCHED, typename T>
rocblas_int rocblas_trsm_blksize(rocblas_int m, rocblas_int n)
{
    rocblas_int blk = 0;

    if(BATCHED)
    {
        if constexpr(rocblas_is_complex<T>)
        {
            rocblas_int M = TRSM_BATCH_NUMROWS_COMPLEX - 1;
            rocblas_int N = TRSM_BATCH_NUMCOLS_COMPLEX - 1;
            blk = trsm_blksizes_complex_batch[get_index(trsm_intervals_row_complex_batch, M, m)]
                                             [get_index(trsm_intervals_col_complex_batch, N, n)];
        }
        else
        {
            rocblas_int M = TRSM_BATCH_NUMROWS_REAL - 1;
            rocblas_int N = TRSM_BATCH_NUMCOLS_REAL - 1;
            blk           = trsm_blksizes_real_batch[get_index(trsm_intervals_row_real_batch, M, m)]
                                          [get_index(trsm_intervals_col_real_batch, N, n)];
        }
    }
    else
    {
        if constexpr(rocblas_is_complex<T>)
        {
            rocblas_int M = TRSM_NUMROWS_COMPLEX - 1;
            rocblas_int N = TRSM_NUMCOLS_COMPLEX - 1;
            blk           = trsm_blksizes_complex_nonbatch
                [get_index(trsm_intervals_row_complex_nonbatch, M, m)]
                [get_index(trsm_intervals_col_complex_nonbatch, N, n)];
        }
        else
        {
            rocblas_int M = TRSM_NUMROWS_REAL - 1;
            rocblas_int N = TRSM_NUMCOLS_REAL - 1;
            blk = trsm_blksizes_real_nonbatch[get_index(trsm_intervals_row_real_nonbatch, M, m)]
                                             [get_index(trsm_intervals_col_real_nonbatch, N, n)];
        }
    }

    if(blk == 1)
        blk = std::min(m, 512);

    // Note: If blk remains zero, we won't be using the substitution method.
    return blk;
}

/*! \brief rocblas_internal_trsm_workspace_size
    Calculates needed memory allocation for trsm, does not allocate any memory.
    Note that for the batched version of trsm, we are also allocating memory to store the
    arrays of pointers for invA and w_x_temp.

    @param[in]
    side rocblas_side
        Whether matrix A is located on the left or right of X
    @param[in]
    m rocblas_int
        Number of rows of matrix B
    @param[in]
    n rocblas_int
        Number of columns of matrix B
    @param[in]
    batch_count rocblas_int
        Number of batches
    @param[in]
    supplied_invA_size rocblas_int
        If the user supplies an invA matrix, this may reduce the needed memory. supplied_invA_size
        specifies the number of elements in device memory of the supplied invA matrix.
    @param[out]
    w_x_tmp_size size_t
        The bytes of workspace memory needed for x_tmp in the trsm calculations
    @param[out]
    w_x_tmp_arr_size size_t
        The bytes of workspace memory needed for the array of pointers for x_tmp
    @param[out]
    w_invA_size size_t
        The bytes of workspace memory needed for invA in the trsm calculations
    @param[out]
    w_invA_arr_size size_t
        The bytes of workspace memory needed for the array of pointers for invA
    @param[out]
    w_x_tmp_size_backup size_t
        If the user is unable to allocate w_x_tmp_arr_size bytes, w_x_tmp_size_backup
        bytes may be used in trsm with degraded performance.
    ********************************************************************/
template <rocblas_int BLOCK, bool BATCHED, typename T>
rocblas_status rocblas_internal_trsm_workspace_size(rocblas_side      side,
                                                    rocblas_operation transA,
                                                    rocblas_int       m,
                                                    rocblas_int       n,
                                                    rocblas_int       batch_count,
                                                    rocblas_int       supplied_invA_size,
                                                    size_t*           w_x_tmp_size,
                                                    size_t*           w_x_tmp_arr_size,
                                                    size_t*           w_invA_size,
                                                    size_t*           w_invA_arr_size,
                                                    size_t*           w_x_tmp_size_backup)
{
    if(!w_x_tmp_size || !w_x_tmp_arr_size || !w_invA_size || !w_invA_arr_size
       || !w_x_tmp_size_backup)
    {
        return rocblas_status_invalid_pointer;
    }

    // can use trsv kernel for n == 1 && left
    if(n == 1 && side == rocblas_side_left)
    {
        *w_x_tmp_size        = batch_count * sizeof(rocblas_int);
        *w_x_tmp_arr_size    = 0;
        *w_invA_size         = 0;
        *w_invA_arr_size     = 0;
        *w_x_tmp_size_backup = 0;
        return rocblas_status_success;
    }

    rocblas_int k = side == rocblas_side_left ? m : n;

    // no memory needed if using small kernels
    bool is_small = (k <= 32) || (m <= 64 && n <= 64);
    if(is_small || !batch_count)
    {
        // return rocblas_status_continue indicating no memory needed
        *w_x_tmp_size        = 0;
        *w_x_tmp_arr_size    = 0;
        *w_invA_size         = 0;
        *w_invA_arr_size     = 0;
        *w_x_tmp_size_backup = 0;
        return rocblas_status_continue;
    }

    // no memory needed for substitution method, only used for specific sizes
    const bool  LEFT    = rocblas_side_left == side;
    rocblas_int blksize = rocblas_trsm_blksize<BATCHED, T>(LEFT ? m : n, LEFT ? n : m);
    const bool  use_sub = rocblas_internal_trsm_use_substitution(side, m, n, batch_count);

    if(use_sub && blksize)
    {
        *w_x_tmp_size        = 0;
        *w_x_tmp_arr_size    = 0;
        *w_invA_size         = 0;
        *w_invA_arr_size     = 0;
        *w_x_tmp_size_backup = 0;
        return rocblas_status_continue;
    }

    // Whether size is an exact multiple of blocksize
    const bool exact_blocks = (k % BLOCK) == 0;
    const bool use_special  = trsm_use_special_kernel<BLOCK, BATCHED, T>(
        side, transA, m, n, batch_count, supplied_invA_size);

    size_t invA_temp_bytes     = 0;
    size_t c_temp_bytes        = 0;
    size_t x_temp_bytes        = 0;
    size_t x_temp_bytes_backup = 0;

    // Only allocate bytes for invA if invA is not supplied or supplied_invA_size is too small
    if(supplied_invA_size / BLOCK < k)
    {
        invA_temp_bytes = BLOCK * k * sizeof(T) * batch_count;

        // When k < BLOCK, C is unnecessary for trtri
        // c_temp is not scaled by batch_count as trtri kernels are currently naive and
        // just iterate through the batches using the same workspace memory
        c_temp_bytes = ((k / BLOCK) * ((BLOCK / 2) * (BLOCK / 2))) * sizeof(T);

        // For the TRTRI last diagonal block we need remainder space if k % BLOCK != 0
        if(!exact_blocks)
        {
            // TODO: Make this more accurate -- right now it's much larger than necessary
            size_t remainder_bytes = ROCBLAS_TRTRI_NB * BLOCK * 2 * sizeof(T);

            // C is the maximum of the temporary space needed for TRTRI
            c_temp_bytes = std::max(c_temp_bytes, remainder_bytes);
        }
    }

    // non-special kernel (regular left/right kernel) when not exact blocks. Also used
    // when exact_blocks, but is a skinny matrix, as this out-performs the special kernel
    if(!use_special)
    {
        // When k % BLOCK != 0, we need m * n space
        x_temp_bytes_backup = x_temp_bytes = size_t(m) * n * sizeof(T) * batch_count;
    }

    if(use_special)
    {
        // Optimal B_chunk_size is the orthogonal dimension to k
        size_t B_chunk_size = size_t(m) + size_t(n) - size_t(k);

        // When k % BLOCK == 0, we only need BLOCK * B_chunk_size space
        x_temp_bytes = BLOCK * B_chunk_size * sizeof(T) * batch_count;

        // backup memory allocation if initial allocation fails, only for exact blocks
        x_temp_bytes_backup = BLOCK * sizeof(T) * batch_count;
    }

    // X and C temporaries can share space, so the maximum size is allocated
    *w_x_tmp_size        = std::max(x_temp_bytes, c_temp_bytes);
    *w_x_tmp_arr_size    = BATCHED ? sizeof(T*) * batch_count : 0;
    *w_invA_size         = invA_temp_bytes;
    *w_invA_arr_size     = BATCHED ? sizeof(T*) * batch_count : 0;
    *w_x_tmp_size_backup = x_temp_bytes_backup;

    return rocblas_status_success;
}

#define TRSM_WORKSPACE_TEMPLATE_PARAMS                                                   \
    side, transA, m, n, batch_count, supplied_invA_size, w_x_tmp_size, w_x_tmp_arr_size, \
        w_invA_size, w_invA_arr_size, w_x_tmp_size_backup

template <typename T>
rocblas_status rocblas_internal_trsm_workspace_size(rocblas_side      side,
                                                    rocblas_operation transA,
                                                    rocblas_int       m,
                                                    rocblas_int       n,
                                                    rocblas_int       batch_count,
                                                    rocblas_int       supplied_invA_size,
                                                    size_t*           w_x_tmp_size,
                                                    size_t*           w_x_tmp_arr_size,
                                                    size_t*           w_invA_size,
                                                    size_t*           w_invA_arr_size,
                                                    size_t*           w_x_tmp_size_backup)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, false, T>(
            TRSM_WORKSPACE_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, false, T>(
            TRSM_WORKSPACE_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, false, T>(
            TRSM_WORKSPACE_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, false, T>(
            TRSM_WORKSPACE_TEMPLATE_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
rocblas_status rocblas_internal_trsm_batched_workspace_size(rocblas_side      side,
                                                            rocblas_operation transA,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            rocblas_int       batch_count,
                                                            rocblas_int       supplied_invA_size,
                                                            size_t*           w_x_tmp_size,
                                                            size_t*           w_x_tmp_arr_size,
                                                            size_t*           w_invA_size,
                                                            size_t*           w_invA_arr_size,
                                                            size_t*           w_x_tmp_size_backup)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, true, T>(
            TRSM_WORKSPACE_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, true, T>(
            TRSM_WORKSPACE_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, true, T>(
            TRSM_WORKSPACE_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trsm_workspace_size<ROCBLAS_TRSM_NB, true, T>(
            TRSM_WORKSPACE_TEMPLATE_PARAMS);

    return rocblas_status_not_implemented;
}

#undef TRSM_WORKSPACE_TEMPLATE_PARAMS

/**
 *  The purpose of this function is to allocate memory for trsm. It is added to remove
 *  memory allocation from the rocblas_internal_trsm_template function, but also allow code reuse
 *  from the _impl functions.
 *
 *  Note that for the batched version of trsm, we are also allocating memory to store the
 *  arrays of pointers for invA and w_x_temp (mem_x_temp_arr, mem_invA_arr).
 */
template <bool BATCHED, typename T, typename U>
rocblas_status rocblas_internal_trsm_template_mem(rocblas_handle              handle,
                                                  rocblas_side                side,
                                                  rocblas_operation           transA,
                                                  rocblas_int                 m,
                                                  rocblas_int                 n,
                                                  rocblas_int                 lda,
                                                  rocblas_int                 ldb,
                                                  rocblas_int                 batch_count,
                                                  rocblas_device_malloc_base& w_mem,
                                                  void*&                      w_mem_x_temp,
                                                  void*&                      w_mem_x_temp_arr,
                                                  void*&                      w_mem_invA,
                                                  void*&                      w_mem_invA_arr,
                                                  U                           supplied_invA,
                                                  rocblas_int                 supplied_invA_size)
{
    // Note, for 32-bit implementation we don't need lda/ldb. They're only used in the 64-bit version
    // to see if we can use this 32-bit implementation or the specialized 64-bit one
    constexpr rocblas_int BLOCK     = ROCBLAS_TRSM_NB;
    auto&                 workspace = static_cast<decltype(handle->device_malloc(0))&>(w_mem);

    // calculate needed memory
    size_t w_x_tmp_size, w_x_tmp_arr_size, w_invA_size, w_invA_arr_size, w_x_tmp_size_backup;
    rocblas_status memory_status;
    if constexpr(BATCHED)
        memory_status = rocblas_internal_trsm_batched_workspace_size<T>(side,
                                                                        transA,
                                                                        m,
                                                                        n,
                                                                        batch_count,
                                                                        supplied_invA_size,
                                                                        &w_x_tmp_size,
                                                                        &w_x_tmp_arr_size,
                                                                        &w_invA_size,
                                                                        &w_invA_arr_size,
                                                                        &w_x_tmp_size_backup);
    else
        memory_status = rocblas_internal_trsm_workspace_size<T>(side,
                                                                transA,
                                                                m,
                                                                n,
                                                                batch_count,
                                                                supplied_invA_size,
                                                                &w_x_tmp_size,
                                                                &w_x_tmp_arr_size,
                                                                &w_invA_size,
                                                                &w_invA_arr_size,
                                                                &w_x_tmp_size_backup);

    if(memory_status != rocblas_status_success && memory_status != rocblas_status_continue)
    {
        return memory_status;
    }

    if(handle->is_device_memory_size_query())
    {
        // indicates no memory needed
        if(memory_status == rocblas_status_continue)
        {
            return rocblas_status_size_unchanged;
        }
        else
        {
            return handle->set_optimal_device_memory_size(
                w_x_tmp_size, w_x_tmp_arr_size, w_invA_size, w_invA_arr_size);
        }
    }

    rocblas_int k = side == rocblas_side_left ? m : n;
    if(supplied_invA && supplied_invA_size / BLOCK < k)
    {
        // One-time warning message
        static auto& once = rocblas_cerr
                            << "WARNING: TRSM invA_size argument is too small; invA "
                               "argument is being ignored; TRSM performance is degraded"
                            << std::endl;
    }

    rocblas_status perf_status = rocblas_status_success;
    if(memory_status == rocblas_status_success)
    {
        // allocate memory
        workspace
            = handle->device_malloc(w_x_tmp_size, w_x_tmp_arr_size, w_invA_size, w_invA_arr_size);

        if(!workspace)
        {
            // if memory allocation fails, try backup. If that fails, return error.
            workspace = handle->device_malloc(
                w_x_tmp_size_backup, w_x_tmp_arr_size, w_invA_size, w_invA_arr_size);

            if(!workspace)
                return rocblas_status_memory_error;

            static auto& once = rocblas_cerr
                                << "WARNING: Device memory allocation size is too small for "
                                   "TRSM; TRSM performance is degraded"
                                << std::endl;

            perf_status = rocblas_status_perf_degraded;
        }

        w_mem_x_temp     = workspace[0];
        w_mem_x_temp_arr = workspace[1];
        w_mem_invA       = workspace[2];
        w_mem_invA_arr   = workspace[3];
    }

    return perf_status;
}

/* T = float, double, etc.
 * SCAL = T* or T
 * ATYPE = const T* or const T* const *
 * BTYPE = T* or T* const *
 *
 *  Uses the substitution method to solve a small problem AX = B.
 */
template <typename T, typename SCAL, typename ATYPE, typename BTYPE, const int NB>
ROCBLAS_KERNEL(NB)
rocblas_trsm_small_right_device(rocblas_fill      uplo,
                                rocblas_operation transA,
                                rocblas_diagonal  diag,
                                int               m,
                                int               n,
                                SCAL              alpha_dev_host,
                                ATYPE             Aa,
                                rocblas_stride    offset_A,
                                int               lda,
                                rocblas_stride    stride_A,
                                BTYPE             Ba,
                                rocblas_stride    offset_B,
                                int               ldb,
                                rocblas_stride    stride_B,
                                int               batch_count)
{
    auto alpha = load_scalar(alpha_dev_host);

    uint32_t batchid = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batchid < batch_count; batchid += c_YZ_grid_launch_limit)
    {
#endif

        auto A = load_ptr_batch(Aa, batchid, offset_A, stride_A);
        auto B = load_ptr_batch(Ba, batchid, offset_B, stride_B);

        bool      LOWER = uplo == rocblas_fill_lower;
        bool      CONJ  = transA == rocblas_operation_conjugate_transpose;
        const int tx    = threadIdx.x;
        const int bx    = blockIdx.x;

        // max A column to read from
        int maxColA = NB - 1 > n - 1 ? n - 1 : NB - 1;
        // NB columns, unless last block, then do leftover
        const int maxColB = (bx < gridDim.x - 1) ? NB : m - bx * NB;

        // offset B into correct block row
        B += size_t(bx) * NB;

        __shared__ T sA[NB * NB];
        __shared__ T sB[NB * NB];

        T resB[4];

        if(tx <= maxColA)
        {
            // Load A into sA, handle conjugation if necessary
            for(int i = 0; i <= maxColA; i++)
                sA[i * NB + tx] = (CONJ) ? conj(A[i * size_t(lda) + tx]) : A[i * size_t(lda) + tx];

            // set unit diagonal if needed
            if(diag == rocblas_diagonal_unit)
                sA[tx * NB + tx] = T(1.0);
        }

        if(tx < maxColB)
        {
            // Load B into sB and multiply by alpha
            for(int i = 0; i < n; i++)
                sB[i * NB + tx] = alpha * B[i * size_t(ldb) + tx];
        }
        __syncthreads();

        // Solve for B in shared memory
        if(transA == rocblas_operation_none && uplo == rocblas_fill_upper)
        {
            int i;
            for(i = 0; i + 3 <= maxColA; i += 4)
            {
                // Subtract previously solved parts
                resB[0] = sB[(i + 0) * NB + tx];
                resB[1] = sB[(i + 1) * NB + tx];
                resB[2] = sB[(i + 2) * NB + tx];
                resB[3] = sB[(i + 3) * NB + tx];

                for(int j = 0; j < i; j++)
                {
                    T sB_reg = sB[j * NB + tx];
                    resB[0] -= sB_reg * sA[(i + 0) * NB + j];
                    resB[1] -= sB_reg * sA[(i + 1) * NB + j];
                    resB[2] -= sB_reg * sA[(i + 2) * NB + j];
                    resB[3] -= sB_reg * sA[(i + 3) * NB + j];
                }

                resB[0] /= sA[(i + 0) * NB + (i + 0)];
                sB[(i + 0) * NB + tx] = resB[0];

                resB[1] -= resB[0] * sA[(i + 1) * NB + (i + 0)];
                resB[1] /= sA[(i + 1) * NB + (i + 1)];
                sB[(i + 1) * NB + tx] = resB[1];

                resB[2] -= resB[0] * sA[(i + 2) * NB + (i + 0)];
                resB[2] -= resB[1] * sA[(i + 2) * NB + (i + 1)];
                resB[2] /= sA[(i + 2) * NB + (i + 2)];
                sB[(i + 2) * NB + tx] = resB[2];

                resB[3] -= resB[0] * sA[(i + 3) * NB + (i + 0)];
                resB[3] -= resB[1] * sA[(i + 3) * NB + (i + 1)];
                resB[3] -= resB[2] * sA[(i + 3) * NB + (i + 2)];
                resB[3] /= sA[(i + 3) * NB + (i + 3)];
                sB[(i + 3) * NB + tx] = resB[3];
            }

            // tail end if not divisible by 4
            for(; i <= maxColA; i++)
            {
                resB[0] = sB[i * NB + tx];
                for(int j = 0; j < i; j++)
                {
                    resB[0] -= sB[j * NB + tx] * sA[i * NB + j];
                }
                sB[i * NB + tx] = resB[0] / sA[i * NB + i];
            }
        }
        else if(transA == rocblas_operation_none && uplo == rocblas_fill_lower)
        {
            int i;
            for(i = maxColA; i >= 3; i -= 4)
            {
                resB[0] = sB[(i - 0) * NB + tx];
                resB[1] = sB[(i - 1) * NB + tx];
                resB[2] = sB[(i - 2) * NB + tx];
                resB[3] = sB[(i - 3) * NB + tx];

                for(int j = maxColA; j > i; j--)
                {
                    T sB_reg = sB[j * NB + tx];
                    resB[0] -= sB_reg * sA[(i - 0) * NB + j];
                    resB[1] -= sB_reg * sA[(i - 1) * NB + j];
                    resB[2] -= sB_reg * sA[(i - 2) * NB + j];
                    resB[3] -= sB_reg * sA[(i - 3) * NB + j];
                }

                resB[0] /= sA[(i - 0) * NB + (i - 0)];
                sB[(i - 0) * NB + tx] = resB[0];

                resB[1] -= resB[0] * sA[(i - 1) * NB + (i - 0)];
                resB[1] /= sA[(i - 1) * NB + (i - 1)];
                sB[(i - 1) * NB + tx] = resB[1];

                resB[2] -= resB[0] * sA[(i - 2) * NB + (i - 0)];
                resB[2] -= resB[1] * sA[(i - 2) * NB + (i - 1)];
                resB[2] /= sA[(i - 2) * NB + (i - 2)];
                sB[(i - 2) * NB + tx] = resB[2];

                resB[3] -= resB[0] * sA[(i - 3) * NB + (i - 0)];
                resB[3] -= resB[1] * sA[(i - 3) * NB + (i - 1)];
                resB[3] -= resB[2] * sA[(i - 3) * NB + (i - 2)];
                resB[3] /= sA[(i - 3) * NB + (i - 3)];
                sB[(i - 3) * NB + tx] = resB[3];
            }

            for(; i >= 0; i--)
            {
                resB[0] = sB[i * NB + tx];
                for(int j = maxColA; j > i; j--)
                {
                    resB[0] -= sB[j * NB + tx] * sA[i * NB + j];
                }
                sB[i * NB + tx] = resB[0] / sA[i * NB + i];
            }
        }
        else if(uplo == rocblas_fill_upper)
        {
            int i;
            for(i = maxColA; i >= 3; i -= 4)
            {
                resB[0] = sB[(i - 0) * NB + tx];
                resB[1] = sB[(i - 1) * NB + tx];
                resB[2] = sB[(i - 2) * NB + tx];
                resB[3] = sB[(i - 3) * NB + tx];

                for(int j = maxColA; j > i; j--)
                {
                    rocblas_int col_off = j * NB;
                    T           sB_reg  = sB[col_off + tx];
                    resB[0] -= sB_reg * sA[col_off + (i - 0)];
                    resB[1] -= sB_reg * sA[col_off + (i - 1)];
                    resB[2] -= sB_reg * sA[col_off + (i - 2)];
                    resB[3] -= sB_reg * sA[col_off + (i - 3)];
                }

                resB[0] /= sA[(i - 0) * NB + (i - 0)];
                sB[(i - 0) * NB + tx] = resB[0];

                resB[1] -= resB[0] * sA[(i - 0) * NB + (i - 1)];
                resB[1] /= sA[(i - 1) * NB + (i - 1)];
                sB[(i - 1) * NB + tx] = resB[1];

                resB[2] -= resB[0] * sA[(i - 0) * NB + (i - 2)];
                resB[2] -= resB[1] * sA[(i - 1) * NB + (i - 2)];
                resB[2] /= sA[(i - 2) * NB + (i - 2)];
                sB[(i - 2) * NB + tx] = resB[2];

                resB[3] -= resB[0] * sA[(i - 0) * NB + (i - 3)];
                resB[3] -= resB[1] * sA[(i - 1) * NB + (i - 3)];
                resB[3] -= resB[2] * sA[(i - 2) * NB + (i - 3)];
                resB[3] /= sA[(i - 3) * NB + (i - 3)];
                sB[(i - 3) * NB + tx] = resB[3];
            }

            for(; i >= 0; i--)
            {
                resB[0] = sB[i * NB + tx];
                for(int j = maxColA; j > i; j--)
                {
                    resB[0] -= sB[j * NB + tx] * sA[j * NB + i];
                }
                sB[i * NB + tx] = resB[0] / sA[i * NB + i];
            }
        }
        else // lower (conjugate-)transpose
        {
            int i;
            for(i = 0; i + 3 <= maxColA; i += 4)
            {
                // Subtract previously solved parts
                resB[0] = sB[(i + 0) * NB + tx];
                resB[1] = sB[(i + 1) * NB + tx];
                resB[2] = sB[(i + 2) * NB + tx];
                resB[3] = sB[(i + 3) * NB + tx];

                for(int j = 0; j < i; j++)
                {
                    rocblas_int col_off = j * NB;
                    T           sB_reg  = sB[col_off + tx];
                    resB[0] -= sB_reg * sA[col_off + (i + 0)];
                    resB[1] -= sB_reg * sA[col_off + (i + 1)];
                    resB[2] -= sB_reg * sA[col_off + (i + 2)];
                    resB[3] -= sB_reg * sA[col_off + (i + 3)];
                }

                resB[0] /= sA[(i + 0) * NB + (i + 0)];
                sB[(i + 0) * NB + tx] = resB[0];

                resB[1] -= resB[0] * sA[(i + 0) * NB + (i + 1)];
                resB[1] /= sA[(i + 1) * NB + (i + 1)];
                sB[(i + 1) * NB + tx] = resB[1];

                resB[2] -= resB[0] * sA[(i + 0) * NB + (i + 2)];
                resB[2] -= resB[1] * sA[(i + 1) * NB + (i + 2)];
                resB[2] /= sA[(i + 2) * NB + (i + 2)];
                sB[(i + 2) * NB + tx] = resB[2];

                resB[3] -= resB[0] * sA[(i + 0) * NB + (i + 3)];
                resB[3] -= resB[1] * sA[(i + 1) * NB + (i + 3)];
                resB[3] -= resB[2] * sA[(i + 2) * NB + (i + 3)];
                resB[3] /= sA[(i + 3) * NB + (i + 3)];
                sB[(i + 3) * NB + tx] = resB[3];
            }

            // tail end if not divisible by 4
            for(; i <= maxColA; i++)
            {
                resB[0] = sB[i * NB + tx];
                for(int j = 0; j < i; j++)
                {
                    resB[0] -= sB[j * NB + tx] * sA[j * NB + i];
                }
                sB[i * NB + tx] = resB[0] / sA[i * NB + i];
            }
        }

        // Save shared memory back into B
        if(tx < maxColB)
        {
            for(int i = 0; i < n; i++)
                B[i * size_t(ldb) + tx] = sB[i * NB + tx];
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/*
 *  Uses a substitution method to solve a small problem AX = B. This version uses
 *  less shared memory for double complex types (currently this means not using
 *  shared memory for A).
 */
template <typename T, typename SCAL, typename ATYPE, typename BTYPE, const int NB>
ROCBLAS_KERNEL(NB)
rocblas_trsm_small_64_right_device(rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_diagonal  diag,
                                   int               m,
                                   int               n,
                                   SCAL              alpha_dev_host,
                                   ATYPE             Aa,
                                   rocblas_stride    offset_A,
                                   int               lda,
                                   rocblas_stride    stride_A,
                                   BTYPE             Ba,
                                   rocblas_stride    offset_B,
                                   int               ldb,
                                   rocblas_stride    stride_B,
                                   int               batch_count)
{
    auto alpha = load_scalar(alpha_dev_host);

    uint32_t batchid = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batchid < batch_count; batchid += c_YZ_grid_launch_limit)
    {
#endif

        auto A = load_ptr_batch(Aa, batchid, offset_A, stride_A);
        auto B = load_ptr_batch(Ba, batchid, offset_B, stride_B);

        bool      LOWER = uplo == rocblas_fill_lower;
        bool      CONJ  = transA == rocblas_operation_conjugate_transpose;
        const int tx    = threadIdx.x;
        const int bx    = blockIdx.x;

        // max A column to read from
        int maxColA = NB - 1 > n - 1 ? n - 1 : NB - 1;
        // NB columns, unless last block, then do leftover
        const int maxColB = (bx < gridDim.x - 1) ? NB : m - bx * NB;

        // offset B into correct block row
        B += bx * size_t(NB);

        __shared__ T sB[NB * NB];

        if(tx < maxColB)
        {
            // Load B into sB and multiply by alpha
            for(int i = 0; i < n; i++)
                sB[i * NB + tx] = alpha * B[i * size_t(ldb) + tx];
        }
        __syncthreads();
        // Solve for B in shared memory
        if(transA == rocblas_operation_none && uplo == rocblas_fill_upper)
        {
            // Note: I didn't find that using 4 registers (as in the above function) to be noticeably faster in this version

            for(int i = 0; i <= maxColA; i++)
            {
                // Subtract previously solved parts
                T temp_reg_B = sB[i * NB + tx];
                for(int j = 0; j < i; j++)
                {
                    T valA = A[i * size_t(lda) + j];
                    temp_reg_B -= sB[j * NB + tx] * valA;
                }
                // Solve
                sB[i * NB + tx] = temp_reg_B;
                if(diag != rocblas_diagonal_unit)
                    sB[i * NB + tx] /= A[i * size_t(lda) + i];
            }
        }
        else if(transA == rocblas_operation_none && uplo == rocblas_fill_lower)
        {
            for(int i = maxColA; i >= 0; i--)
            {
                T temp_reg_B = sB[i * NB + tx];
                for(int j = maxColA; j > i; j--)
                {
                    T valA = A[i * size_t(lda) + j];
                    temp_reg_B -= sB[j * NB + tx] * valA;
                }
                sB[i * NB + tx] = temp_reg_B;
                if(diag != rocblas_diagonal_unit)
                    sB[i * NB + tx] /= A[i * size_t(lda) + i];
            }
        }
        else if(uplo == rocblas_fill_upper)
        {
            for(int i = maxColA; i >= 0; i--)
            {
                T temp_reg_B = sB[i * NB + tx];
                for(int j = maxColA; j > i; j--)
                {
                    T valA = CONJ ? conj(A[j * size_t(lda) + i]) : A[j * size_t(lda) + i];
                    temp_reg_B -= sB[j * NB + tx] * valA;
                }
                sB[i * NB + tx] = temp_reg_B;
                if(diag != rocblas_diagonal_unit)
                    sB[i * NB + tx] /= CONJ ? conj(A[i * size_t(lda) + i]) : A[i * size_t(lda) + i];
            }
        }
        else // lower (conjugate-)transpose
        {
            for(int i = 0; i <= maxColA; i++)
            {
                T temp_reg_B = sB[i * NB + tx];
                for(int j = 0; j < i; j++)
                {
                    T valA = CONJ ? conj(A[j * size_t(lda) + i]) : A[j * size_t(lda) + i];
                    temp_reg_B -= sB[j * NB + tx] * valA;
                }
                sB[i * NB + tx] = temp_reg_B;
                if(diag != rocblas_diagonal_unit)
                    sB[i * NB + tx] /= CONJ ? conj(A[i * size_t(lda) + i]) : A[i * size_t(lda) + i];
            }
        }

        // Save shared memory back into B
        if(tx < maxColB)
        {
            for(int i = 0; i < n; i++)
                B[i * size_t(ldb) + tx] = sB[i * NB + tx];
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/* T = float, double, etc.
 * SCAL = T* or T
 * ATYPE = const T* or const T* const *
 * BTYPE = T* or T* const *
 *
 * Uses the substitution method to solve a small problem XA = B.
 */
template <const int NB,
          const int STEP_SIZE,
          bool      LOWER,
          typename T,
          typename SCAL,
          typename ATYPE,
          typename BTYPE>
ROCBLAS_KERNEL(NB)
rocblas_trsm_small_left_device(rocblas_fill      uplo,
                               rocblas_operation transA,
                               rocblas_diagonal  diag,
                               int               m,
                               int               n,
                               SCAL              alpha_dev_host,
                               ATYPE             Aa,
                               rocblas_stride    offset_A,
                               int               lda,
                               rocblas_stride    stride_A,
                               BTYPE             Ba,
                               rocblas_stride    offset_B,
                               int               ldb,
                               rocblas_stride    stride_B,
                               int               batch_count)
{
    bool CONJ  = transA == rocblas_operation_conjugate_transpose;
    auto alpha = load_scalar(alpha_dev_host);

    uint32_t batchid = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batchid < batch_count; batchid += c_YZ_grid_launch_limit)
    {
#endif
        auto A = load_ptr_batch(Aa, batchid, offset_A, stride_A);
        auto B = load_ptr_batch(Ba, batchid, offset_B, stride_B);

        const int tx = threadIdx.x;
        const int bx = blockIdx.x;

        // max A column to read from
        int maxColA = NB - 1 > m - 1 ? m - 1 : NB - 1;
        // NB columns, unless last block, then do leftover
        const int maxColB = (bx < gridDim.x - 1) ? NB : n - bx * NB;

        constexpr int num_step_sizes = 3;
        constexpr int step_size_1    = STEP_SIZE;
        constexpr int step_size_2    = STEP_SIZE < NB ? (4) : // special case for NB = 64
                                        (NB - 4) > 0 ? (NB - 4)
                                                        : 1;
        constexpr int step_sizes[]   = {step_size_1, step_size_2, 1};

        // offset B into correct block column
        B += (tx + bx * NB) * size_t(ldb);

        // shared A, registers for B
        __shared__ T sA[NB * NB];
        T            resB[step_size_1];

        if(tx <= maxColA)
        {
            for(int i = 0; i <= maxColA; i++)
            {
                // indexing will be transposed for lower-transpose and upper-non-transpose for better memory access
                sA[i * NB + tx] = (CONJ) ? conj(A[i * size_t(lda) + tx]) : A[i * size_t(lda) + tx];
            }

            // set unit diagonal if needed
            if(diag == rocblas_diagonal_unit)
                sA[tx * NB + tx] = T(1.0);
            else
                sA[tx * NB + tx]
                    = T(1.0)
                      / sA[tx * NB + tx]; // invert diagonal here so just have to multiply later
        }
        __syncthreads();

        if(tx >= maxColB)
            return;

        // Solve for B in shared memory
        if(LOWER && transA == rocblas_operation_none)
        {
            int i = 0;
            for(int idx = 0; idx < num_step_sizes; idx++)
            {
                const int step_size = step_sizes[idx];
                for(; i + (step_size - 1) <= maxColA; i += step_size)
                {
                    // Subtract previously solved parts
                    for(int j = 0; j < step_size; j++)
                        resB[j] = alpha * B[i + j];

                    for(int j1 = 0; j1 < i; j1++)
                    {
                        rocblas_int col_off = j1 * NB;
                        T           sB_reg  = B[j1];
                        for(int j2 = 0; j2 < step_size; j2++)
                            resB[j2] -= sB_reg * sA[col_off + i + j2];
                    }

                    for(int j1 = 0; j1 < step_size; j1++)
                    {
                        for(int j2 = 0; j2 < j1; j2++)
                            resB[j1] -= resB[j2] * sA[(i + j2) * NB + (i + j1)];

                        resB[j1] *= sA[(i + j1) * NB + (i + j1)];
                        B[i + j1] = resB[j1];
                    }
                }
                if(i > maxColA)
                    break;
            }
        }
        else if(!LOWER && transA == rocblas_operation_none)
        {
            int i = maxColA;
            for(int idx = 0; idx < num_step_sizes; idx++)
            {
                const int step_size = step_sizes[idx];
                for(; i >= (step_size - 1); i -= step_size)
                {
                    for(int j = 0; j < step_size; j++)
                        resB[j] = alpha * B[i - j];

                    for(int j1 = maxColA; j1 > i; j1--)
                    {
                        rocblas_int col_off = j1;
                        T           sB_reg  = B[j1];
                        for(int j2 = 0; j2 < step_size; j2++)
                            resB[j2] -= sB_reg * sA[(col_off * NB + (i - j2))];
                    }

                    for(int j1 = 0; j1 < step_size; j1++)
                    {
                        for(int j2 = 0; j2 < j1; j2++)
                            resB[j1] -= resB[j2] * sA[(i - j1) + NB * (i - j2)];

                        resB[j1] *= sA[(i - j1) + NB * (i - j1)];
                        B[i - j1] = resB[j1];
                    }
                }
                if(i < 0)
                    break;
            }
        }
        else if(LOWER)
        {
            int i = maxColA;
            for(int idx = 0; idx < num_step_sizes; idx++)
            {
                const int step_size = step_sizes[idx];
                for(; i >= (step_size - 1); i -= step_size)
                {
                    for(int j = 0; j < step_size; j++)
                        resB[j] = alpha * B[i - j];

                    for(int j1 = maxColA; j1 > i; j1--)
                    {
                        rocblas_int col_off = j1;
                        T           sB_reg  = B[j1];
                        for(int j2 = 0; j2 < step_size; j2++)
                            resB[j2] -= sB_reg * sA[(i - j2) * NB + j1];
                    }

                    for(int j1 = 0; j1 < step_size; j1++)
                    {
                        for(int j2 = 0; j2 < j1; j2++)
                            resB[j1] -= resB[j2] * sA[(i - j2) + NB * (i - j1)];

                        resB[j1] *= sA[(i - j1) + NB * (i - j1)];
                        B[i - j1] = resB[j1];
                    }
                }
                if(i < 0)
                    break;
            }
        }
        else if(!LOWER)
        {
            int i = 0;
            for(int idx = 0; idx < num_step_sizes; idx++)
            {
                const int step_size = step_sizes[idx];
                for(; i + (step_size - 1) <= maxColA; i += step_size)
                {
                    // Subtract previously solved parts
                    for(int j = 0; j < step_size; j++)
                        resB[j] = alpha * B[i + j];

                    for(int j1 = 0; j1 < i; j1++)
                    {
                        T sB_reg = B[j1];
                        for(int j2 = 0; j2 < step_size; j2++)
                            resB[j2] -= sB_reg * sA[(i + j2) * NB + j1];
                    }

                    for(int j1 = 0; j1 < step_size; j1++)
                    {
                        for(int j2 = 0; j2 < j1; j2++)
                            resB[j1] -= resB[j2] * sA[(i + j1) * NB + (i + j2)];

                        resB[j1] *= sA[(i + j1) * NB + (i + j1)];
                        B[i + j1] = resB[j1];
                    }
                }
                if(i > maxColA)
                    break;
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <const int NB,
          const int STEP_SIZE,
          bool      LOWER,
          typename T,
          typename SCAL,
          typename ATYPE,
          typename BTYPE>
ROCBLAS_KERNEL(NB)
rocblas_trsm_small_left_device_sharedB(rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_diagonal  diag,
                                       int               m,
                                       int               n,
                                       SCAL              alpha_dev_host,
                                       ATYPE             Aa,
                                       rocblas_stride    offset_A,
                                       int               lda,
                                       rocblas_stride    stride_A,
                                       BTYPE             Ba,
                                       rocblas_stride    offset_B,
                                       int               ldb,
                                       rocblas_stride    stride_B,
                                       int               batch_count)
{
    bool CONJ  = transA == rocblas_operation_conjugate_transpose;
    auto alpha = load_scalar(alpha_dev_host);

    uint32_t batchid = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batchid < batch_count; batchid += c_YZ_grid_launch_limit)
    {
#endif
        auto A = load_ptr_batch(Aa, batchid, offset_A, stride_A);
        auto B = load_ptr_batch(Ba, batchid, offset_B, stride_B);

        const int tx = threadIdx.x;
        const int bx = blockIdx.x;

        // max A column to read from
        int maxColA = NB - 1 > m - 1 ? m - 1 : NB - 1;
        // NB columns, unless last block, then do leftover
        const int maxColB = (bx < gridDim.x - 1) ? NB : n - bx * NB;

        constexpr int num_step_sizes = 3;
        constexpr int step_size_1    = STEP_SIZE;
        constexpr int step_size_2    = STEP_SIZE < NB ? (4) : // special case for NB = 64
                                        (NB - 4) > 0 ? (NB - 4)
                                                        : 1;
        constexpr int step_sizes[]   = {step_size_1, step_size_2, 1};

        // offset B into correct block column
        B += (bx * NB) * size_t(ldb);

        // shared A, registers for B
        __shared__ T sA[NB * NB];
        __shared__ T sB[NB * NB];
        T            resB[step_size_1];

        if(tx <= maxColA)
        {
            for(int i = 0; i <= maxColA; i++)
            {
                // indexing will be transposed for lower-transpose and upper-non-transpose for better memory access
                sA[i * NB + tx] = (CONJ) ? conj(A[i * size_t(lda) + tx]) : A[i * size_t(lda) + tx];
            }

            // set unit diagonal if needed
            if(diag == rocblas_diagonal_unit)
                sA[tx * NB + tx] = T(1.0);
            else
                sA[tx * NB + tx]
                    = T(1.0)
                      / sA[tx * NB + tx]; // invert diagonal here so just have to multiply later
        }

        // Load B into sB and multiply by alpha, transpose for better mem. access
        if(tx < maxColB)
            for(int i = 0; i <= maxColA; i++)
                sB[tx + NB * i] = alpha * B[i + tx * size_t(ldb)];
        __syncthreads();

        // Solve for B in shared memory
        if(LOWER && transA == rocblas_operation_none)
        {
            int i = 0;
            for(int idx = 0; idx < num_step_sizes; idx++)
            {
                const int step_size = step_sizes[idx];
                for(; i + (step_size - 1) <= maxColA; i += step_size)
                {
                    // Subtract previously solved parts
                    for(int j = 0; j < step_size; j++)
                        resB[j] = sB[tx + NB * (i + j)];

                    for(int j1 = 0; j1 < i; j1++)
                    {
                        rocblas_int col_off = j1 * NB;
                        T           sB_reg  = sB[tx + NB * j1];
                        for(int j2 = 0; j2 < step_size; j2++)
                            resB[j2] -= sB_reg * sA[col_off + i + j2];
                    }

                    for(int j1 = 0; j1 < step_size; j1++)
                    {
                        for(int j2 = 0; j2 < j1; j2++)
                            resB[j1] -= resB[j2] * sA[(i + j2) * NB + (i + j1)];

                        resB[j1] *= sA[(i + j1) * NB + (i + j1)];
                        sB[tx + NB * (i + j1)] = resB[j1];
                    }
                }
                if(i > maxColA)
                    break;
            }
        }
        else if(!LOWER && transA == rocblas_operation_none)
        {
            int i = maxColA;
            for(int idx = 0; idx < num_step_sizes; idx++)
            {
                const int step_size = step_sizes[idx];
                for(; i >= (step_size - 1); i -= step_size)
                {
                    for(int j = 0; j < step_size; j++)
                        resB[j] = sB[tx + NB * (i - j)];

                    for(int j1 = maxColA; j1 > i; j1--)
                    {
                        rocblas_int col_off = j1;
                        T           sB_reg  = sB[tx + NB * j1];
                        for(int j2 = 0; j2 < step_size; j2++)
                            resB[j2] -= sB_reg * sA[(col_off * NB + (i - j2))];
                    }

                    for(int j1 = 0; j1 < step_size; j1++)
                    {
                        for(int j2 = 0; j2 < j1; j2++)
                            resB[j1] -= resB[j2] * sA[(i - j1) + NB * (i - j2)];

                        resB[j1] *= sA[(i - j1) + NB * (i - j1)];
                        sB[tx + NB * (i - j1)] = resB[j1];
                    }
                }
                if(i < 0)
                    break;
            }
        }
        else if(LOWER)
        {
            int i = maxColA;
            for(int idx = 0; idx < num_step_sizes; idx++)
            {
                const int step_size = step_sizes[idx];
                for(; i >= (step_size - 1); i -= step_size)
                {
                    for(int j = 0; j < step_size; j++)
                        resB[j] = sB[tx + NB * (i - j)];

                    for(int j1 = maxColA; j1 > i; j1--)
                    {
                        rocblas_int col_off = j1;
                        T           sB_reg  = sB[tx + NB * j1];
                        for(int j2 = 0; j2 < step_size; j2++)
                            resB[j2] -= sB_reg * sA[(i - j2) * NB + j1];
                    }

                    for(int j1 = 0; j1 < step_size; j1++)
                    {
                        for(int j2 = 0; j2 < j1; j2++)
                            resB[j1] -= resB[j2] * sA[(i - j2) + NB * (i - j1)];

                        resB[j1] *= sA[(i - j1) + NB * (i - j1)];
                        sB[tx + NB * (i - j1)] = resB[j1];
                    }
                }
                if(i < 0)
                    break;
            }
        }
        else if(!LOWER)
        {
            int i = 0;
            for(int idx = 0; idx < num_step_sizes; idx++)
            {
                const int step_size = step_sizes[idx];
                for(; i + (step_size - 1) <= maxColA; i += step_size)
                {
                    // Subtract previously solved parts
                    for(int j = 0; j < step_size; j++)
                        resB[j] = sB[tx + NB * (i + j)];

                    for(int j1 = 0; j1 < i; j1++)
                    {
                        T sB_reg = sB[tx + NB * j1];
                        for(int j2 = 0; j2 < step_size; j2++)
                            resB[j2] -= sB_reg * sA[(i + j2) * NB + j1];
                    }

                    for(int j1 = 0; j1 < step_size; j1++)
                    {
                        for(int j2 = 0; j2 < j1; j2++)
                            resB[j1] -= resB[j2] * sA[(i + j1) * NB + (i + j2)];

                        resB[j1] *= sA[(i + j1) * NB + (i + j1)];
                        sB[tx + NB * (i + j1)] = resB[j1];
                    }
                }
                if(i > maxColA)
                    break;
            }
        }

        __syncthreads();

        // Save shared memory back into B
        if(tx < maxColB)
        {
            for(int i = 0; i <= maxColA; i++)
                B[i + tx * size_t(ldb)] = sB[i * NB + tx];
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/*
 *  Uses a substitution method to solve a small problem XA = B. This version uses
 *  less shared memory for double complex types (currently this means not using
 *  shared memory for A).
 */
template <typename T, typename SCAL, typename ATYPE, typename BTYPE, const int NB>
ROCBLAS_KERNEL(NB)
rocblas_trsm_small_64_left_device(rocblas_fill      uplo,
                                  rocblas_operation transA,
                                  rocblas_diagonal  diag,
                                  int               m,
                                  int               n,
                                  SCAL              alpha_dev_host,
                                  ATYPE             Aa,
                                  rocblas_stride    offset_A,
                                  int               lda,
                                  rocblas_stride    stride_A,
                                  BTYPE             Ba,
                                  rocblas_stride    offset_B,
                                  int               ldb,
                                  rocblas_stride    stride_B,
                                  int               batch_count)
{
    auto alpha = load_scalar(alpha_dev_host);

    uint32_t batchid = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batchid < batch_count; batchid += c_YZ_grid_launch_limit)
    {
#endif

        auto A = load_ptr_batch(Aa, batchid, offset_A, stride_A);
        auto B = load_ptr_batch(Ba, batchid, offset_B, stride_B);

        bool      LOWER = uplo == rocblas_fill_lower;
        bool      CONJ  = transA == rocblas_operation_conjugate_transpose;
        const int tx    = threadIdx.x;
        const int bx    = blockIdx.x;

        // max A column to read from
        int maxColA = NB - 1 > m - 1 ? m - 1 : NB - 1;
        // NB columns, unless last block, then do leftover
        const int maxColB = (bx < gridDim.x - 1) ? NB : n - bx * NB;

        // offset B into correct block column
        B += bx * NB * size_t(ldb);

        // shared B
        __shared__ T sB[NB * NB];
        if(tx <= maxColA)
        {
            // Load B into sB and multiply by alpha
            for(int i = 0; i < maxColB; i++)
                sB[i * NB + tx] = alpha * B[i * size_t(ldb) + tx];
        }
        __syncthreads();

        // Solve for B in shared memory
        if(LOWER && transA == rocblas_operation_none)
        {
            // Note: I didn't find that using 4 registers (as in the above function) to be noticeably faster in this version
            for(int i = 0; i <= maxColA; i++)
            {
                // Subtract previously solved parts
                for(int j = 0; j < i; j++)
                {
                    T valA = A[j * size_t(lda) + i];
                    sB[tx * NB + i] -= sB[tx * NB + j] * valA;
                }
                if(diag != rocblas_diagonal_unit)
                    sB[tx * NB + i] /= A[i * size_t(lda) + i];
            }
        }
        else if(!LOWER && transA == rocblas_operation_none)
        {
            for(int i = maxColA; i >= 0; i--)
            {
                T temp_reg_B = sB[tx * NB + i];
                for(int j = maxColA; j > i; j--)
                {
                    T valA = A[j * size_t(lda) + i];
                    temp_reg_B -= sB[tx * NB + j] * valA;
                }
                sB[tx * NB + i] = temp_reg_B;
                if(diag != rocblas_diagonal_unit)
                    sB[tx * NB + i] /= A[i * size_t(lda) + i];
            }
        }
        else if(LOWER)
        {
            for(int i = maxColA; i >= 0; i--)
            {
                T temp_reg_B = sB[tx * NB + i];
                for(int j = maxColA; j > i; j--)
                {
                    T valA = (CONJ) ? conj(A[i * size_t(lda) + j]) : A[i * size_t(lda) + j];
                    temp_reg_B -= sB[tx * NB + j] * valA;
                }
                sB[tx * NB + i] = temp_reg_B;
                if(diag != rocblas_diagonal_unit)
                    sB[tx * NB + i]
                        /= (CONJ) ? conj(A[i * size_t(lda) + i]) : A[i * size_t(lda) + i];
            }
        }
        else if(!LOWER)
        {
            for(int i = 0; i <= maxColA; i++)
            {
                T temp_reg_B = sB[tx * NB + i];
                for(int j = 0; j < i; j++)
                {
                    T valA = (CONJ) ? conj(A[i * size_t(lda) + j]) : A[i * size_t(lda) + j];
                    temp_reg_B -= sB[tx * NB + j] * valA;
                }
                sB[tx * NB + i] = temp_reg_B;
                if(diag != rocblas_diagonal_unit)
                    sB[tx * NB + i]
                        /= (CONJ) ? conj(A[i * size_t(lda) + i]) : A[i * size_t(lda) + i];
            }
        }

        __syncthreads();

        // Save shared memory back into B
        if(tx < m)
        {
            for(int i = 0; i < maxColB; i++)
                B[i * size_t(ldb) + tx] = sB[i * NB + tx];
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/* T = float, double, etc.
 * SCAL = T* or T
 * ATYPE = const T* or const T* const *
 * BTYPE = T* or T* const *
 *
 * Sets kernel parameters and launches the appropriate substitution kernel to solve
 * a small trsm problem.
 */
template <typename T,
          typename SCAL,
          typename ATYPE,
          typename BTYPE,
          const int NB,
          const int STEP_SIZE>
rocblas_status rocblas_trsm_small(rocblas_handle    handle,
                                  rocblas_side      side,
                                  rocblas_fill      uplo,
                                  rocblas_operation transA,
                                  rocblas_diagonal  diag,
                                  rocblas_int       m,
                                  rocblas_int       n,
                                  SCAL              alpha,
                                  ATYPE             dA,
                                  rocblas_stride    offset_A,
                                  rocblas_int       lda,
                                  rocblas_stride    stride_A,
                                  BTYPE             dB,
                                  rocblas_stride    offset_B,
                                  rocblas_int       ldb,
                                  rocblas_stride    stride_B,
                                  rocblas_int       batch_count)
{
    hipStream_t rocblas_stream = handle->get_stream();
    int         batches        = handle->getBatchGridDim((int)batch_count);

    // threadIdx.x = NB >= m
    dim3 threads(NB);

    // blockIdx.x = divide B's columns into NB sized blocks
    // blockIdx.y = batch_count
#define TRSM_SMALL_KERNEL_PARAM                                                           \
    grid, threads, 0, rocblas_stream, uplo, transA, diag, m, n, alpha, dA, offset_A, lda, \
        stride_A, dB, offset_B, ldb, stride_B, batch_count
    if(side == rocblas_side_left)
    {
        dim3 grid((n - 1) / NB + 1, 1, batches);
        if(transA == rocblas_operation_none)
        {
            constexpr bool CONJ = false;
            if(uplo == rocblas_fill_upper)
            {
                constexpr bool LOWER = false;
                if(n > 2 * NB && m > 12)
                    ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                               (rocblas_trsm_small_left_device_sharedB<NB,
                                                                                       STEP_SIZE,
                                                                                       LOWER,
                                                                                       T,
                                                                                       SCAL,
                                                                                       ATYPE,
                                                                                       BTYPE>),
                                               TRSM_SMALL_KERNEL_PARAM);
                else
                    ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                               (rocblas_trsm_small_left_device<NB,
                                                                               STEP_SIZE,
                                                                               LOWER,
                                                                               T,
                                                                               SCAL,
                                                                               ATYPE,
                                                                               BTYPE>),
                                               TRSM_SMALL_KERNEL_PARAM);
            }
            else
            {
                constexpr bool LOWER = true;
                if(n > 2 * NB && m > 16)
                    ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                               (rocblas_trsm_small_left_device_sharedB<NB,
                                                                                       STEP_SIZE,
                                                                                       LOWER,
                                                                                       T,
                                                                                       SCAL,
                                                                                       ATYPE,
                                                                                       BTYPE>),
                                               TRSM_SMALL_KERNEL_PARAM);
                else
                    ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                               (rocblas_trsm_small_left_device<NB,
                                                                               STEP_SIZE,
                                                                               LOWER,
                                                                               T,
                                                                               SCAL,
                                                                               ATYPE,
                                                                               BTYPE>),
                                               TRSM_SMALL_KERNEL_PARAM);
            }
        }
        else if(transA == rocblas_operation_transpose
                || transA == rocblas_operation_conjugate_transpose)
        {
            if(uplo == rocblas_fill_upper)
            {
                constexpr bool LOWER = false;
                if(n > 2 * NB && m > 16)
                    ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                               (rocblas_trsm_small_left_device_sharedB<NB,
                                                                                       STEP_SIZE,
                                                                                       LOWER,
                                                                                       T,
                                                                                       SCAL,
                                                                                       ATYPE,
                                                                                       BTYPE>),
                                               TRSM_SMALL_KERNEL_PARAM);
                else
                    ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                               (rocblas_trsm_small_left_device<NB,
                                                                               STEP_SIZE,
                                                                               LOWER,
                                                                               T,
                                                                               SCAL,
                                                                               ATYPE,
                                                                               BTYPE>),
                                               TRSM_SMALL_KERNEL_PARAM);
            }
            else
            {
                constexpr bool LOWER = true;
                if(n > 2 * NB && m > 8)
                    ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                               (rocblas_trsm_small_left_device_sharedB<NB,
                                                                                       STEP_SIZE,
                                                                                       LOWER,
                                                                                       T,
                                                                                       SCAL,
                                                                                       ATYPE,
                                                                                       BTYPE>),
                                               TRSM_SMALL_KERNEL_PARAM);
                else
                    ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                               (rocblas_trsm_small_left_device<NB,
                                                                               STEP_SIZE,
                                                                               LOWER,
                                                                               T,
                                                                               SCAL,
                                                                               ATYPE,
                                                                               BTYPE>),
                                               TRSM_SMALL_KERNEL_PARAM);
            }
        }
    }
    else
    {
        dim3 grid((m - 1) / NB + 1, 1, batches);
        ROCBLAS_LAUNCH_KERNEL_GRID(grid,
                                   (rocblas_trsm_small_right_device<T, SCAL, ATYPE, BTYPE, NB>),
                                   TRSM_SMALL_KERNEL_PARAM);
    }
#undef TRSM_SMALL_KERNEL_PARAM

    return rocblas_status_success;
}

template <typename T,
          typename SCAL,
          typename ATYPE,
          typename BTYPE,
          bool TRANSA,
          bool TRANSB,
          bool UNIT>
ROCBLAS_KERNEL_NO_BOUNDS rocblas_trsm_block_backward_substitution(rocblas_operation transA,
                                                                  int64_t           m,
                                                                  int64_t           n,
                                                                  SCAL              alpha_dev_host,
                                                                  ATYPE             Aa,
                                                                  rocblas_stride    offset_A,
                                                                  int64_t           lda,
                                                                  rocblas_stride    stride_A,
                                                                  BTYPE             Ba,
                                                                  rocblas_stride    offset_B,
                                                                  int64_t           ldb,
                                                                  rocblas_stride    stride_B,
                                                                  int               batch_count,
                                                                  const bool shared_mem_A = false)
{
    const bool CONJ  = transA == rocblas_operation_conjugate_transpose;
    auto       alpha = load_scalar(alpha_dev_host);

    uint32_t batchid = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batchid < batch_count; batchid += c_YZ_grid_launch_limit)
    {
#endif
        auto A = load_ptr_batch(Aa, batchid, offset_A, stride_A);
        auto B = load_ptr_batch(Ba, batchid, offset_B, stride_B);

        int64_t       lda_norm  = TRANSA ? lda : 1;
        int64_t       lda_trans = TRANSA ? 1 : lda;
        const int64_t ldb_norm  = TRANSB ? ldb : 1;
        const int64_t ldb_trans = TRANSB ? 1 : ldb;

        const int     tx   = threadIdx.x;
        const int     ty   = threadIdx.y;
        const int64_t offY = blockIdx.y * blockDim.y + threadIdx.y;

        // passing as extern shared memory to avoid templating NB size
        // casting fails when going from double -> double complex and otherwise
        extern __shared__ rocblas_double_complex smem[];
        T*                                       sB = reinterpret_cast<T*>(smem);
        T*                                       sA;
        const T*                                 A_shared_or_global = A;

        if(shared_mem_A)
        {
            sA = reinterpret_cast<T*>(smem) + blockDim.y;
            for(int rowOff = 0; rowOff < m; rowOff += blockDim.y)
            {
                int aRow = tx;
                int aCol = ty + (rowOff);
                if(aRow < m && aCol < m && aRow < aCol)
                    sA[aCol * blockDim.x + aRow] = A[aCol * lda_norm + aRow * lda_trans];
                else if(aRow == aCol && aRow < m && !UNIT)
                    sA[aCol * blockDim.x + aRow] = 1.0 / A[aCol * lda_norm + aRow * lda_trans];
            }
            A_shared_or_global = sA;
            lda_norm           = blockDim.x;
            lda_trans          = 1;
        }

        if(offY < n && tx < m)
        {
            T valB = alpha * B[offY * size_t(ldb_norm) + tx * size_t(ldb_trans)];
            for(int64_t i = m - 1; i > 0; i--)
            {
                // tx is row of B, ty is col of B
                // tx is row of A, i is col of A
                __syncthreads();
                if(tx == i)
                {
                    // solve cur row
                    if(!UNIT)
                    {
                        auto valA = (A_shared_or_global[tx * lda_norm + tx * lda_trans]);
                        if(!shared_mem_A)
                            valA = 1.0 / valA;
                        valB *= valA;
                    }

                    sB[ty] = valB;
                }

                __syncthreads();

                if(tx < i)
                    valB -= (CONJ ? conj(
                                 A_shared_or_global[i * size_t(lda_norm) + tx * size_t(lda_trans)])
                                  : A_shared_or_global[i * size_t(lda_norm)
                                                       + tx * size_t(lda_trans)])
                            * sB[ty];
            }

            if(!UNIT && tx == 0)
            {
                auto valA = (A_shared_or_global[tx * lda_norm + tx * lda_trans]);
                if(!shared_mem_A)
                    valA = 1.0 / valA;
                valB *= valA;
            }

            // store back to mem
            B[offY * size_t(ldb_norm) + tx * size_t(ldb_trans)] = valB;
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename T,
          typename SCAL,
          typename ATYPE,
          typename BTYPE,
          bool TRANSA,
          bool TRANSB,
          bool UNIT>
ROCBLAS_KERNEL_NO_BOUNDS rocblas_trsm_block_forward_substitution(rocblas_operation transA,
                                                                 int64_t           m,
                                                                 int64_t           n,
                                                                 SCAL              alpha_dev_host,
                                                                 ATYPE             Aa,
                                                                 rocblas_stride    offset_A,
                                                                 int64_t           lda,
                                                                 rocblas_stride    stride_A,
                                                                 BTYPE             Ba,
                                                                 rocblas_stride    offset_B,
                                                                 int64_t           ldb,
                                                                 rocblas_stride    stride_B,
                                                                 int               batch_count,
                                                                 const bool shared_mem_A = false)
{
    const bool CONJ  = transA == rocblas_operation_conjugate_transpose;
    auto       alpha = load_scalar(alpha_dev_host);

    int64_t       lda_norm  = TRANSA ? 1 : lda;
    int64_t       lda_trans = TRANSA ? lda : 1;
    const int64_t ldb_norm  = TRANSB ? 1 : ldb;
    const int64_t ldb_trans = TRANSB ? ldb : 1;

    uint32_t batchid = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batchid < batch_count; batchid += c_YZ_grid_launch_limit)
    {
#endif
        auto A = load_ptr_batch(Aa, batchid, offset_A, stride_A);
        auto B = load_ptr_batch(Ba, batchid, offset_B, stride_B);

        const int     tx   = threadIdx.x;
        const int     ty   = threadIdx.y;
        const int64_t offY = blockIdx.y * blockDim.y + threadIdx.y;

        extern __shared__ rocblas_double_complex smem[];
        T*                                       sB = reinterpret_cast<T*>(smem);
        T*                                       sA;
        const T*                                 A_shared_or_global = A;

        if(shared_mem_A)
        {
            sA = reinterpret_cast<T*>(smem) + blockDim.y;
            for(int rowOff = 0; rowOff < m; rowOff += blockDim.y)
            {
                int aRow = tx;
                int aCol = ty + (rowOff);
                if(aRow < m && aCol < m && aRow > aCol)
                    sA[aCol * blockDim.x + aRow] = A[aCol * lda_norm + aRow * lda_trans];
                else if(aRow == aCol && aRow < m && !UNIT)
                    sA[aCol * blockDim.x + aRow] = 1.0 / A[aCol * lda_norm + aRow * lda_trans];
            }
            A_shared_or_global = sA;
            lda_norm           = blockDim.x;
            lda_trans          = 1;
        }

        if(offY < n && tx < m)
        {
            T   valB  = alpha * B[offY * size_t(ldb_norm) + tx * size_t(ldb_trans)];
            int sAoff = 1;
            for(int64_t i = 0; i < m - 1; i++)
            {
                // tx is row of B, ty is col of B
                // tx is row of A, i is col of A
                __syncthreads();
                if(tx == i)
                {
                    // solve cur row
                    if(!UNIT)
                    {
                        auto valA = (A_shared_or_global[tx * lda_norm + tx * lda_trans]);
                        if(!shared_mem_A)
                            valA = 1.0 / valA;
                        valB *= valA;
                    }
                    sB[ty] = valB;
                }
                __syncthreads();

                if(tx > i)
                    valB -= (CONJ ? conj(A_shared_or_global[i * lda_norm + tx * lda_trans])
                                  : A_shared_or_global[i * lda_norm + tx * lda_trans])
                            * sB[ty];
            }

            if(!UNIT && tx == m - 1)
            {
                auto valA = (A_shared_or_global[tx * lda_norm + tx * lda_trans]);
                if(!shared_mem_A)
                    valA = 1.0 / valA;
                valB *= valA;
            }

            // store back to mem
            B[offY * size_t(ldb_norm) + tx * size_t(ldb_trans)] = valB;
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename T,
          typename SCAL,
          typename ATYPE,
          typename BTYPE,
          bool UPPER,
          bool TRANSA,
          bool UNIT,
          bool BATCHED>
rocblas_status rocblas_trsm_small_substitution(rocblas_handle    handle,
                                               rocblas_side      side,
                                               rocblas_operation transA,
                                               int64_t           m,
                                               int64_t           n,
                                               SCAL              alpha,
                                               ATYPE             dA,
                                               rocblas_stride    offset_A,
                                               int64_t           lda,
                                               rocblas_stride    stride_A,
                                               BTYPE             dB,
                                               rocblas_stride    offset_B,
                                               int64_t           ldb,
                                               rocblas_stride    stride_B,
                                               rocblas_int       batch_count,
                                               rocblas_int       blk_size)
{
    bool LEFT       = side == rocblas_side_left;
    bool CONJ       = transA == rocblas_operation_conjugate_transpose;
    bool use_64_bit = m > c_i32_max || n > c_i32_max || lda > c_i32_max || ldb > c_i32_max;

    constexpr bool TRANSB = (!UPPER && TRANSA) || (UPPER && !TRANSA);
    const int64_t  k      = LEFT ? n : m;
    const int64_t  k2     = LEFT ? m : n;
    rocblas_int    NBX    = blk_size;

    rocblas_int blocks = (k - 1) / (1024 / NBX) + 1;

    int batches = handle->getBatchGridDim((int)batch_count);

    // kernel params for trsm substitution/solve portion
    dim3 grid(1, blocks, batches); // TODO use x grid
    dim3 threads(NBX, 1024 / NBX);

    // for updating B with solved portions
    T       negative_one = -1;
    T       one          = 1;
    int64_t j            = 0;
    size_t  offA_sub, offB_sub;
    size_t  smem_size;

    // Different kernels for forward substitution vs. backward substitution
    bool FORWARD_SUB = (LEFT && ((!TRANSA && !UPPER) || (TRANSA && UPPER)))
                       || (!LEFT && ((TRANSA && !UPPER) || (!TRANSA && UPPER)));
    for(j = 0; j < k2 - NBX; j += NBX)
    {
        const int64_t j_next = j + NBX;

        size_t offA_gemm = LEFT ? (!TRANSA ? j * size_t(lda) + j_next : j + j_next * size_t(lda))
                                : (!TRANSA ? j + j_next * size_t(lda) : j * size_t(lda) + j_next);
        size_t offB_gemm = LEFT ? j : j * size_t(ldb);
        size_t offC_gemm = LEFT ? j_next : j_next * size_t(ldb);
        smem_size        = ((1024 / NBX) + (NBX * NBX)) * sizeof(T);

        int  max_mem     = handle->getMaxSharedMemPerBlock();
        bool sharedA_mem = (max_mem >= smem_size);
        if(!sharedA_mem)
        {
            // fallback to no shared mem for A matrix
            smem_size = (1024 / NBX) * sizeof(T);
        }

        // 1. call trsm subtitution/solve
        if(FORWARD_SUB)
        {
            offA_sub = j * size_t(lda) + j;
            offB_sub = LEFT ? j : j * size_t(ldb);

            ROCBLAS_LAUNCH_KERNEL((rocblas_trsm_block_forward_substitution<T,
                                                                           SCAL,
                                                                           ATYPE,
                                                                           BTYPE,
                                                                           UPPER,
                                                                           TRANSB,
                                                                           UNIT>),
                                  grid,
                                  threads,
                                  smem_size,
                                  handle->get_stream(),
                                  transA,
                                  NBX,
                                  k,
                                  j == 0 ? alpha : 1,
                                  dA,
                                  offset_A + offA_sub,
                                  lda,
                                  stride_A,
                                  dB,
                                  offset_B + offB_sub,
                                  ldb,
                                  stride_B,
                                  batch_count,
                                  sharedA_mem);
        }
        else
        {
            offA_sub  = LEFT ? (m - j_next) * size_t(lda) + (m - j_next)
                             : (n - j_next) * size_t(lda) + (n - j_next);
            offB_sub  = LEFT ? m - j_next : (n - j_next) * size_t(ldb);
            offA_gemm = LEFT ? (!TRANSA ? (m - j_next) * size_t(lda) : m - j_next)
                             : (!TRANSA ? n - j_next : (n - j_next) * size_t(lda));
            offB_gemm = LEFT ? m - j_next : (n - j_next) * size_t(ldb);
            offC_gemm = 0;
            ROCBLAS_LAUNCH_KERNEL((rocblas_trsm_block_backward_substitution<T,
                                                                            SCAL,
                                                                            ATYPE,
                                                                            BTYPE,
                                                                            UPPER,
                                                                            TRANSB,
                                                                            UNIT>),
                                  grid,
                                  threads,
                                  smem_size,
                                  handle->get_stream(),
                                  transA,
                                  NBX,
                                  k,
                                  j == 0 ? alpha : 1,
                                  dA,
                                  offset_A + offA_sub,
                                  lda,
                                  stride_A,
                                  dB,
                                  offset_B + offB_sub,
                                  ldb,
                                  stride_B,
                                  batch_count,
                                  sharedA_mem);
        }

        // 2. call gemm to update B matrix
        if(use_64_bit)
        {
            RETURN_IF_ROCBLAS_ERROR(rocblas_internal_gemm_64<BATCHED>(
                handle,
                LEFT ? transA : rocblas_operation_none,
                LEFT ? rocblas_operation_none : transA,
                LEFT ? m - j_next : m,
                LEFT ? n : n - j_next,
                NBX,
                &negative_one,
                LEFT ? dA : (ATYPE)dB,
                LEFT ? offset_A + offA_gemm : offset_B + offB_gemm,
                LEFT ? lda : ldb,
                LEFT ? stride_A : stride_B,
                LEFT ? (ATYPE)dB : dA,
                LEFT ? offset_B + offB_gemm : offset_A + offA_gemm,
                LEFT ? ldb : lda,
                LEFT ? stride_B : stride_A,
                j == 0 ? &alpha : &one,
                dB,
                offset_B + offC_gemm,
                ldb,
                stride_B,
                batch_count));
        }
        else
        {
            RETURN_IF_ROCBLAS_ERROR(
                rocblas_internal_gemm<BATCHED>(handle,
                                               LEFT ? transA : rocblas_operation_none,
                                               LEFT ? rocblas_operation_none : transA,
                                               LEFT ? m - j_next : m,
                                               LEFT ? n : n - j_next,
                                               NBX,
                                               &negative_one,
                                               LEFT ? dA : (ATYPE)dB,
                                               LEFT ? offset_A + offA_gemm : offset_B + offB_gemm,
                                               LEFT ? lda : ldb,
                                               LEFT ? stride_A : stride_B,
                                               LEFT ? (ATYPE)dB : dA,
                                               LEFT ? offset_B + offB_gemm : offset_A + offA_gemm,
                                               LEFT ? ldb : lda,
                                               LEFT ? stride_B : stride_A,
                                               j == 0 ? &alpha : &one,
                                               dB,
                                               offset_B + offC_gemm,
                                               ldb,
                                               stride_B,
                                               batch_count));
        }
    }

    // solve last diagonal
    int64_t leftover = k2 - j;
    blocks           = (k - 1) / (1024 / leftover) + 1;
    grid             = dim3(1, blocks, batches);
    threads          = dim3(leftover, 1024 / leftover);
    smem_size        = ((1024 / leftover) + (leftover * leftover)) * sizeof(T);

    int  max_mem     = handle->getMaxSharedMemPerBlock();
    bool sharedA_mem = (max_mem >= smem_size);
    if(!sharedA_mem)
    {
        // fallback to no shared mem for A matrix
        smem_size = (1024 / leftover) * sizeof(T);
    }
    if(FORWARD_SUB)
    {
        offA_sub = j * size_t(lda) + j;
        offB_sub = LEFT ? j : j * size_t(ldb);

        ROCBLAS_LAUNCH_KERNEL(
            (rocblas_trsm_block_forward_substitution<T, SCAL, ATYPE, BTYPE, UPPER, TRANSB, UNIT>),
            grid,
            threads,
            smem_size,
            handle->get_stream(),
            transA,
            k2 - j,
            k,
            j == 0 ? alpha : 1,
            dA,
            offset_A + offA_sub,
            lda,
            stride_A,
            dB,
            offset_B + offB_sub,
            ldb,
            stride_B,
            batch_count,
            sharedA_mem);
    }
    else
    {
        offA_sub = LEFT ? 0 : 0;
        offB_sub = LEFT ? 0 : 0;
        ROCBLAS_LAUNCH_KERNEL(
            (rocblas_trsm_block_backward_substitution<T, SCAL, ATYPE, BTYPE, UPPER, TRANSB, UNIT>),
            grid,
            threads,
            smem_size,
            handle->get_stream(),
            transA,
            k2 - j,
            k,
            j == 0 ? alpha : 1,
            dA,
            offset_A + offA_sub,
            lda,
            stride_A,
            dB,
            offset_B + offB_sub,
            ldb,
            stride_B,
            batch_count,
            sharedA_mem);
    }

    return rocblas_status_success;
}

template <bool BATCHED, typename T, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_trsm_small_substitution_launcher(rocblas_handle    handle,
                                                                 rocblas_side      side,
                                                                 rocblas_fill      uplo,
                                                                 rocblas_operation transA,
                                                                 rocblas_diagonal  diag,
                                                                 int64_t           m,
                                                                 int64_t           n,
                                                                 T                 alpha_h,
                                                                 TConstPtr         A,
                                                                 rocblas_stride    offset_A,
                                                                 int64_t           lda,
                                                                 rocblas_stride    stride_A,
                                                                 TPtr              B,
                                                                 rocblas_stride    offset_B,
                                                                 int64_t           ldb,
                                                                 rocblas_stride    stride_B,
                                                                 int               batch_count,
                                                                 int               blksize)
{
    rocblas_status status = rocblas_status_success;

#define TRSM_SUBSTITUTION_LAUNCH(T, UPPER, TRANS, DIAG, BATCHED)                                  \
    status = rocblas_trsm_small_substitution<T, T, TConstPtr, TPtr, UPPER, TRANS, DIAG, BATCHED>( \
        handle,                                                                                   \
        side,                                                                                     \
        transA,                                                                                   \
        m,                                                                                        \
        n,                                                                                        \
        alpha_h,                                                                                  \
        A,                                                                                        \
        offset_A,                                                                                 \
        lda,                                                                                      \
        stride_A,                                                                                 \
        B,                                                                                        \
        offset_B,                                                                                 \
        ldb,                                                                                      \
        stride_B,                                                                                 \
        batch_count,                                                                              \
        blksize)

    // A mess of if/else statements to get the template parameters
    if(uplo == rocblas_fill_lower && transA == rocblas_operation_none
       && diag == rocblas_diagonal_non_unit)
        TRSM_SUBSTITUTION_LAUNCH(T, false, false, false, BATCHED);
    else if(uplo == rocblas_fill_lower && transA == rocblas_operation_none
            && diag == rocblas_diagonal_unit)
        TRSM_SUBSTITUTION_LAUNCH(T, false, false, true, BATCHED);
    else if(uplo == rocblas_fill_lower && transA != rocblas_operation_none
            && diag == rocblas_diagonal_non_unit)
        TRSM_SUBSTITUTION_LAUNCH(T, false, true, false, BATCHED);
    else if(uplo == rocblas_fill_lower && transA != rocblas_operation_none
            && diag == rocblas_diagonal_unit)
        TRSM_SUBSTITUTION_LAUNCH(T, false, true, true, BATCHED);
    else if(uplo == rocblas_fill_upper && transA == rocblas_operation_none
            && diag == rocblas_diagonal_non_unit)
        TRSM_SUBSTITUTION_LAUNCH(T, true, false, false, BATCHED);
    else if(uplo == rocblas_fill_upper && transA == rocblas_operation_none
            && diag == rocblas_diagonal_unit)
        TRSM_SUBSTITUTION_LAUNCH(T, true, false, true, BATCHED);
    else if(uplo == rocblas_fill_upper && transA != rocblas_operation_none
            && diag == rocblas_diagonal_non_unit)
        TRSM_SUBSTITUTION_LAUNCH(T, true, true, false, BATCHED);
    else if(uplo == rocblas_fill_upper && transA != rocblas_operation_none
            && diag == rocblas_diagonal_unit)
        TRSM_SUBSTITUTION_LAUNCH(T, true, true, true, BATCHED);
    else
        return rocblas_status_internal_error;

#undef TRSM_SUBSTITUTION_LAUNCH

    return status;
}

//////////////////////////////
//////////////////////////////
//////////////////////////////
template <rocblas_int BLOCK, rocblas_int DIM_X, bool BATCHED, typename T, typename U, typename V>
rocblas_status rocblas_internal_trsm_launcher(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              U                 A,
                                              rocblas_stride    offset_A,
                                              rocblas_int       lda,
                                              rocblas_stride    stride_A,
                                              V                 B,
                                              rocblas_stride    offset_B,
                                              rocblas_int       ldb,
                                              rocblas_stride    stride_B,
                                              rocblas_int       batch_count,
                                              bool              optimal_mem,
                                              void*             w_x_temp,
                                              void*             w_x_temparr,
                                              void*             invA,
                                              void*             invAarr,
                                              U                 supplied_invA,
                                              rocblas_int       supplied_invA_size,
                                              rocblas_stride    offset_invA,
                                              rocblas_stride    stride_invA)
{
    if(batch_count == 0)
        return rocblas_status_success;

    if(!rocblas_is_complex<T> && transA == rocblas_operation_conjugate_transpose)
        transA = rocblas_operation_transpose;

    rocblas_int k  = side == rocblas_side_left ? m : n;
    rocblas_int k1 = side == rocblas_side_left ? n : m;

    if(n == 1 && side == rocblas_side_left)
    {
        // left
        // B is essentially a vector (x, in trsv). Don't need ldb, can use 1 for incx.
        rocblas_internal_trsv_substitution_template<DIM_X, T>(handle,
                                                              uplo,
                                                              transA,
                                                              diag,
                                                              m,
                                                              A,
                                                              offset_A,
                                                              lda,
                                                              stride_A,
                                                              alpha,
                                                              B,
                                                              offset_B,
                                                              1,
                                                              stride_B,
                                                              batch_count,
                                                              (rocblas_int*)w_x_temp);
    }
    else
    {
        // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        // Get alpha - Check if zero for quick return
        T alpha_h;
        if(saved_pointer_mode == rocblas_pointer_mode_host)
            alpha_h = *alpha;
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost, handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        }
        if(alpha_h == T(0.0))
        {
            return set_block_unit<T>(handle, m, n, B, ldb, stride_B, batch_count, 0.0, offset_B);
        }

        // These small substitution kernels still seem faster than full substitution method below
        bool is_small = (k <= 32) || (m <= 64 && n <= 64);
        if(is_small)
        {
#define ROCBLAS_TRSM_SMALL(DIM_)                         \
    rocblas_trsm_small<T, T, U, V, DIM_, DIM_>(handle,   \
                                               side,     \
                                               uplo,     \
                                               transA,   \
                                               diag,     \
                                               m,        \
                                               n,        \
                                               alpha_h,  \
                                               A,        \
                                               offset_A, \
                                               lda,      \
                                               stride_A, \
                                               B,        \
                                               offset_B, \
                                               ldb,      \
                                               stride_B, \
                                               batch_count)

            if(k <= 4)
                RETURN_IF_ROCBLAS_ERROR(ROCBLAS_TRSM_SMALL(4));
            else if(k <= 8)
                RETURN_IF_ROCBLAS_ERROR(ROCBLAS_TRSM_SMALL(8));
            else if(k <= 12)
                RETURN_IF_ROCBLAS_ERROR(ROCBLAS_TRSM_SMALL(12));
            else if(k <= 16)
                RETURN_IF_ROCBLAS_ERROR(ROCBLAS_TRSM_SMALL(16));
            else if(k <= 20)
                RETURN_IF_ROCBLAS_ERROR(ROCBLAS_TRSM_SMALL(20));
            else if(k <= 24)
                RETURN_IF_ROCBLAS_ERROR(ROCBLAS_TRSM_SMALL(24));
            else if(k <= 28)
                RETURN_IF_ROCBLAS_ERROR(ROCBLAS_TRSM_SMALL(28));
            else if(k <= 32)
                RETURN_IF_ROCBLAS_ERROR(ROCBLAS_TRSM_SMALL(32));
#undef ROCBLAS_TRSM_SMALL
            else if(k <= 64)
            {
                if constexpr(std::is_same<T, rocblas_double_complex>{})
                {
                    // This function sets kernel parameters and launches the appropriate kernel (with less shared memory) for a double complex trsm problem.

                    int batches = handle->getBatchGridDim((int)batch_count);

                    // threadIdx.x = NB >= m
                    dim3 threads(64);

                    // blockIdx.x = divide B's columns into NB sized blocks
                    // blockIdx.y = batch_count
                    if(side == rocblas_side_left)
                    {
                        dim3 grid((n + 64 - 1) / 64, 1, batches);
                        ROCBLAS_LAUNCH_KERNEL((rocblas_trsm_small_64_left_device<T, T, U, V, 64>),
                                              grid,
                                              threads,
                                              0,
                                              handle->get_stream(),
                                              uplo,
                                              transA,
                                              diag,
                                              m,
                                              n,
                                              alpha_h,
                                              A,
                                              offset_A,
                                              lda,
                                              stride_A,
                                              B,
                                              offset_B,
                                              ldb,
                                              stride_B,
                                              batch_count);
                    }
                    else
                    {
                        dim3 grid((m + 64 - 1) / 64, 1, batches);
                        ROCBLAS_LAUNCH_KERNEL((rocblas_trsm_small_64_right_device<T, T, U, V, 64>),
                                              grid,
                                              threads,
                                              0,
                                              handle->get_stream(),
                                              uplo,
                                              transA,
                                              diag,
                                              m,
                                              n,
                                              alpha_h,
                                              A,
                                              offset_A,
                                              lda,
                                              stride_A,
                                              B,
                                              offset_B,
                                              ldb,
                                              stride_B,
                                              batch_count);
                    }
                }
                else
                {
                    // This function is for all other cases where we can use the regular substitution kernels.

                    rocblas_trsm_small<T, T, U, V, 64, 32>(handle,
                                                           side,
                                                           uplo,
                                                           transA,
                                                           diag,
                                                           m,
                                                           n,
                                                           alpha_h,
                                                           A,
                                                           offset_A,
                                                           lda,
                                                           stride_A,
                                                           B,
                                                           offset_B,
                                                           ldb,
                                                           stride_B,
                                                           batch_count);
                }
            }
        }
        else
        {
            // --------------------------------------------------------------
            // See if substitution method is optimal, otherwise use inversion
            // --------------------------------------------------------------

            // from rocSOLVER profiling, if we get a blksize of 0,
            // substitution method shouldn't be used
            const bool  LEFT    = rocblas_side_left == side;
            rocblas_int blksize = rocblas_trsm_blksize<BATCHED, T>(LEFT ? m : n, LEFT ? n : m);

            const bool use_sub = rocblas_internal_trsm_use_substitution(side, m, n, batch_count);

            if(use_sub && blksize)
            {
                return rocblas_internal_trsm_small_substitution_launcher<BATCHED>(handle,
                                                                                  side,
                                                                                  uplo,
                                                                                  transA,
                                                                                  diag,
                                                                                  m,
                                                                                  n,
                                                                                  alpha_h,
                                                                                  A,
                                                                                  offset_A,
                                                                                  lda,
                                                                                  stride_A,
                                                                                  B,
                                                                                  offset_B,
                                                                                  ldb,
                                                                                  stride_B,
                                                                                  batch_count,
                                                                                  blksize);
            }

            // perf_status indicates whether optimal performance is obtainable with available memory
            rocblas_status perf_status = rocblas_status_success;

            if(supplied_invA && supplied_invA_size / BLOCK < k)
            {
                perf_status   = rocblas_status_perf_degraded;
                supplied_invA = nullptr;
            }

            rocblas_status status = rocblas_status_success;

            if(supplied_invA)
            {
                invAarr = (void*)(supplied_invA);
                invA    = (void*)(supplied_invA);
            }
            else
            {
                // w_c_temp and w_x_temp can reuse the same device memory
                T* w_c_temp = (T*)w_x_temp;
                stride_invA = BLOCK * k;
                if(BATCHED)
                {
                    // for w_c_temp, we currently can use the same memory from each batch since
                    // trtri_batched is naive (since gemm_batched is naive)
                    RETURN_IF_ROCBLAS_ERROR(setup_batched_array<BLOCK>(
                        handle->get_stream(), (T*)w_c_temp, 0, (T**)w_x_temparr, batch_count));
                    RETURN_IF_ROCBLAS_ERROR(setup_batched_array<BLOCK>(
                        handle->get_stream(), (T*)invA, stride_invA, (T**)invAarr, batch_count));
                }

                status = rocblas_trtri_trsm_template<BLOCK, BATCHED, T>(
                    handle,
                    V(BATCHED ? w_x_temparr : w_c_temp),
                    uplo,
                    diag,
                    k,
                    A,
                    offset_A,
                    lda,
                    stride_A,
                    V(BATCHED ? invAarr : invA),
                    offset_invA,
                    stride_invA,
                    batch_count);

                if(status != rocblas_status_success)
                    return status;
            }

            const bool use_special = trsm_use_special_kernel<BLOCK, BATCHED, T>(
                side, transA, m, n, batch_count, supplied_invA_size);
            size_t B_chunk_size = optimal_mem ? size_t(m) + size_t(n) - size_t(k) : 1;
            size_t x_temp_els   = use_special ? BLOCK * B_chunk_size : size_t(m) * n;
            if(BATCHED)
            {
                RETURN_IF_ROCBLAS_ERROR(setup_batched_array<BLOCK>(
                    handle->get_stream(), (T*)w_x_temp, x_temp_els, (T**)w_x_temparr, batch_count));
            }

#ifdef BUILD_WITH_TENSILE
            if(use_special)
            {
                status = special_trsm_template<BLOCK, BATCHED>(handle,
                                                               side,
                                                               uplo,
                                                               transA,
                                                               diag,
                                                               m,
                                                               n,
                                                               &alpha_h,
                                                               U(A),
                                                               offset_A,
                                                               lda,
                                                               stride_A,
                                                               V(B),
                                                               offset_B,
                                                               ldb,
                                                               stride_B,
                                                               batch_count,
                                                               U(BATCHED ? invAarr : invA),
                                                               offset_invA,
                                                               stride_invA,
                                                               B_chunk_size,
                                                               V(BATCHED ? w_x_temparr : w_x_temp),
                                                               x_temp_els);
            }
            else
#endif
            {
                if(side == rocblas_side_left)
                    status
                        = rocblas_trsm_left<BLOCK, BATCHED, T>(handle,
                                                               uplo,
                                                               transA,
                                                               m,
                                                               n,
                                                               &alpha_h,
                                                               U(A),
                                                               offset_A,
                                                               lda,
                                                               stride_A,
                                                               V(B),
                                                               offset_B,
                                                               ldb,
                                                               stride_B,
                                                               batch_count,
                                                               U(BATCHED ? invAarr : invA),
                                                               offset_invA,
                                                               stride_invA,
                                                               V(BATCHED ? w_x_temparr : w_x_temp),
                                                               x_temp_els);
                else
                    status
                        = rocblas_trsm_right<BLOCK, BATCHED, T>(handle,
                                                                uplo,
                                                                transA,
                                                                m,
                                                                n,
                                                                &alpha_h,
                                                                U(A),
                                                                offset_A,
                                                                lda,
                                                                stride_A,
                                                                V(B),
                                                                offset_B,
                                                                ldb,
                                                                stride_B,
                                                                batch_count,
                                                                U(BATCHED ? invAarr : invA),
                                                                offset_invA,
                                                                stride_invA,
                                                                V(BATCHED ? w_x_temparr : w_x_temp),
                                                                x_temp_els);

                if(status != rocblas_status_success)
                    return status;

                rocblas_copy_block_unit<T>(handle,
                                           m,
                                           n,
                                           U(BATCHED ? w_x_temparr : w_x_temp),
                                           m,
                                           x_temp_els,
                                           V(B),
                                           ldb,
                                           stride_B,
                                           batch_count,
                                           0,
                                           offset_B);
            }

            // If status is successful, return perf_status; else return status
            return status == rocblas_status_success ? perf_status : status;
        }
    }

    return rocblas_status_success;
}

#ifdef INSTANTIATE_TRSM_MEM_TEMPLATE
#error INSTANTIATE_TRSM_MEM_TEMPLATE already defined
#endif

#define INSTANTIATE_TRSM_MEM_TEMPLATE(BATCHED_, T_, U_)                           \
    template rocblas_status rocblas_internal_trsm_template_mem<BATCHED_, T_, U_>( \
        rocblas_handle    handle,                                                 \
        rocblas_side      side,                                                   \
        rocblas_operation transA,                                                 \
        rocblas_int       m,                                                      \
        rocblas_int       n,                                                      \
        rocblas_int       lda,                                                    \
        rocblas_int       ldb,                                                    \
        rocblas_int       batch_count,                                            \
        rocblas_device_malloc_base & w_mem,                                       \
        void*&      w_mem_x_temp,                                                 \
        void*&      w_mem_x_temp_arr,                                             \
        void*&      w_mem_invA,                                                   \
        void*&      w_mem_invA_arr,                                               \
        U_          supplied_invA,                                                \
        rocblas_int supplied_invA_size);
