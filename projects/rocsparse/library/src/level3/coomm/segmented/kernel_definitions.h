/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */
#include "../../coomm_device_segmented.h"
#include "../coomm_common.h"
#include "rocsparse_scalar.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t LOOPS,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void coommnn_segmented_main_kernel(bool    conj_A,
                                           bool    conj_B,
                                           I       M,
                                           I       N,
                                           I       K,
                                           int64_t nnz,
                                           int64_t batch_stride_A,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                           I* __restrict__ row_block_red,
                                           T* __restrict__ val_block_red,
                                           const I* __restrict__ coo_row_ind,
                                           const I* __restrict__ coo_col_ind,
                                           const A* __restrict__ coo_val,
                                           const B* __restrict__ dense_B,
                                           int64_t ldb,
                                           int64_t batch_stride_B,
                                           C* __restrict__ dense_C,
                                           int64_t              ldc,
                                           int64_t              batch_stride_C,
                                           rocsparse_order      order_C,
                                           rocsparse_index_base idx_base,
                                           bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        if(alpha != static_cast<T>(0))
        {
            rocsparse::coommnn_segmented_main_device<BLOCKSIZE, WF_SIZE, LOOPS, TRANSB>(
                conj_A,
                conj_B,
                M,
                N,
                K,
                nnz,
                batch_stride_A,
                alpha,
                row_block_red,
                val_block_red,
                coo_row_ind,
                coo_col_ind,
                coo_val,
                dense_B,
                ldb,
                batch_stride_B,
                dense_C,
                ldc,
                batch_stride_C,
                order_C,
                idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t LOOPS,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void coommnn_segmented_remainder_kernel(bool    conj_A,
                                                bool    conj_B,
                                                I       colB_offset,
                                                I       M,
                                                I       N,
                                                I       K,
                                                int64_t nnz,
                                                int64_t batch_stride_A,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                I* __restrict__ row_block_red,
                                                T* __restrict__ val_block_red,
                                                const I* __restrict__ coo_row_ind,
                                                const I* __restrict__ coo_col_ind,
                                                const A* __restrict__ coo_val,
                                                const B* __restrict__ dense_B,
                                                int64_t ldb,
                                                int64_t batch_stride_B,
                                                C* __restrict__ dense_C,
                                                int64_t              ldc,
                                                int64_t              batch_stride_C,
                                                rocsparse_order      order_C,
                                                rocsparse_index_base idx_base,
                                                bool                 is_host_mode)
    {

        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        if(alpha != static_cast<T>(0))
        {
            rocsparse::coommnn_segmented_remainder_device<BLOCKSIZE, WF_SIZE, LOOPS, TRANSB>(
                conj_A,
                conj_B,
                colB_offset,
                M,
                N,
                K,
                nnz,
                batch_stride_A,
                alpha,
                row_block_red,
                val_block_red,
                coo_row_ind,
                coo_col_ind,
                coo_val,
                dense_B,
                ldb,
                batch_stride_B,
                dense_C,
                ldc,
                batch_stride_C,
                order_C,
                idx_base);
        }
    }
}

#define COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, BLOCKSIZE, WFSIZE, LOOPS, TRANSB) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                              \
        rocsparse::coommnn_segmented_main_kernel<BLOCKSIZE, WFSIZE, LOOPS, TRANSB>(    \
            bool    conj_A,                                                            \
            bool    conj_B,                                                            \
            I       M,                                                                 \
            I       N,                                                                 \
            I       K,                                                                 \
            int64_t nnz,                                                               \
            int64_t batch_stride_A,                                                    \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                             \
            I* __restrict__ row_block_red,                                             \
            T* __restrict__ val_block_red,                                             \
            const I* __restrict__ coo_row_ind,                                         \
            const I* __restrict__ coo_col_ind,                                         \
            const A* __restrict__ coo_val,                                             \
            const B* __restrict__ dense_B,                                             \
            int64_t ldb,                                                               \
            int64_t batch_stride_B,                                                    \
            C* __restrict__ dense_C,                                                   \
            int64_t              ldc,                                                  \
            int64_t              batch_stride_C,                                       \
            rocsparse_order      order_C,                                              \
            rocsparse_index_base idx_base,                                             \
            bool                 is_host_mode);

#define COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, BLOCKSIZE, WFSIZE, LOOPS, TRANSB) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                                   \
        rocsparse::coommnn_segmented_remainder_kernel<BLOCKSIZE, WFSIZE, LOOPS, TRANSB>(    \
            bool    conj_A,                                                                 \
            bool    conj_B,                                                                 \
            I       colB_offset,                                                            \
            I       M,                                                                      \
            I       N,                                                                      \
            I       K,                                                                      \
            int64_t nnz,                                                                    \
            int64_t batch_stride_A,                                                         \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                                  \
            I* __restrict__ row_block_red,                                                  \
            T* __restrict__ val_block_red,                                                  \
            const I* __restrict__ coo_row_ind,                                              \
            const I* __restrict__ coo_col_ind,                                              \
            const A* __restrict__ coo_val,                                                  \
            const B* __restrict__ dense_B,                                                  \
            int64_t ldb,                                                                    \
            int64_t batch_stride_B,                                                         \
            C* __restrict__ dense_C,                                                        \
            int64_t              ldc,                                                       \
            int64_t              batch_stride_C,                                            \
            rocsparse_order      order_C,                                                   \
            rocsparse_index_base idx_base,                                                  \
            bool                 is_host_mode);

#define COOMMNN_SEGMENTED_MAIN_256_1_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, 256, 1, 4, false)
#define COOMMNN_SEGMENTED_MAIN_256_1_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, 256, 1, 4, true)
#define COOMMNN_SEGMENTED_MAIN_256_2_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, 256, 2, 4, false)
#define COOMMNN_SEGMENTED_MAIN_256_2_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, 256, 2, 4, true)
#define COOMMNN_SEGMENTED_MAIN_256_4_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, 256, 4, 4, false)
#define COOMMNN_SEGMENTED_MAIN_256_4_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, 256, 4, 4, true)
#define COOMMNN_SEGMENTED_MAIN_256_8_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, 256, 8, 4, false)
#define COOMMNN_SEGMENTED_MAIN_256_8_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_MAIN_KERNEL(T, I, A, B, C, 256, 8, 4, true)

#define COOMMNN_SEGMENTED_REMAINDER_256_1_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, 256, 1, 4, false)
#define COOMMNN_SEGMENTED_REMAINDER_256_1_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, 256, 1, 4, true)
#define COOMMNN_SEGMENTED_REMAINDER_256_2_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, 256, 2, 4, false)
#define COOMMNN_SEGMENTED_REMAINDER_256_2_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, 256, 2, 4, true)
#define COOMMNN_SEGMENTED_REMAINDER_256_4_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, 256, 4, 4, false)
#define COOMMNN_SEGMENTED_REMAINDER_256_4_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, 256, 4, 4, true)
#define COOMMNN_SEGMENTED_REMAINDER_256_8_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, 256, 8, 4, false)
#define COOMMNN_SEGMENTED_REMAINDER_256_8_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_REMAINDER_KERNEL(T, I, A, B, C, 256, 8, 4, true)
