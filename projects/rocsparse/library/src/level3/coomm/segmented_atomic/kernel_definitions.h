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
#include "../../coomm_device_segmented_atomic.h"
#include "../coomm_common.h"
#include "rocsparse_scalar.hpp"

namespace rocsparse
{
    template <uint32_t WF_SIZE,
              uint32_t LOOPS,
              uint32_t COLS,
              bool     NT,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(WF_SIZE) __global__
        void coommnn_segmented_atomic(rocsparse_operation trans_B,
                                      int64_t             nnz,
                                      I                   n,
                                      int64_t             batch_stride_A,
                                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
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
            rocsparse::coommnn_segmented_atomic_device<WF_SIZE, LOOPS, COLS, NT>(trans_B,
                                                                                 nnz,
                                                                                 n,
                                                                                 batch_stride_A,
                                                                                 alpha,
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

#define COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, WFSIZE, LOOPS, COLS, NT) \
    template __launch_bounds__(WFSIZE) __global__ void                          \
        rocsparse::coommnn_segmented_atomic<WFSIZE, LOOPS, COLS, NT>(           \
            rocsparse_operation trans_B,                                        \
            int64_t             nnz,                                            \
            I                   n,                                              \
            int64_t             batch_stride_A,                                 \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                      \
            const I* __restrict__ coo_row_ind,                                  \
            const I* __restrict__ coo_col_ind,                                  \
            const A* __restrict__ coo_val,                                      \
            const B* __restrict__ dense_B,                                      \
            int64_t ldb,                                                        \
            int64_t batch_stride_B,                                             \
            C* __restrict__ dense_C,                                            \
            int64_t              ldc,                                           \
            int64_t              batch_stride_C,                                \
            rocsparse_order      order_C,                                       \
            rocsparse_index_base idx_base,                                      \
            bool                 is_host_mode);

#define COOMMNN_SEGMENTED_ATOMIC_32_16_1_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 1, false)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_1_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 1, true)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_2_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 2, false)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_2_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 2, true)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_3_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 3, false)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_3_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 3, true)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 4, false)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 4, true)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_5_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 5, false)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_5_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 5, true)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_6_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 6, false)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_6_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 6, true)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_7_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 7, false)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_7_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 7, true)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_8_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 8, false)
#define COOMMNN_SEGMENTED_ATOMIC_32_16_8_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 32, 16, 8, true)

#define COOMMNN_SEGMENTED_ATOMIC_64_16_1_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 1, false)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_1_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 1, true)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_2_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 2, false)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_2_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 2, true)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_3_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 3, false)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_3_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 3, true)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_4_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 4, false)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_4_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 4, true)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_5_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 5, false)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_5_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 5, true)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_6_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 6, false)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_6_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 6, true)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_7_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 7, false)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_7_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 7, true)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_8_0(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 8, false)
#define COOMMNN_SEGMENTED_ATOMIC_64_16_8_1(T, I, A, B, C) \
    COOMMNN_SEGMENTED_ATOMIC_KERNEL(T, I, A, B, C, 64, 16, 8, true)
