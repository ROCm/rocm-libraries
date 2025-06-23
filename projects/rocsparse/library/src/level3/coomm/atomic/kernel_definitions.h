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
#include "../../coomm_device_atomic.h"
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
        void coommnn_atomic_main(bool    conj_A,
                                 bool    conj_B,
                                 I       ncol,
                                 int64_t nnz,
                                 I       n,
                                 int64_t batch_stride_A,
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
            rocsparse::coommnn_atomic_main_device<BLOCKSIZE, WF_SIZE, LOOPS, TRANSB>(
                conj_A,
                conj_B,
                ncol,
                nnz,
                n,
                alpha,
                load_pointer(coo_row_ind, hipBlockIdx_y, batch_stride_A),
                load_pointer(coo_col_ind, hipBlockIdx_y, batch_stride_A),
                load_pointer(coo_val, hipBlockIdx_y, batch_stride_A),
                load_pointer(dense_B, hipBlockIdx_y, batch_stride_B),
                ldb,
                load_pointer(dense_C, hipBlockIdx_y, batch_stride_C),
                ldc,
                order_C,
                idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void coommnn_atomic_remainder(bool    conj_A,
                                      bool    conj_B,
                                      I       ncol_offset,
                                      I       n,
                                      int64_t nnz,
                                      int64_t batch_stride_A,
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
            rocsparse::coommnn_atomic_remainder_device<BLOCKSIZE, WF_SIZE, TRANSB>(
                conj_A,
                conj_B,
                ncol_offset,
                n,
                nnz,
                alpha,
                load_pointer(coo_row_ind, hipBlockIdx_y, batch_stride_A),
                load_pointer(coo_col_ind, hipBlockIdx_y, batch_stride_A),
                load_pointer(coo_val, hipBlockIdx_y, batch_stride_A),
                load_pointer(dense_B, hipBlockIdx_y, batch_stride_B),
                ldb,
                load_pointer(dense_C, hipBlockIdx_y, batch_stride_C),
                ldc,
                order_C,
                idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void coommtn_atomic_main(bool    conj_A,
                                 bool    conj_B,
                                 int64_t nnz,
                                 I       n,
                                 int64_t batch_stride_A,
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
            rocsparse::coommtn_atomic_device<BLOCKSIZE, TRANSB>(
                conj_A,
                conj_B,
                nnz,
                n,
                alpha,
                load_pointer(coo_row_ind, hipBlockIdx_z, batch_stride_A),
                load_pointer(coo_col_ind, hipBlockIdx_z, batch_stride_A),
                load_pointer(coo_val, hipBlockIdx_z, batch_stride_A),
                load_pointer(dense_B, hipBlockIdx_z, batch_stride_B),
                ldb,
                load_pointer(dense_C, hipBlockIdx_z, batch_stride_C),
                ldc,
                order_C,
                idx_base);
        }
    }
}

#define COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, BLOCKSIZE, WFSIZE, LOOPS, TRANSB) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                           \
        rocsparse::coommnn_atomic_main<BLOCKSIZE, WFSIZE, LOOPS, TRANSB>(           \
            bool    conj_A,                                                         \
            bool    conj_B,                                                         \
            I       ncol,                                                           \
            int64_t nnz,                                                            \
            I       n,                                                              \
            int64_t batch_stride_A,                                                 \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                          \
            const I* __restrict__ coo_row_ind,                                      \
            const I* __restrict__ coo_col_ind,                                      \
            const A* __restrict__ coo_val,                                          \
            const B* __restrict__ dense_B,                                          \
            int64_t ldb,                                                            \
            int64_t batch_stride_B,                                                 \
            C* __restrict__ dense_C,                                                \
            int64_t              ldc,                                               \
            int64_t              batch_stride_C,                                    \
            rocsparse_order      order_C,                                           \
            rocsparse_index_base idx_base,                                          \
            bool                 is_host_mode);

#define COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, BLOCKSIZE, WFSIZE, TRANSB) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                         \
        rocsparse::coommnn_atomic_remainder<BLOCKSIZE, WFSIZE, TRANSB>(           \
            bool    conj_A,                                                       \
            bool    conj_B,                                                       \
            I       ncol_offset,                                                  \
            I       n,                                                            \
            int64_t nnz,                                                          \
            int64_t batch_stride_A,                                               \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                        \
            const I* __restrict__ coo_row_ind,                                    \
            const I* __restrict__ coo_col_ind,                                    \
            const A* __restrict__ coo_val,                                        \
            const B* __restrict__ dense_B,                                        \
            int64_t ldb,                                                          \
            int64_t batch_stride_B,                                               \
            C* __restrict__ dense_C,                                              \
            int64_t              ldc,                                             \
            int64_t              batch_stride_C,                                  \
            rocsparse_order      order_C,                                         \
            rocsparse_index_base idx_base,                                        \
            bool                 is_host_mode);

#define COOMMTN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, BLOCKSIZE, TRANSB) \
    template __launch_bounds__(BLOCKSIZE) __global__ void            \
        rocsparse::coommtn_atomic_main<BLOCKSIZE, TRANSB>(           \
            bool    conj_A,                                          \
            bool    conj_B,                                          \
            int64_t nnz,                                             \
            I       n,                                               \
            int64_t batch_stride_A,                                  \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),           \
            const I* __restrict__ coo_row_ind,                       \
            const I* __restrict__ coo_col_ind,                       \
            const A* __restrict__ coo_val,                           \
            const B* __restrict__ dense_B,                           \
            int64_t ldb,                                             \
            int64_t batch_stride_B,                                  \
            C* __restrict__ dense_C,                                 \
            int64_t              ldc,                                \
            int64_t              batch_stride_C,                     \
            rocsparse_order      order_C,                            \
            rocsparse_index_base idx_base,                           \
            bool                 is_host_mode);

#define COOMMNN_ATOMIC_MAIN_256_32_2_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 32, 2, false)
#define COOMMNN_ATOMIC_MAIN_256_32_2_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 32, 2, true)
#define COOMMNN_ATOMIC_MAIN_256_32_4_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 32, 4, false)
#define COOMMNN_ATOMIC_MAIN_256_32_4_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 32, 4, true)
#define COOMMNN_ATOMIC_MAIN_256_32_6_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 32, 6, false)
#define COOMMNN_ATOMIC_MAIN_256_32_6_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 32, 6, true)
#define COOMMNN_ATOMIC_MAIN_256_32_8_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 32, 8, false)
#define COOMMNN_ATOMIC_MAIN_256_32_8_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 32, 8, true)
#define COOMMNN_ATOMIC_MAIN_256_64_1_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 64, 1, false)
#define COOMMNN_ATOMIC_MAIN_256_64_1_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 64, 1, true)
#define COOMMNN_ATOMIC_MAIN_256_64_2_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 64, 2, false)
#define COOMMNN_ATOMIC_MAIN_256_64_2_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 64, 2, true)
#define COOMMNN_ATOMIC_MAIN_256_64_3_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 64, 3, false)
#define COOMMNN_ATOMIC_MAIN_256_64_3_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 64, 3, true)
#define COOMMNN_ATOMIC_MAIN_256_64_4_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 64, 4, false)
#define COOMMNN_ATOMIC_MAIN_256_64_4_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, 64, 4, true)
#define COOMMNN_ATOMIC_REMAINDER_256_1_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 1, false)
#define COOMMNN_ATOMIC_REMAINDER_256_1_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 1, true)
#define COOMMNN_ATOMIC_REMAINDER_256_2_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 2, false)
#define COOMMNN_ATOMIC_REMAINDER_256_2_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 2, true)
#define COOMMNN_ATOMIC_REMAINDER_256_4_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 4, false)
#define COOMMNN_ATOMIC_REMAINDER_256_4_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 4, true)
#define COOMMNN_ATOMIC_REMAINDER_256_8_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 8, false)
#define COOMMNN_ATOMIC_REMAINDER_256_8_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 8, true)
#define COOMMNN_ATOMIC_REMAINDER_256_16_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 16, false)
#define COOMMNN_ATOMIC_REMAINDER_256_16_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 16, true)
#define COOMMNN_ATOMIC_REMAINDER_256_32_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 32, false)
#define COOMMNN_ATOMIC_REMAINDER_256_32_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 32, true)
#define COOMMNN_ATOMIC_REMAINDER_256_64_0(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 64, false)
#define COOMMNN_ATOMIC_REMAINDER_256_64_1(T, I, A, B, C) \
    COOMMNN_ATOMIC_REMAINDER_KERNEL(T, I, A, B, C, 256, 64, true)

#define COOMMTN_ATOMIC_MAIN_KERNEL_256_0(T, I, A, B, C) \
    COOMMTN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, false)
#define COOMMTN_ATOMIC_MAIN_KERNEL_256_1(T, I, A, B, C) \
    COOMMTN_ATOMIC_MAIN_KERNEL(T, I, A, B, C, 256, true)
