/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "../../csrmm_device_nnz_split.h"
#include "../csrmm_common.h"
#include "rocsparse_scalar.hpp"

namespace rocsparse
{

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnn_nnz_split_main_kernel(bool conj_A,
                                           bool conj_B,
                                           J    ncol,
                                           J    m,
                                           J    n,
                                           J    k,
                                           I    nnz,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                           J* __restrict__ row_block_red,
                                           T* __restrict__ val_block_red,
                                           const J* __restrict__ row_limits,
                                           const I* __restrict__ csr_row_ptr,
                                           const J* __restrict__ csr_col_ind,
                                           const A* __restrict__ csr_val,
                                           const B* __restrict__ dense_B,
                                           int64_t ldb,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                           C* __restrict__ dense_C,
                                           int64_t              ldc,
                                           rocsparse_order      order_C,
                                           rocsparse_index_base idx_base,
                                           bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha == 0 && beta == 1)
        {
            row_block_red[hipBlockIdx_x] = -1;
            return;
        }

        rocsparse::csrmmnn_nnz_split_main_device<BLOCKSIZE, WF_SIZE>(conj_A,
                                                                     conj_B,
                                                                     ncol,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     nnz,
                                                                     alpha,
                                                                     row_block_red,
                                                                     val_block_red,
                                                                     row_limits,
                                                                     csr_row_ptr,
                                                                     csr_col_ind,
                                                                     csr_val,
                                                                     dense_B,
                                                                     ldb,
                                                                     beta,
                                                                     dense_C,
                                                                     ldc,
                                                                     order_C,
                                                                     idx_base);
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnn_nnz_split_remainder_kernel(bool conj_A,
                                                bool conj_B,
                                                J    offset,
                                                J    m,
                                                J    n,
                                                J    k,
                                                I    nnz,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                J* __restrict__ row_block_red,
                                                T* __restrict__ val_block_red,
                                                const J* __restrict__ row_limits,
                                                const I* __restrict__ csr_row_ptr,
                                                const J* __restrict__ csr_col_ind,
                                                const A* __restrict__ csr_val,
                                                const B* __restrict__ dense_B,
                                                int64_t ldb,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                C* __restrict__ dense_C,
                                                int64_t              ldc,
                                                rocsparse_order      order_C,
                                                rocsparse_index_base idx_base,
                                                bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha == 0 && beta == 1)
        {
            row_block_red[hipBlockIdx_x] = -1;
            return;
        }

        rocsparse::csrmmnn_nnz_split_remainder_device<BLOCKSIZE, WF_SIZE>(conj_A,
                                                                          conj_B,
                                                                          offset,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          nnz,
                                                                          alpha,
                                                                          row_block_red,
                                                                          val_block_red,
                                                                          row_limits,
                                                                          csr_row_ptr,
                                                                          csr_col_ind,
                                                                          csr_val,
                                                                          dense_B,
                                                                          ldb,
                                                                          beta,
                                                                          dense_C,
                                                                          ldc,
                                                                          order_C,
                                                                          idx_base);
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              unsigned int LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnt_nnz_split_main_kernel(bool conj_A,
                                           bool conj_B,
                                           J    ncol,
                                           J    m,
                                           J    n,
                                           J    k,
                                           I    nnz,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                           const J* __restrict__ row_limits,
                                           const I* __restrict__ csr_row_ptr,
                                           const J* __restrict__ csr_col_ind,
                                           const A* __restrict__ csr_val,
                                           const B* __restrict__ dense_B,
                                           int64_t ldb,
                                           C* __restrict__ dense_C,
                                           int64_t              ldc,
                                           rocsparse_order      order_C,
                                           rocsparse_index_base idx_base,
                                           bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        rocsparse::csrmmnt_nnz_split_main_device<BLOCKSIZE, WF_SIZE, LOOPS>(conj_A,
                                                                            conj_B,
                                                                            ncol,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            nnz,
                                                                            alpha,
                                                                            row_limits,
                                                                            csr_row_ptr,
                                                                            csr_col_ind,
                                                                            csr_val,
                                                                            dense_B,
                                                                            ldb,
                                                                            dense_C,
                                                                            ldc,
                                                                            order_C,
                                                                            idx_base);
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnt_nnz_split_remainder_kernel(bool conj_A,
                                                bool conj_B,
                                                J    offset,
                                                J    m,
                                                J    n,
                                                J    k,
                                                I    nnz,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                const J* __restrict__ row_limits,
                                                const I* __restrict__ csr_row_ptr,
                                                const J* __restrict__ csr_col_ind,
                                                const A* __restrict__ csr_val,
                                                const B* __restrict__ dense_B,
                                                int64_t ldb,
                                                C* __restrict__ dense_C,
                                                int64_t              ldc,
                                                rocsparse_order      order_C,
                                                rocsparse_index_base idx_base,
                                                bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        rocsparse::csrmmnt_nnz_split_remainder_device<BLOCKSIZE, WF_SIZE>(conj_A,
                                                                          conj_B,
                                                                          offset,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          nnz,
                                                                          alpha,
                                                                          row_limits,
                                                                          csr_row_ptr,
                                                                          csr_col_ind,
                                                                          csr_val,
                                                                          dense_B,
                                                                          ldb,
                                                                          dense_C,
                                                                          ldc,
                                                                          order_C,
                                                                          idx_base);
    }
}

#define CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                  \
        rocsparse::csrmmnn_nnz_split_main_kernel<BLOCKSIZE, WFSIZE>(       \
            bool conj_A,                                                   \
            bool conj_B,                                                   \
            J    ncol,                                                     \
            J    m,                                                        \
            J    n,                                                        \
            J    k,                                                        \
            I    nnz,                                                      \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                 \
            J* __restrict__ row_block_red,                                 \
            T* __restrict__ val_block_red,                                 \
            const J* __restrict__ row_limits,                              \
            const I* __restrict__ csr_row_ptr,                             \
            const J* __restrict__ csr_col_ind,                             \
            const A* __restrict__ csr_val,                                 \
            const B* __restrict__ dense_B,                                 \
            int64_t ldb,                                                   \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                  \
            C* __restrict__ dense_C,                                       \
            int64_t              ldc,                                      \
            rocsparse_order      order_C,                                  \
            rocsparse_index_base idx_base,                                 \
            bool                 is_host_mode);

#define CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                       \
        rocsparse::csrmmnn_nnz_split_remainder_kernel<BLOCKSIZE, WFSIZE>(       \
            bool conj_A,                                                        \
            bool conj_B,                                                        \
            J    offset,                                                        \
            J    m,                                                             \
            J    n,                                                             \
            J    k,                                                             \
            I    nnz,                                                           \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                      \
            J* __restrict__ row_block_red,                                      \
            T* __restrict__ val_block_red,                                      \
            const J* __restrict__ row_limits,                                   \
            const I* __restrict__ csr_row_ptr,                                  \
            const J* __restrict__ csr_col_ind,                                  \
            const A* __restrict__ csr_val,                                      \
            const B* __restrict__ dense_B,                                      \
            int64_t ldb,                                                        \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                       \
            C* __restrict__ dense_C,                                            \
            int64_t              ldc,                                           \
            rocsparse_order      order_C,                                       \
            rocsparse_index_base idx_base,                                      \
            bool                 is_host_mode);

#define CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE, LOOPS) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                         \
        rocsparse::csrmmnt_nnz_split_main_kernel<BLOCKSIZE, WFSIZE, LOOPS>(       \
            bool conj_A,                                                          \
            bool conj_B,                                                          \
            J    ncol,                                                            \
            J    m,                                                               \
            J    n,                                                               \
            J    k,                                                               \
            I    nnz,                                                             \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                        \
            const J* __restrict__ row_limits,                                     \
            const I* __restrict__ csr_row_ptr,                                    \
            const J* __restrict__ csr_col_ind,                                    \
            const A* __restrict__ csr_val,                                        \
            const B* __restrict__ dense_B,                                        \
            int64_t ldb,                                                          \
            C* __restrict__ dense_C,                                              \
            int64_t              ldc,                                             \
            rocsparse_order      order_C,                                         \
            rocsparse_index_base idx_base,                                        \
            bool                 is_host_mode);

#define CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                       \
        rocsparse::csrmmnt_nnz_split_remainder_kernel<BLOCKSIZE, WFSIZE>(       \
            bool conj_A,                                                        \
            bool conj_B,                                                        \
            J    offset,                                                        \
            J    m,                                                             \
            J    n,                                                             \
            J    k,                                                             \
            I    nnz,                                                           \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                      \
            const J* __restrict__ row_limits,                                   \
            const I* __restrict__ csr_row_ptr,                                  \
            const J* __restrict__ csr_col_ind,                                  \
            const A* __restrict__ csr_val,                                      \
            const B* __restrict__ dense_B,                                      \
            int64_t ldb,                                                        \
            C* __restrict__ dense_C,                                            \
            int64_t              ldc,                                           \
            rocsparse_order      order_C,                                       \
            rocsparse_index_base idx_base,                                      \
            bool                 is_host_mode);

#define CSRMMNN_NNZ_SPLIT_MAIN_256_1(T, I, J, A, B, C) \
    CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 1)
#define CSRMMNN_NNZ_SPLIT_MAIN_256_2(T, I, J, A, B, C) \
    CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 2)
#define CSRMMNN_NNZ_SPLIT_MAIN_256_4(T, I, J, A, B, C) \
    CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 4)
#define CSRMMNN_NNZ_SPLIT_MAIN_256_8(T, I, J, A, B, C) \
    CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 8)

#define CSRMMNN_NNZ_SPLIT_REMAINDER_256_1(T, I, J, A, B, C) \
    CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 1)
#define CSRMMNN_NNZ_SPLIT_REMAINDER_256_2(T, I, J, A, B, C) \
    CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 2)
#define CSRMMNN_NNZ_SPLIT_REMAINDER_256_4(T, I, J, A, B, C) \
    CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 4)
#define CSRMMNN_NNZ_SPLIT_REMAINDER_256_8(T, I, J, A, B, C) \
    CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 8)

#define CSRMMNT_NNZ_SPLIT_MAIN_256_32_2(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 32, 2)
#define CSRMMNT_NNZ_SPLIT_MAIN_256_32_4(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 32, 4)
#define CSRMMNT_NNZ_SPLIT_MAIN_256_32_6(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 32, 6)
#define CSRMMNT_NNZ_SPLIT_MAIN_256_32_8(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 32, 8)
#define CSRMMNT_NNZ_SPLIT_MAIN_256_64_1(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 64, 1)
#define CSRMMNT_NNZ_SPLIT_MAIN_256_64_2(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 64, 2)
#define CSRMMNT_NNZ_SPLIT_MAIN_256_64_3(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 64, 3)
#define CSRMMNT_NNZ_SPLIT_MAIN_256_64_4(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(T, I, J, A, B, C, 256, 64, 4)

#define CSRMMNT_NNZ_SPLIT_REMAINDER_256_1(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 1)
#define CSRMMNT_NNZ_SPLIT_REMAINDER_256_2(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 2)
#define CSRMMNT_NNZ_SPLIT_REMAINDER_256_4(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 4)
#define CSRMMNT_NNZ_SPLIT_REMAINDER_256_8(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 8)
#define CSRMMNT_NNZ_SPLIT_REMAINDER_256_16(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 16)
#define CSRMMNT_NNZ_SPLIT_REMAINDER_256_32(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 32)
#define CSRMMNT_NNZ_SPLIT_REMAINDER_256_64(T, I, J, A, B, C) \
    CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 64)
