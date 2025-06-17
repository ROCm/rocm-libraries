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
#include "rocsparse_scalar.hpp"

namespace rocsparse
{

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
#if defined(__gfx908__)
        __attribute__((amdgpu_waves_per_eu(6, 6)))
#endif
        void csrmmnn_row_split_shared_kernel(ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                             bool                 is_host_mode,
                                             bool                 conj_A,
                                             bool                 conj_B,
                                             J                    m,
                                             J                    n,
                                             int64_t              offsets_batch_stride_A,
                                             int64_t              columns_values_batch_stride_A,
                                             const I*             csr_row_ptr,
                                             const J*             csr_col_ind,
                                             const A*             csr_val,
                                             const B*             dense_B,
                                             int64_t              ldb,
                                             int64_t              batch_stride_B,
                                             C*                   dense_C,
                                             int64_t              ldc,
                                             int64_t              batch_stride_C,
                                             rocsparse_order      order_C,
                                             rocsparse_index_base idx_base);
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnn_row_split_kernel(ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                      bool                 is_host_mode,
                                      bool                 conj_A,
                                      bool                 conj_B,
                                      J                    start,
                                      J                    m,
                                      J                    n,
                                      int64_t              offsets_batch_stride_A,
                                      int64_t              columns_values_batch_stride_A,
                                      const I*             csr_row_ptr,
                                      const J*             csr_col_ind,
                                      const A*             csr_val,
                                      const B*             dense_B,
                                      int64_t              ldb,
                                      int64_t              batch_stride_B,
                                      C*                   dense_C,
                                      int64_t              ldc,
                                      int64_t              batch_stride_C,
                                      rocsparse_order      order_C,
                                      rocsparse_index_base idx_base);

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t SUB_WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnt_row_split_shared_remainder_kernel(ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T,
                                                                                           alpha),
                                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                       bool     is_host_mode,
                                                       bool     conj_A,
                                                       bool     conj_B,
                                                       J        col_start,
                                                       J        col_end,
                                                       J        m,
                                                       J        n,
                                                       int64_t  offsets_batch_stride_A,
                                                       int64_t  columns_values_batch_stride_A,
                                                       const I* csr_row_ptr,
                                                       const J* csr_col_ind,
                                                       const A* csr_val,
                                                       const B* dense_B,
                                                       int64_t  ldb,
                                                       int64_t  batch_stride_B,
                                                       C*       dense_C,
                                                       int64_t  ldc,
                                                       int64_t  batch_stride_C,
                                                       rocsparse_order      order_C,
                                                       rocsparse_index_base idx_base);

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmtn_row_split_kernel(ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                      bool                 is_host_mode,
                                      bool                 conj_A,
                                      bool                 conj_B,
                                      J                    m,
                                      J                    n,
                                      int64_t              offsets_batch_stride_A,
                                      int64_t              columns_values_batch_stride_A,
                                      const I*             csr_row_ptr,
                                      const J*             csr_col_ind,
                                      const A*             csr_val,
                                      const B*             dense_B,
                                      int64_t              ldb,
                                      int64_t              batch_stride_B,
                                      C*                   dense_C,
                                      int64_t              ldc,
                                      int64_t              batch_stride_C,
                                      rocsparse_order      order_C,
                                      rocsparse_index_base idx_base);

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmtt_row_split_kernel(ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                      bool                 is_host_mode,
                                      bool                 conj_A,
                                      bool                 conj_B,
                                      J                    m,
                                      J                    n,
                                      int64_t              offsets_batch_stride_A,
                                      int64_t              columns_values_batch_stride_A,
                                      const I*             csr_row_ptr,
                                      const J*             csr_col_ind,
                                      const A*             csr_val,
                                      const B*             dense_B,
                                      int64_t              ldb,
                                      int64_t              batch_stride_B,
                                      C*                   dense_C,
                                      int64_t              ldc,
                                      int64_t              batch_stride_C,
                                      rocsparse_order      order_C,
                                      rocsparse_index_base idx_base);
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t SUB_WF_SIZE,
              uint32_t LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__ void csrmmnt_row_split_subwfsize_x_loop_columns_kernel(
        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
        bool                 is_host_mode,
        J                    start,
        J                    end,
        J                    m,
        J                    n,
        int64_t              offsets_batch_stride_A,
        int64_t              columns_values_batch_stride_A,
        int64_t              ldb,
        int64_t              batch_stride_B,
        int64_t              ldc,
        int64_t              batch_stride_C,
        const I*             csr_row_ptr,
        const J*             csr_col_ind,
        const A*             csr_val,
        const B*             dense_B,
        C*                   dense_C,
        rocsparse_order      order_C,
        rocsparse_index_base idx_base,
        bool                 conj_A,
        bool                 conj_B);

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t SUBWFSIZE,
              uint32_t LOOPS,
              uint32_t... SUBWFSIZES_LIST,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_kernel(
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
            bool                 is_host_mode,
            J                    start,
            J                    end,
            J                    m,
            J                    n,
            int64_t              offsets_batch_stride_A,
            int64_t              columns_values_batch_stride_A,
            const I*             csr_row_ptr,
            const J*             csr_col_ind,
            const A*             csr_val,
            const B*             dense_B,
            int64_t              ldb,
            int64_t              batch_stride_B,
            C*                   dense_C,
            int64_t              ldc,
            int64_t              batch_stride_C,
            rocsparse_order      order_C,
            rocsparse_index_base idx_base,
            bool                 conj_A,
            bool                 conj_B);
}
