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
#include "../../csrmm_device_row_split.h"
#include "../csrmm_common.h"
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
                                             rocsparse_index_base idx_base)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == 0 && beta == 1)
        {
            return;
        }

        rocsparse::csrmmnn_row_split_shared_device<BLOCKSIZE, WF_SIZE>(
            alpha,
            beta,
            conj_A,
            conj_B,
            m,
            n,
            offsets_batch_stride_A,
            columns_values_batch_stride_A,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            dense_B,
            ldb,
            batch_stride_B,
            dense_C,
            ldc,
            batch_stride_C,
            order_C,
            idx_base);
    }

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
                                      rocsparse_index_base idx_base)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == 0 && beta == 1)
        {
            return;
        }

        rocsparse::csrmmnn_row_split_device<BLOCKSIZE, WF_SIZE, LOOPS>(
            alpha,
            beta,
            conj_A,
            conj_B,
            start,
            m,
            n,
            offsets_batch_stride_A,
            columns_values_batch_stride_A,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            dense_B,
            ldb,
            batch_stride_B,
            dense_C,
            ldc,
            batch_stride_C,
            order_C,
            idx_base);
    }

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
                                                       rocsparse_index_base idx_base)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == 0 && beta == 1)
        {
            return;
        }

        rocsparse::csrmmnt_row_split_shared_remainder_device<BLOCKSIZE, WF_SIZE, SUB_WF_SIZE>(
            alpha,
            beta,
            conj_A,
            conj_B,
            col_start,
            col_end,
            m,
            n,
            offsets_batch_stride_A,
            columns_values_batch_stride_A,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            dense_B,
            ldb,
            batch_stride_B,
            dense_C,
            ldc,
            batch_stride_C,
            order_C,
            idx_base);
    }

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
                                      rocsparse_index_base idx_base)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha == 0 && beta == 1)
        {
            return;
        }

        rocsparse::csrmmtn_row_split_device<BLOCKSIZE, WF_SIZE>(alpha,
                                                                conj_A,
                                                                conj_B,
                                                                m,
                                                                n,
                                                                offsets_batch_stride_A,
                                                                columns_values_batch_stride_A,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                csr_val,
                                                                dense_B,
                                                                ldb,
                                                                batch_stride_B,
                                                                dense_C,
                                                                ldc,
                                                                batch_stride_C,
                                                                order_C,
                                                                idx_base);
    }

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
                                      rocsparse_index_base idx_base)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == 0 && beta == 1)
        {
            return;
        }
        rocsparse::csrmmtt_row_split_device<BLOCKSIZE, WF_SIZE>(alpha,
                                                                conj_A,
                                                                conj_B,
                                                                m,
                                                                n,
                                                                offsets_batch_stride_A,
                                                                columns_values_batch_stride_A,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                csr_val,
                                                                dense_B,
                                                                ldb,
                                                                batch_stride_B,
                                                                dense_C,
                                                                ldc,
                                                                batch_stride_C,
                                                                order_C,
                                                                idx_base);
    }

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
        bool                 conj_B)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == 0 && beta == 1)
        {
            return;
        }

        rocsparse::csrmmnt_row_split_subwfsize_x_loop_columns_device<BLOCKSIZE,
                                                                     WF_SIZE,
                                                                     SUB_WF_SIZE,
                                                                     LOOPS>(
            alpha,
            beta,
            start,
            end,
            m,
            n,
            offsets_batch_stride_A,
            columns_values_batch_stride_A,
            ldb,
            batch_stride_B,
            ldc,
            batch_stride_C,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            dense_B,
            dense_C,
            order_C,
            idx_base,
            conj_A,
            conj_B);
    }

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
            bool                 conj_B)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha == 0 && beta == 1)
        {
            return;
        }

        rocsparse::csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_device<BLOCKSIZE,
                                                                               WFSIZE,
                                                                               SUBWFSIZE,
                                                                               LOOPS,
                                                                               SUBWFSIZES_LIST...>(
            alpha,
            beta,
            start,
            end,
            m,
            n,
            offsets_batch_stride_A,
            columns_values_batch_stride_A,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            dense_B,
            ldb,
            batch_stride_B,
            dense_C,
            ldc,
            batch_stride_C,
            order_C,
            idx_base,
            conj_A,
            conj_B);
    }
}

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(                                      \
    T, I, J, A, B, C, BLOCKSIZE, WFSIZE, SUBWFSIZE, LOOPS)                                      \
    template __launch_bounds__(BLOCKSIZE) __global__ void rocsparse::                           \
        csrmmnt_row_split_subwfsize_x_loop_columns_kernel<BLOCKSIZE, WFSIZE, SUBWFSIZE, LOOPS>( \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                                      \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                                       \
            bool                 is_host_mode,                                                  \
            J                    start,                                                         \
            J                    end,                                                           \
            J                    m,                                                             \
            J                    n,                                                             \
            int64_t              offsets_batch_stride_A,                                        \
            int64_t              columns_values_batch_stride_A,                                 \
            int64_t              ldb,                                                           \
            int64_t              batch_stride_B,                                                \
            int64_t              ldc,                                                           \
            int64_t              batch_stride_C,                                                \
            const I*             csr_row_ptr,                                                   \
            const J*             csr_col_ind,                                                   \
            const A*             csr_val,                                                       \
            const B*             dense_B,                                                       \
            C*                   dense_C,                                                       \
            rocsparse_order      order_C,                                                       \
            rocsparse_index_base idx_base,                                                      \
            bool                 conj_A,                                                        \
            bool                 conj_B);

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(                      \
    T, I, J, A, B, C, BLOCKSIZE, WFSIZE, SUBWFSIZE, LOOPS, SWF1)                          \
    template __launch_bounds__(BLOCKSIZE) __global__ void                                 \
        rocsparse::csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_kernel<BLOCKSIZE, \
                                                                               WFSIZE,    \
                                                                               SUBWFSIZE, \
                                                                               LOOPS,     \
                                                                               SWF1>(     \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                                \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                                 \
            bool                 is_host_mode,                                            \
            J                    start,                                                   \
            J                    end,                                                     \
            J                    m,                                                       \
            J                    n,                                                       \
            int64_t              offsets_batch_stride_A,                                  \
            int64_t              columns_values_batch_stride_A,                           \
            const I*             csr_row_ptr,                                             \
            const J*             csr_col_ind,                                             \
            const A*             csr_val,                                                 \
            const B*             dense_B,                                                 \
            int64_t              ldb,                                                     \
            int64_t              batch_stride_B,                                          \
            C*                   dense_C,                                                 \
            int64_t              ldc,                                                     \
            int64_t              batch_stride_C,                                          \
            rocsparse_order      order_C,                                                 \
            rocsparse_index_base idx_base,                                                \
            bool                 conj_A,                                                  \
            bool                 conj_B);

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                 \
    T, I, J, A, B, C, BLOCKSIZE, WFSIZE, SUBWFSIZE, LOOPS, SWF1, SWF2)                    \
    template __launch_bounds__(BLOCKSIZE) __global__ void                                 \
        rocsparse::csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_kernel<BLOCKSIZE, \
                                                                               WFSIZE,    \
                                                                               SUBWFSIZE, \
                                                                               LOOPS,     \
                                                                               SWF1,      \
                                                                               SWF2>(     \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                                \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                                 \
            bool                 is_host_mode,                                            \
            J                    start,                                                   \
            J                    end,                                                     \
            J                    m,                                                       \
            J                    n,                                                       \
            int64_t              offsets_batch_stride_A,                                  \
            int64_t              columns_values_batch_stride_A,                           \
            const I*             csr_row_ptr,                                             \
            const J*             csr_col_ind,                                             \
            const A*             csr_val,                                                 \
            const B*             dense_B,                                                 \
            int64_t              ldb,                                                     \
            int64_t              batch_stride_B,                                          \
            C*                   dense_C,                                                 \
            int64_t              ldc,                                                     \
            int64_t              batch_stride_C,                                          \
            rocsparse_order      order_C,                                                 \
            rocsparse_index_base idx_base,                                                \
            bool                 conj_A,                                                  \
            bool                 conj_B);

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(            \
    T, I, J, A, B, C, BLOCKSIZE, WFSIZE, SUBWFSIZE, LOOPS, SWF1, SWF2, SWF3)              \
    template __launch_bounds__(BLOCKSIZE) __global__ void                                 \
        rocsparse::csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_kernel<BLOCKSIZE, \
                                                                               WFSIZE,    \
                                                                               SUBWFSIZE, \
                                                                               LOOPS,     \
                                                                               SWF1,      \
                                                                               SWF2,      \
                                                                               SWF3>(     \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                                \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                                 \
            bool                 is_host_mode,                                            \
            J                    start,                                                   \
            J                    end,                                                     \
            J                    m,                                                       \
            J                    n,                                                       \
            int64_t              offsets_batch_stride_A,                                  \
            int64_t              columns_values_batch_stride_A,                           \
            const I*             csr_row_ptr,                                             \
            const J*             csr_col_ind,                                             \
            const A*             csr_val,                                                 \
            const B*             dense_B,                                                 \
            int64_t              ldb,                                                     \
            int64_t              batch_stride_B,                                          \
            C*                   dense_C,                                                 \
            int64_t              ldc,                                                     \
            int64_t              batch_stride_C,                                          \
            rocsparse_order      order_C,                                                 \
            rocsparse_index_base idx_base,                                                \
            bool                 conj_A,                                                  \
            bool                 conj_B);

#define CSRMMNT_ROW_SPLIT_SHARED_REMAINDER_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE, SUBWFSIZE) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                                         \
        rocsparse::csrmmnt_row_split_shared_remainder_kernel<BLOCKSIZE, WFSIZE, SUBWFSIZE>(       \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                                        \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                                         \
            bool                 is_host_mode,                                                    \
            bool                 conj_A,                                                          \
            bool                 conj_B,                                                          \
            J                    col_start,                                                       \
            J                    col_end,                                                         \
            J                    m,                                                               \
            J                    n,                                                               \
            int64_t              offsets_batch_stride_A,                                          \
            int64_t              columns_values_batch_stride_A,                                   \
            const I*             csr_row_ptr,                                                     \
            const J*             csr_col_ind,                                                     \
            const A*             csr_val,                                                         \
            const B*             dense_B,                                                         \
            int64_t              ldb,                                                             \
            int64_t              batch_stride_B,                                                  \
            C*                   dense_C,                                                         \
            int64_t              ldc,                                                             \
            int64_t              batch_stride_C,                                                  \
            rocsparse_order      order_C,                                                         \
            rocsparse_index_base idx_base);

#define CSRMMNN_ROW_SPLIT_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE, LOOPS) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                    \
        rocsparse::csrmmnn_row_split_kernel<BLOCKSIZE, WFSIZE, LOOPS>(       \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                   \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                    \
            bool                 is_host_mode,                               \
            bool                 conj_A,                                     \
            bool                 conj_B,                                     \
            J                    start,                                      \
            J                    m,                                          \
            J                    n,                                          \
            int64_t              offsets_batch_stride_A,                     \
            int64_t              columns_values_batch_stride_A,              \
            const I*             csr_row_ptr,                                \
            const J*             csr_col_ind,                                \
            const A*             csr_val,                                    \
            const B*             dense_B,                                    \
            int64_t              ldb,                                        \
            int64_t              batch_stride_B,                             \
            C*                   dense_C,                                    \
            int64_t              ldc,                                        \
            int64_t              batch_stride_C,                             \
            rocsparse_order      order_C,                                    \
            rocsparse_index_base idx_base);

#define CSRMMTN_ROW_SPLIT_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE) \
    template __launch_bounds__(BLOCKSIZE) __global__ void             \
        rocsparse::csrmmtn_row_split_kernel<BLOCKSIZE, WFSIZE>(       \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),            \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),             \
            bool                 is_host_mode,                        \
            bool                 conj_A,                              \
            bool                 conj_B,                              \
            J                    m,                                   \
            J                    n,                                   \
            int64_t              offsets_batch_stride_A,              \
            int64_t              columns_values_batch_stride_A,       \
            const I*             csr_row_ptr,                         \
            const J*             csr_col_ind,                         \
            const A*             csr_val,                             \
            const B*             dense_B,                             \
            int64_t              ldb,                                 \
            int64_t              batch_stride_B,                      \
            C*                   dense_C,                             \
            int64_t              ldc,                                 \
            int64_t              batch_stride_C,                      \
            rocsparse_order      order_C,                             \
            rocsparse_index_base idx_base);

#define CSRMMTT_ROW_SPLIT_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE) \
    template __launch_bounds__(BLOCKSIZE) __global__ void             \
        rocsparse::csrmmtt_row_split_kernel<BLOCKSIZE, WFSIZE>(       \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),            \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),             \
            bool                 is_host_mode,                        \
            bool                 conj_A,                              \
            bool                 conj_B,                              \
            J                    m,                                   \
            J                    n,                                   \
            int64_t              offsets_batch_stride_A,              \
            int64_t              columns_values_batch_stride_A,       \
            const I*             csr_row_ptr,                         \
            const J*             csr_col_ind,                         \
            const A*             csr_val,                             \
            const B*             dense_B,                             \
            int64_t              ldb,                                 \
            int64_t              batch_stride_B,                      \
            C*                   dense_C,                             \
            int64_t              ldc,                                 \
            int64_t              batch_stride_C,                      \
            rocsparse_order      order_C,                             \
            rocsparse_index_base idx_base);

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_16_8_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_16_8_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_16_8_3(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 3)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_16_8_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_16_8_5(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 5)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_16_8_6(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 6)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_16_8_7(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 7)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_16_8_8(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 8)

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_2_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 2, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_2_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 2, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_3_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 3, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_3_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 3, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_3_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 3, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_4_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 4, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_5_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 5, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_5_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 5, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_5_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 5, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_6_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 6, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_6_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 6, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_6_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 6, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_7_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 7, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_7_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 7, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_16_8_7_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 16, 8, 7, 4)

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_2_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 2, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_2_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 2, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_2_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 2, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_3_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 3, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_3_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 3, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_3_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 3, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_4_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_4_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 4, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_4_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 4, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_5_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 5, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_5_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 5, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_5_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 5, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_6_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 6, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_6_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 6, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_6_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 6, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_7_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 7, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_7_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 7, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_16_8_7_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 16, 8, 7, 4, 2)

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_16_8_2_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 16, 8, 2, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_16_8_3_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 16, 8, 3, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_16_8_4_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 16, 8, 4, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_16_8_5_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 16, 8, 5, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_16_8_6_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 16, 8, 6, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_16_8_7_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 16, 8, 7, 4, 2, 1)

#define CSRMMNT_ROW_SPLIT_SHARED_REMAINDER_256_8_8(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SHARED_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 8, 8)
#define CSRMMNT_ROW_SPLIT_SHARED_REMAINDER_256_16_16(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SHARED_REMAINDER_KERNEL(T, I, J, A, B, C, 256, 16, 16)

#define CSRMMNN_ROW_SPLIT_256_8_1(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_KERNEL(T, I, J, A, B, C, 256, 8, 1)
#define CSRMMNN_ROW_SPLIT_256_8_8(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_KERNEL(T, I, J, A, B, C, 256, 8, 8)

#define CSRMMTN_ROW_SPLIT_256_4(T, I, J, A, B, C) CSRMMTN_ROW_SPLIT_KERNEL(T, I, J, A, B, C, 256, 4)

#define CSRMMTT_ROW_SPLIT_256_4(T, I, J, A, B, C) CSRMMTT_ROW_SPLIT_KERNEL(T, I, J, A, B, C, 256, 4)
