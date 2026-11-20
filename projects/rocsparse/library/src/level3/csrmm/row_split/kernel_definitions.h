/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
                                             int64_t              batch_count,
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

        // grid.y carries the dense column panel index (WF_SIZE columns each) and
        // is clamped by get_grid_size_y, so stride over the WF_SIZE-wide panels to cover all
        // columns when the panel count exceeds the cap.
        // Unsigned, so the last step past n cannot wrap; same width as the index, so no
        // 64-bit counter stays live in the kernel.
        using U = std::make_unsigned_t<J>;
        for(U col_panel = hipBlockIdx_y * WF_SIZE; col_panel < static_cast<U>(n);
            col_panel += hipGridDim_y * WF_SIZE)
        {
            // Grid-stride loop over the batch dimension (grid z).
            for(int64_t batch = hipBlockIdx_z; batch < batch_count; batch += hipGridDim_z)
            {
                rocsparse::csrmmnn_row_split_shared_device<BLOCKSIZE, WF_SIZE>(
                    alpha,
                    beta,
                    conj_A,
                    conj_B,
                    m,
                    n,
                    load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                    load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                    load_pointer(csr_val, batch, columns_values_batch_stride_A),
                    load_pointer(dense_B, batch, batch_stride_B),
                    ldb,
                    load_pointer(dense_C, batch, batch_stride_C),
                    ldc,
                    order_C,
                    idx_base,
                    static_cast<J>(col_panel));
            }
        }
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
                                      int64_t              batch_count,
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

        // grid.y carries the dense column panel index (LOOPS columns each) and is
        // clamped by get_grid_size_y, so stride over the LOOPS-wide panels to cover all
        // columns [start, ...) even when the panel count exceeds the cap. Every
        // launched panel is fully in-bounds by construction, reproduced by the
        // col_panel + LOOPS <= n condition.
        // Unsigned, so the last step past n cannot wrap; same width as the index, so no
        // 64-bit counter stays live in the kernel.
        using U = std::make_unsigned_t<J>;
        for(U col_panel = static_cast<U>(start) + LOOPS * hipBlockIdx_y;
            col_panel + LOOPS <= static_cast<U>(n);
            col_panel += LOOPS * hipGridDim_y)
        {
            // Grid-stride loop over the batch dimension (grid z).
            for(int64_t batch = hipBlockIdx_z; batch < batch_count; batch += hipGridDim_z)
            {
                rocsparse::csrmmnn_row_split_device<BLOCKSIZE, WF_SIZE, LOOPS>(
                    alpha,
                    beta,
                    conj_A,
                    conj_B,
                    static_cast<J>(col_panel),
                    m,
                    n,
                    load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                    load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                    load_pointer(csr_val, batch, columns_values_batch_stride_A),
                    load_pointer(dense_B, batch, batch_stride_B),
                    ldb,
                    load_pointer(dense_C, batch, batch_stride_C),
                    ldc,
                    order_C,
                    idx_base);
            }
        }
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
                                                       int64_t  batch_count,
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

        // Grid-stride loop over the batch dimension (grid y).
        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            rocsparse::csrmmnt_row_split_shared_remainder_device<BLOCKSIZE, WF_SIZE, SUB_WF_SIZE>(
                alpha,
                beta,
                conj_A,
                conj_B,
                col_start,
                col_end,
                m,
                n,
                load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                load_pointer(csr_val, batch, columns_values_batch_stride_A),
                load_pointer(dense_B, batch, batch_stride_B),
                ldb,
                load_pointer(dense_C, batch, batch_stride_C),
                ldc,
                order_C,
                idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t COL,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnt_row_split_general_kernel(ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                              ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                              bool                 is_host_mode,
                                              bool                 conj_A,
                                              bool                 conj_B,
                                              J                    col_start,
                                              J                    m,
                                              int64_t              batch_count,
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

        // Grid-stride loop over the batch dimension (grid y).
        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            rocsparse::csrmmnt_row_split_general_device<BLOCKSIZE, WF_SIZE, COL>(
                alpha,
                beta,
                conj_A,
                conj_B,
                col_start,
                m,
                load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                load_pointer(csr_val, batch, columns_values_batch_stride_A),
                load_pointer(dense_B, batch, batch_stride_B),
                ldb,
                load_pointer(dense_C, batch, batch_stride_C),
                ldc,
                order_C,
                idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t COL,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmmnn_row_split_general_kernel(ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                              ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                              bool                 is_host_mode,
                                              bool                 conj_A,
                                              bool                 conj_B,
                                              J                    col_start,
                                              J                    m,
                                              int64_t              batch_count,
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

        // Grid-stride loop over the batch dimension (grid y).
        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            rocsparse::csrmmnn_row_split_general_device<BLOCKSIZE, WF_SIZE, COL>(
                alpha,
                beta,
                conj_A,
                conj_B,
                col_start,
                m,
                load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                load_pointer(csr_val, batch, columns_values_batch_stride_A),
                load_pointer(dense_B, batch, batch_stride_B),
                ldb,
                load_pointer(dense_C, batch, batch_stride_C),
                ldc,
                order_C,
                idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t COL,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    __launch_bounds__(BLOCKSIZE) __global__
        void csrmm_row_split_general_kernel(ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                            bool                 is_host_mode,
                                            bool                 conj_A,
                                            bool                 conj_B,
                                            rocsparse_order      order_B,
                                            rocsparse_operation  trans_B,
                                            J                    col_start,
                                            J                    m,
                                            int64_t              batch_count,
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

        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            rocsparse::csrmm_row_split_general_device<BLOCKSIZE, WF_SIZE, COL>(
                alpha,
                beta,
                conj_A,
                conj_B,
                order_B,
                trans_B,
                col_start,
                m,
                load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                load_pointer(csr_val, batch, columns_values_batch_stride_A),
                load_pointer(dense_B, batch, batch_stride_B),
                ldb,
                load_pointer(dense_C, batch, batch_stride_C),
                ldc,
                order_C,
                idx_base);
        }
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
                                      J                    k,
                                      int64_t              batch_count,
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

        // grid.y carries the dense column panel index (WF_SIZE columns each) and
        // is clamped by get_grid_size_y, so stride over the WF_SIZE-wide panels to cover all
        // columns when the panel count exceeds the cap.
        // Unsigned, so the last step past n cannot wrap; same width as the index, so no
        // 64-bit counter stays live in the kernel.
        using U = std::make_unsigned_t<J>;
        for(U col_panel = hipBlockIdx_y * WF_SIZE; col_panel < static_cast<U>(n);
            col_panel += hipGridDim_y * WF_SIZE)
        {
            // Grid-stride loop over the batch dimension (grid z).
            for(int64_t batch = hipBlockIdx_z; batch < batch_count; batch += hipGridDim_z)
            {
                rocsparse::csrmmtn_row_split_device<BLOCKSIZE, WF_SIZE>(
                    alpha,
                    conj_A,
                    conj_B,
                    m,
                    n,
                    k,
                    load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                    load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                    load_pointer(csr_val, batch, columns_values_batch_stride_A),
                    load_pointer(dense_B, batch, batch_stride_B),
                    ldb,
                    load_pointer(dense_C, batch, batch_stride_C),
                    ldc,
                    order_C,
                    idx_base,
                    static_cast<J>(col_panel));
            }
        }
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
                                      J                    k,
                                      int64_t              batch_count,
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
        // grid.y carries the dense column panel index (WF_SIZE columns each) and
        // is clamped by get_grid_size_y, so stride over the WF_SIZE-wide panels to cover all
        // columns when the panel count exceeds the cap.
        // Unsigned, so the last step past n cannot wrap; same width as the index, so no
        // 64-bit counter stays live in the kernel.
        using U = std::make_unsigned_t<J>;
        for(U col_panel = hipBlockIdx_y * WF_SIZE; col_panel < static_cast<U>(n);
            col_panel += hipGridDim_y * WF_SIZE)
        {
            // Grid-stride loop over the batch dimension (grid z).
            for(int64_t batch = hipBlockIdx_z; batch < batch_count; batch += hipGridDim_z)
            {
                rocsparse::csrmmtt_row_split_device<BLOCKSIZE, WF_SIZE>(
                    alpha,
                    conj_A,
                    conj_B,
                    m,
                    n,
                    k,
                    load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                    load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                    load_pointer(csr_val, batch, columns_values_batch_stride_A),
                    load_pointer(dense_B, batch, batch_stride_B),
                    ldb,
                    load_pointer(dense_C, batch, batch_stride_C),
                    ldc,
                    order_C,
                    idx_base,
                    static_cast<J>(col_panel));
            }
        }
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
        int64_t              batch_count,
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

        // Grid-stride loop over the batch dimension (grid y).
        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
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
                ldb,
                ldc,
                load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                load_pointer(csr_val, batch, columns_values_batch_stride_A),
                load_pointer(dense_B, batch, batch_stride_B),
                load_pointer(dense_C, batch, batch_stride_C),
                order_C,
                idx_base,
                conj_A,
                conj_B);
        }
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
            int64_t              batch_count,
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

        // Grid-stride loop over the batch dimension (grid y).
        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            rocsparse::csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_device<
                BLOCKSIZE,
                WFSIZE,
                SUBWFSIZE,
                LOOPS,
                SUBWFSIZES_LIST...>(alpha,
                                    beta,
                                    start,
                                    end,
                                    m,
                                    n,
                                    load_pointer(csr_row_ptr, batch, offsets_batch_stride_A),
                                    load_pointer(csr_col_ind, batch, columns_values_batch_stride_A),
                                    load_pointer(csr_val, batch, columns_values_batch_stride_A),
                                    load_pointer(dense_B, batch, batch_stride_B),
                                    ldb,
                                    load_pointer(dense_C, batch, batch_stride_C),
                                    ldc,
                                    order_C,
                                    idx_base,
                                    conj_A,
                                    conj_B);
        }
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
            int64_t              batch_count,                                                   \
            int64_t              offsets_batch_stride_A,                                        \
            int64_t              columns_values_batch_stride_A,                                 \
            int64_t              ldb,                                                         \
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
            int64_t              batch_count,                                             \
            int64_t              offsets_batch_stride_A,                                  \
            int64_t              columns_values_batch_stride_A,                           \
            const I*             csr_row_ptr,                                             \
            const J*             csr_col_ind,                                             \
            const A*             csr_val,                                                 \
            const B*             dense_B,                                                 \
            int64_t              ldb,                                                   \
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
            int64_t              batch_count,                                             \
            int64_t              offsets_batch_stride_A,                                  \
            int64_t              columns_values_batch_stride_A,                           \
            const I*             csr_row_ptr,                                             \
            const J*             csr_col_ind,                                             \
            const A*             csr_val,                                                 \
            const B*             dense_B,                                                 \
            int64_t              ldb,                                                   \
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
            int64_t              batch_count,                                             \
            int64_t              offsets_batch_stride_A,                                  \
            int64_t              columns_values_batch_stride_A,                           \
            const I*             csr_row_ptr,                                             \
            const J*             csr_col_ind,                                             \
            const A*             csr_val,                                                 \
            const B*             dense_B,                                                 \
            int64_t              ldb,                                                   \
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
            int64_t              batch_count,                                                     \
            int64_t              offsets_batch_stride_A,                                          \
            int64_t              columns_values_batch_stride_A,                                   \
            const I*             csr_row_ptr,                                                     \
            const J*             csr_col_ind,                                                     \
            const A*             csr_val,                                                         \
            const B*             dense_B,                                                         \
            int64_t              ldb,                                                           \
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
            int64_t              batch_count,                                \
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
            J                    k,                                   \
            int64_t              batch_count,                         \
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
            J                    k,                                   \
            int64_t              batch_count,                         \
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

#define CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE, COL) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                          \
        rocsparse::csrmmnt_row_split_general_kernel<BLOCKSIZE, WFSIZE, COL>(       \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                         \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                          \
            bool                 is_host_mode,                                     \
            bool                 conj_A,                                           \
            bool                 conj_B,                                           \
            J                    col_start,                                        \
            J                    m,                                                \
            int64_t              batch_count,                                      \
            int64_t              offsets_batch_stride_A,                           \
            int64_t              columns_values_batch_stride_A,                    \
            const I*             csr_row_ptr,                                      \
            const J*             csr_col_ind,                                      \
            const A*             csr_val,                                          \
            const B*             dense_B,                                          \
            int64_t              ldb,                                            \
            int64_t              batch_stride_B,                                   \
            C*                   dense_C,                                          \
            int64_t              ldc,                                              \
            int64_t              batch_stride_C,                                   \
            rocsparse_order      order_C,                                          \
            rocsparse_index_base idx_base);

#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 1)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 2)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_3(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 3)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 4)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_5(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 5)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_6(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 6)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_7(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 7)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_8(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 8)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_9(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 9)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_10(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 10)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_11(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 11)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_12(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 12)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_13(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 13)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_14(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 14)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_15(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 15)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_16(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 16)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_17(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 17)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_18(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 18)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_19(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 19)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_20(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 20)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_21(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 21)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_22(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 22)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_23(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 23)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_24(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 24)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_25(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 25)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_26(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 26)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_27(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 27)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_28(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 28)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_29(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 29)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_30(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 30)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_31(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 31)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_32(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 32)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_33(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 33)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_34(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 34)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_35(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 35)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_36(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 36)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_37(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 37)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_38(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 38)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_39(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 39)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_32_40(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 32, 40)

#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_8_8_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_8_8_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_8_8_3(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 3)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_8_8_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_8_8_5(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 5)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_8_8_6(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 6)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_8_8_7(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 7)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_8_8_8(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 8)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_2_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 2, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_2_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 2, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_3_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 3, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_3_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 3, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_3_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 3, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_4_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 4, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_5_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 5, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_5_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 5, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_5_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 5, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_6_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 6, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_6_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 6, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_6_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 6, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_7_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 7, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_7_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 7, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_8_8_7_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 8, 8, 7, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_2_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 2, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_2_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 2, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_2_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 2, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_3_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 3, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_3_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 3, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_3_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 3, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_4_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_4_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 4, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_4_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 4, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_5_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 5, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_5_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 5, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_5_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 5, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_6_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 6, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_6_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 6, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_6_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 6, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_7_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 7, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_7_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 7, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_8_8_7_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                             \
        T, I, J, A, B, C, 256, 8, 8, 7, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_8_8_2_4_2_1( \
    T, I, J, A, B, C)                                                                   \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(              \
        T, I, J, A, B, C, 256, 8, 8, 2, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_8_8_3_4_2_1( \
    T, I, J, A, B, C)                                                                   \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(              \
        T, I, J, A, B, C, 256, 8, 8, 3, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_8_8_4_4_2_1( \
    T, I, J, A, B, C)                                                                   \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(              \
        T, I, J, A, B, C, 256, 8, 8, 4, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_8_8_5_4_2_1( \
    T, I, J, A, B, C)                                                                   \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(              \
        T, I, J, A, B, C, 256, 8, 8, 5, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_8_8_6_4_2_1( \
    T, I, J, A, B, C)                                                                   \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(              \
        T, I, J, A, B, C, 256, 8, 8, 6, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_8_8_7_4_2_1( \
    T, I, J, A, B, C)                                                                   \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(              \
        T, I, J, A, B, C, 256, 8, 8, 7, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_32_8_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_32_8_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_32_8_3(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 3)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_32_8_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_32_8_5(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 5)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_32_8_6(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 6)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_32_8_7(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 7)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_256_32_8_8(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 8)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_2_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 2, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_2_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 2, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_3_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 3, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_3_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 3, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_3_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 3, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_4_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 4, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_5_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 5, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_5_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 5, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_5_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 5, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_6_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 6, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_6_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 6, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_6_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 6, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_7_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 7, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_7_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 7, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_256_32_8_7_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_COLUMNS_KERNEL(T, I, J, A, B, C, 256, 32, 8, 7, 4)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_2_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 2, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_2_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 2, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_2_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 2, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_3_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 3, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_3_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 3, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_3_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 3, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_4_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_4_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 4, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_4_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 4, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_5_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 5, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_5_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 5, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_5_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 5, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_6_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 6, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_6_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 6, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_6_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 6, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_7_2_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 7, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_7_4_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 7, 4, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_256_32_8_7_4_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_COLUMNS_KERNEL(                              \
        T, I, J, A, B, C, 256, 32, 8, 7, 4, 2)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_32_8_2_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 32, 8, 2, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_32_8_3_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 32, 8, 3, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_32_8_4_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 32, 8, 4, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_32_8_5_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 32, 8, 5, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_32_8_6_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 32, 8, 6, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_256_32_8_7_4_2_1( \
    T, I, J, A, B, C)                                                                    \
    CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWF1_SWF2_SWF3_COLUMNS_KERNEL(               \
        T, I, J, A, B, C, 256, 32, 8, 7, 4, 2, 1)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 1)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 2)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_3(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 3)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 4)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_5(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 5)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_6(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 6)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_7(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 7)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_8(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 8)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_9(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 9)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_10(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 10)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_11(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 11)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_12(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 12)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_13(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 13)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_14(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 14)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_15(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 15)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_16(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 16)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_17(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 17)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_18(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 18)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_19(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 19)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_20(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 20)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_21(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 21)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_22(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 22)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_23(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 23)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_24(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 24)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_25(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 25)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_26(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 26)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_27(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 27)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_28(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 28)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_29(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 29)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_30(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 30)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_31(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 31)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_32(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 32)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_33(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 33)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_34(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 34)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_35(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 35)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_36(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 36)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_37(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 37)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_38(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 38)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_39(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 39)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_8_40(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 40)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_1(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 1)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_2(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 2)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_3(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 3)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_4(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 4)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_5(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 5)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_6(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 6)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_7(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 7)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_8(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 8)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_9(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 9)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_10(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 10)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_11(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 11)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_12(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 12)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_13(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 13)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_14(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 14)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_15(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 15)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_16(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 16)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_17(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 17)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_18(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 18)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_19(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 19)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_20(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 20)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_21(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 21)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_22(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 22)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_23(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 23)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_24(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 24)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_25(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 25)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_26(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 26)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_27(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 27)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_28(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 28)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_29(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 29)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_30(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 30)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_31(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 31)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_32(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 32)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_33(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 33)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_34(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 34)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_35(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 35)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_36(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 36)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_37(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 37)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_38(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 38)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_39(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 39)
#define CSRMMNT_ROW_SPLIT_GENERAL_256_16_40(T, I, J, A, B, C) \
    CSRMMNT_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 16, 40)

#define CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE, COL) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                          \
        rocsparse::csrmmnn_row_split_general_kernel<BLOCKSIZE, WFSIZE, COL>(       \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                         \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                          \
            bool                 is_host_mode,                                     \
            bool                 conj_A,                                           \
            bool                 conj_B,                                           \
            J                    col_start,                                        \
            J                    m,                                                \
            int64_t              batch_count,                                      \
            int64_t              offsets_batch_stride_A,                           \
            int64_t              columns_values_batch_stride_A,                    \
            const I*             csr_row_ptr,                                      \
            const J*             csr_col_ind,                                      \
            const A*             csr_val,                                          \
            const B*             dense_B,                                          \
            int64_t              ldb,                                              \
            int64_t              batch_stride_B,                                   \
            C*                   dense_C,                                          \
            int64_t              ldc,                                              \
            int64_t              batch_stride_C,                                   \
            rocsparse_order      order_C,                                          \
            rocsparse_index_base idx_base);

#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_1(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 1)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_2(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 2)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_3(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 3)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_4(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 4)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_5(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 5)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_6(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 6)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_7(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 7)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_8(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 8)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_9(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 9)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_10(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 10)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_11(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 11)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_12(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 12)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_13(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 13)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_14(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 14)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_15(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 15)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_16(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 16)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_17(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 17)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_18(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 18)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_19(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 19)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_20(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 20)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_21(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 21)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_22(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 22)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_23(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 23)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_24(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 24)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_25(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 25)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_26(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 26)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_27(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 27)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_28(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 28)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_29(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 29)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_30(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 30)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_31(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 31)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_32(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 32)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_33(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 33)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_34(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 34)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_35(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 35)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_36(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 36)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_37(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 37)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_38(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 38)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_39(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 39)
#define CSRMMNN_ROW_SPLIT_GENERAL_256_8_40(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 40)

#define CSRMMNN_ROW_SPLIT_256_8_1(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_KERNEL(T, I, J, A, B, C, 256, 8, 1)
#define CSRMMNN_ROW_SPLIT_256_8_8(T, I, J, A, B, C) \
    CSRMMNN_ROW_SPLIT_KERNEL(T, I, J, A, B, C, 256, 8, 8)

#define CSRMMTN_ROW_SPLIT_256_4(T, I, J, A, B, C) CSRMMTN_ROW_SPLIT_KERNEL(T, I, J, A, B, C, 256, 4)

#define CSRMMTT_ROW_SPLIT_256_4(T, I, J, A, B, C) CSRMMTT_ROW_SPLIT_KERNEL(T, I, J, A, B, C, 256, 4)


#define CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, BLOCKSIZE, WFSIZE, COL) \
    template __launch_bounds__(BLOCKSIZE) __global__ void                          \
        rocsparse::csrmm_row_split_general_kernel<BLOCKSIZE, WFSIZE, COL>(         \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),                         \
            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),                          \
            bool                 is_host_mode,                                     \
            bool                 conj_A,                                           \
            bool                 conj_B,                                           \
            rocsparse_order      order_B,                                          \
            rocsparse_operation  trans_B,                                          \
            J                    col_start,                                        \
            J                    m,                                                \
            int64_t              batch_count,                                      \
            int64_t              offsets_batch_stride_A,                           \
            int64_t              columns_values_batch_stride_A,                    \
            const I*             csr_row_ptr,                                      \
            const J*             csr_col_ind,                                      \
            const A*             csr_val,                                          \
            const B*             dense_B,                                          \
            int64_t              ldb,                                              \
            int64_t              batch_stride_B,                                   \
            C*                   dense_C,                                          \
            int64_t              ldc,                                              \
            int64_t              batch_stride_C,                                   \
            rocsparse_order      order_C,                                          \
            rocsparse_index_base idx_base);

#define CSRMM_ROW_SPLIT_GENERAL_256_8_1(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 1)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_2(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 2)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_3(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 3)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_4(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 4)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_5(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 5)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_6(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 6)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_7(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 7)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_8(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 8)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_9(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 9)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_10(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 10)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_11(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 11)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_12(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 12)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_13(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 13)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_14(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 14)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_15(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 15)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_16(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 16)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_17(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 17)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_18(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 18)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_19(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 19)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_20(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 20)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_21(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 21)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_22(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 22)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_23(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 23)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_24(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 24)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_25(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 25)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_26(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 26)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_27(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 27)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_28(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 28)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_29(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 29)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_30(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 30)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_31(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 31)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_32(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 32)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_33(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 33)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_34(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 34)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_35(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 35)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_36(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 36)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_37(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 37)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_38(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 38)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_39(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 39)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_40(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 40)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_41(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 41)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_42(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 42)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_43(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 43)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_44(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 44)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_45(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 45)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_46(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 46)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_47(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 47)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_48(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 48)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_49(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 49)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_50(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 50)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_51(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 51)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_52(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 52)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_53(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 53)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_54(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 54)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_55(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 55)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_56(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 56)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_57(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 57)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_58(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 58)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_59(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 59)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_60(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 60)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_61(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 61)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_62(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 62)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_63(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 63)
#define CSRMM_ROW_SPLIT_GENERAL_256_8_64(T, I, J, A, B, C) \
    CSRMM_ROW_SPLIT_GENERAL_KERNEL(T, I, J, A, B, C, 256, 8, 64)
