/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_grid.hpp"
#include "rocsparse_utility.hpp"

#include <map>

#include "csrmm/row_split/kernel_declarations.h"
#include "csrmm_device_row_split.h"
#include "rocsparse_common.h"

namespace rocsparse
{
#define LAUNCH_CSRMMNT_ROW_SPLIT_SHARED_SUBWFSIZE_X_LOOP_COLUMNS(                              \
    CSRMMNT_DIM, WFSIZE, SUBWFSIZE, LOOPS)                                                     \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                        \
        (rocsparse::csrmmnt_row_split_subwfsize_x_loop_columns_kernel<CSRMMNT_DIM,             \
                                                                      WFSIZE,                  \
                                                                      SUBWFSIZE,               \
                                                                      LOOPS>),                 \
        dim3((m - 1) / (CSRMMNT_DIM / WFSIZE) + 1, get_grid_size_y<J>(handle, batch_count_C)), \
        dim3(CSRMMNT_DIM),                                                                     \
        0,                                                                                     \
        handle->stream,                                                                        \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                          \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),                           \
        handle->pointer_mode == rocsparse_pointer_mode_host,                                   \
        start,                                                                                 \
        end,                                                                                   \
        m,                                                                                     \
        n,                                                                                     \
        batch_count_C,                                                                         \
        offsets_batch_stride_A,                                                                \
        columns_values_batch_stride_A,                                                         \
        ldb,                                                                                   \
        batch_stride_B,                                                                        \
        ldc,                                                                                   \
        batch_stride_C,                                                                        \
        csr_row_ptr,                                                                           \
        csr_col_ind,                                                                           \
        csr_val,                                                                               \
        dense_B,                                                                               \
        dense_C,                                                                               \
        order_C,                                                                               \
        descr->base,                                                                           \
        conj_A,                                                                                \
        conj_B);

#define LAUNCH_CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWFS_COLUMNS(                           \
    CSRMMNT_DIM, WFSIZE, SUBWFSIZE, LOOPS, ...)                                                \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                        \
        (rocsparse::csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_kernel<CSRMMNT_DIM,   \
                                                                                WFSIZE,        \
                                                                                SUBWFSIZE,     \
                                                                                LOOPS,         \
                                                                                __VA_ARGS__>), \
        dim3((m - 1) / (CSRMMNT_DIM / WFSIZE) + 1, get_grid_size_y<J>(handle, batch_count_C)), \
        dim3(CSRMMNT_DIM),                                                                     \
        0,                                                                                     \
        handle->stream,                                                                        \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                          \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),                           \
        handle->pointer_mode == rocsparse_pointer_mode_host,                                   \
        start,                                                                                 \
        end,                                                                                   \
        m,                                                                                     \
        n,                                                                                     \
        batch_count_C,                                                                         \
        offsets_batch_stride_A,                                                                \
        columns_values_batch_stride_A,                                                         \
        csr_row_ptr,                                                                           \
        csr_col_ind,                                                                           \
        csr_val,                                                                               \
        dense_B,                                                                               \
        ldb,                                                                                   \
        batch_stride_B,                                                                        \
        dense_C,                                                                               \
        ldc,                                                                                   \
        batch_stride_C,                                                                        \
        order_C,                                                                               \
        descr->base,                                                                           \
        conj_A,                                                                                \
        conj_B);

#define LAUNCH_CSRMMNT_ROW_SPLIT_REMAINDER(CSRMMNT_DIM, WFSIZE, SUBWFSIZE)                      \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                         \
        (rocsparse::csrmmnt_row_split_shared_remainder_kernel<CSRMMNT_DIM, WFSIZE, SUBWFSIZE>), \
        dim3((m - 1) / (CSRMMNT_DIM / WFSIZE) + 1, get_grid_size_y<J>(handle, batch_count_C)),  \
        dim3(CSRMMNT_DIM),                                                                      \
        0,                                                                                      \
        handle->stream,                                                                         \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),                            \
        handle->pointer_mode == rocsparse_pointer_mode_host,                                    \
        conj_A,                                                                                 \
        conj_B,                                                                                 \
        start,                                                                                  \
        end,                                                                                    \
        m,                                                                                      \
        n,                                                                                      \
        batch_count_C,                                                                          \
        offsets_batch_stride_A,                                                                 \
        columns_values_batch_stride_A,                                                          \
        csr_row_ptr,                                                                            \
        csr_col_ind,                                                                            \
        csr_val,                                                                                \
        dense_B,                                                                                \
        ldb,                                                                                    \
        batch_stride_B,                                                                         \
        dense_C,                                                                                \
        ldc,                                                                                    \
        batch_stride_C,                                                                         \
        order_C,                                                                                \
        descr->base);

#define LAUNCH_CSRMMNT_ROW_SPLIT_GENERAL(CSRMMNT_DIM, WFSIZE, COL)                             \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                        \
        (rocsparse::csrmmnt_row_split_general_kernel<CSRMMNT_DIM, WFSIZE, COL>),               \
        dim3((m - 1) / (CSRMMNT_DIM / WFSIZE) + 1, get_grid_size_y<J>(handle, batch_count_C)), \
        dim3(CSRMMNT_DIM),                                                                     \
        0,                                                                                     \
        handle->stream,                                                                        \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                          \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),                           \
        handle->pointer_mode == rocsparse_pointer_mode_host,                                   \
        conj_A,                                                                                \
        conj_B,                                                                                \
        start,                                                                                 \
        m,                                                                                     \
        batch_count_C,                                                                         \
        offsets_batch_stride_A,                                                                \
        columns_values_batch_stride_A,                                                         \
        csr_row_ptr,                                                                           \
        csr_col_ind,                                                                           \
        csr_val,                                                                               \
        dense_B,                                                                               \
        ldb,                                                                                   \
        batch_stride_B,                                                                        \
        dense_C,                                                                               \
        ldc,                                                                                   \
        batch_stride_C,                                                                        \
        order_C,                                                                               \
        descr->base);

#define LAUNCH_CSRMMNN_ROW_SPLIT_GENERAL(CSRMMNN_DIM, WFSIZE, COL)                             \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                        \
        (rocsparse::csrmmnn_row_split_general_kernel<CSRMMNN_DIM, WFSIZE, COL>),               \
        dim3((m - 1) / (CSRMMNN_DIM / WFSIZE) + 1, get_grid_size_y<J>(handle, batch_count_C)), \
        dim3(CSRMMNN_DIM),                                                                     \
        0,                                                                                     \
        handle->stream,                                                                        \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                          \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),                           \
        handle->pointer_mode == rocsparse_pointer_mode_host,                                   \
        conj_A,                                                                                \
        conj_B,                                                                                \
        start,                                                                                 \
        m,                                                                                     \
        batch_count_C,                                                                         \
        offsets_batch_stride_A,                                                                \
        columns_values_batch_stride_A,                                                         \
        csr_row_ptr,                                                                           \
        csr_col_ind,                                                                           \
        csr_val,                                                                               \
        dense_B,                                                                               \
        ldb,                                                                                   \
        batch_stride_B,                                                                        \
        dense_C,                                                                               \
        ldc,                                                                                   \
        batch_stride_C,                                                                        \
        order_C,                                                                               \
        descr->base);

#define ROCSPARSE_CSRMM_TEMPLATE_ROW_SPLIT_IMPL(NAME) \
    NAME(handle,                                      \
         conj_A,                                      \
         conj_B,                                      \
         m,                                           \
         n,                                           \
         k,                                           \
         nnz,                                         \
         batch_count_A,                               \
         offsets_batch_stride_A,                      \
         columns_values_batch_stride_A,               \
         alpha_device_host,                           \
         descr,                                       \
         csr_val,                                     \
         csr_row_ptr,                                 \
         csr_col_ind,                                 \
         dense_B,                                     \
         ldb,                                         \
         batch_count_B,                               \
         batch_stride_B,                              \
         beta_device_host,                            \
         dense_C,                                     \
         ldc,                                         \
         batch_count_C,                               \
         batch_stride_C,                              \
         order_C,                                     \
         max_row_nnz);

// Same arguments as ROCSPARSE_CSRMM_TEMPLATE_ROW_SPLIT_IMPL, then the column
// range [start, end). The callee sets end to the first column it did not cover.
#define ROCSPARSE_CSRMMNT_ROW_SPLIT_RANGE(NAME) \
    NAME(handle,                                \
         conj_A,                                \
         conj_B,                                \
         m,                                     \
         n,                                     \
         k,                                     \
         nnz,                                   \
         batch_count_A,                         \
         offsets_batch_stride_A,                \
         columns_values_batch_stride_A,         \
         alpha_device_host,                     \
         descr,                                 \
         csr_val,                               \
         csr_row_ptr,                           \
         csr_col_ind,                           \
         dense_B,                               \
         ldb,                                   \
         batch_count_B,                         \
         batch_stride_B,                        \
         beta_device_host,                      \
         dense_C,                               \
         ldc,                                   \
         batch_count_C,                         \
         batch_stride_C,                        \
         order_C,                               \
         start,                                 \
         end)

    // Column count selects the general kernel. Block size 256, wavefront 8.
#define CSRMMNN_GENERAL_ENTRY(COL)                         \
    {                                                      \
        COL, csrmmnn_launch_general<I, J, A, B, C, T, COL> \
    }

    template <typename I, typename J, typename A, typename B, typename C, typename T, uint32_t COL>
    rocsparse_status csrmmnn_launch_general(rocsparse_handle          handle,
                                            const T*                  alpha_device_host,
                                            const T*                  beta_device_host,
                                            J                         start,
                                            J                         end,
                                            J                         m,
                                            J                         n,
                                            J                         batch_count_C,
                                            int64_t                   offsets_batch_stride_A,
                                            int64_t                   columns_values_batch_stride_A,
                                            int64_t                   ldb,
                                            int64_t                   batch_stride_B,
                                            int64_t                   ldc,
                                            int64_t                   batch_stride_C,
                                            const I*                  csr_row_ptr,
                                            const J*                  csr_col_ind,
                                            const A*                  csr_val,
                                            const B*                  dense_B,
                                            C*                        dense_C,
                                            rocsparse_order           order_C,
                                            const rocsparse_mat_descr descr,
                                            bool                      conj_A,
                                            bool                      conj_B)
    {
        LAUNCH_CSRMMNN_ROW_SPLIT_GENERAL(256, 8, COL);
        return rocsparse_status_success;
    }

    template <typename I, typename J, typename A, typename B, typename C, typename T>
    rocsparse_status csrmmnn_template_row_split(rocsparse_handle handle,
                                                bool             conj_A,
                                                bool             conj_B,
                                                J                m,
                                                J                n,
                                                J                k,
                                                I                nnz,
                                                J                batch_count_A,
                                                int64_t          offsets_batch_stride_A,
                                                int64_t          columns_values_batch_stride_A,
                                                const T*         alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const A*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const B*                  dense_B,
                                                int64_t                   ldb,
                                                J                         batch_count_B,
                                                int64_t                   batch_stride_B,
                                                const T*                  beta_device_host,
                                                C*                        dense_C,
                                                int64_t                   ldc,
                                                J                         batch_count_C,
                                                int64_t                   batch_stride_C,
                                                rocsparse_order           order_C,
                                                int64_t                   max_row_nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // One launch covers every column. COL is the exact column count, so each
        // width is its own kernel.
        if(n > 0 && n <= 40)
        {
            using launch_t = rocsparse_status (*)(rocsparse_handle handle,
                                                  const T*         alpha_device_host,
                                                  const T*         beta_device_host,
                                                  J                start,
                                                  J                end,
                                                  J                m,
                                                  J                n,
                                                  J                batch_count_C,
                                                  int64_t          offsets_batch_stride_A,
                                                  int64_t          columns_values_batch_stride_A,
                                                  int64_t          ldb,
                                                  int64_t          batch_stride_B,
                                                  int64_t          ldc,
                                                  int64_t          batch_stride_C,
                                                  const I*         csr_row_ptr,
                                                  const J*         csr_col_ind,
                                                  const A*         csr_val,
                                                  const B*         dense_B,
                                                  C*               dense_C,
                                                  rocsparse_order  order_C,
                                                  const rocsparse_mat_descr descr,
                                                  bool                      conj_A,
                                                  bool                      conj_B);

            static const std::map<int, launch_t> s_general_dispatch
                = {CSRMMNN_GENERAL_ENTRY(1),  CSRMMNN_GENERAL_ENTRY(2),  CSRMMNN_GENERAL_ENTRY(3),
                   CSRMMNN_GENERAL_ENTRY(4),  CSRMMNN_GENERAL_ENTRY(5),  CSRMMNN_GENERAL_ENTRY(6),
                   CSRMMNN_GENERAL_ENTRY(7),  CSRMMNN_GENERAL_ENTRY(8),  CSRMMNN_GENERAL_ENTRY(9),
                   CSRMMNN_GENERAL_ENTRY(10), CSRMMNN_GENERAL_ENTRY(11), CSRMMNN_GENERAL_ENTRY(12),
                   CSRMMNN_GENERAL_ENTRY(13), CSRMMNN_GENERAL_ENTRY(14), CSRMMNN_GENERAL_ENTRY(15),
                   CSRMMNN_GENERAL_ENTRY(16), CSRMMNN_GENERAL_ENTRY(17), CSRMMNN_GENERAL_ENTRY(18),
                   CSRMMNN_GENERAL_ENTRY(19), CSRMMNN_GENERAL_ENTRY(20), CSRMMNN_GENERAL_ENTRY(21),
                   CSRMMNN_GENERAL_ENTRY(22), CSRMMNN_GENERAL_ENTRY(23), CSRMMNN_GENERAL_ENTRY(24),
                   CSRMMNN_GENERAL_ENTRY(25), CSRMMNN_GENERAL_ENTRY(26), CSRMMNN_GENERAL_ENTRY(27),
                   CSRMMNN_GENERAL_ENTRY(28), CSRMMNN_GENERAL_ENTRY(29), CSRMMNN_GENERAL_ENTRY(30),
                   CSRMMNN_GENERAL_ENTRY(31), CSRMMNN_GENERAL_ENTRY(32), CSRMMNN_GENERAL_ENTRY(33),
                   CSRMMNN_GENERAL_ENTRY(34), CSRMMNN_GENERAL_ENTRY(35), CSRMMNN_GENERAL_ENTRY(36),
                   CSRMMNN_GENERAL_ENTRY(37), CSRMMNN_GENERAL_ENTRY(38), CSRMMNN_GENERAL_ENTRY(39),
                   CSRMMNN_GENERAL_ENTRY(40)};

            const auto it = s_general_dispatch.find(static_cast<int>(n));
            if(it == s_general_dispatch.end())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }

            RETURN_IF_ROCSPARSE_ERROR(it->second(handle,
                                                 alpha_device_host,
                                                 beta_device_host,
                                                 0,
                                                 n,
                                                 m,
                                                 n,
                                                 batch_count_C,
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
                                                 descr,
                                                 conj_A,
                                                 conj_B));
        }
        else
        {
#define CSRMMNN_DIM 256
#define SUB_WF_SIZE 8
            J start = 0;
            J end   = n - n % 8;

            if((end - start) > 0)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrmmnn_row_split_kernel<CSRMMNN_DIM, SUB_WF_SIZE, 8>),
                    dim3((m - 1) / (CSRMMNN_DIM / SUB_WF_SIZE) + 1,
                         get_grid_size_y<J>(handle, ((end - start) - 1) / 8 + 1),
                         get_grid_size_z<J>(handle, batch_count_C)),
                    dim3(CSRMMNN_DIM),
                    0,
                    handle->stream,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    handle->pointer_mode == rocsparse_pointer_mode_host,
                    conj_A,
                    conj_B,
                    start,
                    m,
                    n,
                    batch_count_C,
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
                    descr->base);
            }

            start = end;
            end   = n;

            if((end - start) > 0)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrmmnn_row_split_kernel<CSRMMNN_DIM, SUB_WF_SIZE, 1>),
                    dim3((m - 1) / (CSRMMNN_DIM / SUB_WF_SIZE) + 1,
                         get_grid_size_y<J>(handle, ((end - start) - 1) / 1 + 1),
                         get_grid_size_z<J>(handle, batch_count_C)),
                    dim3(CSRMMNN_DIM),
                    0,
                    handle->stream,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    handle->pointer_mode == rocsparse_pointer_mode_host,
                    conj_A,
                    conj_B,
                    start,
                    m,
                    n,
                    batch_count_C,
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
                    descr->base);
            }
#undef SUB_WF_SIZE
#undef CSRMMNN_DIM
        }

        return rocsparse_status_success;
    }

#undef CSRMMNN_GENERAL_ENTRY

    // Width, then the sub-wavefront loop count and any 4/2/1 tail groups.
#define CSRMMNT_PANEL_ENTRY(WIDTH, ...)                            \
    {                                                              \
        WIDTH, csrmmnt_launch_panel<I, J, A, B, C, T, __VA_ARGS__> \
    }

    template <typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T,
              uint32_t LOOPS,
              uint32_t... TAIL>
    rocsparse_status csrmmnt_launch_panel(rocsparse_handle          handle,
                                          const T*                  alpha_device_host,
                                          const T*                  beta_device_host,
                                          J                         start,
                                          J                         end,
                                          J                         m,
                                          J                         n,
                                          J                         batch_count_C,
                                          int64_t                   offsets_batch_stride_A,
                                          int64_t                   columns_values_batch_stride_A,
                                          int64_t                   ldb,
                                          int64_t                   batch_stride_B,
                                          int64_t                   ldc,
                                          int64_t                   batch_stride_C,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const A*                  csr_val,
                                          const B*                  dense_B,
                                          C*                        dense_C,
                                          rocsparse_order           order_C,
                                          const rocsparse_mat_descr descr,
                                          bool                      conj_A,
                                          bool                      conj_B)
    {
        if constexpr(sizeof...(TAIL) == 0)
        {
            LAUNCH_CSRMMNT_ROW_SPLIT_SHARED_SUBWFSIZE_X_LOOP_COLUMNS(256, 16, 8, LOOPS);
        }
        else
        {
            LAUNCH_CSRMMNT_ROW_SPLIT_SUBWFSIZE_X_LOOP_PLUS_SWFS_COLUMNS(256, 16, 8, LOOPS, TAIL...);
        }

        return rocsparse_status_success;
    }

    // Column 16 is the 8*2 sub-wavefront kernel. Every other width from 1 to 40
    // is the general kernel.
    template <uint32_t WF_SIZE,
              uint32_t COL,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    rocsparse_status csrmmnt_launch_cols(rocsparse_handle          handle,
                                         const T*                  alpha_device_host,
                                         const T*                  beta_device_host,
                                         J                         start,
                                         J                         end,
                                         J                         m,
                                         J                         n,
                                         J                         batch_count_C,
                                         int64_t                   offsets_batch_stride_A,
                                         int64_t                   columns_values_batch_stride_A,
                                         int64_t                   ldb,
                                         int64_t                   batch_stride_B,
                                         int64_t                   ldc,
                                         int64_t                   batch_stride_C,
                                         const I*                  csr_row_ptr,
                                         const J*                  csr_col_ind,
                                         const A*                  csr_val,
                                         const B*                  dense_B,
                                         C*                        dense_C,
                                         rocsparse_order           order_C,
                                         const rocsparse_mat_descr descr,
                                         bool                      conj_A,
                                         bool                      conj_B)
    {
        if constexpr(COL == 16)
        {
            LAUNCH_CSRMMNT_ROW_SPLIT_SHARED_SUBWFSIZE_X_LOOP_COLUMNS(256, WF_SIZE, 8, 2);
        }
        else
        {
            LAUNCH_CSRMMNT_ROW_SPLIT_GENERAL(256, WF_SIZE, COL);
        }

        return rocsparse_status_success;
    }

#define CSRMMNT_COLS_ENTRY(COL)                                  \
    {                                                            \
        COL, csrmmnt_launch_cols<WF_SIZE, COL, I, J, A, B, C, T> \
    }

    // One panel of 17 or more columns. The wavefront is fixed at 16.
    // On return, end is the first column this panel does not cover.
    template <typename I, typename J, typename A, typename B, typename C, typename T>
    rocsparse_status csrmmnt_template_row_split_columns_gt16(rocsparse_handle handle,
                                                             bool             conj_A,
                                                             bool             conj_B,
                                                             J                m,
                                                             J                n,
                                                             J                k,
                                                             I                nnz,
                                                             J                batch_count_A,
                                                             int64_t  offsets_batch_stride_A,
                                                             int64_t  columns_values_batch_stride_A,
                                                             const T* alpha_device_host,
                                                             const rocsparse_mat_descr descr,
                                                             const A*                  csr_val,
                                                             const I*                  csr_row_ptr,
                                                             const J*                  csr_col_ind,
                                                             const B*                  dense_B,
                                                             int64_t                   ldb,
                                                             J               batch_count_B,
                                                             int64_t         batch_stride_B,
                                                             const T*        beta_device_host,
                                                             C*              dense_C,
                                                             int64_t         ldc,
                                                             J               batch_count_C,
                                                             int64_t         batch_stride_C,
                                                             rocsparse_order order_C,
                                                             J               start,
                                                             J&              end)
    {
        ROCSPARSE_ROUTINE_TRACE;

        using launch_t = rocsparse_status (*)(rocsparse_handle handle,
                                              const T*         alpha_device_host,
                                              const T*         beta_device_host,
                                              J                start,
                                              J                end,
                                              J                m,
                                              J                n,
                                              J                batch_count_C,
                                              int64_t          offsets_batch_stride_A,
                                              int64_t          columns_values_batch_stride_A,
                                              int64_t          ldb,
                                              int64_t          batch_stride_B,
                                              int64_t          ldc,
                                              int64_t          batch_stride_C,
                                              const I*         csr_row_ptr,
                                              const J*         csr_col_ind,
                                              const A*         csr_val,
                                              const B*         dense_B,
                                              C*               dense_C,
                                              rocsparse_order  order_C,
                                              const rocsparse_mat_descr descr,
                                              bool                      conj_A,
                                              bool                      conj_B);

        // Kernel dispatch table for panels of 17 or more columns.
        // A pure multiple of 8 has no tail. Any other width adds 4, 2, and 1.
        static const std::map<int, launch_t> s_panel_dispatch
            = {CSRMMNT_PANEL_ENTRY(64, 8),       CSRMMNT_PANEL_ENTRY(63, 7, 4, 2, 1),
               CSRMMNT_PANEL_ENTRY(62, 7, 4, 2), CSRMMNT_PANEL_ENTRY(61, 7, 4, 1),
               CSRMMNT_PANEL_ENTRY(60, 7, 4),    CSRMMNT_PANEL_ENTRY(59, 7, 2, 1),
               CSRMMNT_PANEL_ENTRY(58, 7, 2),    CSRMMNT_PANEL_ENTRY(57, 7, 1),
               CSRMMNT_PANEL_ENTRY(56, 7),       CSRMMNT_PANEL_ENTRY(55, 6, 4, 2, 1),
               CSRMMNT_PANEL_ENTRY(54, 6, 4, 2), CSRMMNT_PANEL_ENTRY(53, 6, 4, 1),
               CSRMMNT_PANEL_ENTRY(52, 6, 4),    CSRMMNT_PANEL_ENTRY(51, 6, 2, 1),
               CSRMMNT_PANEL_ENTRY(50, 6, 2),    CSRMMNT_PANEL_ENTRY(49, 6, 1),
               CSRMMNT_PANEL_ENTRY(48, 6),       CSRMMNT_PANEL_ENTRY(47, 5, 4, 2, 1),
               CSRMMNT_PANEL_ENTRY(46, 5, 4, 2), CSRMMNT_PANEL_ENTRY(45, 5, 4, 1),
               CSRMMNT_PANEL_ENTRY(44, 5, 4),    CSRMMNT_PANEL_ENTRY(43, 5, 2, 1),
               CSRMMNT_PANEL_ENTRY(42, 5, 2),    CSRMMNT_PANEL_ENTRY(41, 5, 1),
               CSRMMNT_PANEL_ENTRY(40, 5),       CSRMMNT_PANEL_ENTRY(39, 4, 4, 2, 1),
               CSRMMNT_PANEL_ENTRY(38, 4, 4, 2), CSRMMNT_PANEL_ENTRY(37, 4, 4, 1),
               CSRMMNT_PANEL_ENTRY(36, 4, 4),    CSRMMNT_PANEL_ENTRY(35, 4, 2, 1),
               CSRMMNT_PANEL_ENTRY(34, 4, 2),    CSRMMNT_PANEL_ENTRY(33, 4, 1),
               CSRMMNT_PANEL_ENTRY(32, 4),       CSRMMNT_PANEL_ENTRY(31, 3, 4, 2, 1),
               CSRMMNT_PANEL_ENTRY(30, 3, 4, 2), CSRMMNT_PANEL_ENTRY(29, 3, 4, 1),
               CSRMMNT_PANEL_ENTRY(28, 3, 4),    CSRMMNT_PANEL_ENTRY(27, 3, 2, 1),
               CSRMMNT_PANEL_ENTRY(26, 3, 2),    CSRMMNT_PANEL_ENTRY(25, 3, 1),
               CSRMMNT_PANEL_ENTRY(24, 3),       CSRMMNT_PANEL_ENTRY(23, 2, 4, 2, 1),
               CSRMMNT_PANEL_ENTRY(22, 2, 4, 2), CSRMMNT_PANEL_ENTRY(21, 2, 4, 1),
               CSRMMNT_PANEL_ENTRY(20, 2, 4),    CSRMMNT_PANEL_ENTRY(19, 2, 2, 1),
               CSRMMNT_PANEL_ENTRY(18, 2, 2),    CSRMMNT_PANEL_ENTRY(17, 2, 1)};

        const J    num_cols = end - start;
        const int  width    = (num_cols >= 64) ? 64 : static_cast<int>(num_cols);
        const auto it       = s_panel_dispatch.find(width);

        if(it == s_panel_dispatch.end())
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        end = start + (num_cols - (num_cols % static_cast<J>(width)));
        RETURN_IF_ROCSPARSE_ERROR(it->second(handle,
                                             alpha_device_host,
                                             beta_device_host,
                                             start,
                                             end,
                                             m,
                                             n,
                                             batch_count_C,
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
                                             descr,
                                             conj_A,
                                             conj_B));

        return rocsparse_status_success;
    }

    // One panel of 1-40 columns. WF_SIZE is the row-length wavefront.
    // On return, end is the first column this panel does not cover.
    template <uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    rocsparse_status csrmmnt_template_row_split_columns_le16(rocsparse_handle handle,
                                                             bool             conj_A,
                                                             bool             conj_B,
                                                             J                m,
                                                             J                n,
                                                             J                k,
                                                             I                nnz,
                                                             J                batch_count_A,
                                                             int64_t  offsets_batch_stride_A,
                                                             int64_t  columns_values_batch_stride_A,
                                                             const T* alpha_device_host,
                                                             const rocsparse_mat_descr descr,
                                                             const A*                  csr_val,
                                                             const I*                  csr_row_ptr,
                                                             const J*                  csr_col_ind,
                                                             const B*                  dense_B,
                                                             int64_t                   ldb,
                                                             J               batch_count_B,
                                                             int64_t         batch_stride_B,
                                                             const T*        beta_device_host,
                                                             C*              dense_C,
                                                             int64_t         ldc,
                                                             J               batch_count_C,
                                                             int64_t         batch_stride_C,
                                                             rocsparse_order order_C,
                                                             J               start,
                                                             J&              end)
    {
        ROCSPARSE_ROUTINE_TRACE;

        using launch_t = rocsparse_status (*)(rocsparse_handle handle,
                                              const T*         alpha_device_host,
                                              const T*         beta_device_host,
                                              J                start,
                                              J                end,
                                              J                m,
                                              J                n,
                                              J                batch_count_C,
                                              int64_t          offsets_batch_stride_A,
                                              int64_t          columns_values_batch_stride_A,
                                              int64_t          ldb,
                                              int64_t          batch_stride_B,
                                              int64_t          ldc,
                                              int64_t          batch_stride_C,
                                              const I*         csr_row_ptr,
                                              const J*         csr_col_ind,
                                              const A*         csr_val,
                                              const B*         dense_B,
                                              C*               dense_C,
                                              rocsparse_order  order_C,
                                              const rocsparse_mat_descr descr,
                                              bool                      conj_A,
                                              bool                      conj_B);

        // Kernel dispatch table for the remaining 1-40 columns.
        static const std::map<int, launch_t> s_cols_dispatch
            = {CSRMMNT_COLS_ENTRY(40), CSRMMNT_COLS_ENTRY(39), CSRMMNT_COLS_ENTRY(38),
               CSRMMNT_COLS_ENTRY(37), CSRMMNT_COLS_ENTRY(36), CSRMMNT_COLS_ENTRY(35),
               CSRMMNT_COLS_ENTRY(34), CSRMMNT_COLS_ENTRY(33), CSRMMNT_COLS_ENTRY(32),
               CSRMMNT_COLS_ENTRY(31), CSRMMNT_COLS_ENTRY(30), CSRMMNT_COLS_ENTRY(29),
               CSRMMNT_COLS_ENTRY(28), CSRMMNT_COLS_ENTRY(27), CSRMMNT_COLS_ENTRY(26),
               CSRMMNT_COLS_ENTRY(25), CSRMMNT_COLS_ENTRY(24), CSRMMNT_COLS_ENTRY(23),
               CSRMMNT_COLS_ENTRY(22), CSRMMNT_COLS_ENTRY(21), CSRMMNT_COLS_ENTRY(20),
               CSRMMNT_COLS_ENTRY(19), CSRMMNT_COLS_ENTRY(18), CSRMMNT_COLS_ENTRY(17),
               CSRMMNT_COLS_ENTRY(16), CSRMMNT_COLS_ENTRY(15), CSRMMNT_COLS_ENTRY(14),
               CSRMMNT_COLS_ENTRY(13), CSRMMNT_COLS_ENTRY(12), CSRMMNT_COLS_ENTRY(11),
               CSRMMNT_COLS_ENTRY(10), CSRMMNT_COLS_ENTRY(9),  CSRMMNT_COLS_ENTRY(8),
               CSRMMNT_COLS_ENTRY(7),  CSRMMNT_COLS_ENTRY(6),  CSRMMNT_COLS_ENTRY(5),
               CSRMMNT_COLS_ENTRY(4),  CSRMMNT_COLS_ENTRY(3),  CSRMMNT_COLS_ENTRY(2),
               CSRMMNT_COLS_ENTRY(1)};

        const J    num_cols = end - start;
        const int  width    = static_cast<int>(num_cols);
        const auto it       = s_cols_dispatch.find(width);

        if(it == s_cols_dispatch.end())
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        end = start + (num_cols - (num_cols % static_cast<J>(width)));
        RETURN_IF_ROCSPARSE_ERROR(it->second(handle,
                                             alpha_device_host,
                                             beta_device_host,
                                             start,
                                             end,
                                             m,
                                             n,
                                             batch_count_C,
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
                                             descr,
                                             conj_A,
                                             conj_B));

        return rocsparse_status_success;
    }

#undef CSRMMNT_COLS_ENTRY
#undef CSRMMNT_PANEL_ENTRY

    template <typename I, typename J, typename A, typename B, typename C, typename T>
    rocsparse_status csrmmnt_template_row_split(rocsparse_handle handle,
                                                bool             conj_A,
                                                bool             conj_B,
                                                J                m,
                                                J                n,
                                                J                k,
                                                I                nnz,
                                                J                batch_count_A,
                                                int64_t          offsets_batch_stride_A,
                                                int64_t          columns_values_batch_stride_A,
                                                const T*         alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const A*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const B*                  dense_B,
                                                int64_t                   ldb,
                                                J                         batch_count_B,
                                                int64_t                   batch_stride_B,
                                                const T*                  beta_device_host,
                                                C*                        dense_C,
                                                int64_t                   ldc,
                                                J                         batch_count_C,
                                                int64_t                   batch_stride_C,
                                                rocsparse_order           order_C,
                                                int64_t                   max_row_nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        J start = 0;
        J end   = n;

        while((end - start) > 0)
        {
            if((end - start) > 40)
            {
                RETURN_IF_ROCSPARSE_ERROR(ROCSPARSE_CSRMMNT_ROW_SPLIT_RANGE(
                    rocsparse::csrmmnt_template_row_split_columns_gt16));
            }
            else
            {
                // RETURN_IF_ROCSPARSE_ERROR(ROCSPARSE_CSRMMNT_ROW_SPLIT_RANGE(
                //     rocsparse::csrmmnt_template_row_split_columns_le16<32>));

                const I nnz_per_row = nnz / m;

                // Wavefront for the 1-40 column kernels. Pick the largest of 8, 16, 32
                // whose rounding-up of the row length equals the smallest rounding, and
                // that is no wider than the row. Past 32 the choice repeats: r in 1-8 or
                // 17-24 -> 8, r in 9-16 -> 16, otherwise 32. Rows of 1-15 stay at 8, and
                // 25-31 stay at 16, because a wider group cannot be filled.
                const I r = nnz_per_row % 32;

                int wfsize = 32;
                if(nnz_per_row < 16 || (r >= 17 && r <= 24)
                   || (nnz_per_row >= 32 && r >= 1 && r <= 8))
                {
                    wfsize = 8;
                }
                else if(nnz_per_row < 32 || (r >= 9 && r <= 16))
                {
                    wfsize = 16;
                }

                // A small matrix launches too few blocks at a narrow wavefront.
                // Widen it when the longest row can fill the wider group. Uniform
                // short rows stay on the average-nnz choice. max_row_nnz is 0
                // when no line profile was cached.
                if(wfsize < 32 && max_row_nnz > 0 && m > 0)
                {
                    constexpr int dim        = 256;
                    const int64_t nblocks    = (static_cast<int64_t>(m) - 1) / (dim / wfsize) + 1;
                    const int     maxthreads = handle->properties.maxThreadsPerBlock;
                    const int     nprocs     = 2 * handle->properties.multiProcessorCount;
                    const int minblocks = (static_cast<int64_t>(nprocs) * maxthreads - 1) / dim + 1;

                    if(nblocks < minblocks)
                    {
                        const int64_t threads_per_row
                            = static_cast<int64_t>(dim) * minblocks / static_cast<int64_t>(m);
                        int raised = wfsize;
                        if(threads_per_row >= 32)
                        {
                            raised = 32;
                        }
                        else if(threads_per_row >= 16)
                        {
                            raised = 16;
                        }

                        const int wavefront = handle->wavefront_size;
                        if(raised > wavefront)
                        {
                            raised = wavefront;
                        }
                        if(raised > 32)
                        {
                            raised = 32;
                        }

                        if(raised > wfsize && max_row_nnz >= raised)
                        {
                            wfsize = raised;
                        }
                    }
                }

                if(wfsize == 8)
                {
                    RETURN_IF_ROCSPARSE_ERROR(ROCSPARSE_CSRMMNT_ROW_SPLIT_RANGE(
                        rocsparse::csrmmnt_template_row_split_columns_le16<8>));
                }
                else if(wfsize == 16)
                {
                    RETURN_IF_ROCSPARSE_ERROR(ROCSPARSE_CSRMMNT_ROW_SPLIT_RANGE(
                        rocsparse::csrmmnt_template_row_split_columns_le16<16>));
                }
                else
                {
                    RETURN_IF_ROCSPARSE_ERROR(ROCSPARSE_CSRMMNT_ROW_SPLIT_RANGE(
                        rocsparse::csrmmnt_template_row_split_columns_le16<32>));
                }
            }

            start = end;
            end   = n;
        }

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename A, typename B, typename C, typename T>
    rocsparse_status csrmmtn_template_row_split(rocsparse_handle handle,
                                                bool             conj_A,
                                                bool             conj_B,
                                                J                m,
                                                J                n,
                                                J                k,
                                                I                nnz,
                                                J                batch_count_A,
                                                int64_t          offsets_batch_stride_A,
                                                int64_t          columns_values_batch_stride_A,
                                                const T*         alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const A*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const B*                  dense_B,
                                                int64_t                   ldb,
                                                J                         batch_count_B,
                                                int64_t                   batch_stride_B,
                                                const T*                  beta_device_host,
                                                C*                        dense_C,
                                                int64_t                   ldc,
                                                J                         batch_count_C,
                                                int64_t                   batch_stride_C,
                                                rocsparse_order           order_C,
                                                int64_t                   max_row_nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Scale C with beta
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_2d_array(
            handle, k, n, ldc, batch_count_C, batch_stride_C, beta_device_host, dense_C, order_C));

#define CSRMMTN_DIM 256
#define WF_SIZE 4
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrmmtn_row_split_kernel<CSRMMTN_DIM, WF_SIZE>),
            dim3((m - 1) / (CSRMMTN_DIM / WF_SIZE) + 1,
                 get_grid_size_y<J>(handle, (n - 1) / WF_SIZE + 1),
                 get_grid_size_z<J>(handle, batch_count_C)),
            dim3(CSRMMTN_DIM),
            0,
            handle->stream,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            handle->pointer_mode == rocsparse_pointer_mode_host,
            conj_A,
            conj_B,
            m,
            n,
            k,
            batch_count_C,
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
            descr->base);

#undef CSRMMTN_DIM
#undef WF_SIZE

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename A, typename B, typename C, typename T>
    rocsparse_status csrmmtt_template_row_split(rocsparse_handle handle,
                                                bool             conj_A,
                                                bool             conj_B,
                                                J                m,
                                                J                n,
                                                J                k,
                                                I                nnz,
                                                J                batch_count_A,
                                                int64_t          offsets_batch_stride_A,
                                                int64_t          columns_values_batch_stride_A,
                                                const T*         alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const A*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const B*                  dense_B,
                                                int64_t                   ldb,
                                                J                         batch_count_B,
                                                int64_t                   batch_stride_B,
                                                const T*                  beta_device_host,
                                                C*                        dense_C,
                                                int64_t                   ldc,
                                                J                         batch_count_C,
                                                int64_t                   batch_stride_C,
                                                rocsparse_order           order_C,
                                                int64_t                   max_row_nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Scale C with beta
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_2d_array(
            handle, k, n, ldc, batch_count_C, batch_stride_C, beta_device_host, dense_C, order_C));

#define CSRMMTT_DIM 256
#define WF_SIZE 4
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrmmtt_row_split_kernel<CSRMMTT_DIM, WF_SIZE>),
            dim3((m - 1) / (CSRMMTT_DIM / WF_SIZE) + 1,
                 get_grid_size_y<J>(handle, (n - 1) / WF_SIZE + 1),
                 get_grid_size_z<J>(handle, batch_count_C)),
            dim3(CSRMMTT_DIM),
            0,
            handle->stream,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            handle->pointer_mode == rocsparse_pointer_mode_host,
            conj_A,
            conj_B,
            m,
            n,
            k,
            batch_count_C,
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
            descr->base);
#undef CSRMMTT_DIM
#undef WF_SIZE

        return rocsparse_status_success;
    }

#define LAUNCH_CSRMM_ROW_SPLIT_GENERAL(CSRMM_DIM, WFSIZE, COL)                                  \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                         \
        (rocsparse::csrmm_row_split_general_kernel<CSRMM_DIM, WFSIZE, COL>),                    \
        dim3((m - 1) / (CSRMM_DIM / WFSIZE) + 1, get_grid_size_y<J>(handle, batch_count_C)),    \
        dim3(CSRMM_DIM),                                                                        \
        0,                                                                                      \
        handle->stream,                                                                         \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),                            \
        handle->pointer_mode == rocsparse_pointer_mode_host,                                    \
        conj_A,                                                                                 \
        conj_B,                                                                                 \
        order_B,                                                                                \
        trans_B,                                                                                \
        start,                                                                                  \
        m,                                                                                      \
        batch_count_C,                                                                          \
        offsets_batch_stride_A,                                                                 \
        columns_values_batch_stride_A,                                                          \
        csr_row_ptr,                                                                            \
        csr_col_ind,                                                                            \
        csr_val,                                                                                \
        dense_B,                                                                                \
        ldb,                                                                                    \
        batch_stride_B,                                                                         \
        dense_C,                                                                                \
        ldc,                                                                                    \
        batch_stride_C,                                                                         \
        order_C,                                                                                \
        descr->base);

#define CSRMM_GENERAL_ENTRY(COL) {COL, csrmm_launch_general<I, J, A, B, C, T, COL>}

    // One exact column count of the combined NN/NT general kernel. Block 256, wavefront 8.
    template <typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T,
              uint32_t COL>
    rocsparse_status csrmm_launch_general(rocsparse_handle          handle,
                                          const T*                  alpha_device_host,
                                          const T*                  beta_device_host,
                                          J                         start,
                                          J                         end,
                                          J                         m,
                                          J                         n,
                                          J                         batch_count_C,
                                          int64_t                   offsets_batch_stride_A,
                                          int64_t                   columns_values_batch_stride_A,
                                          int64_t                   ldb,
                                          int64_t                   batch_stride_B,
                                          int64_t                   ldc,
                                          int64_t                   batch_stride_C,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const A*                  csr_val,
                                          const B*                  dense_B,
                                          C*                        dense_C,
                                          rocsparse_order           order_C,
                                          const rocsparse_mat_descr descr,
                                          bool                      conj_A,
                                          bool                      conj_B,
                                          rocsparse_order           order_B,
                                          rocsparse_operation       trans_B)
    {
        LAUNCH_CSRMM_ROW_SPLIT_GENERAL(256, 8, COL);
        return rocsparse_status_success;
    }

    // NN and NT, trans_A == none. Columns are covered by the combined general kernel,
    // 64 at a time, with a shorter kernel for a trailing remainder.
    template <typename I, typename J, typename A, typename B, typename C, typename T>
    rocsparse_status csrmm_template_row_split_general(rocsparse_handle          handle,
                                                      bool                      conj_A,
                                                      bool                      conj_B,
                                                      J                         m,
                                                      J                         n,
                                                      J                         k,
                                                      I                         nnz,
                                                      J                         batch_count_A,
                                                      int64_t                   offsets_batch_stride_A,
                                                      int64_t                   columns_values_batch_stride_A,
                                                      const T*                  alpha_device_host,
                                                      const rocsparse_mat_descr descr,
                                                      const A*                  csr_val,
                                                      const I*                  csr_row_ptr,
                                                      const J*                  csr_col_ind,
                                                      const B*                  dense_B,
                                                      int64_t                   ldb,
                                                      J                         batch_count_B,
                                                      int64_t                   batch_stride_B,
                                                      rocsparse_order           order_B,
                                                      rocsparse_operation       trans_B,
                                                      const T*                  beta_device_host,
                                                      C*                        dense_C,
                                                      int64_t                   ldc,
                                                      J                         batch_count_C,
                                                      int64_t                   batch_stride_C,
                                                      rocsparse_order           order_C,
                                                      int64_t                   max_row_nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        using launch_t = rocsparse_status (*)(rocsparse_handle,
                                              const T*,
                                              const T*,
                                              J,
                                              J,
                                              J,
                                              J,
                                              J,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              const I*,
                                              const J*,
                                              const A*,
                                              const B*,
                                              C*,
                                              rocsparse_order,
                                              const rocsparse_mat_descr,
                                              bool,
                                              bool,
                                              rocsparse_order,
                                              rocsparse_operation);

        static const std::map<int, launch_t> s_general_dispatch = {
            CSRMM_GENERAL_ENTRY(1),  CSRMM_GENERAL_ENTRY(2),  CSRMM_GENERAL_ENTRY(3),
            CSRMM_GENERAL_ENTRY(4),  CSRMM_GENERAL_ENTRY(5),  CSRMM_GENERAL_ENTRY(6),
            CSRMM_GENERAL_ENTRY(7),  CSRMM_GENERAL_ENTRY(8),  CSRMM_GENERAL_ENTRY(9),
            CSRMM_GENERAL_ENTRY(10), CSRMM_GENERAL_ENTRY(11), CSRMM_GENERAL_ENTRY(12),
            CSRMM_GENERAL_ENTRY(13), CSRMM_GENERAL_ENTRY(14), CSRMM_GENERAL_ENTRY(15),
            CSRMM_GENERAL_ENTRY(16), CSRMM_GENERAL_ENTRY(17), CSRMM_GENERAL_ENTRY(18),
            CSRMM_GENERAL_ENTRY(19), CSRMM_GENERAL_ENTRY(20), CSRMM_GENERAL_ENTRY(21),
            CSRMM_GENERAL_ENTRY(22), CSRMM_GENERAL_ENTRY(23), CSRMM_GENERAL_ENTRY(24),
            CSRMM_GENERAL_ENTRY(25), CSRMM_GENERAL_ENTRY(26), CSRMM_GENERAL_ENTRY(27),
            CSRMM_GENERAL_ENTRY(28), CSRMM_GENERAL_ENTRY(29), CSRMM_GENERAL_ENTRY(30),
            CSRMM_GENERAL_ENTRY(31), CSRMM_GENERAL_ENTRY(32), CSRMM_GENERAL_ENTRY(33),
            CSRMM_GENERAL_ENTRY(34), CSRMM_GENERAL_ENTRY(35), CSRMM_GENERAL_ENTRY(36),
            CSRMM_GENERAL_ENTRY(37), CSRMM_GENERAL_ENTRY(38), CSRMM_GENERAL_ENTRY(39),
            CSRMM_GENERAL_ENTRY(40), CSRMM_GENERAL_ENTRY(41), CSRMM_GENERAL_ENTRY(42),
            CSRMM_GENERAL_ENTRY(43), CSRMM_GENERAL_ENTRY(44), CSRMM_GENERAL_ENTRY(45),
            CSRMM_GENERAL_ENTRY(46), CSRMM_GENERAL_ENTRY(47), CSRMM_GENERAL_ENTRY(48),
            CSRMM_GENERAL_ENTRY(49), CSRMM_GENERAL_ENTRY(50), CSRMM_GENERAL_ENTRY(51),
            CSRMM_GENERAL_ENTRY(52), CSRMM_GENERAL_ENTRY(53), CSRMM_GENERAL_ENTRY(54),
            CSRMM_GENERAL_ENTRY(55), CSRMM_GENERAL_ENTRY(56), CSRMM_GENERAL_ENTRY(57),
            CSRMM_GENERAL_ENTRY(58), CSRMM_GENERAL_ENTRY(59), CSRMM_GENERAL_ENTRY(60),
            CSRMM_GENERAL_ENTRY(61), CSRMM_GENERAL_ENTRY(62), CSRMM_GENERAL_ENTRY(63),
            CSRMM_GENERAL_ENTRY(64)};

        J col = 0;
        while(col < n)
        {
            const J   remain = n - col;
            const int width  = remain >= 64 ? 64 : static_cast<int>(remain);
            const auto it    = s_general_dispatch.find(width);
            if(it == s_general_dispatch.end())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }

            RETURN_IF_ROCSPARSE_ERROR(it->second(handle,
                                                 alpha_device_host,
                                                 beta_device_host,
                                                 col,
                                                 static_cast<J>(col + width),
                                                 m,
                                                 n,
                                                 batch_count_C,
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
                                                 descr,
                                                 conj_A,
                                                 conj_B,
                                                 order_B,
                                                 trans_B));
            col = static_cast<J>(col + width);
        }

        return rocsparse_status_success;
    }

#undef CSRMM_GENERAL_ENTRY
#undef LAUNCH_CSRMM_ROW_SPLIT_GENERAL

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template_row_split(rocsparse_handle    handle,
                                              rocsparse_operation trans_A,
                                              rocsparse_operation trans_B,
                                              J                   m,
                                              J                   n,
                                              J                   k,
                                              I                   nnz,
                                              J                   batch_count_A,
                                              int64_t             offsets_batch_stride_A,
                                              int64_t             columns_values_batch_stride_A,
                                              const T*            alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const A*                  csr_val,
                                              const I*                  csr_row_ptr,
                                              const J*                  csr_col_ind,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              J                         batch_count_B,
                                              int64_t                   batch_stride_B,
                                              rocsparse_order           order_B,
                                              const T*                  beta_device_host,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              J                         batch_count_C,
                                              int64_t                   batch_stride_C,
                                              rocsparse_order           order_C,
                                              bool                      force_conj_A,
                                              int64_t                   max_row_nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose || force_conj_A);
        const bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);
        // Run different csrmv kernels
        if(trans_A == rocsparse_operation_none)
        {
            return csrmm_template_row_split_general(handle,
                                                    conj_A,
                                                    conj_B,
                                                    m,
                                                    n,
                                                    k,
                                                    nnz,
                                                    batch_count_A,
                                                    offsets_batch_stride_A,
                                                    columns_values_batch_stride_A,
                                                    alpha_device_host,
                                                    descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    dense_B,
                                                    ldb,
                                                    batch_count_B,
                                                    batch_stride_B,
                                                    order_B,
                                                    trans_B,
                                                    beta_device_host,
                                                    dense_C,
                                                    ldc,
                                                    batch_count_C,
                                                    batch_stride_C,
                                                    order_C,
                                                    max_row_nnz);
        }
        else
        {
            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                return ROCSPARSE_CSRMM_TEMPLATE_ROW_SPLIT_IMPL(
                    rocsparse::csrmmtn_template_row_split);
            }
            else if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                return ROCSPARSE_CSRMM_TEMPLATE_ROW_SPLIT_IMPL(
                    rocsparse::csrmmtt_template_row_split);
            }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}
#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE)      \
    template rocsparse_status rocsparse::csrmm_template_row_split( \
        rocsparse_handle          handle,                          \
        rocsparse_operation       trans_A,                         \
        rocsparse_operation       trans_B,                         \
        JTYPE                     m,                               \
        JTYPE                     n,                               \
        JTYPE                     k,                               \
        ITYPE                     nnz,                             \
        JTYPE                     batch_count_A,                   \
        int64_t                   offsets_batch_stride_A,          \
        int64_t                   columns_values_batch_stride_A,   \
        const TTYPE*              alpha_device_host,               \
        const rocsparse_mat_descr descr,                           \
        const ATYPE*              csr_val,                         \
        const ITYPE*              csr_row_ptr,                     \
        const JTYPE*              csr_col_ind,                     \
        const BTYPE*              dense_B,                         \
        int64_t                   ldb,                             \
        JTYPE                     batch_count_B,                   \
        int64_t                   batch_stride_B,                  \
        rocsparse_order           order_B,                         \
        const TTYPE*              beta_device_host,                \
        CTYPE*                    dense_C,                         \
        int64_t                   ldc,                             \
        JTYPE                     batch_count_C,                   \
        int64_t                   batch_stride_C,                  \
        rocsparse_order           order_C,                         \
        bool                      force_conj_A,                    \
        int64_t                   max_row_nnz)

// Uniform precisions
INSTANTIATE(float, int32_t, int32_t, float, float, float);
INSTANTIATE(float, int64_t, int32_t, float, float, float);
INSTANTIATE(float, int64_t, int64_t, float, float, float);
INSTANTIATE(double, int32_t, int32_t, double, double, double);
INSTANTIATE(double, int64_t, int32_t, double, double, double);
INSTANTIATE(double, int64_t, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
// Mixed Precisions
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float);
#undef INSTANTIATE
