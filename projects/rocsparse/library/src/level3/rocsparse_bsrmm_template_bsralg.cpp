/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../level2/rocsparse_bsrmv.hpp"
#include "rocsparse_bsrmm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmmnn_template_small(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            J                         mb,
                                            J                         n,
                                            J                         kb,
                                            I                         nnzb,
                                            J                         batch_count_A,
                                            int64_t                   offsets_batch_stride_A,
                                            int64_t                   columns_values_batch_stride_A,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr,
                                            const A*                  bsr_val,
                                            const I*                  bsr_row_ptr,
                                            const J*                  bsr_col_ind,
                                            J                         block_dim,
                                            const B*                  dense_B,
                                            int64_t                   ldb,
                                            J                         batch_count_B,
                                            int64_t                   batch_stride_B,
                                            rocsparse_order           order_B,
                                            const T*                  beta,
                                            C*                        dense_C,
                                            int64_t                   ldc,
                                            J                         batch_count_C,
                                            int64_t                   batch_stride_C,
                                            rocsparse_order           order_C);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmmnt_template_small(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            J                         mb,
                                            J                         n,
                                            J                         kb,
                                            I                         nnzb,
                                            J                         batch_count_A,
                                            int64_t                   offsets_batch_stride_A,
                                            int64_t                   columns_values_batch_stride_A,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr,
                                            const A*                  bsr_val,
                                            const I*                  bsr_row_ptr,
                                            const J*                  bsr_col_ind,
                                            J                         block_dim,
                                            const B*                  dense_B,
                                            int64_t                   ldb,
                                            J                         batch_count_B,
                                            int64_t                   batch_stride_B,
                                            rocsparse_order           order_B,
                                            const T*                  beta,
                                            C*                        dense_C,
                                            int64_t                   ldc,
                                            J                         batch_count_C,
                                            int64_t                   batch_stride_C,
                                            rocsparse_order           order_C);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmm_template_large_ext(bool                nn,
                                              rocsparse_handle    handle,
                                              rocsparse_direction dir,
                                              rocsparse_operation trans_A,
                                              rocsparse_operation trans_B,
                                              J                   mb,
                                              J                   n,
                                              J                   kb,
                                              I                   nnzb,
                                              J                   batch_count_A,
                                              int64_t             offsets_batch_stride_A,
                                              int64_t             columns_values_batch_stride_A,
                                              const T*            alpha,
                                              const rocsparse_mat_descr descr,
                                              const A*                  bsr_val,
                                              const I*                  bsr_row_ptr,
                                              const J*                  bsr_col_ind,
                                              J                         block_dim,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              J                         batch_count_B,
                                              int64_t                   batch_stride_B,
                                              rocsparse_order           order_B,
                                              const T*                  beta,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              J                         batch_count_C,
                                              int64_t                   batch_stride_C,
                                              rocsparse_order           order_C);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmm_template_general(bool                      nn,
                                            rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            J                         mb,
                                            J                         n,
                                            J                         kb,
                                            I                         nnzb,
                                            J                         batch_count_A,
                                            int64_t                   offsets_batch_stride_A,
                                            int64_t                   columns_values_batch_stride_A,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr,
                                            const A*                  bsr_val,
                                            const I*                  bsr_row_ptr,
                                            const J*                  bsr_col_ind,
                                            J                         block_dim,
                                            const B*                  dense_B,
                                            int64_t                   ldb,
                                            J                         batch_count_B,
                                            int64_t                   batch_stride_B,
                                            rocsparse_order           order_B,
                                            const T*                  beta,
                                            C*                        dense_C,
                                            int64_t                   ldc,
                                            J                         batch_count_C,
                                            int64_t                   batch_stride_C,
                                            rocsparse_order           order_C);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmmnn_template_bsralg(rocsparse_handle    handle,
                                             bool                conj_A,
                                             bool                conj_B,
                                             rocsparse_direction dir,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             J                   mb,
                                             J                   n,
                                             J                   kb,
                                             I                   nnzb,
                                             J                   batch_count_A,
                                             int64_t             offsets_batch_stride_A,
                                             int64_t             columns_values_batch_stride_A,
                                             const T*            alpha,
                                             const rocsparse_mat_descr descr,
                                             const A*                  bsr_val,
                                             const I*                  bsr_row_ptr,
                                             const J*                  bsr_col_ind,
                                             J                         block_dim,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             J                         batch_count_B,
                                             int64_t                   batch_stride_B,
                                             rocsparse_order           order_B,
                                             const T*                  beta,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             J                         batch_count_C,
                                             int64_t                   batch_stride_C,
                                             rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(block_dim == 2)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::bsrmmnn_template_small(handle,
                                                  dir,
                                                  trans_A,
                                                  trans_B,
                                                  mb,
                                                  n,
                                                  kb,
                                                  nnzb,
                                                  batch_count_A,
                                                  offsets_batch_stride_A,
                                                  columns_values_batch_stride_A,
                                                  alpha,
                                                  descr,
                                                  bsr_val,
                                                  bsr_row_ptr,
                                                  bsr_col_ind,
                                                  block_dim,
                                                  dense_B,
                                                  ldb,
                                                  batch_count_B,
                                                  batch_stride_B,
                                                  order_B,
                                                  beta,
                                                  dense_C,
                                                  ldc,
                                                  batch_count_C,
                                                  batch_stride_C,
                                                  order_C));
            return rocsparse_status_success;
        }

        if(block_dim <= 32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::bsrmm_template_large_ext(true,
                                                    handle,
                                                    dir,
                                                    trans_A,
                                                    trans_B,
                                                    mb,
                                                    n,
                                                    kb,
                                                    nnzb,
                                                    batch_count_A,
                                                    offsets_batch_stride_A,
                                                    columns_values_batch_stride_A,
                                                    alpha,
                                                    descr,
                                                    bsr_val,
                                                    bsr_row_ptr,
                                                    bsr_col_ind,
                                                    block_dim,
                                                    dense_B,
                                                    ldb,
                                                    batch_count_B,
                                                    batch_stride_B,
                                                    order_B,
                                                    beta,
                                                    dense_C,
                                                    ldc,
                                                    batch_count_C,
                                                    batch_stride_C,
                                                    order_C));
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::bsrmm_template_general(true,
                                                  handle,
                                                  dir,
                                                  trans_A,
                                                  trans_B,
                                                  mb,
                                                  n,
                                                  kb,
                                                  nnzb,
                                                  batch_count_A,
                                                  offsets_batch_stride_A,
                                                  columns_values_batch_stride_A,
                                                  alpha,
                                                  descr,
                                                  bsr_val,
                                                  bsr_row_ptr,
                                                  bsr_col_ind,
                                                  block_dim,
                                                  dense_B,
                                                  ldb,
                                                  batch_count_B,
                                                  batch_stride_B,
                                                  order_B,
                                                  beta,
                                                  dense_C,
                                                  ldc,
                                                  batch_count_C,
                                                  batch_stride_C,
                                                  order_C));
            return rocsparse_status_success;
        }
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmmnt_template_bsralg(rocsparse_handle    handle,
                                             bool                conj_A,
                                             bool                conj_B,
                                             rocsparse_direction dir,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             J                   mb,
                                             J                   n,
                                             J                   kb,
                                             I                   nnzb,
                                             J                   batch_count_A,
                                             int64_t             offsets_batch_stride_A,
                                             int64_t             columns_values_batch_stride_A,
                                             const T*            alpha,
                                             const rocsparse_mat_descr descr,
                                             const A*                  bsr_val,
                                             const I*                  bsr_row_ptr,
                                             const J*                  bsr_col_ind,
                                             J                         block_dim,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             J                         batch_count_B,
                                             int64_t                   batch_stride_B,
                                             rocsparse_order           order_B,
                                             const T*                  beta,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             J                         batch_count_C,
                                             int64_t                   batch_stride_C,
                                             rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(block_dim == 2)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::bsrmmnt_template_small(handle,
                                                  dir,
                                                  trans_A,
                                                  trans_B,
                                                  mb,
                                                  n,
                                                  kb,
                                                  nnzb,
                                                  batch_count_A,
                                                  offsets_batch_stride_A,
                                                  columns_values_batch_stride_A,
                                                  alpha,
                                                  descr,
                                                  bsr_val,
                                                  bsr_row_ptr,
                                                  bsr_col_ind,
                                                  block_dim,
                                                  dense_B,
                                                  ldb,
                                                  batch_count_B,
                                                  batch_stride_B,
                                                  order_B,
                                                  beta,
                                                  dense_C,
                                                  ldc,
                                                  batch_count_C,
                                                  batch_stride_C,
                                                  order_C));
            return rocsparse_status_success;
        }

        if(block_dim <= 32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::bsrmm_template_large_ext(false,
                                                    handle,
                                                    dir,
                                                    trans_A,
                                                    trans_B,
                                                    mb,
                                                    n,
                                                    kb,
                                                    nnzb,
                                                    batch_count_A,
                                                    offsets_batch_stride_A,
                                                    columns_values_batch_stride_A,
                                                    alpha,
                                                    descr,
                                                    bsr_val,
                                                    bsr_row_ptr,
                                                    bsr_col_ind,
                                                    block_dim,
                                                    dense_B,
                                                    ldb,
                                                    batch_count_B,
                                                    batch_stride_B,
                                                    order_B,
                                                    beta,
                                                    dense_C,
                                                    ldc,
                                                    batch_count_C,
                                                    batch_stride_C,
                                                    order_C));
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::bsrmm_template_general(false,
                                                  handle,
                                                  dir,
                                                  trans_A,
                                                  trans_B,
                                                  mb,
                                                  n,
                                                  kb,
                                                  nnzb,
                                                  batch_count_A,
                                                  offsets_batch_stride_A,
                                                  columns_values_batch_stride_A,
                                                  alpha,
                                                  descr,
                                                  bsr_val,
                                                  bsr_row_ptr,
                                                  bsr_col_ind,
                                                  block_dim,
                                                  dense_B,
                                                  ldb,
                                                  batch_count_B,
                                                  batch_stride_B,
                                                  order_B,
                                                  beta,
                                                  dense_C,
                                                  ldc,
                                                  batch_count_C,
                                                  batch_stride_C,
                                                  order_C));
            return rocsparse_status_success;
        }

        return rocsparse_status_success;
    }

#define ROCSPARSE_BSRMM_TEMPLATE_BSRALG_IMPL(NAME) \
    NAME(handle,                                   \
         conj_A,                                   \
         conj_B,                                   \
         dir,                                      \
         trans_A,                                  \
         trans_B,                                  \
         mb,                                       \
         n,                                        \
         kb,                                       \
         nnzb,                                     \
         batch_count_A,                            \
         offsets_batch_stride_A,                   \
         columns_values_batch_stride_A,            \
         alpha,                                    \
         descr,                                    \
         bsr_val,                                  \
         bsr_row_ptr,                              \
         bsr_col_ind,                              \
         block_dim,                                \
         dense_B,                                  \
         ldb,                                      \
         batch_count_B,                            \
         batch_stride_B,                           \
         order_B,                                  \
         beta,                                     \
         dense_C,                                  \
         ldc,                                      \
         batch_count_C,                            \
         batch_stride_C,                           \
         order_C)

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmm_template_bsralg(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           J                         mb,
                                           J                         n,
                                           J                         kb,
                                           I                         nnzb,
                                           J                         batch_count_A,
                                           int64_t                   offsets_batch_stride_A,
                                           int64_t                   columns_values_batch_stride_A,
                                           const T*                  alpha,
                                           const rocsparse_mat_descr descr,
                                           const A*                  bsr_val,
                                           const I*                  bsr_row_ptr,
                                           const J*                  bsr_col_ind,
                                           J                         block_dim,
                                           const B*                  dense_B,
                                           int64_t                   ldb,
                                           J                         batch_count_B,
                                           int64_t                   batch_stride_B,
                                           rocsparse_order           order_B,
                                           const T*                  beta,
                                           C*                        dense_C,
                                           int64_t                   ldc,
                                           J                         batch_count_C,
                                           int64_t                   batch_stride_C,
                                           rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose);
        const bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

        // Run different bsrmm kernels
        if(trans_A == rocsparse_operation_none)
        {
            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    ROCSPARSE_BSRMM_TEMPLATE_BSRALG_IMPL(rocsparse::bsrmmnn_template_bsralg));
                return rocsparse_status_success;
            }
            else if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    ROCSPARSE_BSRMM_TEMPLATE_BSRALG_IMPL(rocsparse::bsrmmnt_template_bsralg));
                return rocsparse_status_success;
            }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE)    \
    template rocsparse_status rocsparse::bsrmm_template_bsralg(  \
        rocsparse_handle          handle,                        \
        rocsparse_direction       dir,                           \
        rocsparse_operation       trans_A,                       \
        rocsparse_operation       trans_B,                       \
        JTYPE                     mb,                            \
        JTYPE                     n,                             \
        JTYPE                     kb,                            \
        ITYPE                     nnzb,                          \
        JTYPE                     batch_count_A,                 \
        int64_t                   offsets_batch_stride_A,        \
        int64_t                   columns_values_batch_stride_A, \
        const TTYPE*              alpha,                         \
        const rocsparse_mat_descr descr,                         \
        const ATYPE*              bsr_val,                       \
        const ITYPE*              bsr_row_ptr,                   \
        const JTYPE*              bsr_col_ind,                   \
        JTYPE                     block_dim,                     \
        const BTYPE*              dense_B,                       \
        int64_t                   ldb,                           \
        JTYPE                     batch_count_B,                 \
        int64_t                   batch_stride_B,                \
        rocsparse_order           order_B,                       \
        const TTYPE*              beta,                          \
        CTYPE*                    dense_C,                       \
        int64_t                   ldc,                           \
        JTYPE                     batch_count_C,                 \
        int64_t                   batch_stride_C,                \
        rocsparse_order           order_C)

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
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE
