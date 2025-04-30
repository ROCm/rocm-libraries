/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "control.h"
#include "handle.h"

typedef enum rocsparse_bsrmm_alg_
{
    rocsparse_bsrmm_alg_default = 0,
    rocsparse_bsrmm_alg_bsr
} rocsparse_bsrmm_alg;

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A>
    rocsparse_status bsrmm_buffer_size_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_bsrmm_alg       alg,
                                                J                         mb,
                                                J                         n,
                                                J                         kb,
                                                I                         nnzb,
                                                const rocsparse_mat_descr descr,
                                                const A*                  bsr_val,
                                                const I*                  bsr_row_ptr,
                                                const J*                  bsr_col_ind,
                                                J                         block_dim,
                                                size_t*                   buffer_size);

    template <typename T, typename I, typename J, typename A>
    rocsparse_status bsrmm_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_bsrmm_alg       alg,
                                             J                         mb,
                                             J                         n,
                                             J                         kb,
                                             I                         nnzb,
                                             const rocsparse_mat_descr descr,
                                             const A*                  bsr_val,
                                             const I*                  bsr_row_ptr,
                                             const J*                  bsr_col_ind,
                                             J                         block_dim,
                                             void*                     temp_buffer);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmm_template_dispatch(rocsparse_handle    handle,
                                             rocsparse_direction dir,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             rocsparse_bsrmm_alg alg,
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
    rocsparse_status bsrmm_template(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_bsrmm_alg       alg,
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
}
