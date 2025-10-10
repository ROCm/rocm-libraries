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

#include "internal/level3/rocsparse_bsrmm.h"
#include "rocsparse_bsrmm.hpp"
#include "rocsparse_common.h"
#include "rocsparse_csrmm.hpp"

#include "rocsparse_common.hpp"
#include "rocsparse_templates.hpp"
#include "rocsparse_utility.hpp"
#include <hip/hip_runtime.h>

namespace rocsparse
{
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
                                           rocsparse_order           order_C);
}

rocsparse_status rocsparse::bsrmm_buffer_size(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_bsrmm_alg       alg,
                                              int64_t                   mb,
                                              int64_t                   n,
                                              int64_t                   kb,
                                              int64_t                   nnzb,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_datatype        bsr_val_datatype,
                                              const void*               bsr_val,
                                              rocsparse_indextype       bsr_row_ptr_indextype,
                                              const void*               bsr_row_ptr,
                                              rocsparse_indextype       bsr_col_ind_indextype,
                                              const void*               bsr_col_ind,
                                              int64_t                   block_dim,
                                              size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    *buffer_size = 0;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::bsrmm_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_bsrmm_alg       alg,
                                           int64_t                   mb,
                                           int64_t                   n,
                                           int64_t                   kb,
                                           int64_t                   nnzb,
                                           const rocsparse_mat_descr descr,
                                           rocsparse_datatype        bsr_val_datatype,
                                           const void*               bsr_val,
                                           rocsparse_indextype       bsr_row_ptr_indextype,
                                           const void*               bsr_row_ptr,
                                           rocsparse_indextype       bsr_col_ind_indextype,
                                           const void*               bsr_col_ind,
                                           int64_t                   block_dim,
                                           void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    return rocsparse_status_success;
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
rocsparse_status rocsparse::bsrmm_template_dispatch(rocsparse_handle    handle,
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
                                                    int64_t  columns_values_batch_stride_A,
                                                    const T* alpha,
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

    // If block dimension is one we can simply call csrmm
    if(block_dim == 1)
    {
        const I nnz = nnzb;
        const J m   = mb;
        const J k   = kb;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_template_dispatch(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     rocsparse_csrmm_alg_default,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     nnz,
                                                                     batch_count_A,
                                                                     offsets_batch_stride_A,
                                                                     columns_values_batch_stride_A,
                                                                     alpha,
                                                                     descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
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
                                                                     order_C,
                                                                     nullptr,
                                                                     false));
        return rocsparse_status_success;
    }

    switch(alg)
    {
    case rocsparse_bsrmm_alg_default:
    case rocsparse_bsrmm_alg_bsr:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_template_bsralg(handle,
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
}

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A, typename B, typename C>
    static rocsparse_status bsrmm_core(rocsparse_handle          handle,
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
                                       rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_template_dispatch(handle,
                                                                     dir,
                                                                     trans_A,
                                                                     trans_B,
                                                                     alg,
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

    template <typename T, typename C>
    static rocsparse_status bsrmm_quickreturn(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              int64_t                   mb,
                                              int64_t                   n,
                                              int64_t                   kb,
                                              int64_t                   nnzb,
                                              const T*                  alpha,
                                              const rocsparse_mat_descr descr,
                                              const void*               bsr_val,
                                              const void*               bsr_row_ptr,
                                              const void*               bsr_col_ind,
                                              int64_t                   block_dim,
                                              const void*               dense_B,
                                              int64_t                   ldb,
                                              const T*                  beta,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              rocsparse_order           order_B,
                                              rocsparse_order           order_C,
                                              int64_t                   batch_count_C,
                                              int64_t                   batch_stride_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(mb == 0 || n == 0 || kb == 0)
        {
            // matrix never accessed however still need to update C matrix
            const int64_t m      = block_dim * mb;
            const int64_t k      = block_dim * kb;
            const int64_t C_size = m * n;
            if(C_size > 0)
            {
                if(dense_C == nullptr && beta == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
                }
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::scale_2d_array(handle,
                                              (trans_A == rocsparse_operation_none) ? m : k,
                                              n,
                                              ldc,
                                              batch_count_C,
                                              batch_stride_C,
                                              beta,
                                              dense_C,
                                              order_C));
            }

            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename T, typename C>
    static rocsparse_status bsrmm_checkarg(rocsparse_handle          handle, //0
                                           rocsparse_direction       dir, //1
                                           rocsparse_operation       trans_A, //2
                                           rocsparse_operation       trans_B, //3
                                           int64_t                   mb, //4
                                           int64_t                   n, //5
                                           int64_t                   kb, //6
                                           int64_t                   nnzb, //7
                                           const T*                  alpha, //8
                                           const rocsparse_mat_descr descr, //9
                                           const void*               bsr_val, //10
                                           const void*               bsr_row_ptr, //11
                                           const void*               bsr_col_ind, //12
                                           int64_t                   block_dim, //13
                                           const void*               dense_B, //14
                                           int64_t                   ldb, //15
                                           const T*                  beta, //16
                                           C*                        dense_C, //17
                                           int64_t                   ldc, //18
                                           rocsparse_order           order_B,
                                           rocsparse_order           order_C,
                                           int64_t                   batch_count_C,
                                           int64_t                   batch_stride_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans_A);

        ROCSPARSE_CHECKARG(
            2, trans_A, (trans_A != rocsparse_operation_none), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG_ENUM(3, trans_B);
        ROCSPARSE_CHECKARG(
            3,
            trans_B,
            (trans_B != rocsparse_operation_none && trans_B != rocsparse_operation_transpose),
            rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG_SIZE(4, mb);
        ROCSPARSE_CHECKARG_SIZE(5, n);
        ROCSPARSE_CHECKARG_SIZE(6, kb);
        ROCSPARSE_CHECKARG_SIZE(7, nnzb);

        ROCSPARSE_CHECKARG_POINTER(9, descr);
        ROCSPARSE_CHECKARG(9,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(9,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(11, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(12, nnzb, bsr_col_ind);

        ROCSPARSE_CHECKARG_SIZE(13, block_dim);
        ROCSPARSE_CHECKARG(13, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        const rocsparse_status status = rocsparse::bsrmm_quickreturn(handle,
                                                                     dir,
                                                                     trans_A,
                                                                     trans_B,
                                                                     mb,
                                                                     n,
                                                                     kb,
                                                                     nnzb,
                                                                     alpha,
                                                                     descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
                                                                     block_dim,
                                                                     dense_B,
                                                                     ldb,
                                                                     beta,
                                                                     dense_C,
                                                                     ldc,
                                                                     order_B,
                                                                     order_C,
                                                                     batch_count_C,
                                                                     batch_stride_C);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(8, alpha);
        ROCSPARSE_CHECKARG_POINTER(14, dense_B);
        ROCSPARSE_CHECKARG_SIZE(15, ldb);
        ROCSPARSE_CHECKARG_POINTER(16, beta);
        ROCSPARSE_CHECKARG_POINTER(17, dense_C);
        ROCSPARSE_CHECKARG_SIZE(18, ldc);

        static constexpr int64_t s_one = static_cast<int64_t>(1);

        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            ROCSPARSE_CHECKARG(
                18,
                ldc,
                (ldc < rocsparse::max(s_one,
                                      ((order_C == rocsparse_order_column) ? mb * block_dim : n))),
                rocsparse_status_invalid_size);

            // Check leading dimension of B
            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(
                    15,
                    ldb,
                    (ldb < rocsparse::max(
                         s_one, ((order_B == rocsparse_order_column) ? kb * block_dim : n))),
                    rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    15,
                    ldb,
                    (ldb < rocsparse::max(
                         s_one, ((order_B == rocsparse_order_column) ? n : kb * block_dim))),
                    rocsparse_status_invalid_size);
                break;
            }
            }
            break;
        }

        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            ROCSPARSE_CHECKARG(
                18,
                ldc,
                (ldc < rocsparse::max(s_one,
                                      ((order_C == rocsparse_order_column) ? kb * block_dim : n))),
                rocsparse_status_invalid_size);

            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(
                    15,
                    ldb,
                    (ldb < rocsparse::max(
                         s_one, ((order_B == rocsparse_order_column) ? mb * block_dim : n))),
                    rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    15,
                    ldb,
                    (ldb < rocsparse::max(
                         s_one, ((order_B == rocsparse_order_column) ? n : mb * block_dim))),
                    rocsparse_status_invalid_size);
                break;
            }
            }
            break;
        }
        }
        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
rocsparse_status rocsparse::bsrmm_template(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_bsrmm_alg       alg,
                                           int64_t                   mb,
                                           int64_t                   n,
                                           int64_t                   kb,
                                           int64_t                   nnzb,
                                           int64_t                   batch_count_A,
                                           int64_t                   offsets_batch_stride_A,
                                           int64_t                   columns_values_batch_stride_A,
                                           const void*               alpha,
                                           const rocsparse_mat_descr descr,
                                           const void*               bsr_val,
                                           const void*               bsr_row_ptr,
                                           const void*               bsr_col_ind,
                                           int64_t                   block_dim,
                                           const void*               dense_B,
                                           int64_t                   ldb,
                                           int64_t                   batch_count_B,
                                           int64_t                   batch_stride_B,
                                           rocsparse_order           order_B,
                                           const void*               beta,
                                           void*                     dense_C,
                                           int64_t                   ldc,
                                           int64_t                   batch_count_C,
                                           int64_t                   batch_stride_C,
                                           rocsparse_order           order_C)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::bsrmm_quickreturn(handle,
                                                                 dir,
                                                                 trans_A,
                                                                 trans_B,
                                                                 mb,
                                                                 n,
                                                                 kb,
                                                                 nnzb,
                                                                 static_cast<const T*>(alpha),
                                                                 descr,
                                                                 bsr_val,
                                                                 bsr_row_ptr,
                                                                 bsr_col_ind,
                                                                 block_dim,
                                                                 dense_B,
                                                                 ldb,
                                                                 static_cast<const T*>(beta),
                                                                 static_cast<C*>(dense_C),
                                                                 ldc,
                                                                 order_B,
                                                                 order_C,
                                                                 batch_count_C,
                                                                 batch_stride_C);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::bsrmm_core<T, I, J, A, B, C>(handle,
                                                 dir,
                                                 trans_A,
                                                 trans_B,
                                                 alg,
                                                 mb,
                                                 n,
                                                 kb,
                                                 nnzb,
                                                 batch_count_A,
                                                 offsets_batch_stride_A,
                                                 columns_values_batch_stride_A,
                                                 static_cast<const T*>(alpha),
                                                 descr,
                                                 static_cast<const A*>(bsr_val),
                                                 static_cast<const I*>(bsr_row_ptr),
                                                 static_cast<const J*>(bsr_col_ind),
                                                 block_dim,
                                                 static_cast<const B*>(dense_B),
                                                 ldb,
                                                 batch_count_B,
                                                 batch_stride_B,
                                                 order_B,
                                                 static_cast<const T*>(beta),
                                                 static_cast<C*>(dense_C),
                                                 ldc,
                                                 batch_count_C,
                                                 batch_stride_C,
                                                 order_C)));

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status bsrmm_impl(rocsparse_handle          handle,
                                rocsparse_direction       dir,
                                rocsparse_operation       trans_A,
                                rocsparse_operation       trans_B,
                                rocsparse_bsrmm_alg       alg,
                                rocsparse_int             mb,
                                rocsparse_int             n,
                                rocsparse_int             kb,
                                rocsparse_int             nnzb,
                                rocsparse_int             batch_count_A,
                                int64_t                   offsets_batch_stride_A,
                                int64_t                   columns_values_batch_stride_A,
                                const T*                  alpha,
                                const rocsparse_mat_descr descr,
                                const T*                  bsr_val,
                                const rocsparse_int*      bsr_row_ptr,
                                const rocsparse_int*      bsr_col_ind,
                                rocsparse_int             block_dim,
                                const T*                  B,
                                rocsparse_int             ldb,
                                rocsparse_int             batch_count_B,
                                int64_t                   batch_stride_B,
                                rocsparse_order           order_B,
                                const T*                  beta,
                                T*                        C,
                                rocsparse_int             ldc,
                                rocsparse_int             batch_count_C,
                                int64_t                   batch_stride_C,
                                rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xbsrmm"),
                             dir,
                             trans_A,
                             trans_B,
                             mb,
                             n,
                             kb,
                             nnzb,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr,
                             (const void*&)bsr_val,
                             (const void*&)bsr_row_ptr,
                             (const void*&)bsr_col_ind,
                             block_dim,
                             (const void*&)B,
                             ldb,
                             LOG_TRACE_SCALAR_VALUE(handle, beta),
                             (const void*&)C,
                             ldc);

        const rocsparse_status status = rocsparse::bsrmm_checkarg(handle,
                                                                  dir,
                                                                  trans_A,
                                                                  trans_B,
                                                                  mb,
                                                                  n,
                                                                  kb,
                                                                  nnzb,
                                                                  alpha,
                                                                  descr,
                                                                  bsr_val,
                                                                  bsr_row_ptr,
                                                                  bsr_col_ind,
                                                                  block_dim,
                                                                  B,
                                                                  ldb,
                                                                  beta,
                                                                  C,
                                                                  ldc,
                                                                  order_B,
                                                                  order_C,
                                                                  batch_count_C,
                                                                  batch_stride_C);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_core(handle,
                                                        dir,
                                                        trans_A,
                                                        trans_B,
                                                        alg,
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
                                                        B,
                                                        ldb,
                                                        batch_count_B,
                                                        batch_stride_B,
                                                        order_B,
                                                        beta,
                                                        C,
                                                        ldc,
                                                        batch_count_C,
                                                        batch_stride_C,
                                                        order_C));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE)     \
    template rocsparse_status rocsparse::bsrmm_template_dispatch( \
        rocsparse_handle          handle,                         \
        rocsparse_direction       dir,                            \
        rocsparse_operation       trans_A,                        \
        rocsparse_operation       trans_B,                        \
        rocsparse_bsrmm_alg       alg,                            \
        JTYPE                     mb,                             \
        JTYPE                     n,                              \
        JTYPE                     kb,                             \
        ITYPE                     nnzb,                           \
        JTYPE                     batch_count_A,                  \
        int64_t                   offsets_batch_stride_A,         \
        int64_t                   columns_values_batch_stride_A,  \
        const TTYPE*              alpha,                          \
        const rocsparse_mat_descr descr,                          \
        const ATYPE*              bsr_val,                        \
        const ITYPE*              bsr_row_ptr,                    \
        const JTYPE*              bsr_col_ind,                    \
        JTYPE                     block_dim,                      \
        const BTYPE*              dense_B,                        \
        int64_t                   ldb,                            \
        JTYPE                     batch_count_B,                  \
        int64_t                   batch_stride_B,                 \
        rocsparse_order           order_B,                        \
        const TTYPE*              beta,                           \
        CTYPE*                    dense_C,                        \
        int64_t                   ldc,                            \
        JTYPE                     batch_count_C,                  \
        int64_t                   batch_stride_C,                 \
        rocsparse_order           order_C);

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

// Mixed precisions
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

#define INSTANTIATE(T, I, J, A, B, C)                                      \
    template rocsparse_status rocsparse::bsrmm_template<T, I, J, A, B, C>( \
        rocsparse_handle          handle,                                  \
        rocsparse_direction       dir,                                     \
        rocsparse_operation       trans_A,                                 \
        rocsparse_operation       trans_B,                                 \
        rocsparse_bsrmm_alg       alg,                                     \
        int64_t                   mb,                                      \
        int64_t                   n,                                       \
        int64_t                   kb,                                      \
        int64_t                   nnzb,                                    \
        int64_t                   batch_count_A,                           \
        int64_t                   offsets_batch_stride_A,                  \
        int64_t                   columns_values_batch_stride_A,           \
        const void*               alpha,                                   \
        const rocsparse_mat_descr descr,                                   \
        const void*               bsr_val,                                 \
        const void*               bsr_row_ptr,                             \
        const void*               bsr_col_ind,                             \
        int64_t                   block_dim,                               \
        const void*               dense_B,                                 \
        int64_t                   ldb,                                     \
        int64_t                   batch_count_B,                           \
        int64_t                   batch_stride_B,                          \
        rocsparse_order           order_B,                                 \
        const void*               beta,                                    \
        void*                     dense_C,                                 \
        int64_t                   ldc,                                     \
        int64_t                   batch_count_C,                           \
        int64_t                   batch_stride_C,                          \
        rocsparse_order           order_C);

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

// Mixed precisions
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

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                       \
                                     rocsparse_direction       dir,                          \
                                     rocsparse_operation       trans_A,                      \
                                     rocsparse_operation       trans_B,                      \
                                     rocsparse_int             mb,                           \
                                     rocsparse_int             n,                            \
                                     rocsparse_int             kb,                           \
                                     rocsparse_int             nnzb,                         \
                                     const TYPE*               alpha,                        \
                                     const rocsparse_mat_descr descr,                        \
                                     const TYPE*               bsr_val,                      \
                                     const rocsparse_int*      bsr_row_ptr,                  \
                                     const rocsparse_int*      bsr_col_ind,                  \
                                     rocsparse_int             block_dim,                    \
                                     const TYPE*               B,                            \
                                     rocsparse_int             ldb,                          \
                                     const TYPE*               beta,                         \
                                     TYPE*                     C,                            \
                                     rocsparse_int             ldc)                          \
    try                                                                                      \
    {                                                                                        \
        ROCSPARSE_ROUTINE_TRACE;                                                             \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_impl<TYPE>(handle,                        \
                                                              dir,                           \
                                                              trans_A,                       \
                                                              trans_B,                       \
                                                              rocsparse_bsrmm_alg_default,   \
                                                              mb,                            \
                                                              n,                             \
                                                              kb,                            \
                                                              nnzb,                          \
                                                              static_cast<rocsparse_int>(1), \
                                                              static_cast<int64_t>(0),       \
                                                              static_cast<int64_t>(0),       \
                                                              alpha,                         \
                                                              descr,                         \
                                                              bsr_val,                       \
                                                              bsr_row_ptr,                   \
                                                              bsr_col_ind,                   \
                                                              block_dim,                     \
                                                              B,                             \
                                                              ldb,                           \
                                                              static_cast<rocsparse_int>(1), \
                                                              static_cast<int64_t>(0),       \
                                                              rocsparse_order_column,        \
                                                              beta,                          \
                                                              C,                             \
                                                              ldc,                           \
                                                              static_cast<rocsparse_int>(1), \
                                                              static_cast<int64_t>(0),       \
                                                              rocsparse_order_column));      \
        return rocsparse_status_success;                                                     \
    }                                                                                        \
    catch(...)                                                                               \
    {                                                                                        \
        RETURN_ROCSPARSE_EXCEPTION();                                                        \
    }

C_IMPL(rocsparse_sbsrmm, float);
C_IMPL(rocsparse_dbsrmm, double);
C_IMPL(rocsparse_cbsrmm, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmm, rocsparse_double_complex);
#undef C_IMPL
