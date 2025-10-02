/*! \file */
/* ************************************************************************
* Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "../conversion/rocsparse_csr2coo.hpp"
#include "internal/level3/rocsparse_csrmm.h"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrmm.hpp"
#include "rocsparse_utility.hpp"
#include <algorithm>

namespace rocsparse
{
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
                                              const T*            alpha,
                                              const rocsparse_mat_descr descr,
                                              const A*                  csr_val,
                                              const I*                  csr_row_ptr,
                                              const J*                  csr_col_ind,
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
                                              rocsparse_order           order_C,
                                              bool                      force_conj_A);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template_nnz_split(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              J                         m,
                                              J                         n,
                                              J                         k,
                                              I                         nnz,
                                              const T*                  alpha,
                                              const rocsparse_mat_descr descr,
                                              const A*                  csr_val,
                                              const I*                  csr_row_ptr,
                                              const J*                  csr_col_ind,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              rocsparse_order           order_B,
                                              const T*                  beta,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              rocsparse_order           order_C,
                                              void*                     temp_buffer,
                                              bool                      force_conj_A);
    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template_merge(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          J                         m,
                                          J                         n,
                                          J                         k,
                                          I                         nnz,
                                          const T*                  alpha,
                                          const rocsparse_mat_descr descr,
                                          const A*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const B*                  dense_B,
                                          int64_t                   ldb,
                                          rocsparse_order           order_B,
                                          const T*                  beta,
                                          C*                        dense_C,
                                          int64_t                   ldc,
                                          rocsparse_order           order_C,
                                          void*                     temp_buffer,
                                          bool                      force_conj_A);
    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template_dispatch(rocsparse_handle    handle,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             rocsparse_csrmm_alg alg,
                                             J                   m,
                                             J                   n,
                                             J                   k,
                                             I                   nnz,
                                             J                   batch_count_A,
                                             int64_t             offsets_batch_stride_A,
                                             int64_t             columns_values_batch_stride_A,
                                             const T*            alpha,
                                             const rocsparse_mat_descr descr,
                                             const A*                  csr_val,
                                             const I*                  csr_row_ptr,
                                             const J*                  csr_col_ind,
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
                                             rocsparse_order           order_C,
                                             void*                     temp_buffer,
                                             bool                      force_conj_A)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(alg)
        {
        case rocsparse_csrmm_alg_default:
        case rocsparse_csrmm_alg_row_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrmm_template_row_split(handle,
                                                    trans_A,
                                                    trans_B,
                                                    m,
                                                    n,
                                                    k,
                                                    nnz,
                                                    batch_count_A,
                                                    offsets_batch_stride_A,
                                                    columns_values_batch_stride_A,
                                                    alpha,
                                                    descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
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
                                                    force_conj_A));
            return rocsparse_status_success;
        }

        case rocsparse_csrmm_alg_nnz_split:
        {
            switch(trans_A)
            {
            case rocsparse_operation_none:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_template_nnz_split(handle,
                                                                              trans_A,
                                                                              trans_B,
                                                                              m,
                                                                              n,
                                                                              k,
                                                                              nnz,
                                                                              alpha,
                                                                              descr,
                                                                              csr_val,
                                                                              csr_row_ptr,
                                                                              csr_col_ind,
                                                                              dense_B,
                                                                              ldb,
                                                                              order_B,
                                                                              beta,
                                                                              dense_C,
                                                                              ldc,
                                                                              order_C,
                                                                              temp_buffer,
                                                                              force_conj_A));
                return rocsparse_status_success;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrmm_template_row_split(handle,
                                                        trans_A,
                                                        trans_B,
                                                        m,
                                                        n,
                                                        k,
                                                        nnz,
                                                        batch_count_A,
                                                        offsets_batch_stride_A,
                                                        columns_values_batch_stride_A,
                                                        alpha,
                                                        descr,
                                                        csr_val,
                                                        csr_row_ptr,
                                                        csr_col_ind,
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
                                                        force_conj_A));
                return rocsparse_status_success;
            }
            }
        }
        case rocsparse_csrmm_alg_merge_path:
        {
            switch(trans_A)
            {
            case rocsparse_operation_none:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_template_merge(handle,
                                                                          trans_A,
                                                                          trans_B,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          nnz,
                                                                          alpha,
                                                                          descr,
                                                                          csr_val,
                                                                          csr_row_ptr,
                                                                          csr_col_ind,
                                                                          dense_B,
                                                                          ldb,
                                                                          order_B,
                                                                          beta,
                                                                          dense_C,
                                                                          ldc,
                                                                          order_C,
                                                                          temp_buffer,
                                                                          force_conj_A));
                return rocsparse_status_success;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrmm_template_row_split(handle,
                                                        trans_A,
                                                        trans_B,
                                                        m,
                                                        n,
                                                        k,
                                                        nnz,
                                                        batch_count_A,
                                                        offsets_batch_stride_A,
                                                        columns_values_batch_stride_A,
                                                        alpha,
                                                        descr,
                                                        csr_val,
                                                        csr_row_ptr,
                                                        csr_col_ind,
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
                                                        force_conj_A));
                return rocsparse_status_success;
            }
            }
        }
        }
    }
    template <typename T, typename I, typename J, typename A, typename B, typename C>
    static rocsparse_status csrmm_core(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_csrmm_alg       alg,
                                       J                         m,
                                       J                         n,
                                       J                         k,
                                       I                         nnz,
                                       J                         batch_count_A,
                                       int64_t                   offsets_batch_stride_A,
                                       int64_t                   columns_values_batch_stride_A,
                                       const T*                  alpha,
                                       const rocsparse_mat_descr descr,
                                       const A*                  csr_val,
                                       const I*                  csr_row_ptr,
                                       const J*                  csr_col_ind,
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
                                       rocsparse_order           order_C,
                                       void*                     temp_buffer,
                                       bool                      force_conj_A)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
        const bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
        const bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);
        if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_template_dispatch(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     alg,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     nnz,
                                                                     batch_count_A,
                                                                     offsets_batch_stride_A,
                                                                     columns_values_batch_stride_A,
                                                                     alpha,
                                                                     descr,
                                                                     csr_val,
                                                                     csr_row_ptr,
                                                                     csr_col_ind,
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
                                                                     temp_buffer,
                                                                     force_conj_A));
        return rocsparse_status_success;
    }

    template <typename T, typename C>
    static rocsparse_status csrmm_quickreturn(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              int64_t                   m,
                                              int64_t                   n,
                                              int64_t                   k,
                                              int64_t                   nnz,
                                              const T*                  alpha,
                                              const rocsparse_mat_descr descr,
                                              const void*               csr_val,
                                              const void*               csr_row_ptr,
                                              const void*               csr_col_ind,
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

        if(m == 0 || n == 0 || k == 0)
        {
            // matrix never accessed however still need to update C matrix
            const int64_t Csize = (trans_A == rocsparse_operation_none) ? m * n : k * n;
            if(Csize > 0)
            {
                if(dense_C == nullptr || beta == nullptr)
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
        if(handle->pointer_mode == rocsparse_pointer_mode_host
           && (alpha != nullptr && *alpha == static_cast<T>(0))
           && (beta != nullptr && *beta == static_cast<T>(1)))
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename T, typename C>
    static rocsparse_status csrmm_checkarg(rocsparse_handle          handle, //0
                                           rocsparse_operation       trans_A, //1
                                           rocsparse_operation       trans_B, //2
                                           int64_t                   m, //3
                                           int64_t                   n, //4
                                           int64_t                   k, //5
                                           int64_t                   nnz, //6
                                           const T*                  alpha, //7
                                           const rocsparse_mat_descr descr, //8
                                           const void*               csr_val, //9
                                           const void*               csr_row_ptr, //10
                                           const void*               csr_col_ind, //11
                                           const void*               dense_B, //12
                                           int64_t                   ldb, //13
                                           const T*                  beta, //14
                                           C*                        dense_C, //15
                                           int64_t                   ldc, //16
                                           rocsparse_order           order_B,
                                           rocsparse_order           order_C,
                                           int64_t                   batch_count_C,
                                           int64_t                   batch_stride_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, n);
        ROCSPARSE_CHECKARG_SIZE(5, k);
        ROCSPARSE_CHECKARG_SIZE(6, nnz);
        ROCSPARSE_CHECKARG_POINTER(8, descr);
        ROCSPARSE_CHECKARG(8,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(8,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_ARRAY(9, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(10, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(11, nnz, csr_col_ind);

        const rocsparse_status status = rocsparse::csrmm_quickreturn(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     nnz,
                                                                     alpha,
                                                                     descr,
                                                                     csr_val,
                                                                     csr_row_ptr,
                                                                     csr_col_ind,
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
        ROCSPARSE_CHECKARG_POINTER(7, alpha);
        ROCSPARSE_CHECKARG_POINTER(12, dense_B);
        ROCSPARSE_CHECKARG_SIZE(13, ldb);
        ROCSPARSE_CHECKARG_POINTER(14, beta);
        ROCSPARSE_CHECKARG_POINTER(15, dense_C);
        ROCSPARSE_CHECKARG_SIZE(16, ldc);

        static constexpr int64_t s_one = 1;
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            // Check leading dimension of C
            ROCSPARSE_CHECKARG(
                16,
                ldc,
                (ldc < rocsparse::max(s_one, ((order_C == rocsparse_order_column) ? m : n))),
                rocsparse_status_invalid_size);
            // Check leading dimension of B
            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(
                    13,
                    ldb,
                    (ldb < rocsparse::max(s_one, ((order_B == rocsparse_order_column) ? k : n))),
                    rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    13,
                    ldb,
                    (ldb < rocsparse::max(s_one, ((order_B == rocsparse_order_column) ? n : k))),
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
                16,
                ldc,
                (ldc < rocsparse::max(s_one, ((order_C == rocsparse_order_column) ? k : n))),
                rocsparse_status_invalid_size);
            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(
                    13,
                    ldb,
                    (ldb < rocsparse::max(s_one, ((order_B == rocsparse_order_column) ? m : n))),
                    rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    13,
                    ldb,
                    (ldb < rocsparse::max(s_one, ((order_B == rocsparse_order_column) ? n : m))),
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
rocsparse_status rocsparse::csrmm_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_csrmm_alg       alg,
                                           int64_t                   m,
                                           int64_t                   n,
                                           int64_t                   k,
                                           int64_t                   nnz,
                                           int64_t                   batch_count_A,
                                           int64_t                   offsets_batch_stride_A,
                                           int64_t                   columns_values_batch_stride_A,
                                           const void*               alpha,
                                           const rocsparse_mat_descr descr,
                                           const void*               csr_val,
                                           const void*               csr_row_ptr,
                                           const void*               csr_col_ind,
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
                                           rocsparse_order           order_C,
                                           void*                     temp_buffer,
                                           bool                      force_conj_A)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::csrmm_quickreturn(handle,
                                                                 trans_A,
                                                                 trans_B,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 nnz,
                                                                 static_cast<const T*>(alpha),
                                                                 descr,
                                                                 csr_val,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
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
        (rocsparse::csrmm_core<T, I, J, A, B, C>(handle,
                                                 trans_A,
                                                 trans_B,
                                                 alg,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 batch_count_A,
                                                 offsets_batch_stride_A,
                                                 columns_values_batch_stride_A,
                                                 static_cast<const T*>(alpha),
                                                 descr,
                                                 static_cast<const A*>(csr_val),
                                                 static_cast<const I*>(csr_row_ptr),
                                                 static_cast<const J*>(csr_col_ind),
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
                                                 order_C,
                                                 temp_buffer,
                                                 force_conj_A)));
    return rocsparse_status_success;
}
namespace rocsparse
{
    template <typename T>
    rocsparse_status csrmm_impl(rocsparse_handle          handle,
                                rocsparse_operation       trans_A,
                                rocsparse_operation       trans_B,
                                rocsparse_csrmm_alg       alg,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                rocsparse_int             k,
                                rocsparse_int             nnz,
                                rocsparse_int             batch_count_A,
                                int64_t                   offsets_batch_stride_A,
                                int64_t                   columns_values_batch_stride_A,
                                const T*                  alpha,
                                const rocsparse_mat_descr descr,
                                const T*                  csr_val,
                                const rocsparse_int*      csr_row_ptr,
                                const rocsparse_int*      csr_col_ind,
                                const T*                  dense_B,
                                int64_t                   ldb,
                                rocsparse_int             batch_count_B,
                                int64_t                   batch_stride_B,
                                rocsparse_order           order_B,
                                const T*                  beta,
                                T*                        dense_C,
                                int64_t                   ldc,
                                rocsparse_int             batch_count_C,
                                int64_t                   batch_stride_C,
                                rocsparse_order           order_C,
                                void*                     temp_buffer,
                                bool                      force_conj_A)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrmm"),
                             trans_A,
                             trans_B,
                             m,
                             n,
                             k,
                             nnz,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)dense_B,
                             ldb,
                             LOG_TRACE_SCALAR_VALUE(handle, beta),
                             (const void*&)dense_C,
                             ldc);
        const rocsparse_status status = rocsparse::csrmm_checkarg(handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  nnz,
                                                                  alpha,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
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
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_core(handle,
                                                        trans_A,
                                                        trans_B,
                                                        alg,
                                                        m,
                                                        n,
                                                        k,
                                                        nnz,
                                                        batch_count_A,
                                                        offsets_batch_stride_A,
                                                        columns_values_batch_stride_A,
                                                        alpha,
                                                        descr,
                                                        csr_val,
                                                        csr_row_ptr,
                                                        csr_col_ind,
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
                                                        temp_buffer,
                                                        force_conj_A));
        return rocsparse_status_success;
    }
}
#define INSTANTIATE(T, I, J, A, B, C)                                      \
    template rocsparse_status rocsparse::csrmm_template<T, I, J, A, B, C>( \
        rocsparse_handle          handle,                                  \
        rocsparse_operation       trans_A,                                 \
        rocsparse_operation       trans_B,                                 \
        rocsparse_csrmm_alg       alg,                                     \
        int64_t                   m,                                       \
        int64_t                   n,                                       \
        int64_t                   k,                                       \
        int64_t                   nnz,                                     \
        int64_t                   batch_count_A,                           \
        int64_t                   offsets_batch_stride_A,                  \
        int64_t                   columns_values_batch_stride_A,           \
        const void*               alpha,                                   \
        const rocsparse_mat_descr descr,                                   \
        const void*               csr_val,                                 \
        const void*               csr_row_ptr,                             \
        const void*               csr_col_ind,                             \
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
        rocsparse_order           order_C,                                 \
        void*                     temp_buffer,                             \
        bool                      force_conj_A);

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
#define C_IMPL(NAME, TYPE)                                                             \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                 \
                                     rocsparse_operation       trans_A,                \
                                     rocsparse_operation       trans_B,                \
                                     rocsparse_int             m,                      \
                                     rocsparse_int             n,                      \
                                     rocsparse_int             k,                      \
                                     rocsparse_int             nnz,                    \
                                     const TYPE*               alpha,                  \
                                     const rocsparse_mat_descr descr,                  \
                                     const TYPE*               csr_val,                \
                                     const rocsparse_int*      csr_row_ptr,            \
                                     const rocsparse_int*      csr_col_ind,            \
                                     const TYPE*               dense_B,                \
                                     rocsparse_int             ldb,                    \
                                     const TYPE*               beta,                   \
                                     TYPE*                     dense_C,                \
                                     rocsparse_int             ldc)                    \
    try                                                                                \
    {                                                                                  \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_impl(handle,                        \
                                                        trans_A,                       \
                                                        trans_B,                       \
                                                        rocsparse_csrmm_alg_default,   \
                                                        m,                             \
                                                        n,                             \
                                                        k,                             \
                                                        nnz,                           \
                                                        static_cast<rocsparse_int>(1), \
                                                        static_cast<int64_t>(0),       \
                                                        static_cast<int64_t>(0),       \
                                                        alpha,                         \
                                                        descr,                         \
                                                        csr_val,                       \
                                                        csr_row_ptr,                   \
                                                        csr_col_ind,                   \
                                                        dense_B,                       \
                                                        ldb,                           \
                                                        static_cast<rocsparse_int>(1), \
                                                        static_cast<int64_t>(0),       \
                                                        rocsparse_order_column,        \
                                                        beta,                          \
                                                        dense_C,                       \
                                                        ldc,                           \
                                                        static_cast<rocsparse_int>(1), \
                                                        static_cast<int64_t>(0),       \
                                                        rocsparse_order_column,        \
                                                        nullptr,                       \
                                                        false));                       \
        return rocsparse_status_success;                                               \
    }                                                                                  \
    catch(...)                                                                         \
    {                                                                                  \
        RETURN_ROCSPARSE_EXCEPTION();                                                  \
    }
C_IMPL(rocsparse_scsrmm, float);
C_IMPL(rocsparse_dcsrmm, double);
C_IMPL(rocsparse_ccsrmm, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrmm, rocsparse_double_complex);
#undef C_IMPL
