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

#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_coomm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status coomm_template_atomic(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           I                         m,
                                           I                         n,
                                           I                         k,
                                           int64_t                   nnz,
                                           I                         batch_count_A,
                                           int64_t                   batch_stride_A,
                                           const T*                  alpha_device_host,
                                           const rocsparse_mat_descr descr,
                                           const A*                  coo_val,
                                           const I*                  coo_row_ind,
                                           const I*                  coo_col_ind,
                                           const B*                  dense_B,
                                           int64_t                   ldb,
                                           I                         batch_count_B,
                                           int64_t                   batch_stride_B,
                                           rocsparse_order           order_B,
                                           const T*                  beta_device_host,
                                           C*                        dense_C,
                                           int64_t                   ldc,
                                           I                         batch_count_C,
                                           int64_t                   batch_stride_C,
                                           rocsparse_order           order_C);

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status coomm_template_segmented_atomic(rocsparse_handle          handle,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     I                         m,
                                                     I                         n,
                                                     I                         k,
                                                     int64_t                   nnz,
                                                     I                         batch_count_A,
                                                     int64_t                   batch_stride_A,
                                                     const T*                  alpha_device_host,
                                                     const rocsparse_mat_descr descr,
                                                     const A*                  coo_val,
                                                     const I*                  coo_row_ind,
                                                     const I*                  coo_col_ind,
                                                     const B*                  dense_B,
                                                     int64_t                   ldb,
                                                     I                         batch_count_B,
                                                     int64_t                   batch_stride_B,
                                                     rocsparse_order           order_B,
                                                     const T*                  beta_device_host,
                                                     C*                        dense_C,
                                                     int64_t                   ldc,
                                                     I                         batch_count_C,
                                                     int64_t                   batch_stride_C,
                                                     rocsparse_order           order_C);

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status coomm_template_segmented(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              I                         m,
                                              I                         n,
                                              I                         k,
                                              int64_t                   nnz,
                                              I                         batch_count_A,
                                              int64_t                   batch_stride_A,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const A*                  coo_val,
                                              const I*                  coo_row_ind,
                                              const I*                  coo_col_ind,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              I                         batch_count_B,
                                              int64_t                   batch_stride_B,
                                              rocsparse_order           order_B,
                                              const T*                  beta_device_host,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              I                         batch_count_C,
                                              int64_t                   batch_stride_C,
                                              rocsparse_order           order_C,
                                              void*                     temp_buffer);
}

template <typename T, typename I, typename A, typename B, typename C>
rocsparse_status rocsparse::coomm_template_dispatch(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_coomm_alg       alg,
                                                    I                         m,
                                                    I                         n,
                                                    I                         k,
                                                    int64_t                   nnz,
                                                    I                         batch_count_A,
                                                    int64_t                   batch_stride_A,
                                                    const T*                  alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const A*                  coo_val,
                                                    const I*                  coo_row_ind,
                                                    const I*                  coo_col_ind,
                                                    const B*                  dense_B,
                                                    int64_t                   ldb,
                                                    I                         batch_count_B,
                                                    int64_t                   batch_stride_B,
                                                    rocsparse_order           order_B,
                                                    const T*                  beta_device_host,
                                                    C*                        dense_C,
                                                    int64_t                   ldc,
                                                    I                         batch_count_C,
                                                    int64_t                   batch_stride_C,
                                                    rocsparse_order           order_C,
                                                    void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(trans_A == rocsparse_operation_none)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_2d_array(
            handle, m, n, ldc, batch_count_C, batch_stride_C, beta_device_host, dense_C, order_C));
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_2d_array(
            handle, k, n, ldc, batch_count_C, batch_stride_C, beta_device_host, dense_C, order_C));
    }

    switch(alg)
    {
    case rocsparse_coomm_alg_default:
    case rocsparse_coomm_alg_atomic:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_template_atomic(handle,
                                                                   trans_A,
                                                                   trans_B,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   nnz,
                                                                   batch_count_A,
                                                                   batch_stride_A,
                                                                   alpha_device_host,
                                                                   descr,
                                                                   coo_val,
                                                                   coo_row_ind,
                                                                   coo_col_ind,
                                                                   dense_B,
                                                                   ldb,
                                                                   batch_count_B,
                                                                   batch_stride_B,
                                                                   order_B,
                                                                   beta_device_host,
                                                                   dense_C,
                                                                   ldc,
                                                                   batch_count_C,
                                                                   batch_stride_C,
                                                                   order_C));
        return rocsparse_status_success;
    }

    case rocsparse_coomm_alg_segmented:
    {
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_template_segmented(handle,
                                                                          trans_A,
                                                                          trans_B,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          nnz,
                                                                          batch_count_A,
                                                                          batch_stride_A,
                                                                          alpha_device_host,
                                                                          descr,
                                                                          coo_val,
                                                                          coo_row_ind,
                                                                          coo_col_ind,
                                                                          dense_B,
                                                                          ldb,
                                                                          batch_count_B,
                                                                          batch_stride_B,
                                                                          order_B,
                                                                          beta_device_host,
                                                                          dense_C,
                                                                          ldc,
                                                                          batch_count_C,
                                                                          batch_stride_C,
                                                                          order_C,
                                                                          temp_buffer));
            return rocsparse_status_success;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_template_atomic(handle,
                                                                       trans_A,
                                                                       trans_B,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       nnz,
                                                                       batch_count_A,
                                                                       batch_stride_A,
                                                                       alpha_device_host,
                                                                       descr,
                                                                       coo_val,
                                                                       coo_row_ind,
                                                                       coo_col_ind,
                                                                       dense_B,
                                                                       ldb,
                                                                       batch_count_B,
                                                                       batch_stride_B,
                                                                       order_B,
                                                                       beta_device_host,
                                                                       dense_C,
                                                                       ldc,
                                                                       batch_count_C,
                                                                       batch_stride_C,
                                                                       order_C));
            return rocsparse_status_success;
        }
        }
    }

    case rocsparse_coomm_alg_segmented_atomic:
    {
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_template_segmented_atomic(handle,
                                                                                 trans_A,
                                                                                 trans_B,
                                                                                 m,
                                                                                 n,
                                                                                 k,
                                                                                 nnz,
                                                                                 batch_count_A,
                                                                                 batch_stride_A,
                                                                                 alpha_device_host,
                                                                                 descr,
                                                                                 coo_val,
                                                                                 coo_row_ind,
                                                                                 coo_col_ind,
                                                                                 dense_B,
                                                                                 ldb,
                                                                                 batch_count_B,
                                                                                 batch_stride_B,
                                                                                 order_B,
                                                                                 beta_device_host,
                                                                                 dense_C,
                                                                                 ldc,
                                                                                 batch_count_C,
                                                                                 batch_stride_C,
                                                                                 order_C));
            return rocsparse_status_success;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_template_atomic(handle,
                                                                       trans_A,
                                                                       trans_B,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       nnz,
                                                                       batch_count_A,
                                                                       batch_stride_A,
                                                                       alpha_device_host,
                                                                       descr,
                                                                       coo_val,
                                                                       coo_row_ind,
                                                                       coo_col_ind,
                                                                       dense_B,
                                                                       ldb,
                                                                       batch_count_B,
                                                                       batch_stride_B,
                                                                       order_B,
                                                                       beta_device_host,
                                                                       dense_C,
                                                                       ldc,
                                                                       batch_count_C,
                                                                       batch_stride_C,
                                                                       order_C));
            return rocsparse_status_success;
        }
        }
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

namespace rocsparse
{
    template <typename T, typename I, typename A, typename B, typename C>
    static rocsparse_status coomm_core(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_coomm_alg       alg,
                                       I                         m,
                                       I                         n,
                                       I                         k,
                                       int64_t                   nnz,
                                       I                         batch_count_A,
                                       int64_t                   batch_stride_A,
                                       const T*                  alpha_device_host,
                                       const rocsparse_mat_descr descr,
                                       const A*                  coo_val,
                                       const I*                  coo_row_ind,
                                       const I*                  coo_col_ind,
                                       const B*                  dense_B,
                                       int64_t                   ldb,
                                       I                         batch_count_B,
                                       int64_t                   batch_stride_B,
                                       rocsparse_order           order_B,
                                       const T*                  beta_device_host,
                                       C*                        dense_C,
                                       int64_t                   ldc,
                                       I                         batch_count_C,
                                       int64_t                   batch_stride_C,
                                       rocsparse_order           order_C,
                                       void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_template_dispatch(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     alg,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     nnz,
                                                                     batch_count_A,
                                                                     batch_stride_A,
                                                                     alpha_device_host,
                                                                     descr,
                                                                     coo_val,
                                                                     coo_row_ind,
                                                                     coo_col_ind,
                                                                     dense_B,
                                                                     ldb,
                                                                     batch_count_B,
                                                                     batch_stride_B,
                                                                     order_B,
                                                                     beta_device_host,
                                                                     dense_C,
                                                                     ldc,
                                                                     batch_count_C,
                                                                     batch_stride_C,
                                                                     order_C,
                                                                     temp_buffer));
        return rocsparse_status_success;
    }

    template <typename T, typename C>
    static rocsparse_status coomm_quickreturn(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_coomm_alg       alg,
                                              int64_t                   m,
                                              int64_t                   n,
                                              int64_t                   k,
                                              int64_t                   nnz,
                                              int64_t                   batch_count_A,
                                              int64_t                   batch_stride_A,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const void*               coo_val,
                                              const void*               coo_row_ind,
                                              const void*               coo_col_ind,
                                              const void*               dense_B,
                                              int64_t                   ldb,
                                              int64_t                   batch_count_B,
                                              int64_t                   batch_stride_B,
                                              rocsparse_order           order_B,
                                              const T*                  beta_device_host,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              int64_t                   batch_count_C,
                                              int64_t                   batch_stride_C,
                                              rocsparse_order           order_C,
                                              void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0 || n == 0 || k == 0)
        {
            // matrix never accessed however still need to update C matrix
            const int64_t Csize = (trans_A == rocsparse_operation_none) ? m * n : k * n;
            if(Csize > 0)
            {
                if(dense_C == nullptr && beta_device_host == nullptr)
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
                                              beta_device_host,
                                              dense_C,
                                              order_C));
            }
            return rocsparse_status_success;
        }

        if(handle->pointer_mode == rocsparse_pointer_mode_host
           && ((alpha_device_host != nullptr) && (*alpha_device_host == static_cast<T>(0)))
           && ((beta_device_host != nullptr) && (*beta_device_host == static_cast<T>(1))))
        {
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    template <typename T, typename C>
    static rocsparse_status coomm_checkarg(rocsparse_handle          handle, //0
                                           rocsparse_operation       trans_A, //1
                                           rocsparse_operation       trans_B, //2
                                           rocsparse_coomm_alg       alg, //3
                                           int64_t                   m, //4
                                           int64_t                   n, //5
                                           int64_t                   k, //6
                                           int64_t                   nnz, //7
                                           int64_t                   batch_count_A, //8
                                           int64_t                   batch_stride_A, //9
                                           const T*                  alpha_device_host, //10
                                           const rocsparse_mat_descr descr, //11
                                           const void*               coo_val, //12
                                           const void*               coo_row_ind, //13
                                           const void*               coo_col_ind, //14
                                           const void*               dense_B, //15
                                           int64_t                   ldb, //16
                                           int64_t                   batch_count_B, //17
                                           int64_t                   batch_stride_B, //18
                                           rocsparse_order           order_B, //19
                                           const T*                  beta_device_host, //20
                                           C*                        dense_C, //21
                                           int64_t                   ldc, //22
                                           int64_t                   batch_count_C, //23
                                           int64_t                   batch_stride_C, //24
                                           rocsparse_order           order_C, //25
                                           void*                     temp_buffer) //26
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_ENUM(19, order_B);
        ROCSPARSE_CHECKARG_ENUM(25, order_C);

        ROCSPARSE_CHECKARG(25, order_C, (order_C != order_B), rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG_ENUM(3, alg);
        ROCSPARSE_CHECKARG_SIZE(4, m);
        ROCSPARSE_CHECKARG_SIZE(5, n);
        ROCSPARSE_CHECKARG_SIZE(6, k);
        ROCSPARSE_CHECKARG_SIZE(7, nnz);
        ROCSPARSE_CHECKARG_POINTER(11, descr);
        ROCSPARSE_CHECKARG(11,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(11,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_POINTER(10, alpha_device_host);
        ROCSPARSE_CHECKARG_POINTER(20, beta_device_host);
        const rocsparse_status status = rocsparse::coomm_quickreturn<T>(handle,
                                                                        trans_A,
                                                                        trans_B,
                                                                        alg,
                                                                        m,
                                                                        n,
                                                                        k,
                                                                        nnz,
                                                                        batch_count_A,
                                                                        batch_stride_A,
                                                                        alpha_device_host,
                                                                        descr,
                                                                        coo_val,
                                                                        coo_row_ind,
                                                                        coo_col_ind,
                                                                        dense_B,
                                                                        ldb,
                                                                        batch_count_B,
                                                                        batch_stride_B,
                                                                        order_B,
                                                                        beta_device_host,
                                                                        dense_C,
                                                                        ldc,
                                                                        batch_count_C,
                                                                        batch_stride_C,
                                                                        order_C,
                                                                        temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_ARRAY(11, nnz, coo_val);
        ROCSPARSE_CHECKARG_ARRAY(13, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(14, nnz, coo_col_ind);

        ROCSPARSE_CHECKARG_POINTER(15, dense_B);

        ROCSPARSE_CHECKARG_POINTER(21, dense_C);

        // Check leading dimension of matrices
        static constexpr int64_t s_one = static_cast<int64_t>(1);
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            ROCSPARSE_CHECKARG(
                22,
                ldc,
                (ldc < rocsparse::max(s_one, ((order_C == rocsparse_order_column) ? m : n))),
                rocsparse_status_invalid_size);

            // Check leading dimension of B
            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(
                    16,
                    ldb,
                    (ldb < rocsparse::max(s_one, ((order_B == rocsparse_order_column) ? k : n))),
                    rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    16,
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
                22,
                ldc,
                (ldc < rocsparse::max(s_one, ((order_C == rocsparse_order_column) ? k : n))),
                rocsparse_status_invalid_size);

            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(
                    16,
                    ldb,
                    (ldb < rocsparse::max(s_one, ((order_B == rocsparse_order_column) ? m : n))),
                    rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    16,
                    ldb,
                    (ldb < rocsparse::max(s_one, ((order_B == rocsparse_order_column) ? n : m))),
                    rocsparse_status_invalid_size);
                break;
            }
            }
            break;
        }
        }

        ROCSPARSE_CHECKARG(23,
                           batch_count_C,
                           ((batch_count_A == 1) && (batch_count_C != batch_count_B)),
                           rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(23,
                           batch_count_C,
                           ((batch_count_B == 1) && (batch_count_C != batch_count_A)),
                           rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(
            9,
            batch_count_A,
            (((batch_count_A > 1) && (batch_count_B > 1))
             && ((batch_count_A != batch_count_B) || (batch_count_A != batch_count_C))),
            rocsparse_status_invalid_value);

        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename A, typename B, typename C>
rocsparse_status rocsparse::coomm_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_coomm_alg       alg,
                                           int64_t                   m,
                                           int64_t                   n,
                                           int64_t                   k,
                                           int64_t                   nnz,
                                           int64_t                   batch_count_A,
                                           int64_t                   batch_stride_A,
                                           const void*               alpha_device_host,
                                           const rocsparse_mat_descr descr,
                                           const void*               coo_val,
                                           const void*               coo_row_ind,
                                           const void*               coo_col_ind,
                                           const void*               dense_B,
                                           int64_t                   ldb,
                                           int64_t                   batch_count_B,
                                           int64_t                   batch_stride_B,
                                           rocsparse_order           order_B,
                                           const void*               beta_device_host,
                                           void*                     dense_C,
                                           int64_t                   ldc,
                                           int64_t                   batch_count_C,
                                           int64_t                   batch_stride_C,
                                           rocsparse_order           order_C,
                                           void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status
        = rocsparse::coomm_quickreturn(handle,
                                       trans_A,
                                       trans_B,
                                       alg,
                                       m,
                                       n,
                                       k,
                                       nnz,
                                       batch_count_A,
                                       batch_stride_A,
                                       static_cast<const T*>(alpha_device_host),
                                       descr,
                                       coo_val,
                                       coo_row_ind,
                                       coo_col_ind,
                                       dense_B,
                                       ldb,
                                       batch_count_B,
                                       batch_stride_B,
                                       order_B,
                                       static_cast<const T*>(beta_device_host),
                                       static_cast<C*>(dense_C),
                                       ldc,
                                       batch_count_C,
                                       batch_stride_C,
                                       order_C,
                                       temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    const bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    const bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    const bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::coomm_core<T, I, A, B, C>(handle,
                                              trans_A,
                                              trans_B,
                                              alg,
                                              m,
                                              n,
                                              k,
                                              nnz,
                                              batch_count_A,
                                              batch_stride_A,
                                              static_cast<const T*>(alpha_device_host),
                                              descr,
                                              static_cast<const A*>(coo_val),
                                              static_cast<const I*>(coo_row_ind),
                                              static_cast<const I*>(coo_col_ind),
                                              static_cast<const B*>(dense_B),
                                              ldb,
                                              batch_count_B,
                                              batch_stride_B,
                                              order_B,
                                              static_cast<const T*>(beta_device_host),
                                              static_cast<C*>(dense_C),
                                              ldc,
                                              batch_count_C,
                                              batch_stride_C,
                                              order_C,
                                              temp_buffer)));
    return rocsparse_status_success;
}

template <typename T, typename I, typename A, typename B, typename C>
rocsparse_status coomm_impl(rocsparse_handle          handle,
                            rocsparse_operation       trans_A,
                            rocsparse_operation       trans_B,
                            rocsparse_coomm_alg       alg,
                            I                         m,
                            I                         n,
                            I                         k,
                            int64_t                   nnz,
                            I                         batch_count_A,
                            int64_t                   batch_stride_A,
                            const T*                  alpha_device_host,
                            const rocsparse_mat_descr descr,
                            const A*                  coo_val,
                            const I*                  coo_row_ind,
                            const I*                  coo_col_ind,
                            const B*                  dense_B,
                            int64_t                   ldb,
                            I                         batch_count_B,
                            int64_t                   batch_stride_B,
                            rocsparse_order           order_B,
                            const T*                  beta_device_host,
                            C*                        dense_C,
                            int64_t                   ldc,
                            I                         batch_count_C,
                            int64_t                   batch_stride_C,
                            rocsparse_order           order_C,
                            void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcoomm"),
                         trans_A,
                         trans_B,
                         alg,
                         m,
                         n,
                         k,
                         nnz,
                         batch_count_A,
                         batch_stride_A,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         (const void*&)descr,
                         (const void*&)coo_val,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)dense_B,
                         ldb,
                         batch_count_B,
                         batch_stride_B,
                         order_B,
                         LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                         (const void*&)dense_C,
                         ldc,
                         batch_count_C,
                         batch_stride_C,
                         order_C,
                         temp_buffer);

    const rocsparse_status status = rocsparse::coomm_checkarg(handle,
                                                              trans_A,
                                                              trans_B,
                                                              alg,
                                                              m,
                                                              n,
                                                              k,
                                                              nnz,
                                                              batch_count_A,
                                                              batch_stride_A,
                                                              alpha_device_host,
                                                              descr,
                                                              coo_val,
                                                              coo_row_ind,
                                                              coo_col_ind,
                                                              dense_B,
                                                              ldb,
                                                              batch_count_B,
                                                              batch_stride_B,
                                                              order_B,
                                                              beta_device_host,
                                                              dense_C,
                                                              ldc,
                                                              batch_count_C,
                                                              batch_stride_C,
                                                              order_C,
                                                              temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_core(handle,
                                                    trans_A,
                                                    trans_B,
                                                    alg,
                                                    m,
                                                    n,
                                                    k,
                                                    nnz,
                                                    batch_count_A,
                                                    batch_stride_A,
                                                    alpha_device_host,
                                                    descr,
                                                    coo_val,
                                                    coo_row_ind,
                                                    coo_col_ind,
                                                    dense_B,
                                                    ldb,
                                                    batch_count_B,
                                                    batch_stride_B,
                                                    order_B,
                                                    beta_device_host,
                                                    dense_C,
                                                    ldc,
                                                    batch_count_C,
                                                    batch_stride_C,
                                                    order_C,
                                                    temp_buffer));
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I, A, B, C)                                      \
    template rocsparse_status rocsparse::coomm_template<T, I, A, B, C>( \
        rocsparse_handle          handle,                               \
        rocsparse_operation       trans_A,                              \
        rocsparse_operation       trans_B,                              \
        rocsparse_coomm_alg       alg,                                  \
        int64_t                   m,                                    \
        int64_t                   n,                                    \
        int64_t                   k,                                    \
        int64_t                   nnz,                                  \
        int64_t                   batch_count_A,                        \
        int64_t                   batch_stride_A,                       \
        const void*               alpha_device_host,                    \
        const rocsparse_mat_descr descr,                                \
        const void*               coo_val,                              \
        const void*               coo_row_ind,                          \
        const void*               coo_col_ind,                          \
        const void*               dense_B,                              \
        int64_t                   ldb,                                  \
        int64_t                   batch_count_B,                        \
        int64_t                   batch_stride_B,                       \
        rocsparse_order           order_B,                              \
        const void*               beta_device_host,                     \
        void*                     dense_C,                              \
        int64_t                   ldc,                                  \
        int64_t                   batch_count_C,                        \
        int64_t                   batch_stride_C,                       \
        rocsparse_order           order_C,                              \
        void*                     temp_buffer);

// Uniform precisions
INSTANTIATE(float, int32_t, float, float, float);
INSTANTIATE(float, int64_t, float, float, float);
INSTANTIATE(double, int32_t, double, double, double);
INSTANTIATE(double, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed precisions
INSTANTIATE(float, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float);
#undef INSTANTIATE
