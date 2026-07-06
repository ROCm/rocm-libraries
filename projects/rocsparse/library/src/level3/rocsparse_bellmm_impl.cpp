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

#include "rocsparse_bellmm.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status bellmm_template_general(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_direction       dir_A,
                                             I                         mb,
                                             I                         n,
                                             I                         kb,
                                             I                         bell_cols,
                                             I                         bell_block_dim,
                                             const T*                  alpha,
                                             const rocsparse_mat_descr descr,
                                             const I*                  bell_col_ind,
                                             const A*                  bell_val,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             rocsparse_order           order_B,
                                             const T*                  beta,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             rocsparse_order           order_C);

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status bellmm_template_dispatch(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_direction       dir_A,
                                              I                         mb,
                                              I                         n,
                                              I                         kb,
                                              I                         bell_cols,
                                              I                         bell_block_dim,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const I*                  bell_col_ind,
                                              const A*                  bell_val,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              rocsparse_order           order_B,
                                              const T*                  beta_device_host,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bellmm_template_general(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     dir_A,
                                                                     mb,
                                                                     n,
                                                                     kb,
                                                                     bell_cols,
                                                                     bell_block_dim,
                                                                     alpha_device_host,
                                                                     descr,
                                                                     bell_col_ind,
                                                                     bell_val,
                                                                     dense_B,
                                                                     ldb,
                                                                     order_B,
                                                                     beta_device_host,
                                                                     dense_C,
                                                                     ldc,
                                                                     order_C));
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::bellmm_buffer_size(rocsparse_handle          handle,
                                               rocsparse_operation       trans_A,
                                               rocsparse_operation       trans_B,
                                               rocsparse_direction       dir_A,
                                               int64_t                   mb,
                                               int64_t                   n,
                                               int64_t                   kb,
                                               int64_t                   bell_cols,
                                               int64_t                   bell_block_dim,
                                               rocsparse_datatype        alpha_datatype,
                                               const void*               alpha,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_indextype       bell_col_ind_indextype,
                                               const void*               bell_col_ind,
                                               rocsparse_datatype        bell_val_datatype,
                                               const void*               bell_val,
                                               rocsparse_datatype        dense_B_datatype,
                                               const void*               dense_B,
                                               int64_t                   ldb,
                                               rocsparse_order           order_B,
                                               rocsparse_datatype        beta_datatype,
                                               const void*               beta,
                                               rocsparse_datatype        dense_C_datatype,
                                               void*                     dense_C,
                                               int64_t                   ldc,
                                               rocsparse_order           order_C,
                                               size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    *buffer_size = 0;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::bellmm_preprocess(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_direction       dir_A,
                                              int64_t                   mb,
                                              int64_t                   n,
                                              int64_t                   kb,
                                              int64_t                   bell_cols,
                                              int64_t                   bell_block_dim,
                                              rocsparse_datatype        alpha_datatype,
                                              const void*               alpha,
                                              const rocsparse_mat_descr descr,
                                              rocsparse_indextype       bell_col_ind_indextype,
                                              const void*               bell_col_ind,
                                              rocsparse_datatype        bell_datatype,
                                              const void*               bell_val,
                                              rocsparse_datatype        dense_B_datatype,
                                              const void*               dense_B,
                                              int64_t                   ldb,
                                              rocsparse_order           order_B,
                                              rocsparse_datatype        beta_datatype,
                                              const void*               beta,
                                              rocsparse_datatype        dense_C_datatype,
                                              void*                     dense_C,
                                              int64_t                   ldc,
                                              rocsparse_order           order_C,
                                              void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    return rocsparse_status_success;
}

namespace rocsparse
{
    static rocsparse_status bellmm_quickreturn(rocsparse_handle          handle,
                                               rocsparse_operation       trans_A,
                                               rocsparse_operation       trans_B,
                                               rocsparse_direction       dir_A,
                                               int64_t                   mb,
                                               int64_t                   n,
                                               int64_t                   kb,
                                               int64_t                   bell_cols,
                                               int64_t                   block_dim,
                                               int64_t                   batch_count_A,
                                               int64_t                   batch_stride_A,
                                               const void*               alpha,
                                               const rocsparse_mat_descr descr,
                                               const void*               bell_col_ind,
                                               const void*               bell_val,
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
                                               void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(mb == 0 || n == 0 || kb == 0 || bell_cols == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    static rocsparse_status bellmm_checkarg(rocsparse_handle          handle, //0
                                            rocsparse_operation       trans_A, //1
                                            rocsparse_operation       trans_B, //2
                                            rocsparse_direction       dir_A, //3
                                            int64_t                   mb, //4
                                            int64_t                   n, //5
                                            int64_t                   kb, //6
                                            int64_t                   bell_cols, //7
                                            int64_t                   block_dim, //8
                                            int64_t                   batch_count_A, //9
                                            int64_t                   batch_stride_A, //10
                                            const void*               alpha, //11
                                            const rocsparse_mat_descr descr, //12
                                            const void*               bell_col_ind, //13
                                            const void*               bell_val, //14
                                            const void*               dense_B, //15
                                            int64_t                   ldb, //16
                                            int64_t                   batch_count_B, //17
                                            int64_t                   batch_stride_B, //18
                                            rocsparse_order           order_B, //19
                                            const void*               beta, //20
                                            void*                     dense_C, //21
                                            int64_t                   ldc, //22
                                            int64_t                   batch_count_C, //23
                                            int64_t                   batch_stride_C, //24
                                            rocsparse_order           order_C, //25
                                            void*                     temp_buffer) //26
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(12, descr);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_ENUM(19, order_B);
        ROCSPARSE_CHECKARG_ENUM(25, order_C);
        ROCSPARSE_CHECKARG_ENUM(3, dir_A);
        ROCSPARSE_CHECKARG(12,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(12,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_SIZE(4, mb);
        ROCSPARSE_CHECKARG_SIZE(5, n);
        ROCSPARSE_CHECKARG_SIZE(6, kb);
        ROCSPARSE_CHECKARG_SIZE(7, bell_cols);
        ROCSPARSE_CHECKARG_SIZE(8, block_dim);
        ROCSPARSE_CHECKARG(8, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        const rocsparse_status status = rocsparse::bellmm_quickreturn(handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      dir_A,
                                                                      mb,
                                                                      n,
                                                                      kb,
                                                                      bell_cols,
                                                                      block_dim,
                                                                      batch_count_A,
                                                                      batch_stride_A,
                                                                      alpha,
                                                                      descr,
                                                                      bell_col_ind,
                                                                      bell_val,
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
                                                                      temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(12, alpha);
        ROCSPARSE_CHECKARG_POINTER(13, bell_col_ind);
        ROCSPARSE_CHECKARG_POINTER(14, bell_val);
        ROCSPARSE_CHECKARG_POINTER(15, dense_B);
        ROCSPARSE_CHECKARG_POINTER(20, beta);
        ROCSPARSE_CHECKARG_POINTER(21, dense_C);

        ROCSPARSE_CHECKARG(
            1, trans_A, (trans_A != rocsparse_operation_none), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(
            1, trans_A, (trans_A != rocsparse_operation_none), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(
            16,
            ldb,
            ((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column)
             || (trans_B != rocsparse_operation_none && order_B != rocsparse_order_column))
                && (ldb < kb * block_dim),
            rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            16,
            ldb,
            (!((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column)
               || (trans_B != rocsparse_operation_none && order_B != rocsparse_order_column)))
                && (ldb < n),
            rocsparse_status_invalid_size);

        ROCSPARSE_CHECKARG(22,
                           ldc,
                           (ldc < mb * block_dim && order_C == rocsparse_order_column),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            22, ldc, (ldc < n && order_C == rocsparse_order_row), rocsparse_status_invalid_size);

        ROCSPARSE_CHECKARG(9, batch_count_A, (batch_count_A != 1), rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(17, batch_count_B, (batch_count_B != 1), rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(23, batch_count_C, (batch_count_C != 1), rocsparse_status_invalid_value);

        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename A, typename B, typename C>
rocsparse_status rocsparse::bellmm_template(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_direction       dir_A,
                                            int64_t                   mb,
                                            int64_t                   n,
                                            int64_t                   kb,
                                            int64_t                   bell_cols,
                                            int64_t                   block_dim,
                                            int64_t                   batch_count_A,
                                            int64_t                   batch_stride_A,
                                            const void*               alpha,
                                            const rocsparse_mat_descr descr,
                                            const void*               bell_col_ind,
                                            const void*               bell_val,
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
                                            void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xbellmm"),
                         trans_A,
                         trans_B,
                         dir_A,
                         mb,
                         n,
                         kb,
                         bell_cols,
                         block_dim,
                         batch_count_A,
                         batch_stride_A,
                         (const void*&)alpha,
                         (const void*&)descr,
                         (const void*&)bell_col_ind,
                         (const void*&)bell_val,
                         (const void*&)dense_B,
                         ldb,
                         batch_count_B,
                         batch_stride_B,
                         order_B,
                         (const void*&)beta,
                         (const void*&)dense_C,
                         ldc,
                         batch_count_C,
                         batch_stride_C,
                         order_C,
                         temp_buffer);

    const rocsparse_status status = rocsparse::bellmm_checkarg(handle,
                                                               trans_A,
                                                               trans_B,
                                                               dir_A,
                                                               mb,
                                                               n,
                                                               kb,
                                                               bell_cols,
                                                               block_dim,
                                                               batch_count_A,
                                                               batch_stride_A,
                                                               alpha,
                                                               descr,
                                                               bell_col_ind,
                                                               bell_val,
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
                                                               temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::bellmm_template_dispatch<T, I, A, B, C>(handle,
                                                            trans_A,
                                                            trans_B,
                                                            dir_A,
                                                            mb,
                                                            n,
                                                            kb,
                                                            bell_cols,
                                                            block_dim,
                                                            static_cast<const T*>(alpha),
                                                            descr,
                                                            static_cast<const I*>(bell_col_ind),
                                                            static_cast<const A*>(bell_val),
                                                            static_cast<const B*>(dense_B),
                                                            ldb,
                                                            order_B,
                                                            static_cast<const T*>(beta),
                                                            static_cast<C*>(dense_C),
                                                            ldc,
                                                            order_C)));
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I, A, B, C)                                       \
    template rocsparse_status rocsparse::bellmm_template<T, I, A, B, C>( \
        rocsparse_handle          handle,                                \
        rocsparse_operation       trans_A,                               \
        rocsparse_operation       trans_B,                               \
        rocsparse_direction       dir_A,                                 \
        int64_t                   mb,                                    \
        int64_t                   n,                                     \
        int64_t                   kb,                                    \
        int64_t                   bell_cols,                             \
        int64_t                   bell_block_dim,                        \
        int64_t                   batch_count_A,                         \
        int64_t                   batch_stride_A,                        \
        const void*               alpha,                                 \
        const rocsparse_mat_descr descr,                                 \
        const void*               bell_col_ind,                          \
        const void*               bell_val,                              \
        const void*               dense_B,                               \
        int64_t                   ldb,                                   \
        int64_t                   batch_count_B,                         \
        int64_t                   batch_stride_B,                        \
        rocsparse_order           order_B,                               \
        const void*               beta,                                  \
        void*                     dense_C,                               \
        int64_t                   ldc,                                   \
        int64_t                   batch_count_C,                         \
        int64_t                   batch_stride_C,                        \
        rocsparse_order           order_C,                               \
        void*                     temp_buffer)

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
INSTANTIATE(int32_t, int32_t, int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int32_t, int32_t);
INSTANTIATE(float, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE
