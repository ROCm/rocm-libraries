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
#include "rocsparse_utility.hpp"

#include "csrmm/row_split/kernel_declarations.h"
#include "rocsparse_common.h"

namespace rocsparse
{
    template <typename I, typename J, typename A, typename B, typename C, typename T>
    rocsparse_status csrmmtn_template_transpose(rocsparse_handle handle,
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
                                                rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Scale C with beta
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_2d_array(
            handle, k, n, ldc, batch_count_C, batch_stride_C, beta_device_host, dense_C, order_C));

#define CSRMMTN_DIM 256
#define WF_SIZE 4
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrmmtn_row_split_kernel<CSRMMTN_DIM, WF_SIZE>),
            dim3((m - 1) / (CSRMMTN_DIM / WF_SIZE) + 1, (n - 1) / WF_SIZE + 1, batch_count_C),
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
    rocsparse_status csrmmtt_template_transpose(rocsparse_handle handle,
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
                                                rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Scale C with beta
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_2d_array(
            handle, k, n, ldc, batch_count_C, batch_stride_C, beta_device_host, dense_C, order_C));

#define CSRMMTT_DIM 256
#define WF_SIZE 4
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrmmtt_row_split_kernel<CSRMMTT_DIM, WF_SIZE>),
            dim3((m - 1) / (CSRMMTT_DIM / WF_SIZE) + 1, (n - 1) / WF_SIZE + 1, batch_count_C),
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
         order_C);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_transpose_kernel_dispatch(rocsparse_handle    handle,
                                                     rocsparse_operation trans_A,
                                                     rocsparse_operation trans_B,
                                                     J                   m,
                                                     J                   n,
                                                     J                   k,
                                                     I                   nnz,
                                                     J                   batch_count_A,
                                                     int64_t             offsets_batch_stride_A,
                                                     int64_t  columns_values_batch_stride_A,
                                                     const T* alpha_device_host,
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
                                                     bool                      force_conj_A)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose || force_conj_A);
        const bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

        // Run different csrmv kernels
        if(trans_A == rocsparse_operation_transpose
           || trans_A == rocsparse_operation_conjugate_transpose)
        {
            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                return ROCSPARSE_CSRMM_TEMPLATE_ROW_SPLIT_IMPL(
                    rocsparse::csrmmtn_template_transpose);
            }
            else if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                return ROCSPARSE_CSRMM_TEMPLATE_ROW_SPLIT_IMPL(
                    rocsparse::csrmmtt_template_transpose);
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}
#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE)             \
    template rocsparse_status rocsparse::csrmm_transpose_kernel_dispatch( \
        rocsparse_handle          handle,                                 \
        rocsparse_operation       trans_A,                                \
        rocsparse_operation       trans_B,                                \
        JTYPE                     m,                                      \
        JTYPE                     n,                                      \
        JTYPE                     k,                                      \
        ITYPE                     nnz,                                    \
        JTYPE                     batch_count_A,                          \
        int64_t                   offsets_batch_stride_A,                 \
        int64_t                   columns_values_batch_stride_A,          \
        const TTYPE*              alpha_device_host,                      \
        const rocsparse_mat_descr descr,                                  \
        const ATYPE*              csr_val,                                \
        const ITYPE*              csr_row_ptr,                            \
        const JTYPE*              csr_col_ind,                            \
        const BTYPE*              dense_B,                                \
        int64_t                   ldb,                                    \
        JTYPE                     batch_count_B,                          \
        int64_t                   batch_stride_B,                         \
        rocsparse_order           order_B,                                \
        const TTYPE*              beta_device_host,                       \
        CTYPE*                    dense_C,                                \
        int64_t                   ldc,                                    \
        JTYPE                     batch_count_C,                          \
        int64_t                   batch_stride_C,                         \
        rocsparse_order           order_C,                                \
        bool                      force_conj_A)

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
