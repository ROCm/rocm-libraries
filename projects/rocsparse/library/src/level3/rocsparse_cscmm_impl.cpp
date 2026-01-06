/*! \file */
/* ************************************************************************
* Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_cscmm.hpp"
#include "rocsparse_csrmm.hpp"

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

template <typename T, typename I, typename J, typename A>
rocsparse_status rocsparse::cscmm_buffer_size_template(rocsparse_handle          handle,
                                                       rocsparse_operation       trans_A,
                                                       rocsparse_csrmm_alg       alg,
                                                       int64_t                   m,
                                                       int64_t                   n,
                                                       int64_t                   k,
                                                       int64_t                   nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const void*               csc_val,
                                                       const void*               csc_col_ptr,
                                                       const void*               csc_row_ind,
                                                       size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmm_buffer_size_template<T, I, J, A>(handle,
                                                               rocsparse_operation_transpose,
                                                               alg,
                                                               k,
                                                               n,
                                                               m,
                                                               nnz,
                                                               descr,
                                                               csc_val,
                                                               csc_col_ptr,
                                                               csc_row_ind,
                                                               buffer_size)));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmm_buffer_size_template<T, I, J, A>(handle,
                                                               rocsparse_operation_none,
                                                               alg,
                                                               k,
                                                               n,
                                                               m,
                                                               nnz,
                                                               descr,
                                                               csc_val,
                                                               csc_col_ptr,
                                                               csc_row_ind,
                                                               buffer_size)));
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}

template <typename I, typename J, typename A>
rocsparse_status rocsparse::cscmm_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_csrmm_alg       alg,
                                                    int64_t                   m,
                                                    int64_t                   n,
                                                    int64_t                   k,
                                                    int64_t                   nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               csc_val,
                                                    const void*               csc_col_ptr,
                                                    const void*               csc_row_ind,
                                                    void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmm_analysis_template<I, J, A>(handle,
                                                         rocsparse_operation_transpose,
                                                         alg,
                                                         k,
                                                         n,
                                                         m,
                                                         nnz,
                                                         descr,
                                                         csc_val,
                                                         csc_col_ptr,
                                                         csc_row_ind,
                                                         temp_buffer)));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmm_analysis_template<I, J, A>(handle,
                                                         rocsparse_operation_none,
                                                         alg,
                                                         k,
                                                         n,
                                                         m,
                                                         nnz,
                                                         descr,
                                                         csc_val,
                                                         csc_col_ptr,
                                                         csc_row_ind,
                                                         temp_buffer)));
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}

template <typename T, typename I, typename J, typename A, typename B, typename C>
rocsparse_status rocsparse::cscmm_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_csrmm_alg       alg,
                                           int64_t                   m,
                                           int64_t                   n,
                                           int64_t                   k,
                                           int64_t                   nnz,
                                           int64_t                   batch_count_A,
                                           int64_t                   offsets_batch_stride_A,
                                           int64_t                   rows_values_batch_stride_A,
                                           const void*               alpha_device_host,
                                           const rocsparse_mat_descr descr,
                                           const void*               csc_val,
                                           const void*               csc_col_ptr,
                                           const void*               csc_row_ind,
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

    switch(trans_A)
    {
    case rocsparse_operation_none:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmm_template<T, I, J, A, B, C>(handle,
                                                         rocsparse_operation_transpose,
                                                         trans_B,
                                                         alg,
                                                         k,
                                                         n,
                                                         m,
                                                         nnz,
                                                         batch_count_A,
                                                         offsets_batch_stride_A,
                                                         rows_values_batch_stride_A,
                                                         alpha_device_host,
                                                         descr,
                                                         csc_val,
                                                         csc_col_ptr,
                                                         csc_row_ind,
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
                                                         temp_buffer,
                                                         false)));
        return rocsparse_status_success;
    }
    case rocsparse_operation_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmm_template<T, I, J, A, B, C>(handle,
                                                         rocsparse_operation_none,
                                                         trans_B,
                                                         alg,
                                                         k,
                                                         n,
                                                         m,
                                                         nnz,
                                                         batch_count_A,
                                                         offsets_batch_stride_A,
                                                         rows_values_batch_stride_A,
                                                         alpha_device_host,
                                                         descr,
                                                         csc_val,
                                                         csc_col_ptr,
                                                         csc_row_ind,
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
                                                         temp_buffer,
                                                         false)));
        return rocsparse_status_success;
    }
    case rocsparse_operation_conjugate_transpose:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmm_template<T, I, J, A, B, C>(handle,
                                                         rocsparse_operation_none,
                                                         trans_B,
                                                         alg,
                                                         k,
                                                         n,
                                                         m,
                                                         nnz,
                                                         batch_count_A,
                                                         offsets_batch_stride_A,
                                                         rows_values_batch_stride_A,
                                                         alpha_device_host,
                                                         descr,
                                                         csc_val,
                                                         csc_col_ptr,
                                                         csc_row_ind,
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
                                                         temp_buffer,
                                                         true)));
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}

#define INSTANTIATE_BUFFER_SIZE(T, I, J, A)                                      \
    template rocsparse_status rocsparse::cscmm_buffer_size_template<T, I, J, A>( \
        rocsparse_handle          handle,                                        \
        rocsparse_operation       trans_A,                                       \
        rocsparse_csrmm_alg       alg,                                           \
        int64_t                   m,                                             \
        int64_t                   n,                                             \
        int64_t                   k,                                             \
        int64_t                   nnz,                                           \
        const rocsparse_mat_descr descr,                                         \
        const void*               csc_val,                                       \
        const void*               csc_col_ptr,                                   \
        const void*               csc_row_ind,                                   \
        size_t*                   buffer_size)

// Uniform precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, float);
INSTANTIATE_BUFFER_SIZE(double, int32_t, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, int64_t, double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE_BUFFER_SIZE(int32_t, int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, int8_t);
#undef INSTANTIATE_BUFFER_SIZE

#define INSTANTIATE_ANALYSIS(I, J, A)                                      \
    template rocsparse_status rocsparse::cscmm_analysis_template<I, J, A>( \
        rocsparse_handle          handle,                                  \
        rocsparse_operation       trans_A,                                 \
        rocsparse_csrmm_alg       alg,                                     \
        int64_t                   m,                                       \
        int64_t                   n,                                       \
        int64_t                   k,                                       \
        int64_t                   nnz,                                     \
        const rocsparse_mat_descr descr,                                   \
        const void*               csc_val,                                 \
        const void*               csc_col_ptr,                             \
        const void*               csc_row_ind,                             \
        void*                     temp_buffer);

// Uniform precisions
INSTANTIATE_ANALYSIS(int32_t, int32_t, float);
INSTANTIATE_ANALYSIS(int64_t, int32_t, float);
INSTANTIATE_ANALYSIS(int64_t, int64_t, float);
INSTANTIATE_ANALYSIS(int32_t, int32_t, double);
INSTANTIATE_ANALYSIS(int64_t, int32_t, double);
INSTANTIATE_ANALYSIS(int64_t, int64_t, double);
INSTANTIATE_ANALYSIS(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_ANALYSIS(int32_t, int32_t, _Float16);
INSTANTIATE_ANALYSIS(int64_t, int32_t, _Float16);
INSTANTIATE_ANALYSIS(int64_t, int64_t, _Float16);
INSTANTIATE_ANALYSIS(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_ANALYSIS(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_ANALYSIS(int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE_ANALYSIS(int32_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int64_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int64_t, int64_t, int8_t);
#undef INSTANTIATE_ANALYSIS

#define INSTANTIATE(T, I, J, A, B, C)                                      \
    template rocsparse_status rocsparse::cscmm_template<T, I, J, A, B, C>( \
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
        int64_t                   rows_values_batch_stride_A,              \
        const void*               alpha_device_host,                       \
        const rocsparse_mat_descr descr,                                   \
        const void*               csc_val,                                 \
        const void*               csc_col_ptr,                             \
        const void*               csc_row_ind,                             \
        const void*               dense_B,                                 \
        int64_t                   ldb,                                     \
        int64_t                   batch_count_B,                           \
        int64_t                   batch_stride_B,                          \
        rocsparse_order           order_B,                                 \
        const void*               beta_device_host,                        \
        void*                     dense_C,                                 \
        int64_t                   ldc,                                     \
        int64_t                   batch_count_C,                           \
        int64_t                   batch_stride_C,                          \
        rocsparse_order           order_C,                                 \
        void*                     temp_buffer);

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
