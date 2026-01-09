/*! \file */
/* ************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrmm_merge_path.hpp"
#include "rocsparse_csrmm_quickreturn.hpp"
#include "rocsparse_utility.hpp"
#include <algorithm>

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_merge_path_kernel_dispatch(rocsparse_handle          handle,
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
    static rocsparse_status csrmm_merge_path_core(rocsparse_handle    handle,
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

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_merge_path_kernel_dispatch(handle,
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
}
template <typename T, typename I, typename J, typename A, typename B, typename C>
rocsparse_status rocsparse::csrmm_merge_path_template(rocsparse_handle    handle,
                                                      rocsparse_operation trans_A,
                                                      rocsparse_operation trans_B,
                                                      int64_t             m,
                                                      int64_t             n,
                                                      int64_t             k,
                                                      int64_t             nnz,
                                                      int64_t             batch_count_A,
                                                      int64_t             offsets_batch_stride_A,
                                                      int64_t     columns_values_batch_stride_A,
                                                      const void* alpha,
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
        (rocsparse::csrmm_merge_path_core<T, I, J, A, B, C>(handle,
                                                            trans_A,
                                                            trans_B,
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

#define INSTANTIATE(T, I, J, A, B, C)                                                 \
    template rocsparse_status rocsparse::csrmm_merge_path_template<T, I, J, A, B, C>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans_A,                                            \
        rocsparse_operation       trans_B,                                            \
        int64_t                   m,                                                  \
        int64_t                   n,                                                  \
        int64_t                   k,                                                  \
        int64_t                   nnz,                                                \
        int64_t                   batch_count_A,                                      \
        int64_t                   offsets_batch_stride_A,                             \
        int64_t                   columns_values_batch_stride_A,                      \
        const void*               alpha,                                              \
        const rocsparse_mat_descr descr,                                              \
        const void*               csr_val,                                            \
        const void*               csr_row_ptr,                                        \
        const void*               csr_col_ind,                                        \
        const void*               dense_B,                                            \
        int64_t                   ldb,                                                \
        int64_t                   batch_count_B,                                      \
        int64_t                   batch_stride_B,                                     \
        rocsparse_order           order_B,                                            \
        const void*               beta,                                               \
        void*                     dense_C,                                            \
        int64_t                   ldc,                                                \
        int64_t                   batch_count_C,                                      \
        int64_t                   batch_stride_C,                                     \
        rocsparse_order           order_C,                                            \
        void*                     temp_buffer,                                        \
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