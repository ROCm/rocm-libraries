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

#include "rocsparse_csrmm_merge_path.hpp"

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_buffer_size_merge_path_kernel_dispatch(rocsparse_handle          handle,
                                                                  rocsparse_operation       trans_A,
                                                                  J                         m,
                                                                  J                         n,
                                                                  J                         k,
                                                                  I                         nnz,
                                                                  const rocsparse_mat_descr descr,
                                                                  const A*                  csr_val,
                                                                  const I* csr_row_ptr,
                                                                  const J* csr_col_ind,
                                                                  size_t*  buffer_size);

    template <typename T, typename I, typename J, typename A>
    static rocsparse_status csrmm_buffer_size_merge_path_core(rocsparse_handle          handle,
                                                              rocsparse_operation       trans_A,
                                                              J                         m,
                                                              J                         n,
                                                              J                         k,
                                                              I                         nnz,
                                                              const rocsparse_mat_descr descr,
                                                              const A*                  csr_val,
                                                              const I*                  csr_row_ptr,
                                                              const J*                  csr_col_ind,
                                                              size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size_merge_path_kernel_dispatch<T>(
            handle, trans_A, m, n, k, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size));
        return rocsparse_status_success;
    }

    static rocsparse_status
        csrmm_buffer_size_merge_path_quickreturn(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 int64_t                   m,
                                                 int64_t                   n,
                                                 int64_t                   k,
                                                 int64_t                   nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const void*               csr_val,
                                                 const void*               csr_row_ptr,
                                                 const void*               csr_col_ind,
                                                 size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0 || n == 0 || k == 0)
        {
            buffer_size[0] = 0;
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J, typename A>
rocsparse_status rocsparse::csrmm_buffer_size_merge_path_template(rocsparse_handle          handle,
                                                                  rocsparse_operation       trans_A,
                                                                  int64_t                   m,
                                                                  int64_t                   n,
                                                                  int64_t                   k,
                                                                  int64_t                   nnz,
                                                                  const rocsparse_mat_descr descr,
                                                                  const void*               csr_val,
                                                                  const void* csr_row_ptr,
                                                                  const void* csr_col_ind,
                                                                  size_t*     buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::csrmm_buffer_size_merge_path_quickreturn(
        handle, trans_A, m, n, k, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR((
        rocsparse::csrmm_buffer_size_merge_path_core<T, I, J, A>(handle,
                                                                 trans_A,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 nnz,
                                                                 descr,
                                                                 static_cast<const A*>(csr_val),
                                                                 static_cast<const I*>(csr_row_ptr),
                                                                 static_cast<const J*>(csr_col_ind),
                                                                 buffer_size)));

    return rocsparse_status_success;
}

#define INSTANTIATE_BUFFER_SIZE(T, I, J, A)                                                 \
    template rocsparse_status rocsparse::csrmm_buffer_size_merge_path_template<T, I, J, A>( \
        rocsparse_handle          handle,                                                   \
        rocsparse_operation       trans_A,                                                  \
        int64_t                   m,                                                        \
        int64_t                   n,                                                        \
        int64_t                   k,                                                        \
        int64_t                   nnz,                                                      \
        const rocsparse_mat_descr descr,                                                    \
        const void*               csr_val,                                                  \
        const void*               csr_row_ptr,                                              \
        const void*               csr_col_ind,                                              \
        size_t*                   buffer_size);

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