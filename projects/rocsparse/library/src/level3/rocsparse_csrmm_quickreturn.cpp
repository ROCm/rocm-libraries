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
#include "rocsparse_csrmm_quickreturn.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

template <typename T, typename C>
rocsparse_status rocsparse::csrmm_quickreturn(rocsparse_handle          handle,
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

#define INSTANTIATE(T, C)                                                                           \
    template rocsparse_status rocsparse::csrmm_quickreturn<T, C>(rocsparse_handle          handle,  \
                                                                 rocsparse_operation       trans_A, \
                                                                 rocsparse_operation       trans_B, \
                                                                 int64_t                   m,       \
                                                                 int64_t                   n,       \
                                                                 int64_t                   k,       \
                                                                 int64_t                   nnz,     \
                                                                 const T*                  alpha,   \
                                                                 const rocsparse_mat_descr descr,   \
                                                                 const void*               csr_val, \
                                                                 const void*     csr_row_ptr,       \
                                                                 const void*     csr_col_ind,       \
                                                                 const void*     dense_B,           \
                                                                 int64_t         ldb,               \
                                                                 const T*        beta,              \
                                                                 C*              dense_C,           \
                                                                 int64_t         ldc,               \
                                                                 rocsparse_order order_B,           \
                                                                 rocsparse_order order_C,           \
                                                                 int64_t         batch_count_C,     \
                                                                 int64_t         batch_stride_C);

INSTANTIATE(float, float);
INSTANTIATE(double, double);
INSTANTIATE(rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int32_t, int32_t);
#undef INSTANTIATE
