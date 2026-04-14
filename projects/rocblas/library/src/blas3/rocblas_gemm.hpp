/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <cstring> // std::memcpy for graph capture use cases

#include "check_numerics_matrix.hpp"
#include "handle.hpp"

/*********************************************************************************
 * Right now Tensile requires alpha and beta to be passed by value on host.      *
 * If in device pointer mode, copy alpha and beta to host.                       *
 * If k == 0, we set alpha = 0 instead of copying from device.                   *
 * TODO: Make this asynchronous, putting synchronization closer to Tensile call. *
 *********************************************************************************/
template <typename Ta, typename Tac, typename Tb, typename Tbc>
rocblas_status rocblas_copy_alpha_beta_to_host_if_on_device(
    rocblas_handle handle, const Ta*& alpha, const Tb*& beta, Tac& alpha_h, Tbc& beta_h, int64_t k)
{
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        if(alpha)
        {
            if(k == 0)
                alpha_h = 0;
            else
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    &alpha_h, alpha, sizeof(Tac), hipMemcpyDeviceToHost, handle->get_stream()));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
            }
            alpha = &alpha_h;
        }
        if(beta)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &beta_h, beta, sizeof(Tbc), hipMemcpyDeviceToHost, handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
            beta = &beta_h;
        }
    }
    return rocblas_status_success;
}

/*******************************************************************************
 * Validate Arguments
 ******************************************************************************/
template <typename API_INT, typename T>
inline rocblas_status rocblas_gemm_arg_check(rocblas_handle    handle,
                                             rocblas_operation trans_a,
                                             rocblas_operation trans_b,
                                             API_INT           m,
                                             API_INT           n,
                                             API_INT           k,
                                             const T*          alpha,
                                             const void*       a,
                                             API_INT           lda,
                                             const void*       b,
                                             API_INT           ldb,
                                             const T*          beta,
                                             const void*       c,
                                             API_INT           ldc,
                                             API_INT           batch_count = 1)
{
    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    if(trans_a != rocblas_operation_none && trans_a != rocblas_operation_transpose
       && trans_a != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(trans_b != rocblas_operation_none && trans_b != rocblas_operation_transpose
       && trans_b != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    API_INT num_rows_a = trans_a == rocblas_operation_none ? m : k;
    API_INT num_rows_b = trans_b == rocblas_operation_none ? k : n;
    API_INT num_rows_c = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc)
        return rocblas_status_invalid_size;

    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*beta == 1)
        {
            if(!k)
                return rocblas_status_success;

            if(!alpha)
                return rocblas_status_invalid_pointer;

            if(!*alpha)
                return rocblas_status_success;
        }
        // all early return success now handled so
        // pointers must be valid
        bool ab_calc_invalid = !alpha || (*alpha != 0 && (!a || !b));
        if(!c || (k && ab_calc_invalid))
            return rocblas_status_invalid_pointer;
    }
    else
    {
        return rocblas_status_internal_error; // always pushed host_mode prevalidation
    }

    return rocblas_status_continue;
}

/*
 * ===========================================================================
 *    template interface
 * ===========================================================================
 */
template <bool BATCHED, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_gemm(rocblas_handle    handle,
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       k,
                                     const TScal*      alpha,
                                     TConstPtr         A,
                                     rocblas_stride    offset_a,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_a,
                                     TConstPtr         B,
                                     rocblas_stride    offset_b,
                                     rocblas_int       ldb,
                                     rocblas_stride    stride_b,
                                     const TScal*      beta,
                                     TPtr              C,
                                     rocblas_stride    offset_c,
                                     rocblas_int       ldc,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemm_template(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const T*          A,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   const T*          B,
                                   rocblas_stride    offset_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   const T*          beta,
                                   T*                C,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemm_batched_template(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const T*          alpha,
                                           const T* const*   A,
                                           rocblas_stride    offset_a,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           const T* const*   B,
                                           rocblas_stride    offset_b,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_b,
                                           const T*          beta,
                                           T* const*         C,
                                           rocblas_stride    offset_c,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_c,
                                           rocblas_int       batch_count);

template <typename TConstPtrA, typename TConstPtrB, typename TPtr>
rocblas_status rocblas_gemm_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           int64_t           m,
                                           int64_t           n,
                                           int64_t           k,
                                           TConstPtrA        A,
                                           rocblas_stride    offset_a,
                                           int64_t           lda,
                                           rocblas_stride    stride_a,
                                           TConstPtrB        B,
                                           rocblas_stride    offset_b,
                                           int64_t           ldb,
                                           rocblas_stride    stride_b,
                                           TPtr              C,
                                           rocblas_stride    offset_c,
                                           int64_t           ldc,
                                           rocblas_stride    stride_c,
                                           int64_t           batch_count,
                                           const int         check_numerics,
                                           bool              is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans_a,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              m,
                                                              k,
                                                              A,
                                                              offset_a,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans_b,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              k,
                                                              n,
                                                              B,
                                                              offset_b,
                                                              ldb,
                                                              stride_b,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }
    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          C,
                                                          offset_c,
                                                          ldc,
                                                          stride_c,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}
