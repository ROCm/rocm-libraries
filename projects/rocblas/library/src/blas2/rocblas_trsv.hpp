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

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_block_sizes.h"

template <typename API_INT, typename U, typename V>
inline rocblas_status rocblas_trsv_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             API_INT           n,
                                             U                 A,
                                             API_INT           lda,
                                             V                 B,
                                             API_INT           incx,
                                             API_INT           batch_count,
                                             size_t&           dev_bytes)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
       && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)
        return rocblas_status_invalid_value;

    if(n < 0 || lda < n || lda < 1 || !incx || batch_count < 0)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!n || !batch_count)
    {
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
        return rocblas_status_success;
    }

    // Need one int worth of global memory to keep track of completed sections
    dev_bytes = sizeof(rocblas_int) * batch_count;
    if(handle->is_device_memory_size_query())
    {
        return handle->set_optimal_device_memory_size(dev_bytes);
    }

    // pointers are validated if they need to be dereferenced
    if(!A || !B)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocblas_internal_trsv_check_numerics(const char*       function_name,
                                                    rocblas_handle    handle,
                                                    rocblas_fill      uplo,
                                                    int64_t           n,
                                                    T                 A,
                                                    rocblas_stride    offset_a,
                                                    int64_t           lda,
                                                    rocblas_stride    stride_a,
                                                    U                 x,
                                                    rocblas_stride    offset_x,
                                                    int64_t           inc_x,
                                                    rocblas_stride    stride_x,
                                                    int64_t           batch_count,
                                                    const rocblas_int check_numerics,
                                                    bool              is_input);

// used in trsm
template <rocblas_int DIM_X, typename T, typename ATYPE, typename XTYPE>
rocblas_status rocblas_internal_trsv_substitution_template(rocblas_handle    handle,
                                                           rocblas_fill      uplo,
                                                           rocblas_operation transA,
                                                           rocblas_diagonal  diag,
                                                           rocblas_int       n,
                                                           ATYPE             dA,
                                                           rocblas_stride    offset_A,
                                                           int64_t           lda,
                                                           rocblas_stride    stride_A,
                                                           T const*          alpha,
                                                           XTYPE             dx,
                                                           rocblas_stride    offset_x,
                                                           int64_t           incx,
                                                           rocblas_stride    stride_x,
                                                           rocblas_int       batch_count,
                                                           rocblas_int*      w_completed_sec);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsv_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_diagonal  diag,
                                   rocblas_int       n,
                                   const T*          dA,
                                   rocblas_stride    offset_A,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_A,
                                   T*                dx,
                                   rocblas_stride    offset_x,
                                   rocblas_int       incx,
                                   rocblas_stride    stride_x,
                                   rocblas_int       batch_count,
                                   rocblas_int*      w_completed_sec);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsv_batched_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation transA,
                                           rocblas_diagonal  diag,
                                           rocblas_int       n,
                                           const T* const*   dA,
                                           rocblas_stride    offset_A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           T* const*         dx,
                                           rocblas_stride    offset_x,
                                           rocblas_int       incx,
                                           rocblas_stride    stride_x,
                                           rocblas_int       batch_count,
                                           rocblas_int*      w_completed_sec);
