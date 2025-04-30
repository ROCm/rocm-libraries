/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"

template <typename API_INT, typename U, typename X>
inline rocblas_status rocblas_tbmv_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             API_INT           n,
                                             API_INT           k,
                                             U                 A,
                                             API_INT           lda,
                                             X                 x,
                                             API_INT           incx,
                                             API_INT           batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
       && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || lda < k + 1 || !incx || batch_count < 0)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!n || !batch_count)
    {
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
        return rocblas_status_success;
    }

    return rocblas_status_continue;
}

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_tbmv_launcher(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              TConstPtr         A,
                                              rocblas_stride    offseta,
                                              int64_t           lda,
                                              rocblas_stride    strideA,
                                              TPtr              x,
                                              rocblas_stride    offsetx,
                                              int64_t           incx,
                                              rocblas_stride    stridex,
                                              rocblas_int       batch_count,
                                              TPtr              w_x_copy);

template <typename T, typename U>
rocblas_status rocblas_tbmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
