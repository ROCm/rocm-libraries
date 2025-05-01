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
#include "handle.hpp"

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_geam_arg_check(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             rocblas_operation transB,
                                             API_INT           m,
                                             API_INT           n,
                                             TScal             alpha,
                                             TConstPtr         A,
                                             API_INT           lda,
                                             TScal             beta,
                                             TConstPtr         B,
                                             API_INT           ldb,
                                             TPtr              C,
                                             API_INT           ldc,
                                             API_INT           batch_count)
{
    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
       && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(transB != rocblas_operation_none && transB != rocblas_operation_transpose
       && transB != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(batch_count < 0 || m < 0 || n < 0 || ldc < m
       || lda < (transA == rocblas_operation_none ? m : n)
       || ldb < (transB == rocblas_operation_none ? m : n))
        return rocblas_status_invalid_size;

    if((C == A && (lda != ldc || transA != rocblas_operation_none))
       || (C == B && (ldb != ldc || transB != rocblas_operation_none)))
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!C)
        return rocblas_status_invalid_pointer;

    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if((*alpha && !A) || (*beta && !B))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_geam_launcher(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     TScal             alpha,
                                     TConstPtr         A,
                                     rocblas_stride    offset_a,
                                     int64_t           lda,
                                     rocblas_stride    stride_a,
                                     TScal             beta,
                                     TConstPtr         B,
                                     rocblas_stride    offset_b,
                                     int64_t           ldb,
                                     rocblas_stride    stride_b,
                                     TPtr              C,
                                     rocblas_stride    offset_c,
                                     int64_t           ldc,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count);

template <typename T, typename U>
rocblas_status rocblas_geam_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           int64_t           m,
                                           int64_t           n,
                                           T                 A,
                                           int64_t           lda,
                                           rocblas_stride    stride_a,
                                           T                 B,
                                           int64_t           ldb,
                                           rocblas_stride    stride_b,
                                           U                 C,
                                           int64_t           ldc,
                                           rocblas_stride    stride_c,
                                           int64_t           batch_count,
                                           const int         check_numerics,
                                           bool              is_input);
