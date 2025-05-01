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

template <typename API_INT, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_dgmm_arg_check(rocblas_handle handle,
                                             rocblas_side   side,
                                             API_INT        m,
                                             API_INT        n,
                                             TConstPtr      A,
                                             API_INT        lda,
                                             TConstPtr      X,
                                             API_INT        incx,
                                             TPtr           C,
                                             API_INT        ldc,
                                             API_INT        batch_count)
{
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    if(m < 0 || n < 0 || ldc < m || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!A || !X || !C)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/**
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_dgmm_launcher(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              TConstPtr      A,
                                              rocblas_stride offset_a,
                                              int64_t        lda,
                                              rocblas_stride stride_a,
                                              TConstPtr      X,
                                              rocblas_stride offset_x,
                                              int64_t        incx,
                                              rocblas_stride stride_x,
                                              TPtr           C,
                                              rocblas_stride offset_c,
                                              int64_t        ldc,
                                              rocblas_stride stride_c,
                                              rocblas_int    batch_count);

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_dgmm_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_side   side,
                                           int64_t        m,
                                           int64_t        n,
                                           TConstPtr      A,
                                           int64_t        lda,
                                           rocblas_stride stride_A,
                                           TConstPtr      x,
                                           int64_t        incx,
                                           rocblas_stride stride_x,
                                           TPtr           C,
                                           int64_t        ldc,
                                           rocblas_stride stride_c,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
