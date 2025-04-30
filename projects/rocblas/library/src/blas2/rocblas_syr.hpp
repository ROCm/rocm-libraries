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
#include "rocblas.h"

template <typename API_INT, typename T, typename U, typename V, typename W>
inline rocblas_status rocblas_syr_arg_check(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            API_INT        n,
                                            U              alpha,
                                            rocblas_stride stride_alpha,
                                            V              x,
                                            rocblas_stride offsetx,
                                            API_INT        incx,
                                            rocblas_stride stridex,
                                            W              A,
                                            rocblas_stride offseta,
                                            API_INT        lda,
                                            rocblas_stride strideA,
                                            API_INT        batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(n < 0 || !incx || lda < n || lda < 1 || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if(!alpha)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alpha == 0)
            return rocblas_status_success;

        // pointers are validated if they need to be dereferenced
        if(!A || !x)
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <typename T, typename U, typename V, typename W>
rocblas_status rocblas_internal_syr_launcher(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             U              alpha,
                                             rocblas_stride stride_alpha,
                                             V              x,
                                             rocblas_stride offsetx,
                                             int64_t        incx,
                                             rocblas_stride stridex,
                                             W              A,
                                             rocblas_stride offseta,
                                             int64_t        lda,
                                             rocblas_stride strideA,
                                             rocblas_int    batch_count);

template <typename T, typename U, typename V, typename W>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr_template(rocblas_handle handle,
                                  rocblas_fill   uplo,
                                  rocblas_int    n,
                                  U              alpha,
                                  rocblas_stride stride_alpha,
                                  V              x,
                                  rocblas_stride offset_x,
                                  int64_t        incx,
                                  rocblas_stride stride_x,
                                  W              A,
                                  rocblas_stride offset_A,
                                  int64_t        lda,
                                  rocblas_stride stride_A,
                                  rocblas_int    batch_count)
{
    return rocblas_internal_syr_launcher<T>(handle,
                                            uplo,
                                            n,
                                            alpha,
                                            stride_alpha,
                                            x,
                                            offset_x,
                                            incx,
                                            stride_x,
                                            A,
                                            offset_A,
                                            lda,
                                            stride_A,
                                            batch_count);
}

template <typename T, typename U>
rocblas_status rocblas_syr_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          rocblas_fill   uplo,
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
