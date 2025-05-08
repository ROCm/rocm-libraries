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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"

template <typename API_INT, rocblas_int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE
    rocblas_status ROCBLAS_API(rocblas_internal_axpy_launcher)(rocblas_handle handle,
                                                               API_INT        n,
                                                               Ta             alpha,
                                                               rocblas_stride stride_alpha,
                                                               Tx             x,
                                                               rocblas_stride offset_x,
                                                               API_INT        incx,
                                                               rocblas_stride stride_x,
                                                               Ty             y,
                                                               rocblas_stride offset_y,
                                                               API_INT        incy,
                                                               rocblas_stride stride_y,
                                                               API_INT        batch_count);

/**
 * @brief internal axpy template, to be used for regular axpy and axpy_strided_batched.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_axpy_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   rocblas_stride stride_alpha,
                                   const T*       x,
                                   rocblas_stride offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   T*             y,
                                   rocblas_stride offset_y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   rocblas_int    batch_count);

/**
 * @brief internal axpy_batched template.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_axpy_batched_template(rocblas_handle  handle,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           rocblas_stride  stride_alpha,
                                           const T* const* x,
                                           rocblas_stride  offset_x,
                                           rocblas_int     incx,
                                           rocblas_stride  stride_x,
                                           T* const*       y,
                                           rocblas_stride  offset_y,
                                           rocblas_int     incy,
                                           rocblas_stride  stride_y,
                                           rocblas_int     batch_count);

template <typename API_INT, typename Ta, typename Tx, typename Ty>
inline rocblas_status rocblas_axpy_arg_check(rocblas_handle handle,
                                             API_INT        n,
                                             const Ta*      alpha,
                                             Tx             x,
                                             rocblas_stride offset_x,
                                             API_INT        incx,
                                             rocblas_stride stride_x,
                                             Ty             y,
                                             rocblas_stride offset_y,
                                             API_INT        incy,
                                             rocblas_stride stride_y,
                                             API_INT        batch_count)
{
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    if(!alpha)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alpha == 0)
            return rocblas_status_success;

        // pointers are validated if they need to be dereferenced
        if(!x || !y)
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocblas_axpy_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
