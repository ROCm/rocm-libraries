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

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_block_sizes.h"

template <typename API_INT, rocblas_int NB, typename T, typename U>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE
    rocblas_status ROCBLAS_API(rocblas_internal_copy_launcher)(rocblas_handle handle,
                                                               API_INT        n,
                                                               T              x,
                                                               rocblas_stride offsetx,
                                                               API_INT        incx,
                                                               rocblas_stride stridex,
                                                               U              y,
                                                               rocblas_stride offsety,
                                                               API_INT        incy,
                                                               rocblas_stride stridey,
                                                               API_INT        batch_count);

template <typename API_INT, typename T, typename U>
inline rocblas_status rocblas_copy_arg_check(rocblas_handle handle,
                                             API_INT        n,
                                             T              x,
                                             rocblas_stride offsetx,
                                             API_INT        incx,
                                             rocblas_stride stridex,
                                             U              y,
                                             rocblas_stride offsety,
                                             API_INT        incy,
                                             rocblas_stride stridey,
                                             API_INT        batch_count)
{
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    if(!x || !y)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename API_INT, typename T, typename U>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE rocblas_status
    rocblas_internal_copy_template(rocblas_handle handle,
                                   API_INT        n,
                                   T              x,
                                   rocblas_stride offsetx,
                                   API_INT        incx,
                                   rocblas_stride stridex,
                                   U              y,
                                   rocblas_stride offsety,
                                   API_INT        incy,
                                   rocblas_stride stridey,
                                   API_INT        batch_count)
{
    return ROCBLAS_API(rocblas_internal_copy_launcher)<API_INT, ROCBLAS_COPY_NB, T, U>(
        handle, n, x, offsetx, incx, stridex, y, offsety, incy, stridey, batch_count);
}

template <typename T, typename U>
rocblas_status rocblas_copy_check_numerics(const char*    function_name,
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
