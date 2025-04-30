/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "logging.hpp"

template <typename T>
bool rocblas_rotm_quick_return_param(rocblas_handle handle,
                                     const T*       param,
                                     rocblas_stride stride_param);

template <typename T>
bool rocblas_rotm_quick_return_param(rocblas_handle handle,
                                     const T* const param[],
                                     rocblas_stride stride_param);

template <typename T>
rocblas_status rocblas_rotm_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           T              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input);

template <typename API_INT, rocblas_int NB, bool BATCHED_OR_STRIDED, typename T, typename U>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE rocblas_status
    rocblas_internal_rotm_launcher(rocblas_handle handle,
                                   API_INT        n,
                                   T              x,
                                   rocblas_stride offset_x,
                                   int64_t        incx,
                                   rocblas_stride stride_x,
                                   T              y,
                                   rocblas_stride offset_y,
                                   int64_t        incy,
                                   rocblas_stride stride_y,
                                   U              param,
                                   rocblas_stride offset_param,
                                   rocblas_stride stride_param,
                                   API_INT        batch_count);
