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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas.h"

template <typename API_INT, int NB, typename T, typename Tex, typename Ta, typename Tx>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE
    rocblas_status ROCBLAS_API(rocblas_internal_scal_launcher)(rocblas_handle handle,
                                                               API_INT        n,
                                                               const Ta*      alpha,
                                                               rocblas_stride stride_alpha,
                                                               Tx             x,
                                                               rocblas_stride offset_x,
                                                               API_INT        incx,
                                                               rocblas_stride stride_x,
                                                               API_INT        batch_count);

/**
 * @brief internal scal template, to be used for regular scal and scal_strided_batched.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T, typename Ta>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ta*      alpha,
                                   rocblas_stride stride_alpha,
                                   T*             x,
                                   rocblas_stride offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   rocblas_int    batch_count);

/**
 * @brief internal scal_batched template.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T, typename Ta>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_batched_template(rocblas_handle handle,
                                           rocblas_int    n,
                                           const Ta*      alpha,
                                           rocblas_stride stride_alpha,
                                           T* const*      x,
                                           rocblas_stride offset_x,
                                           rocblas_int    incx,
                                           rocblas_stride stride_x,
                                           rocblas_int    batch_count);
