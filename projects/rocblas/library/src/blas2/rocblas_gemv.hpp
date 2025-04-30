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

#include "gemv_device.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_level2_threshold.hpp"

template <typename To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t rocblas_internal_gemv_kernel_workspace_size(
    rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int batch_count);

template <typename API_INT, typename Ti, typename Tex, typename To>
inline rocblas_status rocblas_internal_gemv_arg_check(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      API_INT           m,
                                                      API_INT           n,
                                                      const Tex*        alpha,
                                                      rocblas_stride    stride_alpha,
                                                      const Ti*         A,
                                                      rocblas_stride    offseta,
                                                      API_INT           lda,
                                                      rocblas_stride    strideA,
                                                      const Ti*         x,
                                                      rocblas_stride    offsetx,
                                                      API_INT           incx,
                                                      rocblas_stride    stridex,
                                                      const Tex*        beta,
                                                      rocblas_stride    stride_beta,
                                                      To*               y,
                                                      rocblas_stride    offsety,
                                                      API_INT           incy,
                                                      rocblas_stride    stridey,
                                                      API_INT           batch_count)
{
    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
       && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(m < 0 || n < 0 || lda < m || lda < 1 || !incx || !incy || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alpha == 0 && *beta == 1)
            return rocblas_status_success;

        if(!y || (*alpha != 0 && (!A || !x)))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <typename Ti, typename Tex, typename To>
rocblas_status rocblas_internal_gemv_launcher(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const Tex*        alpha,
                                              rocblas_stride    stride_alpha,
                                              const Ti*         A,
                                              rocblas_stride    offseta,
                                              int64_t           lda,
                                              rocblas_stride    strideA,
                                              const Ti*         x,
                                              rocblas_stride    offsetx,
                                              int64_t           incx,
                                              rocblas_stride    stridex,
                                              const Tex*        beta,
                                              rocblas_stride    stride_beta,
                                              To*               y,
                                              rocblas_stride    offsety,
                                              int64_t           incy,
                                              rocblas_stride    stridey,
                                              rocblas_int       batch_count,
                                              Tex*              workspace = nullptr);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemv_template(rocblas_handle    handle,
                                   rocblas_operation transA,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   const T*          alpha,
                                   rocblas_stride    stride_alpha,
                                   const T*          A,
                                   rocblas_stride    offseta,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   const T*          x,
                                   rocblas_stride    offsetx,
                                   rocblas_int       incx,
                                   rocblas_stride    stridex,
                                   const T*          beta,
                                   rocblas_stride    stride_beta,
                                   T*                y,
                                   rocblas_stride    offsety,
                                   rocblas_int       incy,
                                   rocblas_stride    stridey,
                                   rocblas_int       batch_count,
                                   T*                workspace = nullptr);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemv_batched_template(rocblas_handle    handle,
                                           rocblas_operation transA,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           const T*          alpha,
                                           rocblas_stride    stride_alpha,
                                           const T* const*   A,
                                           rocblas_stride    offseta,
                                           rocblas_int       lda,
                                           rocblas_stride    strideA,
                                           const T* const*   x,
                                           rocblas_stride    offsetx,
                                           rocblas_int       incx,
                                           rocblas_stride    stridex,
                                           const T*          beta,
                                           rocblas_stride    stride_beta,
                                           T* const*         y,
                                           rocblas_stride    offsety,
                                           rocblas_int       incy,
                                           rocblas_stride    stridey,
                                           rocblas_int       batch_count,
                                           T*                workspace = nullptr);

template <typename Ti, typename To>
rocblas_status rocblas_gemv_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           int64_t           m,
                                           int64_t           n,
                                           Ti                A,
                                           rocblas_stride    offset_a,
                                           int64_t           lda,
                                           rocblas_stride    stride_a,
                                           Ti                x,
                                           rocblas_stride    offset_x,
                                           int64_t           inc_x,
                                           rocblas_stride    stride_x,
                                           To                y,
                                           rocblas_stride    offset_y,
                                           int64_t           inc_y,
                                           rocblas_stride    stride_y,
                                           int64_t           batch_count,
                                           const int         check_numerics,
                                           bool              is_input);
