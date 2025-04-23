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

#include "blas2/rocblas_gemv.hpp"

#include "int64_helpers.hpp"

template <typename To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t rocblas_internal_gemv_kernel_workspace_size_64(
    rocblas_operation transA, int64_t m_64, int64_t n_64, int64_t batch_count_64)
{

    rocblas_int m           = std::min(c_i32_max, m_64);
    rocblas_int n           = std::min(c_i32_max, n_64);
    rocblas_int batch_count = std::min(c_i64_grid_YZ_chunk, batch_count_64);
    size_t work_size = rocblas_internal_gemv_kernel_workspace_size<To>(transA, m, n, batch_count);
    if(m_64 > c_ILP64_i32_max || n_64 > c_ILP64_i32_max)
        work_size
            += sizeof(rocblas_double_complex); // for temporary beta during accumulation passes
    return work_size;
}

template <typename Ti, typename Tex, typename To>
rocblas_status rocblas_internal_gemv_launcher_64(rocblas_handle    handle,
                                                 rocblas_operation transA,
                                                 int64_t           m,
                                                 int64_t           n,
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
                                                 int64_t           batch_count,
                                                 Tex*              workspace);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemv_template_64(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      int64_t           m,
                                      int64_t           n,
                                      const T*          alpha,
                                      rocblas_stride    stride_alpha,
                                      const T*          A,
                                      rocblas_stride    offseta,
                                      int64_t           lda,
                                      rocblas_stride    strideA,
                                      const T*          x,
                                      rocblas_stride    offsetx,
                                      int64_t           incx,
                                      rocblas_stride    stridex,
                                      const T*          beta,
                                      rocblas_stride    stride_beta,
                                      T*                y,
                                      rocblas_stride    offsety,
                                      int64_t           incy,
                                      rocblas_stride    stridey,
                                      int64_t           batch_count,
                                      T*                workspace = nullptr);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemv_batched_template_64(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              int64_t           m,
                                              int64_t           n,
                                              const T*          alpha,
                                              rocblas_stride    stride_alpha,
                                              const T* const*   A,
                                              rocblas_stride    offseta,
                                              int64_t           lda,
                                              rocblas_stride    strideA,
                                              const T* const*   x,
                                              rocblas_stride    offsetx,
                                              int64_t           incx,
                                              rocblas_stride    stridex,
                                              const T*          beta,
                                              rocblas_stride    stride_beta,
                                              T* const*         y,
                                              rocblas_stride    offsety,
                                              int64_t           incy,
                                              rocblas_stride    stridey,
                                              int64_t           batch_count,
                                              T*                workspace = nullptr);
