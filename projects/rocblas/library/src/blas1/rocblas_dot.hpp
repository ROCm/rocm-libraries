/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "reduction.hpp"
#include "rocblas.h"
#include "rocblas_reduction.hpp"

// work item number (WIN) of elements to process
template <typename T>
constexpr int rocblas_dot_WIN()
{
    size_t nb = sizeof(T);

    int n = 8;
    if(nb >= 8)
        n = 2;
    else if(nb >= 4)
        n = 4;

    return n;
}

constexpr int rocblas_dot_WIN(size_t nb)
{
    int n = 8;
    if(nb >= 8)
        n = 2;
    else if(nb >= 4)
        n = 4;

    return n;
}

template <typename API_INT, int NB, bool CONJ, typename T, typename U, typename V = T>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_launcher(rocblas_handle __restrict__ handle,
                                  API_INT n,
                                  const U __restrict__ x,
                                  rocblas_stride offsetx,
                                  API_INT        incx,
                                  rocblas_stride stridex,
                                  const U __restrict__ y,
                                  rocblas_stride offsety,
                                  API_INT        incy,
                                  rocblas_stride stridey,
                                  API_INT        batch_count,
                                  T* __restrict__ results,
                                  V* __restrict__ workspace);

/**
 * @brief internal dot template, to be used for regular dot and dot_strided_batched.
 *        For complex versions, is equivalent to dotu. For supported types see rocBLAS documentation.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_template(rocblas_handle __restrict__ handle,
                                  rocblas_int n,
                                  const T* __restrict__ x,
                                  rocblas_stride offsetx,
                                  rocblas_int    incx,
                                  rocblas_stride stridex,
                                  const T* __restrict__ y,
                                  rocblas_stride offsety,
                                  rocblas_int    incy,
                                  rocblas_stride stridey,
                                  rocblas_int    batch_count,
                                  T* __restrict__ results,
                                  Tex* __restrict__ workspace);

/**
 * @brief internal dotc template, to be used for regular dotc and dotc_strided_batched.
 *        For complex versions, is equivalent to dotc. For supported types see rocBLAS documentation.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_template(rocblas_handle __restrict__ handle,
                                   rocblas_int n,
                                   const T* __restrict__ x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const T* __restrict__ y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   T* __restrict__ results,
                                   Tex* __restrict__ workspace);

/**
 * @brief internal dot_batched template. For complex versions, is equivalent to dotu_batched.
 *        For supported types see rocBLAS documentation.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_batched_template(rocblas_handle __restrict__ handle,
                                          rocblas_int n,
                                          const T* const* __restrict__ x,
                                          rocblas_stride offsetx,
                                          rocblas_int    incx,
                                          rocblas_stride stridex,
                                          const T* const* __restrict__ y,
                                          rocblas_stride offsety,
                                          rocblas_int    incy,
                                          rocblas_stride stridey,
                                          rocblas_int    batch_count,
                                          T* __restrict__ results,
                                          Tex* __restrict__ workspace);

/**
 * @brief internal dotc_batched template. For complex versions, is equivalent to dotc_batched.
 *        For supported types see rocBLAS documentation.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_batched_template(rocblas_handle __restrict__ handle,
                                           rocblas_int n,
                                           const T* const* __restrict__ x,
                                           rocblas_stride offsetx,
                                           rocblas_int    incx,
                                           rocblas_stride stridex,
                                           const T* const* __restrict__ y,
                                           rocblas_stride offsety,
                                           rocblas_int    incy,
                                           rocblas_stride stridey,
                                           rocblas_int    batch_count,
                                           T* __restrict__ results,
                                           Tex* __restrict__ workspace);

template <typename T>
rocblas_status rocblas_dot_check_numerics(const char*    function_name,
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
