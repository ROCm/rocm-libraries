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

#include "blas1/rocblas_dot.hpp"
#include "handle.hpp"
#include "rocblas.h"

template <typename API_INT, int NB, bool CONJ, typename T, typename U, typename V = T>
rocblas_status rocblas_internal_dot_launcher_64(rocblas_handle __restrict__ handle,
                                                int64_t n,
                                                const U __restrict__ x,
                                                rocblas_stride offsetx,
                                                int64_t        incx,
                                                rocblas_stride stridex,
                                                const U __restrict__ y,
                                                rocblas_stride offsety,
                                                int64_t        incy,
                                                rocblas_stride stridey,
                                                int64_t        batch_count,
                                                T* __restrict__ results,
                                                V* __restrict__ workspace);

/**
 * @brief internal dot template, to be used for regular dot and dot_strided_batched.
 *        For complex versions, is equivalent to dotu. For supported types see rocBLAS documentation.
 */
template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_template_64(rocblas_handle __restrict__ handle,
                                     int64_t n,
                                     const T* __restrict__ x,
                                     rocblas_stride offsetx,
                                     int64_t        incx,
                                     rocblas_stride stridex,
                                     const T* __restrict__ y,
                                     rocblas_stride offsety,
                                     int64_t        incy,
                                     rocblas_stride stridey,
                                     int64_t        batch_count,
                                     T* __restrict__ results,
                                     Tex* __restrict__ workspace);

/**
 * @brief internal dotc template, to be used for regular dotc and dotc_strided_batched.
 *        For complex versions, is equivalent to dotc. For supported types see rocBLAS documentation.
 */
template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_template_64(rocblas_handle __restrict__ handle,
                                      int64_t n,
                                      const T* __restrict__ x,
                                      rocblas_stride offsetx,
                                      int64_t        incx,
                                      rocblas_stride stridex,
                                      const T* __restrict__ y,
                                      rocblas_stride offsety,
                                      int64_t        incy,
                                      rocblas_stride stridey,
                                      int64_t        batch_count,
                                      T* __restrict__ results,
                                      Tex* __restrict__ workspace);

/**
 * @brief internal dot_batched template. For complex versions, is equivalent to dotu_batched.
 *        For supported types see rocBLAS documentation.
 */
template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_batched_template_64(rocblas_handle __restrict__ handle,
                                             int64_t n,
                                             const T* const* __restrict__ x,
                                             rocblas_stride offsetx,
                                             int64_t        incx,
                                             rocblas_stride stridex,
                                             const T* const* __restrict__ y,
                                             rocblas_stride offsety,
                                             int64_t        incy,
                                             rocblas_stride stridey,
                                             int64_t        batch_count,
                                             T* __restrict__ results,
                                             Tex* __restrict__ workspace);

/**
 * @brief internal dotc_batched template. For complex versions, is equivalent to dotc_batched.
 *        For supported types see rocBLAS documentation.
 *        Used by rocSOLVER, includes offset params for alpha/arrays.
 */
template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_batched_template_64(rocblas_handle __restrict__ handle,
                                              int64_t n,
                                              const T* const* __restrict__ x,
                                              rocblas_stride offsetx,
                                              int64_t        incx,
                                              rocblas_stride stridex,
                                              const T* const* __restrict__ y,
                                              rocblas_stride offsety,
                                              int64_t        incy,
                                              rocblas_stride stridey,
                                              int64_t        batch_count,
                                              T* __restrict__ results,
                                              Tex* __restrict__ workspace);
