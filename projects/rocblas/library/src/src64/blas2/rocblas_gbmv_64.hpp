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
#include "rocblas.h"

/**
  *  Here, U is either a `const T* const*` or a `const T*`
  *  V is either a `T*` or a `T* const*`
  */
template <typename T, typename U, typename V>
rocblas_status rocblas_internal_gbmv_launcher_64(rocblas_handle    handle,
                                                 rocblas_operation transA,
                                                 int64_t           m_64,
                                                 int64_t           n_64,
                                                 int64_t           kl_64,
                                                 int64_t           ku_64,
                                                 const T*          alpha,
                                                 U                 A,
                                                 rocblas_stride    offseta,
                                                 int64_t           lda_64,
                                                 rocblas_stride    strideA,
                                                 U                 x,
                                                 rocblas_stride    offsetx,
                                                 int64_t           incx_64,
                                                 rocblas_stride    stridex,
                                                 const T*          beta,
                                                 V                 y_64,
                                                 rocblas_stride    offsety,
                                                 int64_t           incy_64,
                                                 rocblas_stride    stridey,
                                                 int64_t           batch_count);
