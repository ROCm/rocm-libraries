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

#include "../blas1/rocblas_axpy.hpp"

template <typename API_INT, int NB, bool BATCHED>
rocblas_status rocblas_axpy_ex_template(rocblas_handle   handle,
                                        API_INT          n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        rocblas_stride   stride_alpha,
                                        const void*      x,
                                        rocblas_datatype x_type,
                                        rocblas_stride   offset_x,
                                        API_INT          incx,
                                        rocblas_stride   stride_x,
                                        void*            y,
                                        rocblas_datatype y_type,
                                        rocblas_stride   offset_y,
                                        API_INT          incy,
                                        rocblas_stride   stride_y,
                                        API_INT          batch_count,
                                        rocblas_datatype execution_type);
