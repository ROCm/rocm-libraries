/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hpreby granted, free of charge, to any person obtaining a copy
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
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHPR LIABILITY, WHETHPR
 * IN AN ACTION OF CONTRACT, TORT OR OTHPRWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHPR DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "handle.hpp"
#include "int64_helpers.hpp"

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_hpr_launcher_64(rocblas_handle handle,
                                       rocblas_fill   uplo,
                                       API_INT        n_64,
                                       TScal          alpha,
                                       TConstPtr      x,
                                       rocblas_stride offset_x,
                                       int64_t        incx_64,
                                       rocblas_stride stride_x,
                                       TPtr           AP,
                                       rocblas_stride offset_A,
                                       rocblas_stride stride_A,
                                       API_INT        batch_count_64);
