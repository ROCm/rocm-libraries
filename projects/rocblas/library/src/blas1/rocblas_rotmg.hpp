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

template <typename T, typename U>
rocblas_status rocblas_rotmg_check_numerics_template(const char*    function_name,
                                                     rocblas_handle handle,
                                                     T              d1_in,
                                                     rocblas_stride offset_d1,
                                                     rocblas_stride stride_d1,
                                                     T              d2_in,
                                                     rocblas_stride offset_d2,
                                                     rocblas_stride stride_d2,
                                                     T              x1_in,
                                                     rocblas_stride offset_x1,
                                                     rocblas_stride stride_x1,
                                                     U              y1_in,
                                                     rocblas_stride offset_y1,
                                                     rocblas_stride stride_y1,
                                                     int64_t        batch_count,
                                                     const int      check_numerics,
                                                     bool           is_input);

template <typename API_INT, typename T, typename U>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE
    rocblas_status ROCBLAS_API(rocblas_internal_rotmg_launcher)(rocblas_handle handle,
                                                                T              d1_in,
                                                                rocblas_stride offset_d1,
                                                                rocblas_stride stride_d1,
                                                                T              d2_in,
                                                                rocblas_stride offset_d2,
                                                                rocblas_stride stride_d2,
                                                                T              x1_in,
                                                                rocblas_stride offset_x1,
                                                                rocblas_stride stride_x1,
                                                                U              y1_in,
                                                                rocblas_stride offset_y1,
                                                                rocblas_stride stride_y1,
                                                                T              param,
                                                                rocblas_stride offset_param,
                                                                rocblas_stride stride_param,
                                                                API_INT        batch_count);
