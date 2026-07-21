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

#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas.h"

template <typename T>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE rocblas_status
    rocblas_internal_check_numerics_matrix_template(const char*               function_name,
                                                    rocblas_handle            handle,
                                                    rocblas_operation         trans_a,
                                                    rocblas_fill              uplo,
                                                    rocblas_check_matrix_type matrix_type,
                                                    int64_t                   m,
                                                    int64_t                   n,
                                                    T                         A,
                                                    rocblas_stride            offset_a,
                                                    int64_t                   lda,
                                                    rocblas_stride            stride_a,
                                                    int64_t                   batch_count,
                                                    const int                 check_numerics,
                                                    bool                      is_input);
