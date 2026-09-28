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

#include "blas3/rocblas_gemm.hpp"
#include "int64_helpers.hpp"
#include "src64/blas3/rocblas_gemm_64.hpp"

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemm_grouped_batched_template(rocblas_handle           handle,
                                                   const rocblas_operation* transa_array,
                                                   const rocblas_operation* transb_array,
                                                   const rocblas_int*       m_array,
                                                   const rocblas_int*       n_array,
                                                   const rocblas_int*       k_array,
                                                   const T*                 alpha_array,
                                                   const T* const*          Aarray,
                                                   const rocblas_int*       lda_array,
                                                   const T* const*          Barray,
                                                   const rocblas_int*       ldb_array,
                                                   const T*                 beta_array,
                                                   T* const*                Carray,
                                                   const rocblas_int*       ldc_array,
                                                   rocblas_int              group_count,
                                                   const rocblas_int*       group_size)
{
    int64_t idx = 0;
    for(rocblas_int g = 0; g < group_count; ++g)
    {
        rocblas_status status = (rocblas_internal_gemm_batched_template)(handle,
                                                                         transa_array[g],
                                                                         transb_array[g],
                                                                         m_array[g],
                                                                         n_array[g],
                                                                         k_array[g],
                                                                         alpha_array + g,
                                                                         Aarray + idx,
                                                                         0,
                                                                         lda_array[g],
                                                                         0,
                                                                         Barray + idx,
                                                                         0,
                                                                         ldb_array[g],
                                                                         0,
                                                                         beta_array + g,
                                                                         Carray + idx,
                                                                         0,
                                                                         ldc_array[g],
                                                                         0,
                                                                         group_size[g]);
        if(status != rocblas_status_success)
            return status;
        idx += group_size[g];
    }
    return rocblas_status_success;
}
