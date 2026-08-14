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

#include "rocblas_gemm.hpp"

template <typename API_INT, typename T>
inline rocblas_status rocblas_gemm_grouped_batched_arg_check(rocblas_handle           handle,
                                                             const rocblas_operation* transa_array,
                                                             const rocblas_operation* transb_array,
                                                             const API_INT*           m_array,
                                                             const API_INT*           n_array,
                                                             const API_INT*           k_array,
                                                             const T*                 alpha_array,
                                                             const T* const*          Aarray,
                                                             const API_INT*           lda_array,
                                                             const T* const*          Barray,
                                                             const API_INT*           ldb_array,
                                                             const T*                 beta_array,
                                                             T* const*                Carray,
                                                             const API_INT*           ldc_array,
                                                             API_INT                  group_count,
                                                             const API_INT*           group_size)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    if(group_count < 0)
        return rocblas_status_invalid_size;

    if(group_count == 0)
        return rocblas_status_success;

    if(!transa_array || !transb_array || !m_array || !n_array || !k_array || !alpha_array
       || !beta_array || !lda_array || !ldb_array || !ldc_array || !group_size)
        return rocblas_status_invalid_pointer;

    int64_t problem_count = 0;
    for(API_INT g = 0; g < group_count; ++g)
    {
        if(group_size[g] < 0)
            return rocblas_status_invalid_size;

        problem_count += group_size[g];

        const T* alpha_g = alpha_array + g;
        const T* beta_g  = beta_array + g;

        auto valid = rocblas_gemm_arg_check(handle,
                                            transa_array[g],
                                            transb_array[g],
                                            m_array[g],
                                            n_array[g],
                                            k_array[g],
                                            alpha_g,
                                            Aarray,
                                            lda_array[g],
                                            Barray,
                                            ldb_array[g],
                                            beta_g,
                                            Carray,
                                            ldc_array[g],
                                            group_size[g]);

        if(valid != rocblas_status_continue)
            return valid;
    }

    if(problem_count > 0 && (!Aarray || !Barray || !Carray))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

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
                                                   const rocblas_int*       group_size);
