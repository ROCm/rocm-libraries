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

#include "rocblas_gemm_ex.hpp"

inline size_t rocblas_gemm_ex_compute_type_size(rocblas_datatype compute_type)
{
    switch(compute_type)
    {
    case rocblas_datatype_f16_r:
        return sizeof(rocblas_half);
    case rocblas_datatype_f32_r:
        return sizeof(float);
    case rocblas_datatype_f64_r:
        return sizeof(double);
    case rocblas_datatype_i32_r:
        return sizeof(int32_t);
    case rocblas_datatype_f32_c:
        return sizeof(rocblas_float_complex);
    case rocblas_datatype_f64_c:
        return sizeof(rocblas_double_complex);
    default:
        return 0;
    }
}

template <typename API_INT>
inline rocblas_status
    rocblas_gemm_grouped_batched_ex_arg_check(rocblas_handle           handle,
                                              const rocblas_operation* transa_array,
                                              const rocblas_operation* transb_array,
                                              const API_INT*           m_array,
                                              const API_INT*           n_array,
                                              const API_INT*           k_array,
                                              const void*              alpha_array,
                                              const void* const*       Aarray,
                                              rocblas_datatype         a_type,
                                              const API_INT*           lda_array,
                                              const void* const*       Barray,
                                              rocblas_datatype         b_type,
                                              const API_INT*           ldb_array,
                                              const void*              beta_array,
                                              const void* const*       Carray,
                                              rocblas_datatype         c_type,
                                              const API_INT*           ldc_array,
                                              void* const*             Darray,
                                              rocblas_datatype         d_type,
                                              const API_INT*           ldd_array,
                                              API_INT                  group_count,
                                              const API_INT*           group_size,
                                              rocblas_datatype         compute_type)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    if(group_count < 0)
        return rocblas_status_invalid_size;

    if(group_count == 0)
        return rocblas_status_success;

    if(!transa_array || !transb_array || !m_array || !n_array || !k_array || !alpha_array
       || !beta_array || !lda_array || !ldb_array || !ldc_array || !ldd_array || !group_size)
        return rocblas_status_invalid_pointer;

    const size_t scalar_stride = rocblas_gemm_ex_compute_type_size(compute_type);
    if(scalar_stride == 0)
        return rocblas_status_invalid_value;

    int64_t problem_count = 0;
    for(API_INT g = 0; g < group_count; ++g)
    {
        if(group_size[g] < 0)
            return rocblas_status_invalid_size;

        problem_count += group_size[g];

        const void* alpha_g
            = static_cast<const char*>(alpha_array) + static_cast<size_t>(g) * scalar_stride;
        const void* beta_g
            = static_cast<const char*>(beta_array) + static_cast<size_t>(g) * scalar_stride;

        auto valid = rocblas_gemm_ex_arg_check(handle,
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
                                               c_type,
                                               ldc_array[g],
                                               Darray,
                                               d_type,
                                               ldd_array[g],
                                               compute_type,
                                               group_size[g]);

        if(valid != rocblas_status_continue)
            return valid;
    }

    if(problem_count > 0 && (!Aarray || !Barray || !Carray || !Darray))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename API_INT>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemm_grouped_batched_ex_template(rocblas_handle           handle,
                                                      const rocblas_operation* transa_array,
                                                      const rocblas_operation* transb_array,
                                                      const API_INT*           m_array,
                                                      const API_INT*           n_array,
                                                      const API_INT*           k_array,
                                                      const void*              alpha_array,
                                                      const void* const*       Aarray,
                                                      rocblas_datatype         a_type,
                                                      const API_INT*           lda_array,
                                                      const void* const*       Barray,
                                                      rocblas_datatype         b_type,
                                                      const API_INT*           ldb_array,
                                                      const void*              beta_array,
                                                      const void* const*       Carray,
                                                      rocblas_datatype         c_type,
                                                      const API_INT*           ldc_array,
                                                      void* const*             Darray,
                                                      rocblas_datatype         d_type,
                                                      const API_INT*           ldd_array,
                                                      API_INT                  group_count,
                                                      const API_INT*           group_size,
                                                      rocblas_datatype         compute_type,
                                                      rocblas_gemm_algo        algo,
                                                      int32_t                  solution_index,
                                                      uint32_t                 flags);
