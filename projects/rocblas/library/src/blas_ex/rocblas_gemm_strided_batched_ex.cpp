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

#include "blas_ex/rocblas_gemm_strided_batched_ex_imp.hpp"

INST_GEMM_STRIDED_BATCHED_EX_C_API(rocblas_int)

#ifdef BUILD_WITH_TENSILE
#include "rocblas_gemm_ex_get_solutions.hpp"
#endif

// no 64-bit interface
rocblas_status rocblas_gemm_strided_batched_ex_get_solutions(rocblas_handle    handle,
                                                             rocblas_operation trans_a,
                                                             rocblas_operation trans_b,
                                                             rocblas_int       m,
                                                             rocblas_int       n,
                                                             rocblas_int       k,
                                                             const void*       alpha,
                                                             const void*       a,
                                                             rocblas_datatype  a_type,
                                                             rocblas_int       lda,
                                                             rocblas_stride    stride_a,
                                                             const void*       b,
                                                             rocblas_datatype  b_type,
                                                             rocblas_int       ldb,
                                                             rocblas_stride    stride_b,
                                                             const void*       beta,
                                                             const void*       c,
                                                             rocblas_datatype  c_type,
                                                             rocblas_int       ldc,
                                                             rocblas_stride    stride_c,
                                                             void*             d,
                                                             rocblas_datatype  d_type,
                                                             rocblas_int       ldd,
                                                             rocblas_stride    stride_d,
                                                             rocblas_int       batch_count,
                                                             rocblas_datatype  compute_type,
                                                             rocblas_gemm_algo algo,
                                                             uint32_t          flags,
                                                             rocblas_int*      list_array,
                                                             rocblas_int*      list_size)
{
    try
    {
#ifdef BUILD_WITH_TENSILE
        if(!handle)
            return rocblas_status_invalid_handle;

        const bool HPA = compute_type == rocblas_datatype_f32_r
                         && (a_type == rocblas_datatype_f16_r || a_type == rocblas_datatype_bf16_r);

        if(!HPA)
            RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto validArgs = rocblas_gemm_ex_arg_check(handle,
                                                   trans_a,
                                                   trans_b,
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   a,
                                                   lda,
                                                   b,
                                                   ldb,
                                                   beta,
                                                   c,
                                                   c_type,
                                                   ldc,
                                                   d,
                                                   d_type,
                                                   ldd,
                                                   compute_type,
                                                   batch_count,
                                                   true); // get_solutions

        if(validArgs != rocblas_status_continue)
        {
            if(validArgs == rocblas_status_success)
                RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
            return validArgs;
        }

        return rocblas_gemm_ex_get_solutions_template<false>(handle,
                                                             trans_a,
                                                             trans_b,
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             a,
                                                             a_type,
                                                             0,
                                                             lda,
                                                             stride_a,
                                                             b,
                                                             b_type,
                                                             0,
                                                             ldb,
                                                             stride_b,
                                                             beta,
                                                             c,
                                                             c_type,
                                                             0,
                                                             ldc,
                                                             stride_c,
                                                             d,
                                                             d_type,
                                                             0,
                                                             ldd,
                                                             stride_d,
                                                             batch_count,
                                                             compute_type,
                                                             flags,
                                                             CAN_SOLVE,
                                                             list_array,
                                                             list_size);
#else
        return rocblas_status_excluded_from_build;
#endif
    }
    catch(...)
    {
        return exception_to_rocblas_status();
    }
}
