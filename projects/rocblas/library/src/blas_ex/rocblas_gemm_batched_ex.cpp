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

#include "blas_ex/rocblas_gemm_batched_ex_imp.hpp"

INST_GEMM_BATCHED_EX_C_API(rocblas_int)

#ifdef BUILD_WITH_TENSILE
#include "rocblas_gemm_ex_get_solutions.hpp"
#endif

// no 64-bit interface
rocblas_status rocblas_gemm_batched_ex_get_solutions(rocblas_handle    handle,
                                                     rocblas_operation trans_a,
                                                     rocblas_operation trans_b,
                                                     rocblas_int       m,
                                                     rocblas_int       n,
                                                     rocblas_int       k,
                                                     const void*       alpha,
                                                     const void*       a,
                                                     rocblas_datatype  a_type,
                                                     rocblas_int       lda,
                                                     const void*       b,
                                                     rocblas_datatype  b_type,
                                                     rocblas_int       ldb,
                                                     const void*       beta,
                                                     const void*       c,
                                                     rocblas_datatype  c_type,
                                                     rocblas_int       ldc,
                                                     void*             d,
                                                     rocblas_datatype  d_type,
                                                     rocblas_int       ldd,
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

        auto stride_a = rocblas_stride(lda) * (trans_a == rocblas_operation_none ? k : m);
        auto stride_b = rocblas_stride(ldb) * (trans_b == rocblas_operation_none ? n : k);
        auto stride_c = rocblas_stride(ldc) * n;
        auto stride_d = rocblas_stride(ldd) * n;

        return rocblas_gemm_ex_get_solutions_template<true>(handle,
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

rocblas_status rocblas_gemm_batched_ex_get_solutions_by_type(rocblas_handle   handle,
                                                             rocblas_datatype input_type,
                                                             rocblas_datatype output_type,
                                                             rocblas_datatype compute_type,
                                                             uint32_t         flags,
                                                             rocblas_int*     list_array,
                                                             rocblas_int*     list_size)
{
#ifdef BUILD_WITH_TENSILE
    // Create dummy GEMM problem to take advantage of problem templating
    // Most parameters are ignored, just needs to be valid for all types
    rocblas_double_complex alpha{0, 0};
    rocblas_double_complex beta{0, 0};
    rocblas_stride         stride{1};
    return rocblas_gemm_ex_get_solutions_template<true>(handle,
                                                        rocblas_operation_none,
                                                        rocblas_operation_none,
                                                        4,
                                                        4,
                                                        4,
                                                        &alpha,
                                                        reinterpret_cast<const void*>(0xDEADC0DE),
                                                        input_type,
                                                        0,
                                                        4,
                                                        stride,
                                                        reinterpret_cast<const void*>(0xDEADC0DE),
                                                        input_type,
                                                        0,
                                                        4,
                                                        stride,
                                                        &beta,
                                                        reinterpret_cast<const void*>(0xDEADC0DE),
                                                        output_type,
                                                        0,
                                                        4,
                                                        stride,
                                                        reinterpret_cast<void*>(0xDEADC0DE),
                                                        output_type,
                                                        0,
                                                        4,
                                                        stride,
                                                        2,
                                                        compute_type,
                                                        flags,
                                                        MATCHES_TYPE,
                                                        list_array,
                                                        list_size);
#else
    return rocblas_status_excluded_from_build;
#endif
}
