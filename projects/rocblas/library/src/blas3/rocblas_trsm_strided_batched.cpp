/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "blas3/rocblas_trsm_strided_batched_imp.hpp"

INST_TRSM_STRIDED_BATCHED_C_API(rocblas_int);

// only 32-bit trsm_batched_ex for now
rocblas_status rocblas_trsm_strided_batched_ex(rocblas_handle    handle,
                                               rocblas_side      side,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_diagonal  diag,
                                               rocblas_int       m,
                                               rocblas_int       n,
                                               const void*       alpha,
                                               const void*       A,
                                               rocblas_int       lda,
                                               rocblas_stride    strideA,
                                               void*             B,
                                               rocblas_int       ldb,
                                               rocblas_stride    strideB,
                                               rocblas_int       batch_count,
                                               const void*       invA,
                                               rocblas_int       invA_size,
                                               rocblas_stride    stride_invA,
                                               rocblas_datatype  compute_type)
try
{
#define TRSM_EX_ARGS(T_)                                                                         \
    handle, side, uplo, transA, diag, m, n, static_cast<const T_*>(alpha),                       \
        static_cast<const T_*>(A), lda, strideA, static_cast<T_*>(B), ldb, strideB, batch_count, \
        static_cast<const T_*>(invA), invA_size, stride_invA

    switch(compute_type)
    {
    case rocblas_datatype_f32_r:
        return rocblas_trsm_strided_batched_ex_impl<rocblas_int>(TRSM_EX_ARGS(float));

    case rocblas_datatype_f64_r:
        return rocblas_trsm_strided_batched_ex_impl<rocblas_int>(TRSM_EX_ARGS(double));

    case rocblas_datatype_f32_c:
        return rocblas_trsm_strided_batched_ex_impl<rocblas_int>(
            TRSM_EX_ARGS(rocblas_float_complex));

    case rocblas_datatype_f64_c:
        return rocblas_trsm_strided_batched_ex_impl<rocblas_int>(
            TRSM_EX_ARGS(rocblas_double_complex));

    default:
        return rocblas_status_not_implemented;
    }

#undef TRSM_EX_ARGS
}
catch(...)
{
    return exception_to_rocblas_status();
}
