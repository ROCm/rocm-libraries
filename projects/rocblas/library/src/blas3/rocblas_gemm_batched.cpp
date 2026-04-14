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

#include "blas3/rocblas_gemm_batched_imp.hpp"

INST_GEMM_BATCHED_C_API(rocblas_int);

/*******************************************************************************
 * Batched GEMM Kernel name APIs
 ******************************************************************************/
extern "C" {

rocblas_status rocblas_hgemm_batched_kernel_name(rocblas_handle      handle,
                                                 rocblas_operation   trans_a,
                                                 rocblas_operation   trans_b,
                                                 rocblas_int         m,
                                                 rocblas_int         n,
                                                 rocblas_int         k,
                                                 const rocblas_half* alpha,
                                                 const rocblas_half* A[],
                                                 rocblas_int         lda,
                                                 const rocblas_half* B[],
                                                 rocblas_int         ldb,
                                                 const rocblas_half* beta,
                                                 rocblas_half*       C[],
                                                 rocblas_int         ldc,
                                                 rocblas_int         batch_count)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocblas_sgemm_batched_kernel_name(rocblas_handle    handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int       m,
                                                 rocblas_int       n,
                                                 rocblas_int       k,
                                                 const float*      alpha,
                                                 const float*      A[],
                                                 rocblas_int       lda,
                                                 const float*      B[],
                                                 rocblas_int       ldb,
                                                 const float*      beta,
                                                 float*            C[],
                                                 rocblas_int       ldc,
                                                 rocblas_int       batch_count)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocblas_dgemm_batched_kernel_name(rocblas_handle    handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 rocblas_int       m,
                                                 rocblas_int       n,
                                                 rocblas_int       k,
                                                 const double*     alpha,
                                                 const double*     A[],
                                                 rocblas_int       lda,
                                                 const double*     B[],
                                                 rocblas_int       ldb,
                                                 const double*     beta,
                                                 double*           C[],
                                                 rocblas_int       ldc,
                                                 rocblas_int       batch_count)
{
    return rocblas_status_not_implemented;
}

} // extern "C"
