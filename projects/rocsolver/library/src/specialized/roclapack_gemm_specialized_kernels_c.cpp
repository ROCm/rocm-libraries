/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "roclapack_gemm_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GEMM(rocblas_float_complex,
                 rocblas_int,
                 rocblas_float_complex*,
                 rocblas_float_complex*,
                 rocblas_float_complex*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 rocblas_int,
                 rocblas_float_complex* const*,
                 rocblas_float_complex* const*,
                 rocblas_float_complex* const*);

INSTANTIATE_GEMM(rocblas_float_complex,
                 rocblas_int,
                 rocblas_float_complex*,
                 rocblas_float_complex* const*,
                 rocblas_float_complex* const*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 rocblas_int,
                 rocblas_float_complex* const*,
                 rocblas_float_complex*,
                 rocblas_float_complex* const*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 rocblas_int,
                 rocblas_float_complex* const*,
                 rocblas_float_complex* const*,
                 rocblas_float_complex*);

INSTANTIATE_GEMM(rocblas_float_complex,
                 rocblas_int,
                 rocblas_float_complex* const*,
                 rocblas_float_complex*,
                 rocblas_float_complex*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 rocblas_int,
                 rocblas_float_complex*,
                 rocblas_float_complex* const*,
                 rocblas_float_complex*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 rocblas_int,
                 rocblas_float_complex*,
                 rocblas_float_complex*,
                 rocblas_float_complex* const*);

#ifdef HAVE_ROCBLAS_64
// 64-bit APIs
INSTANTIATE_GEMM(rocblas_float_complex,
                 int64_t,
                 rocblas_float_complex*,
                 rocblas_float_complex*,
                 rocblas_float_complex*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 int64_t,
                 rocblas_float_complex* const*,
                 rocblas_float_complex* const*,
                 rocblas_float_complex* const*);

INSTANTIATE_GEMM(rocblas_float_complex,
                 int64_t,
                 rocblas_float_complex*,
                 rocblas_float_complex* const*,
                 rocblas_float_complex* const*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 int64_t,
                 rocblas_float_complex* const*,
                 rocblas_float_complex*,
                 rocblas_float_complex* const*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 int64_t,
                 rocblas_float_complex* const*,
                 rocblas_float_complex* const*,
                 rocblas_float_complex*);

INSTANTIATE_GEMM(rocblas_float_complex,
                 int64_t,
                 rocblas_float_complex* const*,
                 rocblas_float_complex*,
                 rocblas_float_complex*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 int64_t,
                 rocblas_float_complex*,
                 rocblas_float_complex* const*,
                 rocblas_float_complex*);
INSTANTIATE_GEMM(rocblas_float_complex,
                 int64_t,
                 rocblas_float_complex*,
                 rocblas_float_complex*,
                 rocblas_float_complex* const*);
#endif /* HAVE_ROCBLAS_64 */

/*************************************************************
    Export methods for testing
*************************************************************/

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_cgemm(rocblas_handle handle,
                                                rocblas_operation transA,
                                                rocblas_operation transB,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int k,
                                                const rocblas_float_complex* alpha,
                                                rocblas_float_complex* A,
                                                rocblas_int inca,
                                                rocblas_int lda,
                                                rocblas_float_complex* B,
                                                rocblas_int incb,
                                                rocblas_int ldb,
                                                const rocblas_float_complex* beta,
                                                rocblas_float_complex* C,
                                                rocblas_int incc,
                                                rocblas_int ldc)
{
    rocblas_int batch_count = 1;
    rocblas_float_complex** work = nullptr;
    return rocsolver_gemm<rocblas_float_complex>(handle, transA, transB, m, n, k, alpha, A, 0, inca,
                                                 lda, 0, B, 0, incb, ldb, 0, beta, C, 0, incc, ldc,
                                                 0, batch_count, work);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgemm_batched(rocblas_handle handle,
                                                        rocblas_operation transA,
                                                        rocblas_operation transB,
                                                        rocblas_int m,
                                                        rocblas_int n,
                                                        rocblas_int k,
                                                        const rocblas_float_complex* alpha,
                                                        rocblas_float_complex* const* A,
                                                        rocblas_int inca,
                                                        rocblas_int lda,
                                                        rocblas_float_complex* const* B,
                                                        rocblas_int incb,
                                                        rocblas_int ldb,
                                                        const rocblas_float_complex* beta,
                                                        rocblas_float_complex* const* C,
                                                        rocblas_int incc,
                                                        rocblas_int ldc,
                                                        rocblas_int batch_count)
{
    rocblas_float_complex** work = nullptr;
    return rocsolver_gemm<rocblas_float_complex>(handle, transA, transB, m, n, k, alpha, A, 0, inca,
                                                 lda, 0, B, 0, incb, ldb, 0, beta, C, 0, incc, ldc,
                                                 0, batch_count, work);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgemm_strided_batched(rocblas_handle handle,
                                                                rocblas_operation transA,
                                                                rocblas_operation transB,
                                                                rocblas_int m,
                                                                rocblas_int n,
                                                                rocblas_int k,
                                                                const rocblas_float_complex* alpha,
                                                                rocblas_float_complex* A,
                                                                rocblas_int inca,
                                                                rocblas_int lda,
                                                                rocblas_stride strideA,
                                                                rocblas_float_complex* B,
                                                                rocblas_int incb,
                                                                rocblas_int ldb,
                                                                rocblas_stride strideB,
                                                                const rocblas_float_complex* beta,
                                                                rocblas_float_complex* C,
                                                                rocblas_int incc,
                                                                rocblas_int ldc,
                                                                rocblas_stride strideC,
                                                                rocblas_int batch_count)
{
    rocblas_float_complex** work = nullptr;
    return rocsolver_gemm<rocblas_float_complex>(handle, transA, transB, m, n, k, alpha, A, 0, inca,
                                                 lda, strideA, B, 0, incb, ldb, strideB, beta, C, 0,
                                                 incc, ldc, strideC, batch_count, work);
}

} // extern C

ROCSOLVER_END_NAMESPACE
