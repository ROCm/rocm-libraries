/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <rocblas/rocblas.h>

#include "clientcommon.hpp"

// Templated wrappers around rocBLAS functions, following the same conventions
// as rocsolver.hpp: overloading on scalar type to provide a uniform interface.

/******************** AXPY ********************/
inline rocblas_status rocblas_axpy(rocblas_handle handle,
                                   rocblas_int n,
                                   const float* alpha,
                                   const float* x,
                                   rocblas_int incx,
                                   float* y,
                                   rocblas_int incy)
{
    return rocblas_saxpy(handle, n, alpha, x, incx, y, incy);
}

inline rocblas_status rocblas_axpy(rocblas_handle handle,
                                   rocblas_int n,
                                   const double* alpha,
                                   const double* x,
                                   rocblas_int incx,
                                   double* y,
                                   rocblas_int incy)
{
    return rocblas_daxpy(handle, n, alpha, x, incx, y, incy);
}

inline rocblas_status rocblas_axpy(rocblas_handle handle,
                                   rocblas_int n,
                                   const rocblas_float_complex* alpha,
                                   const rocblas_float_complex* x,
                                   rocblas_int incx,
                                   rocblas_float_complex* y,
                                   rocblas_int incy)
{
    return rocblas_caxpy(handle, n, alpha, x, incx, y, incy);
}

inline rocblas_status rocblas_axpy(rocblas_handle handle,
                                   rocblas_int n,
                                   const rocblas_double_complex* alpha,
                                   const rocblas_double_complex* x,
                                   rocblas_int incx,
                                   rocblas_double_complex* y,
                                   rocblas_int incy)
{
    return rocblas_zaxpy(handle, n, alpha, x, incx, y, incy);
}

/******************** GEMM ********************/
inline rocblas_status rocblas_gemm(rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   const float* alpha,
                                   const float* A,
                                   rocblas_int lda,
                                   const float* B,
                                   rocblas_int ldb,
                                   const float* beta,
                                   float* C,
                                   rocblas_int ldc)
{
    return rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline rocblas_status rocblas_gemm(rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   const double* alpha,
                                   const double* A,
                                   rocblas_int lda,
                                   const double* B,
                                   rocblas_int ldb,
                                   const double* beta,
                                   double* C,
                                   rocblas_int ldc)
{
    return rocblas_dgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline rocblas_status rocblas_gemm(rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   const rocblas_float_complex* alpha,
                                   const rocblas_float_complex* A,
                                   rocblas_int lda,
                                   const rocblas_float_complex* B,
                                   rocblas_int ldb,
                                   const rocblas_float_complex* beta,
                                   rocblas_float_complex* C,
                                   rocblas_int ldc)
{
    return rocblas_cgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline rocblas_status rocblas_gemm(rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   const rocblas_double_complex* alpha,
                                   const rocblas_double_complex* A,
                                   rocblas_int lda,
                                   const rocblas_double_complex* B,
                                   rocblas_int ldb,
                                   const rocblas_double_complex* beta,
                                   rocblas_double_complex* C,
                                   rocblas_int ldc)
{
    return rocblas_zgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
