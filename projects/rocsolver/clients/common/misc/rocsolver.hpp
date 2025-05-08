/* **************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocsolver/rocsolver.h>

#include "clientcommon.hpp"

// Most functions within this file exist to provide a consistent interface for our templated tests.
// Function overloading is used to select between the float, double, rocblas_float_complex
// and rocblas_double_complex variants, and to distinguish the batched case (T**) from the normal
// and strided_batched cases (T*).
//
// The normal and strided_batched cases are distinguished from each other by passing a boolean
// parameter, STRIDED. Variants such as the blocked and unblocked versions of algorithms, may be
// provided in similar ways.

/***** Functions not included in the public API that must be declared *****/
#ifdef __cplusplus
extern "C" {
#endif

rocblas_status rocsolver_sstedcx(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_erange range,
                                 const rocblas_int n,
                                 const float vl,
                                 const float vu,
                                 const rocblas_int il,
                                 const rocblas_int iu,
                                 float* D,
                                 float* E,
                                 rocblas_int* nev,
                                 float* W,
                                 float* C,
                                 const rocblas_int ldc,
                                 rocblas_int* info);

rocblas_status rocsolver_dstedcx(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_erange range,
                                 const rocblas_int n,
                                 const double vl,
                                 const double vu,
                                 const rocblas_int il,
                                 const rocblas_int iu,
                                 double* D,
                                 double* E,
                                 rocblas_int* nev,
                                 double* W,
                                 double* C,
                                 const rocblas_int ldc,
                                 rocblas_int* info);

rocblas_status rocsolver_cstedcx(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_erange range,
                                 const rocblas_int n,
                                 const float vl,
                                 const float vu,
                                 const rocblas_int il,
                                 const rocblas_int iu,
                                 float* D,
                                 float* E,
                                 rocblas_int* nev,
                                 float* W,
                                 rocblas_float_complex* C,
                                 const rocblas_int ldc,
                                 rocblas_int* info);

rocblas_status rocsolver_zstedcx(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_erange range,
                                 const rocblas_int n,
                                 const double vl,
                                 const double vu,
                                 const rocblas_int il,
                                 const rocblas_int iu,
                                 double* D,
                                 double* E,
                                 rocblas_int* nev,
                                 double* W,
                                 rocblas_double_complex* C,
                                 const rocblas_int ldc,
                                 rocblas_int* info);

rocblas_status rocsolver_sstedcj(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_int n,
                                 float* D,
                                 float* E,
                                 float* C,
                                 const rocblas_int ldc,
                                 rocblas_int* info);

rocblas_status rocsolver_dstedcj(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_int n,
                                 double* D,
                                 double* E,
                                 double* C,
                                 const rocblas_int ldc,
                                 rocblas_int* info);

rocblas_status rocsolver_cstedcj(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_int n,
                                 float* D,
                                 float* E,
                                 rocblas_float_complex* C,
                                 const rocblas_int ldc,
                                 rocblas_int* info);

rocblas_status rocsolver_zstedcj(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_int n,
                                 double* D,
                                 double* E,
                                 rocblas_double_complex* C,
                                 const rocblas_int ldc,
                                 rocblas_int* info);

rocblas_status rocsolver_sgeqrf_ptr_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            float* const A[],
                                            const rocblas_int lda,
                                            float* const ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_dgeqrf_ptr_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            double* const A[],
                                            const rocblas_int lda,
                                            double* const ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_cgeqrf_ptr_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            const rocblas_int lda,
                                            rocblas_float_complex* const ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_zgeqrf_ptr_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            const rocblas_int lda,
                                            rocblas_double_complex* const ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_sgesv_outofplace(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          float* B,
                                          const rocblas_int ldb,
                                          float* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_dgesv_outofplace(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          double* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          double* B,
                                          const rocblas_int ldb,
                                          double* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_cgesv_outofplace(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          rocblas_float_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_float_complex* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_zgesv_outofplace(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_double_complex* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          rocblas_double_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_double_complex* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_sgels_outofplace(rocblas_handle handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float* A,
                                          const rocblas_int lda,
                                          float* B,
                                          const rocblas_int ldb,
                                          float* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_dgels_outofplace(rocblas_handle handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          double* A,
                                          const rocblas_int lda,
                                          double* B,
                                          const rocblas_int ldb,
                                          double* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_cgels_outofplace(rocblas_handle handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int lda,
                                          rocblas_float_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_float_complex* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_zgels_outofplace(rocblas_handle handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_double_complex* A,
                                          const rocblas_int lda,
                                          rocblas_double_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_double_complex* X,
                                          const rocblas_int ldx,
                                          rocblas_int* info);

rocblas_status rocsolver_ssyevdx_inplace(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         float* A,
                                         const rocblas_int lda,
                                         const float vl,
                                         const float vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const float abstol,
                                         rocblas_int* nev,
                                         float* W,
                                         rocblas_int* info);

rocblas_status rocsolver_dsyevdx_inplace(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         double* A,
                                         const rocblas_int lda,
                                         const double vl,
                                         const double vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const double abstol,
                                         rocblas_int* nev,
                                         double* W,
                                         rocblas_int* info);

rocblas_status rocsolver_cheevdx_inplace(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         rocblas_float_complex* A,
                                         const rocblas_int lda,
                                         const float vl,
                                         const float vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const float abstol,
                                         rocblas_int* nev,
                                         float* W,
                                         rocblas_int* info);

rocblas_status rocsolver_zheevdx_inplace(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         rocblas_double_complex* A,
                                         const rocblas_int lda,
                                         const double vl,
                                         const double vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const double abstol,
                                         rocblas_int* nev,
                                         double* W,
                                         rocblas_int* info);

rocblas_status rocsolver_ssygvdx_inplace(rocblas_handle handle,
                                         const rocblas_eform itype,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         float* A,
                                         const rocblas_int lda,
                                         float* B,
                                         const rocblas_int ldb,
                                         const float vl,
                                         const float vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const float abstol,
                                         rocblas_int* h_nev,
                                         float* W,
                                         rocblas_int* info);

rocblas_status rocsolver_dsygvdx_inplace(rocblas_handle handle,
                                         const rocblas_eform itype,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         double* A,
                                         const rocblas_int lda,
                                         double* B,
                                         const rocblas_int ldb,
                                         const double vl,
                                         const double vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const double abstol,
                                         rocblas_int* h_nev,
                                         double* W,
                                         rocblas_int* info);

rocblas_status rocsolver_chegvdx_inplace(rocblas_handle handle,
                                         const rocblas_eform itype,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         rocblas_float_complex* A,
                                         const rocblas_int lda,
                                         rocblas_float_complex* B,
                                         const rocblas_int ldb,
                                         const float vl,
                                         const float vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const float abstol,
                                         rocblas_int* h_nev,
                                         float* W,
                                         rocblas_int* info);

rocblas_status rocsolver_zhegvdx_inplace(rocblas_handle handle,
                                         const rocblas_eform itype,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         rocblas_double_complex* A,
                                         const rocblas_int lda,
                                         rocblas_double_complex* B,
                                         const rocblas_int ldb,
                                         const double vl,
                                         const double vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         const double abstol,
                                         rocblas_int* h_nev,
                                         double* W,
                                         rocblas_int* info);

rocblas_status rocsolver_sgesvdj_notransv(rocblas_handle handle,
                                          const rocblas_svect left_svect,
                                          const rocblas_svect right_svect,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          float* A,
                                          const rocblas_int lda,
                                          const float abstol,
                                          float* residual,
                                          const rocblas_int max_sweeps,
                                          rocblas_int* n_sweeps,
                                          float* S,
                                          float* U,
                                          const rocblas_int ldu,
                                          float* V,
                                          const rocblas_int ldv,
                                          rocblas_int* info);

rocblas_status rocsolver_dgesvdj_notransv(rocblas_handle handle,
                                          const rocblas_svect left_svect,
                                          const rocblas_svect right_svect,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          double* A,
                                          const rocblas_int lda,
                                          const double abstol,
                                          double* residual,
                                          const rocblas_int max_sweeps,
                                          rocblas_int* n_sweeps,
                                          double* S,
                                          double* U,
                                          const rocblas_int ldu,
                                          double* V,
                                          const rocblas_int ldv,
                                          rocblas_int* info);

rocblas_status rocsolver_cgesvdj_notransv(rocblas_handle handle,
                                          const rocblas_svect left_svect,
                                          const rocblas_svect right_svect,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          rocblas_float_complex* A,
                                          const rocblas_int lda,
                                          const float abstol,
                                          float* residual,
                                          const rocblas_int max_sweeps,
                                          rocblas_int* n_sweeps,
                                          float* S,
                                          rocblas_float_complex* U,
                                          const rocblas_int ldu,
                                          rocblas_float_complex* V,
                                          const rocblas_int ldv,
                                          rocblas_int* info);

rocblas_status rocsolver_zgesvdj_notransv(rocblas_handle handle,
                                          const rocblas_svect left_svect,
                                          const rocblas_svect right_svect,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          rocblas_double_complex* A,
                                          const rocblas_int lda,
                                          const double abstol,
                                          double* residual,
                                          const rocblas_int max_sweeps,
                                          rocblas_int* n_sweeps,
                                          double* S,
                                          rocblas_double_complex* U,
                                          const rocblas_int ldu,
                                          rocblas_double_complex* V,
                                          const rocblas_int ldv,
                                          rocblas_int* info);

rocblas_status rocsolver_sgesvdj_notransv_strided_batched(rocblas_handle handle,
                                                          const rocblas_svect left_svect,
                                                          const rocblas_svect right_svect,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          float* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const float abstol,
                                                          float* residual,
                                                          const rocblas_int max_sweeps,
                                                          rocblas_int* n_sweeps,
                                                          float* S,
                                                          const rocblas_stride strideS,
                                                          float* U,
                                                          const rocblas_int ldu,
                                                          const rocblas_stride strideU,
                                                          float* V,
                                                          const rocblas_int ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count);

rocblas_status rocsolver_dgesvdj_notransv_strided_batched(rocblas_handle handle,
                                                          const rocblas_svect left_svect,
                                                          const rocblas_svect right_svect,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          double* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const double abstol,
                                                          double* residual,
                                                          const rocblas_int max_sweeps,
                                                          rocblas_int* n_sweeps,
                                                          double* S,
                                                          const rocblas_stride strideS,
                                                          double* U,
                                                          const rocblas_int ldu,
                                                          const rocblas_stride strideU,
                                                          double* V,
                                                          const rocblas_int ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count);

rocblas_status rocsolver_cgesvdj_notransv_strided_batched(rocblas_handle handle,
                                                          const rocblas_svect left_svect,
                                                          const rocblas_svect right_svect,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          rocblas_float_complex* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const float abstol,
                                                          float* residual,
                                                          const rocblas_int max_sweeps,
                                                          rocblas_int* n_sweeps,
                                                          float* S,
                                                          const rocblas_stride strideS,
                                                          rocblas_float_complex* U,
                                                          const rocblas_int ldu,
                                                          const rocblas_stride strideU,
                                                          rocblas_float_complex* V,
                                                          const rocblas_int ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count);

rocblas_status rocsolver_zgesvdj_notransv_strided_batched(rocblas_handle handle,
                                                          const rocblas_svect left_svect,
                                                          const rocblas_svect right_svect,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          rocblas_double_complex* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const double abstol,
                                                          double* residual,
                                                          const rocblas_int max_sweeps,
                                                          rocblas_int* n_sweeps,
                                                          double* S,
                                                          const rocblas_stride strideS,
                                                          rocblas_double_complex* U,
                                                          const rocblas_int ldu,
                                                          const rocblas_stride strideU,
                                                          rocblas_double_complex* V,
                                                          const rocblas_int ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count);

rocblas_status rocsolver_sgesvdx_notransv_strided_batched(rocblas_handle handle,
                                                          const rocblas_svect left_svect,
                                                          const rocblas_svect right_svect,
                                                          const rocblas_srange srange,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          float* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const float vl,
                                                          const float vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          rocblas_int* nsv,
                                                          float* S,
                                                          const rocblas_stride strideS,
                                                          float* U,
                                                          const rocblas_int ldu,
                                                          const rocblas_stride strideU,
                                                          float* V,
                                                          const rocblas_int ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int* ifail,
                                                          const rocblas_stride strideF,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count);

rocblas_status rocsolver_dgesvdx_notransv_strided_batched(rocblas_handle handle,
                                                          const rocblas_svect left_svect,
                                                          const rocblas_svect right_svect,
                                                          const rocblas_srange srange,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          double* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const double vl,
                                                          const double vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          rocblas_int* nsv,
                                                          double* S,
                                                          const rocblas_stride strideS,
                                                          double* U,
                                                          const rocblas_int ldu,
                                                          const rocblas_stride strideU,
                                                          double* V,
                                                          const rocblas_int ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int* ifail,
                                                          const rocblas_stride strideF,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count);

rocblas_status rocsolver_cgesvdx_notransv_strided_batched(rocblas_handle handle,
                                                          const rocblas_svect left_svect,
                                                          const rocblas_svect right_svect,
                                                          const rocblas_srange srange,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          rocblas_float_complex* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const float vl,
                                                          const float vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          rocblas_int* nsv,
                                                          float* S,
                                                          const rocblas_stride strideS,
                                                          rocblas_float_complex* U,
                                                          const rocblas_int ldu,
                                                          const rocblas_stride strideU,
                                                          rocblas_float_complex* V,
                                                          const rocblas_int ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int* ifail,
                                                          const rocblas_stride strideF,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count);

rocblas_status rocsolver_zgesvdx_notransv_strided_batched(rocblas_handle handle,
                                                          const rocblas_svect left_svect,
                                                          const rocblas_svect right_svect,
                                                          const rocblas_srange srange,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          rocblas_double_complex* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const double vl,
                                                          const double vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          rocblas_int* nsv,
                                                          double* S,
                                                          const rocblas_stride strideS,
                                                          rocblas_double_complex* U,
                                                          const rocblas_int ldu,
                                                          const rocblas_stride strideU,
                                                          rocblas_double_complex* V,
                                                          const rocblas_int ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int* ifail,
                                                          const rocblas_stride strideF,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count);

#ifdef __cplusplus
}
#endif
/***************************************************/

/******************** GEMM ********************/
// normal and strided_batched
inline rocblas_status rocblas_gemm(bool STRIDED,
                                   rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   float* alpha,
                                   float* A,
                                   rocblas_int lda,
                                   rocblas_stride stA,
                                   float* B,
                                   rocblas_int ldb,
                                   rocblas_stride stB,
                                   float* beta,
                                   float* C,
                                   rocblas_int ldc,
                                   rocblas_stride stC,
                                   rocblas_int batch_count)
{
    if(!STRIDED)
        return rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        return rocblas_sgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda, stA, B,
                                             ldb, stB, beta, C, ldc, stC, batch_count);
}

inline rocblas_status rocblas_gemm(bool STRIDED,
                                   rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   double* alpha,
                                   double* A,
                                   rocblas_int lda,
                                   rocblas_stride stA,
                                   double* B,
                                   rocblas_int ldb,
                                   rocblas_stride stB,
                                   double* beta,
                                   double* C,
                                   rocblas_int ldc,
                                   rocblas_stride stC,
                                   rocblas_int batch_count)
{
    if(!STRIDED)
        return rocblas_dgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        return rocblas_dgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda, stA, B,
                                             ldb, stB, beta, C, ldc, stC, batch_count);
}

inline rocblas_status rocblas_gemm(bool STRIDED,
                                   rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   rocblas_float_complex* alpha,
                                   rocblas_float_complex* A,
                                   rocblas_int lda,
                                   rocblas_stride stA,
                                   rocblas_float_complex* B,
                                   rocblas_int ldb,
                                   rocblas_stride stB,
                                   rocblas_float_complex* beta,
                                   rocblas_float_complex* C,
                                   rocblas_int ldc,
                                   rocblas_stride stC,
                                   rocblas_int batch_count)
{
    if(!STRIDED)
        return rocblas_cgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        return rocblas_cgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda, stA, B,
                                             ldb, stB, beta, C, ldc, stC, batch_count);
}

inline rocblas_status rocblas_gemm(bool STRIDED,
                                   rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   rocblas_double_complex* alpha,
                                   rocblas_double_complex* A,
                                   rocblas_int lda,
                                   rocblas_stride stA,
                                   rocblas_double_complex* B,
                                   rocblas_int ldb,
                                   rocblas_stride stB,
                                   rocblas_double_complex* beta,
                                   rocblas_double_complex* C,
                                   rocblas_int ldc,
                                   rocblas_stride stC,
                                   rocblas_int batch_count)
{
    if(!STRIDED)
        return rocblas_zgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        return rocblas_zgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda, stA, B,
                                             ldb, stB, beta, C, ldc, stC, batch_count);
}

// batched
inline rocblas_status rocblas_gemm(bool STRIDED,
                                   rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   float* alpha,
                                   float* const A[],
                                   rocblas_int lda,
                                   rocblas_stride stA,
                                   float* const B[],
                                   rocblas_int ldb,
                                   rocblas_stride stB,
                                   float* beta,
                                   float* const C[],
                                   rocblas_int ldc,
                                   rocblas_stride stC,
                                   rocblas_int batch_count)
{
    return rocblas_sgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                 ldc, batch_count);
}

inline rocblas_status rocblas_gemm(bool STRIDED,
                                   rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   double* alpha,
                                   double* const A[],
                                   rocblas_int lda,
                                   rocblas_stride stA,
                                   double* const B[],
                                   rocblas_int ldb,
                                   rocblas_stride stB,
                                   double* beta,
                                   double* const C[],
                                   rocblas_int ldc,
                                   rocblas_stride stC,
                                   rocblas_int batch_count)
{
    return rocblas_dgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                 ldc, batch_count);
}

inline rocblas_status rocblas_gemm(bool STRIDED,
                                   rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   rocblas_float_complex* alpha,
                                   rocblas_float_complex* const A[],
                                   rocblas_int lda,
                                   rocblas_stride stA,
                                   rocblas_float_complex* const B[],
                                   rocblas_int ldb,
                                   rocblas_stride stB,
                                   rocblas_float_complex* beta,
                                   rocblas_float_complex* const C[],
                                   rocblas_int ldc,
                                   rocblas_stride stC,
                                   rocblas_int batch_count)
{
    return rocblas_cgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                 ldc, batch_count);
}

inline rocblas_status rocblas_gemm(bool STRIDED,
                                   rocblas_handle handle,
                                   rocblas_operation transA,
                                   rocblas_operation transB,
                                   rocblas_int m,
                                   rocblas_int n,
                                   rocblas_int k,
                                   rocblas_double_complex* alpha,
                                   rocblas_double_complex* const A[],
                                   rocblas_int lda,
                                   rocblas_stride stA,
                                   rocblas_double_complex* const B[],
                                   rocblas_int ldb,
                                   rocblas_stride stB,
                                   rocblas_double_complex* beta,
                                   rocblas_double_complex* const C[],
                                   rocblas_int ldc,
                                   rocblas_stride stC,
                                   rocblas_int batch_count)
{
    return rocblas_zgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                 ldc, batch_count);
}
/*****************************************************/

/******************** LACGV ********************/
inline rocblas_status
    rocsolver_lacgv(rocblas_handle handle, rocblas_int n, rocblas_float_complex* x, rocblas_int incx)
{
    return rocsolver_clacgv(handle, n, x, incx);
}

inline rocblas_status
    rocsolver_lacgv(rocblas_handle handle, rocblas_int n, rocblas_double_complex* x, rocblas_int incx)
{
    return rocsolver_zlacgv(handle, n, x, incx);
}

inline rocblas_status
    rocsolver_lacgv(rocblas_handle handle, int64_t n, rocblas_float_complex* x, int64_t incx)
{
    return rocsolver_clacgv_64(handle, n, x, incx);
}

inline rocblas_status
    rocsolver_lacgv(rocblas_handle handle, int64_t n, rocblas_double_complex* x, int64_t incx)
{
    return rocsolver_zlacgv_64(handle, n, x, incx);
}
/*****************************************************/

/******************** LASWP ********************/
inline rocblas_status rocsolver_laswp(rocblas_handle handle,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    return rocsolver_slaswp(handle, n, A, lda, k1, k2, ipiv, inc);
}

inline rocblas_status rocsolver_laswp(rocblas_handle handle,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    return rocsolver_dlaswp(handle, n, A, lda, k1, k2, ipiv, inc);
}

inline rocblas_status rocsolver_laswp(rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    return rocsolver_claswp(handle, n, A, lda, k1, k2, ipiv, inc);
}

inline rocblas_status rocsolver_laswp(rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    return rocsolver_zlaswp(handle, n, A, lda, k1, k2, ipiv, inc);
}
/*****************************************************/

/******************** LARFG ********************/
inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      rocblas_int n,
                                      float* alpha,
                                      float* x,
                                      rocblas_int incx,
                                      float* tau)
{
    return rocsolver_slarfg(handle, n, alpha, x, incx, tau);
}

inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      rocblas_int n,
                                      double* alpha,
                                      double* x,
                                      rocblas_int incx,
                                      double* tau)
{
    return rocsolver_dlarfg(handle, n, alpha, x, incx, tau);
}

inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* alpha,
                                      rocblas_float_complex* x,
                                      rocblas_int incx,
                                      rocblas_float_complex* tau)
{
    return rocsolver_clarfg(handle, n, alpha, x, incx, tau);
}

inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* alpha,
                                      rocblas_double_complex* x,
                                      rocblas_int incx,
                                      rocblas_double_complex* tau)
{
    return rocsolver_zlarfg(handle, n, alpha, x, incx, tau);
}

inline rocblas_status
    rocsolver_larfg(rocblas_handle handle, int64_t n, float* alpha, float* x, int64_t incx, float* tau)
{
    return rocsolver_slarfg_64(handle, n, alpha, x, incx, tau);
}

inline rocblas_status
    rocsolver_larfg(rocblas_handle handle, int64_t n, double* alpha, double* x, int64_t incx, double* tau)
{
    return rocsolver_dlarfg_64(handle, n, alpha, x, incx, tau);
}

inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      int64_t n,
                                      rocblas_float_complex* alpha,
                                      rocblas_float_complex* x,
                                      int64_t incx,
                                      rocblas_float_complex* tau)
{
    return rocsolver_clarfg_64(handle, n, alpha, x, incx, tau);
}

inline rocblas_status rocsolver_larfg(rocblas_handle handle,
                                      int64_t n,
                                      rocblas_double_complex* alpha,
                                      rocblas_double_complex* x,
                                      int64_t incx,
                                      rocblas_double_complex* tau)
{
    return rocsolver_zlarfg_64(handle, n, alpha, x, incx, tau);
}
/*****************************************************/

/******************** LARF ********************/
inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_int m,
                                     rocblas_int n,
                                     float* x,
                                     rocblas_int incx,
                                     float* alpha,
                                     float* A,
                                     rocblas_int lda)
{
    return rocsolver_slarf(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_int m,
                                     rocblas_int n,
                                     double* x,
                                     rocblas_int incx,
                                     double* alpha,
                                     double* A,
                                     rocblas_int lda)
{
    return rocsolver_dlarf(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_float_complex* x,
                                     rocblas_int incx,
                                     rocblas_float_complex* alpha,
                                     rocblas_float_complex* A,
                                     rocblas_int lda)
{
    return rocsolver_clarf(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_double_complex* x,
                                     rocblas_int incx,
                                     rocblas_double_complex* alpha,
                                     rocblas_double_complex* A,
                                     rocblas_int lda)
{
    return rocsolver_zlarf(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     int64_t m,
                                     int64_t n,
                                     float* x,
                                     int64_t incx,
                                     float* alpha,
                                     float* A,
                                     int64_t lda)
{
    return rocsolver_slarf_64(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     int64_t m,
                                     int64_t n,
                                     double* x,
                                     int64_t incx,
                                     double* alpha,
                                     double* A,
                                     int64_t lda)
{
    return rocsolver_dlarf_64(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     int64_t m,
                                     int64_t n,
                                     rocblas_float_complex* x,
                                     int64_t incx,
                                     rocblas_float_complex* alpha,
                                     rocblas_float_complex* A,
                                     int64_t lda)
{
    return rocsolver_clarf_64(handle, side, m, n, x, incx, alpha, A, lda);
}

inline rocblas_status rocsolver_larf(rocblas_handle handle,
                                     rocblas_side side,
                                     int64_t m,
                                     int64_t n,
                                     rocblas_double_complex* x,
                                     int64_t incx,
                                     rocblas_double_complex* alpha,
                                     rocblas_double_complex* A,
                                     int64_t lda)
{
    return rocsolver_zlarf_64(handle, side, m, n, x, incx, alpha, A, lda);
}
/*****************************************************/

/******************** LARFT ********************/
inline rocblas_status rocsolver_larft(rocblas_handle handle,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int n,
                                      rocblas_int k,
                                      float* V,
                                      rocblas_int ldv,
                                      float* tau,
                                      float* F,
                                      rocblas_int ldt)
{
    return rocsolver_slarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

inline rocblas_status rocsolver_larft(rocblas_handle handle,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int n,
                                      rocblas_int k,
                                      double* V,
                                      rocblas_int ldv,
                                      double* tau,
                                      double* F,
                                      rocblas_int ldt)
{
    return rocsolver_dlarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

inline rocblas_status rocsolver_larft(rocblas_handle handle,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_float_complex* tau,
                                      rocblas_float_complex* F,
                                      rocblas_int ldt)
{
    return rocsolver_clarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}

inline rocblas_status rocsolver_larft(rocblas_handle handle,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_double_complex* tau,
                                      rocblas_double_complex* F,
                                      rocblas_int ldt)
{
    return rocsolver_zlarft(handle, direct, storev, n, k, V, ldv, tau, F, ldt);
}
/*****************************************************/

/******************** LARFB ********************/
inline rocblas_status rocsolver_larfb(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_operation trans,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      float* V,
                                      rocblas_int ldv,
                                      float* F,
                                      rocblas_int ldt,
                                      float* A,
                                      rocblas_int lda)
{
    return rocsolver_slarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

inline rocblas_status rocsolver_larfb(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_operation trans,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      double* V,
                                      rocblas_int ldv,
                                      double* F,
                                      rocblas_int ldt,
                                      double* A,
                                      rocblas_int lda)
{
    return rocsolver_dlarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

inline rocblas_status rocsolver_larfb(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_operation trans,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_float_complex* F,
                                      rocblas_int ldt,
                                      rocblas_float_complex* A,
                                      rocblas_int lda)
{
    return rocsolver_clarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}

inline rocblas_status rocsolver_larfb(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_operation trans,
                                      rocblas_direct direct,
                                      rocblas_storev storev,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_double_complex* F,
                                      rocblas_int ldt,
                                      rocblas_double_complex* A,
                                      rocblas_int lda)
{
    return rocsolver_zlarfb(handle, side, trans, direct, storev, m, n, k, V, ldv, F, ldt, A, lda);
}
/***************************************************************/

/******************** LASR *************************************/
inline rocblas_status rocsolver_lasr(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_pivot pivot,
                                     rocblas_direct direct,
                                     rocblas_int m,
                                     rocblas_int n,
                                     float* C,
                                     float* S,
                                     float* A,
                                     rocblas_int lda)
{
    return rocsolver_slasr(handle, side, pivot, direct, m, n, C, S, A, lda);
}

inline rocblas_status rocsolver_lasr(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_pivot pivot,
                                     rocblas_direct direct,
                                     rocblas_int m,
                                     rocblas_int n,
                                     double* C,
                                     double* S,
                                     double* A,
                                     rocblas_int lda)
{
    return rocsolver_dlasr(handle, side, pivot, direct, m, n, C, S, A, lda);
}

inline rocblas_status rocsolver_lasr(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_pivot pivot,
                                     rocblas_direct direct,
                                     rocblas_int m,
                                     rocblas_int n,
                                     float* C,
                                     float* S,
                                     rocblas_float_complex* A,
                                     rocblas_int lda)
{
    return rocsolver_clasr(handle, side, pivot, direct, m, n, C, S, A, lda);
}

inline rocblas_status rocsolver_lasr(rocblas_handle handle,
                                     rocblas_side side,
                                     rocblas_pivot pivot,
                                     rocblas_direct direct,
                                     rocblas_int m,
                                     rocblas_int n,
                                     double* C,
                                     double* S,
                                     rocblas_double_complex* A,
                                     rocblas_int lda)
{
    return rocsolver_zlasr(handle, side, pivot, direct, m, n, C, S, A, lda);
}
/***************************************************************/

/******************** LAUUM ********************/

inline rocblas_status rocsolver_lauum(rocblas_handle handle,
                                      const rocblas_fill uplo,
                                      const rocblas_int n,
                                      float* A,
                                      const rocblas_int lda)
{
    return rocsolver_slauum(handle, uplo, n, A, lda);
}

inline rocblas_status rocsolver_lauum(rocblas_handle handle,
                                      const rocblas_fill uplo,
                                      const rocblas_int n,
                                      double* A,
                                      const rocblas_int lda)
{
    return rocsolver_dlauum(handle, uplo, n, A, lda);
}

inline rocblas_status rocsolver_lauum(rocblas_handle handle,
                                      const rocblas_fill uplo,
                                      const rocblas_int n,
                                      rocblas_float_complex* A,
                                      const rocblas_int lda)
{
    return rocsolver_clauum(handle, uplo, n, A, lda);
}

inline rocblas_status rocsolver_lauum(rocblas_handle handle,
                                      const rocblas_fill uplo,
                                      const rocblas_int n,
                                      rocblas_double_complex* A,
                                      const rocblas_int lda)
{
    return rocsolver_zlauum(handle, uplo, n, A, lda);
}
/***************************************************************/

/******************** BDSQR ********************/
inline rocblas_status rocsolver_bdsqr(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nv,
                                      rocblas_int nu,
                                      rocblas_int nc,
                                      float* D,
                                      float* E,
                                      float* V,
                                      rocblas_int ldv,
                                      float* U,
                                      rocblas_int ldu,
                                      float* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_sbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, info);
}

inline rocblas_status rocsolver_bdsqr(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nv,
                                      rocblas_int nu,
                                      rocblas_int nc,
                                      double* D,
                                      double* E,
                                      double* V,
                                      rocblas_int ldv,
                                      double* U,
                                      rocblas_int ldu,
                                      double* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_dbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, info);
}

inline rocblas_status rocsolver_bdsqr(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nv,
                                      rocblas_int nu,
                                      rocblas_int nc,
                                      float* D,
                                      float* E,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_float_complex* U,
                                      rocblas_int ldu,
                                      rocblas_float_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_cbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, info);
}

inline rocblas_status rocsolver_bdsqr(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nv,
                                      rocblas_int nu,
                                      rocblas_int nc,
                                      double* D,
                                      double* E,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_double_complex* U,
                                      rocblas_int ldu,
                                      rocblas_double_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_zbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, info);
}
/***************************************************************/

/******************** LATRD ********************/
inline rocblas_status rocsolver_latrd(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int k,
                                      float* A,
                                      rocblas_int lda,
                                      float* E,
                                      float* tau,
                                      float* W,
                                      rocblas_int ldw)
{
    return rocsolver_slatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

inline rocblas_status rocsolver_latrd(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int k,
                                      double* A,
                                      rocblas_int lda,
                                      double* E,
                                      double* tau,
                                      double* W,
                                      rocblas_int ldw)
{
    return rocsolver_dlatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

inline rocblas_status rocsolver_latrd(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      float* E,
                                      rocblas_float_complex* tau,
                                      rocblas_float_complex* W,
                                      rocblas_int ldw)
{
    return rocsolver_clatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}

inline rocblas_status rocsolver_latrd(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      double* E,
                                      rocblas_double_complex* tau,
                                      rocblas_double_complex* W,
                                      rocblas_int ldw)
{
    return rocsolver_zlatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw);
}
/***************************************************************/

/******************** LABRD ********************/
inline rocblas_status rocsolver_labrd(rocblas_handle handle,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      float* A,
                                      rocblas_int lda,
                                      float* D,
                                      float* E,
                                      float* tauq,
                                      float* taup,
                                      float* X,
                                      rocblas_int ldx,
                                      float* Y,
                                      rocblas_int ldy)
{
    return rocsolver_slabrd(handle, m, n, nb, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

inline rocblas_status rocsolver_labrd(rocblas_handle handle,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      double* A,
                                      rocblas_int lda,
                                      double* D,
                                      double* E,
                                      double* tauq,
                                      double* taup,
                                      double* X,
                                      rocblas_int ldx,
                                      double* Y,
                                      rocblas_int ldy)
{
    return rocsolver_dlabrd(handle, m, n, nb, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

inline rocblas_status rocsolver_labrd(rocblas_handle handle,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      float* D,
                                      float* E,
                                      rocblas_float_complex* tauq,
                                      rocblas_float_complex* taup,
                                      rocblas_float_complex* X,
                                      rocblas_int ldx,
                                      rocblas_float_complex* Y,
                                      rocblas_int ldy)
{
    return rocsolver_clabrd(handle, m, n, nb, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

inline rocblas_status rocsolver_labrd(rocblas_handle handle,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      double* D,
                                      double* E,
                                      rocblas_double_complex* tauq,
                                      rocblas_double_complex* taup,
                                      rocblas_double_complex* X,
                                      rocblas_int ldx,
                                      rocblas_double_complex* Y,
                                      rocblas_int ldy)
{
    return rocsolver_zlabrd(handle, m, n, nb, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}
/***************************************************************/

/******************** ORGxR_UNGxR ********************/
inline rocblas_status rocsolver_orgxr_ungxr(bool GQR,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return GQR ? rocsolver_sorgqr(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_sorg2r(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxr_ungxr(bool GQR,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return GQR ? rocsolver_dorgqr(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_dorg2r(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxr_ungxr(bool GQR,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return GQR ? rocsolver_cungqr(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_cung2r(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxr_ungxr(bool GQR,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return GQR ? rocsolver_zungqr(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_zung2r(handle, m, n, k, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORGLx_UNGLx ********************/
inline rocblas_status rocsolver_orglx_unglx(bool GLQ,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return GLQ ? rocsolver_sorglq(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_sorgl2(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orglx_unglx(bool GLQ,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return GLQ ? rocsolver_dorglq(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_dorgl2(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orglx_unglx(bool GLQ,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return GLQ ? rocsolver_cunglq(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_cungl2(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orglx_unglx(bool GLQ,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return GLQ ? rocsolver_zunglq(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_zungl2(handle, m, n, k, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORGxL_UNGxL ********************/
inline rocblas_status rocsolver_orgxl_ungxl(bool GQL,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return GQL ? rocsolver_sorgql(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_sorg2l(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxl_ungxl(bool GQL,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return GQL ? rocsolver_dorgql(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_dorg2l(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxl_ungxl(bool GQL,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return GQL ? rocsolver_cungql(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_cung2l(handle, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgxl_ungxl(bool GQL,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return GQL ? rocsolver_zungql(handle, m, n, k, A, lda, Ipiv)
               : rocsolver_zung2l(handle, m, n, k, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORGBR_UNGBR ********************/
inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return rocsolver_sorgbr(handle, storev, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return rocsolver_dorgbr(handle, storev, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return rocsolver_cungbr(handle, storev, m, n, k, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgbr_ungbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return rocsolver_zungbr(handle, storev, m, n, k, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORGTR_UNGTR ********************/
inline rocblas_status rocsolver_orgtr_ungtr(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv)
{
    return rocsolver_sorgtr(handle, uplo, n, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgtr_ungtr(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv)
{
    return rocsolver_dorgtr(handle, uplo, n, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgtr_ungtr(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv)
{
    return rocsolver_cungtr(handle, uplo, n, A, lda, Ipiv);
}

inline rocblas_status rocsolver_orgtr_ungtr(rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv)
{
    return rocsolver_zungtr(handle, uplo, n, A, lda, Ipiv);
}
/***************************************************************/

/******************** ORMxR_UNMxR ********************/
inline rocblas_status rocsolver_ormxr_unmxr(bool MQR,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return MQR ? rocsolver_sormqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_sorm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxr_unmxr(bool MQR,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return MQR ? rocsolver_dormqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_dorm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxr_unmxr(bool MQR,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return MQR ? rocsolver_cunmqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_cunm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxr_unmxr(bool MQR,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return MQR ? rocsolver_zunmqr(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_zunm2r(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** ORMLx_UNMLx ********************/
inline rocblas_status rocsolver_ormlx_unmlx(bool MLQ,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return MLQ ? rocsolver_sormlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_sorml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormlx_unmlx(bool MLQ,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return MLQ ? rocsolver_dormlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_dorml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormlx_unmlx(bool MLQ,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return MLQ ? rocsolver_cunmlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_cunml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormlx_unmlx(bool MLQ,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return MLQ ? rocsolver_zunmlq(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_zunml2(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** ORMxL_UNMxL ********************/
inline rocblas_status rocsolver_ormxl_unmxl(bool MQL,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return MQL ? rocsolver_sormql(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_sorm2l(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxl_unmxl(bool MQL,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return MQL ? rocsolver_dormql(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_dorm2l(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxl_unmxl(bool MQL,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return MQL ? rocsolver_cunmql(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_cunm2l(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormxl_unmxl(bool MQL,
                                            rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return MQL ? rocsolver_zunmql(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc)
               : rocsolver_zunm2l(handle, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** ORMBR_UNMBR ********************/
inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return rocsolver_sormbr(handle, storev, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return rocsolver_dormbr(handle, storev, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return rocsolver_cunmbr(handle, storev, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormbr_unmbr(rocblas_handle handle,
                                            rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return rocsolver_zunmbr(handle, storev, side, trans, m, n, k, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** ORMTR_UNMTR ********************/
inline rocblas_status rocsolver_ormtr_unmtr(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            float* Ipiv,
                                            float* C,
                                            rocblas_int ldc)
{
    return rocsolver_sormtr(handle, side, uplo, trans, m, n, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormtr_unmtr(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            double* Ipiv,
                                            double* C,
                                            rocblas_int ldc)
{
    return rocsolver_dormtr(handle, side, uplo, trans, m, n, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormtr_unmtr(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc)
{
    return rocsolver_cunmtr(handle, side, uplo, trans, m, n, A, lda, Ipiv, C, ldc);
}

inline rocblas_status rocsolver_ormtr_unmtr(rocblas_handle handle,
                                            rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_double_complex* Ipiv,
                                            rocblas_double_complex* C,
                                            rocblas_int ldc)
{
    return rocsolver_zunmtr(handle, side, uplo, trans, m, n, A, lda, Ipiv, C, ldc);
}
/***************************************************************/

/******************** STERF ********************/
inline rocblas_status
    rocsolver_sterf(rocblas_handle handle, rocblas_int n, float* D, float* E, rocblas_int* info)
{
    return rocsolver_ssterf(handle, n, D, E, info);
}

inline rocblas_status
    rocsolver_sterf(rocblas_handle handle, rocblas_int n, double* D, double* E, rocblas_int* info)
{
    return rocsolver_dsterf(handle, n, D, E, info);
}
/********************************************************/

/******************** STEBZ ********************/
inline rocblas_status rocsolver_stebz(rocblas_handle handle,
                                      rocblas_erange erange,
                                      rocblas_eorder eorder,
                                      rocblas_int n,
                                      float vl,
                                      float vu,
                                      rocblas_int il,
                                      rocblas_int iu,
                                      float abstol,
                                      float* D,
                                      float* E,
                                      rocblas_int* nev,
                                      rocblas_int* nsplit,
                                      float* W,
                                      rocblas_int* iblock,
                                      rocblas_int* isplit,
                                      rocblas_int* info)
{
    return rocsolver_sstebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, E, nev, nsplit, W,
                            iblock, isplit, info);
}

inline rocblas_status rocsolver_stebz(rocblas_handle handle,
                                      rocblas_erange erange,
                                      rocblas_eorder eorder,
                                      rocblas_int n,
                                      double vl,
                                      double vu,
                                      rocblas_int il,
                                      rocblas_int iu,
                                      double abstol,
                                      double* D,
                                      double* E,
                                      rocblas_int* nev,
                                      rocblas_int* nsplit,
                                      double* W,
                                      rocblas_int* iblock,
                                      rocblas_int* isplit,
                                      rocblas_int* info)
{
    return rocsolver_dstebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, E, nev, nsplit, W,
                            iblock, isplit, info);
}
/********************************************************/

/******************** STEQR ********************/
inline rocblas_status rocsolver_steqr(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      float* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_ssteqr(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_steqr(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      double* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_dsteqr(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_steqr(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      rocblas_float_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_csteqr(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_steqr(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      rocblas_double_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_zsteqr(handle, evect, n, D, E, C, ldc, info);
}
/********************************************************/

/******************** STEDC ********************/
inline rocblas_status rocsolver_stedc(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      float* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_sstedc(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedc(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      double* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_dstedc(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedc(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      rocblas_float_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_cstedc(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedc(rocblas_handle handle,
                                      rocblas_evect evect,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      rocblas_double_complex* C,
                                      rocblas_int ldc,
                                      rocblas_int* info)
{
    return rocsolver_zstedc(handle, evect, n, D, E, C, ldc, info);
}
/********************************************************/

/******************** STEDCJ ********************/
inline rocblas_status rocsolver_stedcj(rocblas_handle handle,
                                       rocblas_evect evect,
                                       rocblas_int n,
                                       float* D,
                                       float* E,
                                       float* C,
                                       rocblas_int ldc,
                                       rocblas_int* info)
{
    return rocsolver_sstedcj(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedcj(rocblas_handle handle,
                                       rocblas_evect evect,
                                       rocblas_int n,
                                       double* D,
                                       double* E,
                                       double* C,
                                       rocblas_int ldc,
                                       rocblas_int* info)
{
    return rocsolver_dstedcj(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedcj(rocblas_handle handle,
                                       rocblas_evect evect,
                                       rocblas_int n,
                                       float* D,
                                       float* E,
                                       rocblas_float_complex* C,
                                       rocblas_int ldc,
                                       rocblas_int* info)
{
    return rocsolver_cstedcj(handle, evect, n, D, E, C, ldc, info);
}

inline rocblas_status rocsolver_stedcj(rocblas_handle handle,
                                       rocblas_evect evect,
                                       rocblas_int n,
                                       double* D,
                                       double* E,
                                       rocblas_double_complex* C,
                                       rocblas_int ldc,
                                       rocblas_int* info)
{
    return rocsolver_zstedcj(handle, evect, n, D, E, C, ldc, info);
}
/********************************************************/

/******************** STEDCX ********************/
inline rocblas_status rocsolver_stedcx(rocblas_handle handle,
                                       rocblas_evect evect,
                                       rocblas_erange range,
                                       rocblas_int n,
                                       float vl,
                                       float vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       float* D,
                                       float* E,
                                       rocblas_int* nev,
                                       float* W,
                                       float* C,
                                       rocblas_int ldc,
                                       rocblas_int* info)
{
    return rocsolver_sstedcx(handle, evect, range, n, vl, vu, il, iu, D, E, nev, W, C, ldc, info);
}

inline rocblas_status rocsolver_stedcx(rocblas_handle handle,
                                       rocblas_evect evect,
                                       rocblas_erange range,
                                       rocblas_int n,
                                       double vl,
                                       double vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       double* D,
                                       double* E,
                                       rocblas_int* nev,
                                       double* W,
                                       double* C,
                                       rocblas_int ldc,
                                       rocblas_int* info)
{
    return rocsolver_dstedcx(handle, evect, range, n, vl, vu, il, iu, D, E, nev, W, C, ldc, info);
}

inline rocblas_status rocsolver_stedcx(rocblas_handle handle,
                                       rocblas_evect evect,
                                       rocblas_erange range,
                                       rocblas_int n,
                                       float vl,
                                       float vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       float* D,
                                       float* E,
                                       rocblas_int* nev,
                                       float* W,
                                       rocblas_float_complex* C,
                                       rocblas_int ldc,
                                       rocblas_int* info)
{
    return rocsolver_cstedcx(handle, evect, range, n, vl, vu, il, iu, D, E, nev, W, C, ldc, info);
}

inline rocblas_status rocsolver_stedcx(rocblas_handle handle,
                                       rocblas_evect evect,
                                       rocblas_erange range,
                                       rocblas_int n,
                                       double vl,
                                       double vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       double* D,
                                       double* E,
                                       rocblas_int* nev,
                                       double* W,
                                       rocblas_double_complex* C,
                                       rocblas_int ldc,
                                       rocblas_int* info)
{
    return rocsolver_zstedcx(handle, evect, range, n, vl, vu, il, iu, D, E, nev, W, C, ldc, info);
}
/********************************************************/

/******************** STEIN ********************/
inline rocblas_status rocsolver_stein(rocblas_handle handle,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      rocblas_int* nev,
                                      float* W,
                                      rocblas_int* iblock,
                                      rocblas_int* isplit,
                                      float* Z,
                                      rocblas_int ldz,
                                      rocblas_int* ifail,
                                      rocblas_int* info)
{
    return rocsolver_sstein(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_stein(rocblas_handle handle,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      rocblas_int* nev,
                                      double* W,
                                      rocblas_int* iblock,
                                      rocblas_int* isplit,
                                      double* Z,
                                      rocblas_int ldz,
                                      rocblas_int* ifail,
                                      rocblas_int* info)
{
    return rocsolver_dstein(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_stein(rocblas_handle handle,
                                      rocblas_int n,
                                      float* D,
                                      float* E,
                                      rocblas_int* nev,
                                      float* W,
                                      rocblas_int* iblock,
                                      rocblas_int* isplit,
                                      rocblas_float_complex* Z,
                                      rocblas_int ldz,
                                      rocblas_int* ifail,
                                      rocblas_int* info)
{
    return rocsolver_cstein(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_stein(rocblas_handle handle,
                                      rocblas_int n,
                                      double* D,
                                      double* E,
                                      rocblas_int* nev,
                                      double* W,
                                      rocblas_int* iblock,
                                      rocblas_int* isplit,
                                      rocblas_double_complex* Z,
                                      rocblas_int ldz,
                                      rocblas_int* ifail,
                                      rocblas_int* info)
{
    return rocsolver_zstein(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, info);
}
/********************************************************/

/******************** LASYF ********************/
inline rocblas_status rocsolver_lasyf(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    return rocsolver_slasyf(handle, uplo, n, nb, kb, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_lasyf(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    return rocsolver_dlasyf(handle, uplo, n, nb, kb, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_lasyf(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    return rocsolver_clasyf(handle, uplo, n, nb, kb, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_lasyf(rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    return rocsolver_zlasyf(handle, uplo, n, nb, kb, A, lda, ipiv, info);
}
/********************************************************/

/******************** BDSVDX ********************/
inline rocblas_status rocsolver_bdsvdx(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_svect svect,
                                       rocblas_srange srange,
                                       rocblas_int n,
                                       float* D,
                                       float* E,
                                       float vl,
                                       float vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* nsv,
                                       float* S,
                                       float* Z,
                                       const rocblas_int ldz,
                                       rocblas_int* ifail,
                                       rocblas_int* info)
{
    return rocsolver_sbdsvdx(handle, uplo, svect, srange, n, D, E, vl, vu, il, iu, nsv, S, Z, ldz,
                             ifail, info);
}

inline rocblas_status rocsolver_bdsvdx(rocblas_handle handle,
                                       rocblas_fill uplo,
                                       rocblas_svect svect,
                                       rocblas_srange srange,
                                       rocblas_int n,
                                       double* D,
                                       double* E,
                                       double vl,
                                       double vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* nsv,
                                       double* S,
                                       double* Z,
                                       const rocblas_int ldz,
                                       rocblas_int* ifail,
                                       rocblas_int* info)
{
    return rocsolver_dbdsvdx(handle, uplo, svect, srange, n, D, E, vl, vu, il, iu, nsv, S, Z, ldz,
                             ifail, info);
}
/********************************************************/

/******************** POTF2_POTRF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_spotrf_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_spotf2_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_spotrf(handle, uplo, n, A, lda, info)
                     : rocsolver_spotf2(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_dpotrf_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_dpotf2_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_dpotrf(handle, uplo, n, A, lda, info)
                     : rocsolver_dpotf2(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_cpotrf_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_cpotf2_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_cpotrf(handle, uplo, n, A, lda, info)
                     : rocsolver_cpotf2(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_zpotrf_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_zpotf2_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_zpotrf(handle, uplo, n, A, lda, info)
                     : rocsolver_zpotf2(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            int64_t n,
                                            float* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* info,
                                            int64_t batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_spotrf_strided_batched_64(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_spotf2_strided_batched_64(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_spotrf_64(handle, uplo, n, A, lda, info)
                     : rocsolver_spotf2_64(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            int64_t n,
                                            double* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* info,
                                            int64_t batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_dpotrf_strided_batched_64(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_dpotf2_strided_batched_64(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_dpotrf_64(handle, uplo, n, A, lda, info)
                     : rocsolver_dpotf2_64(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            int64_t n,
                                            rocblas_float_complex* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* info,
                                            int64_t batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_cpotrf_strided_batched_64(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_cpotf2_strided_batched_64(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_cpotrf_64(handle, uplo, n, A, lda, info)
                     : rocsolver_cpotf2_64(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            int64_t n,
                                            rocblas_double_complex* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* info,
                                            int64_t batch_count)
{
    if(STRIDED)
        return POTRF
            ? rocsolver_zpotrf_strided_batched_64(handle, uplo, n, A, lda, stA, info, batch_count)
            : rocsolver_zpotf2_strided_batched_64(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return POTRF ? rocsolver_zpotrf_64(handle, uplo, n, A, lda, info)
                     : rocsolver_zpotf2_64(handle, uplo, n, A, lda, info);
}
// batched
inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    return POTRF ? rocsolver_spotrf_batched(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_spotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    return POTRF ? rocsolver_dpotrf_batched(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_dpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    return POTRF ? rocsolver_cpotrf_batched(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_cpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* info,
                                            rocblas_int batch_count)
{
    return POTRF ? rocsolver_zpotrf_batched(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_zpotf2_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            int64_t n,
                                            float* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* info,
                                            int64_t batch_count)
{
    return POTRF ? rocsolver_spotrf_batched_64(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_spotf2_batched_64(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            int64_t n,
                                            double* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* info,
                                            int64_t batch_count)
{
    return POTRF ? rocsolver_dpotrf_batched_64(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_dpotf2_batched_64(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            int64_t n,
                                            rocblas_float_complex* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* info,
                                            int64_t batch_count)
{
    return POTRF ? rocsolver_cpotrf_batched_64(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_cpotf2_batched_64(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potf2_potrf(bool STRIDED,
                                            bool POTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            int64_t n,
                                            rocblas_double_complex* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* info,
                                            int64_t batch_count)
{
    return POTRF ? rocsolver_zpotrf_batched_64(handle, uplo, n, A, lda, info, batch_count)
                 : rocsolver_zpotf2_batched_64(handle, uplo, n, A, lda, info, batch_count);
}
/********************************************************/

/******************** POTRS ********************/
// normal and strided_batched
inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_spotrs_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                                batch_count);
    else
        return rocsolver_spotrs(handle, uplo, n, nrhs, A, lda, B, ldb);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_dpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                                batch_count);
    else
        return rocsolver_dpotrs(handle, uplo, n, nrhs, A, lda, B, ldb);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_float_complex* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_cpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                                batch_count);
    else
        return rocsolver_cpotrs(handle, uplo, n, nrhs, A, lda, B, ldb);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_zpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                                batch_count);
    else
        return rocsolver_zpotrs(handle, uplo, n, nrhs, A, lda, B, ldb);
}

// batched
inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    return rocsolver_spotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    return rocsolver_dpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_float_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    return rocsolver_cpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}

inline rocblas_status rocsolver_potrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_double_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int batch_count)
{
    return rocsolver_zpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}
/********************************************************/

/******************** POSV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     float* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_sposv_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, batch_count);
    else
        return rocsolver_sposv(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     double* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_dposv_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, batch_count);
    else
        return rocsolver_dposv(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_cposv_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, batch_count);
    else
        return rocsolver_cposv(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_double_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_zposv_strided_batched(handle, uplo, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, batch_count);
    else
        return rocsolver_zposv(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

// batched
inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     float* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    return rocsolver_sposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, info, batch_count);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     double* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    return rocsolver_dposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, info, batch_count);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_float_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    return rocsolver_cposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, info, batch_count);
}

inline rocblas_status rocsolver_posv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_double_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int batch_count)
{
    return rocsolver_zposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, info, batch_count);
}
/********************************************************/

/******************** POTRI ********************/
// normal and strided_batched
inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_spotri_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return rocsolver_spotri(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_dpotri_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return rocsolver_dpotri(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_cpotri_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return rocsolver_cpotri(handle, uplo, n, A, lda, info);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    if(STRIDED)
        return rocsolver_zpotri_strided_batched(handle, uplo, n, A, lda, stA, info, batch_count);
    else
        return rocsolver_zpotri(handle, uplo, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    return rocsolver_spotri_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    return rocsolver_dpotri_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    return rocsolver_cpotri_batched(handle, uplo, n, A, lda, info, batch_count);
}

inline rocblas_status rocsolver_potri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int batch_count)
{
    return rocsolver_zpotri_batched(handle, uplo, n, A, lda, info, batch_count);
}
/********************************************************/

/******************** GETF2_GETRF_NPVT ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 float* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_sgetrf_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_sgetf2_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_sgetrf_npvt(handle, m, n, A, lda, info)
                     : rocsolver_sgetf2_npvt(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 double* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_dgetrf_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_dgetf2_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_dgetrf_npvt(handle, m, n, A, lda, info)
                     : rocsolver_dgetf2_npvt(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_cgetrf_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_cgetf2_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_cgetrf_npvt(handle, m, n, A, lda, info)
                     : rocsolver_cgetf2_npvt(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_zgetrf_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_zgetf2_npvt_strided_batched(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_zgetrf_npvt(handle, m, n, A, lda, info)
                     : rocsolver_zgetf2_npvt(handle, m, n, A, lda, info);
}
inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 int64_t m,
                                                 int64_t n,
                                                 float* A,
                                                 int64_t lda,
                                                 rocblas_stride stA,
                                                 int64_t* info,
                                                 int64_t bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_sgetrf_npvt_strided_batched_64(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_sgetf2_npvt_strided_batched_64(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_sgetrf_npvt_64(handle, m, n, A, lda, info)
                     : rocsolver_sgetf2_npvt_64(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 int64_t m,
                                                 int64_t n,
                                                 double* A,
                                                 int64_t lda,
                                                 rocblas_stride stA,
                                                 int64_t* info,
                                                 int64_t bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_dgetrf_npvt_strided_batched_64(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_dgetf2_npvt_strided_batched_64(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_dgetrf_npvt_64(handle, m, n, A, lda, info)
                     : rocsolver_dgetf2_npvt_64(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 int64_t m,
                                                 int64_t n,
                                                 rocblas_float_complex* A,
                                                 int64_t lda,
                                                 rocblas_stride stA,
                                                 int64_t* info,
                                                 int64_t bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_cgetrf_npvt_strided_batched_64(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_cgetf2_npvt_strided_batched_64(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_cgetrf_npvt_64(handle, m, n, A, lda, info)
                     : rocsolver_cgetf2_npvt_64(handle, m, n, A, lda, info);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 int64_t m,
                                                 int64_t n,
                                                 rocblas_double_complex* A,
                                                 int64_t lda,
                                                 rocblas_stride stA,
                                                 int64_t* info,
                                                 int64_t bc)
{
    if(STRIDED)
        return GETRF ? rocsolver_zgetrf_npvt_strided_batched_64(handle, m, n, A, lda, stA, info, bc)
                     : rocsolver_zgetf2_npvt_strided_batched_64(handle, m, n, A, lda, stA, info, bc);
    else
        return GETRF ? rocsolver_zgetrf_npvt_64(handle, m, n, A, lda, info)
                     : rocsolver_zgetf2_npvt_64(handle, m, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 float* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return GETRF ? rocsolver_sgetrf_npvt_batched(handle, m, n, A, lda, info, bc)
                 : rocsolver_sgetf2_npvt_batched(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 double* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return GETRF ? rocsolver_dgetrf_npvt_batched(handle, m, n, A, lda, info, bc)
                 : rocsolver_dgetf2_npvt_batched(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_float_complex* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return GETRF ? rocsolver_cgetrf_npvt_batched(handle, m, n, A, lda, info, bc)
                 : rocsolver_cgetf2_npvt_batched(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return GETRF ? rocsolver_zgetrf_npvt_batched(handle, m, n, A, lda, info, bc)
                 : rocsolver_zgetf2_npvt_batched(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 int64_t m,
                                                 int64_t n,
                                                 float* const A[],
                                                 int64_t lda,
                                                 rocblas_stride stA,
                                                 int64_t* info,
                                                 int64_t bc)
{
    return GETRF ? rocsolver_sgetrf_npvt_batched_64(handle, m, n, A, lda, info, bc)
                 : rocsolver_sgetf2_npvt_batched_64(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 int64_t m,
                                                 int64_t n,
                                                 double* const A[],
                                                 int64_t lda,
                                                 rocblas_stride stA,
                                                 int64_t* info,
                                                 int64_t bc)
{
    return GETRF ? rocsolver_dgetrf_npvt_batched_64(handle, m, n, A, lda, info, bc)
                 : rocsolver_dgetf2_npvt_batched_64(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 int64_t m,
                                                 int64_t n,
                                                 rocblas_float_complex* const A[],
                                                 int64_t lda,
                                                 rocblas_stride stA,
                                                 int64_t* info,
                                                 int64_t bc)
{
    return GETRF ? rocsolver_cgetrf_npvt_batched_64(handle, m, n, A, lda, info, bc)
                 : rocsolver_cgetf2_npvt_batched_64(handle, m, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf_npvt(bool STRIDED,
                                                 bool GETRF,
                                                 rocblas_handle handle,
                                                 int64_t m,
                                                 int64_t n,
                                                 rocblas_double_complex* const A[],
                                                 int64_t lda,
                                                 rocblas_stride stA,
                                                 int64_t* info,
                                                 int64_t bc)
{
    return GETRF ? rocsolver_zgetrf_npvt_batched_64(handle, m, n, A, lda, info, bc)
                 : rocsolver_zgetf2_npvt_batched_64(handle, m, n, A, lda, info, bc);
}
/********************************************************/

/******************** GETF2_GETRF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_sgetrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_sgetf2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_sgetrf(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_sgetf2(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_dgetrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_dgetf2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_dgetrf(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_dgetf2(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_cgetrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_cgetf2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_cgetrf(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_cgetf2(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_zgetrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_zgetf2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_zgetrf(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_zgetf2(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            float* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* ipiv,
                                            rocblas_stride stP,
                                            int64_t* info,
                                            int64_t bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_sgetrf_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_sgetf2_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_sgetrf_64(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_sgetf2_64(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            double* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* ipiv,
                                            rocblas_stride stP,
                                            int64_t* info,
                                            int64_t bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_dgetrf_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_dgetf2_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_dgetrf_64(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_dgetf2_64(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            rocblas_float_complex* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* ipiv,
                                            rocblas_stride stP,
                                            int64_t* info,
                                            int64_t bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_cgetrf_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_cgetf2_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_cgetrf_64(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_cgetf2_64(handle, m, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            rocblas_double_complex* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* ipiv,
                                            rocblas_stride stP,
                                            int64_t* info,
                                            int64_t bc)
{
    if(STRIDED)
        return GETRF
            ? rocsolver_zgetrf_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_zgetf2_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return GETRF ? rocsolver_zgetrf_64(handle, m, n, A, lda, ipiv, info)
                     : rocsolver_zgetf2_64(handle, m, n, A, lda, ipiv, info);
}

// batched
inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return GETRF ? rocsolver_sgetrf_batched(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_sgetf2_batched(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return GETRF ? rocsolver_dgetrf_batched(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_dgetf2_batched(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return GETRF ? rocsolver_cgetrf_batched(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_cgetf2_batched(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return GETRF ? rocsolver_zgetrf_batched(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_zgetf2_batched(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            float* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* ipiv,
                                            rocblas_stride stP,
                                            int64_t* info,
                                            int64_t bc)
{
    return GETRF ? rocsolver_sgetrf_batched_64(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_sgetf2_batched_64(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            double* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* ipiv,
                                            rocblas_stride stP,
                                            int64_t* info,
                                            int64_t bc)
{
    return GETRF ? rocsolver_dgetrf_batched_64(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_dgetf2_batched_64(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            rocblas_float_complex* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* ipiv,
                                            rocblas_stride stP,
                                            int64_t* info,
                                            int64_t bc)
{
    return GETRF ? rocsolver_cgetrf_batched_64(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_cgetf2_batched_64(handle, m, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getf2_getrf(bool STRIDED,
                                            bool GETRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            rocblas_double_complex* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            int64_t* ipiv,
                                            rocblas_stride stP,
                                            int64_t* info,
                                            int64_t bc)
{
    return GETRF ? rocsolver_zgetrf_batched_64(handle, m, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_zgetf2_batched_64(handle, m, n, A, lda, ipiv, stP, info, bc);
}
/********************************************************/

/******************** GESVD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      float* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      float* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      float* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_sgesvd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S, stS, U, ldu,
                                           stU, V, ldv, stV, E, stE, fast_alg, info, bc)
        : rocsolver_sgesvd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      double* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      double* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      double* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dgesvd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S, stS, U, ldu,
                                           stU, V, ldv, stV, E, stE, fast_alg, info, bc)
        : rocsolver_dgesvd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      rocblas_float_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      float* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cgesvd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S, stS, U, ldu,
                                           stU, V, ldv, stV, E, stE, fast_alg, info, bc)
        : rocsolver_cgesvd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      rocblas_double_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      double* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zgesvd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S, stS, U, ldu,
                                           stU, V, ldv, stV, E, stE, fast_alg, info, bc)
        : rocsolver_zgesvd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

// batched
inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      float* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      float* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      float* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_sgesvd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, E, stE, fast_alg, info, bc);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      double* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      double* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      double* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_dgesvd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, E, stE, fast_alg, info, bc);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      rocblas_float_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      float* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_cgesvd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, E, stE, fast_alg, info, bc);
}

inline rocblas_status rocsolver_gesvd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      rocblas_double_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      double* E,
                                      rocblas_stride stE,
                                      rocblas_workmode fast_alg,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_zgesvd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, E, stE, fast_alg, info, bc);
}
/********************************************************/

/******************** GESDD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesdd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      float* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      float* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_sgesdd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S,
                                                      stS, U, ldu, stU, V, ldv, stV, info, bc)
                   : rocsolver_sgesdd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesdd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      double* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      double* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dgesdd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S,
                                                      stS, U, ldu, stU, V, ldv, stV, info, bc)
                   : rocsolver_dgesdd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesdd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      rocblas_float_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_cgesdd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S,
                                                      stS, U, ldu, stU, V, ldv, stV, info, bc)
                   : rocsolver_cgesdd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesdd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      rocblas_double_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_zgesdd_strided_batched(handle, leftv, rightv, m, n, A, lda, stA, S,
                                                      stS, U, ldu, stU, V, ldv, stV, info, bc)
                   : rocsolver_zgesdd(handle, leftv, rightv, m, n, A, lda, S, U, ldu, V, ldv, info);
}

// batched
inline rocblas_status rocsolver_gesdd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      float* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      float* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_sgesdd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, info, bc);
}

inline rocblas_status rocsolver_gesdd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      double* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      double* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_dgesdd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, info, bc);
}

inline rocblas_status rocsolver_gesdd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      float* S,
                                      rocblas_stride stS,
                                      rocblas_float_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_cgesdd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, info, bc);
}

inline rocblas_status rocsolver_gesdd(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_svect leftv,
                                      rocblas_svect rightv,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      double* S,
                                      rocblas_stride stS,
                                      rocblas_double_complex* U,
                                      rocblas_int ldu,
                                      rocblas_stride stU,
                                      rocblas_double_complex* V,
                                      rocblas_int ldv,
                                      rocblas_stride stV,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_zgesdd_batched(handle, leftv, rightv, m, n, A, lda, S, stS, U, ldu, stU, V,
                                    ldv, stV, info, bc);
}
/********************************************************/

/******************** GESVDJ ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesvdj(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_int m,
                                       rocblas_int n,
                                       float* A,
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       float abstol,
                                       float* residual,
                                       rocblas_int max_sweeps,
                                       rocblas_int* n_sweeps,
                                       float* S,
                                       rocblas_stride stS,
                                       float* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       float* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return STRIDED ? rocsolver_sgesvdj_strided_batched(handle, leftv, rightv, m, n, A, lda, stA,
                                                       abstol, residual, max_sweeps, n_sweeps, S,
                                                       stS, U, ldu, stU, V, ldv, stV, info, bc)
                   : rocsolver_sgesvdj(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                       max_sweeps, n_sweeps, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesvdj(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_int m,
                                       rocblas_int n,
                                       double* A,
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       double abstol,
                                       double* residual,
                                       rocblas_int max_sweeps,
                                       rocblas_int* n_sweeps,
                                       double* S,
                                       rocblas_stride stS,
                                       double* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       double* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return STRIDED ? rocsolver_dgesvdj_strided_batched(handle, leftv, rightv, m, n, A, lda, stA,
                                                       abstol, residual, max_sweeps, n_sweeps, S,
                                                       stS, U, ldu, stU, V, ldv, stV, info, bc)
                   : rocsolver_dgesvdj(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                       max_sweeps, n_sweeps, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesvdj(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       float abstol,
                                       float* residual,
                                       rocblas_int max_sweeps,
                                       rocblas_int* n_sweeps,
                                       float* S,
                                       rocblas_stride stS,
                                       rocblas_float_complex* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       rocblas_float_complex* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return STRIDED ? rocsolver_cgesvdj_strided_batched(handle, leftv, rightv, m, n, A, lda, stA,
                                                       abstol, residual, max_sweeps, n_sweeps, S,
                                                       stS, U, ldu, stU, V, ldv, stV, info, bc)
                   : rocsolver_cgesvdj(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                       max_sweeps, n_sweeps, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesvdj(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       double abstol,
                                       double* residual,
                                       rocblas_int max_sweeps,
                                       rocblas_int* n_sweeps,
                                       double* S,
                                       rocblas_stride stS,
                                       rocblas_double_complex* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       rocblas_double_complex* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return STRIDED ? rocsolver_zgesvdj_strided_batched(handle, leftv, rightv, m, n, A, lda, stA,
                                                       abstol, residual, max_sweeps, n_sweeps, S,
                                                       stS, U, ldu, stU, V, ldv, stV, info, bc)
                   : rocsolver_zgesvdj(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                       max_sweeps, n_sweeps, S, U, ldu, V, ldv, info);
}

// batched
inline rocblas_status rocsolver_gesvdj(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_int m,
                                       rocblas_int n,
                                       float* const A[],
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       float abstol,
                                       float* residual,
                                       rocblas_int max_sweeps,
                                       rocblas_int* n_sweeps,
                                       float* S,
                                       rocblas_stride stS,
                                       float* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       float* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return rocsolver_sgesvdj_batched(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                     max_sweeps, n_sweeps, S, stS, U, ldu, stU, V, ldv, stV, info,
                                     bc);
}

inline rocblas_status rocsolver_gesvdj(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_int m,
                                       rocblas_int n,
                                       double* const A[],
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       double abstol,
                                       double* residual,
                                       rocblas_int max_sweeps,
                                       rocblas_int* n_sweeps,
                                       double* S,
                                       rocblas_stride stS,
                                       double* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       double* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return rocsolver_dgesvdj_batched(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                     max_sweeps, n_sweeps, S, stS, U, ldu, stU, V, ldv, stV, info,
                                     bc);
}

inline rocblas_status rocsolver_gesvdj(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex* const A[],
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       float abstol,
                                       float* residual,
                                       rocblas_int max_sweeps,
                                       rocblas_int* n_sweeps,
                                       float* S,
                                       rocblas_stride stS,
                                       rocblas_float_complex* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       rocblas_float_complex* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return rocsolver_cgesvdj_batched(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                     max_sweeps, n_sweeps, S, stS, U, ldu, stU, V, ldv, stV, info,
                                     bc);
}

inline rocblas_status rocsolver_gesvdj(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* const A[],
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       double abstol,
                                       double* residual,
                                       rocblas_int max_sweeps,
                                       rocblas_int* n_sweeps,
                                       double* S,
                                       rocblas_stride stS,
                                       rocblas_double_complex* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       rocblas_double_complex* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return rocsolver_zgesvdj_batched(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                     max_sweeps, n_sweeps, S, stS, U, ldu, stU, V, ldv, stV, info,
                                     bc);
}
/********************************************************/

/******************** GESVDJ_NOTRANSV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesvdj_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_int m,
                                                rocblas_int n,
                                                float* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float abstol,
                                                float* residual,
                                                rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                float* S,
                                                rocblas_stride stS,
                                                float* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                float* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocsolver_sgesvdj_notransv_strided_batched(handle, leftv, rightv, m, n, A, lda, stA,
                                                     abstol, residual, max_sweeps, n_sweeps, S, stS,
                                                     U, ldu, stU, V, ldv, stV, info, bc)
        : rocsolver_sgesvdj_notransv(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                     max_sweeps, n_sweeps, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesvdj_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_int m,
                                                rocblas_int n,
                                                double* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double abstol,
                                                double* residual,
                                                rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                double* S,
                                                rocblas_stride stS,
                                                double* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                double* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dgesvdj_notransv_strided_batched(handle, leftv, rightv, m, n, A, lda, stA,
                                                     abstol, residual, max_sweeps, n_sweeps, S, stS,
                                                     U, ldu, stU, V, ldv, stV, info, bc)
        : rocsolver_dgesvdj_notransv(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                     max_sweeps, n_sweeps, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesvdj_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_float_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float abstol,
                                                float* residual,
                                                rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                float* S,
                                                rocblas_stride stS,
                                                rocblas_float_complex* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                rocblas_float_complex* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cgesvdj_notransv_strided_batched(handle, leftv, rightv, m, n, A, lda, stA,
                                                     abstol, residual, max_sweeps, n_sweeps, S, stS,
                                                     U, ldu, stU, V, ldv, stV, info, bc)
        : rocsolver_cgesvdj_notransv(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                     max_sweeps, n_sweeps, S, U, ldu, V, ldv, info);
}

inline rocblas_status rocsolver_gesvdj_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_double_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double abstol,
                                                double* residual,
                                                rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                double* S,
                                                rocblas_stride stS,
                                                rocblas_double_complex* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                rocblas_double_complex* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zgesvdj_notransv_strided_batched(handle, leftv, rightv, m, n, A, lda, stA,
                                                     abstol, residual, max_sweeps, n_sweeps, S, stS,
                                                     U, ldu, stU, V, ldv, stV, info, bc)
        : rocsolver_zgesvdj_notransv(handle, leftv, rightv, m, n, A, lda, abstol, residual,
                                     max_sweeps, n_sweeps, S, U, ldu, V, ldv, info);
}

// batched
inline rocblas_status rocsolver_gesvdj_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_int m,
                                                rocblas_int n,
                                                float* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float abstol,
                                                float* residual,
                                                rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                float* S,
                                                rocblas_stride stS,
                                                float* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                float* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_sgesvdj_notransv_batched(handle, leftv, rightv, m, n, A, lda, abstol,
        // residual, max_sweeps, n_sweeps, S, stS, U, ldu, stU, V, ldv, stV, info, bc);
}

inline rocblas_status rocsolver_gesvdj_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_int m,
                                                rocblas_int n,
                                                double* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double abstol,
                                                double* residual,
                                                rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                double* S,
                                                rocblas_stride stS,
                                                double* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                double* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_dgesvdj_notransv_batched(handle, leftv, rightv, m, n, A, lda, abstol,
        // residual, max_sweeps, n_sweeps, S, stS, U, ldu, stU, V, ldv, stV, info, bc);
}

inline rocblas_status rocsolver_gesvdj_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_float_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float abstol,
                                                float* residual,
                                                rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                float* S,
                                                rocblas_stride stS,
                                                rocblas_float_complex* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                rocblas_float_complex* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_cgesvdj_notransv_batched(handle, leftv, rightv, m, n, A, lda, abstol,
        // residual, max_sweeps, n_sweeps, S, stS, U, ldu, stU, V, ldv, stV, info, bc);
}

inline rocblas_status rocsolver_gesvdj_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_double_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double abstol,
                                                double* residual,
                                                rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                double* S,
                                                rocblas_stride stS,
                                                rocblas_double_complex* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                rocblas_double_complex* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_zgesvdj_notransv_batched(handle, leftv, rightv, m, n, A, lda, abstol,
        // residual, max_sweeps, n_sweeps, S, stS, U, ldu, stU, V, ldv, stV, info, bc);
}
/********************************************************/

/******************** GESVDX ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesvdx(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_srange srange,
                                       rocblas_int m,
                                       rocblas_int n,
                                       float* A,
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       float vl,
                                       float vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* ns,
                                       float* S,
                                       rocblas_stride stS,
                                       float* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       float* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* ifail,
                                       rocblas_stride stF,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return STRIDED ? rocsolver_sgesvdx_strided_batched(handle, leftv, rightv, srange, m, n, A, lda,
                                                       stA, vl, vu, il, iu, ns, S, stS, U, ldu, stU,
                                                       V, ldv, stV, ifail, stF, info, bc)
                   : rocsolver_sgesvdx(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
                                       ns, S, U, ldu, V, ldv, ifail, info);
}

inline rocblas_status rocsolver_gesvdx(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_srange srange,
                                       rocblas_int m,
                                       rocblas_int n,
                                       double* A,
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       double vl,
                                       double vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* ns,
                                       double* S,
                                       rocblas_stride stS,
                                       double* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       double* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* ifail,
                                       rocblas_stride stF,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return STRIDED ? rocsolver_dgesvdx_strided_batched(handle, leftv, rightv, srange, m, n, A, lda,
                                                       stA, vl, vu, il, iu, ns, S, stS, U, ldu, stU,
                                                       V, ldv, stV, ifail, stF, info, bc)
                   : rocsolver_dgesvdx(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
                                       ns, S, U, ldu, V, ldv, ifail, info);
}

inline rocblas_status rocsolver_gesvdx(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_srange srange,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex* A,
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       float vl,
                                       float vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* ns,
                                       float* S,
                                       rocblas_stride stS,
                                       rocblas_float_complex* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       rocblas_float_complex* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* ifail,
                                       rocblas_stride stF,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return STRIDED ? rocsolver_cgesvdx_strided_batched(handle, leftv, rightv, srange, m, n, A, lda,
                                                       stA, vl, vu, il, iu, ns, S, stS, U, ldu, stU,
                                                       V, ldv, stV, ifail, stF, info, bc)
                   : rocsolver_cgesvdx(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
                                       ns, S, U, ldu, V, ldv, ifail, info);
}

inline rocblas_status rocsolver_gesvdx(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_srange srange,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       double vl,
                                       double vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* ns,
                                       double* S,
                                       rocblas_stride stS,
                                       rocblas_double_complex* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       rocblas_double_complex* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* ifail,
                                       rocblas_stride stF,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return STRIDED ? rocsolver_zgesvdx_strided_batched(handle, leftv, rightv, srange, m, n, A, lda,
                                                       stA, vl, vu, il, iu, ns, S, stS, U, ldu, stU,
                                                       V, ldv, stV, ifail, stF, info, bc)
                   : rocsolver_zgesvdx(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
                                       ns, S, U, ldu, V, ldv, ifail, info);
}

// batched
inline rocblas_status rocsolver_gesvdx(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_srange srange,
                                       rocblas_int m,
                                       rocblas_int n,
                                       float* const A[],
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       float vl,
                                       float vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* ns,
                                       float* S,
                                       rocblas_stride stS,
                                       float* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       float* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* ifail,
                                       rocblas_stride stF,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return rocsolver_sgesvdx_batched(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
                                     ns, S, stS, U, ldu, stU, V, ldv, stV, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_gesvdx(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_srange srange,
                                       rocblas_int m,
                                       rocblas_int n,
                                       double* const A[],
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       double vl,
                                       double vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* ns,
                                       double* S,
                                       rocblas_stride stS,
                                       double* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       double* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* ifail,
                                       rocblas_stride stF,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return rocsolver_dgesvdx_batched(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
                                     ns, S, stS, U, ldu, stU, V, ldv, stV, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_gesvdx(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_srange srange,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_float_complex* const A[],
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       float vl,
                                       float vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* ns,
                                       float* S,
                                       rocblas_stride stS,
                                       rocblas_float_complex* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       rocblas_float_complex* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* ifail,
                                       rocblas_stride stF,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return rocsolver_cgesvdx_batched(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
                                     ns, S, stS, U, ldu, stU, V, ldv, stV, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_gesvdx(bool STRIDED,
                                       rocblas_handle handle,
                                       rocblas_svect leftv,
                                       rocblas_svect rightv,
                                       rocblas_srange srange,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* const A[],
                                       rocblas_int lda,
                                       rocblas_stride stA,
                                       double vl,
                                       double vu,
                                       rocblas_int il,
                                       rocblas_int iu,
                                       rocblas_int* ns,
                                       double* S,
                                       rocblas_stride stS,
                                       rocblas_double_complex* U,
                                       rocblas_int ldu,
                                       rocblas_stride stU,
                                       rocblas_double_complex* V,
                                       rocblas_int ldv,
                                       rocblas_stride stV,
                                       rocblas_int* ifail,
                                       rocblas_stride stF,
                                       rocblas_int* info,
                                       rocblas_int bc)
{
    return rocsolver_zgesvdx_batched(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
                                     ns, S, stS, U, ldu, stU, V, ldv, stV, ifail, stF, info, bc);
}
/********************************************************/

/******************** GESVDX_NOTRANSV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesvdx_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_srange srange,
                                                rocblas_int m,
                                                rocblas_int n,
                                                float* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float vl,
                                                float vu,
                                                rocblas_int il,
                                                rocblas_int iu,
                                                rocblas_int* ns,
                                                float* S,
                                                rocblas_stride stS,
                                                float* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                float* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* ifail,
                                                rocblas_stride stF,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocsolver_sgesvdx_notransv_strided_batched(handle, leftv, rightv, srange, m, n, A, lda,
                                                     stA, vl, vu, il, iu, ns, S, stS, U, ldu, stU,
                                                     V, ldv, stV, ifail, stF, info, bc)
        : rocblas_status_not_implemented; // rocsolver_sgesvdx_notransv(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
    // ns, S, U, ldu, V, ldv, ifail, info);
}

inline rocblas_status rocsolver_gesvdx_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_srange srange,
                                                rocblas_int m,
                                                rocblas_int n,
                                                double* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double vl,
                                                double vu,
                                                rocblas_int il,
                                                rocblas_int iu,
                                                rocblas_int* ns,
                                                double* S,
                                                rocblas_stride stS,
                                                double* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                double* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* ifail,
                                                rocblas_stride stF,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dgesvdx_notransv_strided_batched(handle, leftv, rightv, srange, m, n, A, lda,
                                                     stA, vl, vu, il, iu, ns, S, stS, U, ldu, stU,
                                                     V, ldv, stV, ifail, stF, info, bc)
        : rocblas_status_not_implemented; // rocsolver_dgesvdx_notransv(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
    // ns, S, U, ldu, V, ldv, ifail, info);
}

inline rocblas_status rocsolver_gesvdx_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_srange srange,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_float_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float vl,
                                                float vu,
                                                rocblas_int il,
                                                rocblas_int iu,
                                                rocblas_int* ns,
                                                float* S,
                                                rocblas_stride stS,
                                                rocblas_float_complex* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                rocblas_float_complex* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* ifail,
                                                rocblas_stride stF,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cgesvdx_notransv_strided_batched(handle, leftv, rightv, srange, m, n, A, lda,
                                                     stA, vl, vu, il, iu, ns, S, stS, U, ldu, stU,
                                                     V, ldv, stV, ifail, stF, info, bc)
        : rocblas_status_not_implemented; // rocsolver_cgesvdx_notransv(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
    // ns, S, U, ldu, V, ldv, ifail, info);
}

inline rocblas_status rocsolver_gesvdx_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_srange srange,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_double_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double vl,
                                                double vu,
                                                rocblas_int il,
                                                rocblas_int iu,
                                                rocblas_int* ns,
                                                double* S,
                                                rocblas_stride stS,
                                                rocblas_double_complex* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                rocblas_double_complex* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* ifail,
                                                rocblas_stride stF,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zgesvdx_notransv_strided_batched(handle, leftv, rightv, srange, m, n, A, lda,
                                                     stA, vl, vu, il, iu, ns, S, stS, U, ldu, stU,
                                                     V, ldv, stV, ifail, stF, info, bc)
        : rocblas_status_not_implemented; // rocsolver_zgesvdx_notransv(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
    // ns, S, U, ldu, V, ldv, ifail, info);
}

// batched
inline rocblas_status rocsolver_gesvdx_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_srange srange,
                                                rocblas_int m,
                                                rocblas_int n,
                                                float* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float vl,
                                                float vu,
                                                rocblas_int il,
                                                rocblas_int iu,
                                                rocblas_int* ns,
                                                float* S,
                                                rocblas_stride stS,
                                                float* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                float* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* ifail,
                                                rocblas_stride stF,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_sgesvdx_notransv_batched(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
        // ns, S, stS, U, ldu, stU, V, ldv, stV, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_gesvdx_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_srange srange,
                                                rocblas_int m,
                                                rocblas_int n,
                                                double* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double vl,
                                                double vu,
                                                rocblas_int il,
                                                rocblas_int iu,
                                                rocblas_int* ns,
                                                double* S,
                                                rocblas_stride stS,
                                                double* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                double* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* ifail,
                                                rocblas_stride stF,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_dgesvdx_notransv_batched(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
        // ns, S, stS, U, ldu, stU, V, ldv, stV, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_gesvdx_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_srange srange,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_float_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float vl,
                                                float vu,
                                                rocblas_int il,
                                                rocblas_int iu,
                                                rocblas_int* ns,
                                                float* S,
                                                rocblas_stride stS,
                                                rocblas_float_complex* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                rocblas_float_complex* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* ifail,
                                                rocblas_stride stF,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_cgesvdx_notransv_batched(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
        // ns, S, stS, U, ldu, stU, V, ldv, stV, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_gesvdx_notransv(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_svect leftv,
                                                rocblas_svect rightv,
                                                rocblas_srange srange,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_double_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double vl,
                                                double vu,
                                                rocblas_int il,
                                                rocblas_int iu,
                                                rocblas_int* ns,
                                                double* S,
                                                rocblas_stride stS,
                                                rocblas_double_complex* U,
                                                rocblas_int ldu,
                                                rocblas_stride stU,
                                                rocblas_double_complex* V,
                                                rocblas_int ldv,
                                                rocblas_stride stV,
                                                rocblas_int* ifail,
                                                rocblas_stride stF,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_zgesvdx_notransv_batched(handle, leftv, rightv, srange, m, n, A, lda, vl, vu, il, iu,
        // ns, S, stS, U, ldu, stU, V, ldv, stV, ifail, stF, info, bc);
}
/********************************************************/

/******************** GETRS ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      float* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetrs_strided_batched(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                      stP, B, ldb, stB, bc)
                   : rocsolver_sgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      double* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetrs_strided_batched(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                      stP, B, ldb, stB, bc)
                   : rocsolver_dgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_float_complex* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetrs_strided_batched(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                      stP, B, ldb, stB, bc)
                   : rocsolver_cgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetrs_strided_batched(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                      stP, B, ldb, stB, bc)
                   : rocsolver_zgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      int64_t n,
                                      int64_t nrhs,
                                      float* A,
                                      int64_t lda,
                                      rocblas_stride stA,
                                      int64_t* ipiv,
                                      rocblas_stride stP,
                                      float* B,
                                      int64_t ldb,
                                      rocblas_stride stB,
                                      int64_t bc)
{
    return STRIDED ? rocsolver_sgetrs_strided_batched_64(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                         stP, B, ldb, stB, bc)
                   : rocsolver_sgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      int64_t n,
                                      int64_t nrhs,
                                      double* A,
                                      int64_t lda,
                                      rocblas_stride stA,
                                      int64_t* ipiv,
                                      rocblas_stride stP,
                                      double* B,
                                      int64_t ldb,
                                      rocblas_stride stB,
                                      int64_t bc)
{
    return STRIDED ? rocsolver_dgetrs_strided_batched_64(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                         stP, B, ldb, stB, bc)
                   : rocsolver_dgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      int64_t n,
                                      int64_t nrhs,
                                      rocblas_float_complex* A,
                                      int64_t lda,
                                      rocblas_stride stA,
                                      int64_t* ipiv,
                                      rocblas_stride stP,
                                      rocblas_float_complex* B,
                                      int64_t ldb,
                                      rocblas_stride stB,
                                      int64_t bc)
{
    return STRIDED ? rocsolver_cgetrs_strided_batched_64(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                         stP, B, ldb, stB, bc)
                   : rocsolver_cgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      int64_t n,
                                      int64_t nrhs,
                                      rocblas_double_complex* A,
                                      int64_t lda,
                                      rocblas_stride stA,
                                      int64_t* ipiv,
                                      rocblas_stride stP,
                                      rocblas_double_complex* B,
                                      int64_t ldb,
                                      rocblas_stride stB,
                                      int64_t bc)
{
    return STRIDED ? rocsolver_zgetrs_strided_batched_64(handle, trans, n, nrhs, A, lda, stA, ipiv,
                                                         stP, B, ldb, stB, bc)
                   : rocsolver_zgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

// batched
inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      float* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return rocsolver_sgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      double* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return rocsolver_dgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_float_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return rocsolver_cgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_double_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_stride stB,
                                      rocblas_int bc)
{
    return rocsolver_zgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      int64_t n,
                                      int64_t nrhs,
                                      float* const A[],
                                      int64_t lda,
                                      rocblas_stride stA,
                                      int64_t* ipiv,
                                      rocblas_stride stP,
                                      float* const B[],
                                      int64_t ldb,
                                      rocblas_stride stB,
                                      int64_t bc)
{
    return rocsolver_sgetrs_batched_64(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      int64_t n,
                                      int64_t nrhs,
                                      double* const A[],
                                      int64_t lda,
                                      rocblas_stride stA,
                                      int64_t* ipiv,
                                      rocblas_stride stP,
                                      double* const B[],
                                      int64_t ldb,
                                      rocblas_stride stB,
                                      int64_t bc)
{
    return rocsolver_dgetrs_batched_64(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      int64_t n,
                                      int64_t nrhs,
                                      rocblas_float_complex* const A[],
                                      int64_t lda,
                                      rocblas_stride stA,
                                      int64_t* ipiv,
                                      rocblas_stride stP,
                                      rocblas_float_complex* const B[],
                                      int64_t ldb,
                                      rocblas_stride stB,
                                      int64_t bc)
{
    return rocsolver_cgetrs_batched_64(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}

inline rocblas_status rocsolver_getrs(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_operation trans,
                                      int64_t n,
                                      int64_t nrhs,
                                      rocblas_double_complex* const A[],
                                      int64_t lda,
                                      rocblas_stride stA,
                                      int64_t* ipiv,
                                      rocblas_stride stP,
                                      rocblas_double_complex* const B[],
                                      int64_t ldb,
                                      rocblas_stride stB,
                                      int64_t bc)
{
    return rocsolver_zgetrs_batched_64(handle, trans, n, nrhs, A, lda, ipiv, stP, B, ldb, bc);
}
/********************************************************/

/******************** GESV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_int* ipiv,
                                     rocblas_stride stP,
                                     float* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return STRIDED ? rocsolver_sgesv_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B,
                                                     ldb, stB, info, bc)
                   : rocsolver_sgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_int* ipiv,
                                     rocblas_stride stP,
                                     double* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return STRIDED ? rocsolver_dgesv_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B,
                                                     ldb, stB, info, bc)
                   : rocsolver_dgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_int* ipiv,
                                     rocblas_stride stP,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return STRIDED ? rocsolver_cgesv_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B,
                                                     ldb, stB, info, bc)
                   : rocsolver_cgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_int* ipiv,
                                     rocblas_stride stP,
                                     rocblas_double_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return STRIDED ? rocsolver_zgesv_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B,
                                                     ldb, stB, info, bc)
                   : rocsolver_zgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

// batched
inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_int* ipiv,
                                     rocblas_stride stP,
                                     float* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_sgesv_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_int* ipiv,
                                     rocblas_stride stP,
                                     double* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_dgesv_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_int* ipiv,
                                     rocblas_stride stP,
                                     rocblas_float_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_cgesv_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_int* ipiv,
                                     rocblas_stride stP,
                                     rocblas_double_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_zgesv_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}
/********************************************************/

/******************** GESV_OUTOFPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                float* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_int* ipiv,
                                                rocblas_stride stP,
                                                float* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                float* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_sgesv_outofplace_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B, ldb, stB, X, ldx, stX, info, bc)
        : rocsolver_sgesv_outofplace(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                double* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_int* ipiv,
                                                rocblas_stride stP,
                                                double* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                double* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_dgesv_outofplace_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B, ldb, stB, X, ldx, stX, info, bc)
        : rocsolver_dgesv_outofplace(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_float_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_int* ipiv,
                                                rocblas_stride stP,
                                                rocblas_float_complex* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_float_complex* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_cgesv_outofplace_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B, ldb, stB, X, ldx, stX, info, bc)
        : rocsolver_cgesv_outofplace(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_double_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_int* ipiv,
                                                rocblas_stride stP,
                                                rocblas_double_complex* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_double_complex* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_zgesv_outofplace_strided_batched(handle, n, nrhs, A, lda, stA, ipiv, stP, B, ldb, stB, X, ldx, stX, info, bc)
        : rocsolver_zgesv_outofplace(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

// batched
inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                float* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_int* ipiv,
                                                rocblas_stride stP,
                                                float* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                float* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_sgesv_outofplace_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                double* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_int* ipiv,
                                                rocblas_stride stP,
                                                double* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                double* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_dgesv_outofplace_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_float_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_int* ipiv,
                                                rocblas_stride stP,
                                                rocblas_float_complex* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_float_complex* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_cgesv_outofplace_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gesv_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_double_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_int* ipiv,
                                                rocblas_stride stP,
                                                rocblas_double_complex* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_double_complex* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_zgesv_outofplace_batched(handle, n, nrhs, A, lda, ipiv, stP, B, ldb, info, bc);
}
/********************************************************/

/******************** GETRI_OUTOFPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 float* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 float* C,
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetri_outofplace_strided_batched(handle, n, A, lda, stA, ipiv, stP,
                                                                 C, ldc, stC, info, bc)
                   : rocsolver_sgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, info);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 double* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 double* C,
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetri_outofplace_strided_batched(handle, n, A, lda, stA, ipiv, stP,
                                                                 C, ldc, stC, info, bc)
                   : rocsolver_dgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, info);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 rocblas_float_complex* C,
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetri_outofplace_strided_batched(handle, n, A, lda, stA, ipiv, stP,
                                                                 C, ldc, stC, info, bc)
                   : rocsolver_cgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, info);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 rocblas_double_complex* C,
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetri_outofplace_strided_batched(handle, n, A, lda, stA, ipiv, stP,
                                                                 C, ldc, stC, info, bc)
                   : rocsolver_zgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, info);
}

// batched
inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 float* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 float* const C[],
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return rocsolver_sgetri_outofplace_batched(handle, n, A, lda, ipiv, stP, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 double* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 double* const C[],
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return rocsolver_dgetri_outofplace_batched(handle, n, A, lda, ipiv, stP, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_float_complex* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 rocblas_float_complex* const C[],
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return rocsolver_cgetri_outofplace_batched(handle, n, A, lda, ipiv, stP, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_outofplace(bool STRIDED,
                                                 rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_double_complex* const A[],
                                                 rocblas_int lda,
                                                 rocblas_stride stA,
                                                 rocblas_int* ipiv,
                                                 rocblas_stride stP,
                                                 rocblas_double_complex* const C[],
                                                 rocblas_int ldc,
                                                 rocblas_stride stC,
                                                 rocblas_int* info,
                                                 rocblas_int bc)
{
    return rocsolver_zgetri_outofplace_batched(handle, n, A, lda, ipiv, stP, C, ldc, info, bc);
}
/********************************************************/

/******************** GETRI_NPVT_OUTOFPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      float* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float* C,
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetri_npvt_outofplace_strided_batched(handle, n, A, lda, stA, C,
                                                                      ldc, stC, info, bc)
                   : rocsolver_sgetri_npvt_outofplace(handle, n, A, lda, C, ldc, info);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      double* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double* C,
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetri_npvt_outofplace_strided_batched(handle, n, A, lda, stA, C,
                                                                      ldc, stC, info, bc)
                   : rocsolver_dgetri_npvt_outofplace(handle, n, A, lda, C, ldc, info);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      rocblas_float_complex* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_float_complex* C,
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetri_npvt_outofplace_strided_batched(handle, n, A, lda, stA, C,
                                                                      ldc, stC, info, bc)
                   : rocsolver_cgetri_npvt_outofplace(handle, n, A, lda, C, ldc, info);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      rocblas_double_complex* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_double_complex* C,
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetri_npvt_outofplace_strided_batched(handle, n, A, lda, stA, C,
                                                                      ldc, stC, info, bc)
                   : rocsolver_zgetri_npvt_outofplace(handle, n, A, lda, C, ldc, info);
}

// batched
inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      float* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float* const C[],
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocsolver_sgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      double* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double* const C[],
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocsolver_dgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      rocblas_float_complex* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_float_complex* const C[],
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocsolver_cgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, info, bc);
}

inline rocblas_status rocsolver_getri_npvt_outofplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_int n,
                                                      rocblas_double_complex* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_double_complex* const C[],
                                                      rocblas_int ldc,
                                                      rocblas_stride stC,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocsolver_zgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, info, bc);
}
/********************************************************/

/******************** GETRI ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetri_strided_batched(handle, n, A, lda, stA, ipiv, stP, info, bc)
                   : rocsolver_sgetri(handle, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetri_strided_batched(handle, n, A, lda, stA, ipiv, stP, info, bc)
                   : rocsolver_dgetri(handle, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetri_strided_batched(handle, n, A, lda, stA, ipiv, stP, info, bc)
                   : rocsolver_cgetri(handle, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetri_strided_batched(handle, n, A, lda, stA, ipiv, stP, info, bc)
                   : rocsolver_zgetri(handle, n, A, lda, ipiv, info);
}

// batched
inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_sgetri_batched(handle, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_dgetri_batched(handle, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_cgetri_batched(handle, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_getri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* ipiv,
                                      rocblas_stride stP,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_zgetri_batched(handle, n, A, lda, ipiv, stP, info, bc);
}
/********************************************************/

/******************** GETRI_NPVT ********************/
// normal and strided_batched
inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           float* A,
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return STRIDED ? rocsolver_sgetri_npvt_strided_batched(handle, n, A, lda, stA, info, bc)
                   : rocsolver_sgetri_npvt(handle, n, A, lda, info);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           double* A,
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return STRIDED ? rocsolver_dgetri_npvt_strided_batched(handle, n, A, lda, stA, info, bc)
                   : rocsolver_dgetri_npvt(handle, n, A, lda, info);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           rocblas_float_complex* A,
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return STRIDED ? rocsolver_cgetri_npvt_strided_batched(handle, n, A, lda, stA, info, bc)
                   : rocsolver_cgetri_npvt(handle, n, A, lda, info);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           rocblas_double_complex* A,
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return STRIDED ? rocsolver_zgetri_npvt_strided_batched(handle, n, A, lda, stA, info, bc)
                   : rocsolver_zgetri_npvt(handle, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           float* const A[],
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return rocsolver_sgetri_npvt_batched(handle, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           double* const A[],
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return rocsolver_dgetri_npvt_batched(handle, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           rocblas_float_complex* const A[],
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return rocsolver_cgetri_npvt_batched(handle, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_getri_npvt(bool STRIDED,
                                           rocblas_handle handle,
                                           rocblas_int n,
                                           rocblas_double_complex* const A[],
                                           rocblas_int lda,
                                           rocblas_stride stA,
                                           rocblas_int* info,
                                           rocblas_int bc)
{
    return rocsolver_zgetri_npvt_batched(handle, n, A, lda, info, bc);
}
/********************************************************/

/******************** TRTRI ********************/
// normal and strided_batched
inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      float* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_strtri_strided_batched(handle, uplo, diag, n, A, lda, stA, info, bc)
                   : rocsolver_strtri(handle, uplo, diag, n, A, lda, info);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      double* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_dtrtri_strided_batched(handle, uplo, diag, n, A, lda, stA, info, bc)
                   : rocsolver_dtrtri(handle, uplo, diag, n, A, lda, info);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_ctrtri_strided_batched(handle, uplo, diag, n, A, lda, stA, info, bc)
                   : rocsolver_ctrtri(handle, uplo, diag, n, A, lda, info);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return STRIDED ? rocsolver_ztrtri_strided_batched(handle, uplo, diag, n, A, lda, stA, info, bc)
                   : rocsolver_ztrtri(handle, uplo, diag, n, A, lda, info);
}

// batched
inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      float* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_strtri_batched(handle, uplo, diag, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      double* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_dtrtri_batched(handle, uplo, diag, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_ctrtri_batched(handle, uplo, diag, n, A, lda, info, bc);
}

inline rocblas_status rocsolver_trtri(bool STRIDED,
                                      rocblas_handle handle,
                                      rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_stride stA,
                                      rocblas_int* info,
                                      rocblas_int bc)
{
    return rocsolver_ztrtri_batched(handle, uplo, diag, n, A, lda, info, bc);
}
/********************************************************/

/******************** GEQR2_GEQRF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_sgeqrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgeqr2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_sgeqrf(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgeqr2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_dgeqrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgeqr2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_dgeqrf(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgeqr2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_cgeqrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgeqr2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_cgeqrf(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgeqr2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_zgeqrf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgeqr2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_zgeqrf(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgeqr2(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_sgeqrf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgeqr2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_dgeqrf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgeqr2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_cgeqrf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgeqr2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_zgeqrf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgeqr2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

// ptr_batched
inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* const ipiv[],
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_sgeqrf_ptr_batched(handle, m, n, A, lda, ipiv, bc)
                 : rocblas_status_not_implemented;
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* const ipiv[],
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_dgeqrf_ptr_batched(handle, m, n, A, lda, ipiv, bc)
                 : rocblas_status_not_implemented;
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* const ipiv[],
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_cgeqrf_ptr_batched(handle, m, n, A, lda, ipiv, bc)
                 : rocblas_status_not_implemented;
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* const ipiv[],
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQRF ? rocsolver_zgeqrf_ptr_batched(handle, m, n, A, lda, ipiv, bc)
                 : rocblas_status_not_implemented;
}

// normal and strided_batched
inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            float* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            int64_t bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_sgeqrf_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgeqr2_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_sgeqrf_64(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgeqr2_64(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            double* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            int64_t bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_dgeqrf_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgeqr2_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_dgeqrf_64(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgeqr2_64(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            rocblas_float_complex* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            int64_t bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_cgeqrf_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgeqr2_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_cgeqrf_64(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgeqr2_64(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            rocblas_double_complex* A,
                                            int64_t lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            int64_t bc)
{
    if(STRIDED)
        return GEQRF ? rocsolver_zgeqrf_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgeqr2_strided_batched_64(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQRF ? rocsolver_zgeqrf_64(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgeqr2_64(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            float* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            int64_t bc)
{
    return GEQRF ? rocsolver_sgeqrf_batched_64(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgeqr2_batched_64(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            double* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            int64_t bc)
{
    return GEQRF ? rocsolver_dgeqrf_batched_64(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgeqr2_batched_64(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            rocblas_float_complex* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            int64_t bc)
{
    return GEQRF ? rocsolver_cgeqrf_batched_64(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgeqr2_batched_64(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geqr2_geqrf(bool STRIDED,
                                            bool GEQRF,
                                            rocblas_handle handle,
                                            int64_t m,
                                            int64_t n,
                                            rocblas_double_complex* const A[],
                                            int64_t lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            int64_t bc)
{
    return GEQRF ? rocsolver_zgeqrf_batched_64(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgeqr2_batched_64(handle, m, n, A, lda, ipiv, stP, bc);
}
/********************************************************/

/******************** GERQ2_GERQF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GERQF ? rocsolver_sgerqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgerq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GERQF ? rocsolver_sgerqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgerq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GERQF ? rocsolver_dgerqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgerq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GERQF ? rocsolver_dgerqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgerq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GERQF ? rocsolver_cgerqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgerq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GERQF ? rocsolver_cgerqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgerq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GERQF ? rocsolver_zgerqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgerq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GERQF ? rocsolver_zgerqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgerq2(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GERQF ? rocsolver_sgerqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgerq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GERQF ? rocsolver_dgerqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgerq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GERQF ? rocsolver_cgerqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgerq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gerq2_gerqf(bool STRIDED,
                                            bool GERQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GERQF ? rocsolver_zgerqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgerq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}
/********************************************************/

/******************** GEQL2_GEQLF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQLF ? rocsolver_sgeqlf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgeql2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQLF ? rocsolver_sgeqlf(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgeql2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQLF ? rocsolver_dgeqlf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgeql2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQLF ? rocsolver_dgeqlf(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgeql2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQLF ? rocsolver_cgeqlf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgeql2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQLF ? rocsolver_cgeqlf(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgeql2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEQLF ? rocsolver_zgeqlf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgeql2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GEQLF ? rocsolver_zgeqlf(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgeql2(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQLF ? rocsolver_sgeqlf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgeql2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQLF ? rocsolver_dgeqlf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgeql2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQLF ? rocsolver_cgeqlf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgeql2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_geql2_geqlf(bool STRIDED,
                                            bool GEQLF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEQLF ? rocsolver_zgeqlf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgeql2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}
/********************************************************/

/******************** GELQ2_GELQF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GELQF ? rocsolver_sgelqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_sgelq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GELQF ? rocsolver_sgelqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_sgelq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GELQF ? rocsolver_dgelqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_dgelq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GELQF ? rocsolver_dgelqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_dgelq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GELQF ? rocsolver_cgelqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_cgelq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GELQF ? rocsolver_cgelqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_cgelq2(handle, m, n, A, lda, ipiv);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GELQF ? rocsolver_zgelqf_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc)
                     : rocsolver_zgelq2_strided_batched(handle, m, n, A, lda, stA, ipiv, stP, bc);
    else
        return GELQF ? rocsolver_zgelqf(handle, m, n, A, lda, ipiv)
                     : rocsolver_zgelq2(handle, m, n, A, lda, ipiv);
}

// batched
inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GELQF ? rocsolver_sgelqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_sgelq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GELQF ? rocsolver_dgelqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_dgelq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GELQF ? rocsolver_cgelqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_cgelq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}

inline rocblas_status rocsolver_gelq2_gelqf(bool STRIDED,
                                            bool GELQF,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GELQF ? rocsolver_zgelqf_batched(handle, m, n, A, lda, ipiv, stP, bc)
                 : rocsolver_zgelq2_batched(handle, m, n, A, lda, ipiv, stP, bc);
}
/********************************************************/

/******************** GELS ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     float* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_sgels_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, bc);
    else
        return rocsolver_sgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     double* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_dgels_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, bc);
    else
        return rocsolver_dgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_cgels_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, bc);
    else
        return rocsolver_cgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* A,
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_double_complex* B,
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_zgels_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB,
                                               info, bc);
    else
        return rocsolver_zgels(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

// batched
inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     float* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     float* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_sgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     double* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     double* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_dgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_float_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_cgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, info, bc);
}

inline rocblas_status rocsolver_gels(bool STRIDED,
                                     rocblas_handle handle,
                                     rocblas_operation trans,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_double_complex* const A[],
                                     rocblas_int lda,
                                     rocblas_stride stA,
                                     rocblas_double_complex* const B[],
                                     rocblas_int ldb,
                                     rocblas_stride stB,
                                     rocblas_int* info,
                                     rocblas_int bc)
{
    return rocsolver_zgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, info, bc);
}
/********************************************************/

/******************** GELS_OUTOFPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                float* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                float* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    if(STRIDED)
        return rocblas_status_not_implemented; // rocsolver_sgels_outofplace_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB, X, ldx, stX, info, bc);
    else
        return rocsolver_sgels_outofplace(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                double* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                double* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    if(STRIDED)
        return rocblas_status_not_implemented; // rocsolver_dgels_outofplace_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB, X, ldx, stX, info, bc);
    else
        return rocsolver_dgels_outofplace(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_float_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_float_complex* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_float_complex* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    if(STRIDED)
        return rocblas_status_not_implemented; // rocsolver_cgels_outofplace_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB, X, ldx, stX, info, bc);
    else
        return rocsolver_cgels_outofplace(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_double_complex* A,
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_double_complex* B,
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_double_complex* X,
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    if(STRIDED)
        return rocblas_status_not_implemented; // rocsolver_zgels_outofplace_strided_batched(handle, trans, m, n, nrhs, A, lda, stA, B, ldb, stB, X, ldx, stX, info, bc);
    else
        return rocsolver_zgels_outofplace(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

// batched
inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                float* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                float* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                float* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_sgels_outofplace_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, bc);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                double* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                double* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                double* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_dgels_outofplace_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, bc);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_float_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_float_complex* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_float_complex* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_cgels_outofplace_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, bc);
}

inline rocblas_status rocsolver_gels_outofplace(bool STRIDED,
                                                rocblas_handle handle,
                                                rocblas_operation trans,
                                                rocblas_int m,
                                                rocblas_int n,
                                                rocblas_int nrhs,
                                                rocblas_double_complex* const A[],
                                                rocblas_int lda,
                                                rocblas_stride stA,
                                                rocblas_double_complex* const B[],
                                                rocblas_int ldb,
                                                rocblas_stride stB,
                                                rocblas_double_complex* const X[],
                                                rocblas_int ldx,
                                                rocblas_stride stX,
                                                rocblas_int* info,
                                                rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_zgels_outofplace_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info, bc);
}
/********************************************************/

/******************** GEBD2_GEBRD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            float* tauq,
                                            rocblas_stride stQ,
                                            float* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEBRD ? rocsolver_sgebrd_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc)
                     : rocsolver_sgebd2_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc);
    else
        return GEBRD ? rocsolver_sgebrd(handle, m, n, A, lda, D, E, tauq, taup)
                     : rocsolver_sgebd2(handle, m, n, A, lda, D, E, tauq, taup);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            double* tauq,
                                            rocblas_stride stQ,
                                            double* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEBRD ? rocsolver_dgebrd_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc)
                     : rocsolver_dgebd2_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc);
    else
        return GEBRD ? rocsolver_dgebrd(handle, m, n, A, lda, D, E, tauq, taup)
                     : rocsolver_dgebd2(handle, m, n, A, lda, D, E, tauq, taup);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_float_complex* tauq,
                                            rocblas_stride stQ,
                                            rocblas_float_complex* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEBRD ? rocsolver_cgebrd_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc)
                     : rocsolver_cgebd2_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc);
    else
        return GEBRD ? rocsolver_cgebrd(handle, m, n, A, lda, D, E, tauq, taup)
                     : rocsolver_cgebd2(handle, m, n, A, lda, D, E, tauq, taup);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_double_complex* tauq,
                                            rocblas_stride stQ,
                                            rocblas_double_complex* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return GEBRD ? rocsolver_zgebrd_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc)
                     : rocsolver_zgebd2_strided_batched(handle, m, n, A, lda, stA, D, stD, E, stE,
                                                        tauq, stQ, taup, stP, bc);
    else
        return GEBRD ? rocsolver_zgebrd(handle, m, n, A, lda, D, E, tauq, taup)
                     : rocsolver_zgebd2(handle, m, n, A, lda, D, E, tauq, taup);
}

// batched
inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            float* tauq,
                                            rocblas_stride stQ,
                                            float* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEBRD
        ? rocsolver_sgebrd_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc)
        : rocsolver_sgebd2_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            double* tauq,
                                            rocblas_stride stQ,
                                            double* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEBRD
        ? rocsolver_dgebrd_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc)
        : rocsolver_dgebd2_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_float_complex* tauq,
                                            rocblas_stride stQ,
                                            rocblas_float_complex* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEBRD
        ? rocsolver_cgebrd_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc)
        : rocsolver_cgebd2_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc);
}

inline rocblas_status rocsolver_gebd2_gebrd(bool STRIDED,
                                            bool GEBRD,
                                            rocblas_handle handle,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_double_complex* tauq,
                                            rocblas_stride stQ,
                                            rocblas_double_complex* taup,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return GEBRD
        ? rocsolver_zgebrd_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc)
        : rocsolver_zgebd2_batched(handle, m, n, A, lda, D, stD, E, stE, tauq, stQ, taup, stP, bc);
}
/********************************************************/

/******************** SYTD2/SYTRD_HETD2/HETRD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            float* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRD ? rocsolver_ssytrd_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc)
                     : rocsolver_ssytd2_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc);
    else
        return SYTRD ? rocsolver_ssytrd(handle, uplo, n, A, lda, D, E, tau)
                     : rocsolver_ssytd2(handle, uplo, n, A, lda, D, E, tau);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            double* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRD ? rocsolver_dsytrd_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc)
                     : rocsolver_dsytd2_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc);
    else
        return SYTRD ? rocsolver_dsytrd(handle, uplo, n, A, lda, D, E, tau)
                     : rocsolver_dsytd2(handle, uplo, n, A, lda, D, E, tau);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_float_complex* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRD ? rocsolver_chetrd_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc)
                     : rocsolver_chetd2_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc);
    else
        return SYTRD ? rocsolver_chetrd(handle, uplo, n, A, lda, D, E, tau)
                     : rocsolver_chetd2(handle, uplo, n, A, lda, D, E, tau);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_double_complex* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRD ? rocsolver_zhetrd_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc)
                     : rocsolver_zhetd2_strided_batched(handle, uplo, n, A, lda, stA, D, stD, E,
                                                        stE, tau, stP, bc);
    else
        return SYTRD ? rocsolver_zhetrd(handle, uplo, n, A, lda, D, E, tau)
                     : rocsolver_zhetd2(handle, uplo, n, A, lda, D, E, tau);
}

// batched
inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            float* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return SYTRD ? rocsolver_ssytrd_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc)
                 : rocsolver_ssytd2_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            double* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return SYTRD ? rocsolver_dsytrd_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc)
                 : rocsolver_dsytd2_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_float_complex* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return SYTRD ? rocsolver_chetrd_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc)
                 : rocsolver_chetd2_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc);
}

inline rocblas_status rocsolver_sytxx_hetxx(bool STRIDED,
                                            bool SYTRD,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_double_complex* tau,
                                            rocblas_stride stP,
                                            rocblas_int bc)
{
    return SYTRD ? rocsolver_zhetrd_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc)
                 : rocsolver_zhetd2_batched(handle, uplo, n, A, lda, D, stD, E, stE, tau, stP, bc);
}
/********************************************************/

/******************** SYGS2/SYGST_HEGS2/HEGST ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYGST
            ? rocsolver_ssygst_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc)
            : rocsolver_ssygs2_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc);
    else
        return SYGST ? rocsolver_ssygst(handle, itype, uplo, n, A, lda, B, ldb)
                     : rocsolver_ssygs2(handle, itype, uplo, n, A, lda, B, ldb);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYGST
            ? rocsolver_dsygst_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc)
            : rocsolver_dsygs2_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc);
    else
        return SYGST ? rocsolver_dsygst(handle, itype, uplo, n, A, lda, B, ldb)
                     : rocsolver_dsygs2(handle, itype, uplo, n, A, lda, B, ldb);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYGST
            ? rocsolver_chegst_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc)
            : rocsolver_chegs2_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc);
    else
        return SYGST ? rocsolver_chegst(handle, itype, uplo, n, A, lda, B, ldb)
                     : rocsolver_chegs2(handle, itype, uplo, n, A, lda, B, ldb);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYGST
            ? rocsolver_zhegst_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc)
            : rocsolver_zhegs2_strided_batched(handle, itype, uplo, n, A, lda, stA, B, ldb, stB, bc);
    else
        return SYGST ? rocsolver_zhegst(handle, itype, uplo, n, A, lda, B, ldb)
                     : rocsolver_zhegs2(handle, itype, uplo, n, A, lda, B, ldb);
}

// batched
inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    return SYGST ? rocsolver_ssygst_batched(handle, itype, uplo, n, A, lda, B, ldb, bc)
                 : rocsolver_ssygs2_batched(handle, itype, uplo, n, A, lda, B, ldb, bc);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    return SYGST ? rocsolver_dsygst_batched(handle, itype, uplo, n, A, lda, B, ldb, bc)
                 : rocsolver_dsygs2_batched(handle, itype, uplo, n, A, lda, B, ldb, bc);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    return SYGST ? rocsolver_chegst_batched(handle, itype, uplo, n, A, lda, B, ldb, bc)
                 : rocsolver_chegs2_batched(handle, itype, uplo, n, A, lda, B, ldb, bc);
}

inline rocblas_status rocsolver_sygsx_hegsx(bool STRIDED,
                                            bool SYGST,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            rocblas_int bc)
{
    return SYGST ? rocsolver_zhegst_batched(handle, itype, uplo, n, A, lda, B, ldb, bc)
                 : rocsolver_zhegs2_batched(handle, itype, uplo, n, A, lda, B, ldb, bc);
}
/********************************************************/

/******************** SYEV/HEEV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          float* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return STRIDED ? rocsolver_ssyev_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, E,
                                                     stE, info, bc)
                   : rocsolver_ssyev(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          double* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return STRIDED ? rocsolver_dsyev_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, E,
                                                     stE, info, bc)
                   : rocsolver_dsyev(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return STRIDED ? rocsolver_cheev_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, E,
                                                     stE, info, bc)
                   : rocsolver_cheev(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_double_complex* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return STRIDED ? rocsolver_zheev_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, E,
                                                     stE, info, bc)
                   : rocsolver_zheev(handle, evect, uplo, n, A, lda, D, E, info);
}

// batched
inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          float* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_ssyev_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          double* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_dsyev_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_cheev_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syev_heev(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_double_complex* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_zheev_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}
/********************************************************/

/******************** SYEVD/HEEVD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_ssyevd_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD,
                                                      E, stE, info, bc)
                   : rocsolver_ssyevd(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_dsyevd_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD,
                                                      E, stE, info, bc)
                   : rocsolver_dsyevd(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_cheevd_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD,
                                                      E, stE, info, bc)
                   : rocsolver_cheevd(handle, evect, uplo, n, A, lda, D, E, info);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_zheevd_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD,
                                                      E, stE, info, bc)
                   : rocsolver_zheevd(handle, evect, uplo, n, A, lda, D, E, info);
}

// batched
inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_ssyevd_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_dsyevd_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_cheevd_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}

inline rocblas_status rocsolver_syevd_heevd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_zheevd_batched(handle, evect, uplo, n, A, lda, D, stD, E, stE, info, bc);
}
/********************************************************/

/******************** SYEVDJ/HEEVDJ ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syevdj_heevdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              float* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_ssyevdj_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, info, bc)
        : rocsolver_ssyevdj(handle, evect, uplo, n, A, lda, D, info);
}

inline rocblas_status rocsolver_syevdj_heevdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              double* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dsyevdj_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, info, bc)
        : rocsolver_dsyevdj(handle, evect, uplo, n, A, lda, D, info);
}

inline rocblas_status rocsolver_syevdj_heevdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cheevdj_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, info, bc)
        : rocsolver_cheevdj(handle, evect, uplo, n, A, lda, D, info);
}

inline rocblas_status rocsolver_syevdj_heevdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_double_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zheevdj_strided_batched(handle, evect, uplo, n, A, lda, stA, D, stD, info, bc)
        : rocsolver_zheevdj(handle, evect, uplo, n, A, lda, D, info);
}

// batched
inline rocblas_status rocsolver_syevdj_heevdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              float* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_ssyevdj_batched(handle, evect, uplo, n, A, lda, D, stD, info, bc);
}

inline rocblas_status rocsolver_syevdj_heevdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              double* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_dsyevdj_batched(handle, evect, uplo, n, A, lda, D, stD, info, bc);
}

inline rocblas_status rocsolver_syevdj_heevdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_cheevdj_batched(handle, evect, uplo, n, A, lda, D, stD, info, bc);
}

inline rocblas_status rocsolver_syevdj_heevdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_double_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_zheevdj_batched(handle, evect, uplo, n, A, lda, D, stD, info, bc);
}
/********************************************************/

/******************** SYEVDX/HEEVDX ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syevdx_heevdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              float* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float vl,
                                              float vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              float* W,
                                              rocblas_stride stW,
                                              float* Z,
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_ssyevdx_strided_batched(handle, evect, erange, uplo, n, A, lda, stA, vl, vu, il,
                                            iu, nev, W, stW, Z, ldz, stZ, info, bc)
        : rocsolver_ssyevdx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, ldz,
                            info);
}

inline rocblas_status rocsolver_syevdx_heevdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              double* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double vl,
                                              double vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              double* W,
                                              rocblas_stride stW,
                                              double* Z,
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dsyevdx_strided_batched(handle, evect, erange, uplo, n, A, lda, stA, vl, vu, il,
                                            iu, nev, W, stW, Z, ldz, stZ, info, bc)
        : rocsolver_dsyevdx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, ldz,
                            info);
}

inline rocblas_status rocsolver_syevdx_heevdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float vl,
                                              float vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              float* W,
                                              rocblas_stride stW,
                                              rocblas_float_complex* Z,
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cheevdx_strided_batched(handle, evect, erange, uplo, n, A, lda, stA, vl, vu, il,
                                            iu, nev, W, stW, Z, ldz, stZ, info, bc)
        : rocsolver_cheevdx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, ldz,
                            info);
}

inline rocblas_status rocsolver_syevdx_heevdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_double_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double vl,
                                              double vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              double* W,
                                              rocblas_stride stW,
                                              rocblas_double_complex* Z,
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zheevdx_strided_batched(handle, evect, erange, uplo, n, A, lda, stA, vl, vu, il,
                                            iu, nev, W, stW, Z, ldz, stZ, info, bc)
        : rocsolver_zheevdx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, ldz,
                            info);
}

// batched
inline rocblas_status rocsolver_syevdx_heevdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              float* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float vl,
                                              float vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              float* W,
                                              rocblas_stride stW,
                                              float* const Z[],
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_ssyevdx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W,
                                     stW, Z, ldz, info, bc);
}

inline rocblas_status rocsolver_syevdx_heevdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              double* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double vl,
                                              double vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              double* W,
                                              rocblas_stride stW,
                                              double* const Z[],
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_dsyevdx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W,
                                     stW, Z, ldz, info, bc);
}

inline rocblas_status rocsolver_syevdx_heevdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float vl,
                                              float vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              float* W,
                                              rocblas_stride stW,
                                              rocblas_float_complex* const Z[],
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_cheevdx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W,
                                     stW, Z, ldz, info, bc);
}

inline rocblas_status rocsolver_syevdx_heevdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_double_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double vl,
                                              double vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              double* W,
                                              rocblas_stride stW,
                                              rocblas_double_complex* const Z[],
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_zheevdx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W,
                                     stW, Z, ldz, info, bc);
}
/********************************************************/

/******************** SYEVJ/HEEVJ ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syevj_heevj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_esort esort,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float abstol,
                                            float* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED
        ? rocsolver_ssyevj_strided_batched(handle, esort, evect, uplo, n, A, lda, stA, abstol,
                                           residual, max_sweeps, n_sweeps, W, stW, info, bc)
        : rocsolver_ssyevj(handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps,
                           n_sweeps, W, info);
}

inline rocblas_status rocsolver_syevj_heevj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_esort esort,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double abstol,
                                            double* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dsyevj_strided_batched(handle, esort, evect, uplo, n, A, lda, stA, abstol,
                                           residual, max_sweeps, n_sweeps, W, stW, info, bc)
        : rocsolver_dsyevj(handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps,
                           n_sweeps, W, info);
}

inline rocblas_status rocsolver_syevj_heevj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_esort esort,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float abstol,
                                            float* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cheevj_strided_batched(handle, esort, evect, uplo, n, A, lda, stA, abstol,
                                           residual, max_sweeps, n_sweeps, W, stW, info, bc)
        : rocsolver_cheevj(handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps,
                           n_sweeps, W, info);
}

inline rocblas_status rocsolver_syevj_heevj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_esort esort,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double abstol,
                                            double* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zheevj_strided_batched(handle, esort, evect, uplo, n, A, lda, stA, abstol,
                                           residual, max_sweeps, n_sweeps, W, stW, info, bc)
        : rocsolver_zheevj(handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps,
                           n_sweeps, W, info);
}

// batched
inline rocblas_status rocsolver_syevj_heevj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_esort esort,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float abstol,
                                            float* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_ssyevj_batched(handle, esort, evect, uplo, n, A, lda, abstol, residual,
                                    max_sweeps, n_sweeps, W, stW, info, bc);
}

inline rocblas_status rocsolver_syevj_heevj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_esort esort,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double abstol,
                                            double* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_dsyevj_batched(handle, esort, evect, uplo, n, A, lda, abstol, residual,
                                    max_sweeps, n_sweeps, W, stW, info, bc);
}

inline rocblas_status rocsolver_syevj_heevj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_esort esort,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float abstol,
                                            float* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_cheevj_batched(handle, esort, evect, uplo, n, A, lda, abstol, residual,
                                    max_sweeps, n_sweeps, W, stW, info, bc);
}

inline rocblas_status rocsolver_syevj_heevj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_esort esort,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double abstol,
                                            double* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_zheevj_batched(handle, esort, evect, uplo, n, A, lda, abstol, residual,
                                    max_sweeps, n_sweeps, W, stW, info, bc);
}
/********************************************************/

/******************** SYEVX/HEEVX ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syevx_heevx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float vl,
                                            float vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            float abstol,
                                            rocblas_int* nev,
                                            float* W,
                                            rocblas_stride stW,
                                            float* Z,
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_ssyevx_strided_batched(handle, evect, erange, uplo, n, A, lda, stA,
                                                      vl, vu, il, iu, abstol, nev, W, stW, Z, ldz,
                                                      stZ, ifail, stF, info, bc)
                   : rocsolver_ssyevx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu,
                                      abstol, nev, W, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_syevx_heevx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double vl,
                                            double vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            double* W,
                                            rocblas_stride stW,
                                            double* Z,
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_dsyevx_strided_batched(handle, evect, erange, uplo, n, A, lda, stA,
                                                      vl, vu, il, iu, abstol, nev, W, stW, Z, ldz,
                                                      stZ, ifail, stF, info, bc)
                   : rocsolver_dsyevx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu,
                                      abstol, nev, W, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_syevx_heevx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float vl,
                                            float vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            float abstol,
                                            rocblas_int* nev,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_float_complex* Z,
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_cheevx_strided_batched(handle, evect, erange, uplo, n, A, lda, stA,
                                                      vl, vu, il, iu, abstol, nev, W, stW, Z, ldz,
                                                      stZ, ifail, stF, info, bc)
                   : rocsolver_cheevx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu,
                                      abstol, nev, W, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_syevx_heevx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double vl,
                                            double vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_double_complex* Z,
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_zheevx_strided_batched(handle, evect, erange, uplo, n, A, lda, stA,
                                                      vl, vu, il, iu, abstol, nev, W, stW, Z, ldz,
                                                      stZ, ifail, stF, info, bc)
                   : rocsolver_zheevx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu,
                                      abstol, nev, W, Z, ldz, ifail, info);
}

// batched
inline rocblas_status rocsolver_syevx_heevx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float vl,
                                            float vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            float abstol,
                                            rocblas_int* nev,
                                            float* W,
                                            rocblas_stride stW,
                                            float* const Z[],
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_ssyevx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol,
                                    nev, W, stW, Z, ldz, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_syevx_heevx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double vl,
                                            double vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            double* W,
                                            rocblas_stride stW,
                                            double* const Z[],
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_dsyevx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol,
                                    nev, W, stW, Z, ldz, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_syevx_heevx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float vl,
                                            float vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            float abstol,
                                            rocblas_int* nev,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_float_complex* const Z[],
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_cheevx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol,
                                    nev, W, stW, Z, ldz, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_syevx_heevx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double vl,
                                            double vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_double_complex* const Z[],
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_zheevx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol,
                                    nev, W, stW, Z, ldz, ifail, stF, info, bc);
}
/********************************************************/

/******************** SYEVDX/HEEVDX_INPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_syevdx_heevdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      float* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float vl,
                                                      float vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      float abstol,
                                                      rocblas_int* nev,
                                                      float* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_ssyevdx_inplace_strided_batched(handle, evect, erange, uplo, n, A, lda, stA, vl, vu, il, iu, abstol, nev, W, stW, info, bc)
        : rocsolver_ssyevdx_inplace(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol,
                                    nev, W, info);
}

inline rocblas_status rocsolver_syevdx_heevdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      double* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double vl,
                                                      double vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      double* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_dsyevdx_inplace_strided_batched(handle, evect, erange, uplo, n, A, lda, stA, vl, vu, il, iu, abstol, nev, W, stW, info, bc)
        : rocsolver_dsyevdx_inplace(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol,
                                    nev, W, info);
}

inline rocblas_status rocsolver_syevdx_heevdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      rocblas_float_complex* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float vl,
                                                      float vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      float abstol,
                                                      rocblas_int* nev,
                                                      float* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_cheevdx_inplace_strided_batched(handle, evect, erange, uplo, n, A, lda, stA, vl, vu, il, iu, abstol, nev, W, stW, info, bc)
        : rocsolver_cheevdx_inplace(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol,
                                    nev, W, info);
}

inline rocblas_status rocsolver_syevdx_heevdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      rocblas_double_complex* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double vl,
                                                      double vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      double* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_zheevdx_inplace_strided_batched(handle, evect, erange, uplo, n, A, lda, stA, vl, vu, il, iu, abstol, nev, W, stW, info, bc)
        : rocsolver_zheevdx_inplace(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol,
                                    nev, W, info);
}

// batched
inline rocblas_status rocsolver_syevdx_heevdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      float* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float vl,
                                                      float vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      float abstol,
                                                      rocblas_int* nev,
                                                      float* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_ssyevdx_inplace_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, nev, W, stW, info, bc);
}

inline rocblas_status rocsolver_syevdx_heevdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      double* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double vl,
                                                      double vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      double* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_dsyevdx_inplace_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, nev, W, stW, info, bc);
}

inline rocblas_status rocsolver_syevdx_heevdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      rocblas_float_complex* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float vl,
                                                      float vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      float abstol,
                                                      rocblas_int* nev,
                                                      float* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_cheevdx_inplace_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, nev, W, stW, info, bc);
}

inline rocblas_status rocsolver_syevdx_heevdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      rocblas_double_complex* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double vl,
                                                      double vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      double* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_zheevdx_inplace_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, nev, W, stW, info, bc);
}
/********************************************************/

/******************** SYGV_HEGV ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          float* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* B,
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_ssygv_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                               stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_ssygv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          double* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* B,
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_dsygv_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                               stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_dsygv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          rocblas_float_complex* B,
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_chegv_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                               stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_chegv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_double_complex* A,
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          rocblas_double_complex* B,
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_zhegv_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                               stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_zhegv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

// batched
inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          float* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          float* const B[],
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_ssygv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                   info, bc);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          double* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          double* const B[],
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_dsygv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                   info, bc);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          rocblas_float_complex* const B[],
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          float* D,
                                          rocblas_stride stD,
                                          float* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_chegv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                   info, bc);
}

inline rocblas_status rocsolver_sygv_hegv(bool STRIDED,
                                          rocblas_handle handle,
                                          rocblas_eform itype,
                                          rocblas_evect evect,
                                          rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_double_complex* const A[],
                                          rocblas_int lda,
                                          rocblas_stride stA,
                                          rocblas_double_complex* const B[],
                                          rocblas_int ldb,
                                          rocblas_stride stB,
                                          double* D,
                                          rocblas_stride stD,
                                          double* E,
                                          rocblas_stride stE,
                                          rocblas_int* info,
                                          rocblas_int bc)
{
    return rocsolver_zhegv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                   info, bc);
}
/********************************************************/

/******************** SYGVD_HEGVD ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_ssygvd_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_ssygvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_dsygvd_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_dsygvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_chegvd_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_chegvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_zhegvd_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, D, stD, E, stE, info, bc);
    else
        return rocsolver_zhegvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info);
}

// batched
inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_ssygvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                    info, bc);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_dsygvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                    info, bc);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float* D,
                                            rocblas_stride stD,
                                            float* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_chegvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                    info, bc);
}

inline rocblas_status rocsolver_sygvd_hegvd(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double* D,
                                            rocblas_stride stD,
                                            double* E,
                                            rocblas_stride stE,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_zhegvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, E, stE,
                                    info, bc);
}
/********************************************************/

/******************** SYGVDJ_HEGVDJ ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygvdj_hegvdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              float* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_ssygvdj_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                 stB, D, stD, info, bc);
    else
        return rocsolver_ssygvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, info);
}

inline rocblas_status rocsolver_sygvdj_hegvdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              double* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_dsygvdj_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                 stB, D, stD, info, bc);
    else
        return rocsolver_dsygvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, info);
}

inline rocblas_status rocsolver_sygvdj_hegvdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_float_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_chegvdj_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                 stB, D, stD, info, bc);
    else
        return rocsolver_chegvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, info);
}

inline rocblas_status rocsolver_sygvdj_hegvdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_double_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_double_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_zhegvdj_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                 stB, D, stD, info, bc);
    else
        return rocsolver_zhegvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, info);
}

// batched
inline rocblas_status rocsolver_sygvdj_hegvdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              float* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_ssygvdj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, info, bc);
}

inline rocblas_status rocsolver_sygvdj_hegvdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              double* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_dsygvdj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, info, bc);
}

inline rocblas_status rocsolver_sygvdj_hegvdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_float_complex* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_chegvdj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, info, bc);
}

inline rocblas_status rocsolver_sygvdj_hegvdj(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_double_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_double_complex* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double* D,
                                              rocblas_stride stD,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_zhegvdj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, stD, info, bc);
}
/********************************************************/

/******************** SYGVDX/HEGVDX ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygvdx_hegvdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              float* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float vl,
                                              float vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              float* W,
                                              rocblas_stride stW,
                                              float* Z,
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED ? rocsolver_ssygvdx_strided_batched(handle, itype, evect, erange, uplo, n, A,
                                                       lda, stA, B, ldb, stB, vl, vu, il, iu, nev,
                                                       W, stW, Z, ldz, stZ, info, bc)
                   : rocsolver_ssygvdx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl,
                                       vu, il, iu, nev, W, Z, ldz, info);
}

inline rocblas_status rocsolver_sygvdx_hegvdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              double* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double vl,
                                              double vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              double* W,
                                              rocblas_stride stW,
                                              double* Z,
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED ? rocsolver_dsygvdx_strided_batched(handle, itype, evect, erange, uplo, n, A,
                                                       lda, stA, B, ldb, stB, vl, vu, il, iu, nev,
                                                       W, stW, Z, ldz, stZ, info, bc)
                   : rocsolver_dsygvdx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl,
                                       vu, il, iu, nev, W, Z, ldz, info);
}

inline rocblas_status rocsolver_sygvdx_hegvdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_float_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float vl,
                                              float vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              float* W,
                                              rocblas_stride stW,
                                              rocblas_float_complex* Z,
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED ? rocsolver_chegvdx_strided_batched(handle, itype, evect, erange, uplo, n, A,
                                                       lda, stA, B, ldb, stB, vl, vu, il, iu, nev,
                                                       W, stW, Z, ldz, stZ, info, bc)
                   : rocsolver_chegvdx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl,
                                       vu, il, iu, nev, W, Z, ldz, info);
}

inline rocblas_status rocsolver_sygvdx_hegvdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_double_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_double_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double vl,
                                              double vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              double* W,
                                              rocblas_stride stW,
                                              rocblas_double_complex* Z,
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED ? rocsolver_zhegvdx_strided_batched(handle, itype, evect, erange, uplo, n, A,
                                                       lda, stA, B, ldb, stB, vl, vu, il, iu, nev,
                                                       W, stW, Z, ldz, stZ, info, bc)
                   : rocsolver_zhegvdx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl,
                                       vu, il, iu, nev, W, Z, ldz, info);
}

// batched
inline rocblas_status rocsolver_sygvdx_hegvdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              float* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float vl,
                                              float vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              float* W,
                                              rocblas_stride stW,
                                              float* const Z[],
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_ssygvdx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                     il, iu, nev, W, stW, Z, ldz, info, bc);
}

inline rocblas_status rocsolver_sygvdx_hegvdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              double* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double vl,
                                              double vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              double* W,
                                              rocblas_stride stW,
                                              double* const Z[],
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_dsygvdx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                     il, iu, nev, W, stW, Z, ldz, info, bc);
}

inline rocblas_status rocsolver_sygvdx_hegvdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_float_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_float_complex* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float vl,
                                              float vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              float* W,
                                              rocblas_stride stW,
                                              rocblas_float_complex* const Z[],
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_chegvdx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                     il, iu, nev, W, stW, Z, ldz, info, bc);
}

inline rocblas_status rocsolver_sygvdx_hegvdx(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_eform itype,
                                              rocblas_evect evect,
                                              rocblas_erange erange,
                                              rocblas_fill uplo,
                                              rocblas_int n,
                                              rocblas_double_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_double_complex* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double vl,
                                              double vu,
                                              rocblas_int il,
                                              rocblas_int iu,
                                              rocblas_int* nev,
                                              double* W,
                                              rocblas_stride stW,
                                              rocblas_double_complex* const Z[],
                                              rocblas_int ldz,
                                              rocblas_stride stZ,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_zhegvdx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                     il, iu, nev, W, stW, Z, ldz, info, bc);
}
/********************************************************/

/******************** SYGVJ_HEGVJ ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygvj_hegvj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float abstol,
                                            float* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_ssygvj_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, abstol, residual, max_sweeps, n_sweeps, W, stW,
                                                info, bc);
    else
        return rocsolver_ssygvj(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual,
                                max_sweeps, n_sweeps, W, info);
}

inline rocblas_status rocsolver_sygvj_hegvj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double abstol,
                                            double* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_dsygvj_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, abstol, residual, max_sweeps, n_sweeps, W, stW,
                                                info, bc);
    else
        return rocsolver_dsygvj(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual,
                                max_sweeps, n_sweeps, W, info);
}

inline rocblas_status rocsolver_sygvj_hegvj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float abstol,
                                            float* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_chegvj_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, abstol, residual, max_sweeps, n_sweeps, W, stW,
                                                info, bc);
    else
        return rocsolver_chegvj(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual,
                                max_sweeps, n_sweeps, W, info);
}

inline rocblas_status rocsolver_sygvj_hegvj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double abstol,
                                            double* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return rocsolver_zhegvj_strided_batched(handle, itype, evect, uplo, n, A, lda, stA, B, ldb,
                                                stB, abstol, residual, max_sweeps, n_sweeps, W, stW,
                                                info, bc);
    else
        return rocsolver_zhegvj(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual,
                                max_sweeps, n_sweeps, W, info);
}

// batched
inline rocblas_status rocsolver_sygvj_hegvj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float abstol,
                                            float* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_ssygvj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual,
                                    max_sweeps, n_sweeps, W, stW, info, bc);
}

inline rocblas_status rocsolver_sygvj_hegvj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double abstol,
                                            double* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_dsygvj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual,
                                    max_sweeps, n_sweeps, W, stW, info, bc);
}

inline rocblas_status rocsolver_sygvj_hegvj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float abstol,
                                            float* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_chegvj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual,
                                    max_sweeps, n_sweeps, W, stW, info, bc);
}

inline rocblas_status rocsolver_sygvj_hegvj(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double abstol,
                                            double* residual,
                                            rocblas_int max_sweeps,
                                            rocblas_int* n_sweeps,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_zhegvj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual,
                                    max_sweeps, n_sweeps, W, stW, info, bc);
}
/********************************************************/

/******************** SYGVX/HEGVX ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygvx_hegvx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float vl,
                                            float vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            float abstol,
                                            rocblas_int* nev,
                                            float* W,
                                            rocblas_stride stW,
                                            float* Z,
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_ssygvx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda,
                                                      stA, B, ldb, stB, vl, vu, il, iu, abstol, nev,
                                                      W, stW, Z, ldz, stZ, ifail, stF, info, bc)
                   : rocsolver_ssygvx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                      il, iu, abstol, nev, W, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_sygvx_hegvx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double vl,
                                            double vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            double* W,
                                            rocblas_stride stW,
                                            double* Z,
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_dsygvx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda,
                                                      stA, B, ldb, stB, vl, vu, il, iu, abstol, nev,
                                                      W, stW, Z, ldz, stZ, ifail, stF, info, bc)
                   : rocsolver_dsygvx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                      il, iu, abstol, nev, W, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_sygvx_hegvx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float vl,
                                            float vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            float abstol,
                                            rocblas_int* nev,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_float_complex* Z,
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_chegvx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda,
                                                      stA, B, ldb, stB, vl, vu, il, iu, abstol, nev,
                                                      W, stW, Z, ldz, stZ, ifail, stF, info, bc)
                   : rocsolver_chegvx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                      il, iu, abstol, nev, W, Z, ldz, ifail, info);
}

inline rocblas_status rocsolver_sygvx_hegvx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* B,
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double vl,
                                            double vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_double_complex* Z,
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return STRIDED ? rocsolver_zhegvx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda,
                                                      stA, B, ldb, stB, vl, vu, il, iu, abstol, nev,
                                                      W, stW, Z, ldz, stZ, ifail, stF, info, bc)
                   : rocsolver_zhegvx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                      il, iu, abstol, nev, W, Z, ldz, ifail, info);
}

// batched
inline rocblas_status rocsolver_sygvx_hegvx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            float* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float vl,
                                            float vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            float abstol,
                                            rocblas_int* nev,
                                            float* W,
                                            rocblas_stride stW,
                                            float* const Z[],
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_ssygvx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                    il, iu, abstol, nev, W, stW, Z, ldz, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_sygvx_hegvx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            double* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double vl,
                                            double vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            double* W,
                                            rocblas_stride stW,
                                            double* const Z[],
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_dsygvx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                    il, iu, abstol, nev, W, stW, Z, ldz, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_sygvx_hegvx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_float_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            float vl,
                                            float vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            float* W,
                                            rocblas_stride stW,
                                            rocblas_float_complex* const Z[],
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_chegvx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                    il, iu, abstol, nev, W, stW, Z, ldz, ifail, stF, info, bc);
}

inline rocblas_status rocsolver_sygvx_hegvx(bool STRIDED,
                                            rocblas_handle handle,
                                            rocblas_eform itype,
                                            rocblas_evect evect,
                                            rocblas_erange erange,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_double_complex* const B[],
                                            rocblas_int ldb,
                                            rocblas_stride stB,
                                            double vl,
                                            double vu,
                                            rocblas_int il,
                                            rocblas_int iu,
                                            double abstol,
                                            rocblas_int* nev,
                                            double* W,
                                            rocblas_stride stW,
                                            rocblas_double_complex* const Z[],
                                            rocblas_int ldz,
                                            rocblas_stride stZ,
                                            rocblas_int* ifail,
                                            rocblas_stride stF,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return rocsolver_zhegvx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                    il, iu, abstol, nev, W, stW, Z, ldz, ifail, stF, info, bc);
}
/********************************************************/

/******************** SYGVX/HEGVX_INPLACE ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sygvdx_hegvdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_eform itype,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      float* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float* B,
                                                      rocblas_int ldb,
                                                      rocblas_stride stB,
                                                      float vl,
                                                      float vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      float abstol,
                                                      rocblas_int* nev,
                                                      float* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_ssygvdx_inplace_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, stA, B, ldb, stB, vl, vu, il, iu, abstol, nev, W, stW, info, bc)
        : rocsolver_ssygvdx_inplace(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                    il, iu, abstol, nev, W, info);
}

inline rocblas_status rocsolver_sygvdx_hegvdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_eform itype,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      double* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double* B,
                                                      rocblas_int ldb,
                                                      rocblas_stride stB,
                                                      double vl,
                                                      double vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      double* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_dsygvdx_inplace_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, stA, B, ldb, stB, vl, vu, il, iu, abstol, nev, W, stW, info, bc)
        : rocsolver_dsygvdx_inplace(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                    il, iu, abstol, nev, W, info);
}

inline rocblas_status rocsolver_sygvdx_hegvdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_eform itype,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      rocblas_float_complex* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_float_complex* B,
                                                      rocblas_int ldb,
                                                      rocblas_stride stB,
                                                      float vl,
                                                      float vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      float abstol,
                                                      rocblas_int* nev,
                                                      float* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_chegvdx_inplace_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, stA, B, ldb, stB, vl, vu, il, iu, abstol, nev, W, stW, info, bc)
        : rocsolver_chegvdx_inplace(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                    il, iu, abstol, nev, W, info);
}

inline rocblas_status rocsolver_sygvdx_hegvdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_eform itype,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      rocblas_double_complex* A,
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_double_complex* B,
                                                      rocblas_int ldb,
                                                      rocblas_stride stB,
                                                      double vl,
                                                      double vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      double* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return STRIDED
        ? rocblas_status_not_implemented // rocsolver_zhegvdx_inplace_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, stA, B, ldb, stB, vl, vu, il, iu, abstol, nev, W, stW, info, bc)
        : rocsolver_zhegvdx_inplace(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu,
                                    il, iu, abstol, nev, W, info);
}

// batched
inline rocblas_status rocsolver_sygvdx_hegvdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_eform itype,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      float* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      float* const B[],
                                                      rocblas_int ldb,
                                                      rocblas_stride stB,
                                                      float vl,
                                                      float vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      float abstol,
                                                      rocblas_int* nev,
                                                      float* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_ssygvdx_inplace_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, nev, W, stW, info, bc);
}

inline rocblas_status rocsolver_sygvdx_hegvdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_eform itype,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      double* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      double* const B[],
                                                      rocblas_int ldb,
                                                      rocblas_stride stB,
                                                      double vl,
                                                      double vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      double* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_dsygvdx_inplace_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, nev, W, stW, info, bc);
}

inline rocblas_status rocsolver_sygvdx_hegvdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_eform itype,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      rocblas_float_complex* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_float_complex* const B[],
                                                      rocblas_int ldb,
                                                      rocblas_stride stB,
                                                      float vl,
                                                      float vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      float* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_chegvdx_inplace_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, nev, W, stW, info, bc);
}

inline rocblas_status rocsolver_sygvdx_hegvdx_inplace(bool STRIDED,
                                                      rocblas_handle handle,
                                                      rocblas_eform itype,
                                                      rocblas_evect evect,
                                                      rocblas_erange erange,
                                                      rocblas_fill uplo,
                                                      rocblas_int n,
                                                      rocblas_double_complex* const A[],
                                                      rocblas_int lda,
                                                      rocblas_stride stA,
                                                      rocblas_double_complex* const B[],
                                                      rocblas_int ldb,
                                                      rocblas_stride stB,
                                                      double vl,
                                                      double vu,
                                                      rocblas_int il,
                                                      rocblas_int iu,
                                                      double abstol,
                                                      rocblas_int* nev,
                                                      double* W,
                                                      rocblas_stride stW,
                                                      rocblas_int* info,
                                                      rocblas_int bc)
{
    return rocblas_status_not_implemented; // rocsolver_zhegvdx_inplace_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, nev, W, stW, info, bc);
}
/********************************************************/

/******************** SYTF2_SYTRF ********************/
// normal and strided_batched
inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRF
            ? rocsolver_ssytrf_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_ssytf2_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return SYTRF ? rocsolver_ssytrf(handle, uplo, n, A, lda, ipiv, info)
                     : rocsolver_ssytf2(handle, uplo, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRF
            ? rocsolver_dsytrf_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_dsytf2_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return SYTRF ? rocsolver_dsytrf(handle, uplo, n, A, lda, ipiv, info)
                     : rocsolver_dsytf2(handle, uplo, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRF
            ? rocsolver_csytrf_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_csytf2_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return SYTRF ? rocsolver_csytrf(handle, uplo, n, A, lda, ipiv, info)
                     : rocsolver_csytf2(handle, uplo, n, A, lda, ipiv, info);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* A,
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    if(STRIDED)
        return SYTRF
            ? rocsolver_zsytrf_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc)
            : rocsolver_zsytf2_strided_batched(handle, uplo, n, A, lda, stA, ipiv, stP, info, bc);
    else
        return SYTRF ? rocsolver_zsytrf(handle, uplo, n, A, lda, ipiv, info)
                     : rocsolver_zsytf2(handle, uplo, n, A, lda, ipiv, info);
}

// batched
inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            float* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return SYTRF ? rocsolver_ssytrf_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_ssytf2_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            double* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return SYTRF ? rocsolver_dsytrf_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_dsytf2_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return SYTRF ? rocsolver_csytrf_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_csytf2_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc);
}

inline rocblas_status rocsolver_sytf2_sytrf(bool STRIDED,
                                            bool SYTRF,
                                            rocblas_handle handle,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_double_complex* const A[],
                                            rocblas_int lda,
                                            rocblas_stride stA,
                                            rocblas_int* ipiv,
                                            rocblas_stride stP,
                                            rocblas_int* info,
                                            rocblas_int bc)
{
    return SYTRF ? rocsolver_zsytrf_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc)
                 : rocsolver_zsytf2_batched(handle, uplo, n, A, lda, ipiv, stP, info, bc);
}
/********************************************************/

/******************** GEBLTTRF_NPVT ********************/
// normal and strided_batched
inline rocblas_status rocsolver_geblttrf_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              float* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float* C,
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED ? rocsolver_sgeblttrf_npvt_strided_batched(handle, nb, nblocks, A, lda, stA, B,
                                                              ldb, stB, C, ldc, stC, info, bc)
                   : rocsolver_sgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info);
}

inline rocblas_status rocsolver_geblttrf_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              double* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double* C,
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED ? rocsolver_dgeblttrf_npvt_strided_batched(handle, nb, nblocks, A, lda, stA, B,
                                                              ldb, stB, C, ldc, stC, info, bc)
                   : rocsolver_dgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info);
}

inline rocblas_status rocsolver_geblttrf_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_float_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED ? rocsolver_cgeblttrf_npvt_strided_batched(handle, nb, nblocks, A, lda, stA, B,
                                                              ldb, stB, C, ldc, stC, info, bc)
                   : rocsolver_cgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info);
}

inline rocblas_status rocsolver_geblttrf_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_double_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_double_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              rocblas_double_complex* C,
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return STRIDED ? rocsolver_zgeblttrf_npvt_strided_batched(handle, nb, nblocks, A, lda, stA, B,
                                                              ldb, stB, C, ldc, stC, info, bc)
                   : rocsolver_zgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info);
}

// batched
inline rocblas_status rocsolver_geblttrf_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              float* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float* const C[],
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_sgeblttrf_npvt_batched(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info, bc);
}

inline rocblas_status rocsolver_geblttrf_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              double* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double* const C[],
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_dgeblttrf_npvt_batched(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info, bc);
}

inline rocblas_status rocsolver_geblttrf_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_float_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_float_complex* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              rocblas_float_complex* const C[],
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_cgeblttrf_npvt_batched(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info, bc);
}

inline rocblas_status rocsolver_geblttrf_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_double_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_double_complex* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              rocblas_double_complex* const C[],
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_int* info,
                                              rocblas_int bc)
{
    return rocsolver_zgeblttrf_npvt_batched(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info, bc);
}
/********************************************************/

/******************** GEBLTTRF_NPVT_INTERLEAVED ********************/
// interleaved_batched
inline rocblas_status rocsolver_geblttrf_npvt_interleaved(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          float* A,
                                                          rocblas_int inca,
                                                          rocblas_int lda,
                                                          rocblas_stride stA,
                                                          float* B,
                                                          rocblas_int incb,
                                                          rocblas_int ldb,
                                                          rocblas_stride stB,
                                                          float* C,
                                                          rocblas_int incc,
                                                          rocblas_int ldc,
                                                          rocblas_stride stC,
                                                          rocblas_int* info,
                                                          rocblas_int bc)
{
    return rocsolver_sgeblttrf_npvt_interleaved_batched(handle, nb, nblocks, A, inca, lda, stA, B,
                                                        incb, ldb, stB, C, incc, ldc, stC, info, bc);
}

inline rocblas_status rocsolver_geblttrf_npvt_interleaved(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          double* A,
                                                          rocblas_int inca,
                                                          rocblas_int lda,
                                                          rocblas_stride stA,
                                                          double* B,
                                                          rocblas_int incb,
                                                          rocblas_int ldb,
                                                          rocblas_stride stB,
                                                          double* C,
                                                          rocblas_int incc,
                                                          rocblas_int ldc,
                                                          rocblas_stride stC,
                                                          rocblas_int* info,
                                                          rocblas_int bc)
{
    return rocsolver_dgeblttrf_npvt_interleaved_batched(handle, nb, nblocks, A, inca, lda, stA, B,
                                                        incb, ldb, stB, C, incc, ldc, stC, info, bc);
}

inline rocblas_status rocsolver_geblttrf_npvt_interleaved(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          rocblas_float_complex* A,
                                                          rocblas_int inca,
                                                          rocblas_int lda,
                                                          rocblas_stride stA,
                                                          rocblas_float_complex* B,
                                                          rocblas_int incb,
                                                          rocblas_int ldb,
                                                          rocblas_stride stB,
                                                          rocblas_float_complex* C,
                                                          rocblas_int incc,
                                                          rocblas_int ldc,
                                                          rocblas_stride stC,
                                                          rocblas_int* info,
                                                          rocblas_int bc)
{
    return rocsolver_cgeblttrf_npvt_interleaved_batched(handle, nb, nblocks, A, inca, lda, stA, B,
                                                        incb, ldb, stB, C, incc, ldc, stC, info, bc);
}

inline rocblas_status rocsolver_geblttrf_npvt_interleaved(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          rocblas_double_complex* A,
                                                          rocblas_int inca,
                                                          rocblas_int lda,
                                                          rocblas_stride stA,
                                                          rocblas_double_complex* B,
                                                          rocblas_int incb,
                                                          rocblas_int ldb,
                                                          rocblas_stride stB,
                                                          rocblas_double_complex* C,
                                                          rocblas_int incc,
                                                          rocblas_int ldc,
                                                          rocblas_stride stC,
                                                          rocblas_int* info,
                                                          rocblas_int bc)
{
    return rocsolver_zgeblttrf_npvt_interleaved_batched(handle, nb, nblocks, A, inca, lda, stA, B,
                                                        incb, ldb, stB, C, incc, ldc, stC, info, bc);
}
/********************************************************/

/******************** GEBLTTRS_NPVT ********************/
// normal and strided_batched
inline rocblas_status rocsolver_geblttrs_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_int nrhs,
                                              float* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float* C,
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              float* X,
                                              rocblas_int ldx,
                                              rocblas_stride stX,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_sgeblttrs_npvt_strided_batched(handle, nb, nblocks, nrhs, A, lda, stA, B, ldb,
                                                   stB, C, ldc, stC, X, ldx, stX, bc)
        : rocsolver_sgeblttrs_npvt(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx);
}

inline rocblas_status rocsolver_geblttrs_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_int nrhs,
                                              double* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double* C,
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              double* X,
                                              rocblas_int ldx,
                                              rocblas_stride stX,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_dgeblttrs_npvt_strided_batched(handle, nb, nblocks, nrhs, A, lda, stA, B, ldb,
                                                   stB, C, ldc, stC, X, ldx, stX, bc)
        : rocsolver_dgeblttrs_npvt(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx);
}

inline rocblas_status rocsolver_geblttrs_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_int nrhs,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_float_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_float_complex* X,
                                              rocblas_int ldx,
                                              rocblas_stride stX,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_cgeblttrs_npvt_strided_batched(handle, nb, nblocks, nrhs, A, lda, stA, B, ldb,
                                                   stB, C, ldc, stC, X, ldx, stX, bc)
        : rocsolver_cgeblttrs_npvt(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx);
}

inline rocblas_status rocsolver_geblttrs_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_int nrhs,
                                              rocblas_double_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_double_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              rocblas_double_complex* C,
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_double_complex* X,
                                              rocblas_int ldx,
                                              rocblas_stride stX,
                                              rocblas_int bc)
{
    return STRIDED
        ? rocsolver_zgeblttrs_npvt_strided_batched(handle, nb, nblocks, nrhs, A, lda, stA, B, ldb,
                                                   stB, C, ldc, stC, X, ldx, stX, bc)
        : rocsolver_zgeblttrs_npvt(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx);
}

// batched
inline rocblas_status rocsolver_geblttrs_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_int nrhs,
                                              float* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              float* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              float* const C[],
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              float* const X[],
                                              rocblas_int ldx,
                                              rocblas_stride stX,
                                              rocblas_int bc)
{
    return rocsolver_sgeblttrs_npvt_batched(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X,
                                            ldx, bc);
}

inline rocblas_status rocsolver_geblttrs_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_int nrhs,
                                              double* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              double* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              double* const C[],
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              double* const X[],
                                              rocblas_int ldx,
                                              rocblas_stride stX,
                                              rocblas_int bc)
{
    return rocsolver_dgeblttrs_npvt_batched(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X,
                                            ldx, bc);
}

inline rocblas_status rocsolver_geblttrs_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_int nrhs,
                                              rocblas_float_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_float_complex* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              rocblas_float_complex* const C[],
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_float_complex* const X[],
                                              rocblas_int ldx,
                                              rocblas_stride stX,
                                              rocblas_int bc)
{
    return rocsolver_cgeblttrs_npvt_batched(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X,
                                            ldx, bc);
}

inline rocblas_status rocsolver_geblttrs_npvt(bool STRIDED,
                                              rocblas_handle handle,
                                              rocblas_int nb,
                                              rocblas_int nblocks,
                                              rocblas_int nrhs,
                                              rocblas_double_complex* const A[],
                                              rocblas_int lda,
                                              rocblas_stride stA,
                                              rocblas_double_complex* const B[],
                                              rocblas_int ldb,
                                              rocblas_stride stB,
                                              rocblas_double_complex* const C[],
                                              rocblas_int ldc,
                                              rocblas_stride stC,
                                              rocblas_double_complex* const X[],
                                              rocblas_int ldx,
                                              rocblas_stride stX,
                                              rocblas_int bc)
{
    return rocsolver_zgeblttrs_npvt_batched(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X,
                                            ldx, bc);
}
/********************************************************/

/******************** GEBLTTRS_NPVT ********************/
// normal and strided_batched
inline rocblas_status rocsolver_geblttrs_npvt_interleaved(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          rocblas_int nrhs,
                                                          float* A,
                                                          rocblas_int inca,
                                                          rocblas_int lda,
                                                          rocblas_stride stA,
                                                          float* B,
                                                          rocblas_int incb,
                                                          rocblas_int ldb,
                                                          rocblas_stride stB,
                                                          float* C,
                                                          rocblas_int incc,
                                                          rocblas_int ldc,
                                                          rocblas_stride stC,
                                                          float* X,
                                                          rocblas_int incx,
                                                          rocblas_int ldx,
                                                          rocblas_stride stX,
                                                          rocblas_int bc)
{
    return rocsolver_sgeblttrs_npvt_interleaved_batched(handle, nb, nblocks, nrhs, A, inca, lda,
                                                        stA, B, incb, ldb, stB, C, incc, ldc, stC,
                                                        X, incx, ldx, stX, bc);
}

inline rocblas_status rocsolver_geblttrs_npvt_interleaved(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          rocblas_int nrhs,
                                                          double* A,
                                                          rocblas_int inca,
                                                          rocblas_int lda,
                                                          rocblas_stride stA,
                                                          double* B,
                                                          rocblas_int incb,
                                                          rocblas_int ldb,
                                                          rocblas_stride stB,
                                                          double* C,
                                                          rocblas_int incc,
                                                          rocblas_int ldc,
                                                          rocblas_stride stC,
                                                          double* X,
                                                          rocblas_int incx,
                                                          rocblas_int ldx,
                                                          rocblas_stride stX,
                                                          rocblas_int bc)
{
    return rocsolver_dgeblttrs_npvt_interleaved_batched(handle, nb, nblocks, nrhs, A, inca, lda,
                                                        stA, B, incb, ldb, stB, C, incc, ldc, stC,
                                                        X, incx, ldx, stX, bc);
}

inline rocblas_status rocsolver_geblttrs_npvt_interleaved(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          rocblas_int nrhs,
                                                          rocblas_float_complex* A,
                                                          rocblas_int inca,
                                                          rocblas_int lda,
                                                          rocblas_stride stA,
                                                          rocblas_float_complex* B,
                                                          rocblas_int incb,
                                                          rocblas_int ldb,
                                                          rocblas_stride stB,
                                                          rocblas_float_complex* C,
                                                          rocblas_int incc,
                                                          rocblas_int ldc,
                                                          rocblas_stride stC,
                                                          rocblas_float_complex* X,
                                                          rocblas_int incx,
                                                          rocblas_int ldx,
                                                          rocblas_stride stX,
                                                          rocblas_int bc)
{
    return rocsolver_cgeblttrs_npvt_interleaved_batched(handle, nb, nblocks, nrhs, A, inca, lda,
                                                        stA, B, incb, ldb, stB, C, incc, ldc, stC,
                                                        X, incx, ldx, stX, bc);
}

inline rocblas_status rocsolver_geblttrs_npvt_interleaved(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          rocblas_int nrhs,
                                                          rocblas_double_complex* A,
                                                          rocblas_int inca,
                                                          rocblas_int lda,
                                                          rocblas_stride stA,
                                                          rocblas_double_complex* B,
                                                          rocblas_int incb,
                                                          rocblas_int ldb,
                                                          rocblas_stride stB,
                                                          rocblas_double_complex* C,
                                                          rocblas_int incc,
                                                          rocblas_int ldc,
                                                          rocblas_stride stC,
                                                          rocblas_double_complex* X,
                                                          rocblas_int incx,
                                                          rocblas_int ldx,
                                                          rocblas_stride stX,
                                                          rocblas_int bc)
{
    return rocsolver_zgeblttrs_npvt_interleaved_batched(handle, nb, nblocks, nrhs, A, inca, lda,
                                                        stA, B, incb, ldb, stB, C, incc, ldc, stC,
                                                        X, incx, ldx, stX, bc);
}
/********************************************************/

/*************** CREATE_ DESTROY_ RFINFO ****************/
// local rocsolver_rfinfo; automatically created and destroyed
class rocsolver_local_rfinfo
{
    rocsolver_rfinfo l_rfinfo;

public:
    rocsolver_local_rfinfo(rocblas_handle handle)
    {
        rocsolver_create_rfinfo(&l_rfinfo, handle);
    }
    ~rocsolver_local_rfinfo()
    {
        rocsolver_destroy_rfinfo(l_rfinfo);
    }

    operator rocsolver_rfinfo&()
    {
        return l_rfinfo;
    }
    operator const rocsolver_rfinfo&() const
    {
        return l_rfinfo;
    }
};

/******************** CSRRF_ANALYSIS ********************/
inline rocblas_status rocsolver_csrrf_analysis(rocblas_handle handle,
                                               rocblas_int n,
                                               rocblas_int nrhs,
                                               rocblas_int nnzM,
                                               rocblas_int* ptrM,
                                               rocblas_int* indM,
                                               float* valM,
                                               rocblas_int nnzT,
                                               rocblas_int* ptrT,
                                               rocblas_int* indT,
                                               float* valT,
                                               rocblas_int* pivP,
                                               rocblas_int* pivQ,
                                               float* B,
                                               rocblas_int ldb,
                                               rocsolver_rfinfo rfinfo)
{
    return rocsolver_scsrrf_analysis(handle, n, nrhs, nnzM, ptrM, indM, valM, nnzT, ptrT, indT,
                                     valT, pivP, pivQ, B, ldb, rfinfo);
}

inline rocblas_status rocsolver_csrrf_analysis(rocblas_handle handle,
                                               rocblas_int n,
                                               rocblas_int nrhs,
                                               rocblas_int nnzM,
                                               rocblas_int* ptrM,
                                               rocblas_int* indM,
                                               double* valM,
                                               rocblas_int nnzT,
                                               rocblas_int* ptrT,
                                               rocblas_int* indT,
                                               double* valT,
                                               rocblas_int* pivP,
                                               rocblas_int* pivQ,
                                               double* B,
                                               rocblas_int ldb,
                                               rocsolver_rfinfo rfinfo)
{
    return rocsolver_dcsrrf_analysis(handle, n, nrhs, nnzM, ptrM, indM, valM, nnzT, ptrT, indT,
                                     valT, pivP, pivQ, B, ldb, rfinfo);
}
/********************************************************/

/********************* CSRRF_SUMLU ************************/
inline rocblas_status rocsolver_csrrf_sumlu(rocblas_handle handle,
                                            rocblas_int n,
                                            rocblas_int nnzL,
                                            rocblas_int* ptrL,
                                            rocblas_int* indL,
                                            float* valL,
                                            rocblas_int nnzU,
                                            rocblas_int* ptrU,
                                            rocblas_int* indU,
                                            float* valU,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            float* valT)
{
    return rocsolver_scsrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU, valU, ptrT,
                                  indT, valT);
}

inline rocblas_status rocsolver_csrrf_sumlu(rocblas_handle handle,
                                            rocblas_int n,
                                            rocblas_int nnzL,
                                            rocblas_int* ptrL,
                                            rocblas_int* indL,
                                            double* valL,
                                            rocblas_int nnzU,
                                            rocblas_int* ptrU,
                                            rocblas_int* indU,
                                            double* valU,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            double* valT)
{
    return rocsolver_dcsrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU, valU, ptrT,
                                  indT, valT);
}
/********************************************************/

/********************* CSRRF_SPLITLU ************************/
inline rocblas_status rocsolver_csrrf_splitlu(rocblas_handle handle,
                                              rocblas_int n,
                                              rocblas_int nnzT,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              float* valT,
                                              rocblas_int* ptrL,
                                              rocblas_int* indL,
                                              float* valL,
                                              rocblas_int* ptrU,
                                              rocblas_int* indU,
                                              float* valU)
{
    return rocsolver_scsrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL, indL, valL, ptrU, indU,
                                    valU);
}

inline rocblas_status rocsolver_csrrf_splitlu(rocblas_handle handle,
                                              rocblas_int n,
                                              rocblas_int nnzT,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              double* valT,
                                              rocblas_int* ptrL,
                                              rocblas_int* indL,
                                              double* valL,
                                              rocblas_int* ptrU,
                                              rocblas_int* indU,
                                              double* valU)
{
    return rocsolver_dcsrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL, indL, valL, ptrU, indU,
                                    valU);
}
/********************************************************/

/********************* CSRRF_REFACTCHOL ************************/
inline rocblas_status rocsolver_csrrf_refactchol(rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_int nnzA,
                                                 rocblas_int* ptrA,
                                                 rocblas_int* indA,
                                                 float* valA,
                                                 rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 float* valT,
                                                 rocblas_int* pivQ,
                                                 rocsolver_rfinfo rfinfo)
{
    return rocsolver_scsrrf_refactchol(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, valT,
                                       pivQ, rfinfo);
}

inline rocblas_status rocsolver_csrrf_refactchol(rocblas_handle handle,
                                                 rocblas_int n,
                                                 rocblas_int nnzA,
                                                 rocblas_int* ptrA,
                                                 rocblas_int* indA,
                                                 double* valA,
                                                 rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 double* valT,
                                                 rocblas_int* pivQ,
                                                 rocsolver_rfinfo rfinfo)
{
    return rocsolver_dcsrrf_refactchol(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, valT,
                                       pivQ, rfinfo);
}
/********************* CSRRF_REFACTLU ************************/
inline rocblas_status rocsolver_csrrf_refactlu(rocblas_handle handle,
                                               rocblas_int n,
                                               rocblas_int nnzA,
                                               rocblas_int* ptrA,
                                               rocblas_int* indA,
                                               float* valA,
                                               rocblas_int nnzT,
                                               rocblas_int* ptrT,
                                               rocblas_int* indT,
                                               float* valT,
                                               rocblas_int* pivP,
                                               rocblas_int* pivQ,
                                               rocsolver_rfinfo rfinfo)
{
    return rocsolver_scsrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, valT,
                                     pivP, pivQ, rfinfo);
}

inline rocblas_status rocsolver_csrrf_refactlu(rocblas_handle handle,
                                               rocblas_int n,
                                               rocblas_int nnzA,
                                               rocblas_int* ptrA,
                                               rocblas_int* indA,
                                               double* valA,
                                               rocblas_int nnzT,
                                               rocblas_int* ptrT,
                                               rocblas_int* indT,
                                               double* valT,
                                               rocblas_int* pivP,
                                               rocblas_int* pivQ,
                                               rocsolver_rfinfo rfinfo)
{
    return rocsolver_dcsrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, valT,
                                     pivP, pivQ, rfinfo);
}
/********************************************************/

/********************* CSRRF_SOLVE ************************/
inline rocblas_status rocsolver_csrrf_solve(rocblas_handle handle,
                                            rocblas_int n,
                                            rocblas_int nrhs,
                                            rocblas_int nnzT,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            float* valT,
                                            rocblas_int* pivP,
                                            rocblas_int* pivQ,
                                            float* B,
                                            rocblas_int ldb,
                                            rocsolver_rfinfo rfinfo)
{
    return rocsolver_scsrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ, B, ldb,
                                  rfinfo);
}

inline rocblas_status rocsolver_csrrf_solve(rocblas_handle handle,
                                            rocblas_int n,
                                            rocblas_int nrhs,
                                            rocblas_int nnzT,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            double* valT,
                                            rocblas_int* pivP,
                                            rocblas_int* pivQ,
                                            double* B,
                                            rocblas_int ldb,
                                            rocsolver_rfinfo rfinfo)
{
    return rocsolver_dcsrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ, B, ldb,
                                  rfinfo);
}
/********************************************************/
