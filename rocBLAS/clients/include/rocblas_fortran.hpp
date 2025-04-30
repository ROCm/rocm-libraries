/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas-macros.h"

/*!\file
 *  This file interfaces with our Fortran BLAS interface.
 */

/*
 * ============================================================================
 *     Fortran functions
 * ============================================================================
 */
extern "C" {

// APIs that have orginal and _64 forms
#include "rocblas_fortran.h.in"
#define ROCBLAS_INTERNAL_ILP64 1
#include "rocblas_fortran.h.in"
#undef ROCBLAS_INTERNAL_ILP64

/* ==========
 *    L3
 * ========== */

// trtri
rocblas_status rocblas_strtri_fortran(rocblas_handle   handle,
                                      rocblas_fill     uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int      n,
                                      const float*     A,
                                      rocblas_int      lda,
                                      float*           invA,
                                      rocblas_int      ldinvA);

rocblas_status rocblas_dtrtri_fortran(rocblas_handle   handle,
                                      rocblas_fill     uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int      n,
                                      const double*    A,
                                      rocblas_int      lda,
                                      double*          invA,
                                      rocblas_int      ldinvA);

rocblas_status rocblas_ctrtri_fortran(rocblas_handle               handle,
                                      rocblas_fill                 uplo,
                                      rocblas_diagonal             diag,
                                      rocblas_int                  n,
                                      const rocblas_float_complex* A,
                                      rocblas_int                  lda,
                                      rocblas_float_complex*       invA,
                                      rocblas_int                  ldinvA);

rocblas_status rocblas_ztrtri_fortran(rocblas_handle                handle,
                                      rocblas_fill                  uplo,
                                      rocblas_diagonal              diag,
                                      rocblas_int                   n,
                                      const rocblas_double_complex* A,
                                      rocblas_int                   lda,
                                      rocblas_double_complex*       invA,
                                      rocblas_int                   ldinvA);

// trtri_batched
rocblas_status rocblas_strtri_batched_fortran(rocblas_handle     handle,
                                              rocblas_fill       uplo,
                                              rocblas_diagonal   diag,
                                              rocblas_int        n,
                                              const float* const A[],
                                              rocblas_int        lda,
                                              float* const       invA[],
                                              rocblas_int        ldinvA,
                                              rocblas_int        batch_count);

rocblas_status rocblas_dtrtri_batched_fortran(rocblas_handle      handle,
                                              rocblas_fill        uplo,
                                              rocblas_diagonal    diag,
                                              rocblas_int         n,
                                              const double* const A[],
                                              rocblas_int         lda,
                                              double* const       invA[],
                                              rocblas_int         ldinvA,
                                              rocblas_int         batch_count);

rocblas_status rocblas_ctrtri_batched_fortran(rocblas_handle                     handle,
                                              rocblas_fill                       uplo,
                                              rocblas_diagonal                   diag,
                                              rocblas_int                        n,
                                              const rocblas_float_complex* const A[],
                                              rocblas_int                        lda,
                                              rocblas_float_complex* const       invA[],
                                              rocblas_int                        ldinvA,
                                              rocblas_int                        batch_count);

rocblas_status rocblas_ztrtri_batched_fortran(rocblas_handle                      handle,
                                              rocblas_fill                        uplo,
                                              rocblas_diagonal                    diag,
                                              rocblas_int                         n,
                                              const rocblas_double_complex* const A[],
                                              rocblas_int                         lda,
                                              rocblas_double_complex* const       invA[],
                                              rocblas_int                         ldinvA,
                                              rocblas_int                         batch_count);

// trtri_strided_batched
rocblas_status rocblas_strtri_strided_batched_fortran(rocblas_handle   handle,
                                                      rocblas_fill     uplo,
                                                      rocblas_diagonal diag,
                                                      rocblas_int      n,
                                                      const float*     A,
                                                      rocblas_int      lda,
                                                      rocblas_stride   stride_a,
                                                      float*           invA,
                                                      rocblas_int      ldinvA,
                                                      rocblas_stride   stride_invA,
                                                      rocblas_int      batch_count);

rocblas_status rocblas_dtrtri_strided_batched_fortran(rocblas_handle   handle,
                                                      rocblas_fill     uplo,
                                                      rocblas_diagonal diag,
                                                      rocblas_int      n,
                                                      const double*    A,
                                                      rocblas_int      lda,
                                                      rocblas_stride   stride_a,
                                                      double*          invA,
                                                      rocblas_int      ldinvA,
                                                      rocblas_stride   stride_invA,
                                                      rocblas_int      batch_count);

rocblas_status rocblas_ctrtri_strided_batched_fortran(rocblas_handle               handle,
                                                      rocblas_fill                 uplo,
                                                      rocblas_diagonal             diag,
                                                      rocblas_int                  n,
                                                      const rocblas_float_complex* A,
                                                      rocblas_int                  lda,
                                                      rocblas_stride               stride_a,
                                                      rocblas_float_complex*       invA,
                                                      rocblas_int                  ldinvA,
                                                      rocblas_stride               stride_invA,
                                                      rocblas_int                  batch_count);

rocblas_status rocblas_ztrtri_strided_batched_fortran(rocblas_handle                handle,
                                                      rocblas_fill                  uplo,
                                                      rocblas_diagonal              diag,
                                                      rocblas_int                   n,
                                                      const rocblas_double_complex* A,
                                                      rocblas_int                   lda,
                                                      rocblas_stride                stride_a,
                                                      rocblas_double_complex*       invA,
                                                      rocblas_int                   ldinvA,
                                                      rocblas_stride                stride_invA,
                                                      rocblas_int                   batch_count);

// gemmt
rocblas_status rocblas_sgemmt_fortran(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const float*      alpha,
                                      const float*      A,
                                      rocblas_int       lda,
                                      const float*      B,
                                      rocblas_int       ldb,
                                      const float*      beta,
                                      float*            C,
                                      rocblas_int       ldc);

rocblas_status rocblas_dgemmt_fortran(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const double*     alpha,
                                      const double*     A,
                                      rocblas_int       lda,
                                      const double*     B,
                                      rocblas_int       ldb,
                                      const double*     beta,
                                      double*           C,
                                      rocblas_int       ldc);

rocblas_status rocblas_cgemmt_fortran(rocblas_handle               handle,
                                      rocblas_fill                 uplo,
                                      rocblas_operation            transA,
                                      rocblas_operation            transB,
                                      rocblas_int                  n,
                                      rocblas_int                  k,
                                      const rocblas_float_complex* alpha,
                                      const rocblas_float_complex* A,
                                      rocblas_int                  lda,
                                      const rocblas_float_complex* B,
                                      rocblas_int                  ldb,
                                      const rocblas_float_complex* beta,
                                      rocblas_float_complex*       C,
                                      rocblas_int                  ldc);

rocblas_status rocblas_zgemmt_fortran(rocblas_handle                handle,
                                      rocblas_fill                  uplo,
                                      rocblas_operation             transA,
                                      rocblas_operation             transB,
                                      rocblas_int                   n,
                                      rocblas_int                   k,
                                      const rocblas_double_complex* alpha,
                                      const rocblas_double_complex* A,
                                      rocblas_int                   lda,
                                      const rocblas_double_complex* B,
                                      rocblas_int                   ldb,
                                      const rocblas_double_complex* beta,
                                      rocblas_double_complex*       C,
                                      rocblas_int                   ldc);

// gemmt_batched
rocblas_status rocblas_sgemmt_batched_fortran(rocblas_handle     handle,
                                              rocblas_fill       uplo,
                                              rocblas_operation  transA,
                                              rocblas_operation  transB,
                                              rocblas_int        n,
                                              rocblas_int        k,
                                              const float*       alpha,
                                              const float* const A[],
                                              rocblas_int        lda,
                                              const float* const B[],
                                              rocblas_int        ldb,
                                              const float*       beta,
                                              float* const       C[],
                                              rocblas_int        ldc,
                                              rocblas_int        batch_count);

rocblas_status rocblas_dgemmt_batched_fortran(rocblas_handle      handle,
                                              rocblas_fill        uplo,
                                              rocblas_operation   transA,
                                              rocblas_operation   transB,
                                              rocblas_int         n,
                                              rocblas_int         k,
                                              const double*       alpha,
                                              const double* const A[],
                                              rocblas_int         lda,
                                              const double* const B[],
                                              rocblas_int         ldb,
                                              const double*       beta,
                                              double* const       C[],
                                              rocblas_int         ldc,
                                              rocblas_int         batch_count);

rocblas_status rocblas_cgemmt_batched_fortran(rocblas_handle                     handle,
                                              rocblas_fill                       uplo,
                                              rocblas_operation                  transA,
                                              rocblas_operation                  transB,
                                              rocblas_int                        n,
                                              rocblas_int                        k,
                                              const rocblas_float_complex*       alpha,
                                              const rocblas_float_complex* const A[],
                                              rocblas_int                        lda,
                                              const rocblas_float_complex* const B[],
                                              rocblas_int                        ldb,
                                              const rocblas_float_complex*       beta,
                                              rocblas_float_complex* const       C[],
                                              rocblas_int                        ldc,
                                              rocblas_int                        batch_count);

rocblas_status rocblas_zgemmt_batched_fortran(rocblas_handle                      handle,
                                              rocblas_fill                        uplo,
                                              rocblas_operation                   transA,
                                              rocblas_operation                   transB,
                                              rocblas_int                         n,
                                              rocblas_int                         k,
                                              const rocblas_double_complex*       alpha,
                                              const rocblas_double_complex* const A[],
                                              rocblas_int                         lda,
                                              const rocblas_double_complex* const B[],
                                              rocblas_int                         ldb,
                                              const rocblas_double_complex*       beta,
                                              rocblas_double_complex* const       C[],
                                              rocblas_int                         ldc,
                                              rocblas_int                         batch_count);

// gemmt_strided_batched
rocblas_status rocblas_sgemmt_strided_batched_fortran(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const float*      alpha,
                                                      const float*      A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const float*      B,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      const float*      beta,
                                                      float*            C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

rocblas_status rocblas_dgemmt_strided_batched_fortran(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const double*     alpha,
                                                      const double*     A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const double*     B,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      const double*     beta,
                                                      double*           C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

rocblas_status rocblas_cgemmt_strided_batched_fortran(rocblas_handle               handle,
                                                      rocblas_fill                 uplo,
                                                      rocblas_operation            transA,
                                                      rocblas_operation            transB,
                                                      rocblas_int                  n,
                                                      rocblas_int                  k,
                                                      const rocblas_float_complex* alpha,
                                                      const rocblas_float_complex* A,
                                                      rocblas_int                  lda,
                                                      rocblas_stride               stride_a,
                                                      const rocblas_float_complex* B,
                                                      rocblas_int                  ldb,
                                                      rocblas_stride               stride_b,
                                                      const rocblas_float_complex* beta,
                                                      rocblas_float_complex*       C,
                                                      rocblas_int                  ldc,
                                                      rocblas_stride               stride_c,
                                                      rocblas_int                  batch_count);

rocblas_status rocblas_zgemmt_strided_batched_fortran(rocblas_handle                handle,
                                                      rocblas_fill                  uplo,
                                                      rocblas_operation             transA,
                                                      rocblas_operation             transB,
                                                      rocblas_int                   n,
                                                      rocblas_int                   k,
                                                      const rocblas_double_complex* alpha,
                                                      const rocblas_double_complex* A,
                                                      rocblas_int                   lda,
                                                      rocblas_stride                stride_a,
                                                      const rocblas_double_complex* B,
                                                      rocblas_int                   ldb,
                                                      rocblas_stride                stride_b,
                                                      const rocblas_double_complex* beta,
                                                      rocblas_double_complex*       C,
                                                      rocblas_int                   ldc,
                                                      rocblas_stride                stride_c,
                                                      rocblas_int                   batch_count);

/* ==========
 *    Ext.
 * ========== */

// trsm_ex
rocblas_status rocblas_trsm_ex_fortran(rocblas_handle    handle,
                                       rocblas_side      side,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_diagonal  diag,
                                       rocblas_int       m,
                                       rocblas_int       n,
                                       const void*       alpha,
                                       const void*       A,
                                       rocblas_int       lda,
                                       void*             B,
                                       rocblas_int       ldb,
                                       const void*       invA,
                                       rocblas_int       invA_size,
                                       rocblas_datatype  compute_type);

rocblas_status rocblas_trsm_batched_ex_fortran(rocblas_handle    handle,
                                               rocblas_side      side,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_diagonal  diag,
                                               rocblas_int       m,
                                               rocblas_int       n,
                                               const void*       alpha,
                                               const void*       A,
                                               rocblas_int       lda,
                                               void*             B,
                                               rocblas_int       ldb,
                                               rocblas_int       batch_count,
                                               const void*       invA,
                                               rocblas_int       invA_size,
                                               rocblas_datatype  compute_type);

rocblas_status rocblas_trsm_strided_batched_ex_fortran(rocblas_handle    handle,
                                                       rocblas_side      side,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_diagonal  diag,
                                                       rocblas_int       m,
                                                       rocblas_int       n,
                                                       const void*       alpha,
                                                       const void*       A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_A,
                                                       void*             B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_B,
                                                       rocblas_int       batch_count,
                                                       const void*       invA,
                                                       rocblas_int       invA_size,
                                                       rocblas_stride    stride_invA,
                                                       rocblas_datatype  compute_type);

// geam_ex
rocblas_status rocblas_geam_ex_fortran(rocblas_handle            handle,
                                       rocblas_operation         transA,
                                       rocblas_operation         transB,
                                       rocblas_int               m,
                                       rocblas_int               n,
                                       rocblas_int               k,
                                       const void*               alpha,
                                       const void*               a,
                                       rocblas_datatype          a_type,
                                       rocblas_int               lda,
                                       const void*               b,
                                       rocblas_datatype          b_type,
                                       rocblas_int               ldb,
                                       const void*               beta,
                                       const void*               c,
                                       rocblas_datatype          c_type,
                                       rocblas_int               ldc,
                                       void*                     d,
                                       rocblas_datatype          d_type,
                                       rocblas_int               ldd,
                                       rocblas_datatype          compute_type,
                                       rocblas_geam_ex_operation geam_ex_op);
}
