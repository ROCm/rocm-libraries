/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    common location for declarations of specialized kernel launchers.
 *    Specialized kernels and their launchers are defined in cpp files to
 *    help with compile times.
 * ===========================================================================
 */

// larf
template <typename T, typename I, typename U>
rocblas_status larf_run_small(rocblas_handle handle,
                              const rocblas_side side,
                              const I m,
                              const I n,
                              U x,
                              const rocblas_stride shiftX,
                              const I incX,
                              const rocblas_stride strideX,
                              const T* tau,
                              const rocblas_stride strideP,
                              U A,
                              const rocblas_stride shiftA,
                              const I lda,
                              const rocblas_stride strideA,
                              const I batch_count);

// larfg
template <typename T, typename I, typename U>
rocblas_status larfg_run_small(rocblas_handle handle,
                               const I n,
                               U alpha,
                               const rocblas_stride shiftA,
                               const rocblas_stride strideA,
                               T* beta,
                               const rocblas_stride shiftB,
                               const rocblas_stride strideB,
                               U x,
                               const rocblas_stride shiftX,
                               const I incX,
                               const rocblas_stride strideX,
                               T* tau,
                               const rocblas_stride strideP,
                               const I batch_count);

// trsm
template <bool BATCHED, bool STRIDED, typename T, typename I>
rocblas_status rocsolver_trsm_mem(const rocblas_side side,
                                  const rocblas_operation trans,
                                  const I m,
                                  const I n,
                                  const I batch_count,
                                  size_t* size_work1,
                                  size_t* size_work2,
                                  size_t* size_work3,
                                  size_t* size_work4,
                                  bool* optim_mem,
                                  bool inblocked = false,
                                  const I lda = 1,
                                  const I ldb = 1,
                                  const I inca = 1,
                                  const I incb = 1);

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_trsm_lower(rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_diagonal diag,
                                    const I m,
                                    const I n,
                                    U A,
                                    const rocblas_stride shiftA,
                                    const I lda,
                                    const rocblas_stride strideA,
                                    U B,
                                    const rocblas_stride shiftB,
                                    const I ldb,
                                    const rocblas_stride strideB,
                                    const I batch_count,
                                    const bool optim_mem,
                                    void* work1,
                                    void* work2,
                                    void* work3,
                                    void* work4);

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_trsm_lower(rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_diagonal diag,
                                    const I m,
                                    const I n,
                                    U A,
                                    const rocblas_stride shiftA,
                                    const I inca,
                                    const I lda,
                                    const rocblas_stride strideA,
                                    U B,
                                    const rocblas_stride shiftB,
                                    const I incb,
                                    const I ldb,
                                    const rocblas_stride strideB,
                                    const I batch_count,
                                    const bool optim_mem,
                                    void* work1,
                                    void* work2,
                                    void* work3,
                                    void* work4);

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_trsm_upper(rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_diagonal diag,
                                    const I m,
                                    const I n,
                                    U A,
                                    const rocblas_stride shiftA,
                                    const I lda,
                                    const rocblas_stride strideA,
                                    U B,
                                    const rocblas_stride shiftB,
                                    const I ldb,
                                    const rocblas_stride strideB,
                                    const I batch_count,
                                    const bool optim_mem,
                                    void* work1,
                                    void* work2,
                                    void* work3,
                                    void* work4);

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_trsm_upper(rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_diagonal diag,
                                    const I m,
                                    const I n,
                                    U A,
                                    const rocblas_stride shiftA,
                                    const I inca,
                                    const I lda,
                                    const rocblas_stride strideA,
                                    U B,
                                    const rocblas_stride shiftB,
                                    const I incb,
                                    const I ldb,
                                    const rocblas_stride strideB,
                                    const I batch_count,
                                    const bool optim_mem,
                                    void* work1,
                                    void* work2,
                                    void* work3,
                                    void* work4);

// gemm
template <typename T, typename I, typename U1, typename U2, typename U3>
rocblas_status rocsolver_gemm(rocblas_handle handle,
                              rocblas_operation transA,
                              rocblas_operation transB,
                              I m,
                              I n,
                              I k,
                              const T* alpha,
                              U1 A,
                              rocblas_stride shiftA,
                              I lda,
                              rocblas_stride strideA,
                              U2 B,
                              rocblas_stride shiftB,
                              I ldb,
                              rocblas_stride strideB,
                              const T* beta,
                              U3 C,
                              rocblas_stride shiftC,
                              I ldc,
                              rocblas_stride strideC,
                              I batch_count,
                              T** work);

template <typename T, typename I, typename U1, typename U2, typename U3>
rocblas_status rocsolver_gemm(rocblas_handle handle,
                              rocblas_operation transA,
                              rocblas_operation transB,
                              I m,
                              I n,
                              I k,
                              const T* alpha,
                              U1 A,
                              rocblas_stride shiftA,
                              I inca,
                              I lda,
                              rocblas_stride strideA,
                              U2 B,
                              rocblas_stride shiftB,
                              I incb,
                              I ldb,
                              rocblas_stride strideB,
                              const T* beta,
                              U3 C,
                              rocblas_stride shiftC,
                              I incc,
                              I ldc,
                              rocblas_stride strideC,
                              I batch_count,
                              T** work);

// ger
template <bool CONJ, typename T, typename I, typename U>
rocblas_status rocsolver_ger(rocblas_handle handle,
                             I m,
                             I n,
                             const T* alpha,
                             rocblas_stride stridea,
                             U x,
                             rocblas_stride shiftX,
                             I incx,
                             rocblas_stride strideX,
                             U y,
                             rocblas_stride shiftY,
                             I incy,
                             rocblas_stride strideY,
                             U A,
                             rocblas_stride shiftA,
                             I lda,
                             rocblas_stride strideA,
                             I batch_count,
                             T** work);

template <bool CONJ, typename T, typename I, typename U>
rocblas_status rocsolver_ger(rocblas_handle handle,
                             I m,
                             I n,
                             const T* alpha,
                             rocblas_stride stridea,
                             U x,
                             rocblas_stride shiftX,
                             I incx,
                             rocblas_stride strideX,
                             U y,
                             rocblas_stride shiftY,
                             I incy,
                             rocblas_stride strideY,
                             U A,
                             rocblas_stride shiftA,
                             I inca,
                             I lda,
                             rocblas_stride strideA,
                             I batch_count,
                             T** work);

// potf2
template <typename T, typename I, typename INFO, typename U>
rocblas_status potf2_run_small(rocblas_handle handle,
                               const rocblas_fill uplo,
                               const I n,
                               U AA,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               INFO* info,
                               const I batch_count);

#ifdef OPTIMAL

template <typename T, typename I, typename INFO, typename U>
rocblas_status getf2_run_panel(rocblas_handle handle,
                               const I m,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               I* ipiv,
                               const rocblas_stride shiftP,
                               const rocblas_stride strideP,
                               INFO* info,
                               const I batch_count,
                               const bool pivot,
                               const I offset,
                               I* permut_idx,
                               const rocblas_stride stride);

template <typename T, typename I, typename U>
void getf2_run_scale_update(rocblas_handle handle,
                            const I m,
                            const I n,
                            T* pivotval,
                            U A,
                            const rocblas_stride shiftA,
                            const I lda,
                            const rocblas_stride strideA,
                            const I batch_count,
                            const I dimx,
                            const I dimy);

template <typename T, typename I, typename INFO, typename U>
rocblas_status getf2_run_small(rocblas_handle handle,
                               const I m,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               I* ipiv,
                               const rocblas_stride shiftP,
                               const rocblas_stride strideP,
                               INFO* info,
                               const I batch_count,
                               const bool pivot,
                               const I offset,
                               I* permut_idx,
                               const rocblas_stride stride);

template <typename T, typename U>
rocblas_status getri_run_small(rocblas_handle handle,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               rocblas_int* ipiv,
                               const rocblas_int shiftP,
                               const rocblas_stride strideP,
                               rocblas_int* info,
                               const rocblas_int batch_count,
                               const bool complete,
                               const bool pivot);

// trti2
template <typename T, typename U>
void trti2_run_small(rocblas_handle handle,
                     const rocblas_fill uplo,
                     const rocblas_diagonal diag,
                     const rocblas_int n,
                     U A,
                     const rocblas_int shiftA,
                     const rocblas_int lda,
                     const rocblas_stride strideA,
                     const rocblas_int batch_count);

#endif // OPTIMAL

ROCSOLVER_END_NAMESPACE
