/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_getrs.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status rocsolver_getrs_npvt_argCheck(rocblas_handle handle,
                                             const rocblas_operation trans,
                                             const I n,
                                             const I nrhs,
                                             const I lda,
                                             const I ldb,
                                             T A,
                                             T B,
                                             const I batch_count = 1)
{
    const I* ipiv = nullptr;
    bool const pivot = false;

    rocblas_status const istat = rocsolver_getrs_argCheck(handle, trans, n, nrhs, lda, ldb,

                                                          A, B, ipiv, batch_count, pivot);
    return istat;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_getrs_npvt_getMemorySize(rocblas_operation trans,
                                        const I n,
                                        const I nrhs,
                                        const I batch_count,
                                        size_t* size_work1,
                                        size_t* size_work2,
                                        size_t* size_work3,
                                        size_t* size_work4,
                                        bool* optim_mem,
                                        const I lda = 1,
                                        const I ldb = 1,
                                        const I inca = 1,
                                        const I incb = 1)
{
    rocsolver_getrs_getMemorySize<BATCHED, STRIDED, T, I>(trans, n, nrhs, batch_count,

                                                          size_work1, size_work2, size_work3,
                                                          size_work4, optim_mem,

                                                          lda, ldb, inca, incb);
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_getrs_npvt_template(rocblas_handle handle,
                                             const rocblas_operation trans,
                                             const I n,
                                             const I nrhs,
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
                                             void* work1,
                                             void* work2,
                                             void* work3,
                                             void* work4,
                                             const bool optim_mem)
{
    bool const pivot = false;
    const I* const ipiv = nullptr;
    rocblas_stride const strideP = 0;

    return (rocsolver_getrs_template<BATCHED, STRIDED, T, I, U>(handle, trans, n, nrhs,

                                                                A, shiftA, inca, lda, strideA,

                                                                ipiv, strideP,

                                                                B, shiftB, incb, ldb, strideB,

                                                                batch_count,

                                                                work1, work2, work3, work4,
                                                                optim_mem, pivot));
}

ROCSOLVER_END_NAMESPACE
