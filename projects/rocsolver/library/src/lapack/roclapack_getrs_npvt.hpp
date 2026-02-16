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

#include "auxiliary/rocauxiliary_laswp.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

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
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (nrhs && n && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
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
    *size_work1 = 0;
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *optim_mem = true;
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        return;
    }

    // workspace required for calling TRSM
    rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, nrhs, batch_count,
                                            size_work1, size_work2, size_work3, size_work4,
                                            optim_mem, lda, ldb, inca, incb);
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
    I* const ipiv = nullptr;
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
