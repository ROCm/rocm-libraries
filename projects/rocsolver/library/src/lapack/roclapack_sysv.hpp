/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
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

#include "rocblas.hpp"
#include "roclapack_sytrf.hpp"
#include "roclapack_sytrs.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_sysv_argCheck(rocblas_handle handle,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       const rocblas_int lda,
                                       const rocblas_int ldb,
                                       T A,
                                       rocblas_int* ipiv,
                                       T B,
                                       rocblas_int* info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (nrhs && n && !B) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T>
rocblas_status rocsolver_sysv_getMemorySize(rocblas_handle handle,
                                            const rocblas_int n,
                                            const rocblas_int nrhs,
                                            const rocblas_int lda,
                                            const rocblas_int ldb,
                                            const rocblas_int batch_count,
                                            size_t* size_work_sytrf,
                                            size_t* size_work_sytrs,
                                            size_t* size_savedB)
{
    *size_work_sytrf = 0;
    *size_work_sytrs = 0;
    *size_savedB = 0;

    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    // workspace required for sytrf (A is factored even if nrhs = 0, as in LAPACK)
    rocsolver_sytrf_getMemorySize<T>(n, batch_count, size_work_sytrf);

    if(nrhs == 0)
        return rocblas_status_success;

    // workspace required for sytrs
    ROCBLAS_CHECK(rocsolver_sytrs_getMemorySize<T, rocblas_int>(handle, n, nrhs, batch_count, lda,
                                                                ldb, size_work_sytrs));

    // copy of B, restored for the singular instances
    *size_savedB = sizeof(T) * n * size_t(nrhs) * batch_count;

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_sysv_template(rocblas_handle handle,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       U A,
                                       const rocblas_int shiftA,
                                       const rocblas_int lda,
                                       const rocblas_stride strideA,
                                       rocblas_int* ipiv,
                                       const rocblas_stride strideP,
                                       U B,
                                       const rocblas_int shiftB,
                                       const rocblas_int ldb,
                                       const rocblas_stride strideB,
                                       rocblas_int* info,
                                       const rocblas_int batch_count,
                                       T* work_sytrf,
                                       void* work_sytrs,
                                       const size_t size_work_sytrs,
                                       T* savedB)
{
    ROCSOLVER_ENTER("sysv", "uplo:", uplo, "n:", n, "nrhs:", nrhs, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // factorize A (this also sets info, and is done even if nrhs = 0, as in LAPACK)
    ROCBLAS_CHECK(rocsolver_sytrf_template<T>(handle, uplo, n, A, shiftA, lda, strideA, ipiv,
                                              strideP, info, batch_count, work_sytrf));

    // quick return if A or B are empty
    if(n == 0 || nrhs == 0)
        return rocblas_status_success;

    const rocblas_int copyblocksx = (n - 1) / BS2 + 1;
    const rocblas_int copyblocksy = (nrhs - 1) / BS2 + 1;

    // save elements of B that will be overwritten by SYTRS for cases where info is nonzero
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, nrhs, B, shiftB, ldb,
                            strideB, savedB, info_mask(info));

    // solve AX = B, overwriting B with X
    ROCBLAS_CHECK(rocsolver_sytrs_template<T>(handle, uplo, n, nrhs, A, rocblas_stride(shiftA), lda,
                                              strideA, ipiv, strideP, B, rocblas_stride(shiftB), ldb,
                                              strideB, batch_count, work_sytrs, size_work_sytrs));

    // restore elements of B that were overwritten by SYTRS in cases where info is nonzero
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, nrhs, B, shiftB, ldb,
                            strideB, savedB, info_mask(info));

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
