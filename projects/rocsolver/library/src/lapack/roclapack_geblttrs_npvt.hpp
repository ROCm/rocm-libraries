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
#include "roclapack_getrs.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_geblttrs_npvt_getMemorySize(const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           const rocblas_int nrhs,
                                           const rocblas_int batch_count,
                                           size_t* size_work1,
                                           size_t* size_work2,
                                           size_t* size_work3,
                                           size_t* size_work4,
                                           bool* optim_mem,
                                           const rocblas_int ldb = 1,
                                           const rocblas_int ldx = 1,
                                           const rocblas_int incb = 1,
                                           const rocblas_int incx = 1)
{
    // if quick return, no need of workspace
    if(nb == 0 || nblocks == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        return;
    }

    // size requirements for getrs
    rocsolver_getrs_getMemorySize<BATCHED, STRIDED, T>(rocblas_operation_none, nb, nrhs, batch_count,
                                                       size_work1, size_work2, size_work3,
                                                       size_work4, optim_mem, ldb, ldx, incb, incx);
}

template <typename T>
rocblas_status rocsolver_geblttrs_npvt_argCheck(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                const rocblas_int nrhs,
                                                const rocblas_int lda,
                                                const rocblas_int ldb,
                                                const rocblas_int ldc,
                                                const rocblas_int ldx,
                                                T A,
                                                T B,
                                                T C,
                                                T X,
                                                const rocblas_int batch_count = 1,
                                                const rocblas_int inca = 1,
                                                const rocblas_int incb = 1,
                                                const rocblas_int incc = 1,
                                                const rocblas_int incx = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(handle == nullptr)
    {
        return (rocblas_status_invalid_handle);
    };

    // 2. invalid size
    if(nb < 0 || nblocks < 0 || nrhs < 0 || batch_count < 0)
        return rocblas_status_invalid_size;
    if(inca < 1 || lda < inca * nb)
        return rocblas_status_invalid_size;
    if(incb < 1 || ldb < incb * nb)
        return rocblas_status_invalid_size;
    if(incc < 1 || ldc < incc * nb)
        return rocblas_status_invalid_size;
    if(incx < 1 || ldx < incx * nb)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((nb && nblocks > 1 && !A) || (nb && nblocks && !B) || (nb && nblocks > 1 && !C)
       || (nb && nblocks && nrhs && !X))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geblttrs_npvt_template(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                const rocblas_int nrhs,
                                                U A,
                                                const rocblas_int shiftA,
                                                const rocblas_int inca,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                U B,
                                                const rocblas_int shiftB,
                                                const rocblas_int incb,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                U C,
                                                const rocblas_int shiftC,
                                                const rocblas_int incc,
                                                const rocblas_int ldc,
                                                const rocblas_stride strideC,
                                                U X,
                                                const rocblas_int shiftX,
                                                const rocblas_int incx,
                                                const rocblas_int ldx,
                                                const rocblas_stride strideX,
                                                const rocblas_int batch_count,
                                                void* work1,
                                                void* work2,
                                                void* work3,
                                                void* work4,
                                                bool optim_mem)
{
    ROCSOLVER_ENTER("geblttrs_npvt", "nb:", nb, "nblocks:", nblocks, "nrhs:", nrhs,
                    "shiftA:", shiftA, "inca:", inca, "lda:", lda, "shiftB:", shiftB, "incb:", incb,
                    "ldb:", ldb, "shiftC:", shiftC, "incc:", incc, "ldc:", ldc, "shiftX:", shiftX,
                    "incx:", incx, "ldx:", ldx, "bc:", batch_count);

    // quick return
    if(nb == 0 || nblocks == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    T one = T(1);
    T minone = T(-1);

    // block strides
    rocblas_int bsa = lda * nb;
    rocblas_int bsb = ldb * nb;
    rocblas_int bsc = ldc * nb;
    rocblas_int bsx = ldx * nrhs;

    // forward solve
    for(rocblas_int k = 0; k < nblocks; k++)
    {
        if(k > 0)
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nrhs, nb,
                           &minone, A, shiftA + (k - 1) * bsa, inca, lda, strideA, X,
                           shiftX + (k - 1) * bsx, incx, ldx, strideX, &one, X, shiftX + k * bsx,
                           incx, ldx, strideX, batch_count, (T**)nullptr);

        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nrhs, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, X, shiftX + k * bsx, incx, ldx, strideX, batch_count, work1,
            work2, work3, work4, optim_mem, false);
    }

    // backward solve
    for(rocblas_int k = nblocks - 2; k >= 0; k--)
    {
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nrhs, nb,
                       &minone, C, shiftC + k * bsc, incc, ldc, strideC, X, shiftX + (k + 1) * bsx,
                       incx, ldx, strideX, &one, X, shiftX + k * bsx, incx, ldx, strideX,
                       batch_count, (T**)nullptr);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
