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
#include "roclapack_getrf.hpp"
#include "roclapack_getrs.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
ROCSOLVER_KERNEL void
    geblttrf_update_info(T* info, T* iinfo, const rocblas_int k_shift, const rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        if(info[b] == 0 && iinfo[b] != 0)
            info[b] = iinfo[b] + k_shift;
    }
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_geblttrf_npvt_getMemorySize(const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           const rocblas_int batch_count,
                                           size_t* size_scalars,
                                           size_t* size_work1,
                                           size_t* size_work2,
                                           size_t* size_work3,
                                           size_t* size_work4,
                                           size_t* size_pivotval,
                                           size_t* size_pivotidx,
                                           size_t* size_iipiv,
                                           size_t* size_iinfo1,
                                           size_t* size_iinfo2,
                                           bool* optim_mem,
                                           const rocblas_int ldb = 1,
                                           const rocblas_int ldc = 1,
                                           const rocblas_int incb = 1,
                                           const rocblas_int incc = 1)
{
    // if quick return, no need of workspace
    if(nb == 0 || nblocks == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        *size_iipiv = 0;
        *size_iinfo1 = 0;
        *size_iinfo2 = 0;
        return;
    }

    bool unused;
    size_t a1 = 0, a2 = 0;
    size_t b1 = 0, b2 = 0;
    size_t c1 = 0, c2 = 0;
    size_t d1 = 0, d2 = 0;

    // size requirements for getrf
    rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T>(
        nb, nb, false, batch_count, size_scalars, &a1, &b1, &c1, &d1, size_pivotval, size_pivotidx,
        size_iipiv, size_iinfo1, optim_mem, ldb, incb);

    // size requirements for getrs
    rocsolver_getrs_getMemorySize<BATCHED, STRIDED, T>(rocblas_operation_none, nb, nb, batch_count,
                                                       &a2, &b2, &c2, &d2, &unused, ldb, ldc, incb,
                                                       incc);

    *size_work1 = std::max(a1, a2);
    *size_work2 = std::max(b1, b2);
    *size_work3 = std::max(c1, c2);
    *size_work4 = std::max(d1, d2);

    // size for temporary info storage
    *size_iinfo2 = sizeof(rocblas_int) * batch_count;
}

template <typename T>
rocblas_status rocsolver_geblttrf_npvt_argCheck(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                const rocblas_int lda,
                                                const rocblas_int ldb,
                                                const rocblas_int ldc,
                                                T A,
                                                T B,
                                                T C,
                                                rocblas_int* info,
                                                const rocblas_int batch_count = 1,
                                                const rocblas_int inca = 1,
                                                const rocblas_int incb = 1,
                                                const rocblas_int incc = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(handle == nullptr)
    {
        return (rocblas_status_invalid_handle);
    };

    // 2. invalid size
    if(nb < 0 || nblocks < 0 || batch_count < 0)
        return rocblas_status_invalid_size;
    if(inca < 1 || lda < inca * nb)
        return rocblas_status_invalid_size;
    if(incb < 1 || ldb < incb * nb)
        return rocblas_status_invalid_size;
    if(incc < 1 || ldc < incc * nb)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((nb && nblocks > 1 && !A) || (nb && nblocks && !B) || (nb && nblocks > 1 && !C)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geblttrf_npvt_template(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
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
                                                rocblas_int* info,
                                                const rocblas_int batch_count,
                                                T* scalars,
                                                void* work1,
                                                void* work2,
                                                void* work3,
                                                void* work4,
                                                T* pivotval,
                                                rocblas_int* pivotidx,
                                                rocblas_int* iipiv,
                                                rocblas_int* iinfo1,
                                                rocblas_int* iinfo2,
                                                bool optim_mem)
{
    ROCSOLVER_ENTER("geblttrf_npvt", "nb:", nb, "nblocks:", nblocks, "shiftA:", shiftA,
                    "inca:", inca, "lda:", lda, "shiftB:", shiftB, "incb:", incb, "ldb:", ldb,
                    "shiftC:", shiftC, "incc:", incc, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(nb == 0 || nblocks == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    T one = T(1);
    T minone = T(-1);

    // block strides
    rocblas_int bsa = lda * nb;
    rocblas_int bsb = ldb * nb;
    rocblas_int bsc = ldc * nb;

    rocsolver_getrf_template<BATCHED, STRIDED, T>(handle, nb, nb, B, shiftB, incb, ldb, strideB,
                                                  (rocblas_int*)nullptr, 0, 0, info, batch_count,
                                                  scalars, work1, work2, work3, work4, pivotval,
                                                  pivotidx, iipiv, iinfo1, optim_mem, false);

    for(rocblas_int k = 0; k < nblocks - 1; k++)
    {
        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nb, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, C, shiftC + k * bsc, incc, ldc, strideC, batch_count, work1,
            work2, work3, work4, optim_mem, false);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nb, nb, &minone,
                       A, shiftA + k * bsa, inca, lda, strideA, C, shiftC + k * bsc, incc, ldc,
                       strideC, &one, B, shiftB + (k + 1) * bsb, incb, ldb, strideB, batch_count,
                       (T**)nullptr);

        rocsolver_getrf_template<BATCHED, STRIDED, T>(
            handle, nb, nb, B, shiftB + (k + 1) * bsb, incb, ldb, strideB, (rocblas_int*)nullptr, 0,
            0, iinfo2, batch_count, scalars, work1, work2, work3, work4, pivotval, pivotidx, iipiv,
            iinfo1, optim_mem, false);

        ROCSOLVER_LAUNCH_KERNEL(geblttrf_update_info, gridReset, threads, 0, stream, info, iinfo2,
                                (k + 1) * nb, batch_count);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
