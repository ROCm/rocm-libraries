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

#include "rocblas.hpp"
#include "roclapack_getrf.hpp"
#include "roclapack_getrs.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_workspace_helper.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_gesv_argCheck(rocblas_handle handle,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       const rocblas_int lda,
                                       const rocblas_int ldb,
                                       T A,
                                       T B,
                                       const rocblas_int* ipiv,
                                       const rocblas_int* info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

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

template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_gesv_getMemorySize(rocblas_handle handle,
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
                                  rocsolver_workspace_helper* work_helper)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return;

    work_helper->set_nested_capacity(2);

    // PHASE 1
    // workspace required for calling GETRF
    rocsolver_workspace_helper* getrf_work = work_helper->add_nested("getrf");
    rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T>(handle, n, n, A, shiftA, 1, lda, strideA,
                                                       ipiv, 0, strideP, info, batch_count,
                                                       getrf_work, true);

    // PHASE 2
    rocsolver_workspace_helper* phase2_work = work_helper->add_nested("phase2");
    phase2_work->set_nested_capacity(1);

    // workspace required for calling GETRS
    rocsolver_workspace_helper* getrs_work = phase2_work->add_nested("getrs");
    rocsolver_getrs_getMemorySize<BATCHED, STRIDED, T>(
        handle, rocblas_operation_none, n, nrhs, A, shiftA, 1, lda, strideA, ipiv, strideP, B,
        shiftB, 1, ldb, strideB, batch_count, getrs_work, true);

    // extra space to copy B
    size_t size_copyB = sizeof(T) * n * nrhs * batch_count;

    phase2_work->assign_sizes({{"copyB", size_copyB}});
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gesv_template(rocblas_handle handle,
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
                                       rocsolver_workspace_helper* work_helper)
{
    ROCSOLVER_ENTER("gesv", "n:", n, "nrhs:", nrhs, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // prepare kernels
    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a nonsingular matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if A or B are empty
    if(n == 0 || nrhs == 0)
        return rocblas_status_success;

    // prepare workspace
    rocsolver_workspace_helper* getrf_work = work_helper->get_nested("getrf");
    rocsolver_workspace_helper* phase2_work = work_helper->get_nested("phase2");
    rocsolver_workspace_helper* getrs_work = phase2_work->get_nested("getrs");
    T* copyB = (T*)(*phase2_work)["copyB"];

    // constants in host memory
    const rocblas_int copyblocksx = (n - 1) / BS2 + 1;
    const rocblas_int copyblocksy = (nrhs - 1) / BS2 + 1;

    // compute LU factorization of A
    rocsolver_getrf_template<BATCHED, STRIDED, T>(handle, n, n, A, shiftA, 1, lda, strideA, ipiv, 0,
                                                  strideP, info, batch_count, getrf_work, true);

    // save elements of B that will be overwritten by GETRS for cases where info is nonzero
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocksx, copyblocksy, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, nrhs, B, shiftB, ldb,
                            strideB, (T*)copyB, info_mask(info));

    // solve AX = B, overwriting B with X
    rocsolver_getrs_template<BATCHED, STRIDED, T>(handle, rocblas_operation_none, n, nrhs, A,
                                                  shiftA, 1, lda, strideA, ipiv, strideP, B, shiftB,
                                                  1, ldb, strideB, batch_count, getrs_work, true);

    // restore elements of B that were overwritten by GETRS in cases where info is nonzero
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocksx, copyblocksy, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, nrhs, B, shiftB, ldb,
                            strideB, (T*)copyB, info_mask(info));

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
