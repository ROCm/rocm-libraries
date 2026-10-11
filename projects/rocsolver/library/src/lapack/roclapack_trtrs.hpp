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

#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Sets info[b] to the position (1-based) of the first zero diagonal element of A_b, or to zero
    if there is none. Grid (1, 1, batch_count), block BS1. **/
template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) trtrs_check_singularity(const I n,
                                                                     U A,
                                                                     const rocblas_stride shiftA,
                                                                     const I lda,
                                                                     const rocblas_stride strideA,
                                                                     I* info)
{
    const I b = blockIdx.z;
    const I tid = threadIdx.x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    __shared__ I sfirst[BS1];

    // first zero diagonal element seen by this thread (n if none)
    I first = n;
    for(I i = tid; i < n; i += BS1)
    {
        if(a[idx2D(i, i, lda)] == T(0))
        {
            first = i;
            break;
        }
    }

    sfirst[tid] = first;
    __syncthreads();
    for(I k = BS1 / 2; k > 0; k /= 2)
    {
        if(tid < k)
            sfirst[tid] = std::min(sfirst[tid], sfirst[tid + k]);
        __syncthreads();
    }

    if(tid == 0)
        info[b] = (sfirst[0] < n) ? sfirst[0] + 1 : 0;
}

/** Copies B_b to the buffer (TO_BUFFER) or back, for the instances with info[b] != 0, so that
    their B_b is not modified. Grid (ceil(n / BS2), ceil(nrhs / BS2), batch_count), block (BS2, BS2). **/
template <bool TO_BUFFER, typename T, typename I, typename U>
ROCSOLVER_KERNEL void trtrs_save_rhs(const I n,
                                     const I nrhs,
                                     U B,
                                     const rocblas_stride shiftB,
                                     const I ldb,
                                     const rocblas_stride strideB,
                                     T* buffer,
                                     const I* info)
{
    const I b = blockIdx.z;
    if(info[b] == 0)
        return;

    T* bb = load_ptr_batch<T>(B, b, shiftB, strideB);
    T* buf = buffer + rocblas_stride(b) * n * nrhs;

    for(I j = blockIdx.y * I(blockDim.y) + threadIdx.y; j < nrhs; j += I(gridDim.y) * blockDim.y)
    {
        for(I i = blockIdx.x * I(blockDim.x) + threadIdx.x; i < n; i += I(gridDim.x) * blockDim.x)
        {
            if(TO_BUFFER)
                buf[i + rocblas_stride(j) * n] = bb[idx2D(i, j, ldb)];
            else
                bb[idx2D(i, j, ldb)] = buf[i + rocblas_stride(j) * n];
        }
    }
}

template <typename T, typename I>
rocblas_status rocsolver_trtrs_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_operation trans,
                                        const rocblas_diagonal diag,
                                        const I n,
                                        const I nrhs,
                                        const I lda,
                                        const I ldb,
                                        T A,
                                        T B,
                                        I* info,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && nrhs && !B) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_trtrs_getMemorySize(const rocblas_operation trans,
                                   const rocblas_diagonal diag,
                                   const I n,
                                   const I nrhs,
                                   const I batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_workB,
                                   bool* optim_mem)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_workB = 0;
        *optim_mem = true;
        return;
    }

    // workspace required for calling TRSM
    rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, nrhs, batch_count,
                                            size_work1, size_work2, size_work3, size_work4,
                                            optim_mem);

    // copy of B, restored for the singular instances (only if A can be singular)
    if(diag == rocblas_diagonal_non_unit)
        *size_workB = sizeof(T) * n * size_t(nrhs) * batch_count;
    else
        *size_workB = 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_trtrs_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_operation trans,
                                        const rocblas_diagonal diag,
                                        const I n,
                                        const I nrhs,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        U B,
                                        const rocblas_stride shiftB,
                                        const I ldb,
                                        const rocblas_stride strideB,
                                        I* info,
                                        const I batch_count,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* workB,
                                        bool optim_mem)
{
    ROCSOLVER_ENTER("trtrs", "uplo:", uplo, "trans:", trans, "diag:", diag, "n:", n, "nrhs:", nrhs,
                    "shiftA:", shiftA, "lda:", lda, "shiftB:", shiftB, "ldb:", ldb,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // info = 0
    const I blocksReset = (batch_count - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL((reset_info<I, I, I>), dim3(blocksReset), dim3(BS1), 0, stream, info,
                            batch_count, I(0));

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // as in LAPACK, check for singularity even if there is no right hand side
    const bool check = (diag == rocblas_diagonal_non_unit);
    if(check)
        ROCSOLVER_LAUNCH_KERNEL((trtrs_check_singularity<T>), dim3(1, 1, batch_count), dim3(BS1), 0,
                                stream, n, A, shiftA, lda, strideA, info);

    // quick return
    if(nrhs == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    const I blocksx = I(std::min<int64_t>((int64_t(n) - 1) / BS2 + 1, 65535));
    const I blocksy = I(std::min<int64_t>((int64_t(nrhs) - 1) / BS2 + 1, 65535));
    dim3 gridCopy(blocksx, blocksy, batch_count);
    dim3 threadsCopy(BS2, BS2, 1);

    // save B for the singular instances
    if(check)
        ROCSOLVER_LAUNCH_KERNEL((trtrs_save_rhs<true, T>), gridCopy, threadsCopy, 0, stream, n,
                                nrhs, B, shiftB, ldb, strideB, workB, info);

    // solve op(A) X = B, overwriting B with X
    if(uplo == rocblas_fill_upper)
    {
        ROCBLAS_CHECK(rocsolver_trsm_upper<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, trans, diag, n, nrhs, A, shiftA, lda, strideA, B, shiftB,
            ldb, strideB, batch_count, optim_mem, work1, work2, work3, work4));
    }
    else
    {
        ROCBLAS_CHECK(rocsolver_trsm_lower<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, trans, diag, n, nrhs, A, shiftA, lda, strideA, B, shiftB,
            ldb, strideB, batch_count, optim_mem, work1, work2, work3, work4));
    }

    // restore B for the singular instances
    if(check)
        ROCSOLVER_LAUNCH_KERNEL((trtrs_save_rhs<false, T>), gridCopy, threadsCopy, 0, stream, n,
                                nrhs, B, shiftB, ldb, strideB, workB, info);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
