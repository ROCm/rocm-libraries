/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
ROCSOLVER_KERNEL void invdiag(const rocblas_diagonal diag,
                              const rocblas_int n,
                              U A,
                              const rocblas_int shiftA,
                              const rocblas_int lda,
                              const rocblas_stride strideA,
                              T* alphas)
{
    int b = hipBlockIdx_y;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* d = alphas + b * n;

        if(a[i + i * lda] != 0 && diag == rocblas_diagonal_non_unit)
        {
            a[i + i * lda] = 1.0 / a[i + i * lda];
            d[i] = -a[i + i * lda];
        }
        else
            d[i] = -1.0;
    }
}

template <bool ISBATCHED>
rocblas_int trtri_get_blksize(const rocblas_int dim)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        rocblas_int size[] = {TRTRI_BATCH_BLKSIZES};
        rocblas_int intervals[] = {TRTRI_BATCH_INTERVALS};
        rocblas_int max = TRTRI_BATCH_NUM_INTERVALS;
        blk = size[get_index(intervals, max, dim)];
    }
    else
    {
        rocblas_int size[] = {TRTRI_BLKSIZES};
        rocblas_int intervals[] = {TRTRI_INTERVALS};
        rocblas_int max = TRTRI_NUM_INTERVALS;
        blk = size[get_index(intervals, max, dim)];
    }

    return blk;
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_trtri_getMemorySize(const rocblas_diagonal diag,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_tmpcopy,
                                   size_t* size_workArr,
                                   bool* optim_mem)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // if quick return, no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_tmpcopy = 0;
        *size_workArr = 0;
        *optim_mem = true;
        return;
    }

    // get block size
    rocblas_int blk = trtri_get_blksize<ISBATCHED>(n);

    // size of temporary array required for copies
    if(diag == rocblas_diagonal_unit && blk > 0)
        *size_tmpcopy = 0;
    else
        *size_tmpcopy = n * n * sizeof(T) * batch_count;

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    size_t w1a, w1b, w3a, w3b;

    // requirements for TRTI2
    rocblas_int nn = (blk == 1) ? n : blk;
#ifdef OPTIMAL
    if(nn <= TRTRI_MAX_COLS)
    {
        // if very small size, no workspace needed
        w1a = 0;
        w3a = 0;
    }
    else
    {
        // size for TRMV
        w1a = nn * sizeof(T) * batch_count;
        // size for alphas
        w3a = nn * sizeof(T) * batch_count;
    }
#else
    // size for TRMV
    w1a = nn * sizeof(T) * batch_count;
    // size for alphas
    w3a = nn * sizeof(T) * batch_count;
#endif

    if(blk == 0)
    {
        // requirements for calling rocBLAS TRTRI
        rocblasCall_trtri_mem<BATCHED, T>(n, batch_count, size_work1, size_work2);
        *size_work3 = 0;
        *size_work4 = 0;
        *optim_mem = true;
    }
    else if(blk == 1)
    {
        *size_work1 = w1a;
        *size_work2 = 0;
        *size_work3 = w3a;
        *size_work4 = 0;
        *optim_mem = true;
    }
    else
    {
        rocblas_int nn = (n % 128 != 0) ? n : n + 1;
        rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, rocblas_operation_none, nn, blk, 1, 1,
                                         batch_count, &w1b, size_work2, &w3b, size_work4);
        *size_work1 = std::max(w1a, w1b);
        *size_work3 = std::max(w3a, w3b);

        // always allocate all required memory for TRSM optimal performance
        *optim_mem = true;
    }
}

template <typename T>
rocblas_status rocsolver_trtri_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_diagonal diag,
                                        const rocblas_int n,
                                        const rocblas_int lda,
                                        T A,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;
    if(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void trti2(rocblas_handle handle,
           const rocblas_fill uplo,
           const rocblas_diagonal diag,
           const rocblas_int n,
           U A,
           const rocblas_int shiftA,
           const rocblas_int lda,
           const rocblas_stride strideA,
           const rocblas_int batch_count,
           T* work,
           T* alphas)
{
#ifdef OPTIMAL
    // if very small size, use optimized kernel
    if(n <= TRTRI_MAX_COLS)
    {
        trti2_run_small<T>(handle, uplo, diag, n, A, shiftA, lda, strideA, batch_count);
        return;
    }
#endif

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_stride stdw = rocblas_stride(n);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // inverse of the diagonal (reciprocals)
    rocblas_int blocks = (n - 1) / 32 + 1;
    ROCSOLVER_LAUNCH_KERNEL(invdiag<T>, dim3(blocks, batch_count), dim3(32, 1), 0, stream, diag, n,
                            A, shiftA, lda, strideA, alphas);

    if(uplo == rocblas_fill_upper)
    {
        for(rocblas_int j = 1; j < n; ++j)
        {
            rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, diag, j, A, shiftA, lda,
                                strideA, A, shiftA + idx2D(0, j, lda), 1, strideA, work, stdw,
                                batch_count);

            rocblasCall_scal<T>(handle, j, alphas + j, stdw, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count);
        }
    }
    else //rocblas_fill_lower
    {
        for(rocblas_int j = n - 2; j >= 0; --j)
        {
            rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, diag, n - j - 1, A,
                                shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, A,
                                shiftA + idx2D(j + 1, j, lda), 1, strideA, work, stdw, batch_count);

            rocblasCall_scal<T>(handle, n - j - 1, alphas + j, stdw, A,
                                shiftA + idx2D(j + 1, j, lda), 1, strideA, batch_count);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_trtri_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_diagonal diag,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* tmpcopy,
                                        T** workArr,
                                        const bool optim_mem)
{
    ROCSOLVER_ENTER("trtri", "uplo:", uplo, "diag:", diag, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // start with info = 0
    rocblas_int blocks = (batch_count - 1) / 32 + 1;
    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocks, 1, 1), dim3(32, 1, 1), 0, stream, info,
                            batch_count, 0);

    // quick return if no dimensions
    if(n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    T one = 1;
    T minone = -1;

    blocks = (n - 1) / 32 + 1;
    rocblas_int ldw = n;
    rocblas_stride strideW = n * n;

    // check for singularities if non-unit diagonal
    if(diag == rocblas_diagonal_non_unit)
    {
        ROCSOLVER_LAUNCH_KERNEL(check_singularity<T>, dim3(batch_count, 1, 1), dim3(1, 64, 1), 0,
                                stream, n, A, shiftA, lda, strideA, info);
    }

    // get block size
    rocblas_int blk = trtri_get_blksize<ISBATCHED>(n);
    rocblas_int jb;

    if(diag == rocblas_diagonal_non_unit && blk > 0)
    {
        // save copy of A to restore it in cases where info is nonzero
        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                                stream, copymat_to_buffer, n, n, A, shiftA, lda, strideA, tmpcopy,
                                info_mask(info));
    }

    if(blk == 0)
    {
        // simply use rocblas_trtri
        rocblasCall_trtri(handle, uplo, diag, n, A, shiftA, lda, strideA, tmpcopy, 0, ldw, strideW,
                          batch_count, (T*)work1, (T**)work2, workArr);

        // copy result to A if info is zero
        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                                stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, tmpcopy,
                                info_mask(info, info_mask::negate), uplo, diag);
    }

    else if(blk == 1)
    {
        // use the unblocked algorithm
        trti2<T>(handle, uplo, diag, n, A, shiftA, lda, strideA, batch_count, (T*)work1, (T*)work3);
    }

    else
    {
        // use blocked algorithm with block size blk
        if(uplo == rocblas_fill_upper)
        {
            for(rocblas_int j = 0; j < n; j += blk)
            {
                jb = std::min(n - j, blk);

                // update current block column
                rocblasCall_trmm(handle, rocblas_side_left, uplo, rocblas_operation_none, diag, j,
                                 jb, &one, 0, A, shiftA, lda, strideA, A, shiftA + idx2D(0, j, lda),
                                 lda, strideA, batch_count);

                rocblasCall_trsm(handle, rocblas_side_right, uplo, rocblas_operation_none, diag, j,
                                 jb, &minone, A, shiftA + idx2D(j, j, lda), lda, strideA, A,
                                 shiftA + idx2D(0, j, lda), lda, strideA, batch_count, optim_mem,
                                 work1, work2, work3, work4);

                trti2<T>(handle, uplo, diag, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                         batch_count, (T*)work1, (T*)work3);
            }
        }
        else // rocblas_fill_lower
        {
            rocblas_int nn = ((n - 1) / blk) * blk + 1;
            for(rocblas_int j = nn - 1; j >= 0; j -= blk)
            {
                jb = std::min(n - j, blk);

                // update current block column
                rocblasCall_trmm(handle, rocblas_side_left, uplo, rocblas_operation_none, diag,
                                 n - j - jb, jb, &one, 0, A, shiftA + idx2D(j + jb, j + jb, lda),
                                 lda, strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA,
                                 batch_count);

                rocblasCall_trsm(handle, rocblas_side_right, uplo, rocblas_operation_none, diag,
                                 n - j - jb, jb, &minone, A, shiftA + idx2D(j, j, lda), lda,
                                 strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA,
                                 batch_count, optim_mem, work1, work2, work3, work4);

                // inverse of current diagonal block
                trti2<T>(handle, uplo, diag, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                         batch_count, (T*)work1, (T*)work3);
            }
        }
    }

    if(diag == rocblas_diagonal_non_unit && blk > 0)
    {
        // restore A in cases where info is nonzero
        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                                stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, tmpcopy,
                                info_mask(info));
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
