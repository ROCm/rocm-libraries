/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "roclapack_trtri.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
__device__ void copy_and_zero(const rocblas_int m,
                              const rocblas_int n,
                              T* a,
                              const rocblas_int lda,
                              T* w,
                              const rocblas_int ldw)
{
    // Copies the lower triangular part of the matrix to the workspace and then
    // replaces it with zeroes
    for(int k = hipThreadIdx_y; k < m * n; k += hipBlockDim_y)
    {
        int i = k % m;
        int j = k / m;
        if(i > j)
        {
            w[i + j * ldw] = a[i + j * lda];
            a[i + j * lda] = 0;
        }
    }
    __syncthreads();
}

template <typename T>
__device__ void zero_work(const rocblas_int m, const rocblas_int n, T* w, const rocblas_int ldw)
{
    // Zeroes the workspace so that calls to gemm and trsm do not alter the matrix
    // (used for singular matrices)
    for(int k = hipThreadIdx_y; k < m * n; k += hipBlockDim_y)
    {
        int i = k % m;
        int j = k / m;
        w[i + j * ldw] = 0;
    }
    __syncthreads();
}

template <typename T>
__device__ void getri_pivot(const rocblas_int n, T* a, const rocblas_int lda, rocblas_int* p)
{
    // Applies the pivots specified in ipiv to the inverted matrix
    rocblas_int jp;
    T temp;
    for(rocblas_int j = n - 2; j >= 0; --j)
    {
        jp = p[j] - 1;
        if(jp != j)
        {
            for(int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
                swap(a[i + j * lda], a[i + jp * lda]);
            __syncthreads();
        }
    }
}

template <typename T, typename U, typename V>
ROCSOLVER_KERNEL void getri_kernel_large1(const rocblas_int n,
                                          const rocblas_int j,
                                          const rocblas_int jb,
                                          U A,
                                          const rocblas_int shiftA,
                                          const rocblas_int lda,
                                          const rocblas_stride strideA,
                                          rocblas_int* info,
                                          V work,
                                          const rocblas_stride strideW)
{
    // Helper kernel for large-size matrices. Preps the matrix for calls to
    // gemm and trsm.
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* w = load_ptr_batch<T>(work, b, 0, strideW);

    if(info[b] != 0)
        zero_work(n - j, jb, w + j, n);
    else
        copy_and_zero(n - j, jb, a + j + j * lda, lda, w + j, n);
}

template <typename T, typename U>
ROCSOLVER_KERNEL void getri_kernel_large2(const rocblas_int n,
                                          U A,
                                          const rocblas_int shiftA,
                                          const rocblas_int lda,
                                          const rocblas_stride strideA,
                                          rocblas_int* ipiv,
                                          const rocblas_int shiftP,
                                          const rocblas_stride strideP,
                                          rocblas_int* info)
{
    // Helper kernel for large-size matrices. Applies the pivots to the inverted
    // matrix.
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    rocblas_int* p = load_ptr_batch<rocblas_int>(ipiv, b, shiftP, strideP);

    if(info[b] == 0)
        getri_pivot(n, a, lda, p);
}

template <bool ISBATCHED>
rocblas_int getri_get_blksize(const rocblas_int dim)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        rocblas_int size[] = {GETRI_BATCH_BLKSIZES};
        rocblas_int intervals[] = {GETRI_BATCH_INTERVALS};
        rocblas_int max = GETRI_BATCH_NUM_INTERVALS;
        blk = size[get_index(intervals, max, dim)];
    }
    else
    {
        rocblas_int size[] = {GETRI_BLKSIZES};
        rocblas_int intervals[] = {GETRI_INTERVALS};
        rocblas_int max = GETRI_NUM_INTERVALS;
        blk = size[get_index(intervals, max, dim)];
    }

    return blk;
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_getri_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_tmpcopy,
                                   size_t* size_workArr,
                                   bool* optim_mem)
{
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

    static constexpr bool ISBATCHED = BATCHED || STRIDED;

#ifdef OPTIMAL
    // if tiny size, no workspace needed
    if((n <= GETRI_TINY_SIZE && !ISBATCHED) || (n <= GETRI_BATCH_TINY_SIZE && ISBATCHED))
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
#endif

    bool opt1, opt2;
    size_t unused, w1a = 0, w1b = 0, w2a = 0, w2b = 0, w3a = 0, w3b = 0, w4a = 0, w4b = 0, t1, t2;

    // requirements for calling TRTRI
    rocsolver_trtri_getMemorySize<BATCHED, STRIDED, T>(rocblas_diagonal_non_unit, n, batch_count,
                                                       &w1b, &w2b, &w3b, &w4b, &t2, &unused, &opt1);

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

#ifdef OPTIMAL
    // if small size nothing else is needed
    if(n <= TRTRI_MAX_COLS)
    {
        *size_work1 = w1b;
        *size_work2 = w2b;
        *size_work3 = w3b;
        *size_work4 = w4b;
        *size_tmpcopy = t2;
        return;
    }
#endif

    // get block size
    rocblas_int blk = getri_get_blksize<ISBATCHED>(n);
    if(blk == 0)
        blk = n;

    // size of temporary array required for copies
    t1 = n * blk * sizeof(T) * batch_count;

    // requirements for calling TRSM
    rocblas_int nn = (n % 128 != 0) ? n : n + 1;
    rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, rocblas_operation_none, nn, blk + 1, 1, 1,
                                     batch_count, &w1a, &w2a, &w3a, &w4a);

    *size_work1 = std::max(w1a, w1b);
    *size_work2 = std::max(w2a, w2b);
    *size_work3 = std::max(w3a, w3b);
    *size_work4 = std::max(w4a, w4b);
    *size_tmpcopy = std::max(t1, t2);

    // always allocate all required memory for TRSM optimal performance
    opt2 = true;

    *optim_mem = opt1 && opt2;
}

template <typename T>
rocblas_status rocsolver_getri_argCheck(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int lda,
                                        T A,
                                        rocblas_int* ipiv,
                                        rocblas_int* info,
                                        const bool pivot,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && pivot && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_getri_template(rocblas_handle handle,
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
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* tmpcopy,
                                        T** workArr,
                                        const bool optim_mem,
                                        const bool pivot)
{
    ROCSOLVER_ENTER("getri", "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftP:", shiftP,
                    "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if(n == 0)
    {
        rocblas_int blocks = (batch_count - 1) / 32 + 1;
        ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocks, 1, 1), dim3(32, 1, 1), 0, stream, info,
                                batch_count, 0);
        return rocblas_status_success;
    }

    static constexpr bool ISBATCHED = BATCHED || STRIDED;

#ifdef OPTIMAL
    if((n <= GETRI_TINY_SIZE && !ISBATCHED) || (n <= GETRI_BATCH_TINY_SIZE && ISBATCHED))
    {
        return getri_run_small<T>(handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info,
                                  batch_count, true, pivot);
    }
#endif

    // compute inverse of U (also check singularity and update info)
    rocsolver_trtri_template<BATCHED, STRIDED, T>(
        handle, rocblas_fill_upper, rocblas_diagonal_non_unit, n, A, shiftA, lda, strideA, info,
        batch_count, work1, work2, work3, work4, tmpcopy, workArr, optim_mem);

    // ************************************************ //
    // Next, compute inv(A) solving inv(A) * L = inv(U) //

#ifdef OPTIMAL
    // if small size, use optimized kernel for stage 2
    if(n <= TRTRI_MAX_COLS)
    {
        return getri_run_small<T>(handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info,
                                  batch_count, false, pivot);
    }
#endif

    rocblas_int threads = std::min(((n - 1) / 64 + 1) * 64, BS1);
    rocblas_int ldw = n;
    rocblas_stride strideW = n * n;

    // get block size
    rocblas_int blk = getri_get_blksize<ISBATCHED>(n);
    if(blk == 0)
        blk = n;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    T minone = -1;
    T one = 1;
    rocblas_int jb;

    rocblas_int nn = ((n - 1) / blk) * blk + 1;
    for(rocblas_int j = nn - 1; j >= 0; j -= blk)
    {
        jb = std::min(n - j, blk);

        // copy and zero entries in case info is nonzero
        ROCSOLVER_LAUNCH_KERNEL(getri_kernel_large1<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                                0, stream, n, j, jb, A, shiftA, lda, strideA, info, tmpcopy, strideW);

        if(j + jb < n)
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, jb,
                           n - j - jb, &minone, A, shiftA + idx2D(0, j + jb, lda), lda, strideA,
                           tmpcopy, j + jb, ldw, strideW, &one, A, shiftA + idx2D(0, j, lda), lda,
                           strideA, batch_count, workArr);

        rocblasCall_trsm(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                         rocblas_diagonal_unit, n, jb, &one, tmpcopy, j, ldw, strideW, A,
                         shiftA + idx2D(0, j, lda), lda, strideA, batch_count, optim_mem, work1,
                         work2, work3, work4, workArr);
    }

    // apply pivoting (column interchanges)
    if(pivot)
        ROCSOLVER_LAUNCH_KERNEL(getri_kernel_large2<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                                0, stream, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info);

    rocblas_set_pointer_mode(handle, old_mode);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
