/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
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

#include "rocblas.hpp"
#include "roclapack_potf2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename I>
static I get_lds_size()
{
    I const default_lds_size = 64 * 1024;

    I lds_size = 0;
    I deviceId = 0;
    auto istat_device = hipGetDevice(&deviceId);
    if(istat_device != hipSuccess)
    {
        return (default_lds_size);
    };
    auto const attr = hipDeviceAttributeMaxSharedMemoryPerBlock;
    auto istat_attr = hipDeviceGetAttribute(&lds_size, attr, deviceId);
    if(istat_attr != hipSuccess)
    {
        return (default_lds_size);
    };

    return (lds_size);
}

template <typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void chk_positive(INFO* iinfo, INFO* info, I j, I batch_count)
{
    I id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(id < batch_count && info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_potrf_getMemorySize(const I n,
                                   const rocblas_fill uplo,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_pivots,
                                   size_t* size_iinfo,
                                   bool* optim_mem)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivots = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    I nb = POTRF_BLOCKSIZE(T);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
    {
        // requirements for calling a single POTF2
        rocsolver_potf2_getMemorySize<T>(n, batch_count, size_scalars, size_work1, size_pivots);
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iinfo = 0;
        *optim_mem = true;
    }
    else
    {
        I jb = nb;
        size_t s1, s2;

        // size to store info about positiveness of each subblock
        *size_iinfo = sizeof(I) * batch_count;

        // requirements for calling POTF2 for the subblocks
        rocsolver_potf2_getMemorySize<T>(jb, batch_count, size_scalars, &s1, size_pivots);

        // extra requirements for calling TRSM
        if(uplo == rocblas_fill_upper)
        {
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                rocblas_side_left, rocblas_operation_conjugate_transpose, jb, n - jb, batch_count,
                &s2, size_work2, size_work3, size_work4, optim_mem);
        }
        else
        {
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                rocblas_side_right, rocblas_operation_conjugate_transpose, n - jb, jb, batch_count,
                &s2, size_work2, size_work3, size_work4, optim_mem);
        }

        *size_work1 = std::max(s1, s2);
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename S, typename U>
rocblas_status rocsolver_potrf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        INFO* info,
                                        const I batch_count,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* pivots,
                                        INFO* iinfo,
                                        bool optim_mem)
{
    ROCSOLVER_ENTER("potrf", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a positive definite matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    I nb = POTRF_BLOCKSIZE(T);
    if(n <= POTRF_POTF2_SWITCHSIZE(T))
        return rocsolver_potf2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                           batch_count, scalars, (T*)work1, pivots);

    // constants for rocblas functions calls
    T t_one = 1;
    S s_one = 1;
    S s_minone = -1;

    I jb, j = 0;

    // (TODO: When the matrix is detected to be non positive definite, we need to
    //  prevent TRSM and HERK to modify further the input matrix; ideally with no
    //  synchronizations.)

    if(uplo == rocblas_fill_upper)
    {
        // Compute the Cholesky factorization A = U'*U.
        while(j < n - POTRF_POTF2_SWITCHSIZE(T))
        {
            // Factor diagonal and subdiagonal blocks
            jb = std::min(n - j, nb); // number of columns in the block
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
            rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda,
                                        strideA, iinfo, batch_count, scalars, (T*)work1, pivots);

            // test for non-positive-definiteness.
            ROCSOLVER_LAUNCH_KERNEL((chk_positive<I, INFO, U>), gridReset, threads, 0, stream,
                                    iinfo, info, j, batch_count);

            if(j + jb < n)
            {
                // update trailing submatrix
                rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, jb, (n - j - jb), A, shiftA + idx2D(j, j, lda), lda,
                    strideA, A, shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count,
                    optim_mem, work1, work2, work3, work4);

                rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_conjugate_transpose, n - j - jb, jb, &s_minone,
                    A, shiftA + idx2D(j, j + jb, lda), lda, strideA, &s_one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);
            }
            j += nb;
        }
    }
    else
    {
        // Compute the Cholesky factorization A = L*L'.
        while(j < n - POTRF_POTF2_SWITCHSIZE(T))
        {
            // Factor diagonal and subdiagonal blocks
            jb = std::min(n - j, nb); // number of columns in the block
            ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
            rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda,
                                        strideA, iinfo, batch_count, scalars, (T*)work1, pivots);

            // test for non-positive-definiteness.
            ROCSOLVER_LAUNCH_KERNEL((chk_positive<I, INFO, U>), gridReset, threads, 0, stream,
                                    iinfo, info, j, batch_count);

            if(j + jb < n)
            {
                // update trailing submatrix
                rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                    handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, (n - j - jb), jb, A, shiftA + idx2D(j, j, lda), lda,
                    strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
                    optim_mem, work1, work2, work3, work4);

                rocblasCall_syrk_herk<BATCHED, T>(
                    handle, uplo, rocblas_operation_none, n - j - jb, jb, &s_minone, A,
                    shiftA + idx2D(j + jb, j, lda), lda, strideA, &s_one, A,
                    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);
            }
            j += nb;
        }
    }

    // factor last block
    if(j < n)
    {
        rocsolver_potf2_template<T>(handle, uplo, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    iinfo, batch_count, scalars, (T*)work1, pivots);
        ROCSOLVER_LAUNCH_KERNEL((chk_positive<I, INFO, U>), gridReset, threads, 0, stream, iinfo,
                                info, j, batch_count);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
