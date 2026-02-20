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

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename INFO, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void sqrtDiagOnward(U A,
                                     const rocblas_stride shiftA,
                                     const rocblas_stride strideA,
                                     const size_t loc,
                                     const I j,
                                     T* res,
                                     INFO* info)
{
    I id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);
    T t = M[loc] - res[id];

    if(t <= 0.0)
    {
        // error for non-positive definiteness
        if(info[id] == 0)
            info[id] = j + 1; // use fortran 1-based index
        M[loc] = t;
        res[id] = 0;
    }

    else
    {
        // minor is positive definite
        M[loc] = sqrt(t);
        res[id] = 1 / M[loc];
    }
}

template <typename T, typename I, typename INFO, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void sqrtDiagOnward(U A,
                                     const rocblas_stride shiftA,
                                     const rocblas_stride strideA,
                                     const size_t loc,
                                     const I j,
                                     T* res,
                                     INFO* info)
{
    I id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);
    auto t = M[loc].real() - res[id].real();

    if(t <= 0.0)
    {
        // error for non-positive definiteness
        if(info[id] == 0)
            info[id] = j + 1; // use fortran 1-based index
        M[loc] = t;
        res[id] = 0;
    }

    else
    {
        // minor is positive definite
        M[loc] = sqrt(t);
        res[id] = 1 / M[loc];
    }
}

template <typename T, typename I>
void rocsolver_potf2_getMemorySize(const I n,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_pivots)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_pivots = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    if(n <= POTF2_MAX_SMALL_SIZE(T))
    {
        *size_work = 0;
        *size_pivots = 0;
        return;
    }

    // size of workspace
    // TODO: replace with rocBLAS call
    constexpr I ROCBLAS_DOT_NB = 512;
    *size_work = sizeof(T) * ((n - 1) / ROCBLAS_DOT_NB + 2) * batch_count;

    // size of array to store pivots
    *size_pivots = sizeof(T) * batch_count;
}

template <typename T, typename I, typename INFO>
rocblas_status rocsolver_potf2_potrf_argCheck(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const I n,
                                              const I lda,
                                              T A,
                                              INFO* info,
                                              const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
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

template <typename T, typename I, typename INFO, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_potf2_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        INFO* info,
                                        const I batch_count,
                                        T* scalars,
                                        T* work,
                                        T* pivots)
{
    ROCSOLVER_ENTER("potf2", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a positive definite matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if no dimensions
    if(n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    if(n <= POTRF_BLOCKSIZE(T))
    {
        // ----------------------
        // use specialized kernel
        // ----------------------
        potf2_run_small<T>(handle, uplo, n, A, shiftA, lda, strideA, info, batch_count);
    }
    else
    {
        // (TODO: When the matrix is detected to be non positive definite, we need
        // to prevent GEMV and SCAL to modify further the input matrix; ideally with
        // no synchronizations.)

        if(uplo == rocblas_fill_upper)
        {
            // Compute the Cholesky factorization A = U'*U.
            for(I j = 0; j < n; ++j)
            {
                // Compute U(J,J) and test for non-positive-definiteness.
                rocblasCall_dot<COMPLEX, T>(handle, j, A, shiftA + idx2D(0, j, lda), 1, strideA, A,
                                            shiftA + idx2D(0, j, lda), 1, strideA, batch_count,
                                            pivots, work);

                ROCSOLVER_LAUNCH_KERNEL((sqrtDiagOnward<T, I>), dim3(batch_count), dim3(1), 0, stream,
                                        A, shiftA, strideA, idx2D(j, j, lda), j, pivots, info);

                // Compute elements J+1:N of row J
                if(j < n - 1)
                {
                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(0, j, lda), (I)1,
                                                    strideA, batch_count);

                    rocblasCall_gemv<T>(handle, rocblas_operation_transpose, j, n - j - 1, scalars,
                                        0, A, shiftA + idx2D(0, j + 1, lda), lda, strideA, A,
                                        shiftA + idx2D(0, j, lda), 1, strideA, scalars + 2, 0, A,
                                        shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                        nullptr);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(0, j, lda), (I)1,
                                                    strideA, batch_count);

                    rocblasCall_scal<T>(handle, n - j - 1, pivots, 1, A,
                                        shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count);
                }
            }
        }
        else
        {
            // Compute the Cholesky factorization A = L'*L.
            for(I j = 0; j < n; ++j)
            {
                // Compute L(J,J) and test for non-positive-definiteness.
                rocblasCall_dot<COMPLEX, T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            A, shiftA + idx2D(j, 0, lda), lda, strideA, batch_count,
                                            pivots, work);

                ROCSOLVER_LAUNCH_KERNEL((sqrtDiagOnward<T, I>), dim3(batch_count), dim3(1), 0, stream,
                                        A, shiftA, strideA, idx2D(j, j, lda), j, pivots, info);

                // Compute elements J+1:N of column J
                if(j < n - 1)
                {
                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                                    strideA, batch_count);

                    rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j, scalars, 0, A,
                                        shiftA + idx2D(j + 1, 0, lda), lda, strideA, A,
                                        shiftA + idx2D(j, 0, lda), lda, strideA, scalars + 2, 0, A,
                                        shiftA + idx2D(j + 1, j, lda), 1, strideA, batch_count,
                                        nullptr);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                                    strideA, batch_count);

                    rocblasCall_scal<T>(handle, n - j - 1, pivots, 1, A,
                                        shiftA + idx2D(j + 1, j, lda), (I)1, strideA, batch_count);
                }
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
