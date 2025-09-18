/************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_ormtr_unmtr.hpp"
#include "auxiliary/rocauxiliary_stedcj.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "roclapack_sytrd_hetrd.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevdj_heevdj_getMemorySize(const rocblas_evect evect,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int batch_count,
                                           size_t* size_scalars,
                                           size_t* size_workE,
                                           size_t* size_workTau,
                                           size_t* size_workVec,
                                           size_t* size_workSplits,
                                           size_t* size_work1,
                                           size_t* size_work2,
                                           size_t* size_work3,
                                           size_t* size_work4,
                                           size_t* size_workArr)
{
    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_workSplits = 0;
        *size_workE = 0;
        *size_workTau = 0;
        *size_workVec = 0;
        *size_workArr = 0;
        return;
    }

    // if size too small or no vectors required
    if(evect != rocblas_evect_original || n < SYEVDJ_MIN_DC_SIZE)
    {
        // space to store the residual
        *size_workE = sizeof(S) * batch_count;

        // space to store the number of sweeps
        *size_workSplits = sizeof(rocblas_int) * batch_count;

        // requirements for jacobi
        rocsolver_syevj_heevj_getMemorySize<BATCHED, T, S>(evect, uplo, n, batch_count,
                                                           size_workVec, size_workTau, size_work1,
                                                           size_work2, size_work3, size_work4);

        *size_scalars = 0;
        *size_workArr = 0;

        return;
    }

    size_t unused;
    size_t w11 = 0, w12 = 0, w13 = 0;
    size_t w21 = 0, w22 = 0, w23 = 0;
    size_t w31 = 0, w32 = 0, w33 = 0;

    // space for the superdiagonal of tridiag form
    *size_workE = sizeof(S) * n * batch_count;

    // space for the householder scalars
    *size_workTau = sizeof(T) * n * batch_count;

    // temp space for eigenvectors
    *size_workVec = sizeof(T) * n * n * batch_count;

    // requirements for tridiagonalization (sytrd/hetrd)
    rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w11, &w21, &w31,
                                                    &unused, false);

    // extra requirements for computing eigenvalues and vectors (stedcj)
    rocsolver_stedcj_getMemorySize<BATCHED, T, S>(rocblas_evect_tridiagonal, n, batch_count, &w12,
                                                  &w22, &w32, size_work4, size_workSplits, &unused);

    // extra requirements for ormtr/unmtr
    rocsolver_ormtr_unmtr_getMemorySize<BATCHED, T>(rocblas_side_left, uplo, n, n, batch_count,
                                                    &unused, &w13, &w23, &w33, &unused);

    // get max values
    *size_work1 = std::max({w11, w12, w13});
    *size_work2 = std::max({w21, w22, w23});
    *size_work3 = std::max({w31, w32, w33});

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;
}

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevdj_heevdj_argCheck(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                T A,
                                                const rocblas_int lda,
                                                S* D,
                                                rocblas_int* info,
                                                const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((evect != rocblas_evect_original && evect != rocblas_evect_none)
       || (uplo != rocblas_fill_lower && uplo != rocblas_fill_upper))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !D) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W>
rocblas_status rocsolver_syevdj_heevdj_template(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                W A,
                                                const rocblas_int shiftA,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                S* D,
                                                const rocblas_stride strideD,
                                                rocblas_int* info,
                                                const rocblas_int batch_count,
                                                T* scalars,
                                                S* workE,
                                                T* workTau,
                                                T* workVec,
                                                rocblas_int* workSplits,
                                                void* work1,
                                                void* work2,
                                                void* work3,
                                                void* work4,
                                                void* workArr)
{
    ROCSOLVER_ENTER("syevdj_heevdj", "evect:", evect, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // quick return for n = 1 (scalar case)
    if(n == 1)
    {
        ROCSOLVER_LAUNCH_KERNEL(syev_scalar_case<T>, gridReset, threads, 0, stream, evect, A,
                                strideA, D, strideD, batch_count);
        return rocblas_status_success;
    }

    // TODO: Scale the matrix

    if(evect != rocblas_evect_original || n < SYEVDJ_MIN_DC_SIZE)
    {
        // **** do not use D&C approach ****

        rocsolver_syevj_heevj_template<BATCHED, STRIDED, T>(
            handle, rocblas_esort_ascending, evect, uplo, n, A, shiftA, lda, strideA, (S)0, workE,
            20, workSplits, D, strideD, info, batch_count, workVec, workTau, (S*)work1,
            (rocblas_int*)work2, (rocblas_int*)work3, (rocblas_int*)work4);
    }
    else
    {
        // **** Use D&C approach ****

        // reduce A to tridiagonal form
        // (Note: a tridiag form is necessary to apply D&C. To solve the subblocks with Jacobi will
        // require copy D and E into a full tridiag matrix however, given all the zeros above the super diagonal,
        // it is expected that the algorithm converges in fewer sweeps)
        rocsolver_sytrd_hetrd_template<BATCHED>(handle, uplo, n, A, shiftA, lda, strideA, D,
                                                strideD, workE, n, workTau, n, batch_count, scalars,
                                                (T*)work1, (T*)work2, (T*)work3, (T**)workArr, false);

        constexpr bool ISBATCHED = BATCHED || STRIDED;
        const rocblas_int ldv = n;
        const rocblas_stride strideV = n * n;

        // solve with Jacobi solver
        rocsolver_stedcj_template<false, ISBATCHED, T>(
            handle, rocblas_evect_tridiagonal, n, D, strideD, workE, n, workVec, 0, ldv, strideV,
            info, batch_count, work1, (S*)work2, (S*)work3, (S*)work4, workSplits, (S**)workArr);

        // update vectors
        rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
            handle, rocblas_side_left, uplo, rocblas_operation_none, n, n, A, shiftA, lda, strideA,
            workTau, n, workVec, 0, ldv, strideV, batch_count, scalars, (T*)work1, (T*)work2,
            (T*)work3, (T**)workArr);

        // copy vectors into A
        const rocblas_int copyblocks = (n - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                dim3(BS2, BS2), 0, stream, n, n, workVec, 0, ldv, strideV, A,
                                shiftA, lda, strideA);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
