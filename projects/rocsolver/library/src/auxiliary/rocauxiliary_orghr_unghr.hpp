/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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

#include "rocauxiliary_laset.hpp"
#include "rocauxiliary_orgqr_ungqr.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_orghr_unghr_getMemorySize(const rocblas_int n,
                                         const rocblas_int ilo,
                                         const rocblas_int ihi,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_Abyx_tmptr,
                                         size_t* size_trfact,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    rocblas_int nh = ihi - ilo;
    if(n == 0 || nh == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_Abyx_tmptr = 0;
        *size_trfact = 0;
        *size_workArr = 0;
        return;
    }

    // extra workspace for the copyshift_right staging buffer (nh*(nh+1)/2 elements per batch)
    size_t w1 = sizeof(T) * batch_count * nh * (nh + 1) / 2;

    // requirements for calling orgqr/ungqr on the nh x nh subblock
    size_t w2;
    rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(nh, nh, nh, batch_count, size_scalars, &w2,
                                                    size_Abyx_tmptr, size_trfact, size_workArr);

    *size_work = std::max(w1, w2);
}

template <typename T, typename U>
rocblas_status rocsolver_orghr_argCheck(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int ilo,
                                        const rocblas_int ihi,
                                        const rocblas_int lda,
                                        T A,
                                        U tau,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0 || (n && (ilo < 1 || ihi < ilo || ihi > n)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && ihi > ilo && !tau))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orghr_unghr_template(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int ilo,
                                              const rocblas_int ihi,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* tau,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* work,
                                              T* Abyx_tmptr,
                                              T* trfact,
                                              T** workArr)
{
    ROCSOLVER_ENTER("orghr_unghr", "n:", n, "ilo:", ilo, "ihi:", ihi, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int nh = ihi - ilo;

    // when nh == 0 (ilo == ihi), Q is the identity
    if(nh == 0)
    {
        rocsolver_laset_template<T>(handle, rocblas_fill_full, n, n, T{0}, T{1}, A, shiftA, lda,
                                    strideA, batch_count);
        return rocblas_status_success;
    }

    // initialize border columns to identity
    // left border: columns 0..ilo-2 (0-indexed)
    if(ilo > 1)
    {
        rocsolver_laset_template<T>(handle, rocblas_fill_full, n, ilo - 1, T{0}, T{1}, A, shiftA,
                                    lda, strideA, batch_count);
    }

    // right border: columns ihi..n-1 (0-indexed)
    if(ihi < n)
    {
        rocblas_int sub_dim = n - ihi;

        // zero rows 0..ihi-1
        rocblas_int bx = (ihi - 1) / BS2 + 1;
        rocblas_int by = (sub_dim - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(bx, by, batch_count), dim3(BS2, BS2), 0, stream,
                                ihi, sub_dim, A, shiftA + idx2D(0, ihi, lda), lda, strideA);

        // rows ihi..n-1
        rocsolver_laset_template<T>(handle, rocblas_fill_full, sub_dim, sub_dim, T{0}, T{1}, A,
                                    shiftA + idx2D(ihi, ihi, lda), lda, strideA, batch_count);
    }

    // Shift the nh Householder vectors one column to the right within the active subblock.
    // sets A[ilo:ihi-1, ilo:ihi-1] = A[ilo-1:ihi-1, ilo-1:ihi-2] (0-indexed)
    // sets A[ilo-1:ihi-1, ilo-1] = e0
    // sets A[ilo-1, ilo-1:ihi-1] = e0
    rocblas_int ldw = nh;
    rocblas_stride strideW = rocblas_stride(nh) * (nh + 1) / 2;
    rocblas_int shift_blocks = (nh - 1) / BS2 + 1;

    // copy phase: save the strict lower triangle of the nh x nh subblock into work
    ROCSOLVER_LAUNCH_KERNEL(copyshift_right<T>, dim3(shift_blocks, shift_blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, true, nh, A,
                            shiftA + idx2D(ilo - 1, ilo - 1, lda), lda, strideA, work, 0, ldw,
                            strideW);

    // shift phase: write the staged data back one column to the right
    ROCSOLVER_LAUNCH_KERNEL(copyshift_right<T>, dim3(shift_blocks, shift_blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, false, nh, A,
                            shiftA + idx2D(ilo - 1, ilo - 1, lda), lda, strideA, work, 0, ldw,
                            strideW);

    // zero the top part of the active columns A[0:ilo-2, ilo-1:ihi-1]
    if(ilo > 1)
    {
        rocblas_int bx = (ilo - 2) / BS2 + 1;
        rocblas_int by = nh / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(bx, by, batch_count), dim3(BS2, BS2), 0, stream,
                                ilo - 1, nh + 1, A, shiftA + idx2D(0, ilo - 1, lda), lda, strideA);
    }

    // zero the bottom part of the active columns A[ihi:n-1, ilo-1:ihi-1]
    if(ihi < n)
    {
        rocblas_int sub_dim = n - ihi;
        rocblas_int bx = (sub_dim - 1) / BS2 + 1;
        rocblas_int by = nh / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(bx, by, batch_count), dim3(BS2, BS2), 0, stream,
                                sub_dim, nh + 1, A, shiftA + idx2D(ihi, ilo - 1, lda), lda, strideA);
    }

    // apply orgqr/ungqr to the nh x nh subblock at A[ilo, ilo] (0-indexed)
    rocsolver_orgqr_ungqr_template<BATCHED, STRIDED, T>(
        handle, nh, nh, nh, A, shiftA + idx2D(ilo, ilo, lda), lda, strideA, tau + (ilo - 1),
        strideP, batch_count, scalars, work, Abyx_tmptr, trfact, workArr);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
