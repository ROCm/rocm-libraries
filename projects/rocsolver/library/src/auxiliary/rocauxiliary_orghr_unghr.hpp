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

#define ORGHR_SHIFT_THREADS 64 // threads per block in orghr_shift_right
#define ORGHR_SHIFT_UNROLL 4 // columns per thread and iteration in orghr_shift_right

/** ORGHR_SHIFT_RIGHT sets the nh+1 columns ilo-1:ihi-1 (0-indexed) of the n x n matrix A as
    required by ORGHR/UNGHR before calling ORGQR/UNGQR on A[ilo:ihi-1, ilo:ihi-1]:
    - the Householder vectors are shifted one column to the right, i.e.
      A[i, j] = A[i, j-1] for ilo <= i <= ihi-1 and ilo <= j <= i-1,
    - A[ilo-1:ihi-1, ilo-1] = A[ilo-1, ilo-1:ihi-1] = e0, and
    - A[0:ilo-2, ilo-1:ihi-1] = A[ihi:n-1, ilo-1:ihi-1] = 0.
    The diagonal and upper triangular part of A[ilo:ihi-1, ilo:ihi-1] are not referenced.
    Each thread owns one row, so the shift is done in place without synchronization;
    the threads of a block access consecutive rows of a column at a time (coalesced).
    Call this kernel with ORGHR_SHIFT_THREADS threads per block and a grid of
    (ceil(n / ORGHR_SHIFT_THREADS), 1, batch_count) blocks. **/
template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(ORGHR_SHIFT_THREADS)
    orghr_shift_right(const rocblas_int n,
                      const rocblas_int ilo,
                      const rocblas_int ihi,
                      U A,
                      const rocblas_stride shiftA,
                      const rocblas_int lda,
                      const rocblas_stride strideA)
{
    const rocblas_int b = hipBlockIdx_z;
    const rocblas_int i = hipBlockIdx_x * ORGHR_SHIFT_THREADS + hipThreadIdx_x;
    if(i >= n)
        return;

    // Ar points to A[i, ilo-1]
    T* Ar = load_ptr_batch<T>(A, b, shiftA + idx2D(i, ilo - 1, lda), strideA);

    if(i >= ilo && i < ihi)
    {
        // row of the active block: A[i, ilo-1:i-1] = [0, A[i, ilo-1:i-2]]
        // (left to right, carrying the overwritten entry in a register)
        const rocblas_int nc = i - ilo + 1;
        T carry = 0;
        for(rocblas_int c = 0; c < nc; c += ORGHR_SHIFT_UNROLL)
        {
            T v[ORGHR_SHIFT_UNROLL];
#pragma unroll
            for(rocblas_int k = 0; k < ORGHR_SHIFT_UNROLL; k++)
                if(c + k < nc)
                    v[k] = Ar[idx2D(0, c + k, lda)];
#pragma unroll
            for(rocblas_int k = 0; k < ORGHR_SHIFT_UNROLL; k++)
            {
                if(c + k < nc)
                {
                    Ar[idx2D(0, c + k, lda)] = carry;
                    carry = v[k];
                }
            }
        }
    }
    else
    {
        // row ilo-1 is e0; rows above ilo-1 or below ihi-1 are zero
        const rocblas_int nc = ihi - ilo + 1;
        for(rocblas_int c = 0; c < nc; c++)
            Ar[idx2D(0, c, lda)] = (i == ilo - 1 && c == 0) ? T(1) : T(0);
    }
}

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

    // requirements for calling orgqr/ungqr on the nh x nh subblock
    // (the shift of the Householder vectors is done in place)
    rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(nh, nh, nh, batch_count, size_scalars, size_work,
                                                    size_Abyx_tmptr, size_trfact, size_workArr);
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
                                              const rocblas_stride shiftA,
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
        rocsolver_laset_template<T>(handle, rocblas_fill_full, ihi, sub_dim, T{0}, T{0}, A,
                                    shiftA + idx2D(0, ihi, lda), lda, strideA, batch_count);

        // rows ihi..n-1
        rocsolver_laset_template<T>(handle, rocblas_fill_full, sub_dim, sub_dim, T{0}, T{1}, A,
                                    shiftA + idx2D(ihi, ihi, lda), lda, strideA, batch_count);
    }

    // active columns ilo-1..ihi-1 (0-indexed): shift the nh Householder vectors one column
    // to the right in place, and set the first column and row of A[ilo-1:ihi-1, ilo-1:ihi-1]
    // to e0 and the rest of the columns to zero
    rocblas_int blocks = (n - 1) / ORGHR_SHIFT_THREADS + 1;
    ROCSOLVER_LAUNCH_KERNEL(orghr_shift_right<T>, dim3(blocks, 1, batch_count),
                            dim3(ORGHR_SHIFT_THREADS), 0, stream, n, ilo, ihi, A, shiftA, lda,
                            strideA);

    // apply orgqr/ungqr to the nh x nh subblock at A[ilo, ilo] (0-indexed)
    rocsolver_orgqr_ungqr_template<BATCHED, STRIDED, T>(
        handle, nh, nh, nh, A, shiftA + idx2D(ilo, ilo, lda), lda, strideA, tau + (ilo - 1),
        strideP, batch_count, scalars, work, Abyx_tmptr, trfact, workArr);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
