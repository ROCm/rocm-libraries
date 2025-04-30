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

#include "rocauxiliary_ormlq_unmlq.hpp"
#include "rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_ormbr_unmbr_getMemorySize(const rocblas_storev storev,
                                         const rocblas_side side,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int k,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_AbyxORwork,
                                         size_t* size_diagORtmptr,
                                         size_t* size_trfact,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_AbyxORwork = 0;
        *size_diagORtmptr = 0;
        *size_trfact = 0;
        *size_workArr = 0;
        return;
    }

    rocblas_int nq = side == rocblas_side_left ? m : n;

    // requirements for calling ORMQR/UNMQR or ORMLQ/UNMLQ
    if(storev == rocblas_column_wise)
        rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(side, m, n, std::min(nq, k), batch_count,
                                                        size_scalars, size_AbyxORwork,
                                                        size_diagORtmptr, size_trfact, size_workArr);

    else
        rocsolver_ormlq_unmlq_getMemorySize<BATCHED, T>(side, m, n, std::min(nq, k), batch_count,
                                                        size_scalars, size_AbyxORwork,
                                                        size_diagORtmptr, size_trfact, size_workArr);
}

template <bool COMPLEX, typename T, typename U>
rocblas_status rocsolver_ormbr_argCheck(rocblas_handle handle,
                                        const rocblas_storev storev,
                                        const rocblas_side side,
                                        const rocblas_operation trans,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int lda,
                                        const rocblas_int ldc,
                                        T A,
                                        T C,
                                        U ipiv)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if((COMPLEX && trans == rocblas_operation_transpose)
       || (!COMPLEX && trans == rocblas_operation_conjugate_transpose))
        return rocblas_status_invalid_value;
    if(storev != rocblas_column_wise && storev != rocblas_row_wise)
        return rocblas_status_invalid_value;
    bool left = (side == rocblas_side_left);
    bool row = (storev == rocblas_row_wise);

    // 2. invalid size
    rocblas_int nq = left ? m : n;
    if(m < 0 || n < 0 || k < 0 || ldc < m)
        return rocblas_status_invalid_size;
    if(!row && lda < nq)
        return rocblas_status_invalid_size;
    if(row && lda < std::min(nq, k))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((std::min(nq, k) > 0 && !A) || (std::min(nq, k) > 0 && !ipiv) || (m && n && !C))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_ormbr_unmbr_template(rocblas_handle handle,
                                              const rocblas_storev storev,
                                              const rocblas_side side,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              U C,
                                              const rocblas_int shiftC,
                                              const rocblas_int ldc,
                                              const rocblas_stride strideC,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* AbyxORwork,
                                              T* diagORtmptr,
                                              T* trfact,
                                              T** workArr)
{
    ROCSOLVER_ENTER("ormbr_unmbr", "storev:", storev, "side:", side, "trans:", trans, "m:", m,
                    "n:", n, "k:", k, "shiftA:", shiftA, "lda:", lda, "shiftC:", shiftC,
                    "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(!n || !m || !k || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int nq = side == rocblas_side_left ? m : n;
    rocblas_int cols, rows, colC, rowC;
    if(side == rocblas_side_left)
    {
        rows = m - 1;
        cols = n;
        rowC = 1;
        colC = 0;
    }
    else
    {
        rows = m;
        cols = n - 1;
        rowC = 0;
        colC = 1;
    }

    // if column-wise, apply the orthogonal matrix Q generated in the
    // bi-diagonalization gebrd to a general matrix C
    if(storev == rocblas_column_wise)
    {
        if(nq >= k)
        {
            rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                handle, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC,
                ldc, strideC, batch_count, scalars, AbyxORwork, diagORtmptr, trfact, workArr);
        }
        else
        {
            // shift the householder vectors provided by gebrd as they come below the
            // first subdiagonal
            rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                handle, side, trans, rows, cols, nq - 1, A, shiftA + idx2D(1, 0, lda), lda, strideA,
                ipiv, strideP, C, shiftC + idx2D(rowC, colC, ldc), ldc, strideC, batch_count,
                scalars, AbyxORwork, diagORtmptr, trfact, workArr);
        }
    }

    // if row-wise, apply the orthogonal matrix P generated in the
    // bi-diagonalization gebrd to a general matrix C
    else
    {
        rocblas_operation transP;
        if(trans == rocblas_operation_none)
            transP = (COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose);
        else
            transP = rocblas_operation_none;
        if(nq > k)
        {
            rocsolver_ormlq_unmlq_template<BATCHED, STRIDED, T>(
                handle, side, transP, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC,
                ldc, strideC, batch_count, scalars, AbyxORwork, diagORtmptr, trfact, workArr);
        }
        else
        {
            // shift the householder vectors provided by gebrd as they come above the
            // first superdiagonal
            rocsolver_ormlq_unmlq_template<BATCHED, STRIDED, T>(
                handle, side, transP, rows, cols, nq - 1, A, shiftA + idx2D(0, 1, lda), lda,
                strideA, ipiv, strideP, C, shiftC + idx2D(rowC, colC, ldc), ldc, strideC,
                batch_count, scalars, AbyxORwork, diagORtmptr, trfact, workArr);
        }
    }

    return rocblas_status_success;
}

/** Adapts A and C to be of the same type **/
template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_ormbr_unmbr_template(rocblas_handle handle,
                                    const rocblas_storev storev,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    T* const A[],
                                    const rocblas_int shiftA,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    T* ipiv,
                                    const rocblas_stride strideP,
                                    T* C,
                                    const rocblas_int shiftC,
                                    const rocblas_int ldc,
                                    const rocblas_stride strideC,
                                    const rocblas_int batch_count,
                                    T* scalars,
                                    T* AbyxORwork,
                                    T* diagORtmptr,
                                    T* trfact,
                                    T** workArr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, C, strideC,
                            batch_count);

    rocsolver_ormbr_unmbr_template<BATCHED, STRIDED>(
        handle, storev, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP,
        (T* const*)workArr, shiftC, ldc, strideC, batch_count, scalars, AbyxORwork, diagORtmptr,
        trfact, (workArr + batch_count));
}

/** Adapts A and C to be of the same type **/
template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_ormbr_unmbr_template(rocblas_handle handle,
                                    const rocblas_storev storev,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    T* A,
                                    const rocblas_int shiftA,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    T* ipiv,
                                    const rocblas_stride strideP,
                                    T* const C[],
                                    const rocblas_int shiftC,
                                    const rocblas_int ldc,
                                    const rocblas_stride strideC,
                                    const rocblas_int batch_count,
                                    T* scalars,
                                    T* AbyxORwork,
                                    T* diagORtmptr,
                                    T* trfact,
                                    T** workArr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, A, strideA,
                            batch_count);

    rocsolver_ormbr_unmbr_template<BATCHED, STRIDED>(
        handle, storev, side, trans, m, n, k, (T* const*)workArr, shiftA, lda, strideA, ipiv,
        strideP, C, shiftC, ldc, strideC, batch_count, scalars, AbyxORwork, diagORtmptr, trfact,
        (workArr + batch_count));
}

ROCSOLVER_END_NAMESPACE
