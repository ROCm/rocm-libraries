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

#include "rocauxiliary_ormql_unmql.hpp"
#include "rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_ormtr_unmtr_getMemorySize(const rocblas_side side,
                                         const rocblas_fill uplo,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_AbyxORwork,
                                         size_t* size_diagORtmptr,
                                         size_t* size_trfact,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_AbyxORwork = 0;
        *size_diagORtmptr = 0;
        *size_trfact = 0;
        *size_workArr = 0;
        return;
    }

    rocblas_int nq = side == rocblas_side_left ? m : n;

    // requirements for calling ORMQL/UNMQL or ORMQR/UNMQR
    if(uplo == rocblas_fill_upper)
        rocsolver_ormql_unmql_getMemorySize<BATCHED, T>(side, m, n, nq, batch_count, size_scalars,
                                                        size_AbyxORwork, size_diagORtmptr,
                                                        size_trfact, size_workArr);

    else
        rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(side, m, n, nq, batch_count, size_scalars,
                                                        size_AbyxORwork, size_diagORtmptr,
                                                        size_trfact, size_workArr);
}

template <bool COMPLEX, typename T, typename U>
rocblas_status rocsolver_ormtr_argCheck(rocblas_handle handle,
                                        const rocblas_side side,
                                        const rocblas_fill uplo,
                                        const rocblas_operation trans,
                                        const rocblas_int m,
                                        const rocblas_int n,
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
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    bool left = (side == rocblas_side_left);

    // 2. invalid size
    rocblas_int nq = left ? m : n;
    if(m < 0 || n < 0 || ldc < m || lda < nq)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((nq > 0 && !A) || (nq > 1 && !ipiv) || (m && n && !C))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_ormtr_unmtr_template(rocblas_handle handle,
                                              const rocblas_side side,
                                              const rocblas_fill uplo,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
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
    ROCSOLVER_ENTER("ormtr_unmtr", "side:", side, "uplo:", uplo, "trans:", trans, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "shiftC:", shiftC, "ldc:", ldc,
                    "bc:", batch_count);

    // quick return
    if(!n || !m || !batch_count)
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

    if(uplo == rocblas_fill_upper)
    {
        rocsolver_ormql_unmql_template<BATCHED, STRIDED, T>(
            handle, side, trans, rows, cols, nq - 1, A, shiftA + idx2D(0, 1, lda), lda, strideA,
            ipiv, strideP, C, shiftC, ldc, strideC, batch_count, scalars, AbyxORwork, diagORtmptr,
            trfact, workArr);
    }
    else
    {
        rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
            handle, side, trans, rows, cols, nq - 1, A, shiftA + idx2D(1, 0, lda), lda, strideA,
            ipiv, strideP, C, shiftC + idx2D(rowC, colC, ldc), ldc, strideC, batch_count, scalars,
            AbyxORwork, diagORtmptr, trfact, workArr);
    }

    return rocblas_status_success;
}

/** Adapts A and C to be of the same type **/
template <bool BATCHED, bool STRIDED, typename T>
rocblas_status rocsolver_ormtr_unmtr_template(rocblas_handle handle,
                                              const rocblas_side side,
                                              const rocblas_fill uplo,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
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

    return rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
        handle, side, uplo, trans, m, n, A, shiftA, lda, strideA, ipiv, strideP,
        cast2constType(workArr), shiftC, ldc, strideC, batch_count, scalars, AbyxORwork,
        diagORtmptr, trfact, workArr + batch_count);
}

ROCSOLVER_END_NAMESPACE
