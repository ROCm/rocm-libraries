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

#include "rocauxiliary_lacgv.hpp"
#include "rocauxiliary_larf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_orm2l_unm2l_getMemorySize(const rocblas_side side,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int k,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_Abyx,
                                         size_t* size_diag,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_Abyx = 0;
        *size_diag = 0;
        *size_workArr = 0;
        return;
    }

    // size of temporary array for diagonal elements
    *size_diag = sizeof(T) * batch_count;

    // memory requirements to call larf
    rocsolver_larf_getMemorySize<BATCHED, T>(side, m, n, batch_count, size_scalars, size_Abyx,
                                             size_workArr);
}

template <bool COMPLEX, typename T, typename U>
rocblas_status rocsolver_orm2l_ormql_argCheck(rocblas_handle handle,
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
    bool left = (side == rocblas_side_left);

    // 2. invalid size
    if(m < 0 || n < 0 || k < 0 || ldc < m)
        return rocblas_status_invalid_size;
    if(left && (lda < m || k > m))
        return rocblas_status_invalid_size;
    if(!left && (lda < n || k > n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !C) || (k && !ipiv) || (left && m && k && !A) || (!left && n && k && !A))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_orm2l_unm2l_template(rocblas_handle handle,
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
                                              T* Abyx,
                                              T* diag,
                                              T** workArr)
{
    ROCSOLVER_ENTER("orm2l_unm2l", "side:", side, "trans:", trans, "m:", m, "n:", n, "k:", k,
                    "shiftA:", shiftA, "lda:", lda, "shiftC:", shiftC, "ldc:", ldc,
                    "bc:", batch_count);

    // quick return
    if(!n || !m || !k || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // determine limits and indices
    bool left = (side == rocblas_side_left);
    bool transpose = (trans != rocblas_operation_none);
    rocblas_int start, step, nq, ncol, nrow;
    if(left)
    {
        nq = m;
        ncol = n;
        if(!transpose)
        {
            start = -1;
            step = 1;
        }
        else
        {
            start = k;
            step = -1;
        }
    }
    else
    {
        nq = n;
        nrow = m;
        if(!transpose)
        {
            start = k;
            step = -1;
        }
        else
        {
            start = -1;
            step = 1;
        }
    }

    // conjugate tau
    if(COMPLEX && transpose)
        rocsolver_lacgv_template<T>(handle, k, ipiv, 0, 1, strideP, batch_count);

    rocblas_int i;
    for(rocblas_int j = 1; j <= k; ++j)
    {
        i = start + step * j; // current householder vector
        if(left)
        {
            nrow = m - k + i + 1;
        }
        else
        {
            ncol = n - k + i + 1;
        }

        // insert one in A(nq-k+i,i), i.e. the i-th element of the (nq-k)-th
        // subdiagonal, to build/apply the householder matrix
        ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                                0, stream, diag, 0, 1, A, shiftA + idx2D(nq - k + i, i, lda), lda,
                                strideA, 1, true);

        // Apply current Householder reflector
        rocsolver_larf_template(handle, side, nrow, ncol, A, shiftA + idx2D(0, i, lda), 1, strideA,
                                (ipiv + i), strideP, C, shiftC, ldc, strideC, batch_count, scalars,
                                Abyx, workArr);

        // restore original value of A(nq-k+i,i)
        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                dim3(1, 1, 1), 0, stream, diag, 0, 1, A,
                                shiftA + idx2D(nq - k + i, i, lda), lda, strideA, 1);
    }

    // restore tau
    if(COMPLEX && transpose)
        rocsolver_lacgv_template<T>(handle, k, ipiv, 0, 1, strideP, batch_count);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
