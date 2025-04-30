/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
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

#include "auxiliary/rocauxiliary_larfb.hpp"
#include "auxiliary/rocauxiliary_larft.hpp"
#include "rocblas.hpp"
#include "roclapack_geqr2.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T, typename I>
void rocsolver_geqrf_getMemorySize(const I m,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms_trfact,
                                   size_t* size_diag_tmptr,
                                   size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms_trfact = 0;
        *size_diag_tmptr = 0;
        *size_workArr = 0;
        return;
    }

    if(m <= GEQxF_GEQx2_SWITCHSIZE || n <= GEQxF_GEQx2_SWITCHSIZE)
    {
        // requirements for a single GEQR2 call
        rocsolver_geqr2_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars, size_work_workArr,
                                                  size_Abyx_norms_trfact, size_diag_tmptr);
        *size_workArr = 0;
    }
    else
    {
        size_t w1, w2, unused, s1, s2;
        I jb = GEQxF_BLOCKSIZE;

        // size to store the temporary triangular factor
        *size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

        // requirements for calling GEQR2 with sub blocks
        rocsolver_geqr2_getMemorySize<BATCHED, T>(m, jb, batch_count, size_scalars, &w1, &s2, &s1);
        *size_Abyx_norms_trfact = std::max(s2, *size_Abyx_norms_trfact);

        // requirements for calling LARFT
        rocsolver_larft_getMemorySize<BATCHED, T>(m, jb, batch_count, &unused, &w2, size_workArr);

        // requirements for calling LARFB
        rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, m, n - jb, jb, batch_count,
                                                  &s2, &unused);

        *size_work_workArr = std::max(w1, w2);
        *size_diag_tmptr = std::max(s1, s2);

        // size of workArr is double to accommodate
        // LARFB's TRMM calls in the batched case
        if(BATCHED)
            *size_workArr *= 2;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_geqrf_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms_trfact,
                                        T* diag_tmptr,
                                        T** workArr)
{
    ROCSOLVER_ENTER("geqrf", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    if(m <= GEQxF_GEQx2_SWITCHSIZE || n <= GEQxF_GEQx2_SWITCHSIZE)
    {
        rocsolver_geqr2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count,
                                    scalars, work_workArr, Abyx_norms_trfact, diag_tmptr);
        return rocblas_status_success;
    }

    I dim = std::min(m, n); // total number of pivots
    I jb, j = 0;

    I nb = GEQxF_BLOCKSIZE;
    I ldw = GEQxF_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) * ldw;

    while(j < dim - GEQxF_GEQx2_SWITCHSIZE)
    {
        // Factor diagonal and subdiagonal blocks
        jb = std::min(dim - j, nb); // number of columns in the block
        rocsolver_geqr2_template<T>(handle, m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    (ipiv + j), strideP, batch_count, scalars, work_workArr,
                                    Abyx_norms_trfact, diag_tmptr);

        // apply transformation to the rest of the matrix
        if(j + jb < n)
        {
            // compute block reflector
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_column_wise,
                                        m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                        (ipiv + j), strideP, Abyx_norms_trfact, ldw, strideW,
                                        batch_count, scalars, (T*)work_workArr, workArr);

            // apply the block reflector
            rocsolver_larfb_template<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                rocblas_forward_direction, rocblas_column_wise, m - j, n - j - jb, jb, A,
                shiftA + idx2D(j, j, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW, A,
                shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count, diag_tmptr, workArr);
        }
        j += nb;
    }

    // factor last block
    if(j < dim)
        rocsolver_geqr2_template<T>(handle, m - j, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                    strideA, (ipiv + j), strideP, batch_count, scalars,
                                    work_workArr, Abyx_norms_trfact, diag_tmptr);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
