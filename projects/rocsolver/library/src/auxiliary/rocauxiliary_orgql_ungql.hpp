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

#include "rocauxiliary_larfb.hpp"
#include "rocauxiliary_larft.hpp"
#include "rocauxiliary_org2l_ung2l.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_orgql_ungql_getMemorySize(const rocblas_int m,
                                         const rocblas_int n,
                                         const rocblas_int k,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_Abyx_tmptr,
                                         size_t* size_trfact,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_Abyx_tmptr = 0;
        *size_trfact = 0;
        *size_workArr = 0;
        return;
    }

    size_t temp, unused;
    rocsolver_org2l_ung2l_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars,
                                                    size_Abyx_tmptr, size_workArr);

    if(k <= xxGQx_xxGQx2_SWITCHSIZE)
    {
        *size_work = 0;
        *size_trfact = 0;
    }

    else
    {
        rocblas_int jb = xxGQx_BLOCKSIZE;
        rocblas_int j = ((k - xxGQx_xxGQx2_SWITCHSIZE - 1) / jb) * jb;
        rocblas_int kk = std::min(k, j + jb);

        // size of workspace is maximum of what is needed by larft and larfb.
        // size of Abyx_tmptr is maximum of what is needed by org2r/ung2r and larfb.
        rocsolver_larft_getMemorySize<BATCHED, T>(m, jb, batch_count, &unused, size_work, &unused);
        rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, m, n - jb, jb, batch_count,
                                                  &temp, &unused);

        *size_Abyx_tmptr = *size_Abyx_tmptr >= temp ? *size_Abyx_tmptr : temp;

        // size of temporary array for triangular factor
        *size_trfact = sizeof(T) * jb * jb * batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_orgql_ungql_template(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* work,
                                              T* Abyx_tmptr,
                                              T* trfact,
                                              T** workArr)
{
    ROCSOLVER_ENTER("orgql_ungql", "m:", m, "n:", n, "k:", k, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if the matrix is small, use the unblocked variant of the algorithm
    if(k <= xxGQx_xxGQx2_SWITCHSIZE)
        return rocsolver_org2l_ung2l_template<T>(handle, m, n, k, A, shiftA, lda, strideA, ipiv,
                                                 strideP, batch_count, scalars, Abyx_tmptr, workArr);

    rocblas_int ldw = xxGQx_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) * ldw;

    // size of unblocked part
    rocblas_int jb = ldw;
    rocblas_int kk = std::min(k, ((k - xxGQx_xxGQx2_SWITCHSIZE + jb - 1) / jb) * jb);

    // start of first blocked block is j + n - k = n - kk
    rocblas_int j = k - kk;

    rocblas_int blocksy, blocksx;

    // compute the unblocked part and set to zero the
    // corresponding bottom submatrix
    if(kk < m)
    {
        blocksx = (kk - 1) / 32 + 1;
        blocksy = (n - kk - 1) / 32 + 1;
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocksx, blocksy, batch_count), dim3(32, 32), 0,
                                stream, kk, n - kk, A, shiftA + idx2D(m - kk, 0, lda), lda, strideA);

        rocsolver_org2l_ung2l_template<T>(handle, m - kk, n - kk, k - kk, A, shiftA, lda, strideA,
                                          ipiv, strideP, batch_count, scalars, Abyx_tmptr, workArr);
    }

    // compute the blocked part
    while(j < k)
    {
        // first update the already computed part
        // applying the current block reflector using larft + larfb
        if(n - k + j > 0)
        {
            rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_column_wise,
                                        m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                                        lda, strideA, (ipiv + j), strideP, trfact, ldw, strideW,
                                        batch_count, scalars, work, workArr);

            rocsolver_larfb_template<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_backward_direction,
                rocblas_column_wise, m - k + j + jb, n - k + j, jb, A,
                shiftA + idx2D(0, n - k + j, lda), lda, strideA, trfact, 0, ldw, strideW, A, shiftA,
                lda, strideA, batch_count, Abyx_tmptr, workArr);
        }

        // now compute the current block and set to zero
        // the corresponding bottom submatrix
        if(j > 0)
        {
            blocksx = (k - j - jb - 1) / 32 + 1;
            blocksy = (jb - 1) / 32 + 1;
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocksx, blocksy, batch_count), dim3(32, 32),
                                    0, stream, k - j - jb, jb, A,
                                    shiftA + idx2D(m - k + j + jb, n - k + j, lda), lda, strideA);
        }
        rocsolver_org2l_ung2l_template<T>(handle, m - k + j + jb, jb, jb, A,
                                          shiftA + idx2D(0, n - k + j, lda), lda, strideA, (ipiv + j),
                                          strideP, batch_count, scalars, Abyx_tmptr, workArr);

        j += jb;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
