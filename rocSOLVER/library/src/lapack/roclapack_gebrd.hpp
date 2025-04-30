/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.8.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2017
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

#include "auxiliary/rocauxiliary_labrd.hpp"
#include "rocblas.hpp"
#include "roclapack_gebd2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_gebrd_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_X,
                                   size_t* size_Y)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_X = 0;
        *size_Y = 0;
        return;
    }

    if(m <= GEBRD_GEBD2_SWITCHSIZE || n <= GEBRD_GEBD2_SWITCHSIZE)
    {
        // requirements for calling a single GEBD2
        rocsolver_gebd2_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars,
                                                  size_work_workArr, size_Abyx_norms);
        *size_X = 0;
        *size_Y = 0;
    }

    else
    {
        size_t s1, s2, w1, w2, unused;
        rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;
        rocblas_int d = std::min(m / k, n / k);

        // sizes are maximum of what is required by GEBD2 and LABRD
        rocsolver_gebd2_getMemorySize<BATCHED, T>(m - d * k, n - d * k, batch_count, &unused, &w1,
                                                  &s1);
        rocsolver_labrd_getMemorySize<BATCHED, T>(m, n, k, batch_count, size_scalars, &w2, &s2);
        *size_work_workArr = std::max(w1, w2);
        *size_Abyx_norms = std::max(s1, s2);

        // size of matrix X
        *size_X = m * k;
        *size_X *= sizeof(T) * batch_count;

        // size of matrix Y
        *size_Y = n * k;
        *size_Y *= sizeof(T) * batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_gebrd_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tauq,
                                        const rocblas_stride strideQ,
                                        T* taup,
                                        const rocblas_stride strideP,
                                        T* X,
                                        const rocblas_int shiftX,
                                        const rocblas_int ldx,
                                        const rocblas_stride strideX,
                                        T* Y,
                                        const rocblas_int shiftY,
                                        const rocblas_int ldy,
                                        const rocblas_stride strideY,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms)
{
    ROCSOLVER_ENTER("gebrd", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    T minone = -1;
    T one = 1;
    rocblas_int nb = GEBRD_BLOCKSIZE;
    rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;
    rocblas_int dim = std::min(m, n); // total number of pivots
    rocblas_int jb, j = 0;
    rocblas_int blocks;

    // if the matrix is small, use the unblocked variant of the algorithm
    if(m <= k || n <= k)
        return rocsolver_gebd2_template<T>(handle, m, n, A, shiftA, lda, strideA, D, strideD, E,
                                           strideE, tauq, strideQ, taup, strideP, batch_count,
                                           scalars, work_workArr, Abyx_norms);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    while(j < dim - k)
    {
        // Reduce block to bidiagonal form
        jb = std::min(dim - j, nb); // number of rows and columns in the block
        rocsolver_labrd_template<T>(handle, m - j, n - j, jb, A, shiftA + idx2D(j, j, lda), lda,
                                    strideA, D + j, strideD, E + j, strideE, tauq + j, strideQ,
                                    taup + j, strideP, X, shiftX, ldx, strideX, Y, shiftY, ldy,
                                    strideY, batch_count, scalars, work_workArr, Abyx_norms);

        // update the rest of the matrix
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                       m - j - jb, n - j - jb, jb, &minone, A, shiftA + idx2D(j + jb, j, lda), lda,
                       strideA, Y, shiftY + jb, ldy, strideY, &one, A,
                       shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count,
                       (T**)work_workArr);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m - j - jb, n - j - jb,
                       jb, &minone, X, shiftX + jb, ldx, strideX, A, shiftA + idx2D(j, j + jb, lda),
                       lda, strideA, &one, A, shiftA + idx2D(j + jb, j + jb, lda), lda, strideA,
                       batch_count, (T**)work_workArr);

        blocks = (jb - 1) / 64 + 1;
        if(m >= n)
        {
            ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, blocks, 1),
                                    dim3(1, 64, 1), 0, stream, D, j, strideD, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, jb);
            ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, blocks, 1),
                                    dim3(1, 64, 1), 0, stream, E, j, strideE, A,
                                    shiftA + idx2D(j, j + 1, lda), lda, strideA, jb);
        }
        else
        {
            ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, blocks, 1),
                                    dim3(1, 64, 1), 0, stream, D, j, strideD, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, jb);
            ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, blocks, 1),
                                    dim3(1, 64, 1), 0, stream, E, j, strideE, A,
                                    shiftA + idx2D(j + 1, j, lda), lda, strideA, jb);
        }

        j += nb;
    }

    // factor last block
    if(j < dim)
        rocsolver_gebd2_template<T>(handle, m - j, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    D + j, strideD, E + j, strideE, tauq + j, strideQ, taup + j,
                                    strideP, batch_count, scalars, work_workArr, Abyx_norms);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
