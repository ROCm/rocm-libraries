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

#include "auxiliary/rocauxiliary_lahr2.hpp"
#include "auxiliary/rocauxiliary_larfb.hpp"
#include "rocblas.hpp"
#include "roclapack_gehd2.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T, typename I>
void rocsolver_gehrd_getMemorySize(const I n,
                                   const I ilo,
                                   const I ihi,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_norms_tmptr,
                                   size_t* size_diag_beta,
                                   size_t* size_F,
                                   size_t* size_work_vec,
                                   size_t* size_Y)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_norms_tmptr = 0;
        *size_diag_beta = 0;
        *size_F = 0;
        *size_work_vec = 0;
        *size_Y = 0;
        return;
    }

    const I nb = GEHRD_BLOCKSIZE;
    const I dim = ihi - ilo;

    // workspace from gehd2 (unblocked fallback)
    size_t w_d2, n_d2, d_d2;
    rocsolver_gehd2_getMemorySize<BATCHED, T>(n, ilo, ihi, batch_count, size_scalars, &w_d2, &n_d2,
                                              &d_d2);

    if(dim <= GEHRD_GEHD2_SWITCHSIZE)
    {
        // only unblocked path
        *size_work_workArr = w_d2;
        *size_norms_tmptr = n_d2;
        *size_diag_beta = d_d2;
        *size_F = 0;
        *size_work_vec = 0;
        *size_Y = 0;
        return;
    }

    // workspace from lahr2 (panel reduction)
    size_t w_lh, n_lh, wv_lh, b_lh;
    rocsolver_lahr2_getMemorySize<BATCHED, T>(n, ilo, nb, batch_count, size_scalars, &w_lh, &n_lh,
                                              &wv_lh, &b_lh);

    // workspace from larfb (left application to trailing submatrix)
    // first block: m = ihi-i-1, n_cols = n-i-ib (i = ilo-1, ib = nb)
    size_t tmptr_lb, wArr_lb;
    rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, dim, n - ilo - nb + 1, nb,
                                              batch_count, &tmptr_lb, &wArr_lb);

    // work_workArr is shared among gehd2, lahr2 work_workArr and larfb workArr
    *size_work_workArr = std::max({w_d2, w_lh, wArr_lb});

    // norms_tmptr: gehd2 Abyx_norms, lahr2 norms, larfb tmptr
    *size_norms_tmptr = std::max({n_d2, n_lh, tmptr_lb});

    // diag_beta: gehd2 diag, lahr2 beta, and buffer for set/restore_diag
    *size_diag_beta = std::max({d_d2, b_lh, sizeof(T) * batch_count});

    // F: nb x nb upper-triangular block reflector factor per batch (lahr2 output)
    *size_F = sizeof(T) * nb * nb * batch_count;

    // work_vec: lahr2 work_vec
    *size_work_vec = wv_lh;

    // Y: ihi x nb workspace per batch (Y = A*V*T output from lahr2)
    *size_Y = sizeof(T) * ihi * nb * batch_count;
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_gehrd_argCheck(rocblas_handle handle,
                                        const I n,
                                        const I ilo,
                                        const I ihi,
                                        const I lda,
                                        T A,
                                        U ipiv,
                                        const I batch_count = 1)
{
    return rocsolver_gehd2_argCheck(handle, n, ilo, ihi, lda, A, ipiv, batch_count);
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_gehrd_template(rocblas_handle handle,
                                        const I n,
                                        const I ilo,
                                        const I ihi,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* norms_tmptr,
                                        void* diag_beta,
                                        T* F,
                                        T* work_vec,
                                        T* Y)
{
    ROCSOLVER_ENTER("gehrd", "n:", n, "ilo:", ilo, "ihi:", ihi, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    const I dim = ihi - ilo;
    const I nb = GEHRD_BLOCKSIZE;

    // if the active submatrix is small, use the unblocked algorithm directly
    if(dim <= GEHRD_GEHD2_SWITCHSIZE)
        return rocsolver_gehd2_template<T>(handle, n, ilo, ihi, A, shiftA, lda, strideA, ipiv,
                                           strideP, batch_count, scalars, work_workArr, norms_tmptr,
                                           diag_beta);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // F: nb x nb upper-triangular block reflector factor, column-major with ldf = nb
    const I ldf = nb;
    const rocblas_stride strideF = rocblas_stride(nb) * nb;

    // Y: ihi x nb workspace per batch (Y = A*V*T from lahr2), ldy = ihi
    const I ldy = ihi;
    const rocblas_stride strideY = rocblas_stride(ihi) * nb;

    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    const T one = T(1);
    const T minone = T(-1);

    I i = ilo - 1;
    while(i < ihi - 1 - GEHRD_GEHD2_SWITCHSIZE)
    {
        I ib = std::min(static_cast<I>(nb), static_cast<I>(ihi - 1 - i));

        // Reduce columns i:i+ib-1 to Hessenberg form and generate matrix Y = A * V * T
        rocsolver_lahr2_template<T>(handle, ihi, i + 1, ib, A, shiftA + idx2D(0, i, lda), lda,
                                    strideA, ipiv + i, strideP, F, ldf, strideF, Y, 0, ldy, strideY,
                                    batch_count, scalars, work_workArr, norms_tmptr, work_vec,
                                    (T*)diag_beta);

        // Apply H from right to A(0:ihi-1, i+ib:ihi-1):  A -= Y * V^H (V = A(i+ib:ihi, i:i+ib-1))
        // Temporarily set A(i+ib, i+ib-1) = 1 (unit lower triangular V convention).
        constexpr rocblas_int DIAG_NTHREADS = BS1;
        ROCSOLVER_LAUNCH_KERNEL((set_diag<T, I, T, U>), dim3(batch_count, 1, 1),
                                dim3(1, DIAG_NTHREADS, 1), 0, stream, (T*)diag_beta, 0,
                                (rocblas_stride)1, A, shiftA + idx2D(i + ib, i + ib - 1, lda), lda,
                                strideA, 1, true);

        rocsolver_gemm<T>(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                          ihi, ihi - i - ib, ib, &minone, Y, 0, ldy, strideY, A,
                          shiftA + idx2D(i + ib, i, lda), lda, strideA, &one, A,
                          shiftA + idx2D(0, i + ib, lda), lda, strideA, batch_count,
                          (T**)work_workArr);

        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, I, T, U>), dim3(batch_count, 1, 1),
                                dim3(1, DIAG_NTHREADS, 1), 0, stream, (T*)diag_beta, 0,
                                (rocblas_stride)1, A, shiftA + idx2D(i + ib, i + ib - 1, lda), lda,
                                strideA, 1);

        // Apply H from right to A(0:i, i+1:i+ib-1)
        // Y(0:i, 0:ib-2) = Y(0:i, 0:ib-2) * A(i+1:i+ib-1, i:i+ib-2)^H
        // A(0:i, i+j+1) -= Y(0:i, j+1)  for j = 0..ib-2
        rocblasCall_trmm<T>(handle, rocblas_side_right, rocblas_fill_lower,
                            rocblas_operation_conjugate_transpose, rocblas_diagonal_unit, i + 1,
                            ib - 1, &one, 0, A, shiftA + idx2D(i + 1, i, lda), lda, strideA, Y, 0,
                            ldy, strideY, batch_count, (T**)work_workArr);

        for(I j = 0; j < ib - 1; ++j)
        {
            rocblasCall_axpy<T>(handle, i + 1, &minone, 0, Y, rocblas_stride(j) * ldy, 1, strideY,
                                A, shiftA + idx2D(0, i + j + 1, lda), 1, strideA, batch_count,
                                (T**)work_workArr);
        }

        // Apply H^H from left to A(i+1:ihi-1, i+ib:n-1)
        rocsolver_larfb_template<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
            rocblas_forward_direction, rocblas_column_wise, ihi - i - 1, n - i - ib, ib, A,
            shiftA + idx2D(i + 1, i, lda), lda, strideA, F, 0, ldf, strideF, A,
            shiftA + idx2D(i + 1, i + ib, lda), lda, strideA, batch_count, norms_tmptr,
            (T**)work_workArr);

        i += ib;
    }

    // reduce the remaining columns with the unblocked algorithm
    if(i < ihi - 1)
        rocsolver_gehd2_template<T>(handle, n, i + 1, ihi, A, shiftA, lda, strideA, ipiv, strideP,
                                    batch_count, scalars, work_workArr, norms_tmptr, diag_beta);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
