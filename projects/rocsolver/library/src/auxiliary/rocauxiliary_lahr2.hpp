/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/**************************************************************************************/
/***************** Kernels/Device functions *******************************************/
/**************************************************************************************/

/***** Kernels to compute the column of Y *****/
/**********************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void lahr2_computeY_kernel(const rocblas_int mm,
                                            const rocblas_int k,
                                            const rocblas_int c,
                                            U AA,
                                            const rocblas_int shiftA,
                                            const rocblas_int lda,
                                            const rocblas_stride strideA,
                                            T* YA,
                                            const rocblas_int shiftY,
                                            const rocblas_int ldy,
                                            const rocblas_stride strideY,
                                            T* FA,
                                            const rocblas_int shiftF,
                                            const rocblas_int ldf,
                                            const rocblas_stride strideF,
                                            T* tauA,
                                            const rocblas_stride strideP)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Y = load_ptr_batch<T>(YA, bid, shiftY, strideY);
    T* F = load_ptr_batch<T>(FA, bid, shiftF, strideF);
    T* tau = tauA + bid * strideP;

    /* ------------------------
    formulate gemv problem:

        components:
            y  = Y(k:mm-1, c)
            A1 = A(k:mm-1, c+1:mm-1)
            A2 = Y(k:mm-1, 0:c-1)
            x1 = A(k+c:mm-k, c)
            x2 = F(0:c-1, c)
            t  = tau(c)

        operation:
            y = t * (A1 * x1 - A2 * x2)
    ------------------------ */
    int m = mm - k;
    int n = std::max((mm - k - c), c);
    T* y = Y + idx2D(k, c, ldy);
    T* A1 = A + idx2D(k, c + 1, lda);
    int lda1 = lda;
    T* A2 = Y + idx2D(k, 0, ldy);
    int lda2 = ldy;
    T* x1 = A + idx2D(k + c, c, lda);
    T* x2 = F + idx2D(0, c, ldf);
    T* t = tau + c;

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        ac = 0;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            // sx1 = (j < mm - k - c) ? x1[j] : 0;
            // sx2 = (j < c) ? x2[j] : 0;

            // operation for all rows
            if(i < m && j < mm - k - c)
                ac += A1[i + j * lda1] * x1[j];
            if(i < m && j < c)
                ac -= A2[i + j * lda2] * x2[j];
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            y[i] = ac * t[0];
    }
}

template <int MAX_THDS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) lahr2_scale_set_tau(const I j,
                                                                      U FA,
                                                                      const rocblas_stride shiftF,
                                                                      const rocblas_stride strideF,
                                                                      T* tauA,
                                                                      const rocblas_stride strideP)
{
    const auto bid = blockIdx.z;
    const auto tid = threadIdx.x;

    // select batch instance
    T* F = load_ptr_batch<T>(FA, bid, shiftF, strideF);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    const T t = *tau;

    I i;
    for(i = tid; i < j; i += MAX_THDS)
    {
        F[i] *= -t;
    }

    if(tid == j)
    {
        F[i] = t;
    }
}

template <bool BATCHED, typename T>
void rocsolver_lahr2_getMemorySize(const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int nb,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_norms,
                                   size_t* size_work_vec)
{
    // if quick return no workspace needed
    if(n <= 1 || nb == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_norms = 0;
        *size_work_vec = 0;
        return;
    }

    // size of scalars (constants) for rocblas calls
    *size_scalars = sizeof(T) * 3;

    size_t s1, s2;

    // size of array of pointers (batched cases)
    if(BATCHED)
        s1 = 2 * sizeof(T*) * batch_count;
    else
        s1 = 0;

    // extra requirements for calling larfg
    size_t larfg_norms;
    rocsolver_larfg_getMemorySize<T>(n - k, batch_count, &s2, &larfg_norms);

    // norms[] stores nb EI values per batch (subdiagonal elements displaced by larfg)
    *size_norms = std::max(larfg_norms, sizeof(T) * nb * batch_count);

    // work_workArr also used as trmv scratch (length nb per batch)
    *size_work_workArr = std::max({s1, s2, sizeof(T) * nb * batch_count});

    // separate w vector buffer (length nb per batch) for the update step trmv operations,
    // kept separate from Tmat to avoid rocblas aliasing checks
    *size_work_vec = sizeof(T) * nb * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_lahr2_argCheck(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int nb,
                                        const rocblas_int lda,
                                        const rocblas_int ldt,
                                        const rocblas_int ldy,
                                        U A,
                                        T* tau,
                                        T* Tmat,
                                        U Y,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    // n=0 or n=1: quick return, not an error (LAPACK: IF(N.LE.1) RETURN)
    if(n < 0 || k < 1 || nb < 1 || (n > 1 && (k >= n || nb > n - k)) || lda < std::max(1, n)
       || ldt < nb || ldy < std::max(1, n) || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    // n=0 is a quick return (no pointers needed)
    if(!n)
        return rocblas_status_success;
    if(!A || !tau || !Tmat || !Y)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_lahr2_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int nb,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* tau,
                                        const rocblas_stride strideT,
                                        T* Tmat,
                                        const rocblas_int ldt,
                                        const rocblas_stride strideN,
                                        U Y,
                                        const rocblas_int shiftY,
                                        const rocblas_int ldy,
                                        const rocblas_stride strideY,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* norms,
                                        T* work_vec)
{
    ROCSOLVER_ENTER("lahr2", "n:", n, "k:", k, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "ldt:", ldt, "shiftY:", shiftY, "ldy:", ldy, "bc:", batch_count);
    using S = decltype(std::real(T{}));

    // quick return (LAPACK: IF( N.LE.1 ) RETURN)
    if(n <= 1 || nb == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // scalars[0] = -1,  scalars[1] = 0,  scalars[2] = 1
    // work_vec: dedicated length-nb w vector per batch for the update step.
    // Kept separate from Tmat to avoid rocblas aliasing checks in trmv.
    // stride_work: per-batch stride for both work_vec and the trmv scratch (work_workArr).
    rocblas_stride stride_work = rocblas_stride(nb);
    rocblas_stride stride_norm = rocblas_stride(nb);

    // norms[] stores EI values: norms[j] = A(k+j, j) saved before larfg sets it to 1,
    // restored at the start of the next iteration (or after the loop for j=nb-1).

    // Grid/block for copy_mat (vector copy: m=j, n=1 -> simple 1D launch)
    // Computed inline where needed.

    // thread config for update Y kernel.
    rocblas_int thr_updates = BS2;
    rocblas_int thc_updates = BS2;
    rocblas_int grr_updates = (n - 1) / thr_updates + 1;
    rocblas_int grc_updates = 1;
    size_t lmemsize_updates = sizeof(T) * (thr_updates * thc_updates);

    // -----------------------------------------------------------------------
    // Main loop: i = 1..NB (LAPACK 1-based) -> j = 0..nb-1 (0-based)
    // All matrix indices below are 0-based. Mapping to LAPACK 1-based:
    //   LAPACK A(K+1, I)  ->  A[k + j,     j]      (diagonal pivot)
    //   LAPACK A(K+1, 1)  ->  A[k,         0]      (start of panel)
    //   LAPACK Y(K+1, I)  ->  Y[k,         j]
    // -----------------------------------------------------------------------
    for(rocblas_int j = 0; j < nb; ++j)
    {
        if(j > 0)
        {
            // ----------------------------------------------------------------
            // Update A(k:n-1, j):
            //
            // (a) A(k:n-1, j) -= Y(k:n-1, 0:j-1) * conj(A(k+j-1, 0:j-1))
            //     LAPACK: DGEMV('N', N-K, I-1, -ONE, Y(K+1,1), LDY, A(K+I-1,1), LDA, ONE, A(K+1,I), 1)
            // ----------------------------------------------------------------
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(k + j - 1, 0, lda), lda,
                                            strideA, batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - k, j,
                                cast2constType<T>(scalars), 0, Y, shiftY + idx2D(k, 0, ldy), ldy,
                                strideY, A, shiftA + idx2D(k + j - 1, 0, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(k, j, lda), 1,
                                strideA, batch_count, (T**)work_workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(k + j - 1, 0, lda), lda,
                                            strideA, batch_count);

            // ----------------------------------------------------------------
            // (b) Apply (I - V*T^H*V^H) to A(k:n-1, j) from the left.
            //     Uses Tmat(:, nb-1) as workspace w (length j).
            //
            //     V = A(k:n-1, 0:j-1), split as:
            //       V1 = A(k:k+j-1, 0:j-1)   -- unit lower triangular
            //       V2 = A(k+j:n-1, 0:j-1)
            //     b = A(k:n-1, j) = [b1; b2]
            //
            //     w  = V1^H * b1         (DCOPY + DTRMV lower ^H unit)
            //     w += V2^H * b2         (DGEMV ^T)
            //     w  = T^H * w           (DTRMV upper ^H non-unit)
            //     b2 -= V2 * w           (DGEMV no-trans)
            //     b1 -= V1 * w           (DTRMV lower no-trans unit + DAXPY)
            // ----------------------------------------------------------------

            // w = b1  (DCOPY: copy A(k:k+j-1, j) -> work_vec[0:j-1])
            {
                rocblas_int grid_x = (j - 1) / 64 + 1;
                ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U, T*>), dim3(grid_x, 1, batch_count),
                                        dim3(64, 1, 1), 0, stream, j, 1, A, shiftA + idx2D(k, j, lda),
                                        1, strideA, work_vec, 0, 1, stride_work);
            }

            // w = V1^H * w  (DTRMV lower ^H unit, in-place on work_vec)
            rocblasCall_trmv<T>(handle, rocblas_fill_lower, rocblas_operation_conjugate_transpose,
                                rocblas_diagonal_unit, j, A, shiftA + idx2D(k, 0, lda), lda,
                                strideA, work_vec, 0, 1, stride_work, (T*)work_workArr, stride_work,
                                batch_count);

            // w += V2^H * b2
            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - k - j, j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(k + j, 0, lda),
                                lda, strideA, A, shiftA + idx2D(k + j, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 2), 0, work_vec, 0, 1, stride_work,
                                batch_count, (T**)work_workArr);

            // w = T^H * w  (DTRMV upper ^H non-unit)
            rocblasCall_trmv<T>(handle, rocblas_fill_upper, rocblas_operation_conjugate_transpose,
                                rocblas_diagonal_non_unit, j, Tmat, 0, ldt, strideN, work_vec, 0, 1,
                                stride_work, (T*)work_workArr, stride_work, batch_count);

            // b2 -= V2 * w
            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - k - j, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(k + j, 0, lda),
                                lda, strideA, work_vec, 0, 1, stride_work,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(k + j, j, lda),
                                1, strideA, batch_count, (T**)work_workArr);

            // w = V1 * w  (DTRMV lower no-trans unit, in-place on work_vec)
            rocblasCall_trmv<T>(handle, rocblas_fill_lower, rocblas_operation_none,
                                rocblas_diagonal_unit, j, A, shiftA + idx2D(k, 0, lda), lda,
                                strideA, work_vec, 0, 1, stride_work, (T*)work_workArr, stride_work,
                                batch_count);

            // b1 -= w  (DAXPY with alpha = -1)
            rocblasCall_axpy<T>(handle, j, cast2constType<T>(scalars), 0, work_vec, 0, 1,
                                stride_work, A, shiftA + idx2D(k, j, lda), 1, strideA, batch_count);

            // Restore A(k+j-1, j-1) = EI saved from the previous iteration
            // LAPACK: A(K+I-1, I-1) = EI
            ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), dim3(batch_count, 1, 1), dim3(1, 1, 1), 0,
                                    stream, (S*)norms, j - 1, stride_norm, A,
                                    shiftA + idx2D(k + j - 1, j - 1, lda), lda, strideA,
                                    (rocblas_int)1);
        }

        // --------------------------------------------------------------------
        // Generate Householder reflector H(j+1) to annihilate A(k+j+1:n-1, j)
        //  - Save EI = A(k+j, j) into norms[j], then set A(k+j, j) = 1
        // --------------------------------------------------------------------
        rocsolver_larfg_template(handle, n - k - j, A, shiftA + idx2D(k + j, j, lda), (S*)norms, j,
                                 stride_norm, A, shiftA + idx2D(std::min(k + j + 1, n - 1), j, lda),
                                 1, strideA, tau + j, strideT, batch_count, (T*)work_workArr, norms);

        // --------------------------------------------------------------------
        // Compute Y(k:n-1, j)
        // --------------------------------------------------------------------
        if(j > 0)
        {
            // T(0:j-1, j) = A(k+j:n-1, 0:j-1)^H * A(k+j:n-1, j)
            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - k - j, j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(k + j, 0, lda),
                                lda, strideA, A, shiftA + idx2D(k + j, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, Tmat, idx2D(0, j, ldt), 1,
                                strideN, batch_count, (T**)work_workArr);
        }

        // Y(k:n-1, j) = t * (A(k:n-1, j+1:n-k) * A(k+j:n-1, j) - Y(k:n-1, 0:j-1) * T(0:j-1, j))
        ROCSOLVER_LAUNCH_KERNEL(lahr2_computeY_kernel<T>, dim3(grr_updates, grc_updates, batch_count),
                                dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream, n, k,
                                j, A, shiftA, lda, strideA, Y, shiftY, ldy, strideY, Tmat, 0, ldt,
                                strideN, tau, strideT);

        // --------------------------------------------------------------------
        // Compute T(0:j, j)
        // --------------------------------------------------------------------

        // T(0:j-1, j) *= -tau(j) and T(j, j) = tau(j)
        ROCSOLVER_LAUNCH_KERNEL((lahr2_scale_set_tau<BS1, T>), dim3(1, 1, batch_count),
                                dim3(BS1, 1, 1), 0, stream, j, Tmat, idx2D(0, j, ldt), strideN,
                                tau + j, strideT);
        if(j > 0)
        {
            // T(0:j-1, j) = T(0:j-1, 0:j-1) * T(0:j-1, j)  (upper non-unit)
            rocblasCall_trmv<T>(handle, rocblas_fill_upper, rocblas_operation_none,
                                rocblas_diagonal_non_unit, j, Tmat, 0, ldt, strideN, Tmat,
                                idx2D(0, j, ldt), 1, strideN, (T*)work_workArr, stride_work,
                                batch_count);
        }
    }

    // Restore A(k+nb-1, nb-1) = EI  (LAPACK: A(K+NB, NB) = EI, 1-based)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                            (S*)norms, nb - 1, stride_norm, A,
                            shiftA + idx2D(k + nb - 1, nb - 1, lda), lda, strideA, (rocblas_int)1);

    // ------------------------------------------------------------------------
    // Compute Y(0:k-1, 0:nb-1)  (top k rows of Y, LAPACK post-loop block)
    // LAPACK:
    //   DLACPY('ALL', K, NB, A(1,2), LDA, Y, LDY)
    //   DTRMM('R','L','N','U', K, NB, ONE, A(K+1,1), LDA, Y, LDY)
    //   if N > K+NB:
    //     DGEMM('N','N', K, NB, N-K-NB, ONE, A(1,2+NB), LDA, A(K+1+NB,1), LDA, ONE, Y, LDY)
    //   DTRMM('R','U','N','N', K, NB, ONE, T, LDT, Y, LDY)
    // ------------------------------------------------------------------------
    if(k > 0)
    {
        // Y(0:k-1, 0:nb-1) = A(0:k-1, 1:nb)
        {
            rocblas_int bx = (k - 1) / 32 + 1;
            rocblas_int by = (nb - 1) / 32 + 1;
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U, T*>), dim3(bx, by, batch_count),
                                    dim3(32, 32, 1), 0, stream, k, nb, A, shiftA + idx2D(0, 1, lda),
                                    lda, strideA, Y, shiftY, ldy, strideY);
        }

        // rocblas_internal_trmm_template may recursively call rocblas_internal_gemm_64 with
        // host-side beta constants (&beta_1<T>). If the handle is in device pointer mode those
        // host addresses are misinterpreted as device pointers, corrupting results.  Switch to
        // host pointer mode for these trmm (and gemm) calls and use a host alpha constant.
        static const T one = T(1);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

        // Y(0:k-1,:) *= V1  (right trmm: Y = Y * V1, V1 = A(k:k+nb-1, 0:nb-1) lower unit)
        rocblasCall_trmm<T>(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                            rocblas_diagonal_unit, k, nb, &one, 0, A, shiftA + idx2D(k, 0, lda),
                            lda, strideA, Y, shiftY, ldy, strideY, batch_count, (T**)work_workArr);

        if(n > k + nb)
        {
            // Y(0:k-1,:) += A(0:k-1, nb+1:n-1) * A(k+nb:n-1, 0:nb-1)
            rocsolver_gemm<T>(handle, rocblas_operation_none, rocblas_operation_none, k, nb,
                              n - k - nb, &one, A, shiftA + idx2D(0, nb + 1, lda), lda, strideA, A,
                              shiftA + idx2D(k + nb, 0, lda), lda, strideA, &one, Y, shiftY, ldy,
                              strideY, batch_count, (T**)work_workArr);
        }

        // Y(0:k-1,:) *= T  (right trmm: Y = Y * T, upper non-unit)
        rocblasCall_trmm<T>(handle, rocblas_side_right, rocblas_fill_upper, rocblas_operation_none,
                            rocblas_diagonal_non_unit, k, nb, &one, 0, Tmat, 0, ldt, strideN, Y,
                            shiftY, ldy, strideY, batch_count, (T**)work_workArr);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
