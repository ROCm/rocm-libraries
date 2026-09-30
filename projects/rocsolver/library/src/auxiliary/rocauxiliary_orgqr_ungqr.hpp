/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocauxiliary_org2r_ung2r.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/** The columns j:j+jb-1 of the blocked algorithm (ORGQR_PANEL_*): with the block of reflectors
    H_j ... H_{j+jb-1} = I - V T V^H (V in A(j:m-1, j:j+jb-1), unit lower trapezoidal, T its
    triangular factor from larft), the columns are (I - V T V^H) [I; 0] = [I - V1 W; -V2 W], with
    W = T V1^H (upper triangular), V1 the first jb rows of V and V2 the others. This replaces the
    unblocked algorithm (org2r/ung2r) on these columns, which applies the jb reflectors one at a
    time. ORGQR_PANEL_W computes W, ORGQR_PANEL_TOP computes I - V1 W (into a separate buffer, as
    it overwrites V1), and ORGQR_PANEL_STORE writes the columns (the product V2 W is a gemm). **/
template <typename T>
__device__ inline T orgqr_conj(const T x)
{
    if constexpr(rocblas_is_complex<T>)
        return conj(x);
    else
        return x;
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void orgqr_panel_w(const I jb,
                                    U A,
                                    const rocblas_stride shiftV,
                                    const I lda,
                                    const rocblas_stride strideA,
                                    const T* Tf,
                                    const I ldt,
                                    const rocblas_stride strideT,
                                    T* W,
                                    const rocblas_stride strideW)
{
    const I b = hipBlockIdx_z;
    const I r = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I c = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if(r >= jb || c >= jb)
        return;
    const T* V = load_ptr_batch<T>(A, b, shiftV, strideA);
    const T* Tb = Tf + b * strideT;
    // W(r, c) = sum_{l = r..c} T(r, l) conj(V1(c, l)), V1(c, c) = 1
    T w = 0;
    for(I l = r; l <= c; l++)
        w += Tb[r + l * ldt] * (l == c ? T(1) : orgqr_conj(V[c + l * lda]));
    W[b * strideW + r + c * jb] = w;
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void orgqr_panel_top(const I jb,
                                      U A,
                                      const rocblas_stride shiftV,
                                      const I lda,
                                      const rocblas_stride strideA,
                                      const T* W,
                                      T* Q1,
                                      const rocblas_stride strideW)
{
    const I b = hipBlockIdx_z;
    const I r = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I c = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if(r >= jb || c >= jb)
        return;
    const T* V = load_ptr_batch<T>(A, b, shiftV, strideA);
    const T* Wb = W + b * strideW;
    // Q1(r, c) = delta(r, c) - sum_{l = 0..min(r, c)} V1(r, l) W(l, c), V1(r, r) = 1
    T q = (r == c) ? T(1) : T(0);
    for(I l = 0; l <= std::min(r, c); l++)
        q -= (l == r ? T(1) : V[r + l * lda]) * Wb[l + c * jb];
    Q1[b * strideW + r + c * jb] = q;
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void orgqr_panel_store(const I mj,
                                        const I jb,
                                        U A,
                                        const rocblas_stride shiftV,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const T* Q1,
                                        const T* P,
                                        const I ldp,
                                        const rocblas_stride strideW)
{
    const I b = hipBlockIdx_z;
    const I r = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I c = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if(r >= mj || c >= jb)
        return;
    T* V = load_ptr_batch<T>(A, b, shiftV, strideA);
    V[r + c * lda] = (r < jb) ? Q1[b * strideW + r + c * jb] : -P[b * strideW + (r - jb) + c * ldp];
}

/** ORGQR_PANEL_WORK_SIZE: entries of the workspace of the columns of a block (per matrix): W and
    I - V1 W (jb x jb each) and V2 W (m x jb). **/
template <typename I>
inline size_t orgqr_panel_work_size(const I m, const I jb)
{
    return size_t(2) * jb * jb + size_t(m) * jb;
}

template <bool BATCHED, typename T>
void rocsolver_orgqr_ungqr_getMemorySize(const rocblas_int m,
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
    rocsolver_org2r_ung2r_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars,
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

        // the columns of each block (see orgqr_panel_w)
        temp = sizeof(T) * orgqr_panel_work_size(m, jb) * batch_count;
        *size_Abyx_tmptr = *size_Abyx_tmptr >= temp ? *size_Abyx_tmptr : temp;

        // size of temporary array for triangular factor
        *size_trfact = sizeof(T) * jb * jb * batch_count;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename I = rocblas_int>
rocblas_status rocsolver_orgqr_ungqr_template(rocblas_handle handle,
                                              const I m,
                                              const I n,
                                              const I k,
                                              U A,
                                              const rocblas_stride shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              const I batch_count,
                                              T* scalars,
                                              T* work,
                                              T* Abyx_tmptr,
                                              T* trfact,
                                              T** workArr)
{
    ROCSOLVER_ENTER("orgqr_ungqr", "m:", m, "n:", n, "k:", k, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(!n || !m || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if the matrix is small, use the unblocked variant of the algorithm
    if(k <= xxGQx_xxGQx2_SWITCHSIZE)
        return rocsolver_org2r_ung2r_template<T>(handle, m, n, k, A, shiftA, lda, strideA, ipiv,
                                                 strideP, batch_count, scalars, Abyx_tmptr, workArr);

    I ldw = xxGQx_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) * ldw;

    // start of first blocked block
    I jb = ldw;
    I j = ((k - xxGQx_xxGQx2_SWITCHSIZE - 1) / jb) * jb;

    // start of the unblocked block
    I kk = std::min(k, j + jb);

    I blocksy, blocksx;

    // compute the unblockled part and set to zero the
    // corresponding top submatrix
    if(kk < n)
    {
        blocksx = (kk - 1) / BS2 + 1;
        blocksy = (n - kk - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocksx, blocksy, batch_count), dim3(BS2, BS2), 0,
                                stream, kk, n - kk, A, shiftA + idx2D(0, kk, lda), lda, strideA);

        rocsolver_org2r_ung2r_template<T>(handle, m - kk, n - kk, k - kk, A,
                                          shiftA + idx2D(kk, kk, lda), lda, strideA, (ipiv + kk),
                                          strideP, batch_count, scalars, Abyx_tmptr, workArr);
    }

    // compute the blocked part
    while(j >= 0)
    {
        // triangular factor of the block reflector
        rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_column_wise, (m - j),
                                    jb, A, shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j),
                                    strideP, trfact, ldw, strideW, batch_count, scalars, work,
                                    workArr);

        // first update the already computed part
        // applying the current block reflector using larfb
        if(j + jb < n)
        {
            rocsolver_larfb_template<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_forward_direction,
                rocblas_column_wise, m - j, n - j - jb, jb, A, shiftA + idx2D(j, j, lda), lda,
                strideA, trfact, 0, ldw, strideW, A, shiftA + idx2D(j, j + jb, lda), lda, strideA,
                batch_count, Abyx_tmptr, workArr);
        }

        // now compute the current block and set to zero
        // the corresponding top submatrix
        if(j > 0)
        {
            blocksx = (j - 1) / BS2 + 1;
            blocksy = (jb - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocksx, blocksy, batch_count), dim3(BS2, BS2),
                                    0, stream, j, jb, A, shiftA + idx2D(0, j, lda), lda, strideA);
        }
        // the columns of the block: [I - V1 W; -V2 W], W = T V1^H (see orgqr_panel_w)
        {
            const I mj = m - j;
            const rocblas_stride strideQ = orgqr_panel_work_size(m, jb);
            T* Wb = Abyx_tmptr;
            T* Q1 = Abyx_tmptr + jb * jb;
            T* P = Abyx_tmptr + 2 * jb * jb;
            const I bx = (jb - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL((orgqr_panel_w<T, I>), dim3(bx, bx, batch_count),
                                    dim3(BS2, BS2), 0, stream, jb, A, shiftA + idx2D(j, j, lda),
                                    lda, strideA, trfact, ldw, strideW, Wb, strideQ);
            ROCSOLVER_LAUNCH_KERNEL((orgqr_panel_top<T, I>), dim3(bx, bx, batch_count),
                                    dim3(BS2, BS2), 0, stream, jb, A, shiftA + idx2D(j, j, lda),
                                    lda, strideA, Wb, Q1, strideQ);
            if(mj > jb)
            {
                // (gemm kernels use scalars on host)
                rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);
                const T one = T(1);
                const T zero = T(0);
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, mj - jb, jb,
                               jb, &one, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, Wb, 0,
                               jb, strideQ, &zero, P, 0, m, strideQ, batch_count, workArr);
            }
            ROCSOLVER_LAUNCH_KERNEL((orgqr_panel_store<T, I>),
                                    dim3((mj - 1) / BS2 + 1, bx, batch_count), dim3(BS2, BS2), 0,
                                    stream, mj, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, Q1,
                                    P, m, strideQ);
        }

        j -= jb;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
