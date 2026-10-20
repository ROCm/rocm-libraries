/****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * and
 *     Joffrain, Low, Quintana-Orti, et al. (2006). Accumulating householder
 *     transformations, revisited.
 *     ACM Transactions on Mathematical Software 32(2), p. 169-179.
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************** Main kernels *********************************************************/
/**************************************************************************************/

template <typename T, typename I, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_triangular(const I n,
                                     const I k,
                                     U V,
                                     const rocblas_stride shiftV,
                                     const I ldv,
                                     const rocblas_stride strideV,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const I ldf,
                                     const rocblas_stride strideF,
                                     const rocblas_direct direct,
                                     const rocblas_storev storev,
                                     const bool add_fp,
                                     const I k1)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    // if k1 < k, the off-diagonal blocks (i < k1 <= j or j < k1 <= i) are left to larft_offdiag
    if(i < k && j < k && (i < k1) == (j < k1))
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
            Fp[idx2D(j, i, ldf)] = tp[i];
        else if(direct == rocblas_forward_direction)
        {
            if(j < i)
            {
                if(storev == rocblas_column_wise)
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(i, j, ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(i, j, ldv)]);
                    }
                }
                else
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(j, i, ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(j, i, ldv)]);
                    }
                }
            }
            else
                Fp[idx2D(j, i, ldf)] = 0;
        }
        else
        {
            if(j > i)
            {
                if(storev == rocblas_column_wise)
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D((n - k + i), j, ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D((n - k + i), j, ldv)]);
                    }
                }
                else
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(j, (n - k + i), ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(j, (n - k + i), ldv)]);
                    }
                }
            }
            else
                Fp[idx2D(j, i, ldf)] = 0;
        }
    }
}

template <typename T, typename I, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_triangular(const I n,
                                     const I k,
                                     U V,
                                     const rocblas_stride shiftV,
                                     const I ldv,
                                     const rocblas_stride strideV,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const I ldf,
                                     const rocblas_stride strideF,
                                     const rocblas_direct direct,
                                     const rocblas_storev storev,
                                     const bool add_fp,
                                     const I k1)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    // if k1 < k, the off-diagonal blocks (i < k1 <= j or j < k1 <= i) are left to larft_offdiag
    if(i < k && j < k && (i < k1) == (j < k1))
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
            Fp[idx2D(j, i, ldf)] = tp[i];
        else if(direct == rocblas_forward_direction)
        {
            if(j < i)
            {
                if(storev == rocblas_column_wise)
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * conj(Vp[idx2D(i, j, ldv)]);
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + conj(Vp[idx2D(i, j, ldv)]));
                    }
                }
                else
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(j, i, ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(j, i, ldv)]);
                    }
                }
            }
            else
                Fp[idx2D(j, i, ldf)] = 0;
        }
        else
        {
            if(j > i)
            {
                if(storev == rocblas_column_wise)
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * conj(Vp[idx2D((n - k + i), j, ldv)]);
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + conj(Vp[idx2D((n - k + i), j, ldv)]));
                    }
                }
                else
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(j, (n - k + i), ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(j, (n - k + i), ldv)]);
                    }
                }
            }
            else
                Fp[idx2D(j, i, ldf)] = 0;
        }
    }
}

template <typename T, typename I>
ROCSOLVER_KERNEL void set_tau(const I k, T* tau, const rocblas_stride strideT)
{
    const auto b = hipBlockIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < k)
    {
        T* tp = tau + b * strideT;
        tp[i] = -tp[i];
    }
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void larft_kernel_forward(const rocblas_storev storev,
                                           const I n,
                                           const I k,
                                           const I k1,
                                           U VA,
                                           const rocblas_stride shiftV,
                                           const I ldv,
                                           const rocblas_stride strideV,
                                           T* tauA,
                                           const rocblas_stride strideT,
                                           T* FA,
                                           const I ldfA,
                                           const rocblas_stride strideF)
{
    const I bid = hipBlockIdx_y;
    const I tid = hipThreadIdx_x;
    const I tid_inc = hipBlockDim_x;

    // if k1 < k, thread-block 0 computes the diagonal block of T of the reflectors 0:k1-1, and
    // thread-block 1 that of the reflectors k1:k-1
    const I k0 = (hipBlockIdx_x == 0) ? 0 : k1;
    const I kb = (hipBlockIdx_x == 0) ? k1 : k - k1;
    I nb = n - k0;

    // select batch instance
    T* V = load_ptr_batch<T>(VA, bid, shiftV + idx2D(k0, k0, ldv), strideV);
    T* tau = tauA + bid * strideT + k0;
    T* Ftemp = FA + bid * strideF + idx2D(k0, k0, ldfA);

    // shared memory setup (work uses the strictly lower triangular part of F)
    extern __shared__ double lmem[];
    T* F = reinterpret_cast<T*>(lmem);
    T* work = F + 1;
    I ldf = kb;

    // copy F to shared memory
    for(I i = tid; i < kb; i += tid_inc)
        for(I j = i; j < kb; j++)
            F[i + j * ldf] = Ftemp[i + j * ldfA];
    __syncthreads();

    // if T is split, rows kb:nb-1 of V (columns if row-wise) are full in all the reflectors of
    // thread-block 0: add their products in parallel
    if(nb > kb)
    {
        for(I e = tid; e < kb * kb; e += tid_inc)
        {
            const I i = e % kb;
            const I j = e / kb;
            if(i < j)
            {
                T temp = 0;
                if(storev == rocblas_column_wise)
                    for(I r = kb; r < nb; r++)
                        temp += conj(V[r + i * ldv]) * V[r + j * ldv];
                else
                    for(I r = kb; r < nb; r++)
                        temp += V[i + r * ldv] * conj(V[j + r * ldv]);
                F[i + j * ldf] += tau[j] * temp;
            }
        }
        nb = kb;
        __syncthreads();
    }

    // --------- MAIN BODY ---------
    for(I kk = 1; kk < kb; kk++)
    {
        const I mm = kk;
        const I nn = nb - 1 - kk;

        T* Fx = F + kk * ldf;

        // compute the matrix vector product, using the householder vectors
        if(storev == rocblas_column_wise)
        {
            T* Vm = V + (kk + 1);
            T* Vx = V + (kk + 1) + kk * ldv;

            // gemv (conjugate transpose)
            for(I i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(I j = 0; j < nn; j++)
                    temp += conj(Vm[j + i * ldv]) * Vx[j];
                work[i] = tau[kk] * temp + Fx[i];
            }
        }
        else
        {
            T* Vm = V + (kk + 1) * ldv;
            T* Vx = V + kk + (kk + 1) * ldv;

            // gemv (no transpose)
            for(I i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(I j = 0; j < nn; j++)
                    temp += Vm[i + j * ldv] * conj(Vx[j * ldv]);
                work[i] = tau[kk] * temp + Fx[i];
            }
        }

        __syncthreads();

        // multiply by previous triangular factor
        // trmv (no transpose)
        for(I i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(I j = i; j < mm; j++)
                temp += F[i + j * ldf] * work[j];
            Fx[i] = temp;
        }

        __syncthreads();
    }

    // copy shared memory back to F
    for(I i = tid; i < kb; i += tid_inc)
        for(I j = i; j < kb; j++)
            Ftemp[i + j * ldfA] = F[i + j * ldf];
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void larft_kernel_backward(const rocblas_storev storev,
                                            const I n,
                                            const I k,
                                            const I k1,
                                            U VA,
                                            const rocblas_stride shiftV,
                                            const I ldv,
                                            const rocblas_stride strideV,
                                            T* tauA,
                                            const rocblas_stride strideT,
                                            T* FA,
                                            const I ldfA,
                                            const rocblas_stride strideF)
{
    const I bid = hipBlockIdx_y;
    const I tid = hipThreadIdx_x;
    const I tid_inc = hipBlockDim_x;

    // if k1 < k, thread-block 0 computes the diagonal block of T of the reflectors 0:k1-1, and
    // thread-block 1 that of the reflectors k1:k-1
    const I k0 = (hipBlockIdx_x == 0) ? 0 : k1;
    const I kb = (hipBlockIdx_x == 0) ? k1 : k - k1;
    I nb = n - (k - k0 - kb);

    // select batch instance
    T* V = load_ptr_batch<T>(
        VA, bid, shiftV + ((storev == rocblas_column_wise) ? idx2D(0, k0, ldv) : idx2D(k0, 0, ldv)),
        strideV);
    T* tau = tauA + bid * strideT + k0;
    T* Ftemp = FA + bid * strideF + idx2D(k0, k0, ldfA);

    // shared memory setup (work uses the strictly upper triangular part of F)
    extern __shared__ double lmem[];
    T* F = reinterpret_cast<T*>(lmem);
    I ldf = kb;
    T* work = F + (kb - 1) * ldf;

    // copy F to shared memory
    for(I i = tid; i < kb; i += tid_inc)
        for(I j = 0; j <= i; j++)
            F[i + j * ldf] = Ftemp[i + j * ldfA];
    __syncthreads();

    // if T is split, rows 0:nb-kb-1 of V (columns if row-wise) are full in all the reflectors
    // of thread-block 1: add their products in parallel
    if(nb > kb)
    {
        const I nr = nb - kb;
        for(I e = tid; e < kb * kb; e += tid_inc)
        {
            const I i = e % kb;
            const I j = e / kb;
            if(i > j)
            {
                T temp = 0;
                if(storev == rocblas_column_wise)
                    for(I r = 0; r < nr; r++)
                        temp += conj(V[r + i * ldv]) * V[r + j * ldv];
                else
                    for(I r = 0; r < nr; r++)
                        temp += V[i + r * ldv] * conj(V[j + r * ldv]);
                F[i + j * ldf] += tau[j] * temp;
            }
        }
        V += (storev == rocblas_column_wise) ? nr : nr * ldv;
        nb = kb;
        __syncthreads();
    }

    // --------- MAIN BODY ---------
    for(I kk = kb - 2; kk >= 0; kk--)
    {
        const I mm = kb - kk - 1;
        const I nn = nb - kb + kk;

        T* Fm = F + (kk + 1) + (kk + 1) * ldf;
        T* Fx = F + (kk + 1) + kk * ldf;

        // compute the matrix vector product, using the householder vectors
        if(storev == rocblas_column_wise)
        {
            T* Vm = V + (kk + 1) * ldv;
            T* Vx = V + kk * ldv;

            // gemv (conjugate transpose)
            for(I i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(I j = 0; j < nn; j++)
                    temp += conj(Vm[j + i * ldv]) * Vx[j];
                work[i] = tau[kk] * temp + Fx[i];
            }
        }
        else
        {
            T* Vm = V + (kk + 1);
            T* Vx = V + kk;

            // gemv (no transpose)
            for(I i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(I j = 0; j < nn; j++)
                    temp += Vm[i + j * ldv] * conj(Vx[j * ldv]);
                work[i] = tau[kk] * temp + Fx[i];
            }
        }

        __syncthreads();

        // multiply by previous triangular factor
        // trmv (no transpose)
        for(I i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(I j = 0; j <= i; j++)
                temp += Fm[i + j * ldf] * work[j];
            Fx[i] = temp;
        }

        __syncthreads();
    }

    // copy shared memory back to F
    for(I i = tid; i < kb; i += tid_inc)
        for(I j = 0; j <= i; j++)
            Ftemp[i + j * ldfA] = F[i + j * ldf];
}

/** LARFT_OFFDIAG_LEFT and LARFT_OFFDIAG_RIGHT complete T when its diagonal blocks T1
    (reflectors 0:k1-1, V1) and T2 (reflectors k1:k-1, V2) were computed separately: the
    off-diagonal block is -T1 * (V1^H V2) * T2 (forward direction) or -T2 * (V2^H V1) * T1
    (backward direction), with V1 V2^H and V2 V1^H instead if row-wise. Each thread-block of
    LARFT_OFFDIAG_LEFT forms a column of the product of the reflectors (adding the unit diagonal
    and the triangular part of V to the product with the rest of V, already in F if add_fp), and
    multiplies it by the triangular factor on the left; it also zeros the opposite block of F.
    Each thread-block of LARFT_OFFDIAG_RIGHT multiplies a row of the result by the triangular
    factor on the right. **/
template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LARFT_SWITCHSIZE)
    larft_offdiag_left(const rocblas_direct direct,
                       const rocblas_storev storev,
                       const I n,
                       const I k,
                       const I k1,
                       U VA,
                       const rocblas_stride shiftV,
                       const I ldv,
                       const rocblas_stride strideV,
                       T* FA,
                       const I ldf,
                       const rocblas_stride strideF,
                       const bool add_fp)
{
    __shared__ T x[LARFT_SWITCHSIZE];
    const I bid = hipBlockIdx_y;
    const I tid = hipThreadIdx_x;
    const bool forward = (direct == rocblas_forward_direction);

    // column q of the off-diagonal block, with rows p0:p1-1; the unit diagonal entry of reflector
    // q is in row (or column) rd of V, and the rest of its triangular part in rows r0:r1-1
    const I q = hipBlockIdx_x + (forward ? k1 : 0);
    const I p0 = forward ? 0 : k1;
    const I p1 = forward ? k1 : k;
    const I rd = forward ? q : n - k + q;
    const I r0 = forward ? q + 1 : n - k;
    const I r1 = forward ? k : n - k + q;

    T* V = load_ptr_batch<T>(VA, bid, shiftV, strideV);
    T* F = FA + bid * strideF;

    for(I p = p0 + tid; p < p1; p += hipBlockDim_x)
    {
        T temp;
        if(storev == rocblas_column_wise)
        {
            temp = conj(V[idx2D(rd, p, ldv)]);
            for(I r = r0; r < r1; r++)
                temp += conj(V[idx2D(r, p, ldv)]) * V[idx2D(r, q, ldv)];
        }
        else
        {
            temp = V[idx2D(p, rd, ldv)];
            for(I r = r0; r < r1; r++)
                temp += V[idx2D(p, r, ldv)] * conj(V[idx2D(q, r, ldv)]);
        }
        x[p - p0] = add_fp ? F[idx2D(p, q, ldf)] + temp : temp;
        F[idx2D(q, p, ldf)] = 0;
    }
    __syncthreads();

    // multiply by T1 (upper triangular) or T2 (lower triangular)
    for(I p = p0 + tid; p < p1; p += hipBlockDim_x)
    {
        T temp = 0;
        for(I l = (forward ? p : p0); l < (forward ? p1 : p + 1); l++)
            temp += F[idx2D(p, l, ldf)] * x[l - p0];
        F[idx2D(p, q, ldf)] = temp;
    }
}

template <typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(LARFT_SWITCHSIZE)
    larft_offdiag_right(const rocblas_direct direct,
                        const I k,
                        const I k1,
                        T* FA,
                        const I ldf,
                        const rocblas_stride strideF)
{
    __shared__ T x[LARFT_SWITCHSIZE];
    const I bid = hipBlockIdx_y;
    const I tid = hipThreadIdx_x;
    const bool forward = (direct == rocblas_forward_direction);

    // row p of the off-diagonal block, with columns q0:q1-1
    const I p = hipBlockIdx_x + (forward ? 0 : k1);
    const I q0 = forward ? k1 : 0;
    const I q1 = forward ? k : k1;

    T* F = FA + bid * strideF;

    for(I q = q0 + tid; q < q1; q += hipBlockDim_x)
        x[q - q0] = F[idx2D(p, q, ldf)];
    __syncthreads();

    // multiply by -T2 (upper triangular) or -T1 (lower triangular)
    for(I q = q0 + tid; q < q1; q += hipBlockDim_x)
    {
        T temp = 0;
        for(I l = (forward ? q0 : q); l < (forward ? q + 1 : q1); l++)
            temp += x[l - q0] * F[idx2D(l, q, ldf)];
        F[idx2D(p, q, ldf)] = -temp;
    }
}

/******************* Host functions *********************************************/
/*******************************************************************************/

/** LARFT_GRAM_PARTIAL and LARFT_GRAM_SUM compute the strictly upper triangular part of
    V2^H V2 (V2: the rows u1_n:n-1 of the k columns of V, forward direction, column-wise; the
    only part of the product that larft uses) for large n, where the matrix product with a
    k x k result and a long inner dimension would run on few compute units. Each thread-block of
    LARFT_GRAM_PARTIAL computes, for a chunk of LARFT_SPLITK_ROWS rows (through shared memory, in
    tiles of rows), the partial products of a group of BS * NPT pairs (i, j), i < j (the third
    grid dimension runs over the groups), into the workspace P (k x k per chunk);
    LARFT_GRAM_SUM adds the chunks in a fixed order (so that the result is deterministic) into F. **/
template <int BS, int NPT, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) larft_gram_partial(const I rows,
                                                               const I k,
                                                               U VA,
                                                               const rocblas_stride shiftV,
                                                               const I ldv,
                                                               const rocblas_stride strideV,
                                                               T* P,
                                                               const rocblas_stride strideP)
{
    constexpr int LDSE = 2048; // entries of the tile in shared memory (32 KB for complex double)
    static_assert(LDSE >= LARFT_SPLITK_MAXK, "a row of V2 must fit in the tile");
    __shared__ T tile[LDSE];

    const I b = hipBlockIdx_y;
    const I chunk = hipBlockIdx_x;
    const I tid = hipThreadIdx_x;
    const T* V = load_ptr_batch<T>(VA, b, shiftV, strideV);
    const I r0 = chunk * I(LARFT_SPLITK_ROWS);
    const I r1 = std::min(rows, r0 + I(LARFT_SPLITK_ROWS));
    I tr = std::max(I(1), I(LDSE) / k); // rows per tile
    if(tr > 1 && tr % 2 == 0)
        tr--; // odd column stride: no bank conflicts between the columns of the tile
    const I npairs = k * (k - 1) / 2;

    // the pairs (i, j), i < j, of this thread: e = e0 + tid + q * BS, in column order
    const I e0 = hipBlockIdx_z * I(BS * NPT);
    I pi[NPT], pj[NPT];
    T acc[NPT];
#pragma unroll
    for(int q = 0; q < NPT; q++)
    {
        acc[q] = T(0);
        I e = e0 + tid + q * BS;
        I j = 1;
        while(j < k && e >= j)
        {
            e -= j;
            j++;
        }
        pi[q] = e;
        pj[q] = j;
    }

    for(I t0 = r0; t0 < r1; t0 += tr)
    {
        const I nt = std::min(tr, r1 - t0);
        __syncthreads();
        for(I e = tid; e < nt * k; e += BS)
        {
            const I r = e % nt;
            const I c = e / nt;
            tile[r + c * tr] = V[(t0 + r) + c * size_t(ldv)];
        }
        __syncthreads();
#pragma unroll
        for(int q = 0; q < NPT; q++)
        {
            if(e0 + tid + q * BS < npairs)
            {
                const T* vi = tile + pi[q] * tr;
                const T* vj = tile + pj[q] * tr;
                T a = acc[q];
                for(I r = 0; r < nt; r++)
                {
                    if constexpr(rocblas_is_complex<T>)
                        a += conj(vi[r]) * vj[r];
                    else
                        a += vi[r] * vj[r];
                }
                acc[q] = a;
            }
        }
    }

    T* Pb = P + b * strideP + chunk * size_t(k) * k;
#pragma unroll
    for(int q = 0; q < NPT; q++)
        if(e0 + tid + q * BS < npairs)
            Pb[pi[q] + pj[q] * k] = acc[q];
}

template <typename T, typename I>
ROCSOLVER_KERNEL void larft_gram_sum(const I k,
                                     const I nchunks,
                                     const T* P,
                                     const rocblas_stride strideP,
                                     T* F,
                                     const I ldf,
                                     const rocblas_stride strideF)
{
    const I b = hipBlockIdx_y;
    const I e = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I i = e % k;
    const I j = e / k;
    if(j >= k || i >= j)
        return;
    const T* Pb = P + b * strideP;
    T a = 0;
    for(I c = 0; c < nchunks; c++)
        a += Pb[c * size_t(k) * k + i + j * k];
    F[b * strideF + i + j * ldf] = a;
}

/** LARFT_SPLITK returns whether larft computes V2^H V2 with larft_gram_partial (forward,
    column-wise, k <= LARFT_SPLITK_MAXK, at least LARFT_SPLITK_MIN rows in V2), and the number
    of chunks. **/
template <typename I>
inline bool larft_splitk(const rocblas_direct direct,
                         const rocblas_storev storev,
                         const I n,
                         const I k,
                         I* nchunks = nullptr)
{
    const bool use = direct == rocblas_forward_direction && storev == rocblas_column_wise
        && k <= I(LARFT_SPLITK_MAXK) && n - k >= I(LARFT_SPLITK_MIN);
    if(nchunks)
        *nchunks = use ? (n - k - 1) / I(LARFT_SPLITK_ROWS) + 1 : 0;
    return use;
}

template <bool BATCHED, typename T, typename I>
void rocsolver_larft_getMemorySize(const I n,
                                   const I k,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_workArr)
{
    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_workArr = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of re-usable workspace (and of the partial products of larft_gram_partial, in the
    // forward column-wise case with many rows). Callers may size the workspace once and then
    // call larft with fewer rows or columns, so cover any call with at most n rows and k
    // columns: at most LARFT_SPLITK_MAXK columns and n - 1 rows in V2.
    *size_work = sizeof(T) * k * batch_count;
    if(n - 1 >= I(LARFT_SPLITK_MIN))
    {
        const size_t ks = std::min(k, I(LARFT_SPLITK_MAXK));
        const size_t nchunks = (n - 2) / I(LARFT_SPLITK_ROWS) + 1;
        *size_work = std::max(*size_work, sizeof(T) * nchunks * ks * ks * batch_count);
    }

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_larft_argCheck(rocblas_handle handle,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const I n,
                                        const I k,
                                        const I ldv,
                                        const I ldf,
                                        T V,
                                        U tau,
                                        U F)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(direct != rocblas_backward_direction && direct != rocblas_forward_direction)
        return rocblas_status_invalid_value;
    if(storev != rocblas_column_wise && storev != rocblas_row_wise)
        return rocblas_status_invalid_value;
    bool row = (storev == rocblas_row_wise);

    // 2. invalid size
    if(n < 0 || k < 1 || ldf < k)
        return rocblas_status_invalid_size;
    if((row && ldv < k) || (!row && ldv < n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !V) || !tau || !F)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larft_template(rocblas_handle handle,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const I n,
                                        const I k,
                                        U V,
                                        const rocblas_stride shiftV,
                                        const I ldv,
                                        const rocblas_stride strideV,
                                        T* tau,
                                        const rocblas_stride strideT,
                                        T* F,
                                        const I ldf,
                                        const rocblas_stride strideF,
                                        const I batch_count,
                                        T* scalars,
                                        T* work,
                                        T** workArr)
{
    ROCSOLVER_ENTER("larft", "direct:", direct, "storev:", storev, "n:", n, "k:", k,
                    "shiftV:", shiftV, "ldv:", ldv, "ldf:", ldf, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // gemm kernels use scalars on host
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    rocblas_stride stridew = rocblas_stride(k);
    rocblas_diagonal diag = rocblas_diagonal_non_unit;
    rocblas_fill uplo;
    rocblas_operation trans;

    const bool use_gemm = n > k;
    const T zero = T(0);
    const T one = T(1);

    const I u1_n = use_gemm ? k : n;
    const I u2_n = use_gemm ? n - k : 0;

    // Compute T=V2'*V2 or V2*V2' (V'=[V1' V2'] where V1 is triangular and V is trapezoidal)
    // SYRK/HERK can be used alternatively, but GEMM is currently more performant.
    if(use_gemm)
    {
        I nchunks;
        if(larft_splitk(direct, storev, n, k, &nchunks))
        {
            const rocblas_stride strideP = rocblas_stride(nchunks) * k * k;
            constexpr int GBS = 128;
            constexpr int GNPT = 16;
            const I npg = (k * (k - 1) / 2 - 1) / (GBS * GNPT) + 1;
            ROCSOLVER_LAUNCH_KERNEL((larft_gram_partial<GBS, GNPT, T>),
                                    dim3(static_cast<uint32_t>(nchunks),
                                         static_cast<uint32_t>(batch_count),
                                         static_cast<uint32_t>(npg)),
                                    dim3(GBS), 0, stream, u2_n, k, V, shiftV + idx2D(u1_n, 0, ldv),
                                    ldv, strideV, work, strideP);
            ROCSOLVER_LAUNCH_KERNEL((larft_gram_sum<T>),
                                    dim3(static_cast<uint32_t>((k * k - 1) / 256 + 1),
                                         static_cast<uint32_t>(batch_count)),
                                    dim3(256), 0, stream, k, nchunks, (const T*)work, strideP, F,
                                    ldf, strideF);
        }
        else if(direct == rocblas_forward_direction && storev == rocblas_column_wise)
        {
            rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, k,
                           k, u2_n, &one, V, shiftV + idx2D(u1_n, 0, ldv), ldv, strideV, V,
                           shiftV + idx2D(u1_n, 0, ldv), ldv, strideV, &zero, F, 0, ldf, strideF,
                           batch_count, workArr);
        }
        else if(direct == rocblas_backward_direction && storev == rocblas_column_wise)
        {
            rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, k,
                           k, u2_n, &one, V, shiftV, ldv, strideV, V, shiftV, ldv, strideV, &zero,
                           F, 0, ldf, strideF, batch_count, workArr);
        }
        else if(direct == rocblas_forward_direction && storev == rocblas_row_wise)
        {
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, k,
                           k, u2_n, &one, V, shiftV + idx2D(0, u1_n, ldv), ldv, strideV, V,
                           shiftV + idx2D(0, u1_n, ldv), ldv, strideV, &zero, F, 0, ldf, strideF,
                           batch_count, workArr);
        }
        else if(direct == rocblas_backward_direction && storev == rocblas_row_wise)
        {
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, k,
                           k, u2_n, &one, V, shiftV, ldv, strideV, V, shiftV, ldv, strideV, &zero,
                           F, 0, ldf, strideF, batch_count, workArr);
        }
    }

    // The fused kernels larft_kernel_forward/backward keep the k x k triangular factor in shared
    // memory. For LARFT_SWITCHSIZE < k <= 2 * LARFT_SWITCHSIZE, two thread-blocks of the fused
    // kernel compute the diagonal blocks of T (reflectors 0:k1-1 and k1:k-1), and
    // larft_offdiag_left/right the off-diagonal block. Otherwise, use rocBLAS for each column of T.
    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    const bool fused = k <= LARFT_SWITCHSIZE && sizeof(T) * k * k <= props->sharedMemPerBlock;
    const I k1 = fused ? k : (k + 1) / 2;
    const bool split = !fused && k > 1 && k <= 2 * LARFT_SWITCHSIZE && n >= k
        && sizeof(T) * k1 * k1 <= props->sharedMemPerBlock;
    const size_t lmemsize = sizeof(T) * k1 * k1;

    // Fix diagonal of T, make zero the not used triangular part,
    // setup tau (changing signs) and account for the non-stored 1's on the
    // householder vectors
    I blocks = (k - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL((set_triangular<T, I, U>),
                            dim3(static_cast<uint32_t>(blocks), static_cast<uint32_t>(blocks),
                                 static_cast<uint32_t>(batch_count)),
                            dim3(BS2, BS2), 0, stream, n, k, V, shiftV, ldv, strideV, tau, strideT,
                            F, ldf, strideF, direct, storev, use_gemm, split ? k1 : k);
    ROCSOLVER_LAUNCH_KERNEL((set_tau<T, I>),
                            dim3(static_cast<uint32_t>(blocks), static_cast<uint32_t>(batch_count)),
                            dim3(BS2, 1), 0, stream, k, tau, strideT);

    // Remaining kernels take scalars on device.
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    if(direct == rocblas_forward_direction)
    {
        uplo = rocblas_fill_upper;

        // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS
        //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
        //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
        //      ZERO ENTRIES ****

        if(fused || split)
        {
            ROCSOLVER_LAUNCH_KERNEL((larft_kernel_forward<T, I, U>), dim3(split ? 2 : 1, batch_count),
                                    dim3(BS1, 1), lmemsize, stream, storev, u1_n, k, k1, V, shiftV,
                                    ldv, strideV, tau, strideT, F, ldf, strideF);
            if(split)
            {
                ROCSOLVER_LAUNCH_KERNEL(
                    (larft_offdiag_left<T, I, U>),
                    dim3(static_cast<uint32_t>(k - k1), static_cast<uint32_t>(batch_count)),
                    dim3(LARFT_SWITCHSIZE), 0, stream, direct, storev, n, k, k1, V, shiftV, ldv,
                    strideV, F, ldf, strideF, use_gemm);
                ROCSOLVER_LAUNCH_KERNEL(
                    (larft_offdiag_right<T, I>),
                    dim3(static_cast<uint32_t>(k1), static_cast<uint32_t>(batch_count)),
                    dim3(LARFT_SWITCHSIZE), 0, stream, direct, k, k1, F, ldf, strideF);
            }
        }
        else
        {
            for(I i = 1; i < k; ++i)
            {
                // compute the matrix vector product, using the householder vectors
                if(storev == rocblas_column_wise)
                {
                    trans = rocblas_operation_conjugate_transpose;
                    rocblasCall_gemv<T>(handle, trans, u1_n - 1 - i, i, tau + i, strideT, V,
                                        shiftV + idx2D(i + 1, 0, ldv), ldv, strideV, V,
                                        shiftV + idx2D(i + 1, i, ldv), 1, strideV, scalars + 2, 0,
                                        F, idx2D(0, i, ldf), 1, strideF, batch_count, workArr);
                }
                else
                {
                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - i - 1, V,
                                                    shiftV + idx2D(i, i + 1, ldv), ldv, strideV,
                                                    batch_count);

                    trans = rocblas_operation_none;
                    rocblasCall_gemv<T>(handle, trans, i, u1_n - 1 - i, tau + i, strideT, V,
                                        shiftV + idx2D(0, i + 1, ldv), ldv, strideV, V,
                                        shiftV + idx2D(i, i + 1, ldv), ldv, strideV, scalars + 2, 0,
                                        F, idx2D(0, i, ldf), 1, strideF, batch_count, workArr);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - i - 1, V,
                                                    shiftV + idx2D(i, i + 1, ldv), ldv, strideV,
                                                    batch_count);
                }

                // multiply by the previous triangular factor
                trans = rocblas_operation_none;
                rocblasCall_trmv<T>(handle, uplo, trans, diag, i, F, 0, ldf, strideF, F,
                                    idx2D(0, i, ldf), 1, strideF, work, stridew, batch_count);
            }
        }
    }
    else
    {
        uplo = rocblas_fill_lower;

        // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS
        //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
        //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
        //      ZERO ENTRIES ****

        if(fused || split)
        {
            auto shiftU2 = shiftV
                + ((storev == rocblas_column_wise) ? idx2D(u2_n, 0, ldv) : idx2D(0, u2_n, ldv));
            ROCSOLVER_LAUNCH_KERNEL((larft_kernel_backward<T, I, U>),
                                    dim3(split ? 2 : 1, batch_count), dim3(BS1, 1), lmemsize,
                                    stream, storev, u1_n, k, k1, V, shiftU2, ldv, strideV, tau,
                                    strideT, F, ldf, strideF);
            if(split)
            {
                ROCSOLVER_LAUNCH_KERNEL(
                    (larft_offdiag_left<T, I, U>),
                    dim3(static_cast<uint32_t>(k1), static_cast<uint32_t>(batch_count)),
                    dim3(LARFT_SWITCHSIZE), 0, stream, direct, storev, n, k, k1, V, shiftV, ldv,
                    strideV, F, ldf, strideF, use_gemm);
                ROCSOLVER_LAUNCH_KERNEL(
                    (larft_offdiag_right<T, I>),
                    dim3(static_cast<uint32_t>(k - k1), static_cast<uint32_t>(batch_count)),
                    dim3(LARFT_SWITCHSIZE), 0, stream, direct, k, k1, F, ldf, strideF);
            }
        }
        else
        {
            for(I i = k - 2; i >= 0; --i)
            {
                // compute the matrix vector product, using the householder vectors
                if(storev == rocblas_column_wise)
                {
                    trans = rocblas_operation_conjugate_transpose;
                    rocblasCall_gemv<T>(handle, trans, u1_n - k + i, k - i - 1, tau + i, strideT, V,
                                        shiftV + idx2D(u2_n, i + 1, ldv), ldv, strideV, V,
                                        shiftV + idx2D(u2_n, i, ldv), 1, strideV, scalars + 2, 0, F,
                                        idx2D(i + 1, i, ldf), 1, strideF, batch_count, workArr);
                }
                else
                {
                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - k + i, V, shiftV + idx2D(i, 0, ldv),
                                                    ldv, strideV, batch_count);

                    trans = rocblas_operation_none;
                    rocblasCall_gemv<T>(handle, trans, k - i - 1, u1_n - k + i, tau + i, strideT, V,
                                        shiftV + idx2D(i + 1, u2_n, ldv), ldv, strideV, V,
                                        shiftV + idx2D(i, u2_n, ldv), ldv, strideV, scalars + 2, 0,
                                        F, idx2D(i + 1, i, ldf), 1, strideF, batch_count, workArr);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - k + i, V, shiftV + idx2D(i, 0, ldv),
                                                    ldv, strideV, batch_count);
                }

                // multiply by the previous triangular factor
                trans = rocblas_operation_none;
                rocblasCall_trmv<T>(handle, uplo, trans, diag, k - i - 1, F,
                                    idx2D(i + 1, i + 1, ldf), ldf, strideF, F, idx2D(i + 1, i, ldf),
                                    1, strideF, work, stridew, batch_count);
            }
        }
    }

    // restore tau
    ROCSOLVER_LAUNCH_KERNEL(set_tau, dim3(blocks, batch_count), dim3(BS2, 1), 0, stream, k, tau,
                            strideT);

    return rocblas_status_success;
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void larft_set_tri(const rocblas_fill uplo,
                                    const I k,
                                    U A,
                                    const rocblas_stride shiftA,
                                    const I lda,
                                    const rocblas_stride strideA,
                                    T* buffer)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const I ldb = k;
    const rocblas_stride strideB = rocblas_stride(ldb) * k;

    const bool upper = (uplo == rocblas_fill_upper);
    const bool lower = (uplo == rocblas_fill_lower);

    if(i < k && j < k)
    {
        if((upper && j >= i) || (lower && i >= j))
        {
            T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
            T* Bp = &buffer[b * strideB];

            // copy A to buffer
            Bp[i + j * ldb] = Ap[i + j * lda];

            // set A to unit triangular
            Ap[i + j * lda] = (i == j) ? 1 : 0;
        }
    }
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void larft_restore_tri(const rocblas_fill uplo,
                                        const I k,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* buffer)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const I ldb = k;
    const rocblas_stride strideB = rocblas_stride(ldb) * k;

    const bool upper = (uplo == rocblas_fill_upper);
    const bool lower = (uplo == rocblas_fill_lower);

    if(i < k && j < k)
    {
        if((upper && j >= i) || (lower && i >= j))
        {
            T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
            T* Bp = &buffer[b * strideB];

            // copy buffer to A
            Ap[i + j * lda] = Bp[i + j * ldb];
        }
    }
}

template <typename T, typename I>
ROCSOLVER_KERNEL void larft_set_diag(I k,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const I ldf,
                                     const rocblas_stride strideF)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < k)
    {
        T *tp, *Fp;
        tp = tau + b * strideT;
        Fp = F + b * strideF;

        Fp[i + i * ldf] = 1 / tp[i];
    }
}

template <bool BATCHED, typename T, typename I>
void rocsolver_larft_inverse_getMemorySize(const I n,
                                           const I k,
                                           const I batch_count,
                                           size_t* size_work,
                                           size_t* size_workArr)
{
    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
    {
        *size_work = 0;
        *size_workArr = 0;
        return;
    }

    // size of re-usable workspace
    *size_work = sizeof(T) * k * k * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;
}

template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larft_inverse_template(rocblas_handle handle,
                                                const rocblas_direct direct,
                                                const rocblas_storev storev,
                                                const I n,
                                                const I k,
                                                U V,
                                                const rocblas_stride shiftV,
                                                const I ldv,
                                                const rocblas_stride strideV,
                                                T* tau,
                                                const rocblas_stride strideT,
                                                T* F,
                                                const I ldf,
                                                const rocblas_stride strideF,
                                                const I batch_count,
                                                T* work,
                                                T** workArr)
{
    ROCSOLVER_ENTER("larft_inverse", "direct:", direct, "storev:", storev, "n:", n, "k:", k,
                    "shiftV:", shiftV, "ldv:", ldv, "ldf:", ldf, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    T one = 1;
    T zero = 0;

    const bool colwise = (storev == rocblas_column_wise);
    const bool forward = (direct == rocblas_forward_direction);

    rocblas_operation transA
        = colwise ? rocblas_operation_conjugate_transpose : rocblas_operation_none;
    rocblas_operation transB
        = colwise ? rocblas_operation_none : rocblas_operation_conjugate_transpose;

    I tri_offset;
    rocblas_fill tri_uplo;

    if(colwise)
    {
        tri_uplo = forward ? rocblas_fill_upper : rocblas_fill_lower;
        tri_offset = (!forward && n > k) ? idx2D(n - k, 0, ldv) : 0;
    }
    else
    {
        tri_uplo = forward ? rocblas_fill_lower : rocblas_fill_upper;
        tri_offset = (!forward && n > k) ? idx2D(0, n - k, ldv) : 0;
    }

    I blocks = (k - 1) / BS2 + 1;
    dim3 gridTri(blocks, blocks, batch_count);
    dim3 blockTri(BS2, BS2);

    // set V to unit triangular/trapezoidal
    ROCSOLVER_LAUNCH_KERNEL((larft_set_tri<T, I, U>), gridTri, blockTri, 0, stream, tri_uplo, k, V,
                            shiftV + tri_offset, ldv, strideV, work);

    // compute: V' * V or V * V'
    rocsolver_gemm(handle, transA, transB, k, k, n, &one, V, shiftV, ldv, strideV, V, shiftV, ldv,
                   strideV, &zero, F, 0, ldf, strideF, batch_count, workArr);

    // set F diag to 1 / tau
    ROCSOLVER_LAUNCH_KERNEL((larft_set_diag<T, I>), dim3(blocks, 1, batch_count), dim3(BS2, 1), 0,
                            stream, k, tau, strideT, F, ldf, strideF);

    // restore original V
    ROCSOLVER_LAUNCH_KERNEL((larft_restore_tri<T, I, U>), gridTri, blockTri, 0, stream, tri_uplo, k,
                            V, shiftV + tri_offset, ldv, strideV, work);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
