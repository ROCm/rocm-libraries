/****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <cmath>

#include "rocauxiliary_bdsqr_hybrid.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/******************** Device functions *************************/
/***************************************************************/

/** BDSQR_ESTIMATE device function computes an estimate of the smallest
    singular value of a n-by-n upper bidiagonal matrix given by D and E
    It also applies convergence test if conver = 1 **/
template <typename T>
__device__ T bdsqr_estimate(const rocblas_int n, T* D, T* E, int t2b, T tol, int conver = 0)
{
    T smin = t2b ? std::abs(D[0]) : std::abs(D[n - 1]);
    T t = smin;

    rocblas_int je, jd;

    for(rocblas_int i = 1; i < n; ++i)
    {
        jd = t2b ? i : n - 1 - i;
        je = jd - t2b;
        if(conver && (std::abs(E[je]) <= tol * t))
        {
            E[je] = 0;
            smin = -1;
            break;
        }

        // Note: Order of operations is important to prevent t from being rounded down to zero
        t = std::abs(D[jd]) * (t / (t + std::abs(E[je])));

        smin = (t < smin) ? t : smin;
    }

    return smin;
}

/** BDSQR_QRSTEP device function applies implicit QR interation to
    the n-by-n bidiagonal matrix given by D and E using shift = sh **/
template <typename T, typename S>
__device__ void bdsqr_QRstep(const rocblas_int tid,
                             const rocblas_int t2b,
                             const rocblas_int n,
                             const rocblas_int nv,
                             const rocblas_int nu,
                             const rocblas_int nc,
                             S* D,
                             S* E,
                             T* V,
                             const rocblas_stride shiftV,
                             const rocblas_int ldv,
                             T* U,
                             const rocblas_stride shiftU,
                             const rocblas_int ldu,
                             T* C,
                             const rocblas_stride shiftC,
                             const rocblas_int ldc,
                             const S sh,
                             S* rots)
{
    S f, g, c, s, r;
    T temp1, temp2;

    const rocblas_int b2t = 1 - t2b;
    const rocblas_int dir = t2b - b2t; // +1 for t2b, -1 for b2t
    const rocblas_int nr = nv ? 2 * n : 0;

    if(tid == 0)
    {
        rocblas_int dk = (t2b ? 0 : n - 1);
        rocblas_int ek = dk - b2t;

        int sgn = (S(0) < D[dk]) - (D[dk] < S(0));
        if(D[dk] == 0)
            f = 0;
        else
            f = (std::abs(D[dk]) - sh) * (S(sgn) + sh / D[dk]);
        g = E[ek];

        for(rocblas_int kk = 0; kk < n - 1; kk++)
        {
            // first apply rotation by columns (t2b) or rows (b2t)
            lartg(f, g, c, s, r);
            if(kk > 0)
                E[ek - dir] = r;
            f = c * D[dk] - s * E[ek];
            E[ek] = c * E[ek] + s * D[dk];
            g = -s * D[dk + dir];
            D[dk + dir] = c * D[dk + dir];

            // save rotations to update singular vectors
            if(t2b && nv)
            {
                rots[ek] = c;
                rots[ek + n] = s;
            }
            if(b2t && (nu || nc))
            {
                rots[ek + nr] = c;
                rots[ek + nr + n] = s;
            }

            // then apply rotation by rows (t2b) or columns (b2t)
            lartg(f, g, c, s, r);
            D[dk] = r;
            f = c * E[ek] - s * D[dk + dir];
            D[dk + dir] = c * D[dk + dir] + s * E[ek];
            if(kk < n - 2)
            {
                g = -s * E[ek + dir];
                E[ek + dir] = c * E[ek + dir];
            }

            // save rotations to update singular vectors
            if(b2t && nv)
            {
                rots[ek] = c;
                rots[ek + n] = s;
            }
            if(t2b && (nu || nc))
            {
                rots[ek + nr] = c;
                rots[ek + nr + n] = s;
            }

            dk += dir;
            ek += dir;
        }

        ek = (t2b ? n - 2 : 0);
        E[ek] = f;
    }
    __syncthreads();

    // update singular vectors
    if(V && nv)
    {
        V += shiftV;

        // rotate from the left
        for(rocblas_int j = tid; j < nv; j += hipBlockDim_x)
        {
            rocblas_int k = (t2b ? 0 : n - 1);
            rocblas_int rk = (t2b ? 0 : n - 2);

            temp1 = V[k + j * ldv];
            for(rocblas_int kk = 0; kk < n - 1; kk++)
            {
                temp2 = V[(k + dir) + j * ldv];
                c = rots[rk];
                s = rots[rk + n];
                V[k + j * ldv] = c * temp1 - s * temp2;
                temp1 = c * temp2 + s * temp1;

                k += dir;
                rk += dir;
            }
            V[k + j * ldv] = temp1;
        }
    }
    if(U && nu)
    {
        U += shiftU;

        // rotate from the right
        for(rocblas_int i = tid; i < nu; i += hipBlockDim_x)
        {
            rocblas_int k = (t2b ? 0 : n - 1);
            rocblas_int rk = (t2b ? nr : (n - 2) + nr);

            temp1 = U[i + k * ldu];
            for(rocblas_int kk = 0; kk < n - 1; kk++)
            {
                temp2 = U[i + (k + dir) * ldu];
                c = rots[rk];
                s = rots[rk + n];
                U[i + k * ldu] = c * temp1 - s * temp2;
                temp1 = c * temp2 + s * temp1;

                k += dir;
                rk += dir;
            }
            U[i + k * ldu] = temp1;
        }
    }
    if(C && nc)
    {
        C += shiftC;

        // rotate from the left
        for(rocblas_int j = tid; j < nc; j += hipBlockDim_x)
        {
            rocblas_int k = (t2b ? 0 : n - 1);
            rocblas_int rk = (t2b ? nr : (n - 2) + nr);

            temp1 = C[k + j * ldc];
            for(rocblas_int kk = 0; kk < n - 1; kk++)
            {
                temp2 = C[(k + dir) + j * ldc];
                c = rots[rk];
                s = rots[rk + n];
                C[k + j * ldc] = c * temp1 - s * temp2;
                temp1 = c * temp2 + s * temp1;

                k += dir;
                rk += dir;
            }
            C[k + j * ldc] = temp1;
        }
    }
}

/** BDSQR_PERMUTE_SWAP device function performs swaps to implement permutation vector.
    The permutation vector will be restored to the identity permutation 0,1,2,...

    Note: this routine works in a single thread block **/
template <typename T, typename I>
__device__ static void bdsqr_permute_swap(const I n,
                                          const I nv,
                                          T* V,
                                          const I ldv,
                                          const I nu,
                                          T* U,
                                          const I ldu,
                                          const I nc,
                                          T* C,
                                          const I ldc,
                                          I* map)
{
    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    if(n <= 0)
        return;

    assert(map != nullptr);

    for(I i = 0; i < n; i++)
    {
        while(map[i] != i)
        {
            auto const map_i = map[i];
            auto const map_ii = map[map[i]];

            __syncthreads();

            if(tid == 0)
            {
                map[map_i] = map_i;
                map[i] = map_ii;
            }

            __syncthreads();

            // -----------
            // swap arrays
            // -----------

            auto const j_start = tid;
            auto const j_inc = nthreads;

            if(nv > 0)
            {
                auto const m = map_i;
                auto const i = map_ii;
                for(auto j = j_start; j < nv; j += j_inc)
                    swap(V[m + j * ((int64_t)ldv)], V[i + j * ((int64_t)ldv)]);
                __syncthreads();
            }

            if(nu > 0)
            {
                auto const m = map_i;
                auto const i = map_ii;
                for(auto j = j_start; j < nu; j += j_inc)
                    swap(U[j + m * ((int64_t)ldu)], U[j + i * ((int64_t)ldu)]);
                __syncthreads();
            }

            if(nc > 0)
            {
                auto const m = map_i;
                auto const i = map_ii;
                for(auto j = j_start; j < nc; j += j_inc)
                    swap(C[m + j * ((int64_t)ldc)], C[i + j * ((int64_t)ldc)]);
                __syncthreads();
            }
        }
    }
}

/********************* Device kernels **************************/
/** The work array is organized as follows:
    work[0] is the estimate of the smallest singular value.
    work[1] is the convergence threshold.
    work[2] is the current number of split blocks.
    work[3] is the number of new split blocks to spawn.
    Remaining elements contain the cosines and sines for Givens rotations as 2-tuples.

    The splits array is organized as follows:
    Each split block is represented by a 4-tuple, which encodes 5 data points.
    tuple[0] != 0 indicates that the QR step was applied to the block.
    sgn(tuple[0]) indicates if the block is t2b (positive sign) or b2t (negative sign).
    tuple[1] is the 0-based index of the current start point for the block.
    tuple[2] is the 0-based index of the current end point for the block.
    tuple[3] is the number of iterations applied to the block.

    The completed array is organized as follows:
    completed[0] is the number of batch instances that have finished all computations.
    completed[1] is the maximum number of split blocks identified across all batch instances.
    Remaining elements contain values indicating if each batch instance has finished all
        computations. 0 means computations are ongoing, 1 means computations are finished,
        and 2 means the input is invalid. */
/***************************************************************/

/** BDSQR_INIT kernel checks if there are any NaNs or Infs in the input, calculates the
    convergence threshold and initial estimate for the smallest singular value, and splits
    the matrix into diagonal blocks. **/
template <typename T, typename S>
ROCSOLVER_KERNEL void bdsqr_init(const rocblas_int n,
                                 S* DD,
                                 const rocblas_stride strideD,
                                 S* EE,
                                 const rocblas_stride strideE,
                                 rocblas_int* info,
                                 const rocblas_int maxiter,
                                 const S sfm,
                                 const S tol,
                                 rocblas_int* splitsA,
                                 S* workA,
                                 const rocblas_stride strideW,
                                 rocblas_int* completed)
{
    rocblas_int bid = hipBlockIdx_y;

    // select batch instance to work with
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* splits = splitsA + bid * (2 * n);
    S* work = workA + bid * strideW;

    bool found = false;
    rocblas_int ii = 0;
    rocblas_int start = 0;

    // calculate threshold for zeroing elements (convergence threshold)
    // direction
    int t2b = (D[0] >= D[n - 1]) ? 1 : 0;
    // estimate of the smallest singular value
    S smin = bdsqr_estimate<S>(n, D, E, t2b, tol);
    // threshold
    S thresh = std::max(tol * smin / S(std::sqrt(n)), S(maxiter) * sfm);

    work[0] = smin;
    work[1] = thresh;

    // search for NaNs, Infs, and splits in the input
    for(rocblas_int i = 0; i < n - 1; i++)
    {
        if(4 * i + 3 < 2 * n)
        {
            splits[4 * i] = 0;
            splits[4 * i + 1] = 0;
            splits[4 * i + 2] = 0;
            splits[4 * i + 3] = 0;
            __threadfence();
        }

        if(!std::isfinite(D[i]) || !std::isfinite(E[i]))
            found = true;

        if(std::abs(E[i]) < thresh)
        {
            E[i] = 0;
            if(start < i)
            {
                // save block endpoints
                splits[4 * ii + 1] = start;
                splits[4 * ii + 2] = i;
                ii++;
            }
            start = i + 1;
        }
    }

    if(!std::isfinite(D[n - 1]))
        found = true;

    if(start < n - 1)
    {
        // save block endpoints
        splits[4 * ii + 1] = start;
        splits[4 * ii + 2] = n - 1;
    }

    // update output
    if(found)
    {
        for(rocblas_int i = 0; i < n - 1; i++)
        {
            D[i] = nan("");
            E[i] = nan("");
        }
        D[n - 1] = nan("");

        info[bid] = n;
        completed[bid + 2] = 2; // 2 indicates bad input
        atomicAdd(completed, 1);
    }
    else
    {
        rocblas_int num_splits = ii + 1; // number of split blocks
        work[2] = num_splits;
        work[3] = 0;
        info[bid] = 0;

        // store largest number of split blocks in completed[1]
        rocblas_int temp_splits = 0;
        while(temp_splits < num_splits)
            temp_splits = atomicCAS(completed + 1, temp_splits, num_splits);
    }
}

/** BDSQR_LOWER2UPPER kernel transforms a lower bidiagonal matrix given by D and E
    into an upper bidiagonal matrix via givens rotations **/
template <typename T, typename S, typename W1, typename W2>
ROCSOLVER_KERNEL void bdsqr_lower2upper(const rocblas_int n,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        S* DD,
                                        const rocblas_stride strideD,
                                        S* EE,
                                        const rocblas_stride strideE,
                                        W1 UU,
                                        const rocblas_int shiftU,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        W2 CC,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        S* workA,
                                        const rocblas_stride strideW,
                                        rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    if(completed[bid + 2])
        return;

    // local variables
    rocblas_int i, j;
    S f, g, c, s, r;
    T temp1, temp2;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    T* U = (UU ? load_ptr_batch<T>(UU, bid, shiftU, strideU) : nullptr);
    T* C = (CC ? load_ptr_batch<T>(CC, bid, shiftC, strideC) : nullptr);
    S* rots = workA + bid * strideW + 4;

    if(tid == 0)
    {
        f = D[0];
        g = E[0];
        for(i = 0; i < n - 1; ++i)
        {
            // apply rotations by rows
            lartg(f, g, c, s, r);
            D[i] = r;
            E[i] = -s * D[i + 1];
            f = c * D[i + 1];
            g = E[i + 1];

            // save rotation to update singular vectors
            if(nu || nc)
            {
                rots[i] = c;
                rots[i + n] = s;
            }
        }
        D[n - 1] = f;
    }
    __syncthreads();

    // update singular vectors
    if(nu)
    {
        // rotate from the right (forward direction)
        for(i = tid; i < nu; i += hipBlockDim_x)
        {
            temp1 = U[i + 0 * ldu];
            for(j = 0; j < n - 1; j++)
            {
                temp2 = U[i + (j + 1) * ldu];
                c = rots[j];
                s = rots[j + n];
                U[i + j * ldu] = c * temp1 - s * temp2;
                temp1 = c * temp2 + s * temp1;
            }
            U[i + (n - 1) * ldu] = temp1;
        }
    }
    if(nc)
    {
        // rotate from the left (forward direction)
        for(j = tid; j < nc; j += hipBlockDim_x)
        {
            temp1 = C[0 + j * ldc];
            for(i = 0; i < n - 1; i++)
            {
                temp2 = C[(i + 1) + j * ldc];
                c = rots[i];
                s = rots[i + n];
                C[i + j * ldc] = c * temp1 - s * temp2;
                temp1 = c * temp2 + s * temp1;
            }
            C[(n - 1) + j * ldc] = temp1;
        }
    }
}

/** BDSQR_COMPUTE implements one iteration of the main loop of the
    BDSQR algorithm to compute the SVD of an upper bidiagonal matrix given by D and E **/
template <int MAX_THDS, typename T, typename S, typename W1, typename W2, typename W3>
ROCSOLVER_KERNEL void bdsqr_compute(const rocblas_int n,
                                    const rocblas_int nv,
                                    const rocblas_int nu,
                                    const rocblas_int nc,
                                    S* DD,
                                    const rocblas_stride strideD,
                                    S* EE,
                                    const rocblas_stride strideE,
                                    W1 VV,
                                    const rocblas_int shiftV,
                                    const rocblas_int ldv,
                                    const rocblas_stride strideV,
                                    W2 UU,
                                    const rocblas_int shiftU,
                                    const rocblas_int ldu,
                                    const rocblas_stride strideU,
                                    W3 CC,
                                    const rocblas_int shiftC,
                                    const rocblas_int ldc,
                                    const rocblas_stride strideC,
                                    const rocblas_int maxiter,
                                    const S eps,
                                    const S sfm,
                                    const S tol,
                                    const S minshift,
                                    rocblas_int* splitsA,
                                    S* workA,
                                    const rocblas_int incW,
                                    const rocblas_stride strideW,
                                    rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int sid_start = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;

    if(completed[bid + 2])
        return;

    // select batch instance
    // (avoiding arithmetics with possible nullptrs)
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    T* V = (VV ? load_ptr_batch<T>(VV, bid, shiftV, strideV) : nullptr);
    T* U = (UU ? load_ptr_batch<T>(UU, bid, shiftU, strideU) : nullptr);
    T* C = (CC ? load_ptr_batch<T>(CC, bid, shiftC, strideC) : nullptr);
    rocblas_int* splits = splitsA + bid * (2 * n);
    S* work = workA + bid * strideW;
    S* rots = work + 4;

    // shared variables
    __shared__ int t2b;
    __shared__ bool applyqr;
    __shared__ S smin, smax, sh;
    __shared__ S sval[MAX_THDS + 1];

    // local variables
    rocblas_int i, k;
    rocblas_int num_splits, iter;

    // main loop
    // iterate over each diagonal block
    num_splits = work[2];
    for(rocblas_int sid = sid_start; sid < num_splits; sid += hipGridDim_y)
    {
        // read diagonal block endpoints
        i = splits[4 * sid + 1];
        k = splits[4 * sid + 2];

        // number of iterations (QR steps) applied to current block
        iter = splits[4 * sid + 3];

        // skip block if it has already been processed
        if(i >= k || iter >= maxiter)
            continue;

        // find largest value of D and E
        iamax<MAX_THDS>(tid, k - i + 1, D + i, 1, sval);
        __syncthreads();
        iamax<MAX_THDS>(tid, k - i, E + i, 1, sval + 1);
        __syncthreads();

        if(tid == 0)
        {
            // current block goes from i until k
            // determine shift for the QR step
            // (apply convergence test to find gaps)
            t2b = std::abs(D[i]) >= std::abs(D[k]);
            sh = (t2b ? std::abs(D[i]) : std::abs(D[k]));

            // shift
            smin = bdsqr_estimate<S>(k - i + 1, D + i, E + i, t2b, tol, 1);
            // estimate of the largest singular value in the block
            smax = std::max(sval[0], sval[1]);

            // check for gaps, if none then continue
            applyqr = false;
            if(smin >= 0)
            {
                if(smin / smax <= minshift)
                    smin = 0; // shift set to zero if less than accepted value
                else if(sh > 0)
                {
                    if(smin * smin / sh / sh < eps)
                        smin = 0; // shift set to zero if negligible
                }

                applyqr = true;
            }
        }
        __syncthreads();

        // apply QR step
        if(applyqr)
        {
            if(tid == 0)
                splits[4 * sid] = (t2b ? 1 : -1);

            bdsqr_QRstep(tid, t2b, k - i + 1, nv, nu, nc, D + i, E + i, V, i, ldv, U, i * ldu, ldu,
                         C, i, ldc, smin, rots + incW * i);
        }
        else
        {
            if(tid == 0)
                splits[4 * sid] = 0;
        }
    }
}

/** BDSQR_ROTATE rotates all singular vectors using cosines and sines from
    BDSQR_SINGLE_ITER **/
template <typename T, typename S, typename W1, typename W2, typename W3>
ROCSOLVER_KERNEL void bdsqr_rotate(const rocblas_int n,
                                   const rocblas_int nv,
                                   const rocblas_int nu,
                                   const rocblas_int nc,
                                   W1 VV,
                                   const rocblas_int shiftV,
                                   const rocblas_int ldv,
                                   const rocblas_stride strideV,
                                   W2 UU,
                                   const rocblas_int shiftU,
                                   const rocblas_int ldu,
                                   const rocblas_stride strideU,
                                   W3 CC,
                                   const rocblas_int shiftC,
                                   const rocblas_int ldc,
                                   const rocblas_stride strideC,
                                   const rocblas_int maxiter,
                                   rocblas_int* splitsA,
                                   S* workA,
                                   const rocblas_int incW,
                                   const rocblas_stride strideW,
                                   rocblas_int* completed)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int sid_start = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;

    if(completed[bid + 2])
        return;

    // select batch instance to work with
    // (avoiding arithmetics with possible nullptrs)
    T* V = (VV ? load_ptr_batch<T>(VV, bid, shiftV, strideV) : nullptr);
    T* U = (UU ? load_ptr_batch<T>(UU, bid, shiftU, strideU) : nullptr);
    T* C = (CC ? load_ptr_batch<T>(CC, bid, shiftC, strideC) : nullptr);
    rocblas_int* splits = splitsA + bid * (2 * n);
    S* work = workA + bid * strideW;

    // local variables
    rocblas_int dir, k_start, k_end;
    rocblas_int num_splits, iter;

    S c, s;
    T temp1, temp2;

    // main loop
    // iterate over each diagonal block
    num_splits = work[2];
    for(rocblas_int sid = sid_start; sid < num_splits; sid += hipGridDim_y)
    {
        // read diagonal block endpoints
        dir = splits[4 * sid];
        k_start = splits[4 * sid + 1];
        k_end = splits[4 * sid + 2];

        // number of iterations (QR steps) applied to current block
        iter = splits[4 * sid + 3];

        // skip block if it has already been processed
        if(k_start >= k_end || iter >= maxiter)
            continue;

        // apply QR step
        if(dir != 0)
        {
            S* rots = work + 4 + incW * k_start;

            rocblas_int t2b = (dir > 0 ? 1 : 0);
            rocblas_int b2t = 1 - t2b;

            rocblas_int nn = k_end - k_start + 1;
            rocblas_int nr = nv ? 2 * nn : 0;

            if(V && tid < nv)
            {
                // rotate from the left
                rocblas_int k = (t2b ? k_start : k_end);
                rocblas_int rk = (t2b ? 0 : nn - 2);

                temp1 = V[k + tid * ldv];
                for(rocblas_int kk = k_start; kk < k_end; kk++)
                {
                    temp2 = V[(k + dir) + tid * ldv];
                    c = rots[rk];
                    s = rots[rk + nn];
                    V[k + tid * ldv] = c * temp1 - s * temp2;
                    temp1 = c * temp2 + s * temp1;

                    k += dir;
                    rk += dir;
                }
                V[k + tid * ldv] = temp1;
            }
            if(U && tid < nu)
            {
                // rotate from the right
                rocblas_int k = (t2b ? k_start : k_end);
                rocblas_int rk = (t2b ? nr : (nn - 2) + nr);

                temp1 = U[tid + k * ldu];
                for(rocblas_int kk = k_start; kk < k_end; kk++)
                {
                    temp2 = U[tid + (k + dir) * ldu];
                    c = rots[rk];
                    s = rots[rk + nn];
                    U[tid + k * ldu] = c * temp1 - s * temp2;
                    temp1 = c * temp2 + s * temp1;

                    k += dir;
                    rk += dir;
                }
                U[tid + k * ldu] = temp1;
            }
            if(C && tid < nc)
            {
                // rotate from the left
                rocblas_int k = (t2b ? k_start : k_end);
                rocblas_int rk = (t2b ? nr : (nn - 2) + nr);

                temp1 = C[k + tid * ldc];
                for(rocblas_int kk = k_start; kk < k_end; kk++)
                {
                    temp2 = C[(k + dir) + tid * ldc];
                    c = rots[rk];
                    s = rots[rk + nn];
                    C[k + tid * ldc] = c * temp1 - s * temp2;
                    temp1 = c * temp2 + s * temp1;

                    k += dir;
                    rk += dir;
                }
                C[k + tid * ldc] = temp1;
            }
        }
    }
}

/** BDSQR_UPDATE_ENDPOINTS updates the endpoint of the split blocks **/
template <typename T, typename S>
ROCSOLVER_KERNEL void bdsqr_update_endpoints(const rocblas_int n,
                                             S* EE,
                                             const rocblas_stride strideE,
                                             rocblas_int* splitsA,
                                             S* workA,
                                             const rocblas_stride strideW,
                                             rocblas_int* completed)
{
    rocblas_int sid_start = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;

    if(completed[bid + 2])
        return;

    // select batch instance to work with
    S* E = EE + bid * strideE;
    rocblas_int* splits = splitsA + bid * (2 * n);
    S* work = workA + bid * strideW;

    // local variables
    rocblas_int k_start, k_end, applyqr, iter;
    S thresh = work[1];
    rocblas_int num_splits = work[2];

    // iterate over each diagonal block
    for(rocblas_int sid = sid_start; sid < num_splits; sid += hipGridDim_y)
    {
        // read diagonal block endpoints
        applyqr = splits[4 * sid];
        k_start = splits[4 * sid + 1];
        k_end = splits[4 * sid + 2];

        // number of iterations (QR steps) applied to current block
        iter = splits[4 * sid + 3];
        if(applyqr != 0)
            splits[4 * sid + 3] = iter += k_end - k_start;

        // trim the block
        while(k_start < k_end && std::abs(E[k_start]) < thresh)
        {
            E[k_start] = 0;
            k_start++;
        }
        while(k_end > k_start && std::abs(E[k_end - 1]) < thresh)
        {
            E[k_end - 1] = 0;
            k_end--;
        }

        // check for zeroes
        if(applyqr == 0)
        {
            for(rocblas_int k = k_start; k < k_end; k++)
            {
                if(std::abs(E[k]) < thresh)
                {
                    E[k] = 0;
                    if(k_start < k)
                    {
                        // spawn new split block
                        rocblas_int new_sid = num_splits + atomicAdd(work + 3, 1);
                        splits[4 * new_sid + 1] = k_start;
                        splits[4 * new_sid + 2] = k;
                        splits[4 * new_sid + 3] = iter;
                    }
                    k_start = k + 1;
                }
            }
        }

        // save endpoints
        splits[4 * sid + 1] = k_start;
        splits[4 * sid + 2] = k_end;
    }
}

/** BDSQR_CHK_COMPLETED updates the split block count, checks if all split blocks have
    been fully processed, and marks the batch instance as completed if they have **/
template <typename T, typename S>
ROCSOLVER_KERNEL void bdsqr_chk_completed(const rocblas_int n,
                                          const rocblas_int maxiter,
                                          rocblas_int* splitsA,
                                          S* workA,
                                          const rocblas_stride strideW,
                                          rocblas_int* completed)
{
    rocblas_int bid = hipBlockIdx_y;

    if(completed[bid + 2])
        return;

    // array pointers
    rocblas_int* splits = splitsA + bid * (2 * n);
    S* work = workA + bid * strideW;

    // local variables
    rocblas_int k_start, k_end, iter;

    // update split block count
    rocblas_int num_splits = work[2] + work[3];
    work[2] = num_splits;
    work[3] = 0;

    // store largest number of split blocks in completed[1]
    rocblas_int temp_splits = completed[1];
    while(temp_splits < num_splits)
        temp_splits = atomicCAS(completed + 1, temp_splits, num_splits);

    // check all split blocks
    for(rocblas_int sid = 0; sid < num_splits; sid++)
    {
        k_start = splits[4 * sid + 1];
        k_end = splits[4 * sid + 2];
        iter = splits[4 * sid + 3];

        // if split block is not fully processed, return
        if(k_start < k_end && iter < maxiter)
            return;
    }

    // mark as completed
    completed[bid + 2] = 1;
    atomicAdd(completed, 1);
}

/** BDSQR_FINALIZE sets the output values for BDSQR, and sorts the singular values and
    vectors by shell sort or selection sort if applicable. **/
template <typename T, typename S, typename W1, typename W2, typename W3>
ROCSOLVER_KERNEL void bdsqr_finalize(const rocblas_int n,
                                     const rocblas_int nv,
                                     const rocblas_int nu,
                                     const rocblas_int nc,
                                     S* DD,
                                     const rocblas_stride strideD,
                                     S* EE,
                                     const rocblas_stride strideE,
                                     W1 VV,
                                     const rocblas_int shiftV,
                                     const rocblas_int ldv,
                                     const rocblas_stride strideV,
                                     W2 UU,
                                     const rocblas_int shiftU,
                                     const rocblas_int ldu,
                                     const rocblas_stride strideU,
                                     W3 CC,
                                     const rocblas_int shiftC,
                                     const rocblas_int ldc,
                                     const rocblas_stride strideC,
                                     rocblas_int* info,
                                     rocblas_int* splits_map,
                                     rocblas_int* completed)
{
    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const j_start = tid;
    auto const j_inc = nthreads;
    rocblas_int bid = hipBlockIdx_y;

    // if a NaN or Inf was detected in the input, return
    if(completed[bid + 2] > 1)
        return;

    // local variables
    rocblas_int i, j, m;
    rocblas_int local_info = 0;

    // array pointers
    S* const D = DD + bid * strideD;
    S* const E = EE + bid * strideE;
    T* const V = (VV ? load_ptr_batch<T>(VV, bid, shiftV, strideV) : nullptr);
    T* const U = (UU ? load_ptr_batch<T>(UU, bid, shiftU, strideU) : nullptr);
    T* const C = (CC ? load_ptr_batch<T>(CC, bid, shiftC, strideC) : nullptr);
    rocblas_int* map = (splits_map ? splits_map + bid * (2 * n) : nullptr);

    // ensure all singular values converged and are positive
    for(rocblas_int i = 0; i < n; i++)
    {
        if(i < n - 1 && E[i] != 0)
            local_info++;

        if(D[i] < 0)
        {
            if(nv)
            {
                for(auto j = j_start; j < nv; j += j_inc)
                    V[i + j * ((int64_t)ldv)] = -V[i + j * ((int64_t)ldv)];
                __syncthreads();
            }

            if(tid == 0)
                D[i] = -D[i];
        }
    }

    if(local_info > 0)
    {
        if(tid == 0)
            info[bid] = local_info;
        return;
    }
    __syncthreads();

    // sort singular values & vectors
    if(map)
    {
        if(nv || nu || nc)
        {
            shell_sort_descending(n, D, map);
            __syncthreads();
            bdsqr_permute_swap(n, nv, V, ldv, nu, U, ldu, nc, C, ldc, map);
        }
        else
        {
            rocblas_int* const null_map = nullptr;
            shell_sort_descending(n, D, null_map);
        }

        __syncthreads();
    }
    else
    {
        S p;
        for(i = 0; i < n - 1; i++)
        {
            m = i;
            p = D[i];
            for(j = i + 1; j < n; j++)
            {
                if(D[j] > p)
                {
                    m = j;
                    p = D[j];
                }
            }
            __syncthreads();

            if(m != i)
            {
                if(tid == 0)
                {
                    D[m] = D[i];
                    D[i] = p;
                }

                if(nv)
                {
                    for(j = tid; j < nv; j += hipBlockDim_x)
                        swap(V[m + j * ldv], V[i + j * ldv]);
                    __syncthreads();
                }
                if(nu)
                {
                    for(j = tid; j < nu; j += hipBlockDim_x)
                        swap(U[j + m * ldu], U[j + i * ldu]);
                    __syncthreads();
                }
                if(nc)
                {
                    for(j = tid; j < nc; j += hipBlockDim_x)
                        swap(C[m + j * ldc], C[i + j * ldc]);
                    __syncthreads();
                }
            }
        }
    }
}

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

template <typename T>
void rocsolver_bdsqr_getMemorySize(const rocblas_int n,
                                   const rocblas_int nv,
                                   const rocblas_int nu,
                                   const rocblas_int nc,
                                   const rocblas_int batch_count,
                                   size_t* size_splits_map,
                                   size_t* size_work,
                                   size_t* size_completed)
{
    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
    {
        *size_splits_map = 0;
        *size_work = 0;
        *size_completed = 0;
        return;
    }

    // size of split indices array
    *size_splits_map = sizeof(rocblas_int) * (2 * n) * batch_count;

    // size of workspace
    rocblas_int incW = 0;
    if(nv)
        incW += 2;
    if(nu || nc)
        incW += 2;
    *size_work = sizeof(T) * (4 + incW * n) * batch_count;

    // size of temporary workspace to indicate problem completion
    *size_completed = sizeof(rocblas_int) * (batch_count + 2);
}

template <typename S, typename W>
rocblas_status rocsolver_bdsqr_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nv,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        const rocblas_int ldv,
                                        const rocblas_int ldu,
                                        const rocblas_int ldc,
                                        S D,
                                        S E,
                                        W V,
                                        W U,
                                        W C,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nv < 0 || nu < 0 || nc < 0 || ldu < nu || ldv < 1 || ldc < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if((nv > 0 && ldv < n) || (nc > 0 && ldc < n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n > 1 && !E) || (n && nv && !V) || (n && nu && !U) || (n && nc && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename W1, typename W2, typename W3>
rocblas_status rocsolver_bdsqr_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nv,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        W1 V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        W2 U,
                                        const rocblas_int shiftU,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        W3 C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        rocblas_int* splits_map,
                                        S* work,
                                        rocblas_int* completed)
{
    ROCSOLVER_ENTER("bdsqr", "uplo:", uplo, "n:", n, "nv:", nv, "nu:", nu, "nc:", nc,
                    "shiftV:", shiftV, "ldv:", ldv, "shiftU:", shiftU, "ldu:", ldu,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocsolver_alg_mode alg_mode;
    ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_bdsqr, &alg_mode));

    // set tolerance and max number of iterations:
    // machine precision (considering rounding strategy)
    S eps = get_epsilon<S>() / 2;
    // safest minimum value such that 1/sfm does not overflow
    S sfm = get_safemin<S>();
    // max number of iterations (QR steps) before declaring not convergence
    rocblas_int maxiter = 6 * n * n;
    // relative accuracy tolerance
    S tol = std::max(S(10.0), std::min(S(100.0), S(pow(eps, -0.125)))) * eps;
    //(minimum accepted shift to not ruin relative accuracy) / (max singular
    // value)
    S minshift = std::max(eps, tol / S(100)) / (n * tol);

    rocblas_int incW = 0;
    if(nv)
        incW += 2;
    if(nu || nc)
        incW += 2;
    rocblas_stride strideW = 4 + incW * n;

    // grid dimensions
    rocblas_int nuc_max = std::max(nu, nc);
    rocblas_int nvuc_max = std::max(nv, nuc_max);
    rocblas_int split_groups;

    dim3 gridReset(batch_count / BS1 + 1, 1, 1);
    dim3 gridBasic(1, batch_count, 1);
    dim3 threadsReset(BS1, 1, 1);
    dim3 threadsBasic(1, 1, 1);
    dim3 threadsBS1(BS1, 1, 1);

    dim3 threadsUC((nuc_max ? std::min(nuc_max, BS1) : 1), 1, 1);
    dim3 threadsVUC((nvuc_max ? std::min(nvuc_max, BS1) : 1), 1, 1);

    // set completed = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, completed,
                            batch_count + 2, 0);

    // check for NaNs and Infs in input
    ROCSOLVER_LAUNCH_KERNEL((bdsqr_init<T>), gridBasic, threadsBasic, 0, stream, n, D, strideD, E,
                            strideE, info, maxiter, sfm, tol, splits_map, work, strideW, completed);

    if(n > 1)
    {
        if(alg_mode == rocsolver_alg_mode_hybrid)
        {
            ROCBLAS_CHECK(rocsolver_bdsqr_host_batch_template<T, S, W1, W2, W3, rocblas_int>(
                handle, uplo, n, nv, nu, nc, D, strideD, E, strideE, V, shiftV, ldv, strideV, U,
                shiftU, ldu, strideU, C, shiftC, ldc, strideC, info, batch_count, splits_map, work));
        }
        else
        {
            // rotate to upper bidiagonal if necessary
            if(uplo == rocblas_fill_lower)
            {
                ROCSOLVER_LAUNCH_KERNEL((bdsqr_lower2upper<T>), gridBasic, threadsUC, 0, stream, n,
                                        nu, nc, D, strideD, E, strideE, U, shiftU, ldu, strideU, C,
                                        shiftC, ldc, strideC, info, work, strideW, completed);
            }

            rocblas_int h_iter = 0;
            struct
            {
                rocblas_int completed;
                rocblas_int num_splits;
            } h_params;

            while(h_iter < maxiter)
            {
                // if all instances in the batch have finished, exit the loop
                HIP_CHECK(hipMemcpyAsync(&h_params, completed, sizeof(h_params),
                                         hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));

                if(h_params.completed == batch_count)
                    break;

                dim3 gridSplits(1, h_params.num_splits, batch_count);
                dim3 gridVUC((nvuc_max - 1) / BS1 + 1, h_params.num_splits, batch_count);

                for(rocblas_int inner_iters = 0; inner_iters < BDSQR_ITERS_PER_SYNC; inner_iters++)
                {
                    if(nvuc_max <= BDSQR_SWITCH_SIZE)
                    {
                        // main computation of SVD
                        ROCSOLVER_LAUNCH_KERNEL((bdsqr_compute<BS1, T>), gridSplits, threadsBS1, 0,
                                                stream, n, nv, nu, nc, D, strideD, E, strideE, V,
                                                shiftV, ldv, strideV, U, shiftU, ldu, strideU, C,
                                                shiftC, ldc, strideC, maxiter, eps, sfm, tol,
                                                minshift, splits_map, work, incW, strideW, completed);
                    }
                    else
                    {
                        // main computation of SVD
                        ROCSOLVER_LAUNCH_KERNEL((bdsqr_compute<BS1, T>), gridSplits, threadsBS1, 0,
                                                stream, n, nv, nu, nc, D, strideD, E, strideE,
                                                (W1) nullptr, shiftV, ldv, strideV, (W2) nullptr,
                                                shiftU, ldu, strideU, (W3) nullptr, shiftC, ldc,
                                                strideC, maxiter, eps, sfm, tol, minshift,
                                                splits_map, work, incW, strideW, completed);

                        // update singular vectors
                        ROCSOLVER_LAUNCH_KERNEL((bdsqr_rotate<T>), gridVUC, threadsVUC, 0, stream,
                                                n, nv, nu, nc, V, shiftV, ldv, strideV, U, shiftU,
                                                ldu, strideU, C, shiftC, ldc, strideC, maxiter,
                                                splits_map, work, incW, strideW, completed);
                    }

                    // update split block endpoints
                    ROCSOLVER_LAUNCH_KERNEL((bdsqr_update_endpoints<T>), gridSplits, threadsBasic,
                                            0, stream, n, E, strideE, splits_map, work, strideW,
                                            completed);
                }

                // check for completion
                h_iter += BDSQR_ITERS_PER_SYNC;
                ROCSOLVER_LAUNCH_KERNEL((bdsqr_chk_completed<T>), gridBasic, threadsBasic, 0,
                                        stream, n, maxiter, splits_map, work, strideW, completed);
            }
        }
    }

    // sort the singular values and vectors
    ROCSOLVER_LAUNCH_KERNEL((bdsqr_finalize<T>), gridBasic, threadsVUC, 0, stream, n, nv, nu, nc, D,
                            strideD, E, strideE, V, shiftV, ldv, strideV, U, shiftU, ldu, strideU,
                            C, shiftC, ldc, strideC, info, splits_map, completed);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
