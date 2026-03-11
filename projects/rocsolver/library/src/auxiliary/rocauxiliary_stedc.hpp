/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocauxiliary_steqr.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <algorithm>

ROCSOLVER_BEGIN_NAMESPACE

#define STEDC_BDIM 512 // Number of threads per thread-block used in main stedc kernels
#define STEDC_BDIM_VALUES 4 // Number of therads per thread-block used in mergeValues kernel
#define STEDC_BDIM_SOLVE 64 // Number of threads per thread-block used in the QR eigensolver

// STEDC_USE_EXTERNAL_UPDATE=true forces the use of external gemms for the vector updates.
// STEDC_WITH_STRIDED_BATCHED=true forces the use of strided_batched gemms when possible.
#define STEDC_USE_EXTERNAL_UPDATE true
#define STEDC_WITH_STRIDED_BATCHED_GEMM false

/*************** Main kernels *********************************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_DIVIDE_KERNEL implements the divide phase of the DC algorithm. It
    divides the input matrix into 'blks' sub-blocks.
        - This kernel is to be called with as many sroups in x as needed to cover all
        the batch_count problems.
        - Each thread will work with a matrix in the batch.
        - Size of groups is set to STEDC_BDIM. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_divide_kernel(const rocblas_int levs,
                        const rocblas_int blks,
                        const rocblas_int n,
                        S* DD,
                        const rocblas_stride strideD,
                        S* EE,
                        const rocblas_stride strideE,
                        const rocblas_int batch_count,
                        rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // for each matrix in the batch
    if(bid < batch_count)
    {
        // select batch instance to work with
        S* D = DD + bid * strideD;
        S* E = EE + bid * strideE;

        // temporary arrays in global memory
        rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
        rocblas_int* ns = ps + blks;

        // find sizes of sub-blocks
        if(STEDC_USE_EXTERNAL_UPDATE && STEDC_WITH_STRIDED_BATCHED_GEMM && batch_count == 1)
        {
            // division schema when using strided_batched_gemms for updates
            rocblas_int sz = n / blks;
            rocblas_int res = n - sz * blks;
            if(res < blks / 2)
            {
                res = blks - res;
                for(auto i = 0; i < blks; ++i)
                    ns[i] = i < res ? sz : sz + 1;
            }
            else
            {
                for(auto i = 0; i < blks; ++i)
                    ns[i] = i < res ? sz + 1 : sz;
            }
        }
        else
        {
            // normal division schema
            ns[0] = n;
            rocblas_int t, t2;
            for(auto i = 0; i < levs; ++i)
            {
                for(auto j = (1 << i); j > 0; --j)
                {
                    t = ns[j - 1];
                    t2 = t / 2;
                    ns[j * 2 - 1] = (2 * t2 < t) ? t2 + 1 : t2;
                    ns[j * 2 - 2] = t2;
                }
            }
        }

        // find beginning of sub-blocks and update elements in D
        rocblas_int p2 = 0;
        ps[0] = p2;
        for(auto i = 1; i < blks; ++i)
        {
            p2 += ns[i - 1];
            ps[i] = p2;

            // perform sub-block division
            S p = E[p2 - 1];
            D[p2] -= p;
            D[p2 - 1] -= p;
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_SOLVE_KERNEL implements the solver phase of the DC algorithm to
   compute the eigenvalues/eigenvectors of the 'blks' different sub-blocks of a matrix.
        - Call this kernel with batch_count groups in y, and 'blks' groups in x.
        - Each group will solve a sub-block.
        - Groups contain a single wavefront **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM_SOLVE)
    stedc_solve_kernel(const rocblas_int levs,
                       const rocblas_int blks,
                       const rocblas_int n,
                       S* DD,
                       const rocblas_stride strideD,
                       S* EE,
                       const rocblas_stride strideE,
                       S* CC,
                       const rocblas_int shiftC,
                       const rocblas_int ldc,
                       const rocblas_stride strideC,
                       rocblas_int* iinfo,
                       S* WA,
                       rocblas_int* workInt,
                       const S eps,
                       const S ssfmin,
                       const S ssfmax)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int sid = hipBlockIdx_x;
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tidb_inc = hipBlockDim_x;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* info = iinfo + bid;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    S* W = WA + bid * (2 * n);

    // Solve the blks sub-blocks in parallel (using classic QR iteration).
    if(sid < blks)
    {
        rocblas_int tmp = sid + 1;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;
        rocblas_int pin = ps[sid]; // start position of sub-block
        rocblas_int sbs = pout - pin; // size of sub-block

        run_steqr(tidb, tidb_inc, sbs, D + pin, E + pin, C + pin + pin * ldc, ldc, info,
                  W + pin * 2, 30 * sbs, eps, ssfmin, ssfmax, true);
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGESORT_KERNEL combines the two sorted arrays containing the eigenvalues of
    every pair of sub-blocks that need to be merged, and gets its corresponding vector z.
        - Call this kernel with batch_count groups in y, and as many groups in x as needed
          to cover the n values of the matrix.
        - Each thread will deal with one value.
        - Size of groups is set to STEDC_BDIM.**/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeSort_kernel(const rocblas_int levs,
                           const rocblas_int blks,
                           const rocblas_int k,
                           const rocblas_int n,
                           S* DD,
                           const rocblas_stride strideD,
                           S* CC,
                           const rocblas_int shiftC,
                           const rocblas_int ldc,
                           const rocblas_stride strideC,
                           S* workSvec,
                           rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int gid = hipBlockIdx_x;
    rocblas_int nofg = hipGridDim_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int totdim = nofg * dim;
    rocblas_int tid = gid * dim + hipThreadIdx_x;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* idd1 = ps + 2 * blks;
    rocblas_int* bp = idd1 + 4 * n;
    S* z1 = workSvec + bid * (std::max(7, n) * n);
    S* ev1 = z1 + 2 * n;

    // work with all the values (items) in parallel
    for(auto tx = tid; tx < n; tx += totdim)
    {
        rocblas_int dm = 1 << k;
        rocblas_int dm2 = dm << 1;

        // item 'tx' belongs to sub-block 'bx' and thus participates
        // in the merge to create the new sub-block 'nbx'
        rocblas_int bx = bisearch(tx, ps, blks, false, false) - 1;
        rocblas_int nbx = bx / dm2;
        bp[tx] = bx;

        // the new sub-block starts at 'pin', the middle point is 'pmid', and
        // it ends at 'pout'
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        rocblas_int pmid = ps[tmp + dm];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;

        // the position where the item 'tx' will end up in the ordered array is 'pos'
        S val = D[tx];
        rocblas_int pos1 = tx < pmid ? bisearch(val, D + pmid, pout - pmid, true, false)
                                     : bisearch(val, D + pin, pmid - pin, false, false);
        rocblas_int pos2 = tx < pmid ? tx - pin : tx - pmid;
        rocblas_int pos = pos1 + pos2;

        // get merged ordered array 'ev' and permutation map 'per'
        rocblas_int* idd = idd1 + pin;
        S* ev = ev1 + pin;
        ev[pos] = val;
        idd[pos] = tx;

        // get vector Z
        const S inv_sqrt2 = 1 / std::sqrt(2);
        val = tx < pmid ? C[pmid - 1 + tx * ldc] : C[pmid + tx * ldc];
        z1[tx] = val * inv_sqrt2;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGESEQUENCES_KERNEL finds the sequences of repeated eigenvalues for the
    relative tolerance on every pair of sub-blocks that need to be merged.
        - Call this kernel with batch_count groups in y, and as many groups as pairs of
          sub-blocks to be merged in x. Each group will deal with one merge.
        - Size of groups is set to STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeSequences_kernel(const rocblas_int levs,
                                const rocblas_int blks,
                                const rocblas_int k,
                                const rocblas_int n,
                                S* EE,
                                const rocblas_stride strideE,
                                S* workSvec,
                                rocblas_int* workInt,
                                const S eps)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int nbx = hipBlockIdx_x;
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* nrs = ps + blks;
    rocblas_int* idd1 = nrs + blks;
    rocblas_int* idd2 = idd1 + n;
    rocblas_int* bp = idd1 + 4 * n;
    rocblas_int* dcount = idd2 + n;
    rocblas_int* rmap = dcount + n;
    S* z1 = workSvec + bid * (std::max(7, n) * n);
    S* z2 = z1 + n;
    S* ev1 = z2 + n;
    S* ev2 = ev1 + n;
    S* ev3 = ev2 + n;
    S* c = ev3 + n;
    S* s = c + n;

    // temporary arrays in shared memory
    // used to store temp values during the different reductions
    extern __shared__ rocblas_int shmem[];
    rocblas_int* posi = shmem;
    rocblas_int* posf = posi + (1 << (k + 1));
    S* shmaxz = reinterpret_cast<S*>(posf + (1 << (k + 1)));
    S shmaxd;

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // the new sub-block starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Its size is 'sz'. Element 'p' is found at middle point
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    rocblas_int pmid = ps[tmp + dm];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    S p = 2 * E[pmid - 1];
    rocblas_int sz = pout - pin;

    // find max values of evs and z in the sub-blocks
    S valz, vald, maxz = 0, maxd = 0;
    for(auto ii = tid; ii < sz; ii += dim)
    {
        rocblas_int i = ii + pin;
        valz = std::abs(z1[i]);
        maxz = (valz > maxz) ? valz : maxz;
    }
    shmaxz[tid] = maxz;
    if(tid == 0)
    {
        maxd = abs(ev1[pin]);
        vald = abs(ev1[pout - 1]);
        maxd = (vald > maxd) ? vald : maxd;
    }
    __syncthreads();

    // reduction
    for(auto r = dim / 2; r > 0; r /= 2)
    {
        if(tid < r)
        {
            valz = shmaxz[tid + r];
            maxz = (valz > maxz) ? valz : maxz;
            shmaxz[tid] = maxz;
        }
        __syncthreads();
    }
    if(tid == 0)
        shmaxz[0] = (maxz > maxd) ? maxz : maxd;
    __syncthreads();
    maxd = shmaxz[0];

    // tol should be 8 * eps * (max diagonal or z element participating in the merge)
    S tol = 8 * eps * maxd;

    // Mark deflated values in each sub-block
    for(auto tx = tid; tx < dm2; tx += dim)
    {
        rocblas_int miposi = -1;
        rocblas_int miposf = -1;

        // each sub-block 'bx' starts and ends at 'in' and 'out', respectively
        rocblas_int bx = nbx * dm2 + tx;
        rocblas_int in = ps[bx];
        tmp = bx + 1;
        rocblas_int out = tmp < blks ? ps[tmp] : n;

        // find sequences of repeated values
        rocblas_int i = in;
        while(i < out)
        {
            rocblas_int count = 0;
            rocblas_int map = idd1[i];
            vald = ev1[i];
            valz = z1[map];

            if(std::abs(p * valz) <= tol)
            {
                // if element in z is zero, i cannot be the base of a new sequence
                dcount[i] = 0;
                i++;
            }
            else
            {
                // otherwise, take base and search for sequence
                miposf = i;
                miposi = (miposi == -1) ? i : miposi;
                rocblas_int oldi = i;
                count = 1;
                i++;
                while(i < out)
                {
                    S valdt = ev1[i];
                    if(abs(vald - valdt) <= tol)
                    {
                        // value repeated for given tolerance. It is part of the sequence
                        count++;
                        dcount[i] = 0;
                        i++;
                    }
                    else
                        break;
                }
                dcount[oldi] = count;
            }
        }

        // posi and posf contains the base of the first and last sequences of values
        // in each sub-block
        posi[tx] = miposi;
        posf[tx] = miposf;
    }
    __syncthreads();

    // now reduce the results of all the sub-blocks
    // merging the different sequences when required.
    dm = 1;
    for(auto kk = 0; kk <= k; ++kk)
    {
        dm *= 2;
        for(auto tx = tid; tx < dm2 / dm; tx += dim)
        {
            rocblas_int sh = nbx * dm2;
            rocblas_int bin = sh + tx * dm;
            rocblas_int in = posf[bin - sh];
            rocblas_int bj = bin + dm / 2;
            rocblas_int j = posi[bj - sh];
            rocblas_int bout = bin + dm;
            rocblas_int out = bout < blks ? ps[bout] : n;
            bool go = (j >= 0 && in >= 0);

            // find sequences to merge
            while(go && j < out)
            {
                go = false;

                // the begining of base sequence is 'vald' and there are 'i'
                // elements of the sequence in front that are within tolerance
                vald = ev1[in];
                rocblas_int count = dcount[j];
                rocblas_int i = bisearch(vald + tol, ev1 + j, count, false, false);

                // if i == 0, there is nothing to merge; we are done
                if(i > 0)
                {
                    // otherwise there is a sequence to merge; i elements will be merged
                    dcount[in] = i + j - in;
                    dcount[j] = 0;
                    j += i;

                    // if i == count, everything was merged; we are done
                    if(i < count)
                    {
                        // otherwise not everything was merged; see if there is a new sequence
                        // in the 'count - i' not merged elements
                        count -= i;
                        i = 0;
                        while(i < count && std::abs(p * z1[idd1[j + i]]) <= tol)
                            i++;

                        // if i == count, there is no new sequence; we are done
                        if(i < count)
                        {
                            // otherwise we have a new base sequence; see if there are more
                            // sequences in front
                            in = j + i;
                            dcount[in] = count - i;
                            rocblas_int inj = j + count;
                            count = out - inj;
                            i = 0;
                            while(i < count && dcount[inj + i] == 0)
                                i++;

                            // if i == count, no more sequences; we are done
                            if(i < count)
                            {
                                // otherwise there are more sequences in front
                                // that need to be analyzed
                                go = true;
                                j = inj + i;
                            }
                        }
                    }
                }
            }

            // merges are done...
            // update position of first and last sequences in each merged block
            rocblas_int tmp1 = posf[bj - sh];
            rocblas_int tmp2 = j > tmp1 ? in : tmp1;
            tmp1 = posi[bin - sh];
            tmp = std::max(tmp1, tmp2);
            go = (tmp1 > 0 && tmp2 < 0) || (tmp1 < 0 && tmp2 > 0);
            posf[bin - sh] = go ? tmp : tmp2;
            posi[bin - sh] = go ? tmp : tmp1;
        }
        __syncthreads();
    }

    // compute final number of non-deflated elementes in each sub-block
    for(auto tx = tid; tx < dm2; tx += dim)
    {
        // each sub-block 'bx' starts and ends at 'in' and 'out', respectively
        rocblas_int bx = nbx * dm2 + tx;
        rocblas_int in = ps[bx];
        tmp = bx + 1;
        rocblas_int out = tmp < blks ? ps[tmp] : n;

        rocblas_int count = 0;
        rocblas_int j = in;
        while(j < out)
        {
            tmp = dcount[j];
            rocblas_int jinc = 1;
            if(tmp > 0)
            {
                jinc = tmp;
                count++;
                for(auto i = 1; i < tmp; ++i)
                {
                    rocblas_int map = idd1[j + i];
                    valz = z1[map];
                    if(std::abs(p * valz) <= tol)
                        dcount[j + i] = -1;
                }
            }
            j += jinc;
        }
        posf[tx] = count;
    }
    __syncthreads();

    // reduce the results of all the sub-blocks
    for(auto kk = 0; kk <= k; ++kk)
    {
        for(auto tx = tid; tx < dm2 / 2; tx += dim)
        {
            dm = 1 << kk;
            rocblas_int bx = (tx / dm) * dm * 2 + dm - 1;
            rocblas_int bxx = bx + tx % dm + 1;
            posf[bxx] += posf[bx];
        }
        __syncthreads();
    }

    // store final number of non-deflated elements in 'nrs'
    for(auto tx = tid; tx < dm2; tx += dim)
        nrs[nbx * dm2 + tx] = posf[tx];
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEDEFLATE_KERNEL performs deflation in the sequences of repeated values of
    every pair of sub-blocks that need to be merged.
        - Call this kernel with batch_count groups in y, and as many groups in x as needed
          to cover the n values of the matrix.
        - Each thread will deal with one value.
        - Size of groups is set to STEDC_BDIM_VALUES.**/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_mergeDeflate_kernel(const rocblas_int levs,
                                                                              const rocblas_int blks,
                                                                              const rocblas_int k,
                                                                              const rocblas_int n,
                                                                              S* workSvec,
                                                                              rocblas_int* workInt,
                                                                              const S eps)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int gid = hipBlockIdx_x;
    rocblas_int nofg = hipGridDim_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int totdim = nofg * dim;
    rocblas_int tid = gid * dim + hipThreadIdx_x;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* nrs = ps + blks;
    rocblas_int* idd1 = nrs + blks;
    rocblas_int* idd2 = idd1 + n;
    rocblas_int* bp = idd1 + 4 * n;
    rocblas_int* dcount = idd2 + n;
    rocblas_int* rmap = dcount + n;
    S* z1 = workSvec + bid * (std::max(7, n) * n);
    S* z2 = z1 + n;
    S* ev1 = z2 + n;
    S* ev2 = ev1 + n;
    S* ev3 = ev2 + n;
    S* c = ev3 + n;
    S* s = c + n;

    // work with all the values (items) in parallel
    for(auto tx = tid; tx < n; tx += totdim)
    {
        rocblas_int dm = 1 << k;
        rocblas_int dm2 = dm << 1;

        // value 'tx' belongs to sub-block 'bx' and thus participates
        // in the merge to create the new sub-block 'nbx'
        rocblas_int bx = bp[tx];
        rocblas_int nbx = bx / dm2;

        // the sub-block 'bx' starts at 'in', and ends at 'out'
        // the new sub-block 'nbx' starts at 'pin', and ends at 'pout'
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;
        rocblas_int in = ps[bx];
        tmp = bx + 1;
        rocblas_int out = tmp < blks ? ps[tmp] : n;

        // 'count' is the number of non-deflated values until value 'tx'
        rocblas_int count = (bx % dm2 == 0) ? 0 : nrs[bx - 1];
        rocblas_int pj = tx - pin;
        rocblas_int j = tx - in;
        for(auto i = 0; i < j; ++i)
        {
            if(dcount[in + i] > 0)
                count++;
        }

        rocblas_int map = idd1[tx];
        S vald = ev1[tx];
        S valz = z1[map];
        rocblas_int dcnt = dcount[tx];

        // if value ev is marked as repeated, move it to the deflated list and finish
        if(dcnt < 1)
        {
            rocblas_int idx = pout - 1 - pj + count;
            ev3[idx] = vald;
            idd2[idx] = map;
        }

        // otehrwise move it to the non-deflated list, and compute rotations to zero out
        // the corresponding z element when required
        else
        {
            dcnt--;
            dcount[tx] = dcnt;
            rocblas_int idx = pin + count;
            ev2[idx] = vald;
            idd2[idx] = -(map + 1);

            for(auto i = 0; i < dcnt; ++i)
            {
                rocblas_int idx2 = tx + 1 + i;
                rocblas_int mapt = dcount[idx2];
                if(mapt == 0)
                {
                    // a rotation is needed
                    mapt = idd1[idx2];
                    S valzt = z1[mapt];
                    S cc, ss, rr;
                    lartg(valz, valzt, cc, ss, rr);
                    valz = rr;

                    // save the rotation encoded for mergeRotate
                    rmap[idx2] = mapt;
                    c[mapt] = cc;
                    s[mapt] = ss;
                }
                else
                {
                    // no rotation required
                    rmap[idx2] = -1;
                }
            }
            z2[idx] = valz;
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_KERNEL prepares the components for the secular equations of every
    pair of sub-blocks that need to be merged.
        - Call this kernel with batch_count groups in y, and n groups in x.
        - Size of groups is set to STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_kernel(const rocblas_int levs,
                              const rocblas_int blks,
                              const rocblas_int k,
                              const rocblas_int n,
                              S* EE,
                              const rocblas_stride strideE,
                              S* workSvec,
                              S* workStmp,
                              rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int jj = hipBlockIdx_x;
    rocblas_int dimr = hipBlockDim_x;
    rocblas_int rid = hipThreadIdx_x;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* nrs = ps + blks;
    rocblas_int* bp = ps + 2 * blks + 4 * n;
    S* z1 = workSvec + bid * (std::max(7, n) * n);
    S* z2 = z1 + n;
    S* ev2 = z2 + 2 * n;
    S* temps = workStmp + bid * (n * n);

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // column 'jj' belongs to sub-block 'bx' and thus forms part of
    // the new sub-block 'nbx'
    rocblas_int bx = bp[jj];
    rocblas_int nbx = bx / dm2;

    // the new sub-block starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Element 'p' is found at middle point
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    rocblas_int pmid = ps[tmp + dm];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    S p = 2 * E[pmid - 1];
    rocblas_int nr = nrs[(nbx + 1) * dm2 - 1]; // number of non-deflated values in sub-block
    rocblas_int j = jj - pin;

    if(j < nr)
    {
        S* tmpd = temps + pin * n;
        S* ev = ev2 + pin;
        S* Z = z1 + pin;

        // if 'p' is negative, the values are copied as negative in reverse order
        // as required by the secular equation solvers
        bool pneg = (p < 0);
        rocblas_int sig = pneg ? -1 : 1;
        rocblas_int start = pneg ? nr - 1 : 0;

        for(auto i = rid; i < nr; i += dimr)
        {
            int id = start + sig * i;
            tmpd[i + j * n] = sig * ev[id];
            if(j == 0)
                Z[i] = z2[id + pin];
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEROTATE_KERNEL performs rotation of vectors corresponding to deflations
        - Call this kernel with batch_count groups in y, and n (matrix size) groups in x.
        - Each group will deal with one deflation group, groups that don't correspond to
          a deflation group will do nothing.
        - Size of groups is set to STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeRotate_kernel(const rocblas_int levs,
                             const rocblas_int blks,
                             const rocblas_int k,
                             const rocblas_int n,
                             S* CC,
                             const rocblas_int shiftC,
                             const rocblas_int ldc,
                             const rocblas_stride strideC,
                             S* workSvec,
                             rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* idd1 = ps + 2 * blks;
    rocblas_int* dcount = idd1 + 2 * n;
    rocblas_int* rmap = dcount + n;
    S* z1 = workSvec + bid * (std::max(7, n) * n);
    S* cc = z1 + 5 * n;
    S* ss = cc + n;

    constexpr int regs = 16;
    const int chunk_width = regs * hipBlockDim_x;
    const int n_chunks = (n - 1) / chunk_width + 1;
    S bval[regs];
    S tval[regs];

    rocblas_int dgs = hipBlockIdx_x;
    rocblas_int dcnt = dcount[dgs];
    if(dcnt)
    {
        rocblas_int base = idd1[dgs];
        S* Cbase = C + base * ldc;

        for(auto chunk = 0; chunk < n_chunks; chunk++)
        {
            for(auto i = 0; i < regs; i++)
            {
                int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                if(x < n)
                    bval[i] = Cbase[x];
            }

            for(auto dn = 0; dn < dcnt; dn++)
            {
                rocblas_int top = rmap[dgs + dn + 1];
                if(top > -1)
                {
                    S c = cc[top];
                    S s = ss[top];
                    S* Ctop = C + top * ldc;

                    for(auto i = 0; i < regs; i++)
                    {
                        int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                        if(x < n)
                            tval[i] = Ctop[x];
                    }

                    for(auto i = 0; i < regs; i++)
                    {
                        S valf = bval[i];
                        S valg = tval[i];
                        bval[i] = valf * c - valg * s;
                        tval[i] = valf * s + valg * c;
                    }

                    for(auto i = 0; i < regs; i++)
                    {
                        int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                        if(x < n)
                            Ctop[x] = tval[i];
                    }
                }
                __syncthreads();
            }

            for(auto i = 0; i < regs; i++)
            {
                int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                if(x < n)
                    Cbase[x] = bval[i];
            }
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVALUES_KERNEL solves the secular equation for every value of every pair of
    sub-blocks that need to be merged, and re-scales vector z accordingly.
        - Call this kernel with batch_count groups in y, and as many groups in x as needed
          to cover the n values of the matrix.
        - Each thread will deal with one value.
        - Size of groups is set to STEDC_BDIM_VALUES.**/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM_VALUES)
    stedc_mergeValues_kernel(const rocblas_int levs,
                             const rocblas_int blks,
                             const rocblas_int k,
                             const rocblas_int n,
                             S* EE,
                             const rocblas_stride strideE,
                             S* workSvec,
                             S* workStmp,
                             rocblas_int* workInt,
                             const S eps,
                             const S ssfmin,
                             const S ssfmax)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int gid = hipBlockIdx_x;
    rocblas_int nofg = hipGridDim_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int totdim = nofg * dim;
    rocblas_int tid = gid * dim + hipThreadIdx_x;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* nrs = ps + blks;
    rocblas_int* idd2 = nrs + blks + n;
    rocblas_int* bp = ps + 2 * blks + 4 * n;
    S* z1 = workSvec + bid * (std::max(7, n) * n);
    S* ev3 = z1 + 4 * n;
    S* temps = workStmp + bid * (n * n);

    // work with all the values (items) in parallel
    for(auto tx = tid; tx < n; tx += totdim)
    {
        rocblas_int dm = 1 << k;
        rocblas_int dm2 = dm << 1;

        // item 'tx' belongs to sub-block 'bx' and thus participates
        // in the merge to create the new sub-block 'nbx'
        rocblas_int bx = bp[tx];
        rocblas_int nbx = bx / dm2;

        // the new sub-block starts at 'pin', the middle point is 'pmid', and
        // it ends at 'pout'. Element 'p' is found at middle point
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        rocblas_int pmid = ps[tmp + dm];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;
        S p = 2 * E[pmid - 1];
        rocblas_int nr = nrs[(nbx + 1) * dm2 - 1]; // number of non-deflated values in sub-block

        // solve secular equation for every non-deflated value
        rocblas_int linfo;

        if(idd2[tx] < 0)
        {
#if defined(ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
            linfo = slaed4(nr, tx - pin, temps + tx * n, z1 + pin, std::abs(p), ev3[tx]);
#else
            if(tx - pin == nr - 1)
                linfo = seq_solve_ext(nr, temps + tx * n, z1 + pin, std::abs(p), ev3[tx], eps,
                                      ssfmin, ssfmax);
            else
                linfo = seq_solve(nr, temps + tx * n, z1 + pin, std::abs(p), ev3[tx], eps, ssfmin,
                                  ssfmax);
#endif
            if(p < 0)
                ev3[tx] *= -1;
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEREINSERT_KERNEL combines and sort the new eigenvalues with the deflated values
        - Call this kernel with batch_count groups in y, and as many groups in x as needed
          to cover the n values of the matrix.
        - Each thread will deal with one value.
        - Size of groups is set to STEDC_BDIM.**/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeReinsert_kernel(const rocblas_int levs,
                               const rocblas_int blks,
                               const rocblas_int k,
                               const rocblas_int n,
                               S* DD,
                               const rocblas_stride strideD,
                               S* EE,
                               const rocblas_stride strideE,
                               S* workSvec,
                               rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int gid = hipBlockIdx_x;
    rocblas_int nofg = hipGridDim_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int totdim = nofg * dim;
    rocblas_int tid = gid * dim + hipThreadIdx_x;

    // select batch instance to work with
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* nrs = ps + blks;
    rocblas_int* idd1 = nrs + blks;
    rocblas_int* idd2 = idd1 + n;
    rocblas_int* bp = ps + 2 * blks + 4 * n;
    S* z1 = workSvec + bid * (std::max(7, n) * n);
    S* ev3 = z1 + 4 * n;

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // work with all the values (items) in parallel
    for(auto j = tid; j < n; j += totdim)
    {
        // item 'j' belongs to sub-block 'bx' and thus form vector of
        // the new sub-block 'nbx'
        rocblas_int bx = bp[j];
        rocblas_int nbx = bx / dm2;

        // the new sub-block starts at 'pin', the middle point is 'pmid', and
        // it ends at 'pout'. Element 'p' is found at middle point
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        rocblas_int pmid = ps[tmp + dm];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;
        S p = 2 * E[pmid - 1];
        rocblas_int nr = nrs[(nbx + 1) * dm2 - 1]; // number of non-deflated values in sub-block

        // re-insert deflated values to keep new sub-blocks ordered
        rocblas_int nf = pout - pin - nr; // number of deflated values

        // the position where the item 'j' will end up in the ordered array is 'pos'
        S val = ev3[j];
        rocblas_int pos1 = (j < nr + pin) ? bisearch(val, ev3 + nr + pin, nf, true, true)
                                          : bisearch(val, ev3 + pin, nr, false, (p < 0));
        rocblas_int pos2 = (j < nr + pin) ? (p < 0 ? pin + nr - 1 - j : j - pin) : pout - j - 1;
        rocblas_int pos = pos1 + pos2 + pin;

        // get merged ordered array 'ev' and permutation map 'ord'
        D[pos] = val;

        rocblas_int ind = idd2[j];
        if(ind < 0)
            idd1[pos] = -(j + 1);
        else
            idd1[pos] = ind;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGERESCALE_KERNEL reconstructs perturbed vector Z of the rank-1 system.
        - Call this kernel with batch_count groups in y, and n groups in x.
        - Each group will deal with one row of Z corresponding to each merge.
        - Size of groups is set to STEDC_BDIM.**/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeRescale_kernel(const rocblas_int levs,
                              const rocblas_int blks,
                              const rocblas_int k,
                              const rocblas_int n,
                              S* EE,
                              const rocblas_stride strideE,
                              S* workSvec,
                              S* workStmp,
                              S* workSz,
                              rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int ii = hipBlockIdx_x;
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* nrs = ps + blks;
    rocblas_int* bp = ps + 2 * blks + 4 * n;
    S* z1 = workSvec + bid * (std::max(7, n) * n);
    S* ev2 = z1 + 3 * n;
    S* temps = workStmp + bid * (n * n);
    S* zf = workSz + bid * n;

    // temporary arrays in shared memory
    // used to store temp values during the different reductions
    __shared__ S inrms[STEDC_BDIM];

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // 'ii' belongs to sub-block 'bx' and thus form vector of
    // the new sub-block 'nbx'
    rocblas_int bx = bp[ii];
    rocblas_int nbx = bx / dm2;

    // the new sub-block starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Element 'p' is found at middle point
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    rocblas_int pmid = ps[tmp + dm];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    S p = 2 * E[pmid - 1];
    rocblas_int nr = nrs[(nbx + 1) * dm2 - 1]; // number of non-deflated values in sub-block

    S* evd = ev2 + pin;
    rocblas_int start = (p < 0) ? nr - 1 : 0;
    rocblas_int inc = (p < 0) ? -1 : 1;

    rocblas_int i = ii - pin;

    // compute re-scaled vector Z of rank-1 perturbed system
    if(i < nr)
    {
        rocblas_int sgnz = (z1[i + pin] < 0) ? -1 : 1;
        S dd = evd[start + inc * i];
        S mul = 1;

        for(auto j = tidb; j < nr; j += dim)
        {
            S num = std::abs(temps[i + (j + pin) * n]);
            S den = (j == i) ? 1 : std::abs(dd - evd[start + inc * j]);
            mul *= num / den;
        }
        inrms[tidb] = mul;
        __syncthreads();

        // reduction (for the norms)
        for(auto r = dim / 2; r > 0; r /= 2)
        {
            if(tidb < r)
            {
                mul *= inrms[tidb + r];
                inrms[tidb] = mul;
            }
            __syncthreads();
        }

        if(tidb == 0)
            zf[i + pin] = sgnz * std::sqrt(mul);
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVECTORS_KERNEL computes vectors of the rank-1 system for
    every pair of sub-blocks that need to be merged.
        - Call this kernel with batch_count groups in y, and n groups in x.
        - Each group works with a column/vector.
        - Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeVectors_kernel(const rocblas_int levs,
                              const rocblas_int blks,
                              const rocblas_int k,
                              const rocblas_int n,
                              S* EE,
                              const rocblas_stride strideE,
                              S* workSvec,
                              S* workStmp,
                              S* workSz,
                              rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int j = hipBlockIdx_x;
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* nrs = ps + blks;
    rocblas_int* idd2 = nrs + blks + n;
    rocblas_int* bp = ps + 2 * blks + 4 * n;
    S* vecs = workSvec + bid * (std::max(7, n) * n);
    S* temps = workStmp + bid * (n * n);
    S* zf = workSz + bid * n;

    // temporary arrays in shared memory
    // used to store temp values during the different reductions
    __shared__ S inrms[STEDC_BDIM];

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // column 'j' belongs to sub-block 'bx' and thus form vector of
    // the new sub-block 'nbx'
    rocblas_int bx = bp[j];
    rocblas_int nbx = bx / dm2;

    // the new sub-block starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Element 'p' is found at middle point
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    rocblas_int pmid = ps[tmp + dm];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    S p = 2 * E[pmid - 1];
    rocblas_int nr = nrs[(nbx + 1) * dm2 - 1]; // number of non-deflated values in sub-block
    rocblas_int start = (p < 0) ? nr - 1 : 0;
    rocblas_int inc = (p < 0) ? -1 : 1;

    // compute vectors of rank-1 perturbed system and their norms
    if(idd2[j] < 0 && j < n)
    {
        S tm, nrm = 0;
        for(auto i = tidb; i < nr; i += dim)
        {
            S tot = zf[i + pin] / temps[i + j * n];
            vecs[i + j * n] = tot;
            nrm += tot * tot;
        }
        inrms[tidb] = nrm;
        __syncthreads();

        // reduction (for the norms)
        for(auto r = dim / 2; r > 0; r /= 2)
        {
            if(tidb < r)
            {
                nrm += inrms[tidb + r];
                inrms[tidb] = nrm;
            }
            __syncthreads();
        }
        nrm = std::sqrt(inrms[0]);

        // normalize
        for(auto i = tidb; i < nr; i += dim)
            vecs[i + j * n] /= nrm;
    }

    /*if(!STEDC_USE_EXTERNAL_UPDATE)
    {
        // Vectors should be updated at this point when not using external gemm update.
        // TODO: the code needs to be revisited and adapted for this new implementation
        // of stedc. Performance of the internal gemm needs to be re-evaluated.
    }*/
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPGEMM1_KERNEL prepares the matrix of vectors of the rank-1 system for
    the gemm to update eigenvectors (pad with zeros and insert 1 for deflated values).
        - Call this kernel with batch_count groups in y, and n groups in x.
        - Groups are size STEDC_BDIM **/
template <typename S>
__launch_bounds__(STEDC_BDIM) ROCSOLVER_KERNEL
    void stedc_mergePrepgemm1_kernel(const rocblas_int levs,
                                     const rocblas_int blks,
                                     const rocblas_int k,
                                     const rocblas_int n,
                                     S* workStmp,
                                     rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int j = hipBlockIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid = hipThreadIdx_x;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* idd1 = ps + 2 * blks;
    rocblas_int* bp = ps + 2 * blks + 4 * n;
    S* temps = workStmp + bid * (n * n);

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // column 'j' belongs to sub-block 'bx' and thus form vector of
    // the new sub-block 'nbx'
    rocblas_int bx = bp[j];
    rocblas_int nbx = bx / dm2;

    // the new sub-block starts at 'pin', and
    // it ends at 'pout'. Its size is 'sz'
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    rocblas_int sz = pout - pin;

    rocblas_int t = idd1[j];

    for(auto ii = tid; ii < sz; ii += dim)
    {
        rocblas_int i = ii + pin;
        temps[i + j * n] = (i == t) ? 1 : 0;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPGEMM_KERNEL prepares the matrix of vectors of the rank-1 system for
    the gemm to update eigenvectors (pad with zeros and permutate rows and columns).
        - Call this kernel with batch_count groups in y, and n groups in x
        - Groups are size STEDC_BDIM **/
template <typename S>
__launch_bounds__(STEDC_BDIM) ROCSOLVER_KERNEL
    void stedc_mergePrepgemm_kernel(const rocblas_int levs,
                                    const rocblas_int blks,
                                    const rocblas_int k,
                                    const rocblas_int n,
                                    S* EE,
                                    const rocblas_stride strideE,
                                    S* workSvec,
                                    S* workStmp,
                                    rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int j = hipBlockIdx_x;
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* nrs = ps + blks;
    rocblas_int* idd1 = nrs + blks;
    rocblas_int* idd2 = idd1 + n;
    rocblas_int* bp = ps + 2 * blks + 4 * n;
    S* vecs = workSvec + bid * (std::max(7, n) * n);
    S* temps = workStmp + bid * (n * n);

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // column 'j' belongs to sub-block 'bx' and thus form vector of
    // the new sub-block 'nbx'
    rocblas_int bx = bp[j];
    rocblas_int nbx = bx / dm2;

    // the new sub-block starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Element 'p' is found at middle point
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    rocblas_int pmid = ps[tmp + dm];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    S p = 2 * E[pmid - 1];
    rocblas_int nr = nrs[(nbx + 1) * dm2 - 1]; // number of non-deflated values in sub-block

    rocblas_int start = (p < 0) ? pin + nr - 1 : pin;
    rocblas_int inc = (p < 0) ? -1 : 1;

    // put vectors in padded matrix 'temps' to use external gemm for the update
    rocblas_int ind = idd1[j];

    if(ind < 0)
    {
        for(auto i = tid; i < nr; i += dim)
        {
            // read rank-1 vector value from 'vecs' (this permutates columns)
            rocblas_int jv = -(ind + 1);
            S val = vecs[i + jv * n];

            // write in final position in 'temps' (this permutates rows)
            rocblas_int it = -(idd2[start + inc * i] + 1);
            temps[it + j * n] = val;
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEUPDATE_KERNEL updates vectors after a merge is done.
    (simply copy results from temporary arrays into V)
        - Call this kernel with batch_count groups in y, and n groups in x
        - Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeUpdate_kernel(const rocblas_int levs,
                             const rocblas_int blks,
                             const rocblas_int k,
                             const rocblas_int n,
                             S* CC,
                             const rocblas_int shiftC,
                             const rocblas_int ldc,
                             const rocblas_stride strideC,
                             S* workSvec,
                             rocblas_int* workInt)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int j = hipBlockIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid = hipThreadIdx_x;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* W = workSvec + bid * (n * n);

    // temporary arrays in global memory
    rocblas_int* ps = workInt + bid * (5 * n + 2 * blks);
    rocblas_int* bp = ps + 2 * blks + 4 * n;

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // column 'j' belongs to sub-block 'bx' and thus form vector of
    // the new sub-block 'nbx'
    rocblas_int bx = bp[j];
    rocblas_int nbx = bx / dm2;

    // the new sub-block starts at 'pin', and
    // it ends at 'pout'. Its size is 'sz'
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    rocblas_int sz = pout - pin;

    for(auto ii = tid; ii < sz; ii += dim)
    {
        rocblas_int i = ii + pin;
        C[i + j * ldc] = W[i + j * n];
    }
}

/******************* Host functions *********************************************/
/*******************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_NUM_LEVELS returns the ideal number of times/levels in which a matrix
    will be divided during the divide phase of divide & conquer algorithm
    i.e. number of sub-blocks = 2^levels **/
inline rocblas_int stedc_num_levels(const rocblas_int n)
{
    rocblas_int levels;

    if(n <= 16)
        levels = 0;
    else
        levels = std::ceil(std::log2(n)) - 4;

    return levels;
}

//--------------------------------------------------------------------------------------//
/** This helper calculates required workspace size **/
template <bool BATCHED, typename T, typename S>
void rocsolver_stedc_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_tempvect,
                                   size_t* size_workSvec,
                                   size_t* size_workStmp,
                                   size_t* size_workSz,
                                   size_t* size_workInt,
                                   size_t* size_workArr)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // if quick return no workspace needed
    if(n <= 1 || !batch_count)
    {
        *size_tempvect = 0;
        *size_workSvec = 0;
        *size_workStmp = 0;
        *size_workArr = 0;
        *size_workInt = 0;
        *size_workSz = 0;
        return;
    }

    // if no eigenvectors required with classic solver
    if(evect == rocblas_evect_none)
    {
        *size_tempvect = 0;
        *size_workStmp = 0;
        *size_workArr = 0;
        *size_workInt = 0;
        *size_workSz = 0;
        rocsolver_sterf_getMemorySize<S>(n, batch_count, size_workSvec);
    }

    // if size is too small with classic solver
    else if(n < STEDC_MIN_DC_SIZE)
    {
        *size_tempvect = 0;
        *size_workStmp = 0;
        *size_workArr = 0;
        *size_workInt = 0;
        *size_workSz = 0;
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_workSvec);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        // find number of sub-blocks
        rocblas_int levs = stedc_num_levels(n);
        rocblas_int blks = 1 << levs;

        // requirements for batched operations
        *size_workArr = 0;
        if(batch_count > 1)
        {
            if(BATCHED && !COMPLEX)
                *size_workArr = sizeof(S*) * batch_count;
        }
        else
        {
            if(STEDC_USE_EXTERNAL_UPDATE && !STEDC_WITH_STRIDED_BATCHED_GEMM)
            {
                rocblas_int max_n_merges = 1 << (levs - 1);
                *size_workArr = sizeof(S*) * max_n_merges * 3;
            }
        }

        // requirements for solver of small independent blocks
        size_t vec1;
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, &vec1);

        // extra requirements for original eigenvectors when needed
        if(evect != rocblas_evect_tridiagonal)
            *size_tempvect = sizeof(S) * (n * n) * batch_count;
        else
            *size_tempvect = 0;

        // extra requirements for divde and conquer process
        size_t vec2 = sizeof(S) * (std::max(7, n) * n) * batch_count;
        *size_workSvec = std::max(vec1, vec2);

        *size_workStmp = sizeof(S) * (n * n) * batch_count;

        *size_workInt = sizeof(rocblas_int) * (5 * n + 2 * blks) * batch_count;

        *size_workSz = sizeof(S) * (n)*batch_count;
    }
}

//--------------------------------------------------------------------------------------//
/** This helper check argument correctness for stedc API **/
template <typename T, typename S>
rocblas_status rocsolver_stedc_argCheck(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_int n,
                                        S D,
                                        S E,
                                        T C,
                                        const rocblas_int ldc,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(evect != rocblas_evect_none && evect != rocblas_evect_tridiagonal
       && evect != rocblas_evect_original)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if(evect != rocblas_evect_none && ldc < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n > 1 && !E) || (evect != rocblas_evect_none && n && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//--------------------------------------------------------------------------------------//
/** STEDC templated function **/
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_stedc_template(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_int n,
                                        S* D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        U C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        S* tempvect,
                                        void* workSvec,
                                        S* workStmp,
                                        S* workSz,
                                        rocblas_int* workInt,
                                        S** workArr)
{
    ROCSOLVER_ENTER("stedc", "evect:", evect, "n:", n, "shiftD:", shiftD, "shiftE:", shiftE,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 1 && evect != rocblas_evect_none)
        ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0, stream, C,
                                strideC, n, 1);
    if(n <= 1)
        return rocblas_status_success;

    // if no eigenvectors required with the classic solver, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_template<S>(handle, n, D, shiftD, strideD, E, shiftE, strideE, info,
                                    batch_count, static_cast<rocblas_int*>(workSvec));
    }

    // if size is too small with classic solver, use steqr
    else if(n < STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_template<T>(handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C,
                                    shiftC, ldc, strideC, info, batch_count, workSvec);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        S* workSvecs = (S*)workSvec;

        // everything must be executed with scalars on the host
        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle, &old_mode);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        S one = 1.0;
        S zero = 0.0;

        // numerics constants
        S eps = get_epsilon<S>();
        S ssfmin = get_safemin<S>();
        S ssfmax = S(1.0) / ssfmin;
        ssfmin = sqrt(ssfmin) / (eps * eps);
        ssfmax = sqrt(ssfmax) / S(3.0);

        // find number of sub-blocks
        rocblas_int levs = stedc_num_levels(n);
        rocblas_int blks = 1 << levs;

        // initialize identity matrix in V
        // if evect is tridiagonal we can store V directly in C
        // otherwise, they must be kept separate to compute C*V
        S* V = tempvect;
        rocblas_int ldv = n;
        rocblas_stride strideV = n * n;
        if(evect == rocblas_evect_tridiagonal)
        {
            V = (S*)(C + shiftC);
            ldv = (rocblas_int)(sizeof(T) / sizeof(S)) * ldc;
            strideV = (rocblas_int)(sizeof(T) / sizeof(S)) * strideC;
        }
        rocblas_int groupsn = (n - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(groupsn, groupsn, batch_count), dim3(BS2, BS2),
                                0, stream, n, n, V, 0, ldv, strideV);

        // 1. divide phase
        //-----------------------------
        rocblas_int groups = (batch_count - 1) / STEDC_BDIM + 1;
        ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<S>), dim3(groups), dim3(STEDC_BDIM), 0, stream,
                                levs, blks, n, D + shiftD, strideD, E + shiftE, strideE,
                                batch_count, workInt);

        // 2. solve phase
        //-----------------------------
        ROCSOLVER_LAUNCH_KERNEL((stedc_solve_kernel<S>), dim3(blks, batch_count),
                                dim3(STEDC_BDIM_SOLVE), 0, stream, levs, blks, n, D + shiftD,
                                strideD, E + shiftE, strideE, V, 0, ldv, strideV, info, workSvecs,
                                workInt, eps, ssfmin, ssfmax);

        // 3. merge phase
        //----------------
        rocblas_int numgrps = (n - 1) / STEDC_BDIM + 1;
        rocblas_int nng = (n - 1) / STEDC_BDIM_VALUES + 1;

        // prepare for batched gemms if necessary
        // using this approach only in the non-batch syevd calls
        bool use_batched_gemm = (batch_count == 1 && !STEDC_WITH_STRIDED_BATCHED_GEMM);
        bool use_strided_batched_gemm = (batch_count == 1 && STEDC_WITH_STRIDED_BATCHED_GEMM);
        std::vector<rocblas_int> ns(blks);
        rocblas_int res, bc, dm, nn, shv, shw, stv, stw;
        rocblas_int dm2 = 1;
        rocblas_int bbs = blks;

        if(STEDC_USE_EXTERNAL_UPDATE && use_strided_batched_gemm)
        {
            rocblas_int sz = n / blks;
            res = n - sz * blks;
            if(res < blks / 2)
            {
                res = blks - res;
                for(auto i = 0; i < blks; ++i)
                    ns[i] = i < res ? sz : sz + 1;
            }
            else
            {
                for(auto i = 0; i < blks; ++i)
                    ns[i] = i < res ? sz + 1 : sz;
            }
        }

        // ****************** launch merge for level k **********************//
        // ------------------------------------------------------------------//
        for(auto k = 0; k < levs; ++k)
        {
            // a. merge sort and deflation
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeSort_kernel<S>), dim3(numgrps, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, D + shiftD,
                                    strideD, V, 0, ldv, strideV, workSvecs, workInt);

            rocblas_int ngps = blks / (1 << (k + 1));
            size_t lmemsize = sizeof(S) * blks + sizeof(rocblas_int) * 2 * (1 << (k + 1));
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeSequences_kernel<S>), dim3(ngps, batch_count),
                                    dim3(STEDC_BDIM), lmemsize, stream, levs, blks, k, n,
                                    E + shiftE, strideE, workSvecs, workInt, eps);

            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeDeflate_kernel<S>), dim3(numgrps, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, workSvecs,
                                    workInt, eps);

            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_kernel<S>), dim3(n, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, E + shiftE,
                                    strideE, workSvecs, workStmp, workInt);

            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeRotate_kernel<S>), dim3(n, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, V, 0, ldv,
                                    strideV, workSvecs, workInt);

            // b. compute new values
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_kernel<S>), dim3(nng, batch_count),
                                    dim3(STEDC_BDIM_VALUES), 0, stream, levs, blks, k, n, E + shiftE,
                                    strideE, workSvecs, workStmp, workInt, eps, ssfmin, ssfmax);

            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeReinsert_kernel<S>), dim3(numgrps, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, D + shiftD,
                                    strideD, E + shiftE, strideE, workSvecs, workInt);

            // c. compute new vectors
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeRescale_kernel<S>), dim3(n, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, E + shiftE,
                                    strideE, workSvecs, workStmp, workSz, workInt);

            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeVectors_kernel<S>), dim3(n, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, E + shiftE,
                                    strideE, workSvecs, workStmp, workSz, workInt);

            // d. vector updates
            if(STEDC_USE_EXTERNAL_UPDATE)
            {
                ROCSOLVER_LAUNCH_KERNEL(stedc_mergePrepgemm1_kernel<S>, dim3(n, batch_count),
                                        dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, workStmp,
                                        workInt);

                ROCSOLVER_LAUNCH_KERNEL(stedc_mergePrepgemm_kernel<S>, dim3(n, batch_count),
                                        dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, E + shiftE,
                                        strideE, workSvecs, workStmp, workInt);

                if(use_strided_batched_gemm)
                {
                    HIP_CHECK(hipMemsetAsync((void*)(workSvecs), 0, sizeof(S) * (n * n), stream));

                    dm = dm2;
                    dm2 *= 2;
                    res /= 2;
                    bbs /= 2;
                    rocblas_int idx = res;
                    for(auto kk = 0; kk < blks; kk += dm2)
                        ns[kk] = ns[kk] + ns[kk + dm];

                    // first batch call
                    shv = 0;
                    shw = 0;
                    bc = idx;
                    if(bc > 0)
                    {
                        nn = ns[(idx - 1) * dm2];
                        stv = nn * (ldv + 1);
                        stw = nn * (n + 1);
                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nn,
                                       nn, nn, &one, V, shv, ldv, stv, workStmp, shw, n, stw, &zero,
                                       workSvecs, shw, n, stw, bc, workArr);

                        shv = bc * nn * (ldv + 1);
                        shw = bc * nn * (n + 1);
                    }

                    // middle batch call
                    if(idx < bbs - 1)
                    {
                        nn = ns[idx * dm2];
                        bc = (nn == ns[(idx + 1) * dm2]) ? 0 : 1;
                        if(bc > 0)
                        {
                            stv = nn * (ldv + 1);
                            stw = nn * (n + 1);
                            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                                           nn, nn, nn, &one, V, shv, ldv, stv, workStmp, shw, n,
                                           stw, &zero, workSvecs, shw, n, stw, bc, workArr);

                            shv += bc * nn * (ldv + 1);
                            shw += bc * nn * (n + 1);
                        }
                        idx += bc;
                    }

                    // last batch call
                    bc = bbs - idx;
                    if(bc > 0)
                    {
                        nn = ns[idx * dm2];
                        stv = nn * (ldv + 1);
                        stw = nn * (n + 1);
                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nn,
                                       nn, nn, &one, V, shv, ldv, stv, workStmp, shw, n, stw, &zero,
                                       workSvecs, shw, n, stw, bc, workArr);
                    }
                }

                else if(use_batched_gemm)
                {
                    HIP_CHECK(hipMemsetAsync((void*)(workSvecs), 0, sizeof(S) * (n * n), stream));
                    rocblas_int n_merges = 1 << (levs - k - 1);

                    if(n % n_merges == 0)
                    {
                        // if all sub-blocks are of same size, only one uniform batch call is required
                        nn = n / n_merges;
                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nn,
                                       nn, nn, &one, V, 0, ldv, nn * ldv + nn, workStmp, 0, n,
                                       nn * n + nn, &zero, workSvecs, 0, n, nn * n + nn, n_merges,
                                       workArr);
                    }
                    else
                    {
                        // otherwise 2 batched calls, with sizes ns[0] and ns[0] + 1, are required
                        ns[0] = n;
                        rocblas_int t, t2;
                        for(auto i = 0; i < levs - k - 1; ++i)
                        {
                            for(auto j = (1 << i); j > 0; --j)
                            {
                                t = ns[j - 1];
                                t2 = t / 2;
                                ns[j * 2 - 1] = (2 * t2 < t) ? t2 + 1 : t2;
                                ns[j * 2 - 2] = t2;
                            }
                        }

                        std::array<std::vector<rocblas_int>, 2> uniform_batch;
                        uniform_batch[0].reserve(n_merges);
                        uniform_batch[1].reserve(n_merges);
                        for(rocblas_int i = 0, ps = 0; i < n_merges; ps += ns[i++])
                            uniform_batch[ns[i] != ns[0]].push_back(ps);
                        for(rocblas_int i = 0, nsb = ns[0]; i < 2; ++i, ++nsb)
                        {
                            auto& b = uniform_batch[i];
                            auto nbb = b.size();
                            std::vector<S*> hABC(nbb * 3);
                            for(size_t j = 0; j < nbb; ++j)
                            {
                                auto ps = b[j];
                                hABC[j + 0 * nbb] = ps + ps * ldv + V;
                                hABC[j + 1 * nbb] = ps + ps * n + workStmp;
                                hABC[j + 2 * nbb] = ps + ps * n + workSvecs;
                            }
                            HIP_CHECK(hipMemcpyAsync(workArr, hABC.data(), 3 * nbb * sizeof(S*),
                                                     hipMemcpyHostToDevice, stream));
                            rocsolver_gemm<S, rocblas_int, S* const*, S* const*, S* const*>(
                                handle, rocblas_operation_none, rocblas_operation_none, nsb, nsb,
                                nsb, &one, workArr, 0, ldv, 0, workArr + nbb, 0, n, 0, &zero,
                                workArr + 2 * nbb, 0, n, 0, nbb, nullptr);
                        }
                    }
                }

                else
                {
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n,
                                   &one, V, 0, ldv, strideV, workStmp, 0, n, n * n, &zero,
                                   workSvecs, 0, n, n * n, batch_count, workArr);
                }
            }

            // e. update for next level
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeUpdate_kernel<S>), dim3(n, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, V, 0, ldv,
                                    strideV, workSvecs, workInt);
        }

        // 4. Final update
        //----------------------
        if(evect != rocblas_evect_tridiagonal)
        {
            // eigenvectors C <- C*V
            local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, V, workSvecs,
                                            workStmp, 0, ldv, strideV, batch_count, workArr);
        }
        else if constexpr(rocblas_is_complex<T>)
        {
            // V is stored in C but is of type S; need to convert to type T
            // tempgemm = V
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<S>, dim3(groupsn, groupsn, batch_count), dim3(BS2, BS2),
                                    0, stream, copymat_to_buffer, n, n, V, 0, ldv, strideV, workStmp);

            // imag(C) = zeros
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(groupsn, groupsn, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, C, shiftC, ldc, strideC);

            // real(C) = tempgemm
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(groupsn, groupsn, batch_count),
                                    dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, C, shiftC,
                                    ldc, strideC, workStmp);
        }

        rocblas_set_pointer_mode(handle, old_mode);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
