/************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_stebz.hpp"
#include "auxiliary/rocauxiliary_stein.hpp"
#include "auxiliary/rocauxiliary_steqr.hpp"
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

#define STEDCX_BDIM 512 // Number of threads per thread-block used in main stedc kernels

// TODO: using macro STEDCX_EXTERNAL_GEMM = false for now. We can enable the use of
// external gemm updates once the development is completed for stedc.
#define STEDCX_EXTERNAL_GEMM false

/***************** Device auxiliary functions *****************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDCX_NUM_LEVELS returns the ideal number of times/levels in which a matrix (or split block)
    will be divided during the divide phase of divide & conquer algorithm.
    i.e. number of sub-blocks = 2^levels **/
__host__ __device__ inline rocblas_int stedcx_num_levels(const rocblas_int n)
{
    rocblas_int levels = 0;
    // return the max number of levels such that the sub-blocks are at least of size 1
    // (i.e. 2^levels <= n), and there are no more than 256 sub-blocks (i.e. 2^levels <= 256)
    if(n <= 2)
        return levels;

    // TODO: tuning will be required; using the same tuning as QR for now
    if(n <= 4)
    {
        levels = 1;
    }
    else if(n <= 32)
    {
        levels = 2;
    }
    else if(n <= 232)
    {
        levels = 4;
    }
    else
    {
        if(n <= 1946)
        {
            if(n <= 1692)
            {
                if(n <= 295)
                {
                    levels = 5;
                }
                else
                {
                    levels = 7;
                }
            }
            else
            {
                levels = 7;
            }
        }
        else
        {
            levels = 8;
        }
    }

    return levels;
}

/*************** Main kernels *********************************************************/
/**************************************************************************************/

/** This kernel deals with the case n = 1
    (one split block and a single eigenvalue which is the element in D) **/
template <typename S>
ROCSOLVER_KERNEL void stedcx_case1_kernel(const rocblas_erange range,
                                          const S vlow,
                                          const S vup,
                                          S* DA,
                                          const rocblas_stride strideD,
                                          rocblas_int* nev,
                                          S* WA,
                                          const rocblas_stride strideW)
{
    int bid = hipBlockIdx_x;

    // select batch instance
    S* D = DA + bid * strideD;
    S* W = WA + bid * strideW;

    // check if diagonal element is in range and return
    S d = D[0];
    if(range == rocblas_erange_value && (d <= vlow || d > vup))
    {
        nev[bid] = 0;
    }
    else
    {
        nev[bid] = 1;
        W[0] = d;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDCX_SPLIT_KERNEL splits the matrix into independent blocks and determines range
    for the partial decomposition **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEBZ_SPLIT_THDS)
    stedcx_split_kernel(const rocblas_erange range,
                        const rocblas_int n,
                        const S vl,
                        const S vu,
                        const rocblas_int il,
                        const rocblas_int iu,
                        S* DD,
                        const rocblas_stride strideD,
                        S* EE,
                        const rocblas_stride strideE,
                        S* WW,
                        const rocblas_stride strideW,
                        rocblas_int* splitsA,
                        S* workA,
                        const S eps,
                        const S ssfmin)
{
    // batch instance
    const int tid = hipThreadIdx_x;
    const int bid = hipBlockIdx_y;
    const int bdim = hipBlockDim_x;
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // workspace
    rocblas_int* ninter = splits + n + 2;
    rocblas_int* tmpIS = ninter + 2 * n;
    // using W as temp array to store the spit off-diagonal
    // (to use in case range = index)
    S* W = WW + bid * strideW;
    //nsplit is not needed; the number of split blocks goes into last entry
    //of splits when compact = true
    bool compact = true;
    rocblas_int* nsplit = nullptr;
    // range bounds
    S* bounds = workA + bid * (4 * n + 2);
    S* pivmin = bounds + 2;
    S* Esqr = pivmin + 1;
    S* Dcpy = Esqr + n - 1;
    S* inter = Dcpy + n;

    // make copy of D for future use if necessary
    if(range == rocblas_erange_index)
    {
        for(rocblas_int i = tid; i < n; i += bdim)
            Dcpy[i] = D[i];
    }

    // shared memory setup for iamax.
    // (sidx also temporarily stores the number of blocks found by each thread)
    __shared__ S sval[STEBZ_SPLIT_THDS];
    __shared__ rocblas_int sidx[STEBZ_SPLIT_THDS];

    run_stebz_splitting<STEBZ_SPLIT_THDS>(tid, range, n, vl, vu, il, iu, D, E, nsplit, W, splits,
                                          tmpIS, pivmin, Esqr, bounds, inter, ninter, sval, sidx,
                                          eps, ssfmin, compact);
}

//--------------------------------------------------------------------------------------//
/** STEDCX_DIVIDE_KERNEL implements the divide phase of the DC algorithm. It
   divides each split-block into a number of sub-blocks.
        - Call this kernel with batch_count groups in x. Groups are of size
   STEDCX_BDIM.
        - If there are actually more split-blocks than STEDCX_BDIM, some threads
   will work with more than one split-block sequentially. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_BDIM)
    stedcx_divide_kernel(const rocblas_int n,
                         S* DD,
                         const rocblas_stride strideD,
                         S* EE,
                         const rocblas_stride strideE,
                         rocblas_int* splitsA)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_x;
    // split-block id
    rocblas_int sid = hipThreadIdx_x;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDCX_BDIM split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDCX_BDIM)
    {
        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks in split-block
        levs = stedcx_num_levels(bs);
        blks = 1 << levs;

        // 1. DIVIDE PHASE
        /* ----------------------------------------------------------------- */
        // (artificially divide split-block into blks sub-blocks
        // find initial positions of each sub-blocks)

        // find sizes of sub-blocks
        ns[0] = bs;
        rocblas_int t, t2;
        for(int i = 0; i < levs; ++i)
        {
            for(int j = (1 << i); j > 0; --j)
            {
                t = ns[j - 1];
                t2 = t / 2;
                ns[j * 2 - 1] = (2 * t2 < t) ? t2 + 1 : t2;
                ns[j * 2 - 2] = t2;
            }
        }

        // find beginning of sub-blocks and update D elements
        p2 = p1;
        ps[0] = p2;
        for(int i = 1; i < blks; ++i)
        {
            p2 += ns[i - 1];
            ps[i] = p2;

            // perform sub-block division
            p = E[p2 - 1];
            D[p2] -= p;
            D[p2 - 1] -= p;
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDCX_SOLVE_KERNEL implements the solver phase of the DC algorithm to
        compute the eigenvalues/eigenvectors of the different sub-blocks of each
   split-block. A matrix in the batch could have many split-blocks, and each
   split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS
   groups in y. Groups are single-thread.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
   be analysed in parallel). If there are actually more split-blocks, some
   groups will work with more than one split-block sequentially. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_BDIM)
    stedcx_solve_kernel(const rocblas_int n,
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
                        rocblas_int* splitsA,
                        const S eps,
                        const S ssfmin,
                        const S ssfmax)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split-block id
    rocblas_int sid = hipBlockIdx_y;
    // sub-block id
    rocblas_int tid = hipBlockIdx_x;
    // thread index
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tidb_inc = hipBlockDim_x;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* info = iinfo + bid;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // workspace for solvers
    S* W = WA + bid * (2 * n);
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // size of sub block
    rocblas_int sbs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        levs = stedcx_num_levels(bs);
        blks = 1 << levs;

        // 2. SOLVE PHASE
        /* ----------------------------------------------------------------- */
        // Solve the blks sub-blocks in parallel.

        if(tid < blks)
        {
            sbs = ns[tid];
            p2 = ps[tid];

            run_steqr(tidb, tidb_inc, sbs, D + p2, E + p2, C + p2 + p2 * ldc, ldc, info, W + p2 * 2,
                      30 * bs, eps, ssfmin, ssfmax, false);
            __syncthreads();
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDCX_SYNTHESIS_KERNEL synthesizes the results of the partial decomposition **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_BDIM)
    stedcx_synthesis_kernel(const rocblas_erange range,
                            const rocblas_int n,
                            const rocblas_int il,
                            const rocblas_int iu,
                            S* DD,
                            const rocblas_stride strideD,
                            rocblas_int* nevA,
                            S* WW,
                            const rocblas_stride strideW,
                            S* VV,
                            const rocblas_int ldv,
                            const rocblas_stride strideV,
                            const rocblas_int batch_count,
                            rocblas_int* splitsA,
                            S* workA,
                            const S eps)
{
    // batch instance
    const int tid = hipThreadIdx_x;
    const int bid = hipBlockIdx_y;
    const int bdim = hipBlockDim_x;
    S* D = DD + bid * strideD;
    S* W = WW + bid * strideW;
    S* V = VV + bid * strideV;
    rocblas_int* nev = nevA + bid;
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // workspace
    rocblas_int* ninter = splits + n + 2;
    rocblas_int* idd = ninter + 2 * n;
    // range bounds
    S* bounds = workA + bid * (4 * n + 2);
    S* pmin = bounds + 2;
    S* Esqr = pmin + 1;
    S* Dcpy = Esqr + n - 1;
    S* inter = Dcpy + n;

    // aux variables
    S tmp, tmp2;
    rocblas_int nn = 0, nnt = 0, ntmp = 0;
    bool index = (range == rocblas_erange_index);
    bool all = (range == rocblas_erange_all);
    S low, up;

    // if computing all eigenvalues, quick return
    if(all)
    {
        *nev = n;
        for(int k = tid; k < n; k += bdim)
            W[k] = D[k];
        return;
    }

    // otherwise, only keep eigenvalues in desired range
    if(tid == 0)
    {
        low = bounds[0];
        up = bounds[1];

        if(!index)
        {
            for(int k = 0; k < n; ++k)
            {
                tmp = D[k];
                idd[k] = 0;
                if(tmp >= low && tmp <= up)
                {
                    idd[k] = 1;
                    W[nn] = tmp;
                    nn++;
                }
            }
        }

        else
        {
            for(int k = 0; k < n; ++k)
            {
                tmp = D[k];
                idd[k] = 0;
                if(tmp >= low && tmp <= up)
                {
                    idd[k] = 1;
                    inter[nnt] = tmp;
                    inter[nnt + n] = tmp;
                    ninter[nnt] = k;
                    nnt++;
                }
            }

            // discard extra values
            increasing_order(nnt, inter + n, (rocblas_int*)nullptr);
            for(int i = 0; i < nnt; ++i)
            {
                tmp = inter[i];
                for(int j = 0; j < nnt; ++j)
                {
                    tmp2 = inter[n + j];
                    if(tmp == tmp2)
                    {
                        tmp2 = (j == nnt - 1) ? (up - tmp2) / 2 : (inter[n + j + 1] - tmp2) / 2;
                        tmp2 += tmp;
                        ntmp = sturm_count(n, Dcpy, Esqr, *pmin, tmp2);
                        if(ntmp >= il && ntmp <= iu)
                        {
                            W[nn] = tmp;
                            nn++;
                        }
                        else
                            idd[ninter[i]] = 0;
                        break;
                    }
                }
            }
        }

        // final total of number of eigenvalues in desired range
        *nev = nn;
    }
    __syncthreads();

    // and keep corresponding eigenvectors
    nn = 0;
    for(int j = 0; j < n; ++j)
    {
        if(idd[j] == 1)
        {
            if(j != nn)
            {
                for(int i = tid; i < n; i += bdim)
                    V[i + nn * ldv] = V[i + j * ldv];
            }
            nn++;
        }
        __syncthreads();
    }
}

//--------------------------------------------------------------------------------------//
/** STEDCX_MERGEPREPARE_KERNEL performs deflation and prepares the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as half of the unmerged sub-blocks in current level. Each group works
          with a merge. Groups are size STEDCX_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than half the actual number of unmerged sub-blocks
          in the level, it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_BDIM)
    stedcx_mergePrepare_kernel(const rocblas_int k,
                               const rocblas_int n,
                               S* DD,
                               const rocblas_stride strideD,
                               S* EE,
                               const rocblas_stride strideE,
                               S* CC,
                               const rocblas_int shiftC,
                               const rocblas_int ldc,
                               const rocblas_stride strideC,
                               S* tmpzA,
                               S* vecsA,
                               rocblas_int* splitsA,
                               const S eps)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split block id
    rocblas_int sid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int mid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tid, tx;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = psA + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);
    /* --------------------------------------------------- */

    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    extern __shared__ rocblas_int lsmem[];
    // used to store temp values during the different reductions
    S* inrmsd = reinterpret_cast<S*>(lsmem);
    S* inrmsz = inrmsd + hipBlockDim_x;
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    rocblas_int tn;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        __syncthreads();

        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        // tn is the number of thread-groups needed
        levs = stedcx_num_levels(bs);
        blks = levs - 1 - k;
        tn = (blks < 0) ? 0 : 1 << blks;

        // 3. MERGE PHASE
        /* ----------------------------------------------------------------- */
        // Work with merges on level k. A thread-group works with two leaves in the merge tree.
        if(mid < tn)
        {
            rocblas_int iam, sz, bdm, dim, dim2;
            S* ptz;
            rocblas_int bd = 1 << k;
            bdm = bd << 1;
            dim = hipBlockDim_x / 2;

            // tid indexes the sub-blocks in the entire split block
            // iam indexes the sub-blocks in the context of the merge
            // (according to its level in the merge tree)
            iam = tidb / dim;
            tx = tidb % dim;
            tid = mid * bdm + iam * bd;
            p2 = ps[tid];

            // 3a. find rank-1 modification components (z and p) for this merge
            /* ----------------------------------------------------------------- */
            // Threads with iam = 0 work with components below the merge point;
            // threads with iam = 1 work above the merge point
            sz = ns[tid];
            for(int j = 1; j < bd; ++j)
                sz += ns[tid + j];
            // with this, all threads involved in a merge
            // will point to the same row of C and the same off-diag element
            ptz = (iam == 0) ? C + p2 - 1 + sz : C + p2;
            p = (iam == 0) ? 2 * E[p2 - 1 + sz] : 2 * E[p2 - 1];

            // copy elements of z
            for(int j = tx; j < sz; j += dim)
                z[p2 + j] = ptz[(p2 + j) * ldc] / sqrt(2);
            /* ----------------------------------------------------------------- */

            // 3b. calculate deflation tolerance
            /* ----------------------------------------------------------------- */
            // compute maximum of diagonal and z in each merge block
            S valf, valg, maxd, maxz;
            maxd = 0;
            maxz = 0;
            for(int i = tx; i < sz; i += dim)
            {
                valf = std::abs(D[p2 + i]);
                valg = std::abs(z[p2 + i]);
                maxd = valf > maxd ? valf : maxd;
                maxz = valg > maxz ? valg : maxz;
            }
            inrmsd[tidb] = maxd;
            inrmsz[tidb] = maxz;
            __syncthreads();

            dim2 = dim;
            while(dim2 > 0)
            {
                if(tidb < dim2)
                {
                    valf = inrmsd[tidb + dim2];
                    valg = inrmsz[tidb + dim2];
                    maxd = valf > maxd ? valf : maxd;
                    maxz = valg > maxz ? valg : maxz;
                    inrmsd[tidb] = maxd;
                    inrmsz[tidb] = maxz;
                }
                dim2 /= 2;
                __syncthreads();
            }

            // tol should be  8 * eps * (max diagonal or z element participating in
            // merge)
            maxd = inrmsd[0];
            maxz = inrmsz[0];
            maxd = maxz > maxd ? maxz : maxd;

            S tol = 8 * eps * maxd;
            /* ----------------------------------------------------------------- */

            // 3c. deflate eigenvalues
            /* ----------------------------------------------------------------- */
            // determine boundaries of what would be the new merged sub-block
            // 'in' will be its initial position.
            // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
            rocblas_int in = tid - iam * bd;
            sz = ns[in];
            for(int i = 1; i < bdm; ++i)
                sz += ns[in + i];
            in = ps[in];

            // first deflate zero components
            S f, g, c, s, rr;
            for(int i = tidb; i < sz; i += hipBlockDim_x)
            {
                tx = in + i;
                g = z[tx];
                if(abs(p * g) <= tol)
                    // deflated ev because component in z is zero
                    idd[tx] = 0;
                else
                    idd[tx] = 1;
            }
            __syncthreads();

            // now deflate repeated values
            rocblas_int sz_even, sz_half, base, top, com;
            sz_even = (sz % 2 == 1) ? sz + 1 : sz;
            sz_half = sz_even / 2;

            // the number of rounds needed is sz_even - 1
            for(int r = 0; r < sz_even - 1; ++r)
            {
                // in each round threads analyze pairs of values in parallel
                // sz_half pairs are needed
                for(int i = tidb; i < sz_half; i += hipBlockDim_x)
                {
                    // determine pair of values (base, top)
                    com = 2 * (i - r);
                    base = (i == 0)             ? 0
                        : (r < i)               ? com
                        : (r > i - 1 + sz_half) ? 2 * (sz_even - 1) + com
                                                : 1 - com;

                    com = 2 * (i + r);
                    top = (r < sz_half - i)     ? 1 + com
                        : (r > sz_even - 2 - i) ? 3 - 2 * sz_even + com
                                                : 2 * (sz_even - 1) - com;

                    if(base > top)
                    {
                        com = base;
                        base = top;
                        top = com;
                    }

                    // compare values and deflate if needed
                    base += in;
                    top += in;
                    if(idd[base] == 1 && idd[top] == 1 && top < sz + in)
                    {
                        if(abs(D[base] - D[top]) <= tol)
                        {
                            // deflated ev because it is repeated
                            idd[top] = 0;
                            // rotation to eliminate component in z
                            g = z[top];
                            f = z[base];
                            lartg(f, g, c, s, rr);
                            z[base] = rr;
                            z[top] = 0;
                            // update C with the rotation
                            for(int ii = 0; ii < n; ++ii)
                            {
                                valf = C[ii + base * ldc];
                                valg = C[ii + top * ldc];
                                C[ii + base * ldc] = valf * c - valg * s;
                                C[ii + top * ldc] = valf * s + valg * c;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            /* ----------------------------------------------------------------- */

            // 3d.1. Organize data with non-deflated values to prepare secular equation
            /* ----------------------------------------------------------------- */
            // define shifted arrays
            S* tmpd = temps + in * n;
            S* diag = D + in;
            rocblas_int* mask = idd + in;
            S* zz = z + in;
            rocblas_int* per = pers + in;
            S* ev = evs + in;

            // find degree and components of secular equation
            // tmpd contains the non-deflated diagonal elements (ie. poles of the
            // secular eqn) zz contains the corresponding non-zero elements of the
            // rank-1 modif vector
            rocblas_int dd = 0;
            for(int i = 0; i < sz; ++i)
            {
                if(mask[i] == 1)
                {
                    if(tidb == 0)
                    {
                        per[dd] = i;
                        tmpd[dd] = p < 0 ? -diag[i] : diag[i];
                        if(dd != i)
                            zz[dd] = zz[i];
                    }
                    dd++;
                }
            }
            /* ----------------------------------------------------------------- */
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDCX_MERGEVALUES_KERNEL solves the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as half of the unmerged sub-blocks in current level. Each group works
          with a merge. Groups are size STEDCX_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than half the actual number of unmerged sub-blocks
          in the level, it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_BDIM)
    stedcx_mergeValues_kernel(const rocblas_int k,
                              const rocblas_int n,
                              S* DD,
                              const rocblas_stride strideD,
                              S* EE,
                              const rocblas_stride strideE,
                              S* tmpzA,
                              S* vecsA,
                              rocblas_int* splitsA,
                              const S eps,
                              const S ssfmin,
                              const S ssfmax)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split block id
    rocblas_int sid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int mid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tid;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = psA + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    rocblas_int tn;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        __syncthreads();

        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        // tn is the number of thread-groups needed
        levs = stedcx_num_levels(bs);
        blks = levs - 1 - k;
        tn = (blks < 0) ? 0 : 1 << blks;
        blks = 1 << levs;

        // 3. MERGE PHASE
        /* ----------------------------------------------------------------- */
        // Work with merges on level k. A thread-group works with two leaves in the merge tree;
        // all threads work together to solve the secular equation.
        if(mid < tn)
        {
            rocblas_int iam, sz, bdm, dim;
            S valf, valg;
            rocblas_int bd = 1 << k;
            bdm = bd << 1;
            dim = hipBlockDim_x / 2;

            // tid indexes the sub-blocks in the entire split block
            // iam indexes the sub-blocks in the context of the merge
            // (according to its level in the merge tree)
            iam = tidb / dim;
            tid = mid * bdm + iam * bd;
            p2 = ps[tid];

            // Find off-diagonal element of the merge
            // Threads with iam = 0 work with components below the merge point;
            // threads with iam = 1 work above the merge point
            sz = ns[tid];
            for(int j = 1; j < bd; ++j)
                sz += ns[tid + j];
            // with this, all threads involved in a merge
            // will point to the same row of C and the same off-diag element
            p = (iam == 0) ? 2 * E[p2 - 1 + sz] : 2 * E[p2 - 1];

            // determine boundaries of what would be the new merged sub-block
            // 'in' will be its initial position.
            // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
            rocblas_int in = tid - iam * bd;
            sz = ns[in];
            for(int i = 1; i < bdm; ++i)
                sz += ns[in + i];
            in = ps[in];

            // 3d.2. Organize data with non-deflated values to prepare secular equation
            /* ----------------------------------------------------------------- */
            rocblas_int tsz = 1 << (levs - 1 - k);
            tsz = (bs - 1) / tsz + 1;

            // All threads of the group participating in the merge will work together
            // to solve the correspondinbg secular eqn. Now 'iam' indexes those threads
            iam = tidb;
            bdm = hipBlockDim_x;

            // define shifted arrays
            S* tmpd = temps + in * n;
            S* ev = evs + in;
            S* diag = D + in;
            rocblas_int* mask = idd + in;
            S* zz = z + in;
            rocblas_int* per = pers + in;

            // find degree of secular equation
            rocblas_int dd = 0;
            for(int i = 0; i < sz; ++i)
            {
                if(mask[i] == 1)
                    dd++;
            }

            // Order the elements in tmpd and zz using a simple parallel selection/bubble sort.
            // This will allow us to find initial intervals for eigenvalue guesses
            for(int i = 0; i < dd; i++)
            {
                for(int j = 2 * iam + i % 2; j < dd - 1; j += 2 * bdm)
                {
                    if(tmpd[j] > tmpd[j + 1])
                    {
                        swap(tmpd[j], tmpd[j + 1]);
                        swap(zz[j], zz[j + 1]);
                        swap(per[j], per[j + 1]);
                    }
                }
                __syncthreads();
            }

            // make dd copies of the non-deflated ordered diagonal elements
            // (i.e. the poles of the secular eqn) so that the distances to the
            // eigenvalues (D - lambda_i) are updated while computing each eigenvalue.
            // This will prevent collapses and division by zero when an eigenvalue
            // is too close to a pole.
            for(int i = iam; i < dd; i += bdm)
            {
                for(int j = i + n; j < i + sz * n; j += n)
                    tmpd[j] = tmpd[i];
            }

            // finally copy over all diagonal elements in ev. ev will be overwritten
            // by the new computed eigenvalues of the merged block
            for(int i = iam; i < sz; i += bdm)
                ev[i] = diag[i];
            __syncthreads();
            /* ----------------------------------------------------------------- */

            // 3e. Solve secular eqns, i.e. find the dd zeros
            // corresponding to non-deflated new eigenvalues of the merged block
            /* ----------------------------------------------------------------- */
            // each thread will find a different zero in parallel
            S a, b;
            for(int j = iam; j < sz; j += bdm)
            {
                if(mask[j] == 1)
                {
                    // find position in the ordered array
                    valf = p < 0 ? -ev[j] : ev[j];
                    int count = dd, cc = 0;
                    while(count > 0)
                    {
                        auto step = count / 2;
                        auto it = cc + step;
                        if(tmpd[it + j * n] < valf)
                        {
                            cc = ++it;
                            count -= step + 1;
                        }
                        else
                            count = step;
                    }

                    // computed zero will overwrite 'ev' at the corresponding position.
                    // 'tmpd' will be updated with the distances D - lambda_i.
                    // deflated values are not changed.
                    rocblas_int linfo;

#if defined(ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
                    linfo = slaed4(dd, cc, tmpd + j * n, zz, std::abs(p), ev[j]);
#else
                    if(cc == dd - 1)
                        linfo = seq_solve_ext(dd, tmpd + j * n, zz, (p < 0 ? -p : p), ev + j, eps,
                                              ssfmin, ssfmax);
                    else
                        linfo = seq_solve(dd, tmpd + j * n, zz, (p < 0 ? -p : p), cc, ev + j, eps,
                                          ssfmin, ssfmax);
#endif

                    if(p < 0)
                        ev[j] *= -1;
                }
            }
            __syncthreads();

            // Re-scale vector Z to avoid bad numerics when an eigenvalue
            // is too close to a pole
            for(int i = iam; i < dd; i += bdm)
            {
                valf = 1;
                for(int j = 0; j < sz; ++j)
                {
                    if(mask[j] == 1)
                    {
                        valg = tmpd[i + j * n];
                        valf *= (per[i] == j) ? valg : valg / (diag[per[i]] - diag[j]);
                    }
                }
                valf = sqrt(std::abs(valf));
                zz[i] = zz[i] < 0 ? -valf : valf;
            }
            /* ----------------------------------------------------------------- */
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDCX_MERGEVECTORS_KERNEL prepares vectors from the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as columns would be in the matrix if its size is exact multiple of nn.
          Each group works with a column. Groups are size STEDCX_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than the actual number of columns n,
          it will do nothing. **/
template <bool USEGEMM, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_BDIM)
    stedcx_mergeVectors_kernel(const rocblas_int k,
                               const rocblas_int n,
                               S* DD,
                               const rocblas_stride strideD,
                               S* EE,
                               const rocblas_stride strideE,
                               S* CC,
                               const rocblas_int shiftC,
                               const rocblas_int ldc,
                               const rocblas_stride strideC,
                               S* tmpzA,
                               S* vecsA,
                               rocblas_int* splitsA)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split block id
    rocblas_int sid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int mid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid, vidb;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = psA + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);
    /* --------------------------------------------------- */

    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    extern __shared__ rocblas_int lsmem[];
    // used to store temp values during the different reductions
    S* inrms = reinterpret_cast<S*>(lsmem);
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    rocblas_int tn;
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        __syncthreads();

        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        levs = stedcx_num_levels(bs);
        blks = 1 << levs;

        // tn is max number of vectors in each sub-block
        tn = (bs - 1) / blks + 1;

        // 3. MERGE PHASE
        /* ----------------------------------------------------------------- */
        // Work with merges on level k. Each thread-group works with one vector.
        if(mid < tn * blks && k < levs)
        {
            rocblas_int iam, sz, bdm;
            S* ptz;
            S valf, valg;
            rocblas_int bd = 1 << k;
            bdm = bd << 1;

            // tid indexes the sub-blocks in the entire split block
            tid = mid / tn;
            p2 = ps[tid];
            // vidb indexes the vectors associated with each sub-block
            vidb = mid % tn;
            // iam indexes the sub-blocks in the context of the merge
            // (according to its level in the merge tree)
            iam = tid % bdm;

            // determine boundaries of what would be the new merged sub-block
            // 'in' will be its initial position
            rocblas_int in = ps[tid - iam];
            // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
            sz = ns[tid];
            for(int i = iam; i > 0; --i)
                sz += ns[tid - i];
            for(int i = bdm - 1 - iam; i > 0; --i)
                sz += ns[tid + i];

            // final number of merged sub-blocks
            rocblas_int tsz = 1 << (levs - 1 - k);
            tsz = (bs - 1) / tsz + 1;

            // define shifted arrays
            S* tmpd = temps + in * n;
            S* ev = evs + in;
            S* diag = D + in;
            rocblas_int* mask = idd + in;
            S* zz = z + in;
            rocblas_int* per = pers + in;

            // find degree of secular equation
            rocblas_int dd = 0;
            for(int i = 0; i < sz; ++i)
            {
                if(mask[i] == 1)
                    dd++;
            }
            __syncthreads();

            // 3f. Prepare vectors corresponding to non-deflated values
            /* ----------------------------------------------------------------- */
            S temp, nrm;
            rocblas_int j = vidb;
            bool go = (j < ns[tid]);
            S* putvec = USEGEMM ? vecs : temps;

            if(go)
            {
                if(idd[p2 + j] == 1)
                {
                    // compute vectors of rank-1 perturbed system and their norms
                    nrm = 0;
                    for(int i = tidb; i < dd; i += dim)
                    {
                        valf = zz[i] / temps[i + (p2 + j) * n];
                        nrm += valf * valf;
                        putvec[i + (p2 + j) * n] = valf;
                    }
                    inrms[tidb] = nrm;
                    __syncthreads();

                    // reduction (for the norms)
                    for(int r = dim / 2; r > 0; r /= 2)
                    {
                        if(tidb < r)
                        {
                            nrm += inrms[tidb + r];
                            inrms[tidb] = nrm;
                        }
                        __syncthreads();
                    }
                    nrm = sqrt(inrms[0]);
                }

                if(USEGEMM)
                {
                    // when using external gemms for the update, we need to
                    // put vectors in padded matrix 'temps'
                    // (this is to compute 'vecs = C * temps' using external gemm call)
                    for(int i = tidb; i < in + sz; i += dim)
                    {
                        if(i >= in && idd[p2 + j] == 1 && idd[i] == 1)
                        {
                            dd = 0;
                            for(int k = in; k < i; ++k)
                            {
                                if(idd[k] == 0)
                                    dd++;
                            }
                            temps[pers[i - dd] + in + (p2 + j) * n]
                                = vecs[i - dd - in + (p2 + j) * n] / nrm;
                        }
                        else
                            temps[i + (p2 + j) * n] = 0;
                    }
                }
                else
                {
                    // otherwise, use internal gemm-like procedure to
                    // multiply by C (row by row)
                    if(idd[p2 + j] == 1)
                    {
                        for(int ii = 0; ii < tsz; ++ii)
                        {
                            rocblas_int i = in + ii;

                            // inner products
                            temp = 0;
                            if(ii < sz)
                            {
                                for(int kk = tidb; kk < dd; kk += dim)
                                    temp += C[i + (per[kk] + in) * ldc] * temps[kk + (p2 + j) * n];
                            }
                            inrms[tidb] = temp;
                            __syncthreads();

                            // reduction
                            for(int r = dim / 2; r > 0; r /= 2)
                            {
                                if(ii < sz && tidb < r)
                                {
                                    temp += inrms[tidb + r];
                                    inrms[tidb] = temp;
                                }
                                __syncthreads();
                            }

                            // result
                            if(ii < sz && tidb == 0)
                                vecs[i + (p2 + j) * n] = temp / nrm;
                            __syncthreads();
                        }
                    }
                }
            }
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDCX_MERGEUPDATE_KERNEL updates vectors after merges are done. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as columns would be in the matrix if its size is exact multiple of nn.
          Each group works with a column. Groups are size STEDCX_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than the actual number of columns n,
          it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_BDIM)
    stedcx_mergeUpdate_kernel(const rocblas_int k,
                              const rocblas_int n,
                              S* DD,
                              const rocblas_stride strideD,
                              S* CC,
                              const rocblas_int shiftC,
                              const rocblas_int ldc,
                              const rocblas_stride strideC,
                              S* tmpzA,
                              S* vecsA,
                              rocblas_int* splitsA)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split block id
    rocblas_int sid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int mid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid, vidb;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = psA + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    /* --------------------------------------------------- */

    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    extern __shared__ rocblas_int lsmem[];
    // used to store temp values during the different reductions
    S* inrms = reinterpret_cast<S*>(lsmem);
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    rocblas_int tn;
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        __syncthreads();

        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        levs = stedcx_num_levels(bs);
        blks = 1 << levs;

        // tn is max number of vectors in each sub-block
        tn = (bs - 1) / blks + 1;

        // 3. MERGE PHASE
        /* ----------------------------------------------------------------- */
        // Work with merges on level k. Each thread-group works with one vector.
        if(mid < tn * blks && k < levs)
        {
            rocblas_int iam, sz, bdm;
            S* ptz;
            S valf, valg;
            rocblas_int bd = 1 << k;
            bdm = bd << 1;

            // tid indexes the sub-blocks in the entire split block
            tid = mid / tn;
            p2 = ps[tid];
            // vidb indexes the vectors associated with each sub-block
            vidb = mid % tn;
            // iam indexes the sub-blocks in the context of the merge
            // (according to its level in the merge tree)
            iam = tid % bdm;

            // determine boundaries of what would be the new merged sub-block
            // 'in' will be its initial position
            rocblas_int in = ps[tid - iam];
            // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
            sz = ns[tid];
            for(int i = iam; i > 0; --i)
                sz += ns[tid - i];
            for(int i = bdm - 1 - iam; i > 0; --i)
                sz += ns[tid + i];

            // 3g. update D and C with computed values and vectors
            /* ----------------------------------------------------------------- */
            rocblas_int j = vidb;
            bool go = (j < ns[tid] && idd[p2 + j] == 1);
            if(go)
            {
                if(tidb == 0)
                    D[p2 + j] = evs[p2 + j];
                for(int i = in + tidb; i < in + sz; i += dim)
                    C[i + (p2 + j) * ldc] = vecs[i + (p2 + j) * n];
            }
            /* ----------------------------------------------------------------- */
        }
    }
}

/** STEDCX_SORT sorts computed eigenvalues and eigenvectors in increasing order **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) stedcx_sort(const rocblas_int n,
                                                         S* DD,
                                                         const rocblas_stride strideD,
                                                         U CC,
                                                         const rocblas_int shiftC,
                                                         const rocblas_int ldc,
                                                         const rocblas_stride strideC,
                                                         const rocblas_int batch_count,
                                                         rocblas_int* work,
                                                         rocblas_int* nev = nullptr)
{
    // -----------------------------------
    // use z-grid dimension as batch index
    // -----------------------------------
    rocblas_int bid_start = hipBlockIdx_z;
    rocblas_int bid_inc = hipGridDim_z;

    int tid = hipThreadIdx_x;

    rocblas_int* const map = work + bid_start * ((int64_t)n);

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // ---------------------------------------------
        // select batch instance to work with
        // (avoiding arithmetics with possible nullptrs)
        // ---------------------------------------------
        T* C = nullptr;
        if(CC)
            C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
        S* D = DD + (bid * strideD);
        rocblas_int nn;
        if(nev)
            nn = nev[bid];
        else
            nn = n;

        bool constexpr use_shell_sort = true;

        __syncthreads();

        if(use_shell_sort)
            shell_sort(nn, D, map);
        else
            selection_sort(nn, D, map);
        __syncthreads();

        permute_swap(n, C, ldc, map, nn);
        __syncthreads();
    }
}

/******************* Host functions ********************************************/
/*******************************************************************************/

//--------------------------------------------------------------------------------------//
/** This helper calculates required workspace size **/
template <bool BATCHED, typename T, typename S>
void rocsolver_stedcx_getMemorySize(const rocblas_evect evect,
                                    const rocblas_int n,
                                    const rocblas_int batch_count,
                                    size_t* size_work_stack,
                                    size_t* size_work_steqr,
                                    size_t* size_tempvect,
                                    size_t* size_tempgemm,
                                    size_t* size_tmpz,
                                    size_t* size_splits,
                                    size_t* size_workArr)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // if quick return no workspace needed
    if(n <= 1 || !batch_count)
    {
        *size_work_stack = 0;
        *size_work_steqr = 0;
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits = 0;
        *size_tmpz = 0;
        return;
    }

    size_t s1, s2;

    // requirements for solver of small independent blocks
    rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_steqr);
    s1 = sizeof(S) * (4 * n + 2) * batch_count;

    // extra requirements for original eigenvectors of small independent blocks
    *size_tempvect = (n * n) * batch_count * sizeof(S);
    *size_tempgemm = 2 * (n * n) * batch_count * sizeof(S);
    if(COMPLEX)
        s2 = n * n * batch_count * sizeof(S);
    else
        s2 = 0;
    if(BATCHED && !COMPLEX)
        *size_workArr = sizeof(S*) * batch_count;
    else
        *size_workArr = 0;
    *size_work_stack = std::max(s1, s2);

    // size for split blocks and sub-blocks positions
    *size_splits = sizeof(rocblas_int) * (5 * n + 2) * batch_count;

    // size for temporary diagonal and rank-1 modif vector
    *size_tmpz = sizeof(S) * (3 * n) * batch_count;
}

//--------------------------------------------------------------------------------------//
/** Helper to check argument correctnesss **/
template <typename T, typename S>
rocblas_status rocsolver_stedcx_argCheck(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_erange range,
                                         const rocblas_int n,
                                         const S vlow,
                                         const S vup,
                                         const rocblas_int ilow,
                                         const rocblas_int iup,
                                         S* D,
                                         S* E,
                                         rocblas_int* nev,
                                         S* W,
                                         T* C,
                                         const rocblas_int ldc,
                                         rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(range != rocblas_erange_all && range != rocblas_erange_value && range != rocblas_erange_index)
        return rocblas_status_invalid_value;
    if(evect != rocblas_evect_none && evect != rocblas_evect_tridiagonal
       && evect != rocblas_evect_original)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if(evect != rocblas_evect_none && ldc < n)
        return rocblas_status_invalid_size;
    if(range == rocblas_erange_value && vlow >= vup)
        return rocblas_status_invalid_size;
    if(range == rocblas_erange_index && (iup > n || (n > 0 && ilow > iup)))
        return rocblas_status_invalid_size;
    if(range == rocblas_erange_index && (ilow < 1 || iup < 0))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && (!D || !W || !C)) || (n > 1 && !E) || !info || !nev)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//--------------------------------------------------------------------------------------//
/** STEDCX templated function **/
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_stedcx_template(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_erange erange,
                                         const rocblas_int n,
                                         const S vl,
                                         const S vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         S* D,
                                         const rocblas_stride strideD,
                                         S* E,
                                         const rocblas_stride strideE,
                                         rocblas_int* nev,
                                         S* W,
                                         const rocblas_stride strideW,
                                         U C,
                                         const rocblas_int shiftC,
                                         const rocblas_int ldc,
                                         const rocblas_stride strideC,
                                         rocblas_int* info,
                                         const rocblas_int batch_count,
                                         S* work_stack,
                                         S* work_steqr,
                                         S* tempvect,
                                         S* tempgemm,
                                         S* tmpz,
                                         rocblas_int* splits,
                                         S** workArr)
{
    ROCSOLVER_ENTER("stedcx", "erange:", erange, "n:", n, "vl:", vl, "vu:", vu, "il:", il,
                    "iu:", iu, "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // NOTE: case evect = N is not implemented for now. This routine always compute vectors
    // as it is only for internal use by syevdx.

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
    if(n == 1)
    {
        if(evect != rocblas_evect_none)
            ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0,
                                    stream, C, strideC, n, 1);
        ROCSOLVER_LAUNCH_KERNEL(stedcx_case1_kernel, dim3(batch_count), dim3(1), 0, stream, erange,
                                vl, vu, D, strideD, nev, W, strideW);
    }
    if(n <= 1)
        return rocblas_status_success;

    // constants
    S eps = get_epsilon<S>();
    S ssfmin = get_safemin<S>();
    S ssfmax = S(1.0) / ssfmin;
    ssfmin = sqrt(ssfmin) / (eps * eps);
    ssfmax = sqrt(ssfmax) / S(3.0);
    rocblas_int blocksn = (n - 1) / BS2 + 1;

    // initialize identity matrix in C if required
    if(evect == rocblas_evect_tridiagonal)
        ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, dim3(blocksn, blocksn, batch_count), dim3(BS2, BS2),
                                0, stream, n, n, C, shiftC, ldc, strideC);

    // initialize identity matrix in tempvect
    rocblas_int ldt = n;
    rocblas_stride strideT = n * n;
    ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(blocksn, blocksn, batch_count), dim3(BS2, BS2), 0,
                            stream, n, n, tempvect, 0, ldt, strideT);

    // find max number of sub-blocks to consider during the divide phase
    rocblas_int maxlevs = stedcx_num_levels(n);
    rocblas_int maxblks = 1 << maxlevs;

    // find independent split blocks in matrix and prepare range for partial decomposition
    ROCSOLVER_LAUNCH_KERNEL(stedcx_split_kernel, dim3(1, batch_count), dim3(STEBZ_SPLIT_THDS), 0,
                            stream, erange, n, vl, vu, il, iu, D, strideD, E, strideE, W, strideW,
                            splits, work_stack, eps, ssfmin);

    // 1. divide phase
    //-----------------------------
    ROCSOLVER_LAUNCH_KERNEL((stedcx_divide_kernel<S>), dim3(batch_count), dim3(STEDCX_BDIM), 0,
                            stream, n, D, strideD, E, strideE, splits);

    // 2. solve phase
    //-----------------------------
    ROCSOLVER_LAUNCH_KERNEL((stedcx_solve_kernel<S>),
                            dim3(maxblks, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(1), 0, stream, n,
                            D, strideD, E, strideE, tempvect, 0, ldt, strideT, info, work_steqr,
                            splits, eps, ssfmin, ssfmax);

    // 3. merge phase
    //----------------
    size_t lmemsize1 = sizeof(S) * 2 * STEDCX_BDIM;
    size_t lmemsize3 = sizeof(S) * STEDCX_BDIM;
    rocblas_int numgrps3 = ((n - 1) / maxblks + 1) * maxblks;

    // launch merge for level k
    /** TODO: using max number of levels for now. Kernels return immediately when passing
        the actual number of levels in the split block. We should explore if synchronizing
        to copy back the actual number of levels makes any difference **/
    for(rocblas_int k = 0; k < maxlevs; ++k)
    {
        /** TODO: at the last level, kernels in steps b, c, and d could skip computations of
            eigen values and vectors that are out of the desired range. Whether this could be
            exploited somehow to improve performance must be explored in the future. For now,
            as all values and vectors are computed concurrently (by different threads), skiping
            the computation of some of them does not seem to make much difference. **/

        // a. prepare secular equations
        rocblas_int numgrps2 = 1 << (maxlevs - 1 - k);
        ROCSOLVER_LAUNCH_KERNEL((stedcx_mergePrepare_kernel<S>),
                                dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(STEDCX_BDIM), lmemsize1, stream, k, n, D, strideD, E, strideE,
                                tempvect, 0, ldt, strideT, tmpz, tempgemm, splits, eps);

        // b. solve to find merged eigen values
        ROCSOLVER_LAUNCH_KERNEL((stedcx_mergeValues_kernel<S>),
                                dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(STEDCX_BDIM), 0, stream, k, n, D, strideD, E, strideE, tmpz,
                                tempgemm, splits, eps, ssfmin, ssfmax);

        // c. find merged eigen vectors
        ROCSOLVER_LAUNCH_KERNEL((stedcx_mergeVectors_kernel<STEDCX_EXTERNAL_GEMM, S>),
                                dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(STEDCX_BDIM), lmemsize3, stream, k, n, D, strideD, E, strideE,
                                tempvect, 0, ldt, strideT, tmpz, tempgemm, splits);

        // d. update level
        ROCSOLVER_LAUNCH_KERNEL((stedcx_mergeUpdate_kernel<S>),
                                dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(STEDCX_BDIM), lmemsize3, stream, k, n, D, strideD, tempvect, 0,
                                ldt, strideT, tmpz, tempgemm, splits);
    }

    // 4. update and sort
    //----------------------
    // Synthesize the results from all the split blocks
    ROCSOLVER_LAUNCH_KERNEL(stedcx_synthesis_kernel, dim3(1, batch_count), dim3(STEDCX_BDIM), 0,
                            stream, erange, n, il, iu, D, strideD, nev, W, strideW, tempvect, ldt,
                            strideT, batch_count, splits, work_stack, eps);

    // eigenvectors C <- C*tempvect
    local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                    work_stack, 0, ldt, strideT, batch_count, workArr);

    // sort eigenvalues and eigenvectors
    ROCSOLVER_LAUNCH_KERNEL((stedcx_sort<T>), dim3(1, 1, batch_count), dim3(BS1), 0, stream, n, W,
                            strideW, C, shiftC, ldc, strideC, batch_count, splits, nev);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
