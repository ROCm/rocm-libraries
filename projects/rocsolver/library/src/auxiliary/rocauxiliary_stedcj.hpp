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

#include <algorithm>

#include "lapack/roclapack_syevj_heevj.hpp"
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

#define STEDCJ_BDIM 512 // Number of threads per thread-block used in main stedc kernels
#define MAXSWEEPS 20 // Max number of sweeps for Jacobi solver (when used)

// TODO: using macro STEDCJ_EXTERNAL_GEMM = false for now. We can enable the use of
// external gemm updates once the development is completed for stedc.
#define STEDCJ_EXTERNAL_GEMM false

/***************** Device auxiliary functions *****************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDCJ_NUM_LEVELS returns the ideal number of times/levels in which a matrix
   (or split block) will be divided during the divide phase of divide & conquer
   algorithm. i.e. number of sub-blocks = 2^levels **/
__host__ __device__ inline rocblas_int stedcj_num_levels(const rocblas_int n)
{
    rocblas_int levels = 0;
    // return the max number of levels such that the sub-blocks are at least of
    // size 1 (i.e. 2^levels <= n), and there are no more than 256 sub-blocks
    // (i.e. 2^levels <= 256)
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

//--------------------------------------------------------------------------------------//
/** DE2TRIDIAG generates a tridiagonal matrix from vectors of diagonal entries
   (D) and off-diagonal entries (E). **/
template <typename S>
__device__ inline void de2tridiag(const int numt,
                                  const rocblas_int id,
                                  const rocblas_int n,
                                  S* D,
                                  S* E,
                                  S* C,
                                  const rocblas_int ldc)
{
    for(rocblas_int k = id; k < n * n; k += numt)
    {
        rocblas_int i = k % n;
        rocblas_int j = k / n;
        S val;
        bool offd = (i == j + 1);
        if(offd || i == j - 1)
            val = offd ? E[j] : E[i];
        else
            val = (i == j) ? D[i] : 0;
        C[i + j * ldc] = val;
    }
}

/*************** Main kernels *********************************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDCJ_SPLIT finds independent blocks (split-blocks) in the tridiagonal
   matrix given by D and E. The independent blocks can then be solved in
    parallel by the DC algorithm.
        - Call this kernel with batch_count single-threaded groups in x **/
template <typename S>
ROCSOLVER_KERNEL void stedcj_split(const rocblas_int n,
                                   S* DD,
                                   const rocblas_stride strideD,
                                   S* EE,
                                   const rocblas_stride strideE,
                                   rocblas_int* splitsA,
                                   const S eps)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance
    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    rocblas_int* splits = splitsA + bid * (5 * n + 2);

    rocblas_int k = 0; // position where the last block starts
    S tol; // tolerance. If an element of E is <= tol we have an independent
        // block
    rocblas_int bs; // size of an independent block
    rocblas_int nb = 1; // number of blocks
    splits[0] = 0; // positions where each block begings

    // main loop
    while(k < n)
    {
        bs = 1;
        for(rocblas_int j = k; j < n - 1; ++j)
        {
            tol = eps * sqrt(abs(D[j])) * sqrt(abs(D[j + 1]));
            if(abs(E[j]) < tol)
            {
                // Split next independent block
                // save its location in matrix
                splits[nb] = j + 1;
                nb++;
                break;
            }
            bs++;
        }
        k += bs;
    }
    splits[nb] = n;
    splits[n + 1] = nb; // also save the number of split blocks
}

//--------------------------------------------------------------------------------------//
/** STEDCJ_DIVIDE_KERNEL implements the divide phase of the DC algorithm. It
   divides each split-block into a number of sub-blocks.
        - Call this kernel with batch_count groups in x. Groups are of size
   STEDCJ_BDIM.
        - If there are actually more split-blocks than STEDCJ_BDIM, some threads
   will work with more than one split-block sequentially. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCJ_BDIM)
    stedcj_divide_kernel(const rocblas_int n,
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

    // work with STEDCJ_BDIM split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDCJ_BDIM)
    {
        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks in split-block
        levs = stedcj_num_levels(bs);
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
/** STEDCJ_SOLVE_KERNEL implements the solver phase of the DC algorithm to
        compute the eigenvalues/eigenvectors of the different sub-blocks of each
   split-block. A matrix in the batch could have many split-blocks, and each
   split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS
   groups in y and nn groups in x. Groups are size STEDCJ_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
   be analysed in parallel). If there are actually more split-blocks, some
   groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
   the size n. If a group has an id larger than the actual number of sub-blocks
   in a split-block, it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCJ_BDIM)
    stedcj_solve_kernel(const rocblas_int n,
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
    S* W = WA + bid * (2 + n * n);
    /* --------------------------------------------------- */

    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    extern __shared__ rocblas_int lsmem[];
    rocblas_int* sj2 = lsmem;
    S* sj1 = reinterpret_cast<S*>(sj2 + n + n % 2);
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
        levs = stedcj_num_levels(bs);
        blks = 1 << levs;

        // 2. SOLVE PHASE
        /* ----------------------------------------------------------------- */
        // Solve the blks sub-blocks in parallel.

        if(tid < blks)
        {
            sbs = ns[tid];
            p2 = ps[tid];

            // transform D and E into full upper tridiag matrix and copy to C
            de2tridiag(STEDCJ_BDIM, tidb, sbs, D + p2, E + p2, C + p2 + p2 * ldc, ldc);

            // set work space
            S* W_Acpy = W;
            S* W_residual = W_Acpy + n * n;
            rocblas_int* W_n_sweeps = reinterpret_cast<rocblas_int*>(W_residual + 1);

            // set shared mem
            rocblas_int even_n = sbs + sbs % 2;
            rocblas_int half_n = even_n / 2;
            S* cosines_res = sj1;
            S* sines_diag = cosines_res + half_n;
            rocblas_int* top = sj2;
            rocblas_int* bottom = top + half_n;

            // re-arrange threads in 2D array
            rocblas_int ddx, ddy;
            syevj_get_dims(sbs, STEDCJ_BDIM, &ddx, &ddy);
            rocblas_int tix = tidb % ddx;
            rocblas_int tiy = tidb / ddx;
            __syncthreads();

            // solve
            run_syevj<S, S>(ddx, ddy, tix, tiy, rocblas_esort_none, rocblas_evect_original,
                            rocblas_fill_upper, sbs, C + p2 + p2 * ldc, ldc, 0, eps, W_residual,
                            MAXSWEEPS, W_n_sweeps, D + p2, info, W_Acpy + p2 + p2 * n, cosines_res,
                            sines_diag, top, bottom);
            __syncthreads();
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDCJ_MERGEPREPARE_KERNEL performs deflation and prepares the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as half of the unmerged sub-blocks in current level. Each group works
          with a merge. Groups are size STEDCJ_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than half the actual number of unmerged sub-blocks
          in the level, it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCJ_BDIM)
    stedcj_mergePrepare_kernel(const rocblas_int k,
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
        levs = stedcj_num_levels(bs);
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
/** STEDCJ_MERGEVALUES_KERNEL solves the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as half of the unmerged sub-blocks in current level. Each group works
          with a merge. Groups are size STEDCJ_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than half the actual number of unmerged sub-blocks
          in the level, it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCJ_BDIM)
    stedcj_mergeValues_kernel(const rocblas_int k,
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
        levs = stedcj_num_levels(bs);
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
/** STEDCJ_MERGEVECTORS_KERNEL prepares vectors from the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as columns would be in the matrix if its size is exact multiple of nn.
          Each group works with a column. Groups are size STEDCJ_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than the actual number of columns n,
          it will do nothing. **/
template <bool USEGEMM, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCJ_BDIM)
    stedcj_mergeVectors_kernel(const rocblas_int k,
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
        levs = stedcj_num_levels(bs);
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
/** STEDCJ_MERGEUPDATE_KERNEL updates vectors after merges are done. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as columns would be in the matrix if its size is exact multiple of nn.
          Each group works with a column. Groups are size STEDCJ_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than the actual number of columns n,
          it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCJ_BDIM)
    stedcj_mergeUpdate_kernel(const rocblas_int k,
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
        levs = stedcj_num_levels(bs);
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

/** STEDCJ_SORT sorts computed eigenvalues and eigenvectors in increasing order **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) stedcj_sort(const rocblas_int n,
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

/******************* Host functions *********************************************/
/*******************************************************************************/

//--------------------------------------------------------------------------------------//
/** This helper check argument correctness for stedc API **/
template <typename T, typename S>
rocblas_status rocsolver_stedcj_argCheck(rocblas_handle handle,
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
/** This helper calculates required workspace size **/
template <bool BATCHED, typename T, typename S>
void rocsolver_stedcj_getMemorySize(const rocblas_evect evect,
                                    const rocblas_int n,
                                    const rocblas_int batch_count,
                                    size_t* size_work_stack,
                                    size_t* size_tempvect,
                                    size_t* size_tempgemm,
                                    size_t* size_tmpz,
                                    size_t* size_splits_map,
                                    size_t* size_workArr)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // if quick return no workspace needed
    if(n <= 1 || !batch_count)
    {
        *size_work_stack = 0;
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits_map = 0;
        *size_tmpz = 0;
        return;
    }

    size_t s1, s2;

    // requirements for solver of small independent blocks
    s1 = sizeof(S) * (n * n + 2) * batch_count;

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
    *size_splits_map = sizeof(rocblas_int) * (5 * n + 2) * batch_count;

    // size for temporary diagonal and rank-1 modif vector
    *size_tmpz = sizeof(S) * (2 * n) * batch_count;
}

//--------------------------------------------------------------------------------------//
/** STEDC templated function **/
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_stedcj_template(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_int n,
                                         S* D,
                                         const rocblas_stride strideD,
                                         S* E,
                                         const rocblas_stride strideE,
                                         U C,
                                         const rocblas_int shiftC,
                                         const rocblas_int ldc,
                                         const rocblas_stride strideC,
                                         rocblas_int* info,
                                         const rocblas_int batch_count,
                                         void* work_stack,
                                         S* tempvect,
                                         S* tempgemm,
                                         S* tmpz,
                                         rocblas_int* splits_map,
                                         S** workArr)
{
    ROCSOLVER_ENTER("stedcj", "evect:", evect, "n:", n, "shiftC:", shiftC, "ldc:", ldc,
                    "bc:", batch_count);

    // NOTE: case evect = N is not implemented for now. This routine always compute vectors
    // as it is only for internal use by syevdj.

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
    rocblas_int maxlevs = stedcj_num_levels(n);
    rocblas_int maxblks = 1 << maxlevs;

    // find independent split blocks in matrix
    ROCSOLVER_LAUNCH_KERNEL(stedcj_split, dim3(batch_count), dim3(1), 0, stream, n, D, strideD, E,
                            strideE, splits_map, eps);

    // 1. divide phase
    //-----------------------------
    ROCSOLVER_LAUNCH_KERNEL((stedcj_divide_kernel<S>), dim3(batch_count), dim3(STEDCJ_BDIM), 0,
                            stream, n, D, strideD, E, strideE, splits_map);

    // 2. solve phase
    //-----------------------------
    size_t lmemsize = (n + n % 2) * (sizeof(rocblas_int) + sizeof(S));

    ROCSOLVER_LAUNCH_KERNEL((stedcj_solve_kernel<S>),
                            dim3(maxblks, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDCJ_BDIM),
                            lmemsize, stream, n, D, strideD, E, strideE, tempvect, 0, ldt, strideT,
                            info, static_cast<S*>(work_stack), splits_map, eps, ssfmin, ssfmax);

    // 3. merge phase
    //----------------
    size_t lmemsize1 = sizeof(S) * 2 * STEDCJ_BDIM;
    size_t lmemsize3 = sizeof(S) * STEDCJ_BDIM;
    rocblas_int numgrps3 = ((n - 1) / maxblks + 1) * maxblks;

    // launch merge for level k
    /** TODO: using max number of levels for now. Kernels return immediately when passing
        the actual number of levels in the split block. We should explore if synchronizing
        to copy back the actual number of levels makes any difference **/
    for(rocblas_int k = 0; k < maxlevs; ++k)
    {
        // a. prepare secular equations
        rocblas_int numgrps2 = 1 << (maxlevs - 1 - k);
        ROCSOLVER_LAUNCH_KERNEL((stedcj_mergePrepare_kernel<S>),
                                dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(STEDCJ_BDIM), lmemsize1, stream, k, n, D, strideD, E, strideE,
                                tempvect, 0, ldt, strideT, tmpz, tempgemm, splits_map, eps);

        // b. solve to find merged eigen values
        ROCSOLVER_LAUNCH_KERNEL((stedcj_mergeValues_kernel<S>),
                                dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(STEDCJ_BDIM), 0, stream, k, n, D, strideD, E, strideE, tmpz,
                                tempgemm, splits_map, eps, ssfmin, ssfmax);

        // c. find merged eigen vectors
        ROCSOLVER_LAUNCH_KERNEL((stedcj_mergeVectors_kernel<STEDCJ_EXTERNAL_GEMM, S>),
                                dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(STEDCJ_BDIM), lmemsize3, stream, k, n, D, strideD, E, strideE,
                                tempvect, 0, ldt, strideT, tmpz, tempgemm, splits_map);

        // c. update level
        ROCSOLVER_LAUNCH_KERNEL((stedcj_mergeUpdate_kernel<S>),
                                dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count),
                                dim3(STEDCJ_BDIM), lmemsize3, stream, k, n, D, strideD, tempvect, 0,
                                ldt, strideT, tmpz, tempgemm, splits_map);
    }

    // 4. update and sort
    //----------------------
    // eigenvectors C <- C*tempvect
    local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                    static_cast<S*>(work_stack), 0, ldt, strideT, batch_count,
                                    workArr);

    ROCSOLVER_LAUNCH_KERNEL((stedcj_sort<T>), dim3(1, 1, batch_count), dim3(BS1), 0, stream, n, D,
                            strideD, C, shiftC, ldc, strideC, batch_count, splits_map);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
