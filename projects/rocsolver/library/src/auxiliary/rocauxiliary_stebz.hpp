/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.10.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#define STEBZ_SPLIT_THDS 256
#define IBISEC_BLKS 64
#define IBISEC_THDS 128

ROCSOLVER_BEGIN_NAMESPACE

/************** Kernels and device functions *********************************/
/*****************************************************************************/

/** This device function orders the elements of vector X
    from smallest to largest, and its corresponding indices in Y **/
template <typename T>
__device__ void increasing_order(const rocblas_int nn, T* X, rocblas_int* Y)
{
    rocblas_int s1, s2, bv;
    T v, vv;
    for(rocblas_int i = 1; i < nn; i++)
    {
        s1 = i - 1;
        s2 = s1;
        v = X[s1];
        for(rocblas_int j = i; j < nn; j++)
        {
            vv = X[j];
            if(vv < v)
            {
                s2 = j;
                v = vv;
            }
        }
        if(s2 != s1)
        {
            X[s2] = X[s1];
            X[s1] = v;
            if(Y)
            {
                bv = Y[s2];
                Y[s2] = Y[s1];
                Y[s1] = bv;
            }
        }
    }
}

/** This device function computes Gershgorin circles for all
    diagonal elements of a tridiagonal matrix, and returns
    the outer bounds**/
template <typename T>
__device__ void gershgorin_bounds(const rocblas_int n, T* D, T* E, T* vlow, T* vup)
{
    /** (TODO: Gershgorin circles for each diagonal element can be computed
    in parallel, and then use iamax/iamin to find the max if this gives
    better performance) **/

    T tmp = D[0];
    T t1 = std::abs(E[0]);
    T gl = tmp - t1;
    T gu = tmp + t1;
    T t2;
    for(rocblas_int i = 1; i < n - 1; ++i)
    {
        tmp = D[i];
        t2 = std::abs(E[i]);
        gl = std::min(gl, tmp - t1 - t2);
        gu = std::max(gu, tmp + t1 + t2);
        t1 = t2;
    }
    tmp = D[n - 1];
    gl = std::min(gl, tmp - t1);
    gu = std::max(gu, tmp + t1);

    *vlow = gl;
    *vup = gu;
}

/** This device function implements the Sturm sequence to compute the
    number of eigenvalues in the half-open interval (-inf,c] **/
template <typename T>
__device__ rocblas_int sturm_count(const rocblas_int n, T* D, T* E, T pmin, T c)
{
    rocblas_int ev;
    T t;

    ev = 0;
    t = D[0] - c;
    if(t <= pmin)
    {
        ev++;
        t = std::min(t, -pmin);
    }

    // main loop
    for(rocblas_int i = 1; i < n; ++i)
    {
        t = D[i] - c - E[i - 1] / t;
        if(t <= pmin)
        {
            ev++;
            t = std::min(t, -pmin);
        }
    }

    return ev;
}

/** This kernel deals with the case n = 1
    (one split block and a single eigenvalue which is the element in D) **/
template <typename T, typename U>
ROCSOLVER_KERNEL void stebz_case1_kernel(const rocblas_erange range,
                                         const T vlow,
                                         const T vup,
                                         U DA,
                                         const rocblas_int shiftD,
                                         const rocblas_stride strideD,
                                         rocblas_int* nev,
                                         rocblas_int* nsplit,
                                         T* WA,
                                         const rocblas_stride strideW,
                                         rocblas_int* IBA,
                                         const rocblas_stride strideIB,
                                         rocblas_int* ISA,
                                         const rocblas_stride strideIS,
                                         const rocblas_int batch_count)
{
    int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(bid < batch_count)
    {
        // select batch instance
        T* D = load_ptr_batch<T>(DA, bid, shiftD, strideD);
        T* W = WA + bid * strideW;
        rocblas_int* IB = IBA + bid * strideIB;
        rocblas_int* IS = ISA + bid * strideIS;

        // one split block
        nsplit[bid] = 1;
        IS[0] = 1;

        // check if diagonal element is in range and return
        T d = D[0];
        if(range == rocblas_erange_value && (d <= vlow || d > vup))
        {
            nev[bid] = 0;
        }
        else
        {
            nev[bid] = 1;
            W[0] = d;
            IB[0] = 1;
        }
    }
}

/** STEBZ_SPLITTING splits the matrix into independent blocks and prepares ranges
    for the computations in the iterative bisection.
    Call the kernel with batch_count groups in Y, of DIM threads in X direction.
    Each thread will work with as many elements of the diagonal as needed.
    Call the device function with compact = true to get the output IS = [0, IS, nsplit]
    as needed by stedcx. **/
template <int DIM, typename T>
__device__ void run_stebz_splitting(const int tid,
                                    const rocblas_erange range,
                                    const rocblas_int n,
                                    const T vlow,
                                    const T vup,
                                    const rocblas_int ilow,
                                    const rocblas_int iup,
                                    T* D,
                                    T* E,
                                    rocblas_int* nsplit,
                                    T* W,
                                    rocblas_int* IS,
                                    rocblas_int* tmpIS,
                                    T* pivmin,
                                    T* Esqr,
                                    T* bounds,
                                    T* inter,
                                    rocblas_int* ninter,
                                    T* sval,
                                    rocblas_int* sidx,
                                    T eps,
                                    T sfmin,
                                    bool compact = false)
{
    // the number of elements worked by this thread is nn
    rocblas_int nn = (n - 1) / DIM;
    if(tid < n - 1 - nn * DIM)
        nn++;
    sidx[tid] = nn;
    __syncthreads();

    // thus, this thread offset is:
    rocblas_int offset = 0;
    for(int i = 0; i < tid; ++i)
        offset += sidx[i];

    // local helper variables
    T tmp, tmp2, vl, vu;
    rocblas_int j, nu, nl;
    rocblas_int tmpns = 0; //temporary number of blocks found

    // this thread find its split-off blocks if necessary
    // tmpIS stores the block indices found by this thread
    tmpIS += offset;
    for(rocblas_int i = 0; i < nn; ++i)
    {
        j = i + offset;

        tmp = E[j];
        tmp2 = tmp * tmp;

        if(std::abs(D[j] * D[j + 1]) * eps * eps + sfmin > tmp2)
        {
            // found split
            tmpIS[tmpns] = j;
            tmpns++;
            Esqr[j] = 0;
            W[j] = 0;
        }
        else
        {
            // no split; E[j] can be pivot
            Esqr[j] = tmp2;
            W[j] = tmp;
        }
    }
    sidx[tid] = tmpns;
    __syncthreads();

    // find split-off blocks in entire matrix
    offset = compact ? 1 : 0;
    for(int i = 0; i < tid; ++i)
        offset += sidx[i];
    for(int i = 0; i < tmpns; ++i)
        IS[i + offset] = tmpIS[i] + 1;

    // total number of split blocks
    if(tid == DIM - 1)
    {
        offset += tmpns;
        IS[offset] = n;
        if(compact)
        {
            IS[0] = 0;
            IS[n + 1] = offset;
        }
        else
            *nsplit = offset + 1;
    }
    __syncthreads();

    // find max squared off-diagonal element
    iamax<DIM>(tid, n - 1, Esqr, 1, sval, sidx);
    __syncthreads();

    // `pmin` is a numerically stable lower bound for pivots
    T pmin = std::max(sval[0] * sfmin, sfmin);
    vl = vlow;
    vu = vup;

    // if range = index, the following code finds a range (vl, vu]
    // containing the desired eigenvalue indices (using the split off-diagonal)
    if(range == rocblas_erange_index)
    {
        // find gershgorin of first diagonal element
        if(tid == 0)
        {
            tmp = D[0];
            tmp2 = std::abs(W[0]);
            vl = tmp - tmp2;
            vu = tmp + tmp2;
            nl = sturm_count(n, D, Esqr, pmin, vl);
            nu = sturm_count(n, D, Esqr, pmin, vu);
            inter[0] = vl;
            ninter[0] = nl;
            inter[1] = vu;
            ninter[1] = nu;
        }
        // work all other elements in parallel
        for(rocblas_int i = tid + 1; i < n - 1; i += DIM)
        {
            tmp = D[i];
            tmp2 = std::abs(W[i]) + std::abs(W[i - 1]);
            vl = tmp - tmp2;
            vu = tmp + tmp2;
            nl = sturm_count(n, D, Esqr, pmin, vl);
            nu = sturm_count(n, D, Esqr, pmin, vu);
            inter[2 * i] = vl;
            ninter[2 * i] = nl;
            inter[2 * i + 1] = vu;
            ninter[2 * i + 1] = nu;
        }
        // find gershgorin of last diagonal element
        if(tid == DIM - 1)
        {
            tmp = D[n - 1];
            tmp2 = std::abs(W[n - 2]);
            vl = tmp - tmp2;
            vu = tmp + tmp2;
            nl = sturm_count(n, D, Esqr, pmin, vl);
            nu = sturm_count(n, D, Esqr, pmin, vu);
            inter[2 * n - 2] = vl;
            ninter[2 * n - 2] = nl;
            inter[2 * n - 1] = vu;
            ninter[2 * n - 1] = nu;
        }
        __syncthreads();

        if(tid == 0)
        {
            // re-order intervals
            increasing_order(2 * n, inter, ninter);

            // adjust outer bounds
            vl = inter[0];
            vu = inter[2 * n - 1];
            tmp = std::max(std::abs(vl), std::abs(vu));
            inter[0] = vl - tmp * eps * n - pmin;
            inter[2 * n - 1] = vu + tmp * eps * n + pmin;

            // find lower bound for the set of indices
            j = 0;
            for(rocblas_int i = 1; i < 2 * n; ++i)
            {
                if(ninter[i] < ilow)
                    j++;
                else
                    break;
            }
            vl = inter[j];

            // find upper bound for the set of indices
            j = 0;
            for(rocblas_int i = 1; i < 2 * n; ++i)
            {
                j++;
                if(ninter[i] >= iup)
                    break;
            }
            vu = inter[j];
        }
    }

    if(tid == 0)
    {
        // set pivmin (minimum value that can be pivot in sturm sequence)
        *pivmin = pmin;

        // set upper and lower bounds vl and vu of the absolute interval (vl, vu] where
        // the eigenvalues will be searched. vl and vu are set to zero when looking for
        // all the eigenvalues in the matrix.
        if(range == rocblas_erange_all)
        {
            bounds[0] = 0;
            bounds[1] = 0;
        }
        else
        {
            bounds[0] = vl;
            bounds[1] = vu;
        }
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(STEBZ_SPLIT_THDS)
    stebz_splitting_kernel(const rocblas_erange range,
                           const rocblas_int n,
                           const T vlow,
                           const T vup,
                           const rocblas_int ilow,
                           const rocblas_int iup,
                           U DA,
                           const rocblas_int shiftD,
                           const rocblas_int strideD,
                           U EA,
                           const rocblas_int shiftE,
                           const rocblas_int strideE,
                           rocblas_int* nsplitA,
                           T* WA,
                           const rocblas_stride strideW,
                           rocblas_int* ISA,
                           const rocblas_stride strideIS,
                           rocblas_int* tmpISA,
                           T* pivminA,
                           T* EsqrA,
                           T* boundsA,
                           T* interA,
                           rocblas_int* ninterA,
                           T eps,
                           T sfmin)
{
    // batch instance
    const int tid = hipThreadIdx_x;
    const int bid = hipBlockIdx_y;
    T* D = load_ptr_batch<T>(DA, bid, shiftD, strideD);
    T* E = load_ptr_batch<T>(EA, bid, shiftE, strideE);
    rocblas_int* nsplit = nsplitA + bid;
    T* pivmin = pivminA + bid;
    T* Esqr = EsqrA + bid * (n - 1);
    T* bounds = boundsA + 2 * bid;
    rocblas_int* IS = ISA + (bid * strideIS);
    // tmpIS stores the block indices found by each thread
    rocblas_int* tmpIS = tmpISA + (bid * n);
    // using W as temp array to store the spit off-diagonal
    // (to use in case range = index)
    T* W = WA + bid * strideW;
    // using inter and ninter as auxiliary arrays to find
    // a range for the case range = index
    T* inter = interA + bid * (2 * n);
    rocblas_int* ninter = ninterA + bid * (2 * n);

    // shared memory setup for iamax.
    // (sidx also temporarily stores the number of blocks found by each thread)
    __shared__ T sval[STEBZ_SPLIT_THDS];
    __shared__ rocblas_int sidx[STEBZ_SPLIT_THDS];

    run_stebz_splitting<STEBZ_SPLIT_THDS>(tid, range, n, vlow, vup, ilow, iup, D, E, nsplit, W, IS,
                                          tmpIS, pivmin, Esqr, bounds, inter, ninter, sval, sidx,
                                          eps, sfmin);
}

/** STEBZ_BISECTION implements the iterative bisection.
    Call the kernel with DIM_BLKS blocks in X and batch_count blocks in Y.
    Each thread-block is working with as many split-off blocks as needed to cover the entire
    matrix in the batch. Each blocks has DIM_THDS threads in X. Each thread works with as many
    non-converged intervals as needed on each iteration. **/
template <int DIM_BLKS, int DIM_THDS, typename T>
__device__ void run_stebz_bisection(const int sbid,
                                    const int tid,
                                    const rocblas_erange range,
                                    const rocblas_int n,
                                    const T abstol,
                                    T* D,
                                    T* E,
                                    const rocblas_int nofb,
                                    T* W,
                                    rocblas_int* IB,
                                    rocblas_int* IS,
                                    rocblas_int* info,
                                    rocblas_int* tmpnev,
                                    const T pmin,
                                    T* Esqr,
                                    T* bounds,
                                    T* inter,
                                    rocblas_int* ninter,
                                    int* sh_newi,
                                    T* sh_inter,
                                    int* sh_ninter,
                                    T eps,
                                    T sfmin)
{
    // sh_nofi accumulates the new number of intervals, for next iteration, in the active block
    __shared__ int sh_nofi;
    // sh_computed = true indicates that the eigenvalues in a block were computed
    __shared__ bool sh_computed;
    // sh_converged = true indicates that the eigenvalues in a block have converged
    __shared__ bool sh_converged;

    // local variables (lc_name):
    // lc_bin, lc_bout are the indices where a block starts and ends; lc_bdim is the dimension
    rocblas_int lc_bin, lc_bout, lc_bdim;
    // tolerance to decide when an interval converged
    T lc_tol;
    // lc_ite counts the number of iterations in the bisection
    // and lc_maxite is the maximum number of iterations allowed
    int lc_ite, lc_maxite;
    // lc_nofi is the number of intervals in active block at current iteration
    int lc_nofi;

    // other local temporary (helper) variables
    T gl, gu, tmp, bnorm;
    rocblas_int offset, offin, offout, nl, nu, ntmp;
    int iid;

    /*********** loop over independent split blocks ***********/
    for(int b = sbid; b < nofb; b += DIM_BLKS)
    {
        // find dimension and indices of current split block
        lc_bin = (b == 0) ? 0 : IS[b - 1];
        lc_bout = IS[b] - 1;
        lc_bdim = lc_bout - lc_bin + 1;

        // >>>>> if current split block has dimension 1, quick return <<<<<<<
        if(lc_bdim == 1)
        {
            if(tid == 0)
            {
                tmp = D[lc_bin];
                if((range == rocblas_erange_all)
                   || (bounds[0] < tmp - pmin && bounds[1] >= tmp - pmin))
                {
                    W[lc_bin] = tmp;
                    tmpnev[b] = 1; // 1 eigenvalue in this split block
                    IB[lc_bin] = b + 1;
                }
                else
                {
                    tmpnev[b] = 0; // no eigenvalues in this split block
                }
            }
        } // ((split block had dimension 1 --did quick return--))

        // <<<<<<<< otherwise do iterative bisection >>>>>>>>>>>>>>>>>
        else
        {
            offset = 2 * lc_bin; //position of the split block in the matrix

            // first find the max Gershgorin circle for this split block
            gershgorin_bounds(lc_bdim, (D + lc_bin), (E + lc_bin), &gl, &gu);

            // set tolerance to consider intervals in this split block as converged
            bnorm = std::max(std::abs(gl), std::abs(gu));
            lc_tol = abstol < 0 ? eps * bnorm : abstol;
            lc_tol = std::max(lc_tol, pmin);

            // initial interval is:
            gl = gl - bnorm * eps * lc_bdim - pmin;
            gu = gu + bnorm * eps * lc_bdim + pmin;
            if(range != rocblas_erange_all)
            {
                gl = std::max(gl, bounds[0]);
                gu = std::min(gu, bounds[1]);
            }

            // decide if there are eigenvalues to search, and set the initial interval
            // and max number of iterations for the bisection process
            if(gl < gu)
            {
                lc_maxite = int((log(gu - gl + pmin) - log(pmin)) / log(2)) + 2;
                lc_ite = 0;
                lc_nofi = 1;

                if(tid == 0)
                {
                    sh_converged = false;
                    nl = sturm_count(lc_bdim, (D + lc_bin), (Esqr + lc_bin), pmin, gl);
                    nu = sturm_count(lc_bdim, (D + lc_bin), (Esqr + lc_bin), pmin, gu);
                    ntmp = nu - nl;

                    if(ntmp > 0)
                    {
                        sh_computed = true;
                        inter[2 * n + offset] = gl;
                        inter[2 * n + offset + 1] = gu;

                        // number of eigenvalues in this split block
                        tmpnev[b] = ntmp;
                        ninter[2 * n + offset] = nl;
                        ninter[2 * n + offset + 1] = nu;
                    }
                    else
                    {
                        sh_computed = false;

                        // no eigenvalues in this split block
                        tmpnev[b] = 0;
                    }
                }
            }
            else
            {
                if(tid == 0)
                {
                    sh_converged = false;
                    sh_computed = false;

                    // no eigenvalues in this split block
                    tmpnev[b] = 0;
                }
            }
            __syncthreads();

            /************************************************/
            /*********** Main iterative loop: ***************/
            /************************************************/
            while(sh_computed && !sh_converged && lc_ite < lc_maxite)
            {
                // start next iteration.
                // (the first 2n elements of inter/ninter are used as inputs in even iterations,
                // the last 2n elements of these arrays are used as inputs in odd iterations)
                lc_ite++;
                offin = (lc_ite % 2) ? 2 * n + offset : offset;
                offout = (lc_ite % 2) ? offset : 2 * n + offset;
                if(tid == 0)
                    sh_nofi = 0; // to start accumulation of new number of intervals

                // work with all intervals in parallel
                for(int i = 0; i < lc_nofi; i += DIM_THDS)
                {
                    iid = tid + i; //position of the interval in the split block

                    if(iid < lc_nofi)
                    {
                        //get interval information
                        gl = inter[offin + 2 * iid];
                        nl = ninter[offin + 2 * iid];
                        gu = inter[offin + 2 * iid + 1];
                        nu = ninter[offin + 2 * iid + 1];

                        // bisect interval
                        tmp = (gl + gu) / 2;

                        // number of eigenvalues less than the middle point
                        ntmp = sturm_count(lc_bdim, (D + lc_bin), (Esqr + lc_bin), pmin, tmp);
                        ntmp = std::min(nu, std::max(ntmp, nl));

                        // re-compute interval; share new interval if necessary
                        if(ntmp == nl)
                        {
                            // no eigenvalues in lower half; adjust interval
                            sh_newi[tid] = 0;
                            sh_inter[4 * tid] = tmp;
                            sh_ninter[4 * tid] = ntmp;
                            sh_inter[4 * tid + 1] = gu;
                            sh_ninter[4 * tid + 1] = nu;
                        }
                        else if(ntmp == nu)
                        {
                            // no eigenvalues in upper half; adjust interval
                            sh_newi[tid] = 0;
                            sh_inter[4 * tid] = gl;
                            sh_ninter[4 * tid] = nl;
                            sh_inter[4 * tid + 1] = tmp;
                            sh_ninter[4 * tid + 1] = ntmp;
                        }
                        else
                        {
                            // eigenvalues in both halves; split interval
                            sh_newi[tid] = 1;
                            sh_inter[4 * tid] = gl;
                            sh_ninter[4 * tid] = nl;
                            sh_inter[4 * tid + 1] = tmp;
                            sh_ninter[4 * tid + 1] = ntmp;
                            sh_inter[4 * tid + 2] = tmp;
                            sh_ninter[4 * tid + 2] = ntmp;
                            sh_inter[4 * tid + 3] = gu;
                            sh_ninter[4 * tid + 3] = nu;
                        }
                    }
                    __syncthreads();

                    if(iid < lc_nofi)
                    {
                        // update main array with new intervals
                        // in preparation for next iteration
                        nu = 0;
                        for(int j = 0; j < tid; ++j)
                            nu += sh_newi[j];
                        ntmp = 2 * (tid + sh_nofi + nu);

                        inter[offout + ntmp] = sh_inter[4 * tid];
                        ninter[offout + ntmp] = sh_ninter[4 * tid];
                        inter[offout + ntmp + 1] = sh_inter[4 * tid + 1];
                        ninter[offout + ntmp + 1] = sh_ninter[4 * tid + 1];
                        if(sh_newi[tid])
                        {
                            inter[offout + ntmp + 2] = sh_inter[4 * tid + 2];
                            ninter[offout + ntmp + 2] = sh_ninter[4 * tid + 2];
                            inter[offout + ntmp + 3] = sh_inter[4 * tid + 3];
                            ninter[offout + ntmp + 3] = sh_ninter[4 * tid + 3];
                        }
                    }
                    __syncthreads();

                    // update new number of intervals
                    nl = std::min(lc_nofi - i, DIM_THDS); //number of active threads
                    if(tid == nl - 1)
                        sh_nofi += (nl + nu + sh_newi[tid]);
                    __syncthreads();
                } // ((((loop bisected all intervals in split block))))

                // set new number of intervals for next iteration
                lc_nofi = sh_nofi;

                // check intervals' width and stop if converged
                if(tid == 0)
                {
                    gl = sh_inter[0];
                    gu = sh_inter[1];
                    tmp = gu - gl;
                    bnorm = std::max(std::abs(gl), std::abs(gu));

                    if(tmp < std::max(lc_tol, 2 * eps * bnorm))
                        sh_converged = true;
                }

                __syncthreads();
            } // (((main loop implemented iterative bisection of split block)))

            // update W and IB and info
            if(tid == 0 && sh_computed)
            {
                int c = 0;
                for(int i = 0; i < lc_nofi; ++i)
                {
                    gl = inter[offout + 2 * i];
                    gu = inter[offout + 2 * i + 1];
                    nl = ninter[offout + 2 * i];
                    nu = ninter[offout + 2 * i + 1];

                    // the value of the eigenvalue is tmp, and it is
                    // repeated ntmp times
                    tmp = (gl + gu) / 2;
                    ntmp = nu - nl;

                    for(int j = 0; j < ntmp; ++j)
                    {
                        W[lc_bin + c] = tmp;
                        IB[lc_bin + c] = sh_converged ? b + 1 : -(b + 1);
                        c++;
                    }
                }

                if(!sh_converged)
                    *info = 1;
            }
        } // ((split block had dimension > 1 --did iterative bisection--))

        __syncthreads();
    } // (loop worked all independent split blocks)
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(IBISEC_THDS)
    stebz_bisection_kernel(const rocblas_erange range,
                           const rocblas_int n,
                           const T abstol,
                           U DA,
                           const rocblas_int shiftD,
                           const rocblas_int strideD,
                           U EA,
                           const rocblas_int shiftE,
                           const rocblas_int strideE,
                           rocblas_int* nsplit,
                           T* WA,
                           const rocblas_stride strideW,
                           rocblas_int* IBA,
                           const rocblas_stride strideIB,
                           rocblas_int* ISA,
                           const rocblas_stride strideIS,
                           rocblas_int* infoA,
                           rocblas_int* tmpnevA,
                           T* pivmin,
                           T* EsqrA,
                           T* boundsA,
                           T* interA,
                           rocblas_int* ninterA,
                           T eps,
                           T sfmin)
{
    const int tid = hipThreadIdx_x;
    const int sbid = hipBlockIdx_x;

    // batch instance
    const int bid = hipBlockIdx_y;
    T* D = load_ptr_batch<T>(DA, bid, shiftD, strideD);
    T* E = load_ptr_batch<T>(EA, bid, shiftE, strideE);
    T* W = WA + bid * strideW;
    rocblas_int* IB = IBA + bid * strideIB;
    rocblas_int* IS = ISA + bid * strideIS;
    T* Esqr = EsqrA + bid * (n - 1);
    T* bounds = boundsA + bid * 2;
    T pmin = pivmin[bid];
    rocblas_int* info = infoA + bid;
    // nofb is the number of split blocks in the matrix
    rocblas_int nofb = nsplit[bid];
    // the bounds and related number of eigenvalues of the intervals
    // in all split blocks are stored in inter and ninter
    T* inter = interA + bid * (4 * n);
    rocblas_int* ninter = ninterA + bid * (4 * n);
    // tmpnev stores the number of eigenvalues found in each split block
    rocblas_int* tmpnev = tmpnevA + bid * n;

    // Shared arrays (sh_name):
    // after each iteration, a thread that found a new interval will activate this flag
    __shared__ int sh_newi[IBISEC_THDS];
    // threads will share the newfound interval bounds and numbers of
    // eigenvalues in these arrays:
    __shared__ T sh_inter[4 * IBISEC_THDS];
    __shared__ int sh_ninter[4 * IBISEC_THDS];

    run_stebz_bisection<IBISEC_BLKS, IBISEC_THDS>(sbid, tid, range, n, abstol, D, E, nofb, W, IB,
                                                  IS, info, tmpnev, pmin, Esqr, bounds, inter,
                                                  ninter, sh_newi, sh_inter, sh_ninter, eps, sfmin);
}

/** STEBZ_SYNTHESIS synthesize the results from all the independent
    split blocks of a given matrix and corrects for values outside of range.
    Call the kernel with as many groups of threads to cover all the matrices in the batch.
    Each thread works with one matrix in the batch. **/
template <typename T>
__device__ void run_stebz_synthesis(const rocblas_erange range,
                                    const rocblas_eorder order,
                                    const rocblas_int n,
                                    const rocblas_int ilow,
                                    const rocblas_int iup,
                                    T* D,
                                    rocblas_int* nev,
                                    const rocblas_int nofb,
                                    T* W,
                                    rocblas_int* IB,
                                    rocblas_int* IS,
                                    rocblas_int* tmpnev,
                                    const T pmin,
                                    T* Esqr,
                                    T* bounds,
                                    T* inter,
                                    rocblas_int* ninter,
                                    T eps)
{
    rocblas_int bin;
    rocblas_int ntmp;
    rocblas_int nn = 0, nnt = 0;
    T tmp, tmp2;
    T bnorm = std::max(std::abs(bounds[0]), std::abs(bounds[1]));

    bool index = (range == rocblas_erange_index);

    // remove gaps in W and IB
    // (if range = index, sometimes the searched interval (vl, vu] could
    // have eigenvalues outside of the desired indices. Thus, the following
    // code also discards those extra values)
    if(index)
    {
        for(int b = 0; b < nofb; ++b)
        {
            bin = (b == 0) ? 0 : IS[b - 1];
            for(int bb = 0; bb < tmpnev[b]; ++bb)
            {
                tmp = W[bin + bb];
                ntmp = IB[bin + bb];
                inter[nnt] = tmp;
                ninter[nnt] = ntmp;
                inter[nnt + n] = tmp;
                nnt++;
            }
        }

        // discard extra values
        increasing_order(nnt, inter + n, ninter + n);

        for(int i = 0; i < nnt; ++i)
        {
            tmp = inter[i];
            for(int j = 0; j < nnt; ++j)
            {
                tmp2 = inter[n + j];
                if(tmp == tmp2)
                {
                    tmp2 = (j == nnt - 1) ? (bounds[1] - tmp2) / 2 : (inter[n + j + 1] - tmp2) / 2;
                    tmp2 += tmp;
                    ntmp = sturm_count(n, D, Esqr, pmin, tmp2);
                    if(ntmp >= ilow && ntmp <= iup)
                    {
                        W[nn] = tmp;
                        IB[nn] = ninter[i];
                        nn++;
                    }
                    break;
                }
            }
        }
    }

    else
    {
        for(int b = 0; b < nofb; ++b)
        {
            bin = (b == 0) ? 0 : IS[b - 1];
            for(int bb = 0; bb < tmpnev[b]; ++bb)
            {
                W[nn] = W[bin + bb];
                IB[nn] = IB[bin + bb];
                nn++;
            }
        }
    }

    // total number of eigenvalues found
    *nev = nn;

    // if ordering is by split blocks, the computed eigenvalues are already in order
    // otherwise re-arrange from smallest to largest
    if(order == rocblas_eorder_entire)
        increasing_order(nn, W, IB);
}

template <typename T, typename U>
ROCSOLVER_KERNEL void stebz_synthesis_kernel(const rocblas_erange range,
                                             const rocblas_eorder order,
                                             const rocblas_int n,
                                             const rocblas_int ilow,
                                             const rocblas_int iup,
                                             U DA,
                                             const rocblas_int shiftD,
                                             const rocblas_int strideD,
                                             rocblas_int* nevA,
                                             rocblas_int* nsplit,
                                             T* WA,
                                             const rocblas_stride strideW,
                                             rocblas_int* IBA,
                                             const rocblas_stride strideIB,
                                             rocblas_int* ISA,
                                             const rocblas_stride strideIS,
                                             const rocblas_int batch_count,
                                             rocblas_int* tmpnevA,
                                             T* pivmin,
                                             T* EsqrA,
                                             T* boundsA,
                                             T* interA,
                                             rocblas_int* ninterA,
                                             T eps)
{
    int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(bid < batch_count)
    {
        // select batch instance
        T* D = load_ptr_batch<T>(DA, bid, shiftD, strideD);
        T* W = WA + bid * strideW;
        rocblas_int* IB = IBA + bid * strideIB;
        rocblas_int* IS = ISA + bid * strideIS;
        rocblas_int nofb = nsplit[bid];
        T* Esqr = EsqrA + bid * (n - 1);
        T pmin = pivmin[bid];
        T* bounds = boundsA + bid * 2;
        rocblas_int* nev = nevA + bid;
        // tmpnev stores the number of eigenvalues found in each split block
        rocblas_int* tmpnev = tmpnevA + bid * n;
        // if range = index, inter and ninter will store temporary ordered eigenvalues
        // with its indices to discard those out of range
        T* inter = interA + bid * 2 * n;
        rocblas_int* ninter = ninterA + bid * 2 * n;

        run_stebz_synthesis(range, order, n, ilow, iup, D, nev, nofb, W, IB, IS, tmpnev, pmin, Esqr,
                            bounds, inter, ninter, eps);
    }
}

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

// Helper to calculate workspace size requirements
template <typename T>
void rocsolver_stebz_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work,
                                   size_t* size_pivmin,
                                   size_t* size_Esqr,
                                   size_t* size_bounds,
                                   size_t* size_inter,
                                   size_t* size_ninter)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_work = 0;
        *size_pivmin = 0;
        *size_Esqr = 0;
        *size_bounds = 0;
        *size_inter = 0;
        *size_ninter = 0;
        return;
    }

    // to store temporary indices or sizes in different kernels
    *size_work = sizeof(rocblas_int) * n * batch_count;

    // to store the value of minimum pivot
    *size_pivmin = sizeof(T) * batch_count;

    // to store the square of the off-diagonal elements
    *size_Esqr = sizeof(T) * (n - 1) * batch_count;

    // to store the bounds of the half-open interval
    // where the eigenvalues will be searched
    *size_bounds = sizeof(T) * 2 * batch_count;

    // to store the bounds of the different intervals during bisection
    *size_inter = sizeof(T) * 4 * n * batch_count;

    // to store the number of eigenvalues corresponding to
    // each bound of the different intervals during bisection
    *size_ninter = sizeof(rocblas_int) * 4 * n * batch_count;
}

// Helper to check argument correctnesss
template <typename T>
rocblas_status rocsolver_stebz_argCheck(rocblas_handle handle,
                                        const rocblas_erange range,
                                        const rocblas_eorder order,
                                        const rocblas_int n,
                                        const T vlow,
                                        const T vup,
                                        const rocblas_int ilow,
                                        const rocblas_int iup,
                                        T* D,
                                        T* E,
                                        rocblas_int* nev,
                                        rocblas_int* nsplit,
                                        T* W,
                                        rocblas_int* IB,
                                        rocblas_int* IS,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(range != rocblas_erange_all && range != rocblas_erange_value && range != rocblas_erange_index)
        return rocblas_status_invalid_value;
    if(order != rocblas_eorder_blocks && order != rocblas_eorder_entire)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
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
    if((n && (!D || !W || !IB || !IS)) || (n > 1 && !E) || !info || !nev || !nsplit)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// stebz template function implementation
template <typename T, typename U>
rocblas_status rocsolver_stebz_template(rocblas_handle handle,
                                        const rocblas_erange range,
                                        const rocblas_eorder order,
                                        const rocblas_int n,
                                        const T vlow,
                                        const T vup,
                                        const rocblas_int ilow,
                                        const rocblas_int iup,
                                        const T abstol,
                                        U D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        U E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        rocblas_int* nev,
                                        rocblas_int* nsplit,
                                        T* W,
                                        const rocblas_stride strideW,
                                        rocblas_int* IB,
                                        const rocblas_stride strideIB,
                                        rocblas_int* IS,
                                        const rocblas_stride strideIS,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        rocblas_int* work,
                                        T* pivmin,
                                        T* Esqr,
                                        T* bounds,
                                        T* inter,
                                        rocblas_int* ninter)
{
    ROCSOLVER_ENTER("stebz", "erange:", range, "eorder:", order, "n:", n, "vl:", vlow, "vu:", vup,
                    "il:", ilow, "iu:", iup, "abstol:", abstol, "shiftD:", shiftD,
                    "shiftE:", shiftE, "bc:", batch_count);

    // quick return (no batch)
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = nev = nsplit = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nev, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nsplit, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return (dimension zero)
    if(n == 0)
        return rocblas_status_success;

    // quick return (dimension 1)
    if(n == 1)
    {
        ROCSOLVER_LAUNCH_KERNEL(stebz_case1_kernel<T>, gridReset, threads, 0, stream, range, vlow,
                                vup, D, shiftD, strideD, nev, nsplit, W, strideW, IB, strideIB, IS,
                                strideIS, batch_count);

        return rocblas_status_success;
    }

    // numerics constants:
    // machine epsilon
    T eps = get_epsilon<T>();
    // smallest safe real (i.e. 1/sfmin does not overflow)
    T sfmin = get_safemin<T>();
    // absolute tolerance for evaluating when an eigenvalue interval is small
    // enough to consider it as converged. By default, if abstol = 0, set this to
    // the best accuracy value
    T atol = (abstol == 0) ? 2 * sfmin : abstol;

    // split matrix into independent blocks and prepare for iterative bisection
    ROCSOLVER_LAUNCH_KERNEL(stebz_splitting_kernel<T>, dim3(1, batch_count), dim3(STEBZ_SPLIT_THDS),
                            0, stream, range, n, vlow, vup, ilow, iup, D, shiftD, strideD, E,
                            shiftE, strideE, nsplit, W, strideW, IS, strideIS, work, pivmin, Esqr,
                            bounds, inter, ninter, eps, sfmin);

    // Implement iterative bisection on each split block.
    // The next kernel has IBISEC_BLKS thread-blocks with IBISEC_THDS threads.
    // Each thread works with as many non-converged intervals as needed on each iteration.
    // Each thread-block is working with as many split-off blocks as needed to cover
    // the entire matrix.

    /** (TODO: in the future, we can evaluate if transferring nsplit -the number of
        split-off blocks- into the host, to launch exactly that amount of thread-blocks,
        could give better performance) **/

    ROCSOLVER_LAUNCH_KERNEL(stebz_bisection_kernel<T>, dim3(IBISEC_BLKS, batch_count),
                            dim3(IBISEC_THDS), 0, stream, range, n, atol, D, shiftD, strideD, E,
                            shiftE, strideE, nsplit, W, strideW, IB, strideIB, IS, strideIS, info,
                            work, pivmin, Esqr, bounds, inter, ninter, eps, sfmin);

    // Finally, synthesize the results from all the split blocks
    ROCSOLVER_LAUNCH_KERNEL(stebz_synthesis_kernel<T>, gridReset, threads, 0, stream, range, order, n,
                            ilow, iup, D, shiftD, strideD, nev, nsplit, W, strideW, IB, strideIB, IS,
                            strideIS, batch_count, work, pivmin, Esqr, bounds, inter, ninter, eps);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
