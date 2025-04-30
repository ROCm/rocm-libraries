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
#include "lapack_device_functions.hpp"
#include "rocauxiliary_stedc.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

// TODO: using macro STEDCX_EXTERNAL_GEMM = false for now. We can enable the use of
// external gemm updates once the development is completed for stedc.
#define STEDCX_EXTERNAL_GEMM false

/***************** Device auxiliary functions *****************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_NUM_LEVELS returns the ideal number of times/levels in which a matrix (or split block)
    will be divided during the divide phase of divide & conquer algorithm.
    i.e. number of sub-blocks = 2^levels **/
template <>
__host__ __device__ inline rocblas_int
    stedc_num_levels<rocsolver_stedc_mode_bisection>(const rocblas_int n)
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
/** STEDCX_SYNTHESIS_KERNEL synthesizes the results of the partial decomposition **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
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
    rocblas_int maxlevs = stedc_num_levels<rocsolver_stedc_mode_bisection>(n);
    rocblas_int maxblks = 1 << maxlevs;

    // find independent split blocks in matrix and prepare range for partial decomposition
    ROCSOLVER_LAUNCH_KERNEL(stedcx_split_kernel, dim3(1, batch_count), dim3(STEBZ_SPLIT_THDS), 0,
                            stream, erange, n, vl, vu, il, iu, D, strideD, E, strideE, W, strideW,
                            splits, work_stack, eps, ssfmin);

    // 1. divide phase
    //-----------------------------
    ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<rocsolver_stedc_mode_bisection, S>),
                            dim3(batch_count), dim3(STEDC_BDIM), 0, stream, n, D, strideD, E,
                            strideE, splits);

    // 2. solve phase
    //-----------------------------
    ROCSOLVER_LAUNCH_KERNEL((stedc_solve_kernel<S>), dim3(maxblks, STEDC_NUM_SPLIT_BLKS, batch_count),
                            dim3(1), 0, stream, n, D, strideD, E, strideE, tempvect, 0, ldt,
                            strideT, info, work_steqr, splits, eps, ssfmin, ssfmax);

    // 3. merge phase
    //----------------
    size_t lmemsize1 = sizeof(S) * 2 * STEDC_BDIM;
    size_t lmemsize3 = sizeof(S) * STEDC_BDIM;
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
        ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_kernel<rocsolver_stedc_mode_bisection, S>),
                                dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM),
                                lmemsize1, stream, k, n, D, strideD, E, strideE, tempvect, 0, ldt,
                                strideT, tmpz, tempgemm, splits, eps);

        // b. solve to find merged eigen values
        ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_kernel<rocsolver_stedc_mode_bisection, S>),
                                dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM),
                                0, stream, k, n, D, strideD, E, strideE, tmpz, tempgemm, splits,
                                eps, ssfmin, ssfmax);

        // c. find merged eigen vectors
        ROCSOLVER_LAUNCH_KERNEL(
            (stedc_mergeVectors_kernel<rocsolver_stedc_mode_bisection, STEDCX_EXTERNAL_GEMM, S>),
            dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM), lmemsize3, stream,
            k, n, D, strideD, E, strideE, tempvect, 0, ldt, strideT, tmpz, tempgemm, splits);

        // d. update level
        ROCSOLVER_LAUNCH_KERNEL((stedc_mergeUpdate_kernel<rocsolver_stedc_mode_bisection, S>),
                                dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM),
                                lmemsize3, stream, k, n, D, strideD, tempvect, 0, ldt, strideT,
                                tmpz, tempgemm, splits);
    }

    // 4. update and sort
    //----------------------
    // Synthesize the results from all the split blocks
    ROCSOLVER_LAUNCH_KERNEL(stedcx_synthesis_kernel, dim3(1, batch_count), dim3(STEDC_BDIM), 0,
                            stream, erange, n, il, iu, D, strideD, nev, W, strideW, tempvect, ldt,
                            strideT, batch_count, splits, work_stack, eps);

    // eigenvectors C <- C*tempvect
    local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                    work_stack, 0, ldt, strideT, batch_count, workArr);

    // sort eigenvalues and eigenvectors
    ROCSOLVER_LAUNCH_KERNEL((stedc_sort<T>), dim3(1, 1, batch_count), dim3(BS1), 0, stream, n, W,
                            strideW, C, shiftC, ldc, strideC, batch_count, splits, nev);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
