/************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocauxiliary_stedc.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

#define MAXSWEEPS 20 // Max number of sweeps for Jacobi solver (when used)

// TODO: using macro STEDCJ_EXTERNAL_GEMM = false for now. We can enable the use of
// external gemm updates once the development is completed for stedc.
#define STEDCJ_EXTERNAL_GEMM false

/***************** Device auxiliary functions *****************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_NUM_LEVELS returns the ideal number of times/levels in which a matrix
   (or split block) will be divided during the divide phase of divide & conquer
   algorithm. i.e. number of sub-blocks = 2^levels **/
template <>
__host__ __device__ inline rocblas_int
    stedc_num_levels<rocsolver_stedc_mode_jacobi>(const rocblas_int n)
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
/** STEDC_SOLVE_KERNEL implements the solver phase of the DC algorithm to
        compute the eigenvalues/eigenvectors of the different sub-blocks of each
   split-block. A matrix in the batch could have many split-blocks, and each
   split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS
   groups in y and nn groups in x. Groups are size STEDC_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
   be analysed in parallel). If there are actually more split-blocks, some
   groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
   the size n. If a group has an id larger than the actual number of sub-blocks
   in a split-block, it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
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
        levs = stedc_num_levels<rocsolver_stedc_mode_jacobi>(bs);
        blks = 1 << levs;

        // 2. SOLVE PHASE
        /* ----------------------------------------------------------------- */
        // Solve the blks sub-blocks in parallel.

        if(tid < blks)
        {
            sbs = ns[tid];
            p2 = ps[tid];

            // transform D and E into full upper tridiag matrix and copy to C
            de2tridiag(STEDC_BDIM, tidb, sbs, D + p2, E + p2, C + p2 + p2 * ldc, ldc);

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
            syevj_get_dims(sbs, STEDC_BDIM, &ddx, &ddy);
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

/******************* Host functions *********************************************/
/*******************************************************************************/

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
    rocblas_int maxlevs = stedc_num_levels<rocsolver_stedc_mode_jacobi>(n);
    rocblas_int maxblks = 1 << maxlevs;

    // find independent split blocks in matrix
    ROCSOLVER_LAUNCH_KERNEL(stedc_split, dim3(batch_count), dim3(1), 0, stream, n, D, strideD, E,
                            strideE, splits_map, eps);

    // 1. divide phase
    //-----------------------------
    ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<rocsolver_stedc_mode_jacobi, S>), dim3(batch_count),
                            dim3(STEDC_BDIM), 0, stream, n, D, strideD, E, strideE, splits_map);

    // 2. solve phase
    //-----------------------------
    size_t lmemsize = (n + n % 2) * (sizeof(rocblas_int) + sizeof(S));

    ROCSOLVER_LAUNCH_KERNEL((stedcj_solve_kernel<S>),
                            dim3(maxblks, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM),
                            lmemsize, stream, n, D, strideD, E, strideE, tempvect, 0, ldt, strideT,
                            info, static_cast<S*>(work_stack), splits_map, eps, ssfmin, ssfmax);

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
        // a. prepare secular equations
        rocblas_int numgrps2 = 1 << (maxlevs - 1 - k);
        ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_kernel<rocsolver_stedc_mode_jacobi, S>),
                                dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM),
                                lmemsize1, stream, k, n, D, strideD, E, strideE, tempvect, 0, ldt,
                                strideT, tmpz, tempgemm, splits_map, eps);

        // b. solve to find merged eigen values
        ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_kernel<rocsolver_stedc_mode_jacobi, S>),
                                dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM),
                                0, stream, k, n, D, strideD, E, strideE, tmpz, tempgemm, splits_map,
                                eps, ssfmin, ssfmax);

        // c. find merged eigen vectors
        ROCSOLVER_LAUNCH_KERNEL(
            (stedc_mergeVectors_kernel<rocsolver_stedc_mode_jacobi, STEDCJ_EXTERNAL_GEMM, S>),
            dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM), lmemsize3, stream,
            k, n, D, strideD, E, strideE, tempvect, 0, ldt, strideT, tmpz, tempgemm, splits_map);

        // c. update level
        ROCSOLVER_LAUNCH_KERNEL((stedc_mergeUpdate_kernel<rocsolver_stedc_mode_jacobi, S>),
                                dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM),
                                lmemsize3, stream, k, n, D, strideD, tempvect, 0, ldt, strideT,
                                tmpz, tempgemm, splits_map);
    }

    // 4. update and sort
    //----------------------
    // eigenvectors C <- C*tempvect
    local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                    static_cast<S*>(work_stack), 0, ldt, strideT, batch_count,
                                    workArr);

    ROCSOLVER_LAUNCH_KERNEL((stedc_sort<T>), dim3(1, 1, batch_count), dim3(BS1), 0, stream, n, D,
                            strideD, C, shiftC, ldc, strideC, batch_count, splits_map);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
