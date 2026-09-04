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

#include "asan_helpers.hpp"
#include "auxiliary/rocauxiliary_stebz.hpp"
#include "auxiliary/rocauxiliary_stein.hpp"
#include "auxiliary/rocauxiliary_stedc.hpp"
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

//#define STEDCX_EXTERNAL_GEMM false
//#define STEDCX_SETRANGE_THDS 256
//#define STEDCX_SYNTHESIS_THDS 256 


/*************** Main kernels *********************************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** This kernel deals with the case n = 1 **/
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
/** STEDCX_SETRANGE_KERNEL determines the range for the partial decomposition **/
/*template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_SETRANGE_THDS)
    stedcx_setrange_kernel(const rocblas_erange range,
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
                        rocblas_int* ninterA,
                        S* workA,
                        S* interA,
                        const S eps,
                        const S ssfmin)
{
    // batch instance
    const int tid = hipThreadIdx_x;
    const int bid = hipBlockIdx_y;
    const int bdim = hipBlockDim_x;
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* ninter = ninterA + bid * (3 * n);
    S* bounds = workA + bid * (2 * n + 2);
    S* inter = interA + bid * (2 * n);
    S* W = WW + bid * strideW;
    
    // workspace
    S* pivmin = bounds + 2;
    S* Esqr = pivmin + 1;
    S* Dcpy = Esqr + n - 1;

    // make copy of D for future use if necessary
    if(range == rocblas_erange_index)
    {
        for(rocblas_int i = tid; i < n; i += bdim)
            Dcpy[i] = D[i];
    }

    // shared memory setup for iamax.
    __shared__ S sval[STEDCX_SETRANGE_THDS];
    __shared__ rocblas_int sidx[STEDCX_SETRANGE_THDS];
    
    // Split blocks are no longer considered during the divide and conquer process.
    // Set nsplit = IS = tmpIS = nullptr
    rocblas_int* nsplit = nullptr;
    rocblas_int* IS = nullptr;
    rocblas_int* tmpIS = nullptr;

    run_stebz_splitting<STEDCX_SETRANGE_THDS>(tid, range, n, vl, vu, il, iu, D, E, nsplit, W, IS,
                                          tmpIS, pivmin, Esqr, bounds, inter, ninter, sval, sidx,
                                          eps, ssfmin);
}*/

//--------------------------------------------------------------------------------------//
/** STEDCX_SELECT_KERNEL selects the results of the partial decomposition **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void stedcx_select_kernel(const rocblas_erange range,
                                               const rocblas_int n,
                                               const S vl,
                                               const S vu,
                                               const rocblas_int il,
                                               const rocblas_int iu,
                                               S* DD,
                                               const rocblas_stride strideD,
                                               rocblas_int* nevA,
                                               S* WW,
                                               const rocblas_stride strideW,
                                               U CC,
                                               const rocblas_int shiftC,
                                               const rocblas_int ldc,
                                               const rocblas_stride strideC,
                                               T* VV,
                                               const rocblas_int ldv,
                                               const rocblas_stride strideV,
                                               const rocblas_int batch_count)
{
    const int tidx = hipThreadIdx_x;
    const int tidy = hipThreadIdx_y;
    const int bidx = hipBlockIdx_x;
    const int bidy = hipBlockIdx_y;
    const int bid = hipBlockIdx_z;
    const int bdimx = hipBlockDim_x;
    const int bdimy = hipBlockDim_y;
    const int gdimx = hipGridDim_x;
    const int gdimy = hipGridDim_y;
    const int myrow = bidx * bdimx + tidx;
    const int mycol = bidy * bdimy + tidy;
    const int step_row = bdimx * gdimx;
    const int step_col = bdimy * gdimy;
    
    // batch instance
    S* D = DD + bid * strideD;
    S* W = WW + bid * strideW;
    T* C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    T* V = VV + bid * strideV;
    rocblas_int* nev = nevA + bid;

    // all values in positions 'in' till 'out' will be selected
    bool value = (range == rocblas_erange_value);
    bool all = (range == rocblas_erange_all);
    rocblas_int in = il - 1;
    rocblas_int out = iu;
    if(all)
    {
        in = 0;
        out = n;
    }
    else if(value)
    {
        in = bisearch(vl, D, n, false, false);
        out = bisearch(vu, D, n, false, false);
    }

    // select values and corresponding vectors
    for(auto j = in + mycol; j < out; j += step_col)
    {
        if(myrow == 0)
            W[j - in] = D[j];

        for(auto i = myrow; i < n; i += step_row)
            C[i + (j - in) * ldc] = V[i + j * ldv];
    }  
        
    // final number of selected values
    if(myrow == 0 && mycol == 0)
        *nev = out - in;
}


//--------------------------------------------------------------------------------------//
/** STEDCX_SYNTHESIS_KERNEL synthesizes the results of the partial decomposition **/
/*template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(STEDCX_SYNTHESIS_THDS)
    stedcx_synthesis_kernel(const rocblas_erange range,
                            const rocblas_int n,
                            const rocblas_int il,
                            const rocblas_int iu,
                            S* DD,
                            const rocblas_stride strideD,
                            rocblas_int* nevA,
                            S* WW,
                            const rocblas_stride strideW,
                            U CC,
                            const rocblas_int shiftC,
                            const rocblas_int ldc,
                            const rocblas_stride strideC,
                            T* VV,
                            const rocblas_int ldv,
                            const rocblas_stride strideV,
                            const rocblas_int batch_count,
                            rocblas_int* ninterA,
                            S* workA,
                            S* interA,
                            const S eps)
{
    // batch instance
    const int tid = hipThreadIdx_x;
    const int bid = hipBlockIdx_y;
    const int bdim = hipBlockDim_x;
    S* D = DD + bid * strideD;
    S* W = WW + bid * strideW;
    T* C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    T* V = VV + bid * strideV;
    rocblas_int* nev = nevA + bid;
    rocblas_int* ninter = ninterA + bid * (3 * n);
    S* bounds = workA + bid * (2 * n + 2);
    S* inter = interA + bid * (2 * n);

    // workspace
    rocblas_int* idd = ninter + 2 * n;
    S* pmin = bounds + 2;
    S* Esqr = pmin + 1;
    S* Dcpy = Esqr + n - 1;

    // aux variables
    S tmp, tmp2;
    rocblas_int nn = 0, nnt = 0, ntmp = 0;
    bool index = (range == rocblas_erange_index);
    bool all = (range == rocblas_erange_all);
    S low, up;

    if(all)
    {
        // if computing all eigenvalues
        *nev = n;
        for(int k = tid; k < n; k += bdim)
        {
            W[k] = D[k];
            idd[k] = 1;
        }
    }
    else if(tid == 0)
    {
        // if only keeping eigenvalues in desired range
        low = bounds[0];
        up = bounds[1];

        if(!index)
        {
            // range given by value
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
            // range given by index
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

    // keep corresponding eigenvectors
    nn = 0;
    for(int j = 0; j < n; ++j)
    {
        if(idd[j] == 1)
        {
            for(int i = tid; i < n; i += bdim)
                C[i + nn * ldc] = V[i + j * ldv];
            nn++;
        }
        __syncthreads();
    }
}*/

//--------------------------------------------------------------------------------------//
/** STEDCX_SORT sorts computed eigenvalues and eigenvectors in increasing order **/
/*template <typename T, typename S, typename U>
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
}*/



/******************* Host functions ********************************************/
/*******************************************************************************/

//--------------------------------------------------------------------------------------//
/** This helper calculates required workspace size **/
template <bool BATCHED, typename T, typename S>
void rocsolver_stedcx_getMemorySize(const rocblas_evect evect,
                                    const rocblas_int n,
                                    const rocblas_int batch_count,
                                    size_t* size_tmpT,
                                    size_t* size_work,
                                    size_t* size_work_stack,
                                    size_t* size_tempvect,
                                    size_t* size_tempgemm,
                                    size_t* size_tmpz,
                                    size_t* size_splits,
                                    size_t* size_workArr)
{
    // if quick return no workspace needed
    *size_tmpT = 0;
    *size_work = 0;
    *size_work_stack = 0;
    *size_tempvect = 0;
    *size_tempgemm = 0;
    *size_tmpz = 0;
    *size_splits = 0;
    *size_workArr = 0;
    if(n <= 1 || !batch_count)
        return;
    
    size_t s1, s2, t1, t2;

    // requirements for D&C solver 
    rocsolver_stedc_getMemorySize<BATCHED, T, S>(evect, n, batch_count, &s1,
                    size_tempvect, size_tempgemm, size_tmpz, &t1, size_workArr);

    // extra requirements for partial decomposition
//    *size_work = sizeof(S) * (2 * n + 2) * batch_count;
    *size_tmpT = sizeof(T) * (n * n) * batch_count;
//    s2 = sizeof(S) * (2 * n) * batch_count;
//    t2 = sizeof(rocblas_int) * (3 * n) * batch_count;
    s2 = 0;
    t2 = 0;
    
    *size_work_stack = std::max(s1, s2);
    *size_splits = std::max(t1, t2);
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
                                         T* tmpT,
                                         S* work,
                                         S* work_stack,
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

//printf("\n-----------INPUTS--------------\n");
//print_device_matrix(std::cout,"D",1,n,D,1);
//print_device_matrix(std::cout,"E",1,n-1,E,1);

    // aux constants
//    S eps = get_epsilon<S>();
//    S ssfmin = get_safemin<S>();
//    S ssfmax = S(1.0) / ssfmin;
//    ssfmin = sqrt(ssfmin) / (eps * eps);
//    ssfmax = sqrt(ssfmax) / S(3.0);

    // find range for partial decomposition
//    ROCSOLVER_LAUNCH_KERNEL(stedcx_setrange_kernel, dim3(1, batch_count), dim3(STEDCX_SETRANGE_THDS), 0,
//                            stream, erange, n, vl, vu, il, iu, D, strideD, E, strideE, W, strideW,
//                            splits, work, work_stack, eps, ssfmin);

//print_device_matrix(std::cout,"bounds",1,n+2,work,1);

    // find values and vectors with divide & conquer
    constexpr bool ISBATCHED = BATCHED || STRIDED;
    rocblas_int ldt = n;
    rocblas_stride strideT = n * n;
    /** TODO: Although stedc accepts batched calls (with C as an array of pointers), in practice it
            only works for strided-batched (a simple array C). This was never caught in tests because 
            syevd always calls stedc as strided-batched. For this reason, we cannot call stedc using C
            directly; we need to pass a temporary array tmpT. We need to decide if we want to fix this
            in the future. **/  
    /** TODO: at the last level of the merge tree, we could skip computations of
            eigen values and vectors that are out of the desired range. Whether this could be
            exploited somehow to improve performance must be explored in the future. **/
    rocsolver_stedc_template<false, ISBATCHED, T>(
        handle, rocblas_evect_tridiagonal, n, D, 0, strideD, E, 0, strideE, 
        tmpT, 0, ldt, strideT, info, batch_count, work_stack, tempvect, 
        tempgemm, tmpz, splits, workArr);        



//printf("\n-----------AFTER D&C--------------\n");
//print_device_matrix(std::cout,"D",1,n,D,1);
//print_device_matrix(std::cout,"E",1,n-1,E,1);
//print_device_matrix(std::cout,"tmpT",n,n,tmpT,ldt);

    // Discard values and vectors out of range
    rocblas_int nblocks = ceildiv(n, BS2); 
    ROCSOLVER_LAUNCH_KERNEL((stedcx_select_kernel<T>), dim3(nblocks, nblocks, batch_count), dim3(BS2, BS2), 0,
                            stream, erange, n, vl, vu, il, iu, D, strideD, nev, W, strideW, 
                            C, shiftC, ldc, strideC, tmpT, ldt, strideT, batch_count);

//printf("\n-----------AFTER SYNTHESIS--------------\n");
//print_device_matrix(std::cout,"nev",1,1,nev,1);
//print_device_matrix(std::cout,"W",1,n,W,1);
//print_device_matrix(std::cout,"C",n,n,C,ldc);

    // sort selected eigenvalues and eigenvectors
//    ROCSOLVER_LAUNCH_KERNEL((stedcx_sort<T>), dim3(1, 1, batch_count), dim3(BS1), 0, stream, n, W,
//                            strideW, C, shiftC, ldc, strideC, batch_count, splits, nev);

//printf("\n-----------SORTED OUTPUTS--------------\n");
//print_device_matrix(std::cout,"nev",1,1,nev,1);
//print_device_matrix(std::cout,"W",1,n,W,1);
//print_device_matrix(std::cout,"C",n,n,C,ldc);

    return rocblas_status_success;
}

#undef STEDCX_EXTERNAL_GEMM
#undef STEDCX_SETRANGE_THDS
#undef STEDCX_SYNTHESIS_THDS

ROCSOLVER_END_NAMESPACE

