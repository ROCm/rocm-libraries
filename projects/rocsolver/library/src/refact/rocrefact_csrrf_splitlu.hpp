/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <rocprim/rocprim.hpp>

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef SPLITLU_SWITCH_SIZE
#define SPLITLU_SWITCH_SIZE 64
#endif

#ifndef SPLITLU_SETUP_THREADS
#define SPLITLU_SETUP_THREADS(lid, wid, waveSize, nwaves)     \
    {                                                         \
        waveSize = hipBlockDim_x;                             \
        nwaves = hipBlockDim_y * hipGridDim_x;                \
        wid = hipThreadIdx_y + hipBlockIdx_x * hipBlockDim_y; \
        lid = hipThreadIdx_x;                                 \
    }
#endif

template <typename I>
__host__ __device__ static I cal_wave_size(I avg_nnzM)
{
    // -----------------------------------------------------------------------------------
    // Ideally to  encourage stride-1 coalesced access to compressed sparse data
    // one would like to use a full warp to process a row. However, if the number
    // of non-zeros per row is small, say about 16, then using 64 threads in a
    // warp may not be most efficient since many threads will have no work. One
    // may consider repartitioning the threads in a warp to assign 4 threads to
    // process a row, and process 16 rows concurrently
    //
    // The threads assigned to process the same row will be called a virtual
    // "wave" of threads.
    //
    // This routine uses a heuristic to determine the number of threads (called
    // wave_size) in a virtual "wave"
    // -----------------------------------------------------------------------------------

    const auto ifactor = 2;

    const auto wave_size = ((avg_nnzM >= ifactor * warpSize)              ? warpSize
                                : (avg_nnzM >= ifactor * (warpSize / 2))  ? (warpSize / 2)
                                : (avg_nnzM >= ifactor * (warpSize / 4))  ? (warpSize / 4)
                                : (avg_nnzM >= ifactor * (warpSize / 8))  ? (warpSize / 8)
                                : (avg_nnzM >= ifactor * (warpSize / 16)) ? (warpSize / 16)
                                                                          : 1);
    return wave_size;
}

template <typename T>
ROCSOLVER_KERNEL __launch_bounds__(BS1) void check_nzLU_kernel(const rocblas_int n,
                                                               const rocblas_int nnzT,
                                                               rocblas_int const* const ptrT,
                                                               rocblas_int const* const indT,
                                                               rocblas_int* const nzLarray,
                                                               rocblas_int* const nzUarray)
{
    auto const tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    auto const nthreads = hipBlockDim_x * hipGridDim_x;
    auto const irow_start = tid;
    auto const irow_inc = nthreads;

    __syncthreads();
    for(rocblas_int irow = irow_start; irow < n; irow += irow_inc)
    {
        auto const istart = ptrT[irow];
        auto const iend = ptrT[irow + 1];
        auto const nnzTrow = (iend - istart);

        rocblas_int nnzLrow = 0;
        rocblas_int nnzUrow = 0;
        for(auto i = istart; i < iend; i++)
        {
            auto const icol = indT[i];
            bool const is_upper = (irow <= icol);
            if(is_upper)
            {
                nnzUrow += 1;
            }
            else
            {
                nnzLrow += 1;
            }
        }
        nnzLrow += 1; // add one for unit diagonal

#ifdef NDEBUG
#else
        {
            bool const isvalid = (nnzUrow == nzUarray[irow]) && (nnzLrow == nzLarray[irow])
                && (nnzTrow == (nnzLrow - 1) + nnzUrow);

            if(!isvalid)
            {
                printf("irow=%d,nnzTrow=%d,nnzLrow=%d,nzLarray[irow]=%d,nnzUrow=%d,nzUarray=%d\n",
                       irow, nnzTrow, nnzLrow, nzLarray[irow], nnzUrow, nzUarray[irow]);
            }
            assert(isvalid);
        }
#endif
    }
    __syncthreads();
}

template <typename T>
ROCSOLVER_KERNEL __launch_bounds__(BS1) void rf_splitLU_gen_nzLU_kernel(const rocblas_int n,
                                                                        const rocblas_int nnzM,
                                                                        rocblas_int const* const Mp,
                                                                        rocblas_int const* const Mi,
                                                                        rocblas_int* const nzLarray,
                                                                        rocblas_int* const nzUarray)
{
    rocblas_int wid = 0;
    rocblas_int lid = 0;
    rocblas_int waveSize = 0;
    rocblas_int nwaves = 0;

    SPLITLU_SETUP_THREADS(lid, wid, waveSize, nwaves);
    auto const lwid = hipThreadIdx_y;

    // ------------------------------------------------------------------
    // Note: the code may not work correctly if number of threads in thread block
    // is not a multiple of warpSize
    // ------------------------------------------------------------------

    auto const ld = waveSize;
    auto idx2D = [=](auto i, auto j, auto ld) { return i + j * ld; };

    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const auto kstart = Mp[irow];
        const auto kend = Mp[irow + 1];
        const auto nnzTrow = (kend - kstart);

        // -------------------------------------
        // calculate number of non-zeros per row
        // -------------------------------------

        rocblas_int lnnzUrow = 0;
        rocblas_int lnnzLrow = 0;

        for(auto k = kstart + lid; (k < kend); k += waveSize)
        {
            const auto icol = Mi[k];
            bool const is_upper_triangular = (irow <= icol);
            if(is_upper_triangular)
            {
                lnnzUrow += 1;
            }
            else
            {
                lnnzLrow += 1;
            }
        }
        atomicAdd(&(nzUarray[irow]), lnnzUrow);

        // add 1 for unit diagonal
        atomicAdd(&(nzLarray[irow]), (lid == 0) ? 1 + lnnzLrow : lnnzLrow);

    } // end for irow
}

template <typename T>
ROCSOLVER_KERNEL void __launch_bounds__(BS1)
    rf_splitLU_copy_kernel(const rocblas_int n,
                           const rocblas_int nnzM,
                           rocblas_int const* const __restrict__ Mp,
                           rocblas_int const* const __restrict__ Mi,
                           T const* const __restrict__ Mx,
                           rocblas_int const* const __restrict__ Lp,
                           rocblas_int* const __restrict__ Li,
                           T* const __restrict__ Lx,
                           rocblas_int const* const __restrict__ Up,
                           rocblas_int* const __restrict__ Ui,
                           T* const __restrict__ Ux)
{
    rocblas_int wid = 0;
    rocblas_int lid = 0;
    rocblas_int waveSize = 0;
    rocblas_int nwaves = 0;

    SPLITLU_SETUP_THREADS(lid, wid, waveSize, nwaves);

    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const auto kstart = Mp[irow];
        const auto kend = Mp[irow + 1];
        const auto nz = (kend - kstart);

        const auto nzU = (Up[irow + 1] - Up[irow]);
        const auto nzL = nz - nzU;
        const auto ubegin = kend - nzU; // start of upper triangular part

        // ------------------------------------------------
        // Note: assume column indices are in increasing order
        // ------------------------------------------------

        // ---------------------
        // lower triangular part
        // ---------------------
        for(auto k = kstart + lid; k < ubegin; k += waveSize)
        {
            const auto aij = Mx[k];
            const auto icol = Mi[k];
            const auto ip = Lp[irow] + (k - kstart);
            Li[ip] = icol;
            Lx[ip] = aij;
        }

        // ------------------------
        // unit diagonal entry of L
        // ------------------------
        if(lid == 0)
        {
            const auto ip = Lp[irow + 1] - 1;
            Li[ip] = irow;
            Lx[ip] = static_cast<T>(1);
        }

        // ---------------------
        // upper triangular part
        // ---------------------
        for(auto k = ubegin + lid; k < kend; k += waveSize)
        {
            const auto aij = Mx[k];
            const auto icol = Mi[k];
            const auto ip = Up[irow] + (k - ubegin);
            Ui[ip] = icol;
            Ux[ip] = aij;
        }
    } // end for irow
}

// ----------------------------------------------
// Note: intended for execution on a single block
// ----------------------------------------------
template <typename T>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) rf_splitLU_kernel(const rocblas_int n,
                                                               const rocblas_int nnzM,
                                                               rocblas_int const* const Mp,
                                                               rocblas_int const* const Mi,
                                                               T const* const Mx,
                                                               rocblas_int* const Lp,
                                                               rocblas_int* const Li,
                                                               T* const Lx,
                                                               rocblas_int* const Up,
                                                               rocblas_int* const Ui,
                                                               T* const Ux,
                                                               rocblas_int* work)
{
    if(hipBlockIdx_x != 0)
    {
        return;
    }

    rocblas_int wid = 0;
    rocblas_int lid = 0;
    rocblas_int waveSize = 0;
    rocblas_int nwaves = 0;

    SPLITLU_SETUP_THREADS(lid, wid, waveSize, nwaves);

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    size_t const LDS_MAX_SIZE = 64 * 1024;
    auto const N_MAX = LDS_MAX_SIZE / sizeof(rocblas_int);
    __shared__ rocblas_int work_lds[N_MAX];

    rocblas_int* const nnzUrow = (n <= N_MAX) ? &(work_lds[0]) : work;

    // -------------------------------------------------
    // 1st pass to determine number of non-zeros per row
    // and set up Lp and Up
    // -------------------------------------------------
    __syncthreads();
    for(auto irow = tid; irow < n; irow += nthreads)
    {
        nnzUrow[irow] = 0;
    };
    __syncthreads();

    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const rocblas_int istart = Mp[irow];
        const rocblas_int iend = Mp[irow + 1];

        rocblas_int lnnzUrow = 0;
        for(auto i = istart + lid; i < iend; i += waveSize)
        {
            const auto icol = Mi[i];

            bool const is_upper_triangular = (irow <= icol);
            if(is_upper_triangular)
            {
                lnnzUrow += 1;
            }
        }
        atomicAdd(&(nnzUrow[irow]), lnnzUrow);
    }
    __syncthreads();

    // ---------------------------------
    // prefix sum to setup Lp[] and Up[]
    // ---------------------------------

    __syncthreads();

    if(tid == 0)
    {
        rocblas_int nnzL = 0;
        rocblas_int nnzU = 0;

        for(rocblas_int irow = 0; irow < n; irow++)
        {
            Lp[irow] = nnzL;
            Up[irow] = nnzU;

            const auto istart = Mp[irow];
            const auto iend = Mp[irow + 1];
            const auto nnzTrow = (iend - istart);

            const auto nzUp_i = nnzUrow[irow];
            const auto nzLp_i = (nnzTrow - nzUp_i) + 1; // add 1 for unit diagonal

            nnzL += nzLp_i;
            nnzU += nzUp_i;
        }

        Lp[n] = nnzL;
        Up[n] = nnzU;
    }
    __syncthreads();
#ifdef NDEBUG
#else
    {
        // ------------------
        // correctness  check
        // ------------------
        auto const nnzL = Lp[n] - Lp[0];
        auto const nnzU = Up[n] - Up[0];
        auto const nnzT = Mp[n] - Mp[0];
        assert(nnzT == ((nnzL - n) + nnzU));
    }
#endif

    // ------------------------------------
    // 2nd pass to populate Li, Lx, Ui, Ux
    // ------------------------------------
    for(auto irow = wid; irow < n; irow += nwaves)
    {
        const auto istart = Mp[irow];
        const auto iend = Mp[irow + 1];

        auto const nnzUrow = Up[irow + 1] - Up[irow];
        auto const Ustart = (iend - nnzUrow);
        // --------------
        // copy into L, U
        // --------------
        for(auto k = istart + lid; k < iend; k += waveSize)
        {
            const auto aij = Mx[k];
            auto const icol = Mi[k];

            bool const is_upper_triangular = (irow <= icol);
            if(is_upper_triangular)
            {
                auto const offset = (k - Ustart);
                auto const ip = Up[irow] + offset;
                Ui[ip] = icol;
                Ux[ip] = aij;
            }
            else
            {
                auto const offset = (k - istart);
                auto const ip = Lp[irow] + offset;
                Li[ip] = icol;
                Lx[ip] = aij;
            }
        }
    }

    __syncthreads();
    // -----------------------------
    // set unit diagonal entry in L
    // -----------------------------
    for(auto irow = tid; irow < n; irow += nthreads)
    {
        const auto j = Lp[irow + 1] - 1;
        Li[j] = irow;
        Lx[j] = static_cast<T>(1);
    }
    __syncthreads();
}

template <typename T>
rocblas_status rocsolver_csrrf_splitlu_getMemorySize(const rocblas_int n,
                                                     const rocblas_int nnzT,
                                                     rocblas_int* ptrT,
                                                     size_t* size_work)
{
    // if quick return, no need of workspace
    if(n == 0 || nnzT == 0)
    {
        *size_work = 0;
        return rocblas_status_success;
    }

    // space to store the number of non-zeros per row in L and U
    const size_t size_work_LU = sizeof(rocblas_int) * 2 * n;

    // ------------------------------------------
    // query amount of temporary storage required
    // ------------------------------------------
    size_t rocprim_size_bytes = 0;
    void* temp_ptr = nullptr;

    HIP_CHECK(rocprim::inclusive_scan(temp_ptr, rocprim_size_bytes, ptrT, ptrT, n,
                                      rocprim::plus<rocblas_int>()));

    *size_work = (rocprim_size_bytes + size_work_LU);

    return rocblas_status_success;
}

template <typename T>
rocblas_status rocsolver_csrrf_splitlu_argCheck(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int nnzT,
                                                rocblas_int* ptrT,
                                                rocblas_int* indT,
                                                T valT,
                                                rocblas_int* ptrL,
                                                rocblas_int* indL,
                                                T valL,
                                                rocblas_int* ptrU,
                                                rocblas_int* indU,
                                                T valU)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    if(handle == nullptr)
    {
        return rocblas_status_invalid_handle;
    }

    // 2. invalid size
    if(n < 0 || nnzT < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!ptrL || !ptrU || !ptrT || (nnzT && (!indT || !valT || !indU || !valU))
       || ((n || nnzT) && (!indL || !valL)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_splitlu_template(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int nnzT,
                                                rocblas_int* ptrT,
                                                rocblas_int* indT,
                                                U valT,
                                                rocblas_int* ptrL,
                                                rocblas_int* indL,
                                                U valL,
                                                rocblas_int* ptrU,
                                                rocblas_int* indU,
                                                U valU,
                                                rocblas_int* work,
                                                size_t size_work)
{
    ROCSOLVER_ENTER("csrrf_splitlu", "n:", n, "nnzT:", nnzT);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    // quick return with matrix zero
    if(nnzT == 0)
    {
        // set ptrU = 0
        const rocblas_int blocks = (n - 1) / BS1 + 1;
        dim3 grid(blocks, 1, 1);
        dim3 threads(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, ptrU, n + 1, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, ptrL, n + 1, 0, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, indL, n, 0, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, valL, n, 1);

        return rocblas_status_success;
    }

    const rocblas_int avg_nnzM = std::max(1, nnzT / n);
    const rocblas_int waveSize = cal_wave_size(avg_nnzM);
    const rocblas_int nx = waveSize;
    const rocblas_int ny = BS1 / nx;

    assert(BS1 == (nx * ny));

    bool const use_splitLU_kernel = (n <= SPLITLU_SWITCH_SIZE);

    if(use_splitLU_kernel)
    {
        // --------------------------------
        // note using a single thread block
        // --------------------------------

        const rocblas_int nblocks = 1;
        ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_kernel<T>, dim3(nblocks, 1, 1), dim3(nx, ny, 1), 0, stream,
                                n, nnzT, ptrT, indT, valT, ptrL, indL, valL, ptrU, indU, valU, work);
    }
    else
    {
        rocblas_int const nblocks = std::max(1, std::min(1024, (n - 1) / BS1 + 1));

        rocblas_int* const Lp = ptrL;
        rocblas_int* const Up = ptrU;

        // ------------------------------------------------
        // setup number of nonzeros in each row of L and U
        // note: reuse arrays Lp[] and Up[]
        // ------------------------------------------------
        HIP_CHECK(hipMemsetAsync(Lp, 0, sizeof(rocblas_int) * (n + 1), stream));
        HIP_CHECK(hipMemsetAsync(Up, 0, sizeof(rocblas_int) * (n + 1), stream));

        ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_gen_nzLU_kernel<T>, dim3(nblocks, 1, 1), dim3(nx, ny, 1),
                                0, stream, n, nnzT, ptrT, indT, Lp + 1, Up + 1);
#ifdef NDEBUG
#else
        {
            // ---------------------------------------------------
            // double check result from rf_splitLU_gen_nzLU_kernel
            // ---------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(check_nzLU_kernel<T>, dim3((n - 1) / BS1 + 1, 1, 1),
                                    dim3(BS1, 1, 1), 0, stream, n, nnzT, ptrT, indT, Lp + 1, Up + 1);
        }
#endif

        // -------------------------------------
        // generate prefix sum for Lp[] and Up[]
        // note: in-place prefix sum
        // -------------------------
        {
            void* temp_ptr = static_cast<void*>(work);
            size_t storage_size_bytes = size_work;

            // ----------------------------------------
            // perform inclusive scan for Lp[] and Up[]
            // ----------------------------------------
            HIP_CHECK(rocprim::inclusive_scan(temp_ptr, storage_size_bytes, (Lp + 1), (Lp + 1), n,
                                              rocprim::plus<rocblas_int>(), stream));

            HIP_CHECK(rocprim::inclusive_scan(temp_ptr, storage_size_bytes, (Up + 1), (Up + 1), n,
                                              rocprim::plus<rocblas_int>(), stream));
        }

        // ------------------------
        // set Lp[0] = 0, Up[0] = 0
        // ------------------------
        {
            const rocblas_int ival = static_cast<rocblas_int>(0);

            HIP_CHECK(hipMemcpyAsync(Lp, &ival, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));

            HIP_CHECK(hipMemcpyAsync(Up, &ival, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
        }

        // -----------------
        // copy into L and U
        // -----------------

        ROCSOLVER_LAUNCH_KERNEL(rf_splitLU_copy_kernel<T>, dim3(nblocks, 1, 1), dim3(nx, ny, 1), 0,
                                stream, n, nnzT, ptrT, indT, valT, Lp, indL, valL, Up, indU, valU);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
