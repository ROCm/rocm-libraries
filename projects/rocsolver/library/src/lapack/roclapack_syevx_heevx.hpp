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

#include "auxiliary/rocauxiliary_ormtr_unmtr.hpp"
#include "auxiliary/rocauxiliary_stebz.hpp"
#include "auxiliary/rocauxiliary_stein.hpp"
#include "rocblas.hpp"
#include "roclapack_sytrd_hetrd.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
__device__ static void syevx_permute_swap(rocblas_int n,
                                          rocblas_int nev,
                                          rocblas_int info,
                                          rocblas_int* map,
                                          T* Z,
                                          rocblas_int ldz,
                                          rocblas_int* ifail)
{
    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const k_start = tid;
    auto const k_inc = nthreads;
    bool const is_root_thread = (tid == 0);

    // ---------------------------------------
    // perform swaps to implement permutation
    // ---------------------------------------
    for(rocblas_int ii = 0; ii < nev; ii++)
    {
        __syncthreads();

        while(map[ii] != ii)
        {
            auto const map_i = map[ii];
            auto const map_ii = map[map[ii]];

            __syncthreads();

            if(is_root_thread)
            {
                map[map_i] = map_i;
                map[ii] = map_ii;
            };

            __syncthreads();

            auto const i = map_i;
            auto const j = map_ii;

            __syncthreads();
            for(int k = k_start; k < n; k += k_inc)
            {
                auto k_i = k + i * ((int64_t)ldz);
                auto k_j = k + j * ((int64_t)ldz);

                auto const ztemp = Z[k_i];
                Z[k_i] = Z[k_j];
                Z[k_j] = ztemp;
            };
            __syncthreads();

            if(ifail)
            {
                __syncthreads();
                for(int k = k_start; k < info; k += k_inc)
                {
                    if(ifail[k] == i + 1)
                        ifail[k] = j + 1;
                    else if(ifail[k] == j + 1)
                        ifail[k] = i + 1;
                }
                __syncthreads();
            }
        }; // end while
    }; // end for

#ifdef NDEBUG
#else
    {
        // ------------------------------------------------------
        // double check map[] is restored to identity permutation
        // ------------------------------------------------------
        __syncthreads();
        for(auto k = k_start; k < nev; k += k_inc)
        {
            assert(map[k] == k);
        };
        __syncthreads();
    };
#endif
}

template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) syevx_sort_eigs(const rocblas_int n,
                                                             rocblas_int* nevA,
                                                             S* WW,
                                                             const rocblas_stride strideW,
                                                             U ZZ,
                                                             const rocblas_int shiftZ,
                                                             const rocblas_int ldz,
                                                             const rocblas_stride strideZ,
                                                             rocblas_int* ifailA,
                                                             const rocblas_stride strideIfail,
                                                             rocblas_int* infoA,
                                                             rocblas_int* isplit_map)
{
    // select batch instance
    auto const bid = hipBlockIdx_y;
    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;
    auto const k_start = tid;
    auto const k_inc = nthreads;

    // local variables
    auto const nev = nevA[bid];
    auto const info = infoA[bid];

    S* W = WW + (bid * strideW);
    T* Z = load_ptr_batch<T>(ZZ, bid, shiftZ, strideZ);
    rocblas_int* ifail = nullptr;
    if(ifailA)
        ifail = ifailA + (bid * strideIfail);

    assert(nev <= n);
    assert(isplit_map != nullptr);

    auto const map = isplit_map + (bid * n);
    bool constexpr use_shell_sort = true;

    __syncthreads();
    if(use_shell_sort)
        shell_sort(nev, W, map);
    else
        selection_sort(nev, W, map);

    __syncthreads();
    syevx_permute_swap(n, nev, info, map, Z, ldz, ifail);
    __syncthreads();
}

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevx_heevx_argCheck(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_erange erange,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              T A,
                                              const rocblas_int lda,
                                              const S vl,
                                              const S vu,
                                              const rocblas_int il,
                                              const rocblas_int iu,
                                              rocblas_int* nev,
                                              S* W,
                                              T Z,
                                              const rocblas_int ldz,
                                              rocblas_int* ifail,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(evect != rocblas_evect_original && evect != rocblas_evect_none)
        return rocblas_status_invalid_value;
    if(erange != rocblas_erange_all && erange != rocblas_erange_value
       && erange != rocblas_erange_index)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || (evect != rocblas_evect_none && ldz < n) || batch_count < 0)
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_value && vl >= vu)
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_index && (il < 1 || iu < 0))
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !W) || (batch_count && !nev) || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if(evect != rocblas_evect_none && ((n && !Z) || (n && !ifail)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevx_heevx_getMemorySize(const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work1,
                                         size_t* size_work2,
                                         size_t* size_work3,
                                         size_t* size_work4,
                                         size_t* size_work5,
                                         size_t* size_work6,
                                         size_t* size_D,
                                         size_t* size_E,
                                         size_t* size_iblock,
                                         size_t* size_isplit_map,
                                         size_t* size_tau,
                                         size_t* size_nsplit_workArr)
{
    // if quick return, set workspace to zero
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_work5 = 0;
        *size_work6 = 0;
        *size_D = 0;
        *size_E = 0;
        *size_iblock = 0;
        *size_isplit_map = 0;
        *size_tau = 0;
        *size_nsplit_workArr = 0;
        return;
    }

    size_t unused;
    size_t a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    size_t c1 = 0, c2 = 0, c3 = 0;

    // requirements for tridiagonalization (sytrd/hetrd)
    rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &a1, &b1, &c1,
                                                    size_nsplit_workArr, false);

    // extra requirements for computing the eigenvalues (stebz)
    rocsolver_stebz_getMemorySize<T>(n, batch_count, &a2, &b2, &c2, size_work4, size_work5,
                                     size_work6);

    if(evect == rocblas_evect_original)
    {
        // extra requirements for ormtr/unmtr
        rocsolver_ormtr_unmtr_getMemorySize<BATCHED, T>(rocblas_side_left, uplo, n, n, batch_count,
                                                        &unused, &a3, &b3, &c3, &unused);

        // extra requirements for computing the eigenvectors (stein)
        rocsolver_stein_getMemorySize<T, S>(n, batch_count, &a4, &b4);
    }

    // get max values
    *size_work1 = std::max({a1, a2, a3, a4});
    *size_work2 = std::max({b1, b2, b3, b4});
    *size_work3 = std::max({c1, c2, c3});

    // size of arrays for temporary tridiagonal elements
    *size_D = sizeof(S) * n * batch_count;
    *size_E = sizeof(S) * n * batch_count;

    // size of arrays for temporary submatrix indices
    *size_iblock = sizeof(rocblas_int) * n * batch_count;
    *size_isplit_map = sizeof(rocblas_int) * n * batch_count;

    // size of array for temporary householder scalars
    *size_tau = sizeof(T) * n * batch_count;

    // size of array for temporary split off block sizes
    *size_nsplit_workArr = std::max(*size_nsplit_workArr, sizeof(rocblas_int) * batch_count);
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_syevx_heevx_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_erange erange,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              const S vl,
                                              const S vu,
                                              const rocblas_int il,
                                              const rocblas_int iu,
                                              const S abstol,
                                              rocblas_int* nev,
                                              S* W,
                                              const rocblas_stride strideW,
                                              U Z,
                                              const rocblas_int shiftZ,
                                              const rocblas_int ldz,
                                              const rocblas_stride strideZ,
                                              rocblas_int* ifail,
                                              const rocblas_stride strideF,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              void* work1,
                                              void* work2,
                                              void* work3,
                                              void* work4,
                                              void* work5,
                                              void* work6,
                                              S* D,
                                              S* E,
                                              rocblas_int* iblock,
                                              rocblas_int* isplit_map,
                                              T* tau,
                                              void* nsplit_workArr)
{
    ROCSOLVER_ENTER("syevx_heevx", "evect:", evect, "erange:", erange, "uplo:", uplo, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "vl:", vl, "vu:", vu, "il:", il, "iu:", iu,
                    "abstol:", abstol, "shiftZ:", shiftZ, "ldz:", ldz, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return with info = 0 and nev = 0
    if(n == 0)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threads(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nev, batch_count, 0);
        return rocblas_status_success;
    }

    // TODO: Scale the matrix

    const rocblas_stride stride = n;

    // reduce A to tridiagonal form
    rocsolver_sytrd_hetrd_template<BATCHED, T>(handle, uplo, n, A, shiftA, lda, strideA, D, stride, E,
                                               stride, tau, stride, batch_count, scalars, (T*)work1,
                                               (T*)work2, (T*)work3, (T**)nsplit_workArr, false);

    // compute eigenvalues
    rocblas_eorder eorder
        = (evect == rocblas_evect_none ? rocblas_eorder_entire : rocblas_eorder_blocks);
    rocsolver_stebz_template<S>(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, 0, stride, E,
                                0, stride, nev, (rocblas_int*)nsplit_workArr, W, strideW, iblock,
                                stride, isplit_map, stride, info, batch_count, (rocblas_int*)work1,
                                (S*)work2, (S*)work3, (S*)work4, (S*)work5, (rocblas_int*)work6);

    if(evect != rocblas_evect_none)
    {
        // compute eigenvectors
        rocsolver_stein_template<T>(handle, n, D, 0, stride, E, 0, stride, nev, W, 0, strideW, iblock,
                                    stride, isplit_map, stride, Z, shiftZ, ldz, strideZ, ifail,
                                    strideF, info, batch_count, (S*)work1, (rocblas_int*)work2);

        // apply unitary matrix to eigenvectors
        rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
        rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
            handle, rocblas_side_left, uplo, rocblas_operation_none, n, h_nev, A, shiftA, lda,
            strideA, tau, stride, Z, shiftZ, ldz, strideZ, batch_count, scalars, (T*)work1,
            (T*)work2, (T*)work3, (T**)nsplit_workArr);

        // sort eigenvalues and eigenvectors
        dim3 grid(1, batch_count, 1);
        dim3 threads(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(syevx_sort_eigs<T>, grid, threads, 0, stream, n, nev, W, strideW, Z,
                                shiftZ, ldz, strideZ, ifail, strideF, info, isplit_map);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
