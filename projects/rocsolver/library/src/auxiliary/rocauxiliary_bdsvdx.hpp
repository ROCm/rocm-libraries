/**************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.10.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lapack/roclapack_syevx_heevx.hpp"
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
ROCSOLVER_KERNEL void bdsvdx_abs_eigs(const rocblas_int n,
                                      rocblas_int* nsvA,
                                      T* SS,
                                      const rocblas_stride strideS,
                                      T* StmpA)
{
    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // local variables
    rocblas_int nsv = nsvA[bid];
    T* S = SS + (bid * strideS);
    T* Stmp = StmpA + (bid * 2 * n);

    if(nsv > n)
        nsvA[bid] = nsv = n;

    if(tid < nsv)
        S[tid] = -Stmp[tid];
}

template <typename T, typename U>
ROCSOLVER_KERNEL void bdsvdx_reorder_vect(const rocblas_fill uplo,
                                          const rocblas_int n,
                                          rocblas_int* nsvA,
                                          T* SS,
                                          const rocblas_stride strideS,
                                          U ZZ,
                                          const rocblas_int shiftZ,
                                          const rocblas_int ldz,
                                          const rocblas_stride strideZ,
                                          T* workA)
{
    using W = decltype(std::real(T{}));

    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    // local variables
    rocblas_int i, j;
    rocblas_int nsv = nsvA[bid];
    T* work = workA + (bid * 2 * n);
    T* S = SS + (bid * strideS);
    T* Z = load_ptr_batch<T>(ZZ, bid, shiftZ, strideZ);

    if(nsv > n)
    {
        nsv = n;
        if(tid == 0)
            nsvA[bid] = n;
    }

    for(i = tid; i < nsv; i += hipBlockDim_x)
        S[i] = -work[i];
    __syncthreads();

    const W scl = W(sqrt(2.0));
    for(j = 0; j < nsv; j++)
    {
        for(i = tid; i < 2 * n; i += hipBlockDim_x)
            work[i] = Z[i + j * ldz];
        __syncthreads();

        if(uplo == rocblas_fill_upper)
        {
            for(i = tid; i < n; i += hipBlockDim_x)
            {
                Z[i + j * ldz] = work[2 * i + 1] * scl;
                Z[(n + i) + j * ldz] = -work[2 * i] * scl;
            }
        }
        else
        {
            for(i = tid; i < n; i += hipBlockDim_x)
            {
                Z[i + j * ldz] = work[2 * i] * scl;
                Z[(n + i) + j * ldz] = -work[2 * i + 1] * scl;
            }
        }
        __syncthreads();
    }
}

// Helper to calculate workspace size requirements
template <typename T>
void rocsolver_bdsvdx_getMemorySize(const rocblas_int n,
                                    const rocblas_int batch_count,
                                    size_t* size_work1_iwork,
                                    size_t* size_work2_pivmin,
                                    size_t* size_Esqr,
                                    size_t* size_bounds,
                                    size_t* size_inter,
                                    size_t* size_ninter,
                                    size_t* size_nsplit,
                                    size_t* size_iblock,
                                    size_t* size_isplit_map,
                                    size_t* size_Dtgk,
                                    size_t* size_Etgk,
                                    size_t* size_Stmp)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_work1_iwork = 0;
        *size_work2_pivmin = 0;
        *size_Esqr = 0;
        *size_bounds = 0;
        *size_inter = 0;
        *size_ninter = 0;
        *size_nsplit = 0;
        *size_iblock = 0;
        *size_isplit_map = 0;
        *size_Dtgk = 0;
        *size_Etgk = 0;
        *size_Stmp = 0;
        return;
    }

    size_t a1, b1;

    // extra requirements for computing the eigenvalues (stebz)
    rocsolver_stebz_getMemorySize<T>(2 * n, batch_count, size_work1_iwork, size_work2_pivmin,
                                     size_Esqr, size_bounds, size_inter, size_ninter);

    // extra requirements for computing the eigenvectors (stein)
    rocsolver_stein_getMemorySize<T, T>(2 * n, batch_count, &b1, &a1);

    *size_work1_iwork = std::max(*size_work1_iwork, a1);
    *size_work2_pivmin = std::max(*size_work2_pivmin, b1);

    // size of arrays for temporary submatrix indices
    *size_nsplit = sizeof(rocblas_int) * batch_count;
    *size_iblock = sizeof(rocblas_int) * 2 * n * batch_count;
    *size_isplit_map = sizeof(rocblas_int) * 2 * n * batch_count;

    // size of arrays for temporary tridiagonal matrix
    *size_Dtgk = sizeof(T) * 2 * n * batch_count;
    *size_Etgk = sizeof(T) * 2 * n * batch_count;

    // size of array for temporary singular values
    *size_Stmp = sizeof(T) * 2 * n * batch_count;
}

// Helper to check argument correctnesss
template <typename T, typename U>
rocblas_status rocsolver_bdsvdx_argCheck(rocblas_handle handle,
                                         const rocblas_fill uplo,
                                         const rocblas_svect svect,
                                         const rocblas_srange srange,
                                         const rocblas_int n,
                                         T* D,
                                         T* E,
                                         const T vl,
                                         const T vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         rocblas_int* nsv,
                                         T* S,
                                         U Z,
                                         const rocblas_int ldz,
                                         rocblas_int* ifail,
                                         rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    if(svect != rocblas_svect_none && svect != rocblas_svect_singular)
        return rocblas_status_invalid_value;
    if(srange != rocblas_srange_all && srange != rocblas_srange_value
       && srange != rocblas_srange_index)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if((svect == rocblas_svect_none && ldz < 1) || (svect != rocblas_svect_none && ldz < 2 * n))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_value && (vl < 0 || vl >= vu))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_index && (iu > n || (n > 0 && il > iu)))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_index && (il < 1 || iu < 0))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && (!D || !S)) || (n > 1 && !E) || !info || !nsv)
        return rocblas_status_invalid_pointer;
    if(svect != rocblas_svect_none && n && (!Z || !ifail))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// bdsvdx template function implementation
template <typename T, typename U>
rocblas_status rocsolver_bdsvdx_template(rocblas_handle handle,
                                         const rocblas_fill uplo,
                                         const rocblas_svect svect,
                                         const rocblas_srange srange,
                                         const rocblas_int n,
                                         T* D,
                                         const rocblas_stride strideD,
                                         T* E,
                                         const rocblas_stride strideE,
                                         const T vl,
                                         const T vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         rocblas_int* nsv,
                                         T* S,
                                         const rocblas_stride strideS,
                                         U Z,
                                         const rocblas_int shiftZ,
                                         const rocblas_int ldz,
                                         const rocblas_stride strideZ,
                                         rocblas_int* ifail,
                                         const rocblas_stride strideF,
                                         rocblas_int* info,
                                         const rocblas_int batch_count,
                                         rocblas_int* work1_iwork,
                                         T* work2_pivmin,
                                         T* Esqr,
                                         T* bounds,
                                         T* inter,
                                         rocblas_int* ninter,
                                         rocblas_int* nsplit,
                                         rocblas_int* iblock,
                                         rocblas_int* isplit_map,
                                         T* Dtgk,
                                         T* Etgk,
                                         T* Stmp)
{
    ROCSOLVER_ENTER("bdsvdx", "uplo:", uplo, "svect:", svect, "srange:", srange, "n:", n, "vl:", vl,
                    "vu:", vu, "il:", il, "iu:", iu, "ldz:", ldz, "shiftZ:", shiftZ,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // set info = 0
    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocksReset, 1, 1), dim3(BS1, 1, 1), 0, stream, info,
                            batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // zero out diagonal of tridiagonal matrix (Dtgk)
    rocblas_int blocksZero = (2 * n * batch_count - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocksZero, 1, 1), dim3(BS1, 1, 1), 0, stream, Dtgk,
                            2 * n * batch_count, 0);

    // populate off-diagonal of tridiagonal matrix (Etgk) by interleaving entries of D and E
    rocblas_int blocksCopy = (n - 1) / BS1 + 1;
    dim3 gridCopy(1, blocksCopy, batch_count);
    dim3 threadsCopy(1, BS1);

    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, T*>), gridCopy, threadsCopy, 0, stream, 1, n, D, 0, 1,
                            strideD, Etgk, 0, 2, 2 * n);
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, T*>), gridCopy, threadsCopy, 0, stream, 1, n - 1, E, 0, 1,
                            strideE, Etgk, 1, 2, 2 * n);

    rocblas_int ntgk = 2 * n;
    rocblas_erange range
        = (srange == rocblas_srange_value ? rocblas_erange_value : rocblas_erange_index);
    rocblas_eorder order
        = (svect == rocblas_svect_none ? rocblas_eorder_entire : rocblas_eorder_blocks);
    T vltgk = (srange == rocblas_srange_value ? -vu : 0);
    T vutgk = (srange == rocblas_srange_value ? -vl : 0);
    rocblas_int iltgk = (srange == rocblas_srange_index ? il : 1);
    rocblas_int iutgk = (srange == rocblas_srange_index ? iu : n);

    // compute eigenvalues of tridiagonal matrix
    rocsolver_stebz_template<T>(handle, range, order, ntgk, vltgk, vutgk, iltgk, iutgk, 0, Dtgk, 0,
                                ntgk, Etgk, 0, ntgk, nsv, nsplit, Stmp, ntgk, iblock, ntgk,
                                isplit_map, ntgk, info, batch_count, work1_iwork, work2_pivmin,
                                Esqr, bounds, inter, ninter);

    if(svect == rocblas_svect_none)
    {
        // take absolute value of eigenvalues
        ROCSOLVER_LAUNCH_KERNEL(bdsvdx_abs_eigs<T>, dim3(blocksCopy, batch_count, 1),
                                dim3(BS1, 1, 1), 0, stream, n, nsv, S, strideS, Stmp);
    }
    else
    {
        // compute eigenvectors of tridiagonal matrix
        rocsolver_stein_template<T>(handle, ntgk, Dtgk, 0, ntgk, Etgk, 0, ntgk, nsv, Stmp, 0, ntgk,
                                    iblock, ntgk, isplit_map, ntgk, Z, shiftZ, ldz, strideZ, ifail,
                                    strideF, info, batch_count, work2_pivmin, work1_iwork);

        // sort eigenvalues and vectors
        ROCSOLVER_LAUNCH_KERNEL(syevx_sort_eigs<T>, dim3(1, batch_count, 1), dim3(BS1, 1, 1), 0,
                                stream, ntgk, nsv, Stmp, ntgk, Z, shiftZ, ldz, strideZ, ifail,
                                strideF, info, isplit_map);

        // take absolute value of eigenvalues, reorder and normalize eigenvector elements, and negate elements of V
        ROCSOLVER_LAUNCH_KERNEL(bdsvdx_reorder_vect<T>, dim3(1, batch_count, 1), dim3(BS1, 1, 1), 0,
                                stream, uplo, n, nsv, S, strideS, Z, shiftZ, ldz, strideZ, Stmp);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
