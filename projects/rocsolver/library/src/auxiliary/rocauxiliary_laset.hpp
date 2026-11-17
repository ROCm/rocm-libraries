/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2013
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
// Initialize matrix
// motivated by xLASET in LAPACK
//
// matrix A is m by n
//
// uplo == rocblas_fill_upper : assign to upper triangular matrix
// uplo == rocblas_fill_lower : assign to lower triangular matrix
// uplo == rocblas_fill_full : assign to entire matrix
//
// Assign offdiag to off-diagonal elements.
// Assign diag to diagonal elements.
//
// Thread block is (dimX, dimY), which can be arbitrary.
// Grid is (ceil( m / dimX ), ceil( n / dimY ), batch_count).
//
template <typename T, typename I, typename UA>
ROCSOLVER_KERNEL void laset_kernel(const rocblas_fill uplo,
                                   const I m,
                                   const I n,
                                   const T offdiag,
                                   const T diag,
                                   UA AA,
                                   const rocblas_stride shiftA,
                                   const I lda,
                                   const rocblas_stride strideA,
                                   const I batch_count)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;
    I const j_inc = blockDim.y * gridDim.y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A = load_ptr_batch<T>(AA, bid, shiftA, strideA);

        if(uplo == rocblas_fill_lower)
        {
            // ---------------------------------
            // assign to lower triangular matrix
            // ---------------------------------
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = j + i_start; i < m; i += i_inc)
                {
                    A[idx2D(i, j, lda)] = (i == j) ? diag : offdiag;
                }
            }
        }
        else if(uplo == rocblas_fill_upper)
        {
            // ---------------------------------
            // assign to upper triangular matrix
            // ---------------------------------
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < std::min(m, j + 1); i += i_inc)
                {
                    A[idx2D(i, j, lda)] = (i == j) ? diag : offdiag;
                }
            }
        }
        else
        {
            // ------------------------
            // assign to entire matrix
            // ------------------------
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < m; i += i_inc)
                {
                    A[idx2D(i, j, lda)] = (i == j) ? diag : offdiag;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename T, typename I>
rocblas_status rocsolver_laset_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I m,
                                        const I n,
                                        const I lda,
                                        T A,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower && uplo != rocblas_fill_full)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < std::max(I(1), m) || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(m && n && batch_count && !A)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
template <typename T, typename I, typename UA>
rocblas_status rocsolver_laset_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I m,
                                        const I n,
                                        const T offdiag,
                                        const T diag,
                                        UA A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("laset", "uplo:", uplo, "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    const int nx = std::min(ceildiv(m, BS2), I(props->maxGridSize[0]));
    const int ny = std::min(ceildiv(n, BS2), I(props->maxGridSize[1]));
    const int nz = std::min(batch_count, I(props->maxGridSize[2]));
    dim3 blocks(nx, ny, nz);
    dim3 threads(BS2, BS2);
    ROCSOLVER_LAUNCH_KERNEL(laset_kernel<T>, blocks, threads, 0, stream, // kernel
                            uplo, m, n, offdiag, diag, // opts
                            A, shiftA, lda, strideA, // A
                            batch_count);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
