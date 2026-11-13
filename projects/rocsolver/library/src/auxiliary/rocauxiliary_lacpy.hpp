/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
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

#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Copies the upper (uplo = upper), lower (uplo = lower) or full part of the m-by-n matrix A
    into B, for each matrix of the batch. One thread per entry, with no shared memory.
    Launch as dim3(blocksx, blocksy, blocksz), dim3(BS2, BS2): any grid works, since the
    threads loop over the rows and columns that the grid does not cover, and the blocks loop
    over the batch (blockIdx.z, blockIdx.z + gridDim.z, ...). **/
template <typename T, typename I, typename U1, typename U2>
ROCSOLVER_KERNEL void lacpy_kernel(const rocblas_fill uplo,
                                   const I m,
                                   const I n,
                                   U1 A,
                                   const rocblas_stride shiftA,
                                   const I lda,
                                   const rocblas_stride strideA,
                                   U2 B,
                                   const rocblas_stride shiftB,
                                   const I ldb,
                                   const rocblas_stride strideB,
                                   const I batch_count)
{
    // first row and column of this thread, and the strides of the loops over the rows and the
    // columns (the indices are 64-bit, so that the loops cannot overflow I near its end)
    const int64_t i_start = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    const int64_t j_start = blockIdx.y * int64_t(blockDim.y) + threadIdx.y;
    const int64_t i_inc = gridDim.x * int64_t(blockDim.x);
    const int64_t j_inc = gridDim.y * int64_t(blockDim.y);

    for(I bid = blockIdx.z; bid < batch_count; bid += gridDim.z)
    {
        const T* const a = load_ptr_batch<T>(A, bid, shiftA, strideA);
        T* const bb = load_ptr_batch<T>(B, bid, shiftB, strideB);

        for(int64_t j = j_start; j < n; j += j_inc)
        {
            // rows of column j in the copied part
            const int64_t lo = (uplo == rocblas_fill_lower) ? j : 0;
            const int64_t hi = (uplo == rocblas_fill_upper) ? std::min<int64_t>(j + 1, m) : m;
            for(int64_t i = lo + i_start; i < hi; i += i_inc)
                bb[idx2D(i, j, ldb)] = a[idx2D(i, j, lda)];
        }
    }
}

template <typename T, typename I>
rocblas_status rocsolver_lacpy_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I m,
                                        const I n,
                                        const I lda,
                                        const I ldb,
                                        T A,
                                        T B)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower && uplo != rocblas_fill_full)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || ldb < m)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(m && n && (!A || !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename U1, typename U2>
rocblas_status rocsolver_lacpy_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I m,
                                        const I n,
                                        U1 A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        U2 B,
                                        const rocblas_stride shiftB,
                                        const I ldb,
                                        const rocblas_stride strideB,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("lacpy", "uplo:", uplo, "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // (at most LACPY_MAX_BLOCKSX blocks in x, which is enough to fill the device, at most 64K - 1
    // in y and z, and at most INT32_MAX threads in all, the limits of ROCSOLVER_LAUNCH_KERNEL: the
    // kernel loops over the rest of the rows, of the columns and of the batch)
    static constexpr int64_t LACPY_MAX_BLOCKSX = 1024;
    static constexpr int64_t LACPY_MAX_BLOCKSYZ = 64 * 1024 - 1;
    static constexpr int64_t LACPY_MAX_BLOCKS = INT32_MAX / (BS2 * BS2);
    const int64_t blocksx = std::min<int64_t>(ceildiv(m, BS2), LACPY_MAX_BLOCKSX);
    const int64_t blocksy = std::max<int64_t>(
        1, std::min<int64_t>({ceildiv(n, BS2), LACPY_MAX_BLOCKSYZ, LACPY_MAX_BLOCKS / blocksx}));
    const int64_t blocksz = std::max<int64_t>(
        1,
        std::min<int64_t>({batch_count, LACPY_MAX_BLOCKSYZ, LACPY_MAX_BLOCKS / (blocksx * blocksy)}));
    assert(blocksx * blocksy * blocksz <= LACPY_MAX_BLOCKS);
    ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T>), dim3(blocksx, blocksy, blocksz), dim3(BS2, BS2, 1),
                            0, stream, uplo, m, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
                            batch_count);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
