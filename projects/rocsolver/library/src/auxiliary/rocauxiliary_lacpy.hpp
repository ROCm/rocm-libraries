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
    for(I bid = blockIdx.z; bid < batch_count; bid += gridDim.z)
    {
        const T* const a = load_ptr_batch<T>(A, bid, shiftA, strideA);
        T* const bb = load_ptr_batch<T>(B, bid, shiftB, strideB);

        // (the indices are 64-bit, so that the strided loops cannot overflow I near its end)
        for(int64_t j = blockIdx.y * int64_t(blockDim.y) + threadIdx.y; j < n;
            j += int64_t(gridDim.y) * blockDim.y)
        {
            // rows of column j in the copied part
            const int64_t lo = (uplo == rocblas_fill_lower) ? j : 0;
            const int64_t hi = (uplo == rocblas_fill_upper) ? std::min<int64_t>(j + 1, m) : m;
            for(int64_t i = lo + blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < hi;
                i += int64_t(gridDim.x) * blockDim.x)
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

    // (at most LACPY_MAX_BLOCKSX blocks in x, which is enough to fill the device, and at most
    // 64K - 1 in y and z, the limit of ROCSOLVER_LAUNCH_KERNEL: the kernel loops over the rest of
    // the rows, of the columns and of the batch)
    static constexpr int LACPY_MAX_BLOCKSX = 1024;
    static constexpr int LACPY_MAX_BLOCKSYZ = 64 * 1024 - 1;
    const I blocksx = std::min(ceildiv(m, BS2), I(LACPY_MAX_BLOCKSX));
    const I blocksy = std::min(ceildiv(n, BS2), I(LACPY_MAX_BLOCKSYZ));
    const I blocksz = std::min(batch_count, I(LACPY_MAX_BLOCKSYZ));
    ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T>), dim3(blocksx, blocksy, blocksz), dim3(BS2, BS2, 1),
                            0, stream, uplo, m, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
                            batch_count);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
