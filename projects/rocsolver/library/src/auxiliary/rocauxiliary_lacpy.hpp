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

/** Copies the upper (uplo = upper), lower (uplo = lower) or full part of A_b into B_b.
    Grid (ceil(m / BS2), ceil(n / BS2), batch_count) (or fewer blocks), block (BS2, BS2). **/
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
                                   const rocblas_stride strideB)
{
    const I b = blockIdx.z;
    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* bb = load_ptr_batch<T>(B, b, shiftB, strideB);

    for(I j = blockIdx.y * I(blockDim.y) + threadIdx.y; j < n; j += I(gridDim.y) * blockDim.y)
    {
        // rows of column j in the copied part
        const I lo = (uplo == rocblas_fill_lower) ? j : 0;
        const I hi = (uplo == rocblas_fill_upper) ? std::min(j + 1, m) : m;
        for(I i = lo + blockIdx.x * I(blockDim.x) + threadIdx.x; i < hi;
            i += I(gridDim.x) * blockDim.x)
            bb[idx2D(i, j, ldb)] = a[idx2D(i, j, lda)];
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

    const I blocksx = I(std::min<int64_t>((int64_t(m) - 1) / BS2 + 1, 1024));
    const I blocksy = I(std::min<int64_t>((int64_t(n) - 1) / BS2 + 1, 65535));
    ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T>), dim3(blocksx, blocksy, batch_count), dim3(BS2, BS2, 1),
                            0, stream, uplo, m, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
