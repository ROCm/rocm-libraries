/* ************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

#define LASWP_THDS 256 // size of thread-blocks for calling the laswp kernel

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void laswp_kernel(const I n,
                                   U AA,
                                   const rocblas_stride shiftA,
                                   const I inca,
                                   const I lda,
                                   const rocblas_stride stride,
                                   const I k1,
                                   const I k2,
                                   const I* ipivA,
                                   const rocblas_stride shiftP,
                                   I incp,
                                   const rocblas_stride strideP)
{
    I id = hipBlockIdx_y;
    I tid = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;

    if(tid < n)
    {
        // batch instance
        // shiftP must be used so that ipiv[k1] is the desired first index of ipiv
        const I* ipiv = ipivA + id * strideP + shiftP;
        T* A = load_ptr_batch(AA, id, shiftA, stride);

        I start, end, inc;
        if(incp < 0)
        {
            start = k2;
            end = k1 - 1;
            inc = -1;
            incp = -incp;
        }
        else
        {
            start = k1;
            end = k2 + 1;
            inc = 1;
        }

        for(I i = start; i != end; i += inc)
        {
            I exch = ipiv[k1 + (i - k1) * incp - 1];

            // will exchange rows i and exch if they are not the same
            if(exch != i)
                swap(A[(i - 1) * inca + tid * lda], A[(exch - 1) * inca + tid * lda]);
        }
    }
}

template <typename T, typename I>
rocblas_status rocsolver_laswp_argCheck(rocblas_handle handle,
                                        const I n,
                                        const I lda,
                                        const I k1,
                                        const I k2,
                                        T A,
                                        const I* ipiv,
                                        const I incp = 1,
                                        const I inca = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < 1 || k1 < 1 || k2 < 1 || k2 < k1)
        return rocblas_status_invalid_size;
    if(incp == 0 || inca < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || !ipiv)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_laswp_template(rocblas_handle handle,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I inca,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const I k1,
                                        const I k2,
                                        const I* ipiv,
                                        const rocblas_stride shiftP,
                                        const I incp,
                                        const rocblas_stride strideP,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("laswp", "n:", n, "shiftA:", shiftA, "inca:", inca, "lda:", lda, "k1:", k1,
                    "k2:", k2, "shiftP:", shiftP, "incp:", incp, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    I blocksPivot = (n - 1) / LASWP_THDS + 1;
    dim3 gridPivot(blocksPivot, batch_count, 1);
    dim3 threads(LASWP_THDS, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCSOLVER_LAUNCH_KERNEL(laswp_kernel<T>, gridPivot, threads, 0, stream, n, A, shiftA, inca, lda,
                            strideA, k1, k2, ipiv, shiftP, incp, strideP);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
