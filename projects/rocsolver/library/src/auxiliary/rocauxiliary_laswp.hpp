/* ************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "lapack_host_functions.hpp"
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
                                   const I incp_arg,
                                   const rocblas_stride strideP,
                                   const I batch_count)
{
    I const tid_start = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;
    I const tid_inc = hipBlockDim_x * hipGridDim_x;

    I const id_start = hipBlockIdx_y;
    I const id_inc = hipGridDim_y;

    bool const is_backward = (incp_arg < 0);
    I const incp = std::abs(incp_arg);

    for(I id = id_start; id < batch_count; id += id_inc)
    {
        // batch instance
        // shiftP must be used so that ipiv[k1] is the desired first index of ipiv
        I const* const ipiv = ipivA + id * strideP + shiftP;
        T* const A = load_ptr_batch(AA, id, shiftA, stride);

        for(I tid = tid_start; tid < n; tid += tid_inc)
        {
            I const start = (is_backward) ? k2 : k1;
            I const end = (is_backward) ? (k1 - 1) : (k2 + 1);
            I const inc = (is_backward) ? -1 : 1;

            for(I i = start; i != end; i += inc)
            {
                I const exch = ipiv[k1 + (i - k1) * incp - 1];

                // will exchange rows i and exch if they are not the same
                if(exch != i)
                    swap(A[(i - 1) * inca + tid * lda], A[(exch - 1) * inca + tid * lda]);
            }
        }

    } // end for id
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

    I const blocksPivot = (n - 1) / LASWP_THDS + 1;
    I const max_blocks = get_nblocks_yz(handle);
    dim3 gridPivot(blocksPivot, std::min(max_blocks, batch_count), 1);
    dim3 threads(LASWP_THDS, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCSOLVER_LAUNCH_KERNEL(laswp_kernel<T>, gridPivot, threads, 0, stream, n, A, shiftA, inca, lda,
                            strideA, k1, k2, ipiv, shiftP, incp, strideP, batch_count);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
