/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "rocblas_block_sizes.h"

#include "blas1/reduction.hpp"
#include "blas1/rocblas_iamax_iamin_kernels.hpp"
#include "blas1/rocblas_reduction.hpp"

#include "rocblas_iamax_iamin_64.hpp" // rocblas_int API called

// iamax, iamin kernels

template <int WARP, typename REDUCE, typename T>
ROCBLAS_KERNEL_ILF rocblas_index_64_value_t<T>
                   rocblas_wavefront_reduce_method_64(rocblas_index_64_value_t<T> x)
{
    constexpr int WFBITS = rocblas_log2ui(WARP);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        rocblas_index_64_value_t<T> y{};
        y.index = __shfl_down(x.index, offset);
        y.value = __shfl_down(x.value, offset);
        REDUCE{}(x, y);
        offset >>= 1;
    }
    return x;
}

template <int WARP, int DIM_X, typename REDUCE, typename T>
ROCBLAS_KERNEL_ILF T rocblas_shuffle_block_reduce_method_64(T val)
{
    __shared__ T psums[WARP];

    rocblas_int wavefront = threadIdx.x / WARP;
    rocblas_int wavelet   = threadIdx.x % WARP;

    if(wavefront == 0)
        psums[wavelet] = T{};
    __syncthreads();

    val = rocblas_wavefront_reduce_method_64<WARP, REDUCE>(val); // sum over wavefront
    if(wavelet == 0)
        psums[wavefront] = val; // store sum for wavefront

    __syncthreads(); // Wait for all wavefront reductions

    // ensure wavefront was run
    static constexpr rocblas_int NUM_WARPS = DIM_X / WARP;
    val                                    = (threadIdx.x < NUM_WARPS) ? psums[wavelet] : T{};
    if(wavefront == 0)
        val = rocblas_wavefront_reduce_method_64<NUM_WARPS, REDUCE>(val); // sum wavefront sums

    return val;
}

// kernel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <int DIM_X, typename FETCH, typename REDUCE, typename TPtrX, typename To>
ROCBLAS_KERNEL(DIM_X)
rocblas_iamax_iamin_kernel_part1_64(int64_t        n,
                                    TPtrX          xvec,
                                    rocblas_stride shiftx,
                                    int64_t        incx,
                                    rocblas_stride stridex,
                                    To*            workspace)
{
    int64_t tid = blockIdx.x * DIM_X + threadIdx.x;

    const auto* x = load_ptr_batch(xvec, blockIdx.z, shiftx, stridex);

    To winner = rocblas_default_value<To>{}(); // pad with default value
    for(int64_t offset = 0; offset < n; offset += DIM_X * gridDim.x)
    {
        int64_t i = tid + offset;

        To sum;

        // bound
        if(i < n)
            sum = FETCH{}(x[i * incx], i + 1); // 1 based indexing
        else
            sum = rocblas_default_value<To>{}(); // pad with default value

        if(warpSize == WARP_32)
            sum = rocblas_shuffle_block_reduce_method_64<WARP_32, DIM_X, REDUCE>(sum);
        else
            sum = rocblas_shuffle_block_reduce_method_64<WARP_64, DIM_X, REDUCE>(sum);

        if(threadIdx.x == 0)
            REDUCE{}(winner, sum);
    }
    if(threadIdx.x == 0)
        workspace[size_t(blockIdx.z) * gridDim.x + blockIdx.x] = winner;
}

// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (DIM_X) loop blocks
template <int DIM_X, typename REDUCE, typename To, typename Tr>
ROCBLAS_KERNEL(DIM_X)
rocblas_iamax_iamin_kernel_part2_64(int nblocks, To* workspace, Tr* result)
{
    int tx = threadIdx.x;

    To winner;

    if(tx < nblocks)
    {
        To* work = workspace + size_t(nblocks) * blockIdx.x;
        winner   = work[tx];

        // bound, loop
        for(int i = tx + DIM_X; i < nblocks; i += DIM_X)
            REDUCE{}(winner, work[i]);
    }
    else
    { // pad with default value
        winner = rocblas_default_value<To>{}();
    }

    if(warpSize == WARP_32)
        winner = rocblas_shuffle_block_reduce_method_64<WARP_32, DIM_X, REDUCE>(winner);
    else
        winner = rocblas_shuffle_block_reduce_method_64<WARP_64, DIM_X, REDUCE>(winner);

    // Store result on device or in workspace
    if(tx == 0)
        result[blockIdx.x] = winner.index;
}
