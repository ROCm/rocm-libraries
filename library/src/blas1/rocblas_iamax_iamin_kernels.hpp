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

#include "device_macros.hpp"
#include "reduction.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_iamax_iamin.hpp"
#include "rocblas_reduction.hpp"

// iamax, iamin kernels

template <int WARP, typename REDUCE, typename T>
__inline__ __device__ rocblas_index_value_t<T>
                      rocblas_wavefront_reduce_method(rocblas_index_value_t<T> x)
{
    constexpr int WFBITS = rocblas_log2ui(WARP);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        rocblas_index_value_t<T> y{};
        y.index = __shfl_down(x.index, offset);
        y.value = __shfl_down(x.value, offset);
        REDUCE{}(x, y);
        offset >>= 1;
    }
    return x;
}

template <int WARP, int NB, typename REDUCE, typename T>
__inline__ __device__ T rocblas_shuffle_block_reduce_method(T val)
{
    __shared__ T psums[WARP];

    rocblas_int wavefront = threadIdx.x / WARP;
    rocblas_int wavelet   = threadIdx.x % WARP;

    if(wavefront == 0)
        psums[wavelet] = T{};
    __syncthreads();

    val = rocblas_wavefront_reduce_method<WARP, REDUCE>(val); // sum over wavefront
    if(wavelet == 0)
        psums[wavefront] = val; // store sum for wavefront

    __syncthreads(); // Wait for all wavefront reductions

    // ensure wavefront was run
    static constexpr rocblas_int NUM_WARPS = NB / WARP;
    val                                    = (threadIdx.x < NUM_WARPS) ? psums[wavelet] : T{};
    if(wavefront == 0)
        val = rocblas_wavefront_reduce_method<NUM_WARPS, REDUCE>(val); // sum wavefront sums

    return val;
}

// kernel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <int NB, typename FETCH, typename REDUCE, typename TPtrX, typename To>
ROCBLAS_KERNEL(NB)
rocblas_iamax_iamin_kernel_part1(rocblas_int    n,
                                 rocblas_int    nblocks,
                                 TPtrX          xvec,
                                 rocblas_stride shiftx,
                                 rocblas_int    incx,
                                 rocblas_stride stridex,
                                 rocblas_int    batch_count,
                                 To*            workspace)
{
    int64_t tid = blockIdx.x * NB + threadIdx.x;
    To      sum;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto* x = load_ptr_batch(xvec, batch, shiftx, stridex);

        // bound
        if(tid < n)
            sum = FETCH{}(x[tid * incx], tid + 1); // 1-based indexing
        else
            sum = rocblas_default_value<To>{}(); // pad with default value

        if(warpSize == WARP_32)
            sum = rocblas_shuffle_block_reduce_method<WARP_32, NB, REDUCE>(sum);
        else
            sum = rocblas_shuffle_block_reduce_method<WARP_64, NB, REDUCE>(sum);

        if(threadIdx.x == 0)
            workspace[batch * nblocks + blockIdx.x] = sum;

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

// kernel 2 gathers all the partial results in workspace and finishes the final reduction;
// number of threads (NB) loop blocks
template <int NB, typename REDUCE, typename To, typename Tr>
ROCBLAS_KERNEL(NB)
rocblas_iamax_iamin_kernel_part2(rocblas_int nblocks, To* workspace, Tr* result)
{
    rocblas_int tx = threadIdx.x;
    To          sum;

    if(tx < nblocks)
    {
        To* work = workspace + blockIdx.x * nblocks;
        sum      = work[tx];

        // bound, loop
        for(rocblas_int i = tx + NB; i < nblocks; i += NB)
            REDUCE{}(sum, work[i]);
    }
    else
    { // pad with default value
        sum = rocblas_default_value<To>{}();
    }

    if(warpSize == WARP_32)
        sum = rocblas_shuffle_block_reduce_method<WARP_32, NB, REDUCE>(sum);
    else
        sum = rocblas_shuffle_block_reduce_method<WARP_64, NB, REDUCE>(sum);

    // Store result on device or in workspace
    if(tx == 0)
        result[blockIdx.x] = sum.index;
}
