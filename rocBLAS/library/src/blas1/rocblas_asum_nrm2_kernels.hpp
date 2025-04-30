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

#include "../blas1/rocblas_reduction.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_asum_nrm2.hpp"

/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

// BLAS Level 1 includes routines and functions performing vector-vector
// operations. Most BLAS 1 routines are about reduction: compute the norm,
// calculate the dot production of two vectors, find the maximum/minimum index
// of the element of the vector. As you may observed, although the computation
// type is different, the core algorithm is the same: scan all element of the
// vector(s) and reduce to one single result.
//
// The primary reduction algorithm now uses shuffle instructions that use a binary
// tree like reduction of masked out channels, but follows almost the
// same pattern as the recursive reduction algorithm on GPU that is called [parallel
// reduction](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce3.png)
// which is also adopted in rocBLAS. At the beginning, all the threads in the thread
// block participate. After each step of reduction (like a tree), the number of
// participating threads decrease by half. At the end of the parallel reduction,
// only one thread (usually thread 0) owns the result in its thread block.
//
// Classically, the BLAS 1 reduction needs more than one GPU kernel to finish,
// because the lack of global synchronization of thread blocks without exiting
// the kernel. The first kernels gather partial results, write into a temporary
// working buffer. The second kernel finishes the final reduction.
//
// For example, BLAS 1 routine i*amax is to find index of the maximum absolute
// value element of a vector. In this routine:
//
// Kernel 1: launch many thread block as needed. Each thread block works on a
// subset of the vector. Each thread block use the parallel reduction to find a
// local index with the maximum absolute value of the subset. There are
// number-of-the-thread-blocks local results.The results are written into a
// temporary working buffer. The working buffer has number-of-the-thread-blocks
// elements.
//
// Kernel 2: launch only one thread block which reads the temporary work buffer and
// reduces to final result still with the parallel reduction.
//
// As you may see, if there is a mechanism to synchronize all the thread blocks
// after local index is obtained in kernel 1 (without ending the kernel), then
// Kernel 2's computation can be merged into Kernel 1. One such mechanism is called
// atomic operation. However, atomic operation is new and is not used in rocBLAS
// yet. rocBLAS still use the classic standard parallel reduction right now.

// kernel 1 writes partial results per thread block in workspace; number of partial results is
// blocks
template <typename API_INT, int NB, int WIN, typename FETCH, typename TPtrX, typename To>
ROCBLAS_KERNEL(NB)
rocblas_reduction_kernel_part1(rocblas_int    n,
                               rocblas_int    nblocks,
                               TPtrX          xvec,
                               rocblas_stride shiftx,
                               API_INT        incx,
                               rocblas_stride stridex,
                               rocblas_int    batch_count,
                               To*            workspace)
{
    int64_t tid = blockIdx.x * NB + threadIdx.x;
    To      sum = 0;

    uint32_t batch = blockIdx.z;
#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        const auto* x = load_ptr_batch(xvec, batch, shiftx, stridex);

        // sum WIN elements per thread
        int inc = NB * gridDim.x;
        for(int j = 0; j < WIN && tid < n; j++, tid += inc)
            sum += FETCH{}(x[tid * incx]);

        if(warpSize == WARP_32)
            sum = rocblas_dot_block_reduce<WARP_32, NB, To>(sum); // sum reduction only
        else
            sum = rocblas_dot_block_reduce<WARP_64, NB, To>(sum);

        if(threadIdx.x == 0)
            workspace[batch * nblocks + blockIdx.x] = sum;
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/*! \brief

    \details
    rocblas_reduction_strided_batched computes a reduction over multiple vectors x_i
              Template parameters allow threads per block, data, and specific phase kernel overrides
              At least two kernels are needed to finish the reduction
              kernel 1 write partial result per thread block in workspace, blocks partial results
              kernel 2 gathers all the partial result in workspace and finishes the final reduction.
    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    n         rocblas_int
              number of elements in each vector x_i
    @param[in]
    x         pointer to the first vector x_i on the GPU.
    @param[in]
    shiftx    rocblas_int
              specifies a base offset increment for the start of each x_i.
    @param[in]
    incx      rocblas_int
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex   rocblas_int
              specifies the pointer increment between batches for x. stridex must be >= n*incx
    @param[in]
    batch_count rocblas_int
              number of instances in the batch
    @param[out]
    workspace To*
              temporary GPU buffer for inidividual block results for each batch
              and results buffer in case result pointer is to host memory
              Size must be (blocks+1)*batch_count*sizeof(To)
    @param[out]
    result
              pointers to array of batch_count size for results. either on the host CPU or device GPU.
              return is 0.0 if n, incx<=0.
    ********************************************************************/
template <typename API_INT,
          int NB,
          typename FETCH,
          typename FINALIZE,
          typename TPtrX,
          typename To,
          typename Tr>
rocblas_status rocblas_internal_asum_nrm2_launcher(rocblas_handle handle,
                                                   API_INT        n,
                                                   TPtrX          x,
                                                   rocblas_stride shiftx,
                                                   API_INT        incx,
                                                   rocblas_stride stridex,
                                                   API_INT        batch_count,
                                                   To*            workspace,
                                                   Tr*            result)
{
    // param REDUCE is always SUM for these kernels so not passed on

    static constexpr int WIN = rocblas_dot_WIN<To>();

    rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB * WIN);

    int batches = handle->getBatchGridDim((int)batch_count);

    ROCBLAS_LAUNCH_KERNEL((rocblas_reduction_kernel_part1<API_INT, NB, WIN, FETCH>),
                          dim3(blocks, 1, batches),
                          dim3(NB),
                          0,
                          handle->get_stream(),
                          n,
                          blocks,
                          x,
                          shiftx,
                          incx,
                          stridex,
                          batch_count,
                          workspace);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_reduction_kernel_part2<NB, WIN, FINALIZE>),
                              dim3(batch_count),
                              dim3(NB),
                              0,
                              handle->get_stream(),
                              blocks,
                              workspace,
                              result);
    }
    else
    {
        // If in host pointer mode, workspace is converted to Tr* and the result is
        // placed there, and then copied from device to host. If To is a class type,
        // it must be a standard layout type and its first member must be of type Tr.
        static_assert(std::is_standard_layout<To>{}, "To must be a standard layout type");

        bool reduceKernel = blocks > 1 || batch_count > 1;
        if(reduceKernel)
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_reduction_kernel_part2<NB, WIN, FINALIZE>),
                                  dim3(batch_count),
                                  dim3(NB),
                                  0,
                                  handle->get_stream(),
                                  blocks,
                                  workspace,
                                  (Tr*)(workspace + size_t(batch_count) * blocks));
        }

        if(std::is_same_v<FINALIZE, rocblas_finalize_identity> || reduceKernel)
        {
            // If FINALIZE is trivial or kernel part2 was called, result is in the
            // beginning of workspace[0]+offset, and can be copied directly.
            size_t offset = reduceKernel ? size_t(batch_count) * blocks : 0;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(result,
                                               workspace + offset,
                                               batch_count * sizeof(Tr),
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        }
        else
        {
            // If FINALIZE is not trivial and kernel part2 was not called, then
            // workspace[0] needs to be finalized on host.
            auto res = std::make_unique<To[]>(batch_count);
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&res[0],
                                               workspace,
                                               batch_count * sizeof(To),
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
            for(rocblas_int i = 0; i < batch_count; i++)
                result[i] = Tr(FINALIZE{}(res[i]));
        }
    }

    return rocblas_status_success;
}
