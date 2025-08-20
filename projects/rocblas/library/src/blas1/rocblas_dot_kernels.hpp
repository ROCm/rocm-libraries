/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblas_block_sizes.h"
#include "rocblas_dot.hpp"
#include "rocblas_level1_threshold.hpp"

#include <cassert>

template <typename T>
constexpr int rocblas_dot_one_block_threshold()
{
    int single_block_threshold = 32768;
    if(std::is_same_v<T, float>)
        single_block_threshold = 31000;
    else if(std::is_same_v<T, rocblas_float_complex>)
        single_block_threshold = 16000;
    else if(std::is_same_v<T, double>)
        single_block_threshold = 13000;
    else if(std::is_same_v<T, rocblas_double_complex>)
        single_block_threshold = 10000;
    return single_block_threshold;
}

template <bool ONE_BLOCK, typename V, typename T>
__inline__ __device__ void
    rocblas_dot_save_sum(V sum, size_t batch, V* __restrict__ workspace, T* __restrict__ out)
{
    if(threadIdx.x == 0)
    {
        if(ONE_BLOCK || gridDim.x == 1) // small N avoid second kernel
            out[batch] = T(sum);
        else
            workspace[blockIdx.x + batch * gridDim.x] = sum;
    }
}

template <bool ONE_BLOCK, int NB, int WIN, bool CONJ, typename T, typename U, typename V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_inc1(rocblas_int n,
                        const U __restrict__ xa,
                        rocblas_stride shiftx,
                        rocblas_stride stridex,
                        const U __restrict__ ya,
                        rocblas_stride shifty,
                        rocblas_stride stridey,
                        rocblas_int    batch_count,
                        V* __restrict__ workspace,
                        T* __restrict__ out)
{
    int      i     = !ONE_BLOCK ? blockIdx.x * NB + threadIdx.x : threadIdx.x;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        const auto* x = load_ptr_batch(xa, batch, shiftx, stridex);
        const auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        V sum = 0;

        // sum WIN elements per thread
        int inc = !ONE_BLOCK ? NB * gridDim.x : NB;
        for(int j = 0; j < WIN && i < n; j++, i += inc)
        {
            sum += V(y[i]) * V(CONJ ? conj(x[i]) : x[i]);
        }

        if(warpSize == WARP_32)
            sum = rocblas_dot_block_reduce<WARP_32, NB>(sum);
        else
            sum = rocblas_dot_block_reduce<WARP_64, NB>(sum);

        rocblas_dot_save_sum<ONE_BLOCK>(sum, batch, workspace, out);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <bool ONE_BLOCK, int NB, int WIN, bool CONJ, typename T, typename U, typename V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_inc1by2(rocblas_int n,
                           const U __restrict__ xa,
                           rocblas_stride shiftx,
                           rocblas_stride stridex,
                           const U __restrict__ ya,
                           rocblas_stride shifty,
                           rocblas_stride stridey,
                           rocblas_int    batch_count,
                           V* __restrict__ workspace,
                           T* __restrict__ out)
{
    int      i     = !ONE_BLOCK ? blockIdx.x * NB + threadIdx.x : threadIdx.x;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto* x = load_ptr_batch(xa, batch, shiftx, stridex);
        const auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        V sum = 0;

        // sum WIN elements per thread
        int inc = !ONE_BLOCK ? NB * gridDim.x : NB;

        if constexpr(
            std::is_same_v<
                T,
                rocblas_half> || std::is_same_v<T, rocblas_bfloat16> || std::is_same_v<T, rocblas_float>)
        {
            i *= 2;
            inc *= 2;
            for(int j = 0; j < WIN && i < n - 1; j++, i += inc)
            {
#pragma unroll
                for(int k = 0; k < 2; ++k)
                {
                    sum += V(y[i + k]) * V(CONJ ? conj(x[i + k]) : x[i + k]);
                }
            }
            // If `n` is odd then the computation of last element is covered below.
            if(n % 2 && i == n - 1)
            {
                sum += V(y[i]) * V(CONJ ? conj(x[i]) : x[i]);
            }
        }
        else
        {
            for(int j = 0; j < WIN && i < n; j++, i += inc)
            {
                sum += V(y[i]) * V(CONJ ? conj(x[i]) : x[i]);
            }
        }

        if(warpSize == WARP_32)
            sum = rocblas_dot_block_reduce<WARP_32, NB>(sum);
        else
            sum = rocblas_dot_block_reduce<WARP_64, NB>(sum);

        rocblas_dot_save_sum<ONE_BLOCK>(sum, batch, workspace, out);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename API_INT,
          bool ONE_BLOCK,
          int  NB,
          int  WIN,
          bool CONJ,
          typename T,
          typename U,
          typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel(rocblas_int n,
                   const U __restrict__ xa,
                   rocblas_stride shiftx,
                   API_INT        incx,
                   rocblas_stride stridex,
                   const U __restrict__ ya,
                   rocblas_stride shifty,
                   API_INT        incy,
                   rocblas_stride stridey,
                   rocblas_int    batch_count,
                   V* __restrict__ workspace,
                   T* __restrict__ out)
{
    int      i     = !ONE_BLOCK ? blockIdx.x * NB + threadIdx.x : threadIdx.x;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto* x = load_ptr_batch(xa, batch, shiftx, stridex);
        const auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        V sum = 0;

        // sum WIN elements per thread
        int inc = NB * gridDim.x;
        for(int j = 0; j < WIN && i < n; j++, i += inc)
        {
            sum += V(y[i * int64_t(incy)])
                   * V(CONJ ? conj(x[i * int64_t(incx)]) : x[i * int64_t(incx)]);
        }
        if(warpSize == WARP_32)
            sum = rocblas_dot_block_reduce<WARP_32, NB>(sum);
        else
            sum = rocblas_dot_block_reduce<WARP_64, NB>(sum);

        rocblas_dot_save_sum<ONE_BLOCK>(sum, batch, workspace, out);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename API_INT, int NB, typename T, typename U, typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_gfx942_float_double(rocblas_int n,
                                       const U __restrict__ xa,
                                       rocblas_stride shiftx,
                                       API_INT        incx,
                                       rocblas_stride stridex,
                                       const U __restrict__ ya,
                                       rocblas_stride shifty,
                                       API_INT        incy,
                                       rocblas_stride stridey,
                                       V* __restrict__ workspace,
                                       T* __restrict__ out)
{
// gfx942 kernels
#if defined(__gfx942__)
    int         i = blockIdx.x * NB + threadIdx.x;
    const auto* x = load_ptr_batch(xa, blockIdx.z, shiftx, stridex);
    const auto* y = load_ptr_batch(ya, blockIdx.z, shifty, stridey);

    V sum = 0;

    //Loop unrolled for i threads
    if((i + (3 * NB * gridDim.x)) < n)
    {
        sum += V(y[i * int64_t(incy)]) * V(x[i * int64_t(incx)]);
        sum += V(y[(i + (NB * gridDim.x)) * int64_t(incy)])
               * V(x[(i + (NB * gridDim.x)) * int64_t(incx)]);
        sum += V(y[(i + (2 * NB * gridDim.x)) * int64_t(incy)])
               * V(x[(i + (2 * NB * gridDim.x)) * int64_t(incx)]);
        sum += V(y[(i + (3 * NB * gridDim.x)) * int64_t(incy)])
               * V(x[(i + (3 * NB * gridDim.x)) * int64_t(incx)]);
        i += (4 * NB * gridDim.x);
    }

    //Loop for other i threads which did not do the computation in the above-unrolled loop
    for(; i < (4 * NB * gridDim.x) && i < n; i += NB * gridDim.x)
        sum += V(y[i * int64_t(incy)]) * V(x[i * int64_t(incx)]);

    if(warpSize == WARP_32)
        sum = rocblas_dot_block_reduce<WARP_32, NB>(sum);
    else
        sum = rocblas_dot_block_reduce<WARP_64, NB>(sum);

    rocblas_dot_save_sum<false>(sum, blockIdx.z, workspace, out);
#endif
}

template <typename API_INT,
          bool ONE_BLOCK,
          int  NB,
          int  WIN,
          bool CONJ,
          typename T,
          typename U,
          typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_magsq(rocblas_int n,
                         const U __restrict__ xa,
                         rocblas_stride shiftx,
                         API_INT        incx,
                         rocblas_stride stridex,
                         rocblas_int    batch_count,
                         V* __restrict__ workspace,
                         T* __restrict__ out)
{
    int      i     = !ONE_BLOCK ? blockIdx.x * NB + threadIdx.x : threadIdx.x;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto* x = load_ptr_batch(xa, batch, shiftx, stridex);

        V sum = 0;

        // sum WIN elements per thread
        int inc = NB * gridDim.x;
        for(int j = 0; j < WIN && i < n; j++, i += inc)
        {
            int64_t idx = i * int64_t(incx);
            sum += V(x[idx]) * V(CONJ ? conj(x[idx]) : x[idx]);
        }
        if(warpSize == WARP_32)
            sum = rocblas_dot_block_reduce<WARP_32, NB>(sum);
        else
            sum = rocblas_dot_block_reduce<WARP_64, NB>(sum);

        rocblas_dot_save_sum<ONE_BLOCK>(sum, batch, workspace, out);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename API_INT, int WARP, int NB_Y, bool CONJ, typename V, typename T, typename U>
ROCBLAS_KERNEL(WARP* NB_Y)
rocblas_dot_batched_4_kernel(rocblas_int n,
                             const U __restrict__ xa,
                             rocblas_stride shiftx,
                             API_INT        incx,
                             rocblas_stride stridex,
                             const U __restrict__ ya,
                             rocblas_stride shifty,
                             API_INT        incy,
                             rocblas_stride stridey,
                             rocblas_int    batch_count,
                             T* __restrict__ out)
{

    uint32_t batch = blockIdx.x * NB_Y + threadIdx.y;

    if(batch >= batch_count)
        return;

    const auto* x = load_ptr_batch(xa, batch, shiftx, stridex);
    const auto* y = load_ptr_batch(ya, batch, shifty, stridey);

    V reg_x = V(0), reg_y = V(0), sum = V(0);

    for(int tid = threadIdx.x; tid < n; tid += WARP)
    {
        reg_x = V(CONJ ? conj(x[tid * int64_t(incx)]) : x[tid * int64_t(incx)]);
        reg_y = V(y[tid * int64_t(incy)]);
        sum += reg_x * reg_y;
    }
    __syncthreads();

    sum = rocblas_wavefront_reduce<WARP>(sum); // sum over wavefront

    if(threadIdx.x == 0)
        out[batch] = T(sum);
}

// assume workspace has already been allocated, recommended for repeated calling of dot_strided_batched product
// routine
template <typename API_INT, int NB, bool CONJ, typename T, typename U, typename V>
rocblas_status rocblas_internal_dot_launcher(rocblas_handle __restrict__ handle,
                                             API_INT n,
                                             const U __restrict__ x,
                                             rocblas_stride offsetx,
                                             API_INT        incx,
                                             rocblas_stride stridex,
                                             const U __restrict__ y,
                                             rocblas_stride offsety,
                                             API_INT        incy,
                                             rocblas_stride stridey,
                                             API_INT        batch_count,
                                             T* __restrict__ results,
                                             V* __restrict__ workspace)
{

    // One or two kernels are used to finish the reduction
    // kernel 1 write partial results per thread block in workspace, number of partial results is blocks
    // kernel 2 if blocks > 1 the partial results in workspace are reduced to output

    // Quick return if possible.
    if(n <= 0 || batch_count == 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        else if(rocblas_pointer_mode_device == handle->pointer_mode && batch_count > 0)
        {
            RETURN_IF_HIP_ERROR(
                hipMemsetAsync(&results[0], 0, batch_count * sizeof(T), handle->get_stream()));
        }
        else
        {
            for(int i = 0; i < batch_count; i++)
            {
                results[i] = T(0);
            }
        }

        return rocblas_status_success;
    }

    //Identifying the precision to have an appropriate optimization
    static constexpr bool is_float  = std::is_same_v<V, float> && std::is_same_v<T, float>;
    static constexpr bool is_double = std::is_same_v<V, double> && std::is_same_v<T, double>;

    //Identifying the architecture to have an appropriate optimization
    int  arch_major = handle->getArchMajor();
    bool is_gfx942  = handle->getArch() == 942 ? true : false;

    static constexpr int WIN = rocblas_dot_WIN<T>();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    int64_t shiftx = incx < 0 ? offsetx - int64_t(incx) * (n - 1) : offsetx;
    int64_t shifty = incy < 0 ? offsety - int64_t(incy) * (n - 1) : offsety;

    static constexpr int single_block_threshold = rocblas_dot_one_block_threshold<T>();

    if(n <= 1024 && batch_count >= 256)
    {
        // Optimized kernel for small n and bigger batch_count
        static constexpr int NB_Y = 4;

        dim3 grid((batch_count - 1) / NB_Y + 1);

        T* output = results; // device mode output directly to results
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            size_t offset = size_t(batch_count);
            output        = (T*)(workspace + offset);
        }

        if(handle->getWarpSize() == WARP_32)
        {
            // threadIdx.x all work on same batch index, threadIdx.y used for batch idx selection
            dim3 threads(WARP_32, NB_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_dot_batched_4_kernel<API_INT, WARP_32, NB_Y, CONJ, V>),
                                  grid,
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  n,
                                  x,
                                  shiftx,
                                  incx,
                                  stridex,
                                  y,
                                  shifty,
                                  incy,
                                  stridey,
                                  batch_count,
                                  output);
        }
        else
        {
            // threadIdx.x all work on same batch index, threadIdx.y used for batch idx selection
            dim3 threads(WARP_64, NB_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_dot_batched_4_kernel<API_INT, WARP_64, NB_Y, CONJ, V>),
                                  grid,
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  n,
                                  x,
                                  shiftx,
                                  incx,
                                  stridex,
                                  y,
                                  shifty,
                                  incy,
                                  stridey,
                                  batch_count,
                                  output);
        }
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[0],
                                               output,
                                               sizeof(T) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        }
    }
    else if(n <= single_block_threshold)
    {
        // we only reduce the block count to 1 so safe to ignore extra workspace allocated in caller

        static constexpr int NB_OB  = 1024;
        static constexpr int WIN_OB = 32; // 32K max n threshold, assert guard below

        rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB_OB * WIN_OB);
        assert(blocks == 1);
        static constexpr bool ONE_BLOCK = true;

        int batches = handle->getBatchGridDim((int)batch_count);

        dim3 grid(blocks, 1, batches);
        dim3 threads(NB_OB);

        T* output = results; // device mode output directly to results
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            size_t offset = size_t(batch_count) * blocks;
            output        = (T*)(workspace + offset);
        }

        if(x != y || incx != incy || offsetx != offsety || stridex != stridey)
        {
            if(incx == 1 && incy == 1)
            {
                ROCBLAS_LAUNCH_KERNEL(
                    (rocblas_dot_kernel_inc1by2<ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
                    grid,
                    threads,
                    0,
                    handle->get_stream(),
                    n,
                    x,
                    shiftx,
                    stridex,
                    y,
                    shifty,
                    stridey,
                    batch_count,
                    workspace,
                    output);
            }
            else
            {
                ROCBLAS_LAUNCH_KERNEL(
                    (rocblas_dot_kernel<API_INT, ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
                    grid,
                    threads,
                    0,
                    handle->get_stream(),
                    n,
                    x,
                    shiftx,
                    incx,
                    stridex,
                    y,
                    shifty,
                    incy,
                    stridey,
                    batch_count,
                    workspace,
                    output);
            }
        }
        else // x dot x
        {
            ROCBLAS_LAUNCH_KERNEL(
                (rocblas_dot_kernel_magsq<API_INT, ONE_BLOCK, NB_OB, WIN_OB, CONJ, T>),
                grid,
                threads,
                0,
                handle->get_stream(),
                n,
                x,
                shiftx,
                incx,
                stridex,
                batch_count,
                workspace,
                output);
        }

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[0],
                                               output,
                                               sizeof(T) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        }
    }
    //optimized gfx942 kernel for very large N
    else if(is_gfx942 && (is_float || is_double) && n > sddot_gfx942_lower_threshold
            && (x != y || incx != incy || offsetx != offsety || stridex != stridey))
    {
        static constexpr bool ONE_BLOCK = false;
        static constexpr int  DOT_NB    = 1024;
        static constexpr int  DOT_NELEM = 4;
        rocblas_int           blocks = rocblas_reduction_kernel_block_count(n, DOT_NB * DOT_NELEM);
        T*                    output = results;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            size_t offset = size_t(batch_count) * blocks;
            output        = (T*)(workspace + offset);
        }

        dim3 grid(blocks, 1, batch_count);
        dim3 threads(DOT_NB);

        ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_gfx942_float_double<API_INT, DOT_NB, T>),
                              grid,
                              threads,
                              0,
                              handle->get_stream(),
                              n,
                              x,
                              shiftx,
                              incx,
                              stridex,
                              y,
                              shifty,
                              incy,
                              stridey,
                              workspace,
                              output);

        ROCBLAS_LAUNCH_KERNEL(
            (rocblas_reduction_kernel_part2<DOT_NB, DOT_NELEM, rocblas_finalize_identity>),
            dim3(batch_count),
            threads,
            0,
            handle->get_stream(),
            blocks,
            workspace,
            output);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[0],
                                               output,
                                               sizeof(T) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        }
    }
    else
    {
        static constexpr bool ONE_BLOCK = false;

        rocblas_int blocks = rocblas_reduction_kernel_block_count(n, NB * WIN);

        int batches = handle->getBatchGridDim((int)batch_count);

        dim3 grid(blocks, 1, batches);
        dim3 threads(NB);

        T* output = results;
        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            size_t offset = size_t(batch_count) * blocks;
            output        = (T*)(workspace + offset);
        }

        if(x != y || incx != incy || offsetx != offsety || stridex != stridey)
        {
            if(incx == 1 && incy == 1)
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_inc1<ONE_BLOCK, NB, WIN, CONJ, T>),
                                      grid,
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      n,
                                      x,
                                      shiftx,
                                      stridex,
                                      y,
                                      shifty,
                                      stridey,
                                      batch_count,
                                      workspace,
                                      output);
            }
            else
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel<API_INT, ONE_BLOCK, NB, WIN, CONJ, T>),
                                      grid,
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      n,
                                      x,
                                      shiftx,
                                      incx,
                                      stridex,
                                      y,
                                      shifty,
                                      incy,
                                      stridey,
                                      batch_count,
                                      workspace,
                                      output);
            }
        }
        else // x dot x
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_dot_kernel_magsq<API_INT, ONE_BLOCK, NB, WIN, CONJ, T>),
                                  grid,
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  n,
                                  x,
                                  shiftx,
                                  incx,
                                  stridex,
                                  batch_count,
                                  workspace,
                                  output);
        }

        if(blocks > 1) // if single block first kernel did all work
            ROCBLAS_LAUNCH_KERNEL(
                (rocblas_reduction_kernel_part2<NB, WIN, rocblas_finalize_identity>),
                dim3(batch_count),
                threads,
                0,
                handle->get_stream(),
                blocks,
                workspace,
                output);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[0],
                                               output,
                                               sizeof(T) * batch_count,
                                               hipMemcpyDeviceToHost,
                                               handle->get_stream()));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        }
    }
    return rocblas_status_success;
}

// for ex interface and _64 reuse
#ifdef INST_DOT_EX_LAUNCHER
#error INST_DOT_EX_LAUNCHER already defined
#endif

#define INST_DOT_EX_LAUNCHER(API_INT_, NB_, CONJ_, T_, U_, V_)           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status             \
        rocblas_internal_dot_launcher<API_INT_, NB_, CONJ_, T_, U_, V_>( \
            rocblas_handle __restrict__ handle,                          \
            API_INT_ n,                                                  \
            U_ __restrict__ x,                                           \
            rocblas_stride offsetx,                                      \
            API_INT_       incx,                                         \
            rocblas_stride stridex,                                      \
            U_ __restrict__ y,                                           \
            rocblas_stride offsety,                                      \
            API_INT_       incy,                                         \
            rocblas_stride stridey,                                      \
            API_INT_       batch_count,                                  \
            T_* __restrict__ results,                                    \
            V_* __restrict__ workspace);
