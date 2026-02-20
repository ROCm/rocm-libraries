// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TEST_SCAN_REDUCE_KERNELS_HPP_
#define TEST_SCAN_REDUCE_KERNELS_HPP_

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_inclusive_scan_kernel(T* device_input, T* device_output)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
        const unsigned int     warp_id  = rocprim::detail::logical_warp_id<LogicalWarpSize>();
        unsigned int           index    = threadIdx.x + (blockIdx.x * blockDim.x);

        T value = device_input[index];

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[warps_no];
        wscan_t().inclusive_scan(value, value, storage[warp_id]);

        device_output[index] = value;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_inclusive_scan_initial_value_kernel(T* device_input, T* device_output, T initial_value)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
        const unsigned int     warp_id  = rocprim::detail::logical_warp_id<LogicalWarpSize>();
        unsigned int           index    = threadIdx.x + (blockIdx.x * blockDim.x);

        T value = device_input[index];

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[warps_no];
        wscan_t().inclusive_scan(value, value, storage[warp_id], initial_value);

        device_output[index] = value;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_inclusive_scan_reduce_kernel(T* device_input,
                                       T* device_output,
                                       T* device_output_reductions)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
        const unsigned int     warp_id  = rocprim::detail::logical_warp_id<LogicalWarpSize>();
        unsigned int           index    = threadIdx.x + (blockIdx.x * BlockSize);

        T value = device_input[index];
        T reduction;

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[warps_no];
        wscan_t().inclusive_scan(value, value, reduction, storage[warp_id]);

        device_output[index] = value;
        if((threadIdx.x % LogicalWarpSize) == 0)
        {
            device_output_reductions[index / LogicalWarpSize] = reduction;
        }
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_inclusive_scan_reduce_initial_value_kernel(T* device_input,
                                                     T* device_output,
                                                     T* device_output_reductions,
                                                     T  initial_value)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
        const unsigned int     warp_id  = rocprim::detail::logical_warp_id<LogicalWarpSize>();
        unsigned int           index    = threadIdx.x + (blockIdx.x * BlockSize);

        T value = device_input[index];
        T reduction;

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[warps_no];
        wscan_t().inclusive_scan(value, value, reduction, storage[warp_id], initial_value);

        device_output[index] = value;
        if((threadIdx.x % LogicalWarpSize) == 0)
        {
            device_output_reductions[index / LogicalWarpSize] = reduction;
        }
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_exclusive_scan_kernel(T* device_input, T* device_output, T init)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
        const unsigned int     warp_id  = rocprim::detail::logical_warp_id<LogicalWarpSize>();
        unsigned int           index    = threadIdx.x + (blockIdx.x * blockDim.x);

        T value = device_input[index];

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[warps_no];
        wscan_t().exclusive_scan(value, value, init, storage[warp_id]);

        device_output[index] = value;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_exclusive_scan_reduce_kernel(T* device_input,
                                       T* device_output,
                                       T* device_output_reductions,
                                       T  init)
{

    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
        const unsigned int     warp_id  = rocprim::detail::logical_warp_id<LogicalWarpSize>();
        unsigned int           index    = threadIdx.x + (blockIdx.x * blockDim.x);

        T value = device_input[index];
        T reduction;

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[warps_no];
        wscan_t().exclusive_scan(value, value, init, reduction, storage[warp_id]);

        device_output[index] = value;
        if((threadIdx.x % LogicalWarpSize) == 0)
        {
            device_output_reductions[index / LogicalWarpSize] = reduction;
        }
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_broadcast_kernel(T* device_input, T* device_output)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        const unsigned int index    = threadIdx.x + (blockIdx.x * blockDim.x);
        const unsigned int warp_id  = index / LogicalWarpSize;
        const unsigned int src_lane = warp_id % LogicalWarpSize;

        T value = device_input[index];

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage;
        value = wscan_t().broadcast(value, src_lane, storage);

        device_output[index] = value;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_exclusive_scan_wo_init_kernel(T* device_input, T* device_output)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        static constexpr unsigned int block_warps_no = BlockSize / LogicalWarpSize;

        const unsigned int global_index  = threadIdx.x + (blockIdx.x * blockDim.x);
        const unsigned int block_warp_id = threadIdx.x / LogicalWarpSize;

        T value = device_input[global_index];

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[block_warps_no];
        wscan_t().exclusive_scan(value, value, storage[block_warp_id]);

        device_output[global_index] = value;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_exclusive_scan_reduce_wo_init_kernel(T* device_input,
                                               T* device_output,
                                               T* device_output_reductions)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        static constexpr unsigned int block_warps_no = BlockSize / LogicalWarpSize;

        const unsigned int global_index   = threadIdx.x + (blockIdx.x * blockDim.x);
        const unsigned int block_warp_id  = threadIdx.x / LogicalWarpSize;
        const unsigned int lane_id        = threadIdx.x % LogicalWarpSize;
        const unsigned int global_warp_id = global_index / LogicalWarpSize;

        T value = device_input[global_index];
        T reduction;

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[block_warps_no];
        wscan_t().exclusive_scan(value, value, storage[block_warp_id], reduction);

        device_output[global_index] = value;
        if(lane_id == 0)
        {
            device_output_reductions[global_warp_id] = reduction;
        }
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_scan_kernel(T* device_input,
                      T* device_inclusive_output,
                      T* device_exclusive_output,
                      T  init)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
        const unsigned int     warp_id  = rocprim::detail::logical_warp_id<LogicalWarpSize>();
        unsigned int           index    = threadIdx.x + (blockIdx.x * blockDim.x);

        T input = device_input[index];
        T inclusive_output, exclusive_output;

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[warps_no];
        wscan_t().scan(input, inclusive_output, exclusive_output, init, storage[warp_id]);

        device_inclusive_output[index] = inclusive_output;
        device_exclusive_output[index] = exclusive_output;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
void warp_scan_reduce_kernel(T* device_input,
                             T* device_inclusive_output,
                             T* device_exclusive_output,
                             T* device_output_reductions,
                             T  init)
{
    if constexpr(LogicalWarpSize <= rocprim::arch::wavefront::max_size())
    {
        constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
        const unsigned int     warp_id  = rocprim::detail::logical_warp_id<LogicalWarpSize>();
        unsigned int           index    = threadIdx.x + (blockIdx.x * blockDim.x);

        T input = device_input[index];
        T inclusive_output, exclusive_output, reduction;

        using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
        __shared__ typename wscan_t::storage_type storage[warps_no];
        wscan_t()
            .scan(input, inclusive_output, exclusive_output, init, reduction, storage[warp_id]);

        device_inclusive_output[index] = inclusive_output;
        device_exclusive_output[index] = exclusive_output;
        if((threadIdx.x % LogicalWarpSize) == 0)
        {
            device_output_reductions[index / LogicalWarpSize] = reduction;
        }
    }
}

#endif // TEST_SCAN_REDUCE_KERNELS_HPP_
