// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_inclusive_scan_kernel(T* device_input, T* device_output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T value = device_input[index];

    using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().inclusive_scan(value, value, storage[warp_id]);

    device_output[index] = value;
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_inclusive_scan_reduce_kernel(
    T* device_input,
    T* device_output,
    T* device_output_reductions)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + ( blockIdx.x * BlockSize );

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

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_exclusive_scan_kernel(T* device_input, T* device_output, T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T value = device_input[index];

    using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().exclusive_scan(value, value, init, storage[warp_id]);

    device_output[index] = value;
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_exclusive_scan_reduce_kernel(
    T* device_input,
    T* device_output,
    T* device_output_reductions,
    T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

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

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_scan_kernel(
    T* device_input,
    T* device_inclusive_output,
    T* device_exclusive_output,
    T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T input = device_input[index];
    T inclusive_output, exclusive_output;

    using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().scan(input, inclusive_output, exclusive_output, init, storage[warp_id]);

    device_inclusive_output[index] = inclusive_output;
    device_exclusive_output[index] = exclusive_output;
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_scan_reduce_kernel(
    T* device_input,
    T* device_inclusive_output,
    T* device_exclusive_output,
    T* device_output_reductions,
    T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T input = device_input[index];
    T inclusive_output, exclusive_output, reduction;

    using wscan_t = rocprim::warp_scan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::storage_type storage[warps_no];
    wscan_t().scan(input, inclusive_output, exclusive_output, init, reduction, storage[warp_id]);

    device_inclusive_output[index] = inclusive_output;
    device_exclusive_output[index] = exclusive_output;
    if((threadIdx.x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

#endif // TEST_SCAN_REDUCE_KERNELS_HPP_
