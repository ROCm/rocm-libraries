// Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TEST_WARP_REDUCE_KERNELS_HPP_
#define TEST_WARP_REDUCE_KERNELS_HPP_

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_reduce_sum_kernel(T* device_input, T* device_output)
{
    static constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T value = device_input[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, storage[warp_id]);

    if(threadIdx.x%LogicalWarpSize == 0)
    {
        device_output[index/LogicalWarpSize] = value;
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_allreduce_sum_kernel(T* device_input, T* device_output)
{
    static constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T value = device_input[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize, true>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, storage[warp_id]);

    device_output[index] = value;
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_reduce_sum_kernel(T* device_input, T* device_output, size_t valid)
{
    static constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T value = device_input[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, valid, storage[warp_id]);

    if(threadIdx.x%LogicalWarpSize == 0)
    {
        device_output[index/LogicalWarpSize] = value;
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_allreduce_sum_kernel(T* device_input, T* device_output, size_t valid)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T value = device_input[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize, true>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().reduce(value, value, valid, storage[warp_id]);

    device_output[index] = value;
}

template<
    class T,
    class Flag,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void head_segmented_warp_reduce_kernel(T* input, Flag* flags, T* output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T value = input[index];
    auto flag = flags[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize, true>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().head_segmented_reduce(value, value, flag, storage[warp_id]);

    output[index] = value;
}

template<
    class T,
    class Flag,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void tail_segmented_warp_reduce_kernel(T* input, Flag* flags, T* output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = rocprim::detail::logical_warp_id<LogicalWarpSize>();
    unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

    T value = input[index];
    auto flag = flags[index];

    using wreduce_t = rocprim::warp_reduce<T, LogicalWarpSize, true>;
    __shared__ typename wreduce_t::storage_type storage[warps_no];
    wreduce_t().tail_segmented_reduce(value, value, flag, storage[warp_id]);

    output[index] = value;
}

#endif // TEST_WARP_REDUCE_KERNELS_HPP_
