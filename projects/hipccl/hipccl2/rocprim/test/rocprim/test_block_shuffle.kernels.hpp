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

#ifndef TEST_BLOCK_SHUFFLE_KERNELS_HPP_
#define TEST_BLOCK_SHUFFLE_KERNELS_HPP_

template<
    unsigned int BlockSize,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void shuffle_offset_kernel(T* device_input, T* device_output, int distance)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    rocprim::block_shuffle<T,BlockSize> b_shuffle;
    b_shuffle.offset(device_input[index],device_output[index],distance);
}

template<
    unsigned int BlockSize,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void shuffle_rotate_kernel(T* device_input, T* device_output, int distance)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    rocprim::block_shuffle<T,BlockSize> b_shuffle;
    b_shuffle.rotate(device_input[index],device_output[index],distance);
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void shuffle_up_kernel(T (*device_input), T (*device_output))
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    rocprim::block_shuffle<T,BlockSize> b_shuffle;
    b_shuffle.template up<ItemsPerThread>(reinterpret_cast<T(&)[ItemsPerThread]>(device_input[index*ItemsPerThread]),reinterpret_cast<T(&)[ItemsPerThread]>(device_output[index*ItemsPerThread]));
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void shuffle_down_kernel(T (*device_input), T (*device_output))
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    rocprim::block_shuffle<T,BlockSize> b_shuffle;
    b_shuffle.template down<ItemsPerThread>(reinterpret_cast<T(&)[ItemsPerThread]>(device_input[index*ItemsPerThread]),reinterpret_cast<T(&)[ItemsPerThread]>(device_output[index*ItemsPerThread]));
}

#endif // TEST_BLOCK_SHUFFLE_KERNELS_HPP_
