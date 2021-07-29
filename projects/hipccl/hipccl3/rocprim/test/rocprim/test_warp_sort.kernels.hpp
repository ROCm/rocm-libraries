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

#ifndef TEST_WARP_SORT_KERNELS_HPP_
#define TEST_WARP_SORT_KERNELS_HPP_

template<
    class T,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void test_hip_warp_sort(T* d_output)
{
    unsigned int i = threadIdx.x + (blockIdx.x * blockDim.x);
    T value = d_output[i];
    rocprim::warp_sort<T, LogicalWarpSize> wsort;
    wsort.sort(value);
    d_output[i] = value;
}

template<
    class KeyType,
    class ValueType,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void test_hip_sort_key_value_kernel(KeyType* d_output_key, ValueType* d_output_value)
{
    unsigned int i = threadIdx.x + (blockIdx.x * blockDim.x);
    KeyType key = d_output_key[i];
    ValueType value = d_output_value[i];
    rocprim::warp_sort<KeyType, LogicalWarpSize, ValueType> wsort;
    wsort.sort(key, value);
    d_output_key[i] = key;
    d_output_value[i] = value;
}

#endif // TEST_WARP_SORT_KERNELS_HPP_
