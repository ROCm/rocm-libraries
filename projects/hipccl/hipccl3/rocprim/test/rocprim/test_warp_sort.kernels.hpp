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
    unsigned int ItemsPerThread,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize,
    class T
>
__global__
__launch_bounds__(BlockSize)
auto test_hip_warp_sort(T* d_output)
    -> typename std::enable_if<
        ItemsPerThread == 1 , void
    >::type
{
    unsigned int i = threadIdx.x + (blockIdx.x * blockDim.x);
    T value = d_output[i];
    rocprim::warp_sort<T, LogicalWarpSize> wsort;
    wsort.sort(value);
    d_output[i] = value;
}

template<
    unsigned int ItemsPerThread,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize,
    class KeyType,
    class ValueType
>
__global__
__launch_bounds__(BlockSize)
auto test_hip_sort_key_value_kernel(KeyType* d_output_key, ValueType* d_output_value)
    -> typename std::enable_if<
        ItemsPerThread == 1 , void
    >::type
{
    unsigned int i = threadIdx.x + (blockIdx.x * blockDim.x);
    KeyType key = d_output_key[i];
    ValueType value = d_output_value[i];
    rocprim::warp_sort<KeyType, LogicalWarpSize, ValueType> wsort;
    wsort.sort(key, value);
    d_output_key[i] = key;
    d_output_value[i] = value;
}

template<
    unsigned int ItemsPerThread,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize,
    class KeyType
>
__global__
__launch_bounds__(BlockSize)
auto test_hip_warp_sort(KeyType * device_key_output)
    -> typename std::enable_if<
        (ItemsPerThread != 1 && LogicalWarpSize <= ::rocprim::device_warp_size()) , void
    >::type
{
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    KeyType keys[ItemsPerThread];
    ::rocprim::block_load_direct_warp_striped<LogicalWarpSize>(lid, device_key_output + block_offset, keys);

    rocprim::warp_sort<KeyType, LogicalWarpSize> wsort;
    wsort.sort(keys);

    ::rocprim::block_store_direct_blocked(lid, device_key_output + block_offset, keys);
}

template<
    unsigned int ItemsPerThread,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize,
    class KeyType
>
__global__
__launch_bounds__(BlockSize)
auto test_hip_warp_sort(KeyType* /*device_key_output*/)
    -> typename std::enable_if<
        (ItemsPerThread != 1 && LogicalWarpSize > ::rocprim::device_warp_size()), void
    >::type
{
    // This kernel will never be actually called, the tests are filtered at runtime if the warp size
    // of the current device is less than the tested LogicalWarpSize
}

template<
    unsigned int ItemsPerThread,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize,
    class KeyType,
    class ValueType
>
__global__
__launch_bounds__(BlockSize)
auto test_hip_sort_key_value_kernel(KeyType* device_key_output, ValueType* device_value_output)
    -> typename std::enable_if<
        (ItemsPerThread != 1 && LogicalWarpSize <= ::rocprim::device_warp_size()), void
    >::type
{
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    KeyType keys[ItemsPerThread];
    ValueType values[ItemsPerThread];
    ::rocprim::block_load_direct_warp_striped<LogicalWarpSize>(lid, device_key_output + block_offset, keys);
    ::rocprim::block_load_direct_warp_striped<LogicalWarpSize>(lid, device_value_output + block_offset, values);

    rocprim::warp_sort<KeyType, LogicalWarpSize, ValueType> wsort;
    wsort.sort(keys, values);

    ::rocprim::block_store_direct_blocked(lid, device_key_output + block_offset, keys);
    ::rocprim::block_store_direct_blocked(lid, device_value_output + block_offset, values);

}

template<
    unsigned int ItemsPerThread,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize,
    class KeyType,
    class ValueType
>
__global__
__launch_bounds__(BlockSize)
auto test_hip_sort_key_value_kernel(KeyType* /*device_key_output*/, ValueType* /*device_value_output*/)
    -> typename std::enable_if<
        (ItemsPerThread != 1 && LogicalWarpSize > ::rocprim::device_warp_size()), void
    >::type
{
    // This kernel will never be actually called, the tests are filtered at runtime if the warp size
    // of the current device is less than the tested LogicalWarpSize
}

#endif // TEST_WARP_SORT_KERNELS_HPP_
