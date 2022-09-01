// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_BLOCK_SORT_KERNELS_HPP_
#define TEST_BLOCK_SORT_KERNELS_HPP_

constexpr bool is_buildable(unsigned int                  BlockSize,
                            unsigned int                  ItemsPerThread,
                            rocprim::block_sort_algorithm algorithm)
{
    switch(algorithm)
    {
        case rocprim::block_sort_algorithm::merge_sort:
            return (rocprim::detail::is_power_of_two(ItemsPerThread)
                    && rocprim::detail::is_power_of_two(BlockSize));
        case rocprim::block_sort_algorithm::bitonic_sort:
            return ItemsPerThread == 1u
                   || (ItemsPerThread > 1u && rocprim::detail::is_power_of_two(ItemsPerThread)
                       && rocprim::detail::is_power_of_two(BlockSize));
    }
    return false;
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp = rocprim::less<key_type>,
    class OffsetT,
    std::enable_if_t<(ItemsPerThread == 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
                     int> = 0>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(key_type* device_key_output,
                                                              OffsetT   size)
{
    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    const unsigned int                  index         = block_offset + threadIdx.x;
    if(size % ItemsPerBlock == 0)
    {
        key_type key = device_key_output[index];
        rocprim::block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type, algorithm>
            bsort;
        bsort.sort(key, BinaryOp());
        device_key_output[index] = key;
    }
    else
    {
        key_type key  = device_key_output[index];
        using st_type = typename rocprim::
            block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type>::storage_type;
        ROCPRIM_SHARED_MEMORY st_type                                                 storage;
        rocprim::block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type> bsort;
        bsort.sort(key,
                   storage,
                   std::min(static_cast<size_t>(ItemsPerBlock), size - block_offset),
                   BinaryOp());
        device_key_output[index] = key;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp = rocprim::less<key_type>,
    class OffsetT,
    std::enable_if_t<(ItemsPerThread > 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
                     int> = 0>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(key_type* device_key_output,
                                                              OffsetT /*size*/)
{
    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  lid           = threadIdx.x;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;

    key_type keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, device_key_output + block_offset, keys);

    rocprim::block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type, algorithm> bsort;
    bsort.sort(keys, BinaryOp());

    rocprim::block_store_direct_blocked(lid, device_key_output + block_offset, keys);
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class key_type,
         rocprim::block_sort_algorithm algorithm,
         class BinaryOp = rocprim::less<key_type>,
         class OffsetT,
         std::enable_if_t<!is_buildable(BlockSize, ItemsPerThread, algorithm), int> = 0>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(key_type* /*device_key_output*/,
                                                              OffsetT /*size*/)
{}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    class value_type,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp        = rocprim::less<key_type>,
    std::enable_if_t<(ItemsPerThread == 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
                     int> = 0>
__global__ __launch_bounds__(BlockSize) void sort_pairs_kernel(key_type*   device_key_output,
                                                               value_type* device_value_output)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    key_type           key   = device_key_output[index];
    value_type         value = device_value_output[index];
    rocprim::block_sort<key_type, BlockSize, ItemsPerThread, value_type, algorithm> bsort;
    bsort.sort(key, value, BinaryOp());
    device_key_output[index]   = key;
    device_value_output[index] = value;
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    class value_type,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp        = rocprim::less<key_type>,
    std::enable_if_t<(ItemsPerThread > 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
                     int> = 0>
__global__ __launch_bounds__(BlockSize) void sort_pairs_kernel(key_type*   device_key_output,
                                                               value_type* device_value_output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    key_type keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, device_key_output + block_offset, keys);
    value_type values[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, device_value_output + block_offset, values);
    rocprim::block_sort<key_type, BlockSize, ItemsPerThread, value_type, algorithm> bsort;
    bsort.sort(keys, values, BinaryOp());

    rocprim::block_store_direct_blocked(lid, device_key_output + block_offset, keys);
    rocprim::block_store_direct_blocked(lid, device_value_output + block_offset, values);
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class key_type,
         class value_type,
         rocprim::block_sort_algorithm algorithm,
         class BinaryOp = rocprim::less<key_type>,
         std::enable_if_t<!is_buildable(BlockSize, ItemsPerThread, algorithm), int> = 0>
__global__ __launch_bounds__(BlockSize) void sort_pairs_kernel(key_type* /*device_key_output*/,
                                                               value_type* /*device_value_output*/)
{}

#endif // TEST_BLOCK_SORT_KERNELS_HPP_
