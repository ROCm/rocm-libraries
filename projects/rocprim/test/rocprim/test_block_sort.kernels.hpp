// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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
        case rocprim::block_sort_algorithm::stable_merge_sort:
            return (rocprim::detail::is_power_of_two(ItemsPerThread)
                    && rocprim::detail::is_power_of_two(BlockSize));
        case rocprim::block_sort_algorithm::bitonic_sort: return true;
    }
    return false;
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeyIterator,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp = rocprim::less<typename std::iterator_traits<KeyIterator>::value_type>,
    class OffsetT,
    std::enable_if_t<(ItemsPerThread == 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
                     int>
    = 0>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(KeyIterator keys, OffsetT size)
{
    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    using bsort_type
        = rocprim::block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type, algorithm>;

    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    const unsigned int                  index         = block_offset + threadIdx.x;
    key_type                            thread_key;

    if(index < size)
    {
        thread_key = keys[index];
    }

    if(size % ItemsPerBlock == 0)
    {
        bsort_type().sort(thread_key, BinaryOp());
    }
    else
    {
        ROCPRIM_SHARED_MEMORY typename bsort_type::storage_type storage;
        bsort_type().sort(thread_key,
                          storage,
                          std::min(static_cast<size_t>(ItemsPerBlock), size - block_offset),
                          BinaryOp());
    }

    if(index < size)
    {
        keys[index] = thread_key;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeyIterator,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp = rocprim::less<typename std::iterator_traits<KeyIterator>::value_type>,
    class OffsetT,
    std::enable_if_t<(ItemsPerThread > 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
                     int>
    = 0>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(KeyIterator keys, OffsetT size)
{
    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    using bsort_type
        = rocprim::block_sort<key_type, BlockSize, ItemsPerThread, rocprim::empty_type, algorithm>;

    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  lid           = threadIdx.x;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    key_type                            thread_keys[ItemsPerThread];
    unsigned int valid = std::min(static_cast<OffsetT>(ItemsPerBlock), size - block_offset);

    rocprim::block_load_direct_blocked(lid, keys + block_offset, thread_keys, valid);

    if(size % ItemsPerBlock == 0)
    {
        bsort_type().sort(thread_keys, BinaryOp());
    }
    else
    {
        ROCPRIM_SHARED_MEMORY typename bsort_type::storage_type storage;
        bsort_type().sort(thread_keys, storage, valid, BinaryOp());
    }

    rocprim::block_store_direct_blocked(lid, keys + block_offset, thread_keys, valid);
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class KeyIterator,
         rocprim::block_sort_algorithm algorithm,
         class BinaryOp = rocprim::less<typename std::iterator_traits<KeyIterator>::value_type>,
         class OffsetT,
         std::enable_if_t<!is_buildable(BlockSize, ItemsPerThread, algorithm), int> = 0>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(KeyIterator /*keys*/,
                                                              OffsetT /*size*/)
{}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    class value_type,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp = rocprim::less<key_type>,
    class OffsetT,
    std::enable_if_t<(ItemsPerThread == 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
                     int>
    = 0>
__global__ __launch_bounds__(BlockSize) void sort_pairs_kernel(key_type*   keys,
                                                               value_type* values,
                                                               OffsetT     size)
{
    using bsort_type
        = rocprim::block_sort<key_type, BlockSize, ItemsPerThread, value_type, algorithm>;

    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    const unsigned int                  index         = block_offset + threadIdx.x;
    key_type                            thread_key;
    value_type                          thread_value;

    if(index < size)
    {
        thread_key   = keys[index];
        thread_value = values[index];
    }

    if(size % ItemsPerBlock == 0)
    {
        bsort_type().sort(thread_key, thread_value, BinaryOp());
    }
    else
    {
        ROCPRIM_SHARED_MEMORY typename bsort_type::storage_type storage;
        bsort_type().sort(thread_key,
                          thread_value,
                          storage,
                          std::min(static_cast<size_t>(ItemsPerBlock), size - block_offset),
                          BinaryOp());
    }

    if(index < size)
    {
        keys[index]   = thread_key;
        values[index] = thread_value;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class key_type,
    class value_type,
    rocprim::block_sort_algorithm algorithm,
    class BinaryOp = rocprim::less<key_type>,
    class OffsetT,
    std::enable_if_t<(ItemsPerThread > 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
                     int>
    = 0>
__global__ __launch_bounds__(BlockSize) void sort_pairs_kernel(key_type*   keys,
                                                               value_type* values,
                                                               OffsetT     size)
{
    using bsort_type
        = rocprim::block_sort<key_type, BlockSize, ItemsPerThread, value_type, algorithm>;

    static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
    const unsigned int                  lid           = threadIdx.x;
    const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
    key_type                            thread_keys[ItemsPerThread];
    value_type                          thread_values[ItemsPerThread];
    unsigned int valid = std::min(static_cast<OffsetT>(ItemsPerBlock), size - block_offset);

    rocprim::block_load_direct_blocked(lid, keys + block_offset, thread_keys, valid);
    rocprim::block_load_direct_blocked(lid, values + block_offset, thread_values, valid);

    if(size % ItemsPerBlock == 0)
    {
        bsort_type().sort(thread_keys, thread_values, BinaryOp());
    }
    else
    {
        ROCPRIM_SHARED_MEMORY typename bsort_type::storage_type storage;
        bsort_type().sort(thread_keys, thread_values, storage, valid, BinaryOp());
    }

    rocprim::block_store_direct_blocked(lid, keys + block_offset, thread_keys, valid);
    rocprim::block_store_direct_blocked(lid, values + block_offset, thread_values, valid);
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class key_type,
         class value_type,
         rocprim::block_sort_algorithm algorithm,
         class BinaryOp = rocprim::less<key_type>,
         class OffsetT,
         std::enable_if_t<!is_buildable(BlockSize, ItemsPerThread, algorithm), int> = 0>
__global__ __launch_bounds__(BlockSize) void sort_pairs_kernel(key_type* /*keys*/,
                                                               value_type* /*values*/,
                                                               OffsetT /*size*/)
{}

#endif // TEST_BLOCK_SORT_KERNELS_HPP_
