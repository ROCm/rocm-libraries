// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR next
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR nextWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR next DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SORT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SORT_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_sort.hpp"
#include "../../block/block_store.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    class KeysInputIterator,
    class KeysOutputIterator
>
ROCPRIM_DEVICE inline
void block_copy_kernel_impl(KeysInputIterator input,
                            const size_t input_size,
                            KeysOutputIterator output)
{
    using key_type = typename std::iterator_traits<KeysOutputIterator>::value_type;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * BlockSize;
    const unsigned int number_of_blocks = (input_size + BlockSize - 1)/BlockSize;
    auto valid_in_last_block = input_size - BlockSize * (number_of_blocks - 1);

    key_type key[1];

    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            key,
            valid_in_last_block
        );

        block_store_direct_striped<BlockSize>(
            flat_id,
            output + block_offset,
            key,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            key
        );

        block_store_direct_striped<BlockSize>(
            flat_id,
            output + block_offset,
            key
        );
    }
}

template<
    unsigned int BlockSize,
    class KeysInputIterator,
    class KeysOutputIterator,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
void block_sort_kernel_impl(KeysInputIterator input,
                            const size_t input_size,
                            KeysOutputIterator output,
                            BinaryFunction compare_function)
{
    using key_type = typename std::iterator_traits<KeysOutputIterator>::value_type;

    using block_sort_type = ::rocprim::block_sort<
        key_type, BlockSize
    >;

    __shared__ typename block_sort_type::storage_type storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * BlockSize;
    const unsigned int number_of_blocks = (input_size + BlockSize - 1)/BlockSize;
    auto valid_in_last_block = input_size - BlockSize * (number_of_blocks - 1);

    key_type key[1];

    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            key,
            valid_in_last_block
        );

        block_sort_type()
            .sort(
                key[0], // input
                storage,
                valid_in_last_block,
                compare_function
            );

        block_store_direct_striped<BlockSize>(
            flat_id,
            output + block_offset,
            key,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            key
        );

        block_sort_type()
            .sort(
                key[0], // input
                storage,
                compare_function
            );

        block_store_direct_striped<BlockSize>(
            flat_id,
            output + block_offset,
            key
        );
    }
}

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
void block_merge_kernel_impl(KeysInputIterator input,
                             const size_t input_size,
                             const unsigned int block_size,
                             KeysOutputIterator output,
                             BinaryFunction compare_function)
{
    using key_type = typename std::iterator_traits<KeysOutputIterator>::value_type;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int flat_block_size = ::rocprim::detail::block_size<0>();
    unsigned int id = (flat_block_id * flat_block_size) + flat_id;

    if (id >= input_size)
    {
        return;
    }

    key_type key;

    key = input[id];

    const unsigned int block_id = id / block_size;
    const bool block_id_is_odd = block_id & 1;
    const unsigned int next_block_id = block_id_is_odd ? block_id - 1 :
                                                         block_id + 1;
    const unsigned int block_start = min(block_id * block_size, (unsigned int) input_size);
    //const unsigned int block_end = min((flat_block_id + 1) * BlockSize, (unsigned int) input_size);
    const unsigned int next_block_start = min(next_block_id * block_size, (unsigned int) input_size);
    const unsigned int next_block_end = min((next_block_id + 1) * block_size, (unsigned int) input_size);

    if(next_block_start == input_size)
    {
        output[id] = key;
        return;
    }

    unsigned int left_id = next_block_start;
    unsigned int right_id = next_block_end;

    while(left_id < right_id)
    {
        unsigned int mid_id = (left_id + right_id) / 2;
        key_type mid_key = input[mid_id];
        bool smaller = compare_function(mid_key, key);
        left_id = smaller ? mid_id + 1 : left_id;
        right_id = smaller ? right_id : mid_id;
    }

    right_id = next_block_end;
    if(block_id_is_odd && left_id != right_id)
    {
        key_type upper_key = input[left_id];
        while(!compare_function(upper_key, key) &&
              !compare_function(key, upper_key) &&
              left_id < right_id)
        {
            unsigned int mid_id = (left_id + right_id) / 2;
            key_type mid_key = input[mid_id];
            bool equal = !compare_function(mid_key, key) &&
                         !compare_function(key, mid_key);
            left_id = equal ? mid_id + 1 : left_id + 1;
            right_id = equal ? right_id : mid_id;
            upper_key = input[left_id];
        }
    }

    unsigned int offset = 0;
    offset += id - block_start;
    offset += left_id - next_block_start;
    offset += min(block_start, next_block_start);
    output[offset] = key;
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SORT_HPP_
