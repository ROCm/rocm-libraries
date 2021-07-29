// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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
    bool WithValues,
    unsigned int BlockSize,
    class KeysInputIterator,
    class ValuesInputIterator,
    class Key,
    class Value,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
typename std::enable_if<!WithValues>::type
block_load_impl(const unsigned int flat_id,
                const unsigned int block_offset,
                const unsigned int valid_in_last_block,
                const bool last_block,
                KeysInputIterator keys_input,
                ValuesInputIterator values_input,
                Key (&keys)[ItemsPerThread],
                Value (&values)[ItemsPerThread])
{
    (void) values_input;
    (void) values;

    if(last_block)
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            keys_input + block_offset,
            keys,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            keys_input + block_offset,
            keys
        );
    }

}

template<
    bool WithValues,
    unsigned int BlockSize,
    class KeysInputIterator,
    class ValuesInputIterator,
    class Key,
    class Value,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
typename std::enable_if<WithValues>::type
block_load_impl(const unsigned int flat_id,
                const unsigned int block_offset,
                const unsigned int valid_in_last_block,
                const bool last_block,
                KeysInputIterator keys_input,
                ValuesInputIterator values_input,
                Key (&keys)[ItemsPerThread],
                Value (&values)[ItemsPerThread])
{
    if(last_block)
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            keys_input + block_offset,
            keys,
            valid_in_last_block
        );

        block_load_direct_striped<BlockSize>(
            flat_id,
            values_input + block_offset,
            values,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            keys_input + block_offset,
            keys
        );

        block_load_direct_striped<BlockSize>(
            flat_id,
            values_input + block_offset,
            values
        );
    }
}

template<
    bool WithValues,
    unsigned int BlockSize,
    class KeysOutputIterator,
    class ValuesOutputIterator,
    class Key,
    class Value,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
typename std::enable_if<!WithValues>::type
block_store_impl(const unsigned int flat_id,
                 const unsigned int block_offset,
                 const unsigned int valid_in_last_block,
                 const bool last_block,
                 KeysOutputIterator keys_output,
                 ValuesOutputIterator values_output,
                 Key (&keys)[ItemsPerThread],
                 Value (&values)[ItemsPerThread])
{
    (void) values_output;
    (void) values;

    if(last_block)
    {
        block_store_direct_striped<BlockSize>(
            flat_id,
            keys_output + block_offset,
            keys,
            valid_in_last_block
        );
    }
    else
    {
        block_store_direct_striped<BlockSize>(
            flat_id,
            keys_output + block_offset,
            keys
        );
    }
}

template<
    bool WithValues,
    unsigned int BlockSize,
    class KeysOutputIterator,
    class ValuesOutputIterator,
    class Key,
    class Value,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
typename std::enable_if<WithValues>::type
block_store_impl(const unsigned int flat_id,
                 const unsigned int block_offset,
                 const unsigned int valid_in_last_block,
                 const bool last_block,
                 KeysOutputIterator keys_output,
                 ValuesOutputIterator values_output,
                 Key (&keys)[ItemsPerThread],
                 Value (&values)[ItemsPerThread])
{
    if(last_block)
    {
        block_store_direct_striped<BlockSize>(
            flat_id,
            keys_output + block_offset,
            keys,
            valid_in_last_block
        );

        block_store_direct_striped<BlockSize>(
            flat_id,
            values_output + block_offset,
            values,
            valid_in_last_block
        );
    }
    else
    {
        block_store_direct_striped<BlockSize>(
            flat_id,
            keys_output + block_offset,
            keys
        );

        block_store_direct_striped<BlockSize>(
            flat_id,
            values_output + block_offset,
            values
        );
    }
}

template<
    bool WithValues,
    unsigned int BlockSize,
    class Key,
    class Value,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
typename std::enable_if<!WithValues>::type
block_sort_impl(Key& key,
                Value& value,
                const unsigned int valid_in_last_block,
                const bool last_block,
                BinaryFunction compare_function)
{
    using block_sort_type = ::rocprim::block_sort<
        Key, BlockSize
    >;

    ROCPRIM_SHARED_MEMORY typename block_sort_type::storage_type storage;

    (void) value;

    if(last_block)
    {
        block_sort_type()
            .sort(
                key, // keys_input
                storage,
                valid_in_last_block,
                compare_function
            );
    }
    else
    {
        block_sort_type()
            .sort(
                key, // keys_input
                storage,
                compare_function
            );
    }
}

template<
    bool WithValues,
    unsigned int BlockSize,
    class Key,
    class Value,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
typename std::enable_if<WithValues>::type
block_sort_impl(Key& key,
                Value& value,
                const unsigned int valid_in_last_block,
                const bool last_block,
                BinaryFunction compare_function)
{
    using block_sort_type = ::rocprim::block_sort<
        Key, BlockSize, Value
    >;

    ROCPRIM_SHARED_MEMORY typename block_sort_type::storage_type storage;

    if(last_block)
    {
        block_sort_type()
            .sort(
                key, // keys_input
                value, // values_input
                storage,
                valid_in_last_block,
                compare_function
            );
    }
    else
    {
        block_sort_type()
            .sort(
                key, // keys_input
                value, // values_input
                storage,
                compare_function
            );
    }
}

template<
    unsigned int BlockSize,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
void block_sort_kernel_impl(KeysInputIterator keys_input,
                            KeysOutputIterator keys_output,
                            ValuesInputIterator values_input,
                            ValuesOutputIterator values_output,
                            const size_t input_size,
                            BinaryFunction compare_function)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using stable_key_type = rocprim::tuple<key_type, unsigned int>;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * BlockSize;
    const unsigned int number_of_blocks = (input_size + BlockSize - 1)/BlockSize;
    auto valid_in_last_block = input_size - BlockSize * (number_of_blocks - 1);
    const bool last_block = flat_block_id == (number_of_blocks - 1);

    key_type key[1];
    value_type value[1];

    block_load_impl<with_values, BlockSize>(
        flat_id,
        block_offset,
        valid_in_last_block,
        last_block,
        keys_input,
        values_input,
        key,
        value
    );

    // Special comparison that preserves relative order of equal keys
    auto stable_compare_function = [compare_function](const stable_key_type& a, const stable_key_type& b) mutable -> bool
    {
        const bool ab = compare_function(rocprim::get<0>(a), rocprim::get<0>(b));
        const bool ba = compare_function(rocprim::get<0>(b), rocprim::get<0>(a));
        return ab || (!ba && (rocprim::get<1>(a) < rocprim::get<1>(b)));
    };

    stable_key_type stable_key = rocprim::make_tuple(key[0], flat_id);
    block_sort_impl<with_values, BlockSize>(
        stable_key,
        value[0],
        valid_in_last_block,
        last_block,
        stable_compare_function
    );
    key[0] = rocprim::get<0>(stable_key);

    block_store_impl<with_values, BlockSize>(
        flat_id,
        block_offset,
        valid_in_last_block,
        last_block,
        keys_output,
        values_output,
        key,
        value
    );
}

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
void block_merge_kernel_impl(KeysInputIterator keys_input,
                             KeysOutputIterator keys_output,
                             ValuesInputIterator values_input,
                             ValuesOutputIterator values_output,
                             const size_t input_size,
                             const unsigned int block_size,
                             BinaryFunction compare_function)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int flat_block_size = ::rocprim::detail::block_size<0>();
    unsigned int id = (flat_block_id * flat_block_size) + flat_id;

    if (id >= input_size)
    {
        return;
    }

    key_type key;
    value_type value;

    key = keys_input[id];
    if(with_values)
    {
        value = values_input[id];
    }

    const unsigned int block_id = id / block_size;
    const bool block_id_is_odd = block_id & 1;
    const unsigned int next_block_id = block_id_is_odd ? block_id - 1 :
                                                         block_id + 1;
    const unsigned int block_start = min(block_id * block_size, (unsigned int) input_size);
    const unsigned int next_block_start = min(next_block_id * block_size, (unsigned int) input_size);
    const unsigned int next_block_end = min((next_block_id + 1) * block_size, (unsigned int) input_size);

    if(next_block_start == input_size)
    {
        keys_output[id] = key;
        if(with_values)
        {
            values_output[id] = value;
        }
        return;
    }

    unsigned int left_id = next_block_start;
    unsigned int right_id = next_block_end;

    while(left_id < right_id)
    {
        unsigned int mid_id = (left_id + right_id) / 2;
        key_type mid_key = keys_input[mid_id];
        bool smaller = compare_function(mid_key, key);
        left_id = smaller ? mid_id + 1 : left_id;
        right_id = smaller ? right_id : mid_id;
    }

    right_id = next_block_end;
    if(block_id_is_odd && left_id != right_id)
    {
        key_type upper_key = keys_input[left_id];
        while(!compare_function(upper_key, key) &&
              !compare_function(key, upper_key) &&
              left_id < right_id)
        {
            unsigned int mid_id = (left_id + right_id) / 2;
            key_type mid_key = keys_input[mid_id];
            bool equal = !compare_function(mid_key, key) &&
                         !compare_function(key, mid_key);
            left_id = equal ? mid_id + 1 : left_id + 1;
            right_id = equal ? right_id : mid_id;
            upper_key = keys_input[left_id];
        }
    }

    unsigned int offset = 0;
    offset += id - block_start;
    offset += left_id - next_block_start;
    offset += min(block_start, next_block_start);
    keys_output[offset] = key;
    if(with_values)
    {
        values_output[offset] = value;
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SORT_HPP_
