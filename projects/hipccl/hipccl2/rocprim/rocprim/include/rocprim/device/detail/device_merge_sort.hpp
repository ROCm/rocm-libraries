// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_HPP_

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
    unsigned int ItemsPerThread,
    class Key
>
struct block_load_keys_impl {
    using block_load_type = ::rocprim::block_load<Key,
                                                  BlockSize,
                                                  ItemsPerThread,
                                                  rocprim::block_load_method::block_load_transpose>;

    using storage_type = typename block_load_type::storage_type;

    template <class KeysInputIterator, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(const OffsetT block_offset,
              const unsigned int valid_in_last_block,
              const bool is_incomplete_block,
              KeysInputIterator keys_input,
              Key (&keys)[ItemsPerThread],
              storage_type& storage)
    {
        if(is_incomplete_block)
        {
            block_load_type().load(
                keys_input + block_offset,
                keys,
                valid_in_last_block,
                storage
            );
        }
        else
        {
            block_load_type().load(
                keys_input + block_offset,
                keys,
                storage
            );
        }

    }
};

template <bool WithValues, unsigned int BlockSize, unsigned int ItemsPerThread, class Value>
struct block_load_values_impl
{
    using storage_type = empty_storage_type;

    template <class ValuesInputIterator, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(const unsigned int flat_id,
              const unsigned int (&ranks)[ItemsPerThread],
              const OffsetT block_offset,
              const unsigned int valid_in_last_block,
              const bool is_incomplete_block,
              ValuesInputIterator values_input,
              Value (&values)[ItemsPerThread],
              storage_type& storage)
    {
        (void) flat_id;
        (void) ranks;
        (void) block_offset;
        (void) valid_in_last_block;
        (void) is_incomplete_block;
        (void) values_input;
        (void) values;
        (void) storage;
    }
};

template <unsigned int BlockSize, unsigned int ItemsPerThread, class Value>
struct block_load_values_impl<true, BlockSize, ItemsPerThread, Value>
{
    using block_exchange = ::rocprim::block_exchange<Value, BlockSize, ItemsPerThread>;

    using storage_type = typename block_exchange::storage_type;

    template <class ValuesInputIterator, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(const unsigned int flat_id,
              const unsigned int (&ranks)[ItemsPerThread],
              const OffsetT block_offset,
              const unsigned int valid_in_last_block,
              const bool is_incomplete_block,
              ValuesInputIterator values_input,
              Value (&values)[ItemsPerThread],
              storage_type& storage)
    {
        if(is_incomplete_block)
        {
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
                values_input + block_offset,
                values
            );
        }

        // Synchronize before reusing shared memory
        ::rocprim::syncthreads();
        block_exchange().gather_from_striped(values, values, ranks, storage);
    }
};

template<
    bool WithValues,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class Key,
    class Value
>
struct block_store_impl {
    using block_store_type
        = block_store<Key, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;

    using storage_type = typename block_store_type::storage_type;

    template <class KeysOutputIterator, class ValuesOutputIterator, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(const OffsetT block_offset,
               const unsigned int valid_in_last_block,
               const bool is_incomplete_block,
               KeysOutputIterator keys_output,
               ValuesOutputIterator values_output,
               Key (&keys)[ItemsPerThread],
               Value (&values)[ItemsPerThread],
               storage_type& storage)
    {
        (void) values_output;
        (void) values;

        // Synchronize before reusing shared memory
        ::rocprim::syncthreads();

        if(is_incomplete_block)
        {
            block_store_type().store(
                keys_output + block_offset,
                keys,
                valid_in_last_block,
                storage
            );
        }
        else
        {
            block_store_type().store(
                keys_output + block_offset,
                keys,
                storage
            );
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class Key,
    class Value
>
struct block_store_impl<true, BlockSize, ItemsPerThread, Key, Value> {
    using block_store_key_type   = block_store<Key, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;
    using block_store_value_type = block_store<Value, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;

    union storage_type {
        typename block_store_key_type::storage_type   keys;
        typename block_store_value_type::storage_type values;
    };

    template <class KeysOutputIterator, class ValuesOutputIterator, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(const OffsetT block_offset,
               const unsigned int valid_in_last_block,
               const bool is_incomplete_block,
               KeysOutputIterator keys_output,
               ValuesOutputIterator values_output,
               Key (&keys)[ItemsPerThread],
               Value (&values)[ItemsPerThread],
               storage_type& storage)
    {
        // Synchronize before reusing shared memory
        ::rocprim::syncthreads();

        if(is_incomplete_block)
        {
            block_store_key_type().store(
                keys_output + block_offset,
                keys,
                valid_in_last_block,
                storage.keys
            );

            ::rocprim::syncthreads();

            block_store_value_type().store(
                values_output + block_offset,
                values,
                valid_in_last_block,
                storage.values
            );
        }
        else
        {
            block_store_key_type().store(
                keys_output + block_offset,
                keys,
                storage.keys
            );

            ::rocprim::syncthreads();

            block_store_value_type().store(
                values_output + block_offset,
                values,
                storage.values
            );
        }
    }
};

template <unsigned int BlockSize, unsigned int ItemsPerThread, class Key>
struct block_sort_impl
{
    using stable_key_type = rocprim::tuple<Key, unsigned int>;
    using block_sort_type = ::rocprim::block_sort<stable_key_type, BlockSize, ItemsPerThread>;

    using storage_type = typename block_sort_type::storage_type;

    template <class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(stable_key_type (&keys)[ItemsPerThread],
              storage_type& storage,
              const unsigned int valid_in_last_block,
              const bool is_incomplete_block,
              BinaryFunction compare_function)
    {
        if(is_incomplete_block)
        {
            // Special comparison that sorts out of range values after any "valid" values
            auto oor_compare
                = [compare_function, valid_in_last_block](
                      const stable_key_type& lhs, const stable_key_type& rhs) mutable -> bool {
                const bool left_oor  = rocprim::get<1>(lhs) >= valid_in_last_block;
                const bool right_oor = rocprim::get<1>(rhs) >= valid_in_last_block;
                return (left_oor || right_oor) ? !left_oor : compare_function(lhs, rhs);
            };
            block_sort_type().sort(keys, // keys_input
                                   storage,
                                   oor_compare);
        }
        else
        {
            block_sort_type()
                .sort(
                    keys, // keys_input
                    storage,
                    compare_function
                );
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetT,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void block_sort_kernel_impl(KeysInputIterator keys_input,
                            KeysOutputIterator keys_output,
                            ValuesInputIterator values_input,
                            ValuesOutputIterator values_output,
                            const OffsetT input_size,
                            BinaryFunction compare_function)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    const unsigned int flat_id = block_thread_id<0>();
    const unsigned int flat_block_id = block_id<0>();
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const OffsetT block_offset = flat_block_id * items_per_block;
    const unsigned int valid_in_last_block = input_size - block_offset;
    const bool is_incomplete_block = flat_block_id == (input_size / items_per_block);

    key_type keys[ItemsPerThread];
    value_type values[ItemsPerThread];

    using block_load_keys_impl = block_load_keys_impl<BlockSize, ItemsPerThread, key_type>;
    using block_sort_impl = block_sort_impl<BlockSize, ItemsPerThread, key_type>;
    using block_load_values_impl = block_load_values_impl<with_values, BlockSize, ItemsPerThread, value_type>;
    using block_store_impl = block_store_impl<with_values, BlockSize, ItemsPerThread, key_type, value_type>;

    ROCPRIM_SHARED_MEMORY union {
        typename block_load_keys_impl::storage_type   load_keys;
        typename block_sort_impl::storage_type        sort;
        typename block_load_values_impl::storage_type load_values;
        typename block_store_impl::storage_type       store;
    } storage;

    block_load_keys_impl().load(
        block_offset,
        valid_in_last_block,
        is_incomplete_block,
        keys_input,
        keys,
        storage.load_keys
    );

    using stable_key_type = typename block_sort_impl::stable_key_type;

    // Special comparison that preserves relative order of equal keys
    auto stable_compare_function = [compare_function](const stable_key_type& a, const stable_key_type& b) mutable -> bool
    {
        const bool ab = compare_function(rocprim::get<0>(a), rocprim::get<0>(b));
        const bool ba = compare_function(rocprim::get<0>(b), rocprim::get<0>(a));
        return ab || (!ba && (rocprim::get<1>(a) < rocprim::get<1>(b)));
    };

    stable_key_type stable_keys[ItemsPerThread];
    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item) {
        stable_keys[item] = rocprim::make_tuple(keys[item], ItemsPerThread * flat_id + item);
    }

    // Synchronize before reusing shared memory
    ::rocprim::syncthreads();

    block_sort_impl().sort(
        stable_keys,
        storage.sort,
        valid_in_last_block,
        is_incomplete_block,
        stable_compare_function
    );

    unsigned int ranks[ItemsPerThread];

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item) {
        keys[item]  = rocprim::get<0>(stable_keys[item]);
        ranks[item] = rocprim::get<1>(stable_keys[item]);
    }

    // Load the values with the already sorted indices
    block_load_values_impl().load(
        flat_id,
        ranks,
        block_offset,
        valid_in_last_block,
        is_incomplete_block,
        values_input,
        values,
        storage.load_values
    );

    block_store_impl().store(
        block_offset,
        valid_in_last_block,
        is_incomplete_block,
        keys_output,
        values_output,
        keys,
        values,
        storage.store
    );
}

template<
    unsigned int BlockSize,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetT,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_merge_kernel_impl(KeysInputIterator keys_input,
                             KeysOutputIterator keys_output,
                             ValuesInputIterator values_input,
                             ValuesOutputIterator values_output,
                             const OffsetT input_size,
                             const unsigned int block_size,
                             BinaryFunction compare_function)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    unsigned int id = (flat_block_id * BlockSize) + flat_id;

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

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_HPP_
