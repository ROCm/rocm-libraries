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
#include "../../block/block_load_func.hpp"
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

    template<class KeysOutputIterator, class ValuesOutputIterator, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE void store(const OffsetT      block_offset,
                                             const unsigned int valid_in_last_block,
                                             const bool         is_incomplete_block,
                                             KeysOutputIterator keys_output,
                                             ValuesOutputIterator /*values_output*/,
                                             Key (&keys)[ItemsPerThread],
                                             Value (&/*values*/)[ItemsPerThread],
                                             storage_type& storage)
    {
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

template<unsigned int BlockSize, unsigned int ItemsPerThread, block_sort_algorithm Algo, class Key>
struct block_sort_impl
{
    using stable_key_type = rocprim::tuple<Key, unsigned int>;
    using block_sort_type = ::rocprim::
        block_sort<stable_key_type, BlockSize, ItemsPerThread, rocprim::empty_type, Algo>;

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

template<unsigned int         BlockSize,
         unsigned int         ItemsPerThread,
         block_sort_algorithm Algo,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetT,
         class BinaryFunction,
         class ValueType = typename std::iterator_traits<ValuesInputIterator>::value_type>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE auto block_sort_kernel_impl(KeysInputIterator    keys_input,
                                                                KeysOutputIterator   keys_output,
                                                                ValuesInputIterator  values_input,
                                                                ValuesOutputIterator values_output,
                                                                const OffsetT        input_size,
                                                                BinaryFunction compare_function)
    -> std::enable_if_t<(!std::is_trivially_copyable<ValueType>::value
                         || rocprim::is_floating_point<ValueType>::value
                         || std::is_integral<ValueType>::value),
                        void>
{
    using key_type             = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type           = typename std::iterator_traits<ValuesInputIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    const unsigned int     flat_id         = block_thread_id<0>();
    const unsigned int     flat_block_id   = block_id<0>();
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const OffsetT      block_offset        = flat_block_id * items_per_block;
    const unsigned int valid_in_last_block = input_size - block_offset;
    const bool         is_incomplete_block = flat_block_id == (input_size / items_per_block);

    key_type keys[ItemsPerThread];

    using block_load_keys_impl = block_load_keys_impl<BlockSize, ItemsPerThread, key_type>;
    using block_sort_impl      = block_sort_impl<BlockSize, ItemsPerThread, Algo, key_type>;
    using block_load_values_impl
        = block_load_values_impl<with_values, BlockSize, ItemsPerThread, value_type>;
    using block_store_impl
        = block_store_impl<with_values, BlockSize, ItemsPerThread, key_type, value_type>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_keys_impl::storage_type   load_keys;
        typename block_sort_impl::storage_type        sort;
        typename block_load_values_impl::storage_type load_values;
        typename block_store_impl::storage_type       store;
    } storage;

    block_load_keys_impl().load(block_offset,
                                valid_in_last_block,
                                is_incomplete_block,
                                keys_input,
                                keys,
                                storage.load_keys);

    using stable_key_type = typename block_sort_impl::stable_key_type;

    // Special comparison that preserves relative order of equal keys
    auto stable_compare_function
        = [compare_function](const stable_key_type& a, const stable_key_type& b) mutable -> bool
    {
        const bool ab = compare_function(rocprim::get<0>(a), rocprim::get<0>(b));
        return ab
               || (!compare_function(rocprim::get<0>(b), rocprim::get<0>(a))
                   && (rocprim::get<1>(a) < rocprim::get<1>(b)));
    };

    stable_key_type stable_keys[ItemsPerThread];
    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        stable_keys[item] = rocprim::make_tuple(keys[item], ItemsPerThread * flat_id + item);
    }

    // Synchronize before reusing shared memory
    ::rocprim::syncthreads();

    block_sort_impl().sort(stable_keys,
                           storage.sort,
                           valid_in_last_block,
                           is_incomplete_block,
                           stable_compare_function);

    unsigned int ranks[ItemsPerThread];

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        keys[item]  = rocprim::get<0>(stable_keys[item]);
        ranks[item] = rocprim::get<1>(stable_keys[item]);
    }

    value_type values[ItemsPerThread];
    // Load the values with the already sorted indices
    block_load_values_impl().load(flat_id,
                                  ranks,
                                  block_offset,
                                  valid_in_last_block,
                                  is_incomplete_block,
                                  values_input,
                                  values,
                                  storage.load_values);

    block_store_impl().store(block_offset,
                             valid_in_last_block,
                             is_incomplete_block,
                             keys_output,
                             values_output,
                             keys,
                             values,
                             storage.store);
}

// The specialization below exists because the compiler creates slow code for
// ValueTypes with misaligned datastructures in them (e.g. custom_char_double)
// when storing/loading those ValueTypes to/from registers.
// Thus this is a temporary workaround.
template<unsigned int         BlockSize,
         unsigned int         ItemsPerThread,
         block_sort_algorithm Algo,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetT,
         class BinaryFunction,
         class ValueType = typename std::iterator_traits<ValuesInputIterator>::value_type>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE auto block_sort_kernel_impl(KeysInputIterator    keys_input,
                                                                KeysOutputIterator   keys_output,
                                                                ValuesInputIterator  values_input,
                                                                ValuesOutputIterator values_output,
                                                                const OffsetT        input_size,
                                                                BinaryFunction compare_function)
    -> std::enable_if_t<(std::is_trivially_copyable<ValueType>::value
                         && !rocprim::is_floating_point<ValueType>::value
                         && !std::is_integral<ValueType>::value),
                        void>
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

    using block_load_keys_impl = block_load_keys_impl<BlockSize, ItemsPerThread, key_type>;
    using block_sort_impl      = block_sort_impl<BlockSize, ItemsPerThread, Algo, key_type>;
    using block_store_impl
        = block_store_impl<false, BlockSize, ItemsPerThread, key_type, rocprim::empty_type>;

    using values_storage_ = value_type[items_per_block];
    ROCPRIM_SHARED_MEMORY union {
        typename block_load_keys_impl::storage_type   load_keys;
        typename block_sort_impl::storage_type        sort;
        detail::raw_storage<values_storage_>          load_values;
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
        return ab
               || (!compare_function(rocprim::get<0>(b), rocprim::get<0>(a))
                   && (rocprim::get<1>(a) < rocprim::get<1>(b)));
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

    rocprim::empty_type values[ItemsPerThread];
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

    if ROCPRIM_IF_CONSTEXPR(with_values)
    {
        ::rocprim::syncthreads();
        auto& values_shared = storage.load_values.get();
        if(is_incomplete_block)
        {
            ROCPRIM_UNROLL
            for(unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                const unsigned int idx = BlockSize * item + flat_id;
                if(idx < valid_in_last_block)
                {
                    values_shared[idx] = values_input[block_offset + idx];
                }
            }
        }
        else
        {
            ROCPRIM_UNROLL
            for(unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                const unsigned int idx = BlockSize * item + flat_id;
                values_shared[idx]     = values_input[block_offset + idx];
            }
        }

        // Synchronize before reusing shared memory
        ::rocprim::syncthreads();

        const OffsetT thread_offset = block_offset + ItemsPerThread * flat_id;
        if(is_incomplete_block)
        {
            ROCPRIM_UNROLL
            for(unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                if(flat_id * ItemsPerThread + item < valid_in_last_block)
                {
                    values_output[thread_offset + item] = values_shared[ranks[item]];
                }
            }
        }
        else
        {
            ROCPRIM_UNROLL
            for(unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                values_output[thread_offset + item] = values_shared[ranks[item]];
            }
        }

        rocprim::syncthreads();
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetT,
         class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE void block_merge_oddeven_kernel(KeysInputIterator    keys_input,
                                                              KeysOutputIterator   keys_output,
                                                              ValuesInputIterator  values_input,
                                                              ValuesOutputIterator values_output,
                                                              const OffsetT        input_size,
                                                              const OffsetT  sorted_block_size,
                                                              BinaryFunction compare_function)
{
    using key_type             = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type           = typename std::iterator_traits<ValuesInputIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    constexpr unsigned int items_per_block     = BlockSize * ItemsPerThread;
    const unsigned int     flat_id             = ::rocprim::detail::block_thread_id<0>();
    const unsigned int     flat_block_id       = ::rocprim::detail::block_id<0>();
    const bool             is_incomplete_block = flat_block_id == (input_size / items_per_block);
    // ^ bounds-checking: if input_size is not a multiple of items_per_block and
    // this is the last block: true, false otherwise
    const OffsetT block_offset        = flat_block_id * items_per_block;
    const OffsetT valid_in_last_block = input_size - block_offset;

    const OffsetT thread_offset = flat_id * ItemsPerThread;
    if(thread_offset >= valid_in_last_block)
    {
        return;
    }

    key_type   keys[ItemsPerThread];
    value_type values[ItemsPerThread];

    if(is_incomplete_block)
    {
        block_load_direct_blocked(flat_id, keys_input + block_offset, keys, valid_in_last_block);

        if ROCPRIM_IF_CONSTEXPR(with_values)
        {
            block_load_direct_blocked(flat_id,
                                      values_input + block_offset,
                                      values,
                                      valid_in_last_block);
        }
    }
    else
    {
        block_load_direct_blocked(flat_id, keys_input + block_offset, keys);
        if ROCPRIM_IF_CONSTEXPR(with_values)
        {
            block_load_direct_blocked(flat_id, values_input + block_offset, values);
        }
    }

    const unsigned int merged_tiles_number = sorted_block_size / items_per_block;
    const unsigned int mask                = merged_tiles_number - 1;
    // tilegroup_id is the id of the input sorted_block
    const unsigned int tilegroup_id = ~mask & flat_block_id;
    const unsigned int block_is_odd = merged_tiles_number & tilegroup_id;
    const OffsetT      block_start  = tilegroup_id * items_per_block;
    const OffsetT      next_block_start_
        = block_is_odd ? block_start - sorted_block_size : block_start + sorted_block_size;
    const OffsetT next_block_start = min(next_block_start_, input_size);
    const OffsetT next_block_end   = min(next_block_start + sorted_block_size, input_size);

    if(next_block_start == input_size)
    {
        // In this case, no merging needs to happen and
        // block_is_odd will always be false here
        if(is_incomplete_block)
        {
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const unsigned int id = block_offset + thread_offset + i;
                if(id < input_size)
                {
                    keys_output[id] = keys[i];
                    if ROCPRIM_IF_CONSTEXPR(with_values)
                    {
                        values_output[id] = values[i];
                    }
                }
            }
        }
        else
        {
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const unsigned int id = block_offset + thread_offset + i;
                keys_output[id]       = keys[i];
                if ROCPRIM_IF_CONSTEXPR(with_values)
                {
                    values_output[id] = values[i];
                }
            }
        }
        return;
    }

    OffsetT left_id = next_block_start;

    const OffsetT dest_offset
        = min(block_start, next_block_start) + block_offset + thread_offset - block_start
          - next_block_start; // Destination offset (base+source+partial target calculation)

    const auto merge_function = [&](const unsigned int i)
    {
        OffsetT right_id = next_block_end;

        while(left_id < right_id)
        {
            OffsetT    mid_id      = (left_id + right_id) / 2;
            key_type   mid_key     = keys_input[mid_id];
            const bool mid_smaller = block_is_odd ? !compare_function(keys[i], mid_key)
                                                  : compare_function(mid_key, keys[i]);
            left_id                = mid_smaller ? mid_id + 1 : left_id;
            right_id               = mid_smaller ? right_id : mid_id;
        }

        OffsetT offset      = dest_offset + i + left_id; // Destination offset (target calculation)
        keys_output[offset] = keys[i];
        if ROCPRIM_IF_CONSTEXPR(with_values)
        {
            values_output[offset] = values[i];
        }
    };

    if(is_incomplete_block)
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(thread_offset + i < valid_in_last_block)
            {
                merge_function(i);
            }
        }
    }
    else
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            merge_function(i);
        }
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_HPP_
