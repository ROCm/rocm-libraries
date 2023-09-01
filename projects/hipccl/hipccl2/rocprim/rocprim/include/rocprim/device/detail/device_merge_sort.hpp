// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

template<typename Value,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         typename Enable = void>
struct block_permute_values_impl
{
    using values_exchange_type = block_exchange<Value, BlockSize, ItemsPerThread>;

    using values_store_type
        = block_store<Value, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;

    union storage_type
    {
        typename values_exchange_type::storage_type exchange;
        typename values_store_type::storage_type    store;
    };

    template<typename ValuesInputIterator, typename ValuesOutputIterator>
    ROCPRIM_DEVICE void permute(unsigned int (&ranks)[ItemsPerThread],
                                ValuesInputIterator  values_input,
                                ValuesOutputIterator values_output,
                                storage_type&        storage)
    {
        syncthreads();
        const auto flat_id = block_thread_id<0>();
        Value      values[ItemsPerThread];
        block_load_direct_striped<BlockSize>(flat_id, values_input, values);
        values_exchange_type().gather_from_striped(values, values, ranks, storage.exchange);
        syncthreads();
        values_store_type().store(values_output, values, storage.store);
    }

    template<typename ValuesOutputIterator, typename ValuesInputIterator>
    ROCPRIM_DEVICE void permute(unsigned int (&ranks)[ItemsPerThread],
                                ValuesInputIterator  values_input,
                                ValuesOutputIterator values_output,
                                const unsigned int   valid_in_last_block,
                                storage_type&        storage)
    {
        syncthreads();
        const auto flat_id = block_thread_id<0>();
        Value      values[ItemsPerThread];
        block_load_direct_striped<BlockSize>(flat_id, values_input, values, valid_in_last_block);
        values_exchange_type().gather_from_striped(values, values, ranks, storage.exchange);
        syncthreads();
        values_store_type().store(values_output, values, valid_in_last_block, storage.store);
    }
};

template<unsigned int BlockSize, unsigned int ItemsPerThread>
struct block_permute_values_impl<rocprim::empty_type, BlockSize, ItemsPerThread>
{
    using storage_type = empty_storage_type;

    template<typename ValuesInputIterator, typename ValuesOutputIterator>
    ROCPRIM_DEVICE void permute(unsigned int (&ranks)[ItemsPerThread],
                                ValuesInputIterator  values_input,
                                ValuesOutputIterator values_output,
                                storage_type&        storage)
    {
        (void)ranks;
        (void)values_input;
        (void)values_output;
        (void)storage;
    }

    template<typename ValuesOutputIterator, typename ValuesInputIterator>
    ROCPRIM_DEVICE void permute(unsigned int (&ranks)[ItemsPerThread],
                                ValuesInputIterator  values_input,
                                ValuesOutputIterator values_output,
                                const unsigned int   valid_in_last_block,
                                storage_type&        storage)
    {
        (void)ranks;
        (void)values_input;
        (void)values_output;
        (void)valid_in_last_block;
        (void)storage;
    }
};

// The specialization below exists because the compiler creates slow code for
// ValueTypes with misaligned datastructures in them (e.g. custom_char_double)
// when storing/loading those ValueTypes to/from registers.
// Thus this is a temporary workaround.
// TODO: Check if also the case for small types like this.
template<typename Value, unsigned int BlockSize, unsigned int ItemsPerThread>
struct block_permute_values_impl<Value,
                                 BlockSize,
                                 ItemsPerThread,
                                 std::enable_if_t<(std::is_trivially_copyable<Value>::value
                                                   && !rocprim::is_floating_point<Value>::value
                                                   && !std::is_integral<Value>::value)>>
{
    static constexpr unsigned int items_per_block = ItemsPerThread * BlockSize;

    struct storage_type_
    {
        Value values[items_per_block];
    };

    using storage_type = raw_storage<storage_type_>;

    template<typename ValuesInputIterator, typename ValuesOutputIterator>
    ROCPRIM_DEVICE void permute(unsigned int (&ranks)[ItemsPerThread],
                                ValuesInputIterator  values_input,
                                ValuesOutputIterator values_output,
                                storage_type&        storage_)
    {
        syncthreads();
        auto&      values_shared = storage_.get().values;
        const auto flat_id       = block_thread_id<0>();

        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            const unsigned int idx = BlockSize * item + flat_id;
            values_shared[idx]     = values_input[idx];
        }

        syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            values_output[ItemsPerThread * flat_id + item] = values_shared[ranks[item]];
        }
    }

    template<typename ValuesOutputIterator, typename ValuesInputIterator>
    ROCPRIM_DEVICE void permute(unsigned int (&ranks)[ItemsPerThread],
                                ValuesInputIterator  values_input,
                                ValuesOutputIterator values_output,
                                const unsigned int   valid_in_last_block,
                                storage_type&        storage_)
    {
        syncthreads();
        auto&      values_shared = storage_.get().values;
        const auto flat_id       = block_thread_id<0>();

        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            const unsigned int idx = BlockSize * item + flat_id;
            if(idx < valid_in_last_block)
            {
                values_shared[idx] = values_input[idx];
            }
        }

        syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            if(flat_id * ItemsPerThread + item < valid_in_last_block)
            {
                values_output[ItemsPerThread * flat_id + item] = values_shared[ranks[item]];
            }
        }
    }
};

template<typename Key,
         typename Value,
         unsigned int         BlockSize,
         unsigned int         ItemsPerThread,
         block_sort_algorithm Algo,
         typename Enable = void>
struct block_sort_impl
{
    using stable_key_type = rocprim::tuple<Key, unsigned int>;

    using keys_load_type
        = block_load<Key, BlockSize, ItemsPerThread, block_load_method::block_load_transpose>;

    using sort_type
        = block_sort<stable_key_type, BlockSize, ItemsPerThread, rocprim::empty_type, Algo>;

    using keys_store_type
        = block_store<Key, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;

    using values_permute_type = block_permute_values_impl<Value, BlockSize, ItemsPerThread>;

    union storage_type
    {
        typename keys_load_type::storage_type      load_keys;
        typename sort_type::storage_type           sort;
        typename keys_store_type::storage_type     store_keys;
        typename values_permute_type::storage_type permute_values;
    };

    template<typename KeysInputIterator,
             typename KeysOutputIterator,
             typename ValuesInputIterator,
             typename ValuesOutputIterator,
             typename BinaryFunction>
    ROCPRIM_DEVICE  ROCPRIM_FORCE_INLINE
    void sort(const unsigned int   valid_in_last_block,
              const bool           is_incomplete_block,
              KeysInputIterator    keys_input,
              KeysOutputIterator   keys_output,
              ValuesInputIterator  values_input,
              ValuesOutputIterator values_output,
              BinaryFunction       compare_function,
              storage_type&        storage)
    {
        // By default, the block sort algorithm is not stable. We can make it stable
        // by adding an index to each key.

        Key keys[ItemsPerThread];

        if(is_incomplete_block)
        {
            keys_load_type().load(keys_input, keys, valid_in_last_block, storage.load_keys);
        }
        else
        {
            keys_load_type().load(keys_input, keys, storage.load_keys);
        }

        const auto flat_id = block_thread_id<0>();

        stable_key_type stable_keys[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            stable_keys[i] = rocprim::make_tuple(keys[i], flat_id * ItemsPerThread + i);
        }

        syncthreads();

        // Special compare function that enforces sorting is stable.
        auto stable_compare_function
            = [compare_function](const stable_key_type& a,
                                 const stable_key_type& b) ROCPRIM_FORCE_INLINE mutable
        {
            const bool ab = compare_function(rocprim::get<0>(a), rocprim::get<0>(b));
            return ab
                   || (!compare_function(rocprim::get<0>(b), rocprim::get<0>(a))
                       && (rocprim::get<1>(a) < rocprim::get<1>(b)));
        };

        if(is_incomplete_block)
        {
            // Special compare function that enforces sorting is stable, and that out-of-bounds elements
            // are not compared.
            auto stable_oob_compare_function
                = [stable_compare_function, valid_in_last_block](const stable_key_type& a,
                                                                 const stable_key_type& b) mutable
            {
                const bool a_oob = rocprim::get<1>(a) >= valid_in_last_block;
                const bool b_oob = rocprim::get<1>(b) >= valid_in_last_block;
                return a_oob || b_oob ? !a_oob : stable_compare_function(a, b);
            };

            // Note: rocprim::block_sort with an algorithm that is not stable_merge_sort does not implement sorting
            // a misaligned amount of items.
            sort_type().sort(stable_keys, storage.sort, stable_oob_compare_function);

            unsigned int ranks[ItemsPerThread];
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                keys[i]  = rocprim::get<0>(stable_keys[i]);
                ranks[i] = rocprim::get<1>(stable_keys[i]);
            }

            syncthreads();
            keys_store_type().store(keys_output, keys, valid_in_last_block, storage.store_keys);
            values_permute_type().permute(ranks,
                                          values_input,
                                          values_output,
                                          valid_in_last_block,
                                          storage.permute_values);
        }
        else
        {
            sort_type().sort(stable_keys, storage.sort, stable_compare_function);

            unsigned int ranks[ItemsPerThread];
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                keys[i]  = rocprim::get<0>(stable_keys[i]);
                ranks[i] = rocprim::get<1>(stable_keys[i]);
            }

            syncthreads();
            keys_store_type().store(keys_output, keys, storage.store_keys);
            values_permute_type().permute(ranks,
                                          values_input,
                                          values_output,
                                          storage.permute_values);
        }
    }
};

template<typename Key, unsigned int BlockSize, unsigned int ItemsPerThread>
struct block_sort_impl<Key,
                       rocprim::empty_type,
                       BlockSize,
                       ItemsPerThread,
                       block_sort_algorithm::stable_merge_sort>
{
    using keys_load_type
        = block_load<Key, BlockSize, ItemsPerThread, block_load_method::block_load_transpose>;

    using sort_type = block_sort<Key,
                                 BlockSize,
                                 ItemsPerThread,
                                 rocprim::empty_type,
                                 block_sort_algorithm::stable_merge_sort>;

    using keys_store_type
        = block_store<Key, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;

    union storage_type
    {
        typename keys_load_type::storage_type  load_keys;
        typename sort_type::storage_type       sort;
        typename keys_store_type::storage_type store_keys;
    };

    template<typename KeysInputIterator,
             typename KeysOutputIterator,
             typename ValuesInputIterator,
             typename ValuesOutputIterator,
             typename BinaryFunction>
    ROCPRIM_DEVICE void sort(unsigned int       valid_in_last_block,
                             const bool         is_incomplete_block,
                             KeysInputIterator  keys_input,
                             KeysOutputIterator keys_output,
                             ValuesInputIterator /*values_input*/,
                             ValuesOutputIterator /*values_output*/,
                             BinaryFunction compare_function,
                             storage_type&  storage)
    {
        Key keys[ItemsPerThread];

        if(is_incomplete_block)
        {
            keys_load_type().load(keys_input, keys, valid_in_last_block, storage.load_keys);
            syncthreads();
            sort_type().sort(keys, storage.sort, valid_in_last_block, compare_function);
            syncthreads();
            keys_store_type().store(keys_output, keys, valid_in_last_block, storage.store_keys);
        }
        else
        {
            keys_load_type().load(keys_input, keys, storage.load_keys);
            syncthreads();
            sort_type().sort(keys, storage.sort, compare_function);
            syncthreads();
            keys_store_type().store(keys_output, keys, storage.store_keys);
        }
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename Key, typename Value, unsigned int BlockSize, unsigned int ItemsPerThread>
struct block_sort_impl<Key,
                       Value,
                       BlockSize,
                       ItemsPerThread,
                       block_sort_algorithm::stable_merge_sort,
                       std::enable_if_t<(sizeof(Value) <= sizeof(int))>>
{
    using keys_load_type
        = block_load<Key, BlockSize, ItemsPerThread, block_load_method::block_load_transpose>;

    using values_load_type
        = block_load<Value, BlockSize, ItemsPerThread, block_load_method::block_load_transpose>;

    using sort_type = block_sort<Key,
                                 BlockSize,
                                 ItemsPerThread,
                                 Value,
                                 block_sort_algorithm::stable_merge_sort>;

    using keys_store_type
        = block_store<Key, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;

    using values_store_type
        = block_store<Value, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;

    union storage_type
    {
        typename keys_load_type::storage_type    load_keys;
        typename values_load_type::storage_type  load_values;
        typename sort_type::storage_type         sort;
        typename keys_store_type::storage_type   store_keys;
        typename values_store_type::storage_type store_values;
    };

    template<typename KeysInputIterator,
             typename KeysOutputIterator,
             typename ValuesInputIterator,
             typename ValuesOutputIterator,
             typename BinaryFunction>
    ROCPRIM_DEVICE void sort(const unsigned int   valid_in_last_block,
                             const bool           is_incomplete_block,
                             KeysInputIterator    keys_input,
                             KeysOutputIterator   keys_output,
                             ValuesInputIterator  values_input,
                             ValuesOutputIterator values_output,
                             BinaryFunction       compare_function,
                             storage_type&        storage)
    {
        Key   keys[ItemsPerThread];
        Value values[ItemsPerThread];

        if(is_incomplete_block)
        {
            keys_load_type().load(keys_input, keys, valid_in_last_block, storage.load_keys);
            syncthreads();
            values_load_type().load(values_input, values, valid_in_last_block, storage.load_values);
            syncthreads();
            sort_type().sort(keys, values, storage.sort, valid_in_last_block, compare_function);
            syncthreads();
            keys_store_type().store(keys_output, keys, valid_in_last_block, storage.store_keys);
            syncthreads();
            values_store_type().store(values_output,
                                      values,
                                      valid_in_last_block,
                                      storage.store_values);
        }
        else
        {
            keys_load_type().load(keys_input, keys, storage.load_keys);
            syncthreads();
            values_load_type().load(values_input, values, storage.load_values);
            syncthreads();
            sort_type().sort(keys, values, storage.sort, compare_function);
            syncthreads();
            keys_store_type().store(keys_output, keys, storage.store_keys);
            syncthreads();
            values_store_type().store(values_output, values, storage.store_values);
        }
    }
};
template<typename Key, typename Value, unsigned int BlockSize, unsigned int ItemsPerThread>
struct block_sort_impl<Key,
                       Value,
                       BlockSize,
                       ItemsPerThread,
                       block_sort_algorithm::stable_merge_sort,
                       std::enable_if_t<(sizeof(Value) > sizeof(int))>>
{
    using keys_load_type
        = block_load<Key, BlockSize, ItemsPerThread, block_load_method::block_load_transpose>;

    using sort_type = block_sort<Key,
                                 BlockSize,
                                 ItemsPerThread,
                                 unsigned int,
                                 block_sort_algorithm::stable_merge_sort>;

    using keys_store_type
        = block_store<Key, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>;

    using values_permute_type = block_permute_values_impl<Value, BlockSize, ItemsPerThread>;

    union storage_type
    {
        typename keys_load_type::storage_type      load_keys;
        typename sort_type::storage_type           sort;
        typename keys_store_type::storage_type     store_keys;
        typename values_permute_type::storage_type permute_values;
    };

    template<typename KeysInputIterator,
             typename KeysOutputIterator,
             typename ValuesInputIterator,
             typename ValuesOutputIterator,
             typename BinaryFunction>
    ROCPRIM_DEVICE void sort(const unsigned int   valid_in_last_block,
                             const bool           is_incomplete_block,
                             KeysInputIterator    keys_input,
                             KeysOutputIterator   keys_output,
                             ValuesInputIterator  values_input,
                             ValuesOutputIterator values_output,
                             BinaryFunction       compare_function,
                             storage_type&        storage)
    {
        Key keys[ItemsPerThread];

        const auto   flat_id = block_thread_id<0>();
        unsigned int ranks[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            ranks[item] = flat_id * ItemsPerThread + item;
        }

        if(is_incomplete_block)
        {
            keys_load_type().load(keys_input, keys, valid_in_last_block, storage.load_keys);
            syncthreads();
            sort_type().sort(keys, ranks, storage.sort, valid_in_last_block, compare_function);
            syncthreads();
            keys_store_type().store(keys_output, keys, valid_in_last_block, storage.store_keys);
            values_permute_type().permute(ranks,
                                          values_input,
                                          values_output,
                                          valid_in_last_block,
                                          storage.permute_values);
        }
        else
        {
            keys_load_type().load(keys_input, keys, storage.load_keys);
            syncthreads();
            sort_type().sort(keys, ranks, storage.sort, compare_function);
            syncthreads();
            keys_store_type().store(keys_output, keys, storage.store_keys);
            values_permute_type().permute(ranks,
                                          values_input,
                                          values_output,
                                          storage.permute_values);
        }
    }
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

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
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    const unsigned int     flat_block_id   = block_id<0>();
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const OffsetT      block_offset        = flat_block_id * items_per_block;
    const unsigned int valid_in_last_block = input_size - block_offset;
    const bool         is_incomplete_block = flat_block_id == (input_size / items_per_block);

    using sort_impl = block_sort_impl<key_type, value_type, BlockSize, ItemsPerThread, Algo>;

    ROCPRIM_SHARED_MEMORY typename sort_impl::storage_type storage;

    sort_impl().sort(valid_in_last_block,
                     is_incomplete_block,
                     keys_input + block_offset,
                     keys_output + block_offset,
                     values_input + block_offset,
                     values_output + block_offset,
                     compare_function,
                     storage);
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
