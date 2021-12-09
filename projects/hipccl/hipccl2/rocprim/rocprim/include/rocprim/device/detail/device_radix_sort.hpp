// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_RADIX_SORT_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../detail/radix_sort.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_discontinuity.hpp"
#include "../../block/block_exchange.hpp"
#include "../../block/block_load.hpp"
#include "../../block/block_load_func.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_radix_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Wrapping functions that allow to call proper methods (with or without values)
// (a variant with values is enabled only when Value is not empty_type)
template<bool Descending = false, class SortType, class SortKey, class SortValue, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void sort_block(SortType sorter,
                SortKey (&keys)[ItemsPerThread],
                SortValue (&values)[ItemsPerThread],
                typename SortType::storage_type& storage,
                unsigned int begin_bit,
                unsigned int end_bit)
{
    if(Descending)
    {
        sorter.sort_desc(keys, values, storage, begin_bit, end_bit);
    }
    else
    {
        sorter.sort(keys, values, storage, begin_bit, end_bit);
    }
}

template<bool Descending = false, class SortType, class SortKey, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void sort_block(SortType sorter,
                SortKey (&keys)[ItemsPerThread],
                ::rocprim::empty_type (&values)[ItemsPerThread],
                typename SortType::storage_type& storage,
                unsigned int begin_bit,
                unsigned int end_bit)
{
    (void) values;
    if(Descending)
    {
        sorter.sort_desc(keys, storage, begin_bit, end_bit);
    }
    else
    {
        sorter.sort(keys, storage, begin_bit, end_bit);
    }
}

template<
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending
>
struct radix_digit_count_helper
{
    static constexpr unsigned int radix_size = 1 << RadixBits;

    static constexpr unsigned int warp_size = WarpSize;
    static constexpr unsigned int warps_no = BlockSize / warp_size;
    static_assert(BlockSize % ::rocprim::device_warp_size() == 0, "BlockSize must be divisible by warp size");
    static_assert(radix_size <= BlockSize, "Radix size must not exceed BlockSize");

    struct storage_type
    {
        unsigned int digit_counts[warps_no][radix_size];
    };

    template<bool IsFull = false, class KeysInputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void count_digits(KeysInputIterator keys_input,
                      unsigned int begin_offset,
                      unsigned int end_offset,
                      unsigned int bit,
                      unsigned int current_radix_bits,
                      storage_type& storage,
                      unsigned int& digit_count)  // i-th thread will get i-th digit's value
    {
        constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

        using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

        using key_codec = radix_key_codec<key_type, Descending>;
        using bit_key_type = typename key_codec::bit_key_type;

        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
        const unsigned int warp_id = ::rocprim::warp_id<0, 1, 1>();

        if(flat_id < radix_size)
        {
            for(unsigned int w = 0; w < warps_no; w++)
            {
                storage.digit_counts[w][flat_id] = 0;
            }
        }
        ::rocprim::syncthreads();

        for(unsigned int block_offset = begin_offset; block_offset < end_offset; block_offset += items_per_block)
        {
            key_type keys[ItemsPerThread];
            unsigned int valid_count;
            // Use loading into a striped arrangement because an order of items is irrelevant,
            // only totals matter
            if(IsFull || (block_offset + items_per_block <= end_offset))
            {
                valid_count = items_per_block;
                block_load_direct_striped<BlockSize>(flat_id, keys_input + block_offset, keys);
            }
            else
            {
                valid_count = end_offset - block_offset;
                block_load_direct_striped<BlockSize>(flat_id, keys_input + block_offset, keys, valid_count);
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const bit_key_type bit_key = key_codec::encode(keys[i]);
                const unsigned int digit = key_codec::extract_digit(bit_key, bit, current_radix_bits);
                const unsigned int pos = i * BlockSize + flat_id;
                lane_mask_type same_digit_lanes_mask = ::rocprim::ballot(IsFull || (pos < valid_count));
                for(unsigned int b = 0; b < RadixBits; b++)
                {
                    const unsigned int bit_set = digit & (1u << b);
                    const lane_mask_type bit_set_mask = ::rocprim::ballot(bit_set);
                    same_digit_lanes_mask &= (bit_set ? bit_set_mask : ~bit_set_mask);
                }
                const unsigned int same_digit_count = ::rocprim::bit_count(same_digit_lanes_mask);
                const unsigned int prev_same_digit_count = ::rocprim::masked_bit_count(same_digit_lanes_mask);
                if(prev_same_digit_count == 0)
                {
                    // Write the number of lanes having this digit,
                    // if the current lane is the first (and maybe only) lane with this digit.
                    storage.digit_counts[warp_id][digit] += same_digit_count;
                }
            }
        }
        ::rocprim::syncthreads();

        digit_count = 0;
        if(flat_id < radix_size)
        {
            for(unsigned int w = 0; w < warps_no; w++)
            {
                digit_count += storage.digit_counts[w][flat_id];
            }
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Descending,
    class Key,
    class Value
>
struct radix_sort_single_helper
{
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using key_type = Key;
    using value_type = Value;

    using key_codec = radix_key_codec<key_type, Descending>;
    using bit_key_type = typename key_codec::bit_key_type;
    using keys_load_type = ::rocprim::block_load<
        key_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose>;
    using values_load_type = ::rocprim::block_load<
        value_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose>;
    using sort_type = ::rocprim::block_radix_sort<key_type, BlockSize, ItemsPerThread, value_type>;

    static constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    struct storage_type
    {
        union
        {
            typename keys_load_type::storage_type keys_load;
            typename values_load_type::storage_type values_load;
            typename sort_type::storage_type sort;
        };
    };

    template<
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_single(KeysInputIterator keys_input,
                     KeysOutputIterator keys_output,
                     ValuesInputIterator values_input,
                     ValuesOutputIterator values_output,
                     unsigned int size,
                     unsigned int bit,
                     unsigned int current_radix_bits,
                     storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
        const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
        const unsigned int block_offset = flat_block_id * items_per_block;
        const unsigned int number_of_blocks = (size + items_per_block - 1) / items_per_block;
        unsigned int valid_in_last_block;
        const bool last_block = flat_block_id == (number_of_blocks - 1);

        using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

        using key_codec = radix_key_codec<key_type, Descending>;
        using bit_key_type = typename key_codec::bit_key_type;

        key_type keys[ItemsPerThread];
        value_type values[ItemsPerThread];
        if(!last_block)
        {
            valid_in_last_block = items_per_block;
            keys_load_type().load(keys_input + block_offset, keys, storage.keys_load);
            if(with_values)
            {
                ::rocprim::syncthreads();
                values_load_type().load(values_input + block_offset, values, storage.values_load);
            }
        }
        else
        {
            const key_type out_of_bounds = key_codec::decode(bit_key_type(-1));
            valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
            keys_load_type().load(keys_input + block_offset, keys, valid_in_last_block, out_of_bounds, storage.keys_load);
            if(with_values)
            {
                ::rocprim::syncthreads();
                values_load_type().load(values_input + block_offset, values, valid_in_last_block, storage.values_load);
            }
        }

        ::rocprim::syncthreads();

        sort_block<Descending>(sort_type(), keys, values, storage.sort, bit, bit + current_radix_bits);

        // Store keys and values
        #pragma unroll
        for (unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            unsigned int item_offset = flat_id * ItemsPerThread + i;
            if (item_offset < valid_in_last_block)
            {
                keys_output[block_offset + item_offset] = keys[i];
                if (with_values)
                    values_output[block_offset + item_offset] = values[i];
            }
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class Key,
    class Value
>
struct radix_sort_and_scatter_helper
{
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    static constexpr unsigned int radix_size = 1 << RadixBits;

    using key_type = Key;
    using value_type = Value;

    using key_codec = radix_key_codec<key_type, Descending>;
    using bit_key_type = typename key_codec::bit_key_type;
    using keys_load_type = ::rocprim::block_load<
        key_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose>;
    using values_load_type = ::rocprim::block_load<
        value_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose>;
    using sort_type = ::rocprim::block_radix_sort<key_type, BlockSize, ItemsPerThread, value_type>;
    using discontinuity_type = ::rocprim::block_discontinuity<unsigned int, BlockSize>;
    using bit_keys_exchange_type = ::rocprim::block_exchange<bit_key_type, BlockSize, ItemsPerThread>;
    using values_exchange_type = ::rocprim::block_exchange<value_type, BlockSize, ItemsPerThread>;

    static constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    struct storage_type
    {
        union
        {
            typename keys_load_type::storage_type keys_load;
            typename values_load_type::storage_type values_load;
            typename sort_type::storage_type sort;
            typename discontinuity_type::storage_type discontinuity;
            typename bit_keys_exchange_type::storage_type bit_keys_exchange;
            typename values_exchange_type::storage_type values_exchange;
        };

        unsigned short starts[radix_size];
        unsigned short ends[radix_size];

        unsigned int digit_starts[radix_size];
    };

    template<
        bool IsFull = false,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_and_scatter(KeysInputIterator keys_input,
                          KeysOutputIterator keys_output,
                          ValuesInputIterator values_input,
                          ValuesOutputIterator values_output,
                          unsigned int begin_offset,
                          unsigned int end_offset,
                          unsigned int bit,
                          unsigned int current_radix_bits,
                          unsigned int digit_start, // i-th thread must pass i-th digit's value
                          storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();

        if(flat_id < radix_size)
        {
            storage.digit_starts[flat_id] = digit_start;
        }

        for(unsigned int block_offset = begin_offset; block_offset < end_offset; block_offset += items_per_block)
        {
            key_type keys[ItemsPerThread];
            value_type values[ItemsPerThread];
            unsigned int valid_count;
            if(IsFull || (block_offset + items_per_block <= end_offset))
            {
                valid_count = items_per_block;
                keys_load_type().load(keys_input + block_offset, keys, storage.keys_load);
                if(with_values)
                {
                    ::rocprim::syncthreads();
                    values_load_type().load(values_input + block_offset, values, storage.values_load);
                }
            }
            else
            {
                valid_count = end_offset - block_offset;
                // Sort will leave "invalid" (out of size) items at the end of the sorted sequence
                const key_type out_of_bounds = key_codec::decode(bit_key_type(-1));
                keys_load_type().load(keys_input + block_offset, keys, valid_count, out_of_bounds, storage.keys_load);
                if(with_values)
                {
                    ::rocprim::syncthreads();
                    values_load_type().load(values_input + block_offset, values, valid_count, storage.values_load);
                }
            }

            if(flat_id < radix_size)
            {
                storage.starts[flat_id] = valid_count;
                storage.ends[flat_id] = valid_count;
            }

            ::rocprim::syncthreads();
            sort_block<Descending>(sort_type(), keys, values, storage.sort, bit, bit + current_radix_bits);

            bit_key_type bit_keys[ItemsPerThread];
            unsigned int digits[ItemsPerThread];
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                bit_keys[i] = key_codec::encode(keys[i]);
                digits[i] = key_codec::extract_digit(bit_keys[i], bit, current_radix_bits);
            }

            bool head_flags[ItemsPerThread];
            bool tail_flags[ItemsPerThread];
            ::rocprim::not_equal_to<unsigned int> flag_op;

            ::rocprim::syncthreads();
            discontinuity_type().flag_heads_and_tails(head_flags, tail_flags, digits, flag_op, storage.discontinuity);

            // Fill start and end position of subsequence for every digit
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const unsigned int digit = digits[i];
                const unsigned int pos = flat_id * ItemsPerThread + i;
                if(head_flags[i])
                {
                    storage.starts[digit] = pos;
                }
                if(tail_flags[i])
                {
                    storage.ends[digit] = pos;
                }
            }

            ::rocprim::syncthreads();
            // Rearrange to striped arrangement to have faster coalesced writes instead of
            // scattering of blocked-arranged items
            bit_keys_exchange_type().blocked_to_striped(bit_keys, bit_keys, storage.bit_keys_exchange);
            if(with_values)
            {
                ::rocprim::syncthreads();
                values_exchange_type().blocked_to_striped(values, values, storage.values_exchange);
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const unsigned int digit = key_codec::extract_digit(bit_keys[i], bit, current_radix_bits);
                const unsigned int pos = i * BlockSize + flat_id;
                if(IsFull || (pos < valid_count))
                {
                    const unsigned int dst = pos - storage.starts[digit] + storage.digit_starts[digit];
                    keys_output[dst] = key_codec::decode(bit_keys[i]);
                    if(with_values)
                    {
                        values_output[dst] = values[i];
                    }
                }
            }

            ::rocprim::syncthreads();

            // Accumulate counts of the current block
            if(flat_id < radix_size)
            {
                const unsigned int digit = flat_id;
                const unsigned int start = storage.starts[digit];
                const unsigned int end = storage.ends[digit];
                if(start < valid_count)
                {
                    storage.digit_starts[digit] += (::rocprim::min(valid_count - 1, end) - start + 1);
                }
            }
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void fill_digit_counts(KeysInputIterator keys_input,
                       unsigned int size,
                       unsigned int * batch_digit_counts,
                       unsigned int bit,
                       unsigned int current_radix_bits,
                       unsigned int blocks_per_full_batch,
                       unsigned int full_batches)
{
    constexpr unsigned int radix_size = 1 << RadixBits;
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using count_helper_type = radix_digit_count_helper<::rocprim::device_warp_size(), BlockSize, ItemsPerThread, RadixBits, Descending>;

    ROCPRIM_SHARED_MEMORY typename count_helper_type::storage_type storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int batch_id = ::rocprim::detail::block_id<0>();

    unsigned int block_offset;
    unsigned int blocks_per_batch;
    if(batch_id < full_batches)
    {
        blocks_per_batch = blocks_per_full_batch;
        block_offset = batch_id * blocks_per_batch;
    }
    else
    {
        blocks_per_batch = blocks_per_full_batch - 1;
        block_offset = batch_id * blocks_per_batch + full_batches;
    }
    block_offset *= items_per_block;

    unsigned int digit_count;
    if(batch_id < ::rocprim::detail::grid_size<0>() - 1)
    {
        count_helper_type().template count_digits<true>(
            keys_input,
            block_offset, block_offset + blocks_per_batch * items_per_block,
            bit, current_radix_bits,
            storage,
            digit_count
        );
    }
    else
    {
        count_helper_type().template count_digits<false>(
            keys_input,
            block_offset, size,
            bit, current_radix_bits,
            storage,
            digit_count
        );
    }

    if(flat_id < radix_size)
    {
        batch_digit_counts[batch_id * radix_size + flat_id] = digit_count;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void scan_batches(unsigned int * batch_digit_counts,
                  unsigned int * digit_counts,
                  unsigned int batches)
{
    constexpr unsigned int radix_size = 1 << RadixBits;

    using scan_type = typename ::rocprim::block_scan<unsigned int, BlockSize>;

    const unsigned int digit = ::rocprim::detail::block_id<0>();
    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();

    unsigned int values[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        const unsigned int batch_id = flat_id * ItemsPerThread + i;
        values[i] = (batch_id < batches ? batch_digit_counts[batch_id * radix_size + digit] : 0);
    }

    unsigned int digit_count;
    scan_type().exclusive_scan(values, values, 0, digit_count);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        const unsigned int batch_id = flat_id * ItemsPerThread + i;
        if(batch_id < batches)
        {
            batch_digit_counts[batch_id * radix_size + digit] = values[i];
        }
    }

    if(flat_id == 0)
    {
        digit_counts[digit] = digit_count;
    }
}

template<unsigned int RadixBits>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void scan_digits(unsigned int * digit_counts)
{
    constexpr unsigned int radix_size = 1 << RadixBits;

    using scan_type = typename ::rocprim::block_scan<unsigned int, radix_size>;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();

    unsigned int value = digit_counts[flat_id];
    scan_type().exclusive_scan(value, value, 0);
    digit_counts[flat_id] = value;
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void sort_single(KeysInputIterator keys_input,
                 KeysOutputIterator keys_output,
                 ValuesInputIterator values_input,
                 ValuesOutputIterator values_output,
                 unsigned int size,
                 unsigned int bit,
                 unsigned int current_radix_bits)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using sort_single_helper = radix_sort_single_helper<
        BlockSize, ItemsPerThread, Descending,
        key_type, value_type
    >;

    ROCPRIM_SHARED_MEMORY typename sort_single_helper::storage_type storage;

    sort_single_helper().template sort_single(
        keys_input, keys_output, values_input, values_output,
        size, bit, current_radix_bits,
        storage
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void sort_and_scatter(KeysInputIterator keys_input,
                      KeysOutputIterator keys_output,
                      ValuesInputIterator values_input,
                      ValuesOutputIterator values_output,
                      unsigned int size,
                      const unsigned int * batch_digit_starts,
                      const unsigned int * digit_starts,
                      unsigned int bit,
                      unsigned int current_radix_bits,
                      unsigned int blocks_per_full_batch,
                      unsigned int full_batches)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    constexpr unsigned int radix_size = 1 << RadixBits;

    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using sort_and_scatter_helper = radix_sort_and_scatter_helper<
        BlockSize, ItemsPerThread, RadixBits, Descending,
        key_type, value_type
    >;

    ROCPRIM_SHARED_MEMORY typename sort_and_scatter_helper::storage_type storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int batch_id = ::rocprim::detail::block_id<0>();

    unsigned int block_offset;
    unsigned int blocks_per_batch;
    if(batch_id < full_batches)
    {
        blocks_per_batch = blocks_per_full_batch;
        block_offset = batch_id * blocks_per_batch;
    }
    else
    {
        blocks_per_batch = blocks_per_full_batch - 1;
        block_offset = batch_id * blocks_per_batch + full_batches;
    }
    block_offset *= items_per_block;

    unsigned int digit_start = 0;
    if(flat_id < radix_size)
    {
        digit_start = digit_starts[flat_id] + batch_digit_starts[batch_id * radix_size + flat_id];
    }

    if(batch_id < ::rocprim::detail::grid_size<0>() - 1)
    {
        sort_and_scatter_helper().template sort_and_scatter<true>(
            keys_input, keys_output, values_input, values_output,
            block_offset, block_offset + blocks_per_batch * items_per_block,
            bit, current_radix_bits,
            digit_start,
            storage
        );
    }
    else
    {
        sort_and_scatter_helper().template sort_and_scatter<false>(
            keys_input, keys_output, values_input, values_output,
            block_offset, size,
            bit, current_radix_bits,
            digit_start,
            storage
        );
    }
}

template<
    bool WithValues,
    class KeysInputIterator,
    class ValuesInputIterator,
    class Key,
    class Value,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<!WithValues>::type
block_load_radix_impl(const unsigned int flat_id,
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
        block_load_direct_blocked(
            flat_id,
            keys_input + block_offset,
            keys,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_blocked(
            flat_id,
            keys_input + block_offset,
            keys
        );
    }
}

template<
    bool WithValues,
    class KeysInputIterator,
    class ValuesInputIterator,
    class Key,
    class Value,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<WithValues>::type
block_load_radix_impl(const unsigned int flat_id,
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
        block_load_direct_blocked(
            flat_id,
            keys_input + block_offset,
            keys,
            valid_in_last_block
        );

        block_load_direct_blocked(
            flat_id,
            values_input + block_offset,
            values,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_blocked(
            flat_id,
            keys_input + block_offset,
            keys
        );

        block_load_direct_blocked(
            flat_id,
            values_input + block_offset,
            values
        );
    }
}

template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto compare_nans(const T& a, const T& b)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, bool>::type
{
    return (a != a) && (b == b);
}

template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto compare_nans(const T&, const T&)
    -> typename std::enable_if<!rocprim::is_floating_point<T>::value, bool>::type
{
    return false;
}

template<
    bool Descending,
    bool UseRadixMask,
    class T,
    class Enable = void
>
struct radix_merge_compare;

template<class T>
struct radix_merge_compare<false, false, T>
{
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const T& a, const T& b) const
    {
        return compare_nans<T>(b, a) || b > a;
    }
};

template<class T>
struct radix_merge_compare<true, false, T>
{
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const T& a, const T& b) const
    {
        return compare_nans<T>(a, b) || a > b;
    }
};

template<class T>
struct radix_merge_compare<false, true, T, typename std::enable_if<rocprim::is_integral<T>::value>::type>
{
    T radix_mask;

    radix_merge_compare(const unsigned int start_bit, const unsigned int current_radix_bits)
    {
        T radix_mask_upper  = (T(1) << (current_radix_bits + start_bit)) - 1;
        T radix_mask_bottom = (T(1) << start_bit) - 1;
        radix_mask = radix_mask_upper ^ radix_mask_bottom;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const T& a, const T& b) const
    {
        const T masked_key_a  = a & radix_mask;
        const T masked_key_b  = b & radix_mask;
        return masked_key_b > masked_key_a;
    }
};

template<class T>
struct radix_merge_compare<true, true, T, typename std::enable_if<rocprim::is_integral<T>::value>::type>
{
    T radix_mask;

    radix_merge_compare(const unsigned int start_bit, const unsigned int current_radix_bits)
    {
        T radix_mask_upper  = (T(1) << (current_radix_bits + start_bit)) - 1;
        T radix_mask_bottom = (T(1) << start_bit) - 1;
        radix_mask = (radix_mask_upper ^ radix_mask_bottom);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const T& a, const T& b) const
    {
        const T masked_key_a  = a & radix_mask;
        const T masked_key_b  = b & radix_mask;
        return masked_key_a > masked_key_b;
    }
};

template<bool Descending, class T>
struct radix_merge_compare<Descending, true, T, typename std::enable_if<rocprim::is_floating_point<T>::value>::type>
{
    // radix_merge_compare supports masks only for integrals.
    // even though masks are never used for floating point-types,
    // it needs to be able to compile.
    radix_merge_compare(const unsigned int, const unsigned int){}

    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const T&, const T&) const { return false; }
};

template<>
struct radix_merge_compare<false, false, rocprim::half>
{
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        return (__hisnan(b) && !__hisnan(a)) || __hgt(b, a);
    }
};

template<>
struct radix_merge_compare<true, false, rocprim::half>
{
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        return (!__hisnan(b) && __hisnan(a)) || __hgt(a, b);
    }
};

template<>
struct radix_merge_compare<false, true, rocprim::half>
{
    using key_codec = radix_key_codec<rocprim::half, true>;
    using bit_key_type = typename key_codec::bit_key_type;

    unsigned int bit, length;

    radix_merge_compare(const unsigned int bit, const unsigned int current_radix_bits)
    {
        this->bit = bit;
        this->length = current_radix_bits;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        const bit_key_type encoded_key_a = key_codec::encode(a);
        const bit_key_type masked_key_a  = key_codec::extract_digit(encoded_key_a, bit, length);

        const bit_key_type encoded_key_b = key_codec::encode(b);
        const bit_key_type masked_key_b  = key_codec::extract_digit(encoded_key_b, bit, length);

        return __hgt(key_codec::decode(masked_key_b), key_codec::decode(masked_key_a));
    }
};

template<>
struct radix_merge_compare<true, true, rocprim::half>
{
    using key_codec = radix_key_codec<rocprim::half, true>;
    using bit_key_type = typename key_codec::bit_key_type;

    unsigned int bit, length;

    radix_merge_compare(const unsigned int bit, const unsigned int current_radix_bits)
    {
        this->bit = bit;
        this->length = current_radix_bits;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        const bit_key_type encoded_key_a = key_codec::encode(a);
        const bit_key_type masked_key_a  = key_codec::extract_digit(encoded_key_a, bit, length);

        const bit_key_type encoded_key_b = key_codec::encode(b);
        const bit_key_type masked_key_b  = key_codec::extract_digit(encoded_key_b, bit, length);

        return __hgt(key_codec::decode(masked_key_a), key_codec::decode(masked_key_b));
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void radix_block_merge_impl(KeysInputIterator keys_input,
                            KeysOutputIterator keys_output,
                            ValuesInputIterator values_input,
                            ValuesOutputIterator values_output,
                            const size_t input_size,
                            const unsigned int merge_items_per_block_size,
                            BinaryFunction compare_function)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_block;
    const unsigned int number_of_blocks = (input_size + items_per_block - 1) / items_per_block;
    const bool last_block = flat_block_id == (number_of_blocks - 1);
    auto valid_in_last_block = last_block ? input_size - items_per_block * (number_of_blocks - 1) : items_per_block;

    unsigned int start_id = (flat_block_id * items_per_block) + flat_id * ItemsPerThread;
    if (start_id >= input_size)
    {
        return;
    }


    key_type keys[ItemsPerThread];
    value_type values[ItemsPerThread];

    block_load_radix_impl<with_values>(
        flat_id,
        block_offset,
        valid_in_last_block,
        last_block,
        keys_input,
        values_input,
        keys,
        values
    );

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        if( flat_id * ItemsPerThread + i < valid_in_last_block )
        {
            const unsigned int id = start_id + i;
            const unsigned int block_id = id / merge_items_per_block_size;
            const bool block_id_is_odd = block_id & 1;
            const unsigned int next_block_id = block_id_is_odd ? block_id - 1 :
                                                                 block_id + 1;
            const unsigned int block_start = min(block_id * merge_items_per_block_size, (unsigned int) input_size);
            const unsigned int next_block_start = min(next_block_id * merge_items_per_block_size, (unsigned int) input_size);
            const unsigned int next_block_end = min((next_block_id + 1) * merge_items_per_block_size, (unsigned int) input_size);

            if(next_block_start == input_size)
            {
                keys_output[id] = keys[i];
                if(with_values)
                {
                    values_output[id] = values[i];
                }
            }

            unsigned int left_id = next_block_start;
            unsigned int right_id = next_block_end;

            while(left_id < right_id)
            {
                unsigned int mid_id = (left_id + right_id) / 2;
                key_type mid_key = keys_input[mid_id];
                bool smaller = compare_function(mid_key, keys[i]);
                left_id = smaller ? mid_id + 1 : left_id;
                right_id = smaller ? right_id : mid_id;
            }


            right_id = next_block_end;
            if(block_id_is_odd && left_id != right_id)
            {
                key_type upper_key = keys_input[left_id];
                while(!compare_function(upper_key, keys[i]) &&
                      !compare_function(keys[i], upper_key) &&
                      left_id < right_id)
                {
                    unsigned int mid_id = (left_id + right_id) / 2;
                    key_type mid_key = keys_input[mid_id];
                    bool equal = !compare_function(mid_key, keys[i]) &&
                                 !compare_function(keys[i], mid_key);
                    left_id = equal ? mid_id + 1 : left_id + 1;
                    right_id = equal ? right_id : mid_id;
                    upper_key = keys_input[left_id];
                }
            }

            unsigned int offset = 0;
            offset += id - block_start;
            offset += left_id - next_block_start;
            offset += min(block_start, next_block_start);

            keys_output[offset] = keys[i];
            if(with_values)
            {
                values_output[offset] = values[i];
            }
        }
    }
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_RADIX_SORT_HPP_
