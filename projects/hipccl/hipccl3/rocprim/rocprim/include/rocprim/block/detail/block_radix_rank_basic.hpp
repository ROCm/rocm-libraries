// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_RANK_BASIC_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_RANK_BASIC_HPP_

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../functional.hpp"

#include "../../detail/radix_sort.hpp"

#include "../block_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<unsigned int BlockSizeX,
         unsigned int RadixBits,
         bool         MemoizeOuterScan = false,
         unsigned int BlockSizeY       = 1,
         unsigned int BlockSizeZ       = 1>
class block_radix_rank
{
    using digit_counter_type  = unsigned short;
    using packed_counter_type = unsigned int;

    using block_scan_type = ::rocprim::block_scan<packed_counter_type,
                                                  BlockSizeX,
                                                  ::rocprim::block_scan_algorithm::using_warp_scan,
                                                  BlockSizeY,
                                                  BlockSizeZ>;

    static constexpr unsigned int block_size   = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int radix_digits = 1 << RadixBits;
    static constexpr unsigned int packing_ratio
        = sizeof(packed_counter_type) / sizeof(digit_counter_type);
    static constexpr unsigned int column_size = radix_digits / packing_ratio;

public:
    static constexpr unsigned int digits_per_thread
        = ::rocprim::detail::ceiling_div(radix_digits, block_size);

private:
    struct storage_type_
    {
        union
        {
            digit_counter_type  digit_counters[block_size * radix_digits];
            packed_counter_type packed_counters[block_size * column_size];
        };

        typename block_scan_type::storage_type block_scan;
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE digit_counter_type& get_digit_counter(const unsigned int digit,
                                                                        const unsigned int thread,
                                                                        storage_type_&     storage)
    {
        const unsigned int column_counter = digit % column_size;
        const unsigned int sub_counter    = digit / column_size;
        const unsigned int counter
            = (column_counter * block_size + thread) * packing_ratio + sub_counter;
        return storage.digit_counters[counter];
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE void reset_counters(const unsigned int flat_id,
                                                      storage_type_&     storage)
    {
        for(unsigned int i = flat_id; i < block_size * column_size; i += block_size)
        {
            storage.packed_counters[i] = 0;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void
        scan_block_counters(storage_type_& storage, packed_counter_type* const packed_counters)
    {
        packed_counter_type block_reduction = 0;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < column_size; ++i)
        {
            block_reduction += packed_counters[i];
        }

        packed_counter_type exclusive_prefix = 0;
        packed_counter_type reduction;
        block_scan_type().exclusive_scan(block_reduction,
                                         exclusive_prefix,
                                         0,
                                         reduction,
                                         storage.block_scan);

        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < packing_ratio; i <<= 1)
        {
            exclusive_prefix += reduction << (sizeof(digit_counter_type) * 8 * i);
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < column_size; ++i)
        {
            packed_counter_type counter = packed_counters[i];
            packed_counters[i]          = exclusive_prefix;
            exclusive_prefix += counter;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void scan_counters(const unsigned int flat_id,
                                                     storage_type_&     storage)
    {
        packed_counter_type* const shared_counters
            = &storage.packed_counters[flat_id * column_size];

        if ROCPRIM_IF_CONSTEXPR(MemoizeOuterScan)
        {
            packed_counter_type local_counters[column_size];
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < column_size; ++i)
            {
                local_counters[i] = shared_counters[i];
            }

            scan_block_counters(storage, local_counters);

            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < column_size; ++i)
            {
                shared_counters[i] = local_counters[i];
            }
        }
        else
        {
            scan_block_counters(storage, shared_counters);
        }
    }

    template<typename Key, unsigned int ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys_impl(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type_& storage,
                                       DigitExtractor digit_extractor)
    {
        static_assert(block_size * ItemsPerThread < 1u << 16,
                      "The maximum amout of items that block_radix_rank can rank is 2**16.");
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        reset_counters(flat_id, storage);

        digit_counter_type  thread_prefixes[ItemsPerThread];
        digit_counter_type* digit_counters[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            const unsigned int digit = digit_extractor(keys[i]);
            digit_counters[i]        = &get_digit_counter(digit, flat_id, storage);
            thread_prefixes[i]       = (*digit_counters[i])++;
        }

        ::rocprim::syncthreads();

        scan_counters(flat_id, storage);

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            ranks[i] = thread_prefixes[i] + *digit_counters[i];
        }
    }

    template<bool Descending, typename Key, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_impl(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type_&     storage,
                                       const unsigned int begin_bit,
                                       const unsigned int pass_bits)
    {
        using key_codec    = ::rocprim::detail::radix_key_codec<Key, Descending>;
        using bit_key_type = typename key_codec::bit_key_type;

        bit_key_type bit_keys[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            bit_keys[i] = key_codec::encode(keys[i]);
        }

        rank_keys_impl(bit_keys,
                       ranks,
                       storage,
                       [begin_bit, pass_bits](const bit_key_type& key)
                       { return key_codec::extract_digit(key, begin_bit, pass_bits); });
    }

    template<unsigned int ItemsPerThread>
    ROCPRIM_DEVICE void digit_prefix_count(unsigned int (&prefix)[digits_per_thread],
                                           unsigned int (&counts)[digits_per_thread],
                                           storage_type_& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < digits_per_thread; ++i)
        {
            const unsigned int digit = flat_id * digits_per_thread + i;
            if(radix_digits % block_size == 0 || digit < radix_digits)
            {
                // The counter for thread 0 holds the prefix of all the digits at this point.
                prefix[i] = get_digit_counter(digit, 0, storage);
                // To find the count, subtract the prefix of the next digit with that of the
                // current digit.
                const unsigned int next_prefix = digit + 1 == radix_digits
                                                     ? block_size * ItemsPerThread
                                                     : get_digit_counter(digit + 1, 0, storage);
                counts[i]                      = next_prefix - prefix[i];
            }
        }
    }

public:
    using storage_type = ::rocprim::detail::raw_storage<storage_type_>;

    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  storage_type& storage,
                                  unsigned int  begin_bit = 0,
                                  unsigned int  pass_bits = RadixBits)
    {
        rank_keys_impl<false>(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_desc(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type& storage,
                                       unsigned int  begin_bit = 0,
                                       unsigned int  pass_bits = RadixBits)
    {
        rank_keys_impl<true>(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    template<typename Key, unsigned ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  storage_type&  storage,
                                  DigitExtractor digit_extractor)
    {
        rank_keys_impl(keys, ranks, storage.get(), digit_extractor);
    }

    template<typename Key, unsigned ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys_desc(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type&  storage,
                                       DigitExtractor digit_extractor)
    {
        rank_keys_impl(keys,
                       ranks,
                       storage.get(),
                       [&digit_extractor](const Key& key)
                       {
                           const unsigned int digit = digit_extractor(key);
                           return radix_digits - 1 - digit;
                       });
    }

    template<typename Key, unsigned ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  storage_type&  storage,
                                  DigitExtractor digit_extractor,
                                  unsigned int (&prefix)[digits_per_thread],
                                  unsigned int (&counts)[digits_per_thread])
    {
        rank_keys(keys, ranks, storage, digit_extractor);
        digit_prefix_count<ItemsPerThread>(prefix, counts, storage.get());
    }
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif
