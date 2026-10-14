// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_RANK_MATCH_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_RANK_MATCH_HPP_

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../block_scan.hpp"
#include "../config.hpp"
#include "rocprim/intrinsics/arch.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<unsigned int            BlockSizeX,
         unsigned int            RadixBits,
         unsigned int            BlockSizeY     = 1,
         unsigned int            BlockSizeZ     = 1,
         block_padding_hint      PaddingHint    = block_padding_hint::avoid_conflicts,
         arch::wavefront::target TargetWaveSize = arch::wavefront::get_target()>
class block_radix_rank_match
{
    using digit_counter_type = unsigned int;

    using block_scan_type = ::rocprim::block_scan<digit_counter_type,
                                                  BlockSizeX,
                                                  ::rocprim::block_scan_algorithm::using_warp_scan,
                                                  BlockSizeY,
                                                  BlockSizeZ,
                                                  TargetWaveSize>;

    static constexpr unsigned int block_size   = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int radix_digits = 1 << RadixBits;

    struct unpadded_config
    {
        // min size is used because we allocate based on the number of warps
        static constexpr unsigned int warps = ::rocprim::detail::ceiling_div(
            block_size, arch::wavefront::size_from_target<TargetWaveSize>());
    };

    struct padded_config
    {
        static constexpr unsigned int warps = unpadded_config::warps | 1u;
    };

    template<typename Config>
    struct build_config : Config
    {
        static constexpr unsigned int active_counters = Config::warps * radix_digits;
        static constexpr unsigned int counters_per_thread
            = ::rocprim::detail::ceiling_div(active_counters, block_size);
        static constexpr unsigned int counters = counters_per_thread * block_size;

        // Compute local data share and theorethical occupancy
        static constexpr size_t       lds_size  = max(sizeof(digit_counter_type) * counters,
                                               sizeof(typename block_scan_type::storage_type));
        static constexpr unsigned int occupancy = detail::get_min_lds_size() / lds_size;
    };

    using config = detail::select_block_padding_config<PaddingHint,
                                                       build_config<padded_config>,
                                                       build_config<unpadded_config>>;

    static constexpr unsigned int warps = config::warps;
    // The number of counters that are actively being used.
    static constexpr unsigned int active_counters = config::active_counters;
    // We want to use a regular block scan to scan the per-warp counters. This requires the
    // total number of counters to be divisible by the block size. To facilitate this, just add
    // a bunch of counters that are not otherwise used.
    static constexpr unsigned int counters_per_thread = config::counters_per_thread;
    // The total number of counters, factoring in the unused ones for the block scan.
    static constexpr unsigned int counters = config::counters;

public:
    constexpr static unsigned int digits_per_thread
        = ::rocprim::detail::ceiling_div(radix_digits, block_size);

private:
    struct storage_type_
    {
        typename block_scan_type::storage_type block_scan;
        digit_counter_type                     counters[counters];
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int get_digit_counter(const unsigned int digit, const unsigned int warp)
    {
        return digit * warps + warp;
    }

    template<typename Key, unsigned int ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys_impl(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type_& storage,
                                       DigitExtractor digit_extractor)
    {
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
        const unsigned int warp_id = ::rocprim::warp_id();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < counters_per_thread; ++i)
        {
            storage.counters[flat_id * counters_per_thread + i] = 0;
        }

        ::rocprim::syncthreads();

        digit_counter_type* digit_counters[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            // Get the digit for this key.
            const unsigned int digit = digit_extractor(keys[i]);

            // Get the digit counter for this key on the current warp.
            digit_counters[i] = &storage.counters[get_digit_counter(digit, warp_id)];

            // Read the prefix sum of that digit. We already know it's 0 on the first iteration. So
            // we can skip a read-after-write dependency. The conditional gets optimized out due to
            // loop unrolling.
            const digit_counter_type warp_digit_prefix
                = i == 0 ? digit_counter_type(0) : *digit_counters[i];

            // Construct a mask of threads in this wave which have the same digit.
            ::rocprim::lane_mask_type peer_mask = ::rocprim::match_any<RadixBits>(digit);

            ::rocprim::wave_barrier();

            // The total number of threads in the warp which also have this digit.
            const unsigned int digit_count = rocprim::bit_count(peer_mask);
            // The number of threads in the warp that have the same digit AND whose lane id is lower
            // than the current thread's.
            const unsigned int peer_digit_prefix = rocprim::masked_bit_count(peer_mask);

            if(::rocprim::group_elect(peer_mask))
            {
                *digit_counters[i] = warp_digit_prefix + digit_count;
            }

            ::rocprim::wave_barrier();

            // Compute the warp-local rank.
            ranks[i] = warp_digit_prefix + peer_digit_prefix;
        }

        ::rocprim::syncthreads();

        // Scan the per-warp counters to get a rank-offset per warp counter.
        digit_counter_type scan_counters[counters_per_thread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < counters_per_thread; ++i)
        {
            scan_counters[i] = storage.counters[flat_id * counters_per_thread + i];
        }

        block_scan_type().exclusive_scan(scan_counters, scan_counters, 0, storage.block_scan);

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < counters_per_thread; ++i)
        {
            storage.counters[flat_id * counters_per_thread + i] = scan_counters[i];
        }

        ::rocprim::syncthreads();

        // Add the per-warp rank counter to get the final rank.
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            ranks[i] += *digit_counters[i];
        }
    }

    template<bool Descending, typename Key, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_impl(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type_&     storage,
                                       const unsigned int begin_bit,
                                       const unsigned int pass_bits)
    {
        using key_codec
            = decltype(::rocprim::traits::get<Key>().template radix_key_codec<Descending>());
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
                // The counter for warp 0 holds the prefix of all the digits at this point.
                prefix[i] = storage.counters[get_digit_counter(digit, 0)];
                // To find the count, subtract the prefix of the next digit with that of the
                // current digit.
                const unsigned int next_prefix
                    = digit + 1 == radix_digits ? block_size * ItemsPerThread
                                                : storage.counters[get_digit_counter(digit + 1, 0)];
                counts[i]                      = next_prefix - prefix[i];
            }
        }
    }

public:
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = ::rocprim::detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

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
