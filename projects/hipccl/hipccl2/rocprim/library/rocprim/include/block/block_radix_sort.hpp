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
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
#define ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"
#include "../detail/radix_sort.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "block_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T, unsigned int Size>
struct buckets
{
    static constexpr unsigned int size = Size;
    T xs[Size];

    buckets operator+(const buckets& b) const [[hc]]
    {
        buckets c;
        for(unsigned int r = 0; r < Size; r++)
        {
            c[r] = xs[r] + b[r];
        }
        return c;
    }

    T& operator[](unsigned int idx) [[hc]]
    {
        return xs[idx];
    }

    T operator[](unsigned int idx) const [[hc]]
    {
        return xs[idx];
    }
};

template<class T>
void warp_bit_plus_exlusive_scan(const T input, T& output) [[hc]]
{
    const unsigned int lane_id = ::rocprim::lane_id();
    const unsigned long long prev_lanes_mask = (1ull << lane_id) - 1;
    output = hc::__popcount_u32_b64(hc::__ballot(input) & prev_lanes_mask);
}

template<class T, unsigned int Size>
void warp_bit_plus_exlusive_scan(const buckets<T, Size>& input, buckets<T, Size>& output) [[hc]]
{
    for(unsigned int r = 0; r < Size; r++)
    {
        warp_bit_plus_exlusive_scan(input[r], output[r]);
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_bit_plus_scan
{
    // Select warp size
    static constexpr unsigned int warp_size =
        detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // Number of warps in block
    static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;

    static constexpr unsigned int prefix_scan_size =
        detail::next_power_of_two(warps_no * ItemsPerThread);
    // TODO use block_scan for such cases?
    static_assert(prefix_scan_size <= ::rocprim::warp_size(),
        "ItemsPerThread is too large for the current BlockSize");

    using prefix_scan = ::rocprim::warp_scan<T, prefix_scan_size>;

public:

    struct storage_type
    {
        T scan_results[warps_no * ItemsPerThread];
        typename prefix_scan::storage_type prefix_scan;
    };

    void exclusive_scan(const T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction) [[hc]]
    {
        tile_static storage_type storage;
        return exclusive_scan(input, output, reduction, storage);
    }

    void exclusive_scan(const T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        storage_type& storage) [[hc]]
    {
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int current_warp_size = (warp_id == warps_no - 1)
            ? (BlockSize % warp_size > 0 ? BlockSize % warp_size : warp_size)
            : warp_size;

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            // Perform exclusive warp scan of bit values
            warp_bit_plus_exlusive_scan(input[i], output[i]);

            // Save the warp reduction result, that is the scan result
            // for last element in each warp including its value
            if(lane_id == current_warp_size - 1)
            {
                const unsigned int prefix_id = warp_id * ItemsPerThread + i;
                storage.scan_results[prefix_id] = output[i] + input[i];
            }
        }
        ::rocprim::syncthreads();

        // Scan the warp reduction results
        if(warp_id == 0)
        {
            const unsigned int prefix_id = lane_id;
            // TODO what about small BlockSize and large ItemsPerThread? Not enough active lanes
            // to scan all values
            if(prefix_id < warps_no * ItemsPerThread)
            {
                auto prefix = storage.scan_results[prefix_id];
                ::rocprim::plus<T> plus;
                prefix_scan().inclusive_scan(prefix, prefix, storage.prefix_scan, plus);
                storage.scan_results[prefix_id] = prefix;
            }
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            // Calculate the final scan result for every thread
            const unsigned int prefix_id = warp_id * ItemsPerThread + i;
            if(prefix_id != 0)
            {
                auto prefix = storage.scan_results[prefix_id - 1];
                output[i] = prefix + output[i];
            }
        }

        // Get the final inclusive reduction result
        reduction = storage.scan_results[warps_no * ItemsPerThread - 1];
    }
};

} // end namespace detail

template<
    class Key,
    unsigned int BlockSize,
    unsigned int ItemsPerThread = 1,
    class Value = detail::empty_type,
    unsigned int RadixBits = 2
>
class block_radix_sort
{
    static constexpr unsigned int radix_size = 1 << RadixBits;

    using key_codec = ::rocprim::detail::radix_key_codec<Key>;
    using bit_key_type = typename key_codec::bit_key_type;

    using buckets = detail::buckets<unsigned int, radix_size>;
    using block_bit_plus_scan = detail::block_bit_plus_scan<buckets, BlockSize, ItemsPerThread>;

    // Select warp size
    static constexpr unsigned int warp_size =
        detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // Number of warps in block
    static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;

public:

    struct storage_type
    {
        bit_key_type bit_keys[BlockSize * ItemsPerThread];
        typename block_bit_plus_scan::storage_type block_bit_plus_scan;
    };

    void sort(Key (&keys)[ItemsPerThread],
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        tile_static storage_type storage;
        sort(keys, storage, begin_bit, end_bit);
    }

    void sort(Key (&keys)[ItemsPerThread],
              storage_type& storage,
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        sort<false>(keys, storage, begin_bit, end_bit);
    }

    void sort_desc(Key (&keys)[ItemsPerThread],
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        tile_static storage_type storage;
        sort_desc(keys, storage, begin_bit, end_bit);
    }

    void sort_desc(Key (&keys)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        sort<true>(keys, storage, begin_bit, end_bit);
    }

private:

    template<bool Descending>
    void sort(Key (&keys)[ItemsPerThread],
              storage_type& storage,
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        const unsigned int thread_id = ::rocprim::flat_block_thread_id();
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const bit_key_type bit_key = key_codec::encode(keys[i]);
            storage.bit_keys[thread_id * ItemsPerThread + i] = (Descending ? ~bit_key : bit_key);
        }
        ::rocprim::syncthreads();

        block_bit_plus_scan scan;

        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int current_warp_size = (warp_id == warps_no - 1)
            ? (BlockSize % warp_size > 0 ? BlockSize % warp_size : warp_size)
            : warp_size;

        for(unsigned int bit = begin_bit; bit < end_bit; bit += RadixBits)
        {
            bit_key_type bit_keys[ItemsPerThread];
            buckets banks[ItemsPerThread];
            buckets positions[ItemsPerThread];
            buckets counts;

            // Handle cases when (end_bit - bit) is not divisible by RadixBits, i.e. the last
            // iteration has a shorter mask.
            const unsigned int radix_mask = (1u << ::rocprim::min(RadixBits, end_bit - bit)) - 1;

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const bit_key_type bit_key = storage.bit_keys[
                    warp_id * warp_size * ItemsPerThread + i * current_warp_size + lane_id
                ];
                bit_keys[i] = bit_key;
                const unsigned int radix = (bit_key >> bit) & radix_mask;
                for(unsigned int r = 0; r < radix_size; r++)
                {
                    banks[i][r] = radix == r;
                }
            }
            scan.exclusive_scan(banks, positions, counts, storage.block_bit_plus_scan);

            // Prefix sum of counts to compute starting positions of keys of each radix value
            unsigned int prefix = 0;
            for(unsigned int r = 0; r < radix_size; r++)
            {
                const unsigned int c = counts[r];
                counts[r] = prefix;
                prefix += c;
            }
            // Scatter keys to computed positions considering starting positions of their
            // radix values
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const bit_key_type bit_key = bit_keys[i];
                const unsigned int radix = (bit_key >> bit) & radix_mask;
                unsigned int position = 0;
                for(unsigned int r = 0; r < radix_size; r++)
                {
                    position = radix == r ? (counts[r] + positions[i][r]) : position;
                }
                storage.bit_keys[position] = bit_key;
            }
            ::rocprim::syncthreads();
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const bit_key_type bit_key = storage.bit_keys[thread_id * ItemsPerThread + i];
            keys[i] = key_codec::decode(Descending ? ~bit_key : bit_key);
        }
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
