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

#include "block_exchange.hpp"
#include "block_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

/// Allows to reduce and scan multimple sequences simultaneously
template<class T, unsigned int Size>
struct buckets
{
    static constexpr unsigned int size = Size;
    T xs[Size];

    buckets() [[hc]]
    {
        for(unsigned int r = 0; r < Size; r++)
        {
            xs[r] = T(0);
        }
    }

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

/// Specialized warp scan and reduce functions of bool (1 bit values)
/// They have much better performance (several times faster) than generic scan and reduce classes
/// because of using hardware ability to calculate which lanes have true predicate values.

template<class T>
void warp_bit_plus_exlusive_scan(const T input, T& output) [[hc]]
{
    output = ::rocprim::masked_bit_count(::rocprim::ballot(input));
}

template<class T, unsigned int Size>
void warp_bit_plus_exlusive_scan(const buckets<T, Size>& input, buckets<T, Size>& output) [[hc]]
{
    for(unsigned int r = 0; r < Size; r++)
    {
        warp_bit_plus_exlusive_scan(input[r], output[r]);
    }
}

template<class T>
void warp_bit_plus_reduce(const T input, T& output) [[hc]]
{
    output = ::rocprim::bit_count(::rocprim::ballot(input));
}

template<class T, unsigned int Size>
void warp_bit_plus_reduce(const buckets<T, Size>& input, buckets<T, Size>& output) [[hc]]
{
    for(unsigned int r = 0; r < Size; r++)
    {
        warp_bit_plus_reduce(input[r], output[r]);
    }
}

/// Specialized block scan of bool (1 bit values)
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

    // typedef of warp_scan primtive that will be used to get prefix values for
    // each warp (scanned carry-outs from warps before it)
    // warp_scan_shuffle is an implementation of warp_scan that does not need storage,
    // but requires logical warp size to be a power of two.
    using warp_scan_prefix_type = ::rocprim::detail::warp_scan_shuffle<T, detail::next_power_of_two(warps_no)>;

public:

    struct storage_type
    {
        T warp_prefixes[warps_no];
        // ---------- Shared memory optimisation ----------
        // Since we use warp_scan_shuffle for warp scan, we don't need to allocate
        // any temporaty memory for it.
    };

    void exclusive_scan(const T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        storage_type& storage) [[hc]]
    {
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();

        T warp_reduction;
        warp_bit_plus_reduce(input[0], warp_reduction);
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            T r;
            warp_bit_plus_reduce(input[i], r);
            warp_reduction = warp_reduction + r;
        }
        if(lane_id == 0)
        {
            storage.warp_prefixes[warp_id] = warp_reduction;
        }
        ::rocprim::syncthreads();

        // Scan the warp reduction results to calculate warp prefixes
        if(warp_id == 0)
        {
            if(lane_id < warps_no)
            {
                T prefix = storage.warp_prefixes[lane_id];
                warp_scan_prefix_type().inclusive_scan(prefix, prefix, ::rocprim::plus<T>());
                storage.warp_prefixes[lane_id] = prefix;
            }
        }
        ::rocprim::syncthreads();

        // Perform exclusive warp scan of bit values
        T lane_prefix;
        warp_bit_plus_exlusive_scan(input[0], lane_prefix);
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            T s;
            warp_bit_plus_exlusive_scan(input[i], s);
            lane_prefix = lane_prefix + s;
        }

        // Scan the lane's items and calculate final scan results
        output[0] = warp_id == 0
            ? lane_prefix
            : lane_prefix + storage.warp_prefixes[warp_id - 1];
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            output[i] = output[i - 1] + input[i - 1];
        }

        // Get the final inclusive reduction result
        reduction = storage.warp_prefixes[warps_no - 1];
    }
};

} // end namespace detail

template<
    class Key,
    unsigned int BlockSize,
    unsigned int ItemsPerThread = 1,
    class Value = detail::empty_type,
    unsigned int RadixBits = 1
>
class block_radix_sort
{
    static constexpr bool with_values = !std::is_same<Value, detail::empty_type>::value;
    static constexpr unsigned int radix_size = 1 << RadixBits;

    using key_codec = ::rocprim::detail::radix_key_codec<Key>;
    using bit_key_type = typename key_codec::bit_key_type;

    // The last radix value does not have its own bucket and hence no scan is performed, because
    // its value can be calculated based on all other values
    using buckets = detail::buckets<unsigned int, radix_size - 1>;
    using block_scan = detail::block_bit_plus_scan<buckets, BlockSize, ItemsPerThread>;

    using bit_keys_exchange_type = ::rocprim::block_exchange<bit_key_type, BlockSize, ItemsPerThread>;
    using values_exchange_type = ::rocprim::block_exchange<Value, BlockSize, ItemsPerThread>;

public:

    struct storage_type
    {
        union
        {
            typename bit_keys_exchange_type::storage_type bit_keys_exchange;
            typename values_exchange_type::storage_type values_exchange;
        };
        typename block_scan::storage_type block_scan;
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
        detail::empty_type * values = nullptr;
        sort_impl<false>(keys, values, storage, begin_bit, end_bit);
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
        detail::empty_type * values = nullptr;
        sort_impl<true>(keys, values, storage, begin_bit, end_bit);
    }

    template<bool WithValues = with_values>
    void sort(Key (&keys)[ItemsPerThread],
              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        tile_static storage_type storage;
        sort(keys, values, storage, begin_bit, end_bit);
    }

    template<bool WithValues = with_values>
    void sort(Key (&keys)[ItemsPerThread],
              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
              storage_type& storage,
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        sort_impl<false>(keys, values, storage, begin_bit, end_bit);
    }

    template<bool WithValues = with_values>
    void sort_desc(Key (&keys)[ItemsPerThread],
                   typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        tile_static storage_type storage;
        sort_desc(keys, values, storage, begin_bit, end_bit);
    }

    template<bool WithValues = with_values>
    void sort_desc(Key (&keys)[ItemsPerThread],
                   typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        sort_impl<true>(keys, values, storage, begin_bit, end_bit);
    }

    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        tile_static storage_type storage;
        sort_to_striped(keys, storage, begin_bit, end_bit);
    }

    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         storage_type& storage,
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        detail::empty_type * values = nullptr;
        sort_impl<false, true>(keys, values, storage, begin_bit, end_bit);
    }

    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        tile_static storage_type storage;
        sort_desc_to_striped(keys, storage, begin_bit, end_bit);
    }

    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              storage_type& storage,
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        detail::empty_type * values = nullptr;
        sort_impl<true, true>(keys, values, storage, begin_bit, end_bit);
    }

    template<bool WithValues = with_values>
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        tile_static storage_type storage;
        sort_to_striped(keys, values, storage, begin_bit, end_bit);
    }

    template<bool WithValues = with_values>
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                         storage_type& storage,
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        sort_impl<false, true>(keys, values, storage, begin_bit, end_bit);
    }

    template<bool WithValues = with_values>
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        tile_static storage_type storage;
        sort_desc_to_striped(keys, values, storage, begin_bit, end_bit);
    }

    template<bool WithValues = with_values>
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                              storage_type& storage,
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key)) [[hc]]
    {
        sort_impl<true, true>(keys, values, storage, begin_bit, end_bit);
    }

private:

    template<bool Descending, bool ToStriped = false, class SortedValue>
    void sort_impl(Key (&keys)[ItemsPerThread],
                   SortedValue * values,
                   storage_type& storage,
                   unsigned int begin_bit,
                   unsigned int end_bit) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();

        bit_key_type bit_keys[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            bit_key_type bit_key = key_codec::encode(keys[i]);
            bit_key = (Descending ? ~bit_key : bit_key);
            bit_keys[i] = bit_key;
        }

        for(unsigned int bit = begin_bit; bit < end_bit; bit += RadixBits)
        {
            buckets banks[ItemsPerThread];
            buckets positions[ItemsPerThread];
            buckets counts;

            // Handle cases when (end_bit - bit) is not divisible by RadixBits, i.e. the last
            // iteration has a shorter mask.
            const unsigned int radix_mask = (1u << ::rocprim::min(RadixBits, end_bit - bit)) - 1;

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const unsigned int radix = (bit_keys[i] >> bit) & radix_mask;
                for(unsigned int r = 0; r < radix_size - 1; r++)
                {
                    banks[i][r] = radix == r;
                }
            }
            block_scan().exclusive_scan(banks, positions, counts, storage.block_scan);

            // Prefix sum of counts to compute starting positions of keys of each radix value
            buckets starts;
            unsigned int last_start = 0;
            for(unsigned int r = 0; r < radix_size - 1; r++)
            {
                const unsigned int c = counts[r];
                starts[r] = last_start;
                last_start += c;
            }
            // Scatter keys to computed positions considering starting positions of their
            // radix values
            unsigned int ranks[ItemsPerThread];
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                const unsigned int radix = (bit_keys[i] >> bit) & radix_mask;
                unsigned int to_position = 0;
                unsigned int last_position = 0;
                for(unsigned int r = 0; r < radix_size - 1; r++)
                {
                    to_position = radix == r ? (starts[r] + positions[i][r]) : to_position;
                    last_position += positions[i][r];
                }
                // Calculate position for the last radix value based on positions of
                // all other previous values
                const unsigned int from_position = flat_id * ItemsPerThread + i;
                ranks[i] = radix == radix_size - 1
                    ? (last_start + from_position - last_position)
                    : to_position;
            }
            exchange_keys(storage, bit_keys, ranks);
            exchange_values(storage, values, ranks);
        }

        if(ToStriped)
        {
            ::rocprim::syncthreads();
            to_striped_keys(storage, bit_keys);
            to_striped_values(storage, values);
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            bit_key_type bit_key = bit_keys[i];
            bit_key = (Descending ? ~bit_key : bit_key);
            keys[i] = key_codec::decode(bit_key);
        }
    }


    void exchange_keys(storage_type& storage,
                       bit_key_type (&bit_keys)[ItemsPerThread],
                       const unsigned int (&ranks)[ItemsPerThread]) [[hc]]
    {
        bit_keys_exchange_type().scatter_to_blocked(bit_keys, bit_keys, ranks, storage.bit_keys_exchange);
    }

    template<class SortedValue>
    void exchange_values(storage_type& storage,
                         SortedValue * values,
                         const unsigned int (&ranks)[ItemsPerThread]) [[hc]]
    {
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        SortedValue (&vs)[ItemsPerThread] = *reinterpret_cast<SortedValue (*)[ItemsPerThread]>(values);
        values_exchange_type().scatter_to_blocked(vs, vs, ranks, storage.values_exchange);
    }

    void exchange_values(storage_type& storage,
                         detail::empty_type * values,
                         const unsigned int (&ranks)[ItemsPerThread]) [[hc]]
    { }

    void to_striped_keys(storage_type& storage,
                         bit_key_type (&bit_keys)[ItemsPerThread]) [[hc]]
    {
        bit_keys_exchange_type().blocked_to_striped(bit_keys, bit_keys, storage.bit_keys_exchange);
    }

    template<class SortedValue>
    void to_striped_values(storage_type& storage,
                           SortedValue * values) [[hc]]
    {
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        SortedValue (&vs)[ItemsPerThread] = *reinterpret_cast<SortedValue (*)[ItemsPerThread]>(values);
        values_exchange_type().blocked_to_striped(vs, vs, storage.values_exchange);
    }

    void to_striped_values(storage_type& storage,
                           detail::empty_type * values) [[hc]]
    { }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
