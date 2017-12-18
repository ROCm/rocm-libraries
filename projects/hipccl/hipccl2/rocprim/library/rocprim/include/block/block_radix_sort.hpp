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
    const unsigned long long mask = hc::__ballot(input);
    int c;
    c = hc::__amdgcn_mbcnt_lo(static_cast<int>(mask), 0);
    c = hc::__amdgcn_mbcnt_hi(static_cast<int>(mask >> 32), c);
    output = c;
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
    output = hc::__activelanecount_u32_b1(input);
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
    using warp_scan_prefix_type = ::rocprim::warp_scan<T, detail::next_power_of_two(warps_no)>;

public:

    struct storage_type
    {
        T warp_prefixes[warps_no];
    };

    void exclusive_scan(const T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        storage_type& storage) [[hc]]
    {
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();

        T reductions[ItemsPerThread];
        reduction = T();
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            warp_bit_plus_reduce(input[i], reductions[i]);
            reduction = reduction + reductions[i];
        }
        storage.warp_prefixes[warp_id] = reduction;
        ::rocprim::syncthreads();

        // Scan the warp reduction results to calculate warp prefixes
        if(warp_id == 0)
        {
            if(lane_id < warps_no)
            {
                T prefix = storage.warp_prefixes[lane_id];
                ::rocprim::plus<T> plus;
                warp_scan_prefix_type().inclusive_scan(prefix, prefix, plus);
                storage.warp_prefixes[lane_id] = prefix;
            }
        }
        ::rocprim::syncthreads();

        // Calculate the final scan result for every thread
        T prefix = warp_id == 0 ? T() : storage.warp_prefixes[warp_id - 1];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            // Perform exclusive warp scan of bit values
            warp_bit_plus_exlusive_scan(input[i], output[i]);
            output[i] = prefix + output[i];

            prefix = prefix + reductions[i];
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
    unsigned int RadixBits = 2
>
class block_radix_sort
{
    static constexpr bool with_values = !std::is_same<Value, detail::empty_type>::value;
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

    struct storage_type_with_values
    {
        Value values[BlockSize * ItemsPerThread];
    };
    struct storage_type_without_values { };

public:

    struct storage_type
        : std::conditional<with_values, storage_type_with_values, storage_type_without_values>::type
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

private:

    template<bool Descending, class SortedValue>
    void sort_impl(Key (&keys)[ItemsPerThread],
                   SortedValue * values,
                   storage_type& storage,
                   unsigned int begin_bit,
                   unsigned int end_bit) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int current_warp_size = (warp_id == warps_no - 1)
            ? (BlockSize % warp_size > 0 ? BlockSize % warp_size : warp_size)
            : warp_size;

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            bit_key_type bit_key = key_codec::encode(keys[i]);
            bit_key = (Descending ? ~bit_key : bit_key);
            store(storage, flat_id * ItemsPerThread + i, bit_key, values[i]);
        }
        ::rocprim::syncthreads();

        block_bit_plus_scan scan;

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
                const unsigned int from_position =
                    warp_id * warp_size * ItemsPerThread + i * current_warp_size + lane_id;
                load(bit_keys[i], values[i], storage, from_position);
                const bit_key_type bit_key = bit_keys[i];
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
                unsigned int to_position = 0;
                for(unsigned int r = 0; r < radix_size; r++)
                {
                    to_position = radix == r ? (counts[r] + positions[i][r]) : to_position;
                }
                store(storage, to_position, bit_key, values[i]);
            }
            ::rocprim::syncthreads();
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            bit_key_type bit_key;
            load(bit_key, values[i], storage, flat_id * ItemsPerThread + i);
            bit_key = (Descending ? ~bit_key : bit_key);
            keys[i] = key_codec::decode(bit_key);
        }
    }

    template<class SortedValue>
    void store(storage_type& storage,
               unsigned int i,
               const bit_key_type &bit_key,
               const SortedValue &value) [[hc]]
    {
        storage.bit_keys[i] = bit_key;
        storage.values[i] = value;
    }

    void store(storage_type& storage,
               unsigned int i,
               const bit_key_type &bit_key,
               const detail::empty_type &value) [[hc]]
    {
        storage.bit_keys[i] = bit_key;
    }

    template<class SortedValue>
    void load(bit_key_type &bit_key,
              SortedValue &value,
              const storage_type& storage,
              unsigned int i) [[hc]]
    {
        bit_key = storage.bit_keys[i];
        value = storage.values[i];
    }

    void load(bit_key_type &bit_key,
              detail::empty_type &value,
              const storage_type& storage,
              unsigned int i) [[hc]]
    {
        bit_key = storage.bit_keys[i];
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
