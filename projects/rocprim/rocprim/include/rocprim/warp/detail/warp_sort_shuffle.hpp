// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../functional.hpp"
#include "../../intrinsics/warp_shuffle.hpp"
#include "../../intrinsics/bit.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Key, unsigned int VirtualWaveSize, class Value>
class warp_sort_shuffle
{
private:
    template<class V, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void wlev_cas(bool dir, BinaryFunction compare_function, unsigned int xor_mask, Key& k, V& v)
    {
        Key  k1   = warp_swizzle_shuffle(k, xor_mask, VirtualWaveSize);
        bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        k         = swap ? k1 : k;
        if(swap)
        {
            v = warp_swizzle_shuffle(v, xor_mask, VirtualWaveSize);
        }
    }

    template<class V, class BinaryFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void wlev_cas(bool           dir,
                  BinaryFunction compare_function,
                  unsigned int   xor_mask,
                  Key (&k)[ItemsPerThread],
                  V (&v)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; item++)
        {
            Key  k0   = k[item];
            Key  k1   = warp_swizzle_shuffle(k0, xor_mask, VirtualWaveSize);
            bool swap = compare_function(dir ? k0 : k1, dir ? k1 : k0);
            k[item]   = swap ? k1 : k0;
            if(swap)
            {
                v[item] = warp_swizzle_shuffle(v[item], xor_mask, VirtualWaveSize);
            }
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void wlev_cas(
        bool dir, BinaryFunction compare_function, unsigned int xor_mask, Key& k)
    {
        Key  k1   = warp_swizzle_shuffle(k, xor_mask, VirtualWaveSize);
        bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        k         = swap ? k1 : k;
    }

    template<class BinaryFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void wlev_cas(bool           dir,
                  BinaryFunction compare_function,
                  unsigned int   xor_mask,
                  Key (&k)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; item++)
        {
            Key  k0   = k[item];
            Key  k1   = warp_swizzle_shuffle(k0, xor_mask, VirtualWaveSize);
            bool swap = compare_function(dir ? k0 : k1, dir ? k1 : k0);
            k[item]   = swap ? k1 : k0;
        }
    }

    /// Applies the thread-level compare and swaps.
    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tlev_cas(bool           dir,
                  BinaryFunction compare_function,
                  unsigned int   group_size,
                  unsigned int   offset,
                  Key (&k)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for(unsigned int base = 0; base < ItemsPerThread; base += 2 * offset)
        {
            const bool local_dir = ((base & group_size) > 0) != dir;

            ROCPRIM_UNROLL
            for(unsigned i = 0; i < offset; ++i)
            {
                unsigned int i_l  = base + i;
                unsigned int i_r  = base + i + offset;
                bool         swap = compare_function(k[i_l], k[i_r]) == local_dir;

                Key k_l = k[i_l];
                Key k_r = k[i_r];
                k[i_l]  = swap ? k_r : k_l;
                k[i_r]  = swap ? k_l : k_r;
            }
        }
    }

    /// Applies the thread-level compare and swaps.
    template<unsigned int ItemsPerThread, class BinaryFunction, class V>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tlev_cas(bool           dir,
                  BinaryFunction compare_function,
                  unsigned int   group_size,
                  unsigned int   offset,
                  Key (&k)[ItemsPerThread],
                  V (&v)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for(unsigned int base = 0; base < ItemsPerThread; base += 2 * offset)
        {
            // The local direction must change every group_size items
            // and is flipped if dir is true
            const bool local_dir = ((base & group_size) > 0) != dir;

            ROCPRIM_UNROLL
            for(unsigned i = 0; i < offset; ++i)
            {
                unsigned int i_l = base + i;
                unsigned int i_r = base + i + offset;

                bool swap = compare_function(k[i_l], k[i_r]) == local_dir;

                Key k_l = k[i_l];
                Key k_r = k[i_r];
                k[i_l]  = swap ? k_r : k_l;
                k[i_r]  = swap ? k_l : k_r;

                V v_l  = v[i_l];
                V v_r  = v[i_r];
                v[i_l] = swap ? v_r : v_l;
                v[i_r] = swap ? v_l : v_r;
            }
        }
    }

    template<unsigned int ItemsPerThread,
             class BinaryFunction,
             int group_size = ItemsPerThread,
             int offset     = group_size / 2,
             class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tlev_pass(bool dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        // Implement the following loop using recursion:
        //   for(unsigned int offset = group_size / 2; offset > 0; offset /= 2)
        if constexpr(offset > 0)
        {
            tlev_cas<ItemsPerThread>(dir, compare_function, group_size, offset, kv...);
            // Recurse
            tlev_pass<ItemsPerThread, BinaryFunction, group_size, offset / 2>(dir,
                                                                              compare_function,
                                                                              kv...);
        }
    }

    template<unsigned int ItemsPerThread,
             class BinaryFunction,
             int group_size = 2,
             class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tlev_sort(bool dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        // Implement the following loop using recursion:
        //   for(unsigned int group_size = 2; group_size <= ItemsPerThread; group_size *= 2)
        if constexpr(group_size <= ItemsPerThread)
        {
            tlev_pass<ItemsPerThread, BinaryFunction, group_size>(dir, compare_function, kv...);
            // Recurse
            tlev_sort<ItemsPerThread, BinaryFunction, group_size * 2, KeyValue...>(dir,
                                                                                   compare_function,
                                                                                   kv...);
        }
    }

    template<class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void bitonic_sort(BinaryFunction compare_function, KeyValue&... kv)
    {
        static_assert(sizeof...(KeyValue) < 3,
                      "KeyValue parameter pack can 1 or 2 elements (key, or key and value)");

        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();
        __builtin_assume(id >= 0 && id < VirtualWaveSize);

        // Construct a bit mask in scalar registers from lane id.
        constexpr int num_id_bits = Log2<VirtualWaveSize>::VALUE;
        bool id_bits[num_id_bits];
        ROCPRIM_UNROLL
        for(int i = 0; i < num_id_bits; ++i)
        {
            id_bits[i] = id & (1u << i);
        }

        ROCPRIM_UNROLL
        for(int group_bit = 1; (1 << group_bit) < VirtualWaveSize; ++group_bit)
        {
            const bool group_dir = id_bits[group_bit];

            ROCPRIM_UNROLL
            for(int pass_bit = group_bit - 1; pass_bit >= 0; --pass_bit)
            {
                const unsigned int pass_mask = 1u << pass_bit;
                const bool         pass_dir  = group_dir != id_bits[pass_bit];
                wlev_cas(pass_dir, compare_function, pass_mask, kv...);
            }
        }

        ROCPRIM_UNROLL
        for(int pass_bit = Log2<VirtualWaveSize>::VALUE - 1; pass_bit >= 0; --pass_bit)
        {
            const unsigned int pass_mask = 1u << pass_bit;
            const bool         pass_dir  = id_bits[pass_bit];
            wlev_cas(pass_dir, compare_function, pass_mask, kv...);
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void bitonic_sort(BinaryFunction compare_function, KeyValue&... kv)
    {
        static_assert(sizeof...(KeyValue) < 3,
                      "KeyValue parameter pack can 1 or 2 elements (key, or key and value)");
        static_assert(detail::is_power_of_two(ItemsPerThread), "ItemsPerThread must be power of 2");

        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();
        __builtin_assume(id >= 0 && id < VirtualWaveSize);

        // Construct a bit mask in scalar registers from lane id.
        constexpr int num_id_bits = Log2<VirtualWaveSize>::VALUE;
        bool id_bits[num_id_bits];
        ROCPRIM_UNROLL
        for(int i = 0; i < num_id_bits; ++i)
        {
            id_bits[i] = id & (1u << i);
        }

        tlev_sort<ItemsPerThread>(id_bits[0], compare_function, kv...);
        ROCPRIM_UNROLL
        for(int group_bit = 1; (1 << group_bit) < VirtualWaveSize; ++group_bit)
        {
            const bool group_dir = id_bits[group_bit];

            ROCPRIM_UNROLL
            for(int pass_bit = group_bit - 1; pass_bit >= 0; --pass_bit)
            {
                const unsigned int pass_mask = 1u << pass_bit;
                const bool         pass_dir  = group_dir != id_bits[pass_bit];
                wlev_cas(pass_dir, compare_function, pass_mask, kv...);
            }
            tlev_pass<ItemsPerThread>(group_dir, compare_function, kv...);
        }

        ROCPRIM_UNROLL
        for(int pass_bit = Log2<VirtualWaveSize>::VALUE - 1; pass_bit >= 0; --pass_bit)
        {
            const unsigned int pass_mask = 1u << pass_bit;
            const bool         pass_dir  = id_bits[pass_bit];
            wlev_cas(pass_dir, compare_function, pass_mask, kv...);
        }
        tlev_pass<ItemsPerThread>(false, compare_function, kv...);
    }

public:
    static_assert(detail::is_power_of_two(VirtualWaveSize), "VirtualWaveSize must be power of 2");

    using storage_type = ::rocprim::detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key& thread_value, BinaryFunction compare_function)
    {
        // sort by value only
        bitonic_sort(compare_function, thread_value);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        sort(Key& thread_value, storage_type& storage, BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_value, compare_function);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_values)[ItemsPerThread],
                                            BinaryFunction compare_function)
    {
        // sort by value only
        bitonic_sort<ItemsPerThread>(compare_function, thread_values);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_values)[ItemsPerThread],
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_values, compare_function);
    }

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function)
    {
        if(sizeof(V) <= sizeof(int))
        {
            bitonic_sort(compare_function, thread_key, thread_value);
        }
        else
        {
            // Instead of passing large values between lanes we pass indices and gather values after sorting.
            unsigned int v = detail::logical_lane_id<VirtualWaveSize>();
            bitonic_sort(compare_function, thread_key, v);
            thread_value = warp_shuffle(thread_value, v, VirtualWaveSize);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key&           thread_key,
                                            Value&         thread_value,
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        (void)storage;
        sort(compare_function, thread_key, thread_value);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function)
    {
        if(sizeof(V) <= sizeof(int))
        {
            bitonic_sort<ItemsPerThread>(compare_function, thread_keys, thread_values);
        }
        else
        {
            // Instead of passing large values between lanes we pass indices and gather values after sorting.
            unsigned int v[ItemsPerThread];
            ROCPRIM_UNROLL
            for(unsigned int item = 0; item < ItemsPerThread; item++)
            {
                v[item] = ItemsPerThread * detail::logical_lane_id<VirtualWaveSize>() + item;
            }

            bitonic_sort<ItemsPerThread>(compare_function, thread_keys, v);

            V copy[ItemsPerThread];
            ROCPRIM_UNROLL
            for(unsigned item = 0; item < ItemsPerThread; ++item)
            {
                copy[item] = thread_values[item];
            }

            ROCPRIM_UNROLL
            for(unsigned int dst_item = 0; dst_item < ItemsPerThread; ++dst_item)
            {
                ROCPRIM_UNROLL
                for(unsigned src_item = 0; src_item < ItemsPerThread; ++src_item)
                {
                    V temp
                        = warp_shuffle(copy[src_item], v[dst_item] / ItemsPerThread, VirtualWaveSize);
                    if(v[dst_item] % ItemsPerThread == src_item)
                    {
                        thread_values[dst_item] = temp;
                    }
                }
            }
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void sort(Key (&thread_keys)[ItemsPerThread],
                                            Value (&thread_values)[ItemsPerThread],
                                            storage_type&  storage,
                                            BinaryFunction compare_function)
    {
        (void)storage;
        sort(thread_keys, thread_values, compare_function);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_
