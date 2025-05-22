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
#include "../../intrinsics.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Key, unsigned int VirtualWaveSize, class Value>
class warp_sort_shuffle
{
private:
    template<int warp, int xor_mask, class V, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<!(VirtualWaveSize > warp)>::type
        swap(Key& k, V& v, bool dir, BinaryFunction compare_function)
    {
        (void)k;
        (void)v;
        (void)dir;
        (void)compare_function;
    }

    template<int warp, int xor_mask, class V, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(VirtualWaveSize > warp)>::type
        swap(Key& k, V& v, bool dir, BinaryFunction compare_function)
    {
        Key  k1   = warp_swizzle_shuffle(k, xor_mask, VirtualWaveSize);
        bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        if(swap)
        {
            k = k1;
            v = warp_swizzle_shuffle(v, xor_mask, VirtualWaveSize);
        }
    }

    template<int warp, int xor_mask, class V, class BinaryFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<!(VirtualWaveSize > warp)>::type swap(Key (&k)[ItemsPerThread],
                                                                  V (&v)[ItemsPerThread],
                                                                  bool           dir,
                                                                  BinaryFunction compare_function)
    {
        (void)k;
        (void)v;
        (void)dir;
        (void)compare_function;
    }

    template<int warp, int xor_mask, class V, class BinaryFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(VirtualWaveSize > warp)>::type swap(Key (&k)[ItemsPerThread],
                                                                 V (&v)[ItemsPerThread],
                                                                 bool           dir,
                                                                 BinaryFunction compare_function)
    {
        Key k1[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; item++)
        {
            k1[item]  = warp_swizzle_shuffle(k[item], xor_mask, VirtualWaveSize);
            bool swap = compare_function(dir ? k[item] : k1[item], dir ? k1[item] : k[item]);
            if(swap)
            {
                k[item] = k1[item];
                v[item] = warp_swizzle_shuffle(v[item], xor_mask, VirtualWaveSize);
            }
        }
    }

    template<int warp, int xor_mask, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<!(VirtualWaveSize > warp)>::type
        swap(Key& k, bool dir, BinaryFunction compare_function)
    {
        (void)k;
        (void)dir;
        (void)compare_function;
    }

    template<int warp, int xor_mask, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(VirtualWaveSize > warp)>::type
        swap(Key& k, bool dir, BinaryFunction compare_function)
    {
        Key  k1   = warp_swizzle_shuffle(k, xor_mask, VirtualWaveSize);
        bool swap = compare_function(dir ? k : k1, dir ? k1 : k);
        if(swap)
        {
            k = k1;
        }
    }

    template<int warp, int xor_mask, class BinaryFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<!(VirtualWaveSize > warp)>::type
        swap(Key (&k)[ItemsPerThread], bool dir, BinaryFunction compare_function)
    {
        (void)k;
        (void)dir;
        (void)compare_function;
    }

    template<int warp, int xor_mask, class BinaryFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(VirtualWaveSize > warp)>::type
        swap(Key (&k)[ItemsPerThread], bool dir, BinaryFunction compare_function)
    {
        Key k1[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; item++)
        {
            k1[item]  = warp_swizzle_shuffle(k[item], xor_mask, VirtualWaveSize);
            bool swap = compare_function(dir ? k[item] : k1[item], dir ? k1[item] : k[item]);
            if(swap)
            {
                k[item] = k1[item];
            }
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void thread_swap(Key (&k)[ItemsPerThread],
                                                   unsigned int   i,
                                                   unsigned int   j,
                                                   bool           dir,
                                                   BinaryFunction compare_function)
    {
        if(compare_function(k[i], k[j]) == dir)
        {
            Key temp = k[i];
            k[i]     = k[j];
            k[j]     = temp;
        }
    }

    template<unsigned int ItemsPerThread, class V, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void thread_swap(Key (&k)[ItemsPerThread],
                                                   V (&v)[ItemsPerThread],
                                                   unsigned int   i,
                                                   unsigned int   j,
                                                   bool           dir,
                                                   BinaryFunction compare_function)
    {
        if(compare_function(k[i], k[j]) == dir)
        {
            Key k_temp = k[i];
            k[i]       = k[j];
            k[j]       = k_temp;
            V v_temp   = v[i];
            v[i]       = v[j];
            v[j]       = v_temp;
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE void thread_shuffle(unsigned int   group_size,
                                                      unsigned int   offset,
                                                      bool           dir,
                                                      BinaryFunction compare_function,
                                                      KeyValue&... kv)
    {
        ROCPRIM_UNROLL
        for(unsigned int base = 0; base < ItemsPerThread; base += 2 * offset)
        {
            // The local direction must change every group_size items
            // and is flipped if dir is true
            const bool local_dir = ((base & group_size) > 0) != dir;

            for(unsigned i = 0; i < offset; ++i)
            {
                thread_swap(kv..., base + i, base + i + offset, local_dir, compare_function);
            }
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        thread_sort(bool dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        ROCPRIM_UNROLL
        for(unsigned int k = 2; k <= ItemsPerThread; k *= 2)
        {
            ROCPRIM_UNROLL
            for(unsigned int j = k / 2; j > 0; j /= 2)
            {
                thread_shuffle<ItemsPerThread>(k, j, dir, compare_function, kv...);
            }
        }
    }

    template<int warp, unsigned int ItemsPerThread, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(VirtualWaveSize > warp)>::type
        thread_merge(bool dir, BinaryFunction compare_function, KeyValue&... kv)
    {
        ROCPRIM_UNROLL
        for(unsigned int j = ItemsPerThread / 2; j > 0; j /= 2)
        {
            thread_shuffle<ItemsPerThread>(ItemsPerThread, j, dir, compare_function, kv...);
        }
    }

    template<int warp, unsigned int ItemsPerThread, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<!(VirtualWaveSize > warp)>::type
        thread_merge(bool /*dir*/, BinaryFunction /*compare_function*/, KeyValue&... /*kv*/)
    {}

    template<class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE void bitonic_sort(BinaryFunction compare_function,
                                                    KeyValue&... kv)
    {
        static_assert(sizeof...(KeyValue) < 3,
                      "KeyValue parameter pack can 1 or 2 elements (key, or key and value)");

        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();

        swap<2, 1>(kv..., get_bit(id, 1) != get_bit(id, 0), compare_function);

        swap<4, 2>(kv..., get_bit(id, 2) != get_bit(id, 1), compare_function);
        swap<4, 1>(kv..., get_bit(id, 2) != get_bit(id, 0), compare_function);

        swap<8, 4>(kv..., get_bit(id, 3) != get_bit(id, 2), compare_function);
        swap<8, 2>(kv..., get_bit(id, 3) != get_bit(id, 1), compare_function);
        swap<8, 1>(kv..., get_bit(id, 3) != get_bit(id, 0), compare_function);

        swap<16, 8>(kv..., get_bit(id, 4) != get_bit(id, 3), compare_function);
        swap<16, 4>(kv..., get_bit(id, 4) != get_bit(id, 2), compare_function);
        swap<16, 2>(kv..., get_bit(id, 4) != get_bit(id, 1), compare_function);
        swap<16, 1>(kv..., get_bit(id, 4) != get_bit(id, 0), compare_function);

        swap<32, 16>(kv..., get_bit(id, 5) != get_bit(id, 4), compare_function);
        swap<32, 8>(kv..., get_bit(id, 5) != get_bit(id, 3), compare_function);
        swap<32, 4>(kv..., get_bit(id, 5) != get_bit(id, 2), compare_function);
        swap<32, 2>(kv..., get_bit(id, 5) != get_bit(id, 1), compare_function);
        swap<32, 1>(kv..., get_bit(id, 5) != get_bit(id, 0), compare_function);

        swap<32, 32>(kv..., get_bit(id, 5) != 0, compare_function);
        swap<16, 16>(kv..., get_bit(id, 4) != 0, compare_function);
        swap<8, 8>(kv..., get_bit(id, 3) != 0, compare_function);
        swap<4, 4>(kv..., get_bit(id, 2) != 0, compare_function);
        swap<2, 2>(kv..., get_bit(id, 1) != 0, compare_function);
        swap<0, 1>(kv..., get_bit(id, 0) != 0, compare_function);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE void bitonic_sort(BinaryFunction compare_function,
                                                    KeyValue&... kv)
    {
        static_assert(sizeof...(KeyValue) < 3,
                      "KeyValue parameter pack can 1 or 2 elements (key, or key and value)");

        static_assert(detail::is_power_of_two(ItemsPerThread), "ItemsPerThread must be power of 2");

        const unsigned int id = detail::logical_lane_id<VirtualWaveSize>();

        thread_sort<ItemsPerThread>(get_bit(id, 0) != 0, compare_function, kv...);

        swap<2, 1>(kv..., get_bit(id, 1) != get_bit(id, 0), compare_function);
        thread_merge<2, ItemsPerThread>(get_bit(id, 1) != 0, compare_function, kv...);

        swap<4, 2>(kv..., get_bit(id, 2) != get_bit(id, 1), compare_function);
        swap<4, 1>(kv..., get_bit(id, 2) != get_bit(id, 0), compare_function);
        thread_merge<4, ItemsPerThread>(get_bit(id, 2) != 0, compare_function, kv...);

        swap<8, 4>(kv..., get_bit(id, 3) != get_bit(id, 2), compare_function);
        swap<8, 2>(kv..., get_bit(id, 3) != get_bit(id, 1), compare_function);
        swap<8, 1>(kv..., get_bit(id, 3) != get_bit(id, 0), compare_function);
        thread_merge<8, ItemsPerThread>(get_bit(id, 3) != 0, compare_function, kv...);

        swap<16, 8>(kv..., get_bit(id, 4) != get_bit(id, 3), compare_function);
        swap<16, 4>(kv..., get_bit(id, 4) != get_bit(id, 2), compare_function);
        swap<16, 2>(kv..., get_bit(id, 4) != get_bit(id, 1), compare_function);
        swap<16, 1>(kv..., get_bit(id, 4) != get_bit(id, 0), compare_function);
        thread_merge<16, ItemsPerThread>(get_bit(id, 4) != 0, compare_function, kv...);

        swap<32, 16>(kv..., get_bit(id, 5) != get_bit(id, 4), compare_function);
        swap<32, 8>(kv..., get_bit(id, 5) != get_bit(id, 3), compare_function);
        swap<32, 4>(kv..., get_bit(id, 5) != get_bit(id, 2), compare_function);
        swap<32, 2>(kv..., get_bit(id, 5) != get_bit(id, 1), compare_function);
        swap<32, 1>(kv..., get_bit(id, 5) != get_bit(id, 0), compare_function);
        thread_merge<32, ItemsPerThread>(get_bit(id, 5) != 0, compare_function, kv...);

        swap<32, 32>(kv..., get_bit(id, 5) != 0, compare_function);
        swap<16, 16>(kv..., get_bit(id, 4) != 0, compare_function);
        swap<8, 8>(kv..., get_bit(id, 3) != 0, compare_function);
        swap<4, 4>(kv..., get_bit(id, 2) != 0, compare_function);
        swap<2, 2>(kv..., get_bit(id, 1) != 0, compare_function);
        swap<0, 1>(kv..., get_bit(id, 0) != 0, compare_function);
        thread_merge<1, ItemsPerThread>(false, compare_function, kv...);
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
    ROCPRIM_DEVICE ROCPRIM_INLINE typename std::enable_if<(sizeof(V) <= sizeof(int))>::type
        sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function)
    {
        bitonic_sort(compare_function, thread_key, thread_value);
    }

    template<class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE typename std::enable_if<!(sizeof(V) <= sizeof(int))>::type
        sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function)
    {
        // Instead of passing large values between lanes we pass indices and gather values after sorting.
        unsigned int v = detail::logical_lane_id<VirtualWaveSize>();
        bitonic_sort(compare_function, thread_key, v);
        thread_value = warp_shuffle(thread_value, v, VirtualWaveSize);
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
    ROCPRIM_DEVICE ROCPRIM_INLINE typename std::enable_if<(sizeof(V) <= sizeof(int))>::type
        sort(Key (&thread_keys)[ItemsPerThread],
             Value (&thread_values)[ItemsPerThread],
             BinaryFunction compare_function)
    {
        bitonic_sort<ItemsPerThread>(compare_function, thread_keys, thread_values);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE typename std::enable_if<!(sizeof(V) <= sizeof(int))>::type
        sort(Key (&thread_keys)[ItemsPerThread],
             Value (&thread_values)[ItemsPerThread],
             BinaryFunction compare_function)
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
                    thread_values[dst_item] = temp;
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
