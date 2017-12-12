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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../../detail/config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class Key,
    unsigned int WarpSize,
    class Value
>
class warp_sort_shuffle
{
private:
    struct key_value_type
    {
        Key k;
        Value v;
    };

    template<class KV, class BinaryFunction>
    KV shuffle_swap(const KV a, int mask, int dir, BinaryFunction compare_function) [[hc]]
    {
        KV b = warp_shuffle_xor(a, mask, WarpSize);
        return compare_function(a, b) == dir ? b : a;
    }

    template<int warp, class KV, class BinaryFunction>
    typename std::enable_if<!(WarpSize > warp), KV>::type
    swap(const KV a, int mask, int dir, BinaryFunction compare_function) [[hc]]
    {
        return a;
    }

    template<int warp, class KV, class BinaryFunction>
    typename std::enable_if<(WarpSize > warp), KV>::type
    swap(const KV a, int mask, int dir, BinaryFunction compare_function) [[hc]]
    {
        return shuffle_swap<KV>(a, mask, dir, compare_function);
    }

    template<class KV, class BinaryFunction>
    void bitonic_sort(KV& val, BinaryFunction compare_function) [[hc]]
    {
        unsigned int id = detail::logical_lane_id<WarpSize>();
        val = swap<2, KV, BinaryFunction>(val, 0x01,
                                         get_bit(id, 1) ^ get_bit(id, 0),
                                         compare_function);

        val = swap<4, KV, BinaryFunction>(val, 0x02,
                                         get_bit(id, 2) ^ get_bit(id, 1),
                                         compare_function);
        val = swap<4, KV, BinaryFunction>(val, 0x01,
                                         get_bit(id, 2) ^ get_bit(id, 0),
                                         compare_function);

        val = swap<8, KV, BinaryFunction>(val, 0x04,
                                         get_bit(id, 3) ^ get_bit(id, 2),
                                         compare_function);
        val = swap<8, KV, BinaryFunction>(val, 0x02,
                                         get_bit(id, 3) ^ get_bit(id, 1),
                                         compare_function);
        val = swap<8, KV, BinaryFunction>(val, 0x01,
                                         get_bit(id, 3) ^ get_bit(id, 0),
                                         compare_function);

        val = swap<16, KV, BinaryFunction>(val, 0x08,
                                          get_bit(id, 4) ^ get_bit(id, 3),
                                          compare_function);
        val = swap<16, KV, BinaryFunction>(val, 0x04,
                                          get_bit(id, 4) ^ get_bit(id, 2),
                                          compare_function);
        val = swap<16, KV, BinaryFunction>(val, 0x02,
                                          get_bit(id, 4) ^ get_bit(id, 1),
                                          compare_function);
        val = swap<16, KV, BinaryFunction>(val, 0x01,
                                          get_bit(id, 4) ^ get_bit(id, 0),
                                          compare_function);

        val = swap<32, KV, BinaryFunction>(val, 0x10,
                                          get_bit(id, 5) ^ get_bit(id, 4),
                                          compare_function);
        val = swap<32, KV, BinaryFunction>(val, 0x08,
                                          get_bit(id, 5) ^ get_bit(id, 3),
                                          compare_function);
        val = swap<32, KV, BinaryFunction>(val, 0x04,
                                          get_bit(id, 5) ^ get_bit(id, 2),
                                          compare_function);
        val = swap<32, KV, BinaryFunction>(val, 0x02,
                                          get_bit(id, 5) ^ get_bit(id, 1),
                                          compare_function);
        val = swap<32, KV, BinaryFunction>(val, 0x01,
                                          get_bit(id, 5) ^ get_bit(id, 0),
                                          compare_function);

        val = swap<32, KV, BinaryFunction>(val, 0x20,
                                          get_bit(id, 5),
                                          compare_function);
        val = swap<16, KV, BinaryFunction>(val, 0x10,
                                          get_bit(id, 4),
                                          compare_function);
        val = swap<8, KV, BinaryFunction>(val, 0x08,
                                         get_bit(id, 3),
                                         compare_function);
        val = swap<4, KV, BinaryFunction>(val, 0x04,
                                         get_bit(id, 2),
                                         compare_function);
        val = swap<2, KV, BinaryFunction>(val, 0x02,
                                         get_bit(id, 1),
                                         compare_function);

        val = swap<0, KV, BinaryFunction>(val, 0x01,
                                         get_bit(id, 0),
                                         compare_function);
    }

public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    struct storage_type
    {
        // can't use empty_type type due to multiple definitions of the same functions
    };

    template<class BinaryFunction>
    void sort(Key& thread_value, BinaryFunction compare_function) [[hc]]
    {
        // sort by value only
        bitonic_sort<Key>(thread_value, compare_function);
    }

    template<class BinaryFunction>
    void sort(Key& thread_value, storage_type& storage, 
              BinaryFunction compare_function) [[hc]]
    {
        (void) storage;
        sort(thread_value, compare_function);
    }

    template<class BinaryFunction>
    void sort(Key& thread_key, Value& thread_value, BinaryFunction compare_function) [[hc]]
    {
        // initialize key and value to a struct
        key_value_type kv = {thread_key, thread_value};
        // pass a custom compare_function to compare key only (sort_by_key)
        bitonic_sort<key_value_type>(
            kv,
            [&compare_function](const key_value_type& kv1, const key_value_type& kv2) [[hc]]
            {
                return compare_function(kv1.k, kv2.k);
            }
        );
        thread_key = kv.k;
        thread_value = kv.v;
    }

    template<class BinaryFunction>
    void sort(Key& thread_key, Value& thread_value, storage_type& storage,
              BinaryFunction compare_function) [[hc]]
    {
        (void) storage;
        return sort(thread_key, thread_value, compare_function);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_
