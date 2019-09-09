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

#include "../../config.hpp"
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
    template<int warp, class BinaryFunction>
    ROCPRIM_DEVICE inline
    typename std::enable_if<!(WarpSize > warp)>::type
    swap(Key& k, Value& v, int mask, int dir, BinaryFunction compare_function)
    {
        (void) k;
        (void) v;
        (void) mask;
        (void) dir;
        (void) compare_function;
    }

    template<int warp, class BinaryFunction>
    ROCPRIM_DEVICE inline
    typename std::enable_if<(WarpSize > warp)>::type
    swap(Key& k, Value& v, int mask, int dir, BinaryFunction compare_function)
    {
        Key k1 = warp_shuffle_xor(k, mask, WarpSize);
        Value v1 = warp_shuffle_xor(v, mask, WarpSize);
        bool swap = dir == 0 ? compare_function(k1, k) : compare_function(k, k1);
        if (swap)
        {
            k = k1;
            v = v1;
        }
    }

    template<int warp, class BinaryFunction>
    ROCPRIM_DEVICE inline
    typename std::enable_if<!(WarpSize > warp)>::type
    swap(Key& k, int mask, int dir, BinaryFunction compare_function)
    {
        (void) k;
        (void) mask;
        (void) dir;
        (void) compare_function;
    }

    template<int warp, class BinaryFunction>
    ROCPRIM_DEVICE inline
    typename std::enable_if<(WarpSize > warp)>::type
    swap(Key& k, int mask, int dir, BinaryFunction compare_function)
    {
        Key k1 = warp_shuffle_xor(k, mask, WarpSize);
        bool swap = dir == 0 ? compare_function(k1, k) : compare_function(k, k1);
        if (swap)
        {
            k = k1;
        }
    }

    template<class BinaryFunction, class... KeyValue>
    ROCPRIM_DEVICE inline
    void bitonic_sort(BinaryFunction compare_function, KeyValue&... kv)
    {
        static_assert(
            sizeof...(KeyValue) < 3,
            "KeyValue parameter pack can 1 or 2 elements (key, or key and value)"
        );

        unsigned int id = detail::logical_lane_id<WarpSize>();
        swap<2,  BinaryFunction>(kv..., 1, get_bit(id, 1) ^ get_bit(id, 0), compare_function);

        swap<4,  BinaryFunction>(kv..., 2, get_bit(id, 2) ^ get_bit(id, 1), compare_function);
        swap<4,  BinaryFunction>(kv..., 1, get_bit(id, 2) ^ get_bit(id, 0), compare_function);

        swap<8,  BinaryFunction>(kv..., 4, get_bit(id, 3) ^ get_bit(id, 2), compare_function);
        swap<8,  BinaryFunction>(kv..., 2, get_bit(id, 3) ^ get_bit(id, 1), compare_function);
        swap<8,  BinaryFunction>(kv..., 1, get_bit(id, 3) ^ get_bit(id, 0), compare_function);

        swap<16, BinaryFunction>(kv..., 8, get_bit(id, 4) ^ get_bit(id, 3), compare_function);
        swap<16, BinaryFunction>(kv..., 4, get_bit(id, 4) ^ get_bit(id, 2), compare_function);
        swap<16, BinaryFunction>(kv..., 2, get_bit(id, 4) ^ get_bit(id, 1), compare_function);
        swap<16, BinaryFunction>(kv..., 1, get_bit(id, 4) ^ get_bit(id, 0), compare_function);

        swap<32, BinaryFunction>(kv..., 16, get_bit(id, 5) ^ get_bit(id, 4), compare_function);
        swap<32, BinaryFunction>(kv..., 8,  get_bit(id, 5) ^ get_bit(id, 3), compare_function);
        swap<32, BinaryFunction>(kv..., 4,  get_bit(id, 5) ^ get_bit(id, 2), compare_function);
        swap<32, BinaryFunction>(kv..., 2,  get_bit(id, 5) ^ get_bit(id, 1), compare_function);
        swap<32, BinaryFunction>(kv..., 1,  get_bit(id, 5) ^ get_bit(id, 0), compare_function);

        swap<32, BinaryFunction>(kv..., 32, get_bit(id, 5), compare_function);
        swap<16, BinaryFunction>(kv..., 16, get_bit(id, 4), compare_function);
        swap<8,  BinaryFunction>(kv..., 8,  get_bit(id, 3), compare_function);
        swap<4,  BinaryFunction>(kv..., 4,  get_bit(id, 2), compare_function);
        swap<2,  BinaryFunction>(kv..., 2,  get_bit(id, 1), compare_function);
        swap<0,  BinaryFunction>(kv..., 1,  get_bit(id, 0), compare_function);
    }

public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = ::rocprim::detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_value, BinaryFunction compare_function)
    {
        // sort by value only
        bitonic_sort(compare_function, thread_value);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_value, storage_type& storage,
              BinaryFunction compare_function)
    {
        (void) storage;
        sort(thread_value, compare_function);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key, Value& thread_value,
              BinaryFunction compare_function)
    {
        bitonic_sort(compare_function, thread_key, thread_value);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key, Value& thread_value,
              storage_type& storage, BinaryFunction compare_function)
    {
        (void) storage;
        return sort(compare_function, thread_key, thread_value);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SORT_SHUFFLE_HPP_
