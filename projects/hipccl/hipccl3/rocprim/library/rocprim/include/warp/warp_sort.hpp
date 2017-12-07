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

#ifndef ROCPRIM_WARP_WARP_SORT_HPP_
#define ROCPRIM_WARP_WARP_SORT_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize,
    class U
>
class warp_sort_shuffle
{
private:
    struct key_value_type
    {
        T k;
        U v;
    };

    template<class KV, class BinaryFunction>
    KV shuffle_swap(const KV a, int mask, int dir, BinaryFunction sort_function) [[hc]]
    {
        KV b = warp_shuffle_xor(a, mask, WarpSize);
        return sort_function(a, b) == dir ? b : a;
    }

    template<int warp, class KV, class BinaryFunction>
    typename std::enable_if<!(WarpSize > warp), KV>::type
    swap(const KV a, int mask, int dir, BinaryFunction sort_function) [[hc]]
    {
        return a;
    }

    template<int warp, class KV, class BinaryFunction>
    typename std::enable_if<(WarpSize > warp), KV>::type
    swap(const KV a, int mask, int dir, BinaryFunction sort_function) [[hc]]
    {
        return shuffle_swap<KV>(a, mask, dir, sort_function);
    }

    template<class KV, class BinaryFunction>
    void bitonic_sort(KV& val, BinaryFunction sort_function) [[hc]]
    {
        unsigned int id = lane_id();
        val = swap<2, KV, BinaryFunction>(val, 0x01,
                                         get_bit(id, 1) ^ get_bit(id, 0),
                                         sort_function);

        val = swap<4, KV, BinaryFunction>(val, 0x02,
                                         get_bit(id, 2) ^ get_bit(id, 1),
                                         sort_function);
        val = swap<4, KV, BinaryFunction>(val, 0x01,
                                         get_bit(id, 2) ^ get_bit(id, 0),
                                         sort_function);

        val = swap<8, KV, BinaryFunction>(val, 0x04,
                                         get_bit(id, 3) ^ get_bit(id, 2),
                                         sort_function);
        val = swap<8, KV, BinaryFunction>(val, 0x02,
                                         get_bit(id, 3) ^ get_bit(id, 1),
                                         sort_function);
        val = swap<8, KV, BinaryFunction>(val, 0x01,
                                         get_bit(id, 3) ^ get_bit(id, 0),
                                         sort_function);

        val = swap<16, KV, BinaryFunction>(val, 0x08,
                                          get_bit(id, 4) ^ get_bit(id, 3),
                                          sort_function);
        val = swap<16, KV, BinaryFunction>(val, 0x04,
                                          get_bit(id, 4) ^ get_bit(id, 2),
                                          sort_function);
        val = swap<16, KV, BinaryFunction>(val, 0x02,
                                          get_bit(id, 4) ^ get_bit(id, 1),
                                          sort_function);
        val = swap<16, KV, BinaryFunction>(val, 0x01,
                                          get_bit(id, 4) ^ get_bit(id, 0),
                                          sort_function);

        val = swap<32, KV, BinaryFunction>(val, 0x10,
                                          get_bit(id, 5) ^ get_bit(id, 4),
                                          sort_function);
        val = swap<32, KV, BinaryFunction>(val, 0x08,
                                          get_bit(id, 5) ^ get_bit(id, 3),
                                          sort_function);
        val = swap<32, KV, BinaryFunction>(val, 0x04,
                                          get_bit(id, 5) ^ get_bit(id, 2),
                                          sort_function);
        val = swap<32, KV, BinaryFunction>(val, 0x02,
                                          get_bit(id, 5) ^ get_bit(id, 1),
                                          sort_function);
        val = swap<32, KV, BinaryFunction>(val, 0x01,
                                          get_bit(id, 5) ^ get_bit(id, 0),
                                          sort_function);

        val = swap<32, KV, BinaryFunction>(val, 0x20,
                                          get_bit(id, 5),
                                          sort_function);
        val = swap<16, KV, BinaryFunction>(val, 0x10,
                                          get_bit(id, 4),
                                          sort_function);
        val = swap<8, KV, BinaryFunction>(val, 0x08,
                                         get_bit(id, 3),
                                         sort_function);
        val = swap<4, KV, BinaryFunction>(val, 0x04,
                                         get_bit(id, 2),
                                         sort_function);
        val = swap<2, KV, BinaryFunction>(val, 0x02,
                                         get_bit(id, 1),
                                         sort_function);

        val = swap<0, KV, BinaryFunction>(val, 0x01,
                                         get_bit(id, 0),
                                         sort_function);
    }

public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    struct storage_type
    {
        // can't use empty_type type
    };

    template<class BinaryFunction>
    void sort(T& thread_value, BinaryFunction sort_function) [[hc]]
    {
        bitonic_sort<T>(thread_value, sort_function);
    }

    template<class BinaryFunction>
    void sort(T& thread_value,
              storage_type& storage,
              BinaryFunction sort_function) [[hc]]
    {
        (void) storage;
        sort(thread_value, sort_function);
    }

    template<class BinaryFunction>
    void sort(T& thread_key, U& thread_value, BinaryFunction sort_function) [[hc]]
    {
        key_value_type kv = {thread_key, thread_value};
        bitonic_sort<key_value_type>(
            kv,
            [&sort_function](const key_value_type& kv1, const key_value_type& kv2) [[hc]]
            {
                return sort_function(kv1.k, kv2.k);
            }
        );
        thread_key = kv.k;
        thread_value = kv.v;
    }

    template<class BinaryFunction>
    void sort(T& thread_key, U& thread_value,
              storage_type& storage,
              BinaryFunction sort_function) [[hc]]
    {
        (void) storage;
        return sort(thread_key, thread_value, sort_function);
    }
};

template<
    class T,
    unsigned int WarpSize,
    class U
>
class warp_sort_shared_mem
{
public:
    static_assert(
        detail::is_power_of_two(WarpSize),
        "warp_sort is not implemented for WarpSizes that are not power of two."
    );

    typedef detail::empty_type storage;
};

// Select warp_sort implementation based WarpSize
template<class T, unsigned int WarpSize, class U>
struct select_warp_sort_impl
{
    typedef typename std::conditional<
        // can we use shuffle-based implementation?
        detail::is_warpsize_shuffleable<WarpSize>::value,
        detail::warp_sort_shuffle<T, WarpSize, U>, // yes
        detail::warp_sort_shared_mem<T, WarpSize, U> // no
    >::type type;
};

} // end namespace detail

/// \brief Parallel sort primitive for warp.
template<
    class T,
    unsigned int WarpSize = warp_size(),
    class U = detail::empty_type
>
class warp_sort : detail::select_warp_sort_impl<T, WarpSize, U>::type
{
    typedef typename detail::select_warp_sort_impl<T, WarpSize, U>::type base_type;

public:
    typedef typename base_type::storage_type storage_type;

    template<class BinaryFunction = ::rocprim::less<T>>
    void sort(T& thread_key, BinaryFunction sort_function = BinaryFunction()) [[hc]]
    {
        base_type::sort(thread_key, sort_function);
    }

    template<class BinaryFunction = ::rocprim::less<T>>
    void sort(T& thread_key,
              storage_type& storage,
              BinaryFunction sort_function = BinaryFunction()) [[hc]]
    {
        base_type::sort(
            thread_key, storage, sort_function
        );
    }

    template<class BinaryFunction = ::rocprim::less<T>>
    void sort(T& thread_key, U& thread_value, BinaryFunction sort_function = BinaryFunction()) [[hc]]
    {
        base_type::sort(
            thread_key, thread_value, sort_function
        );
    }

    template<class BinaryFunction = ::rocprim::less<T>>
    void sort(T& thread_key, U& thread_value,
              storage_type& storage,
              BinaryFunction sort_function = BinaryFunction()) [[hc]]
    {
        base_type::sort(
            thread_key, thread_value, storage, sort_function
        );
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_WARP_SORT_HPP_
