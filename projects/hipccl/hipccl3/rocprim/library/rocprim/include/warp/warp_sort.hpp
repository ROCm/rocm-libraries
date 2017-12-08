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

#include "detail/warp_sort_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class Key,
    unsigned int WarpSize,
    class Value
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
template<class Key, unsigned int WarpSize, class Value>
struct select_warp_sort_impl
{
    typedef typename std::conditional<
        // can we use shuffle-based implementation?
        detail::is_warpsize_shuffleable<WarpSize>::value,
        detail::warp_sort_shuffle<Key, WarpSize, Value>, // yes
        detail::warp_sort_shared_mem<Key, WarpSize, Value> // no
    >::type type;
};

} // end namespace detail

/// \brief Parallel sort primitive for warp.
template<
    class Key,
    unsigned int WarpSize = warp_size(),
    class Value = detail::empty_type
>
class warp_sort : detail::select_warp_sort_impl<Key, WarpSize, Value>::type
{
    typedef typename detail::select_warp_sort_impl<Key, WarpSize, Value>::type base_type;

public:
    typedef typename base_type::storage_type storage_type;

    template<class BinaryFunction = ::rocprim::less<Key>>
    void sort(Key& thread_key, 
              BinaryFunction compare_function = BinaryFunction()) [[hc]]
    {
        base_type::sort(thread_key, compare_function);
    }

    template<class BinaryFunction = ::rocprim::less<Key>>
    void sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction()) [[hc]]
    {
        base_type::sort(
            thread_key, storage, compare_function
        );
    }

    template<class BinaryFunction = ::rocprim::less<Key>>
    void sort(Key& thread_key, 
              Value& thread_value, 
              BinaryFunction compare_function = BinaryFunction()) [[hc]]
    {
        base_type::sort(
            thread_key, thread_value, compare_function
        );
    }

    template<class BinaryFunction = ::rocprim::less<Key>>
    void sort(Key& thread_key, 
              Value& thread_value,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction()) [[hc]]
    {
        base_type::sort(
            thread_key, thread_value, storage, compare_function
        );
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_WARP_SORT_HPP_
