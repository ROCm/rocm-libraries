// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_
#define ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_binary_search.hpp"

#include "device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    class Config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class SearchFunction,
    class CompareFunction
>
inline
hipError_t binary_search(void * temporary_storage,
                         size_t& storage_size,
                         HaystackIterator haystack,
                         NeedlesIterator needles,
                         OutputIterator output,
                         size_t haystack_size,
                         size_t needles_size,
                         SearchFunction search_op,
                         CompareFunction compare_op,
                         hipStream_t stream,
                         bool debug_synchronous)
{
    using value_type = typename std::iterator_traits<NeedlesIterator>::value_type;

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = 4;
        return hipSuccess;
    }

    return transform<Config>(
        needles, output,
        needles_size,
        [haystack, haystack_size, search_op, compare_op]
        ROCPRIM_DEVICE
        (const value_type& value)
        {
            return search_op(haystack, haystack_size, value, compare_op);
        },
        stream, debug_synchronous
    );
}

} // end of detail namespace

template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t lower_bound(void * temporary_storage,
                       size_t& storage_size,
                       HaystackIterator haystack,
                       NeedlesIterator needles,
                       OutputIterator output,
                       size_t haystack_size,
                       size_t needles_size,
                       CompareFunction compare_op = CompareFunction(),
                       hipStream_t stream = 0,
                       bool debug_synchronous = false)
{
    return detail::binary_search<Config>(
        temporary_storage, storage_size,
        haystack, needles, output,
        haystack_size, needles_size,
        detail::lower_bound_search_op(), compare_op,
        stream, debug_synchronous
    );
}

template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t upper_bound(void * temporary_storage,
                       size_t& storage_size,
                       HaystackIterator haystack,
                       NeedlesIterator needles,
                       OutputIterator output,
                       size_t haystack_size,
                       size_t needles_size,
                       CompareFunction compare_op = CompareFunction(),
                       hipStream_t stream = 0,
                       bool debug_synchronous = false)
{
    return detail::binary_search<Config>(
        temporary_storage, storage_size,
        haystack, needles, output,
        haystack_size, needles_size,
        detail::upper_bound_search_op(), compare_op,
        stream, debug_synchronous
    );
}

template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
hipError_t binary_search(void * temporary_storage,
                         size_t& storage_size,
                         HaystackIterator haystack,
                         NeedlesIterator needles,
                         OutputIterator output,
                         size_t haystack_size,
                         size_t needles_size,
                         CompareFunction compare_op = CompareFunction(),
                         hipStream_t stream = 0,
                         bool debug_synchronous = false)
{
    return detail::binary_search<Config>(
        temporary_storage, storage_size,
        haystack, needles, output,
        haystack_size, needles_size,
        detail::binary_search_op(), compare_op,
        stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_
