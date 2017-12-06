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

#ifndef ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_

#include <type_traits>

// HC API
#include <hcc/hc.hpp>

#include "../../detail/config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize
>
class warp_scan_shuffle
{
public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = detail::empty_type;

    template<class BinaryFunction>
    T inclusive_scan(T thread_value, storage_type& storage, BinaryFunction scan_op) [[hc]]
    {
        (void) storage; // disables unused parameter warning

        T value;
        #pragma unroll
        for(unsigned int offset = 1; offset < WarpSize; offset *= 2)
        {
            value = warp_shuffle_up(thread_value, offset, WarpSize);
            unsigned int id = lane_id();
            if(id >= offset) thread_value = scan_op(value, thread_value);
        }
        return thread_value;
    }

    template<class BinaryFunction>
    scan_reduction_type<T>
    inclusive_scan_reduce(T thread_value, storage_type& storage, BinaryFunction scan_op) [[hc]]
    {
        (void) storage; // disables unused parameter warning

        T value;
        #pragma unroll
        for(unsigned int offset = 1; offset < WarpSize; offset *= 2)
        {
            value = warp_shuffle_up(thread_value, offset, WarpSize);
            unsigned int id = lane_id();
            if(id >= offset) thread_value = scan_op(value, thread_value);
        }
        // Broadcast value from the last thread in warp
        T reduction_value = warp_shuffle(thread_value, WarpSize-1, WarpSize);
        return {thread_value, reduction_value};
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_SCAN_SHUFFLE_HPP_
