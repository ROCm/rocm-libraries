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

#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHUFFLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHUFFLE_HPP_

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
    unsigned int WarpSize,
    bool UseAllReduce
>
class warp_reduce_shuffle
{
public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = detail::empty_storage_type;

template<class BinaryFunction>
    void reduce(T input, T& output,
                storage_type& storage, BinaryFunction reduce_op) [[hc]]
    {
        (void) storage; // disables unused parameter warning

        output = input;

        T value;
        #pragma unroll
        for (unsigned int offset = WarpSize >> 1; offset > 0; offset >>= 1)
        {
            value = warp_shuffle_down(output, offset, WarpSize);
            output = reduce_op(output, value);
        }
        set_output<UseAllReduce>(output);
    }

    template<class BinaryFunction>
    void reduce(T input, T& output, unsigned int valid_items,
                storage_type& storage, BinaryFunction reduce_op) [[hc]]
    {
        (void) storage; // disables unused parameter warning

        output = input;

        T value;
        #pragma unroll
        for (unsigned int offset = WarpSize >> 1; offset > 0; offset >>= 1)
        {
            value = warp_shuffle_down(output, offset, WarpSize);
            unsigned int id = detail::logical_lane_id<WarpSize>();
            if (id + offset < valid_items) output = reduce_op(output, value);
        }
        set_output<UseAllReduce>(output);
    }

private:
    template<bool Switch>
    typename std::enable_if<(Switch == false)>::type
    set_output(T& output) [[hc]]
    {
        (void) output;
        // output already set correctly
    }

    template<bool Switch>
    typename std::enable_if<(Switch == true)>::type
    set_output(T& output) [[hc]]
    {
        output = warp_shuffle(output, 0, WarpSize);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHUFFLE_HPP_
