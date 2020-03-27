// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHARED_MEM_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHARED_MEM_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"
#include "../../detail/various.hpp"

#include "warp_segment_bounds.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize,
    bool UseAllReduce
>
class warp_reduce_shared_mem
{
    struct storage_type_
    {
        T values[WarpSize];
    };

public:
    using storage_type = detail::raw_storage<storage_type_>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        constexpr unsigned int ceiling = next_power_of_two(WarpSize);
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        output = input;
        store_volatile(&storage_.values[lid], output);
        #pragma unroll
        for(unsigned int i = ceiling >> 1; i > 0; i >>= 1)
        {
            if (lid + i < WarpSize && lid < i)
            {
                output = load_volatile(&storage_.values[lid]);
                T other = load_volatile(&storage_.values[lid + i]);
                output = reduce_op(output, other);
                store_volatile(&storage_.values[lid], output);
            }
        }
        set_output<UseAllReduce>(output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input, T& output, unsigned int valid_items,
                storage_type& storage, BinaryFunction reduce_op)
    {
        constexpr unsigned int ceiling = next_power_of_two(WarpSize);
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        output = input;
        store_volatile(&storage_.values[lid], output);
        #pragma unroll
        for(unsigned int i = ceiling >> 1; i > 0; i >>= 1)
        {
            if((lid + i) < WarpSize && lid < i && (lid + i) < valid_items)
            {
                output = load_volatile(&storage_.values[lid]);
                T other = load_volatile(&storage_.values[lid + i]);
                output = reduce_op(output, other);
                store_volatile(&storage_.values[lid], output);
            }
        }
        set_output<UseAllReduce>(output, storage);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void head_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {
        this->segmented_reduce<true>(input, output, flag, storage, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void tail_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {
        this->segmented_reduce<false>(input, output, flag, storage, reduce_op);
    }

private:
    template<bool HeadSegmented, class Flag, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void segmented_reduce(T input, T& output, Flag flag,
                          storage_type& storage, BinaryFunction reduce_op)
    {
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        constexpr unsigned int ceiling = next_power_of_two(WarpSize);
        storage_type_& storage_ = storage.get();
        // Get logical lane id of the last valid value in the segment
        auto last = last_in_warp_segment<HeadSegmented, WarpSize>(flag);

        output = input;
        #pragma unroll
        for(unsigned int i = 1; i < ceiling; i *= 2)
        {
            store_volatile(&storage_.values[lid], output);
            if((lid + i) <= last)
            {
                T other = load_volatile(&storage_.values[lid + i]);
                output = reduce_op(output, other);
            }
        }
    }

    template<bool Switch>
    ROCPRIM_DEVICE inline
    typename std::enable_if<(Switch == false)>::type
    set_output(T& output, storage_type& storage)
    {
        (void) output;
        (void) storage;
        // output already set correctly
    }

    template<bool Switch>
    ROCPRIM_DEVICE inline
    typename std::enable_if<(Switch == true)>::type
    set_output(T& output, storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        output = load_volatile(&storage_.values[0]);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHARED_MEM_HPP_
