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

#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHARED_MEM_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHARED_MEM_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "warp_segment_bounds.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T, unsigned int VirtualWaveSize, bool UseAllReduce>
class warp_reduce_shared_mem
{
    struct storage_type_
    {
        T values[VirtualWaveSize];
    };

public:
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        constexpr unsigned int ceiling  = next_power_of_two(VirtualWaveSize);
        const unsigned int     lid      = detail::logical_lane_id<VirtualWaveSize>();
        storage_type_&         storage_ = storage.get();

        output               = input;
        storage_.values[lid] = output;
        ::rocprim::wave_barrier();
        ROCPRIM_UNROLL
        for(unsigned int i = ceiling >> 1; i > 0; i >>= 1)
        {
            const bool do_op = lid + i < VirtualWaveSize && lid < i;
            if(do_op)
            {
                output  = storage_.values[lid];
                T other = storage_.values[lid + i];
                output  = reduce_op(output, other);
            }
            ::rocprim::wave_barrier();
            if(do_op)
            {
                storage_.values[lid] = output;
            }
            ::rocprim::wave_barrier();
        }
        set_output<UseAllReduce>(output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T              input,
                T&             output,
                unsigned int   valid_items,
                storage_type&  storage,
                BinaryFunction reduce_op)
    {
        constexpr unsigned int ceiling  = next_power_of_two(VirtualWaveSize);
        const unsigned int     lid      = detail::logical_lane_id<VirtualWaveSize>();
        storage_type_&         storage_ = storage.get();

        output               = input;
        storage_.values[lid] = output;
        ::rocprim::wave_barrier();
        ROCPRIM_UNROLL
        for(unsigned int i = ceiling >> 1; i > 0; i >>= 1)
        {
            const bool do_op = (lid + i) < VirtualWaveSize && lid < i && (lid + i) < valid_items;
            if(do_op)
            {
                output  = storage_.values[lid];
                T other = storage_.values[lid + i];
                output  = reduce_op(output, other);
            }
            ::rocprim::wave_barrier();
            if(do_op)
            {
                storage_.values[lid] = output;
            }
            ::rocprim::wave_barrier();
        }
        set_output<UseAllReduce>(output, storage);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(
        T input, T& output, Flag flag, storage_type& storage, BinaryFunction reduce_op)
    {
        this->segmented_reduce<true>(input, output, flag, storage, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(
        T input, T& output, Flag flag, storage_type& storage, BinaryFunction reduce_op)
    {
        this->segmented_reduce<false>(input, output, flag, storage, reduce_op);
    }

private:
    template<bool HeadSegmented, class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void segmented_reduce(
        T input, T& output, Flag flag, storage_type& storage, BinaryFunction reduce_op)
    {
        const unsigned int     lid      = detail::logical_lane_id<VirtualWaveSize>();
        constexpr unsigned int ceiling  = next_power_of_two(VirtualWaveSize);
        storage_type_&         storage_ = storage.get();
        // Get logical lane id of the last valid value in the segment
        auto last = last_in_warp_segment<HeadSegmented, VirtualWaveSize>(flag);

        output = input;
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ceiling; i *= 2)
        {
            storage_.values[lid] = output;
            ::rocprim::wave_barrier();
            if((lid + i) <= last)
            {
                T other = storage_.values[lid + i];
                output  = reduce_op(output, other);
            }
            ::rocprim::wave_barrier();
        }
    }

    template<bool Switch>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(Switch == false)>::type set_output(T& output, storage_type& storage)
    {
        (void)output;
        (void)storage;
        // output already set correctly
    }

    template<bool Switch>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(Switch == true)>::type set_output(T& output, storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        output                  = storage_.values[0];
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHARED_MEM_HPP_
