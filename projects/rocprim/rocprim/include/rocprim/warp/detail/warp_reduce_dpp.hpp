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

#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_DPP_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_DPP_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "warp_reduce_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T, unsigned int VirtualWaveSize, bool UseAllReduce>
class warp_reduce_dpp
{
public:
    static_assert(detail::is_power_of_two(VirtualWaveSize), "VirtualWaveSize must be power of 2");

    using storage_type = detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce_impl(T input, T& output, BinaryFunction reduce_op, std::false_type)
    {
        output = input;

        if(VirtualWaveSize > 1)
        {
            // quad_perm:[1,0,3,2] -> 10110001
            output = reduce_op(warp_move_dpp<T, 0xb1>(output), output);
        }
        if(VirtualWaveSize > 2)
        {
            // quad_perm:[2,3,0,1] -> 01001110
            output = reduce_op(warp_move_dpp<T, 0x4e>(output), output);
        }
        if(VirtualWaveSize > 4)
        {
            // row_ror:4
            // Use rotation instead of shift to avoid leaving invalid values in the destination
            // registers (asume warp size of at least hardware warp-size)
            output = reduce_op(warp_move_dpp<T, 0x124>(output), output);
        }
        if(VirtualWaveSize > 8)
        {
            // row_ror:8
            // Use rotation instead of shift to avoid leaving invalid values in the destination
            // registers (asume warp size of at least hardware warp-size)
            output = reduce_op(warp_move_dpp<T, 0x128>(output), output);
        }
#ifdef ROCPRIM_DETAIL_HAS_DPP_BROADCAST
        if(VirtualWaveSize > 16)
        {
            // row_bcast:15
            output = reduce_op(warp_move_dpp<T, 0x142>(output), output);
        }
        if(VirtualWaveSize > 32)
        {
            // row_bcast:31
            output = reduce_op(warp_move_dpp<T, 0x143>(output), output);
        }
        static_assert(VirtualWaveSize <= 64, "VirtualWaveSize > 64 is not supported");
#else
        if(VirtualWaveSize > 16)
        {
            // row_bcast:15
            output = reduce_op(warp_swizzle<T, 0x1e0>(output), output);
        }
        static_assert(VirtualWaveSize <= 32,
                      "VirtualWaveSize > 32 is not supported without DPP broadcasts");
#endif
        // Read the result from the last lane of the logical warp
        output = warp_shuffle(output, VirtualWaveSize - 1, VirtualWaveSize);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce_impl(T input, T& output, BinaryFunction reduce_op, std::true_type)
    {
        warp_reduce_shuffle<T, VirtualWaveSize, UseAllReduce>().reduce(input, output, reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, BinaryFunction reduce_op)
    {
        reduce_impl(
            input,
            output,
            reduce_op,
            std::integral_constant<bool,
                                   (VirtualWaveSize < ::rocprim::arch::wavefront::min_size())>{});
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        (void)storage; // disables unused parameter warning
        this->reduce(input, output, reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, unsigned int valid_items, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, VirtualWaveSize, UseAllReduce>().reduce(input,
                                                                       output,
                                                                       valid_items,
                                                                       reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T              input,
                T&             output,
                unsigned int   valid_items,
                storage_type&  storage,
                BinaryFunction reduce_op)
    {
        (void)storage; // disables unused parameter warning
        this->reduce(input, output, valid_items, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, VirtualWaveSize, UseAllReduce>().head_segmented_reduce(input,
                                                                                      output,
                                                                                      flag,
                                                                                      reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, VirtualWaveSize, UseAllReduce>().tail_segmented_reduce(input,
                                                                                      output,
                                                                                      flag,
                                                                                      reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(
        T input, T& output, Flag flag, storage_type& storage, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, VirtualWaveSize, UseAllReduce>().head_segmented_reduce(input,
                                                                                      output,
                                                                                      flag,
                                                                                      storage,
                                                                                      reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(
        T input, T& output, Flag flag, storage_type& storage, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, VirtualWaveSize, UseAllReduce>().tail_segmented_reduce(input,
                                                                                      output,
                                                                                      flag,
                                                                                      storage,
                                                                                      reduce_op);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_REDUCE_DPP_HPP_
