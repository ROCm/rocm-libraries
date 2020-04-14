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

#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_DPP_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_DPP_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"
#include "../../detail/various.hpp"

#include "warp_reduce_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize,
    bool UseAllReduce
>
class warp_reduce_dpp
{
public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input, T& output, BinaryFunction reduce_op)
    {
        output = input;

        if(WarpSize > 1)
        {
            // quad_perm:[1,0,3,2] -> 10110001
            output = reduce_op(warp_move_dpp<T, 0xb1>(output), output);
        }
        if(WarpSize > 2)
        {
            // quad_perm:[2,3,0,1] -> 01001110
            output = reduce_op(warp_move_dpp<T, 0x4e>(output), output);
        }
        if(WarpSize > 4)
        {
            // row_shr:4
            output = reduce_op(warp_move_dpp<T, 0x114>(output), output);
        }
        if(WarpSize > 8)
        {
            // row_shr:8
            output = reduce_op(warp_move_dpp<T, 0x118>(output), output);
        }
        if(WarpSize > 16)
        {
            // row_bcast:15
            output = reduce_op(warp_move_dpp<T, 0x142>(output), output);
        }
        if(WarpSize > 32)
        {
            // row_bcast:31
            output = reduce_op(warp_move_dpp<T, 0x143>(output), output);
        }

        // Read the result from the last lane of the logical warp
        output = warp_shuffle(output, WarpSize - 1, WarpSize);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        (void) storage; // disables unused parameter warning
        this->reduce(input, output, reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input, T& output, unsigned int valid_items, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .reduce(input, output, valid_items, reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void reduce(T input, T& output, unsigned int valid_items,
                storage_type& storage, BinaryFunction reduce_op)
    {
        (void) storage; // disables unused parameter warning
        this->reduce(input, output, valid_items, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void head_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .head_segmented_reduce(input, output, flag, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void tail_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .tail_segmented_reduce(input, output, flag, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void head_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .head_segmented_reduce(input, output, flag, storage, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void tail_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {
        // Fallback to shuffle-based implementation
        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .tail_segmented_reduce(input, output, flag, storage, reduce_op);
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_WARP_DETAIL_WARP_REDUCE_DPP_HPP_
