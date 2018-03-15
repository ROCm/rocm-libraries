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

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_

#include "../../config.hpp"

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int BLOCK_DIM_X,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH /* ignored */
>
class BlockDiscontinuity
    : private ::rocprim::block_discontinuity<
        T,
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_discontinuity<
            T,
            BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z
        >;

    // Reference to temporary storage (usually shared memory)
    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockDiscontinuity() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockDiscontinuity(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeads(FlagT (&head_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op)
    {
        base_type::flag_heads(head_flags, input, flag_op, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeads(FlagT (&head_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op,
                   T tile_predecessor_item)
    {
        base_type::flag_heads(head_flags, tile_predecessor_item, input, flag_op, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagTails(FlagT (&tail_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op)
    {
        base_type::flag_tails(tail_flags, input, flag_op, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagTails(FlagT (&tail_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op,
                   T tile_successor_item)
    {
        base_type::flag_tails(tail_flags, tile_successor_item, input, flag_op, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        base_type::flag_heads_and_tails(
            head_flags, tail_flags, input,
            flag_op, temp_storage_
        );
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T tile_successor_item,
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        base_type::flag_heads_and_tails(
            head_flags, tail_flags, tile_successor_item, input,
            flag_op, temp_storage_
        );
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           T tile_predecessor_item,
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        base_type::flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, input,
            flag_op, temp_storage_
        );
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           T tile_predecessor_item,
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T tile_successor_item,
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        base_type::flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input,
            flag_op, temp_storage_
        );
    }

private:
    HIPCUB_DEVICE inline
    TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_
