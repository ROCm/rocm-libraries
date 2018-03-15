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

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_

#include "../../config.hpp"

BEGIN_HIPCUB_NAMESPACE

template<
    typename InputT,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    bool WARP_TIME_SLICING = false, /* ignored */
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH /* ignored */
>
class BlockExchange
    : private ::rocprim::block_exchange<
        InputT,
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
        ITEMS_PER_THREAD
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_exchange<
            InputT,
            BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
            ITEMS_PER_THREAD
        >;

    // Reference to temporary storage (usually shared memory)
    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockExchange() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockExchange(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<typename OutputT>
    HIPCUB_DEVICE inline
    void StripedToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                          OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type::striped_to_blocked(input_items, output_items, temp_storage_);
    }

    template<typename OutputT>
    HIPCUB_DEVICE inline
    void BlockedToStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                          OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type::blocked_to_striped(input_items, output_items, temp_storage_);
    }

    template<typename OutputT>
    HIPCUB_DEVICE inline
    void WarpStripedToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                              OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type::warp_striped_to_blocked(input_items, output_items, temp_storage_);
    }

    template<typename OutputT>
    HIPCUB_DEVICE inline
    void BlockedToWarpStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                              OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type::blocked_to_warp_striped(input_items, output_items, temp_storage_);
    }

    template<typename OutputT, typename OffsetT>
    HIPCUB_DEVICE inline
    void ScatterToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                          OutputT (&output_items)[ITEMS_PER_THREAD],
                          OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        base_type::scatter_to_blocked(input_items, output_items, ranks, temp_storage_);
    }

    template<typename OutputT, typename OffsetT>
    HIPCUB_DEVICE inline
    void ScatterToStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                          OutputT (&output_items)[ITEMS_PER_THREAD],
                          OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        base_type::scatter_to_striped(input_items, output_items, ranks, temp_storage_);
    }

    template<typename OutputT, typename OffsetT>
    HIPCUB_DEVICE inline
    void ScatterToStripedGuarded(InputT (&input_items)[ITEMS_PER_THREAD],
                                 OutputT (&output_items)[ITEMS_PER_THREAD],
                                 OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        base_type::scatter_to_striped_guarded(input_items, output_items, ranks, temp_storage_);
    }

    template<typename OutputT, typename OffsetT, typename ValidFlag>
    HIPCUB_DEVICE inline
    void ScatterToStripedFlagged(InputT (&input_items)[ITEMS_PER_THREAD],
                                 OutputT (&output_items)[ITEMS_PER_THREAD],
                                 OffsetT (&ranks)[ITEMS_PER_THREAD],
                                 ValidFlag (&is_valid)[ITEMS_PER_THREAD])
    {
        base_type::scatter_to_striped_flagged(input_items, output_items, ranks, is_valid, temp_storage_);
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

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
