/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_

#include "../../../config.hpp"

#include <rocprim/block/block_exchange.hpp>

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
        BLOCK_DIM_X,
        ITEMS_PER_THREAD,
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_exchange<
            InputT,
            BLOCK_DIM_X,
            ITEMS_PER_THREAD,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
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
