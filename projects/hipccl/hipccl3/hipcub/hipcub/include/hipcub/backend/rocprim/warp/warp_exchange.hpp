/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2021, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_WARP_WARP_EXCHANGE_HPP_
#define HIPCUB_ROCPRIM_WARP_WARP_EXCHANGE_HPP_

#include "../../../config.hpp"
#include "../util_type.hpp"

#include <rocprim/warp/warp_exchange.hpp>

BEGIN_HIPCUB_NAMESPACE

template <
    typename InputT,
    int ITEMS_PER_THREAD,
    int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS,
    int ARCH = HIPCUB_ARCH
>
class WarpExchange
{
    using base_type = typename rocprim::warp_exchange<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>;

public:
    using TempStorage = typename base_type::storage_type;

private:
    TempStorage &temp_storage;

public:
    WarpExchange() = delete;

    explicit HIPCUB_DEVICE __forceinline__
    WarpExchange(TempStorage &temp_storage) :
        temp_storage(temp_storage)
    {
    }

    template <typename OutputT>
    HIPCUB_DEVICE __forceinline__
    void BlockedToStriped(
        const InputT (&input_items)[ITEMS_PER_THREAD],
        OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type rocprim_warp_exchange;
        rocprim_warp_exchange.blocked_to_striped(input_items, output_items, temp_storage);
    }

    template <typename OutputT>
    HIPCUB_DEVICE __forceinline__
    void StripedToBlocked(
        const InputT (&input_items)[ITEMS_PER_THREAD],
        OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type rocprim_warp_exchange;
        rocprim_warp_exchange.striped_to_blocked(input_items, output_items, temp_storage);
    }

    template <typename OffsetT>
    HIPCUB_DEVICE __forceinline__
    void ScatterToStriped(
        InputT (&items)[ITEMS_PER_THREAD],
        OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToStriped(items, items, ranks);
    }

    template <typename OutputT,
              typename OffsetT>
    HIPCUB_DEVICE __forceinline__
    void ScatterToStriped(
        const InputT (&input_items)[ITEMS_PER_THREAD],
        OutputT (&output_items)[ITEMS_PER_THREAD],
        OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        base_type rocprim_warp_exchange;
        rocprim_warp_exchange.scatter_to_striped(input_items, output_items, ranks, temp_storage);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WARP_WARP_EXCHANGE_HPP_
