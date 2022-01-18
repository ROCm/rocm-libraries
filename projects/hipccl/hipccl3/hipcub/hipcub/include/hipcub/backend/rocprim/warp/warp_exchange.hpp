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

BEGIN_HIPCUB_NAMESPACE

template <
    typename InputT,
    int ITEMS_PER_THREAD,
    int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS,
    int ARCH = HIPCUB_ARCH
>
class WarpExchange
{
    static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE,
        "LOGICAL_WARP_THREADS must be a power of two");
    
    constexpr static int SMEM_BANKS = ::rocprim::detail::get_lds_banks_no();

    constexpr static bool HAS_BANK_CONFLICTS =
        ITEMS_PER_THREAD > 4 && PowerOfTwo<ITEMS_PER_THREAD>::VALUE;

    constexpr static int BANK_CONFLICTS_PADDING =
        HAS_BANK_CONFLICTS ? (ITEMS_PER_THREAD / SMEM_BANKS) : 0;

    constexpr static int ITEMS_PER_TILE =
        ITEMS_PER_THREAD * LOGICAL_WARP_THREADS + BANK_CONFLICTS_PADDING;

    constexpr static bool IS_ARCH_WARP = LOGICAL_WARP_THREADS ==
                                         HIPCUB_DEVICE_WARP_THREADS;

    union _TempStorage
    {
        InputT items_shared[ITEMS_PER_TILE];
    };

    _TempStorage &temp_storage;
    unsigned lane_id;
   
public:
    struct TempStorage : Uninitialized<_TempStorage> {};

    WarpExchange() = delete;

    explicit HIPCUB_DEVICE __forceinline__
    WarpExchange(TempStorage &temp_storage) :
        temp_storage(temp_storage.Alias()),
        lane_id(IS_ARCH_WARP ? LaneId() : LaneId() % LOGICAL_WARP_THREADS)
    {
    }

    template <typename OutputT>
    HIPCUB_DEVICE __forceinline__
    void BlockedToStriped(
        const InputT (&input_items)[ITEMS_PER_THREAD],
        OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
            const int idx = ITEMS_PER_THREAD * lane_id + item;
            temp_storage.items_shared[idx] = input_items[item];
        }

        // member mask is unused in rocPRIM
        WARP_SYNC(0);

        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
            const int idx = LOGICAL_WARP_THREADS * item + lane_id;
            output_items[item] = temp_storage.items_shared[idx];
        }
    }

    template <typename OutputT>
    HIPCUB_DEVICE __forceinline__
    void StripedToBlocked(
        const InputT (&input_items)[ITEMS_PER_THREAD],
        OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
            const int idx = LOGICAL_WARP_THREADS * item + lane_id;
            temp_storage.items_shared[idx] = input_items[item];
        }

        // member mask is unused in rocPRIM
        WARP_SYNC(0);

        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
            const int idx = ITEMS_PER_THREAD * lane_id + item;
            output_items[item] = temp_storage.items_shared[idx];
        }
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
        ROCPRIM_UNROLL
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
            temp_storage.items_shared[ranks[item]] = input_items[item];
        }

        // member mask is unused in rocPRIM
        WARP_SYNC(0);
        
        ROCPRIM_UNROLL
        for (int item = 0; item < ITEMS_PER_THREAD; item++)
        {
            int item_offset = (item * LOGICAL_WARP_THREADS) + lane_id;
            output_items[item] = temp_storage.items_shared[item_offset];
        }
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WARP_WARP_EXCHANGE_HPP_
