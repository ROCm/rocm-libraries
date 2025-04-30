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

#ifndef HIPCUB_ROCPRIM_WARP_WARP_LOAD_HPP_
#define HIPCUB_ROCPRIM_WARP_WARP_LOAD_HPP_

#include "../../../config.hpp"

#include "../util_type.hpp"
#include "../iterator/cache_modified_input_iterator.hpp"
#include "./warp_exchange.hpp"

#include <rocprim/block/block_load_func.hpp>

BEGIN_HIPCUB_NAMESPACE

enum WarpLoadAlgorithm
{
    WARP_LOAD_DIRECT,
    WARP_LOAD_STRIPED,
    WARP_LOAD_VECTORIZE,
    WARP_LOAD_TRANSPOSE
};

template<
    class InputT,
    int ITEMS_PER_THREAD,
    WarpLoadAlgorithm ALGORITHM = WARP_LOAD_DIRECT,
    int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS,
    int ARCH = HIPCUB_ARCH
>
class WarpLoad
{
private:
    constexpr static bool IS_ARCH_WARP 
        = static_cast<unsigned>(LOGICAL_WARP_THREADS) == HIPCUB_DEVICE_WARP_THREADS;

    template <WarpLoadAlgorithm _POLICY>
    struct LoadInternal;

    template <>
    struct LoadInternal<WARP_LOAD_DIRECT>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__
        LoadInternal(
            TempStorage & /*temp_storage*/,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_load_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }

        template <typename InputIteratorT, typename DefaultT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items,
            DefaultT oob_default)
        {
            ::rocprim::block_load_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items),
                oob_default
            );
        }
    };

    template <>
    struct LoadInternal<WARP_LOAD_STRIPED>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__
        LoadInternal(
            TempStorage & /*temp_storage*/,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }

        template <typename InputIteratorT, typename DefaultT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items,
            DefaultT oob_default)
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items),
                oob_default
            );
        }
    };

    template <>
    struct LoadInternal<WARP_LOAD_VECTORIZE>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__ LoadInternal(
            TempStorage & /*temp_storage*/,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputT *block_ptr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_ptr,
                items
            );
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            const InputT *block_ptr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_ptr,
                items
            );
        }

        template<
            CacheLoadModifier MODIFIER,
            typename ValueType,
            typename OffsetT
        >
        HIPCUB_DEVICE __forceinline__ void Load(
            CacheModifiedInputIterator<MODIFIER, ValueType, OffsetT> block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename _InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            _InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }

        template <typename InputIteratorT, typename DefaultT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items,
            DefaultT oob_default)
        {
            // vectorized overload does not exist
            // fall back to direct blocked
            ::rocprim::block_load_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items),
                oob_default
            );
        }
    };

    template <>
    struct LoadInternal<WARP_LOAD_TRANSPOSE>
    {
        using WarpExchangeT = WarpExchange<
            InputT,
            ITEMS_PER_THREAD,
            LOGICAL_WARP_THREADS,
            ARCH
        >;
        using TempStorage = typename WarpExchangeT::TempStorage;
        TempStorage& temp_storage;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__ LoadInternal(
            TempStorage &temp_storage,
            int linear_tid) :
            temp_storage(temp_storage),
            linear_tid(linear_tid)
        {
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
            WarpExchangeT(temp_storage).StripedToBlocked(items, items);
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
            WarpExchangeT(temp_storage).StripedToBlocked(items, items);
        }

        template <typename InputIteratorT, typename DefaultT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items,
            DefaultT oob_default)
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items),
                oob_default
            );
            WarpExchangeT(temp_storage).StripedToBlocked(items, items);
        }
    };

    using InternalLoad = LoadInternal<ALGORITHM>;

    using _TempStorage = typename InternalLoad::TempStorage;

    HIPCUB_DEVICE __forceinline__ _TempStorage &PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }

    _TempStorage &temp_storage;
    int linear_tid;

public:
    struct TempStorage : Uninitialized<_TempStorage>
    {
    };

    HIPCUB_DEVICE __forceinline__
    WarpLoad() :
        temp_storage(PrivateStorage()),
        linear_tid(IS_ARCH_WARP ? ::rocprim::lane_id() : (::rocprim::lane_id() % LOGICAL_WARP_THREADS))
    {
    }

    HIPCUB_DEVICE __forceinline__
    WarpLoad(TempStorage &temp_storage) :
        temp_storage(temp_storage.Alias()),
        linear_tid(IS_ARCH_WARP ? ::rocprim::lane_id() : (::rocprim::lane_id() % LOGICAL_WARP_THREADS))
    {
    }

    template <typename InputIteratorT>
    HIPCUB_DEVICE __forceinline__ void Load(
        InputIteratorT block_itr,
        InputT (&items)[ITEMS_PER_THREAD])
    {
        InternalLoad(temp_storage, linear_tid)
            .Load(block_itr, items);
    }

    template <typename InputIteratorT>
    HIPCUB_DEVICE __forceinline__ void Load(
        InputIteratorT block_itr,
        InputT (&items)[ITEMS_PER_THREAD],
        int valid_items)
    {
        InternalLoad(temp_storage, linear_tid)
            .Load(block_itr, items, valid_items);
    }

    template <typename InputIteratorT,
              typename DefaultT>
    HIPCUB_DEVICE __forceinline__ void Load(
        InputIteratorT block_itr,
        InputT (&items)[ITEMS_PER_THREAD],
        int valid_items,
        DefaultT oob_default)
    {
        InternalLoad(temp_storage, linear_tid)
            .Load(block_itr, items, valid_items, oob_default);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WARP_WARP_LOAD_HPP_
