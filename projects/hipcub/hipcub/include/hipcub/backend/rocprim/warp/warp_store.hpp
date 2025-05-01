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

#ifndef HIPCUB_ROCPRIM_WARP_WARP_STORE_HPP_
#define HIPCUB_ROCPRIM_WARP_WARP_STORE_HPP_

#include "../../../config.hpp"

#include "../util_type.hpp"
#include "./warp_exchange.hpp"

#include <rocprim/block/block_store_func.hpp>

BEGIN_HIPCUB_NAMESPACE

enum WarpStoreAlgorithm
{
    WARP_STORE_DIRECT,
    WARP_STORE_STRIPED,
    WARP_STORE_VECTORIZE,
    WARP_STORE_TRANSPOSE
};

template<
    class T,
    int ITEMS_PER_THREAD,
    WarpStoreAlgorithm ALGORITHM = WARP_STORE_DIRECT,
    int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS,
    int ARCH = HIPCUB_ARCH
>
class WarpStore
{
private:
    constexpr static bool IS_ARCH_WARP 
        = static_cast<unsigned>(LOGICAL_WARP_THREADS) == HIPCUB_DEVICE_WARP_THREADS;

    template <WarpStoreAlgorithm _POLICY>
    struct StoreInternal;

    template <>
    struct StoreInternal<WARP_STORE_DIRECT>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__ StoreInternal(
            TempStorage & /*temp_storage*/,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
            OutputIteratorT block_itr,
            T (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_store_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
            OutputIteratorT block_itr,
            T (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_store_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }
    };

    template <>
    struct StoreInternal<WARP_STORE_STRIPED>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__ StoreInternal(
            TempStorage & /*temp_storage*/,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
            OutputIteratorT block_itr,
            T (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_store_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
            OutputIteratorT block_itr,
            T (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_store_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }
    };

    template <>
    struct StoreInternal<WARP_STORE_VECTORIZE>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__ StoreInternal(
            TempStorage & /*temp_storage*/,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
            T *block_ptr,
            T (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_store_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_ptr,
                items
            );
        }

        template <typename _OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
            _OutputIteratorT block_itr,
            T (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_store_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
            OutputIteratorT block_itr,
            T (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            // vectorized overload does not exist
            // fall back to direct blocked
            ::rocprim::block_store_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }
    };

    template <>
    struct StoreInternal<WARP_STORE_TRANSPOSE>
    {
        using WarpExchangeT = WarpExchange<
            T,
            ITEMS_PER_THREAD,
            LOGICAL_WARP_THREADS,
            ARCH
        >;
        using TempStorage = typename WarpExchangeT::TempStorage;
        TempStorage& temp_storage;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__ StoreInternal(
            TempStorage &temp_storage,
            int linear_tid) :
            temp_storage(temp_storage),
            linear_tid(linear_tid)
        {
        }

        template <typename OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
            OutputIteratorT block_itr,
            T (&items)[ITEMS_PER_THREAD])
        {
            WarpExchangeT(temp_storage).BlockedToStriped(items, items);
            ::rocprim::block_store_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename OutputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Store(
        OutputIteratorT block_itr,
        T (&items)[ITEMS_PER_THREAD],
        int valid_items)
        {
            WarpExchangeT(temp_storage).BlockedToStriped(items, items);
            ::rocprim::block_store_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );

        }
    };

    using InternalStore = StoreInternal<ALGORITHM>;

    using _TempStorage = typename InternalStore::TempStorage;

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
    WarpStore() :
        temp_storage(PrivateStorage()),
        linear_tid(IS_ARCH_WARP ? ::rocprim::lane_id() : (::rocprim::lane_id() % LOGICAL_WARP_THREADS))
    {
    }

    HIPCUB_DEVICE __forceinline__
    WarpStore(TempStorage &temp_storage) :
        temp_storage(temp_storage.Alias()),
        linear_tid(IS_ARCH_WARP ? ::rocprim::lane_id() : (::rocprim::lane_id() % LOGICAL_WARP_THREADS))
    {
    }

    template <typename OutputIteratorT>
    HIPCUB_DEVICE __forceinline__ void Store(
        OutputIteratorT block_itr,
        T (&items)[ITEMS_PER_THREAD])
    {
        InternalStore(temp_storage, linear_tid)
            .Store(block_itr, items);
    }

    template <typename OutputIteratorT>
    HIPCUB_DEVICE __forceinline__ void Store(
        OutputIteratorT block_itr,
        T (&items)[ITEMS_PER_THREAD],
        int valid_items)
    {
        InternalStore(temp_storage, linear_tid)
            .Store(block_itr, items, valid_items);
    }

    template <typename OutputIteratorT,
              typename DefaultT>
    HIPCUB_DEVICE __forceinline__ void Store(
        OutputIteratorT block_itr,
        T (&items)[ITEMS_PER_THREAD],
        int valid_items,
        DefaultT oob_default)
    {
        InternalStore(temp_storage, linear_tid)
            .Store(block_itr, items, valid_items, oob_default);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WARP_WARP_STORE_HPP_
