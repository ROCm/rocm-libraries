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

#ifndef ROCPRIM_BLOCK_BLOCK_STORE_HPP_
#define ROCPRIM_BLOCK_BLOCK_STORE_HPP_

// HC API
#include <hcc/hc.hpp>
#include <hcc/hc_short_vector.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "block_store_func.hpp"

/// \addtogroup collectiveblockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

enum block_store_method
{
    block_store_direct,
    block_store_vectorize,
    block_store_transpose,
    block_store_warp_transpose
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    block_store_method Method = block_store_direct
>
class block_store
{
public:
    typedef typename detail::empty_type storage_type;

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread]) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked(flat_id, block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked(flat_id, block_output, items, valid);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage) [[hc]]
    {
        (void) storage;
        store(block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage) [[hc]]
    {
        (void) storage;
        store(block_output, items, valid);
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_store<T, BlockSize, ItemsPerThread, block_store_vectorize>
{
public:
    typedef typename detail::empty_type storage_type;

    template<class U>
    void store(T* block_output,
               U (&items)[ItemsPerThread]) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked_vectorized(flat_id, block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread]) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked(flat_id, block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked(flat_id, block_output, items, valid);
    }

    template<class U>
    void store(T* block_output,
               U (&items)[ItemsPerThread],
               storage_type& storage) [[hc]]
    {
        (void) storage;
        store(block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage) [[hc]]
    {
        (void) storage;
        store(block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage) [[hc]]
    {
        (void) storage;
        store(block_output, items, valid);
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_store<T, BlockSize, ItemsPerThread, block_store_transpose>
{
public:
    block_exchange<T, BlockSize, ItemsPerThread> exchange;

    struct storage_type
    {
        volatile int valid;
    };

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread]) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        exchange.blocked_to_striped(items, items);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        exchange.blocked_to_striped(items, items);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items, valid);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        exchange.blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        exchange.blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items, valid);
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_store<T, BlockSize, ItemsPerThread, block_store_warp_transpose>
{
public:
    static_assert(BlockSize % warp_size() == 0,
                 "BlockSize must be a multiple of hardware warpsize");
    block_exchange<T, BlockSize, ItemsPerThread> exchange;

    struct storage_type
    {
        volatile int valid;
    };

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread]) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        exchange.blocked_to_warp_striped(items, items);
        block_store_direct_warp_striped(flat_id, block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        exchange.blocked_to_warp_striped(items, items);
        block_store_direct_warp_striped(flat_id, block_output, items, valid);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        exchange.blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items);
    }

    template<class IteratorT>
    void store(IteratorT block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage) [[hc]]
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        exchange.blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items, valid);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group collectiveblockmodule

#endif // ROCPRIM_BLOCK_BLOCK_STORE_HPP_
