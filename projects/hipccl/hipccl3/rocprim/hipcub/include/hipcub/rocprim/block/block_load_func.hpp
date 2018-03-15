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

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_

#include "../../config.hpp"

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectBlocked(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_load_direct_blocked(
        linear_id, block_iter, items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectBlocked(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD],
                       int valid_items)
{
    ::rocprim::block_load_direct_blocked(
        linear_id, block_iter, items, valid_items
    );
}

template<
    typename T,
    typename Default,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectBlocked(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD],
                       int valid_items,
                       Default oob_default)
{
    ::rocprim::block_load_direct_blocked(
        linear_id, block_iter, items, valid_items, oob_default
    );
}

template <
    typename T,
    int ITEMS_PER_THREAD
>
HIPCUB_DEVICE inline
void LoadDirectBlockedVectorized(int linear_id,
                                 T* block_iter,
                                 T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_load_direct_blocked_vectorized(
        linear_id, block_iter, items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectStriped(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_load_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectStriped(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD],
                       int valid_items)
{
    ::rocprim::block_load_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items, valid_items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    typename Default,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectStriped(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD],
                       int valid_items,
                       Default oob_default)
{
    ::rocprim::block_load_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items, valid_items, oob_default
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectWarpStriped(int linear_id,
                           InputIteratorT block_iter,
                           T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_load_direct_warp_striped(
        linear_id, block_iter, items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectWarpStriped(int linear_id,
                           InputIteratorT block_iter,
                           T (&items)[ITEMS_PER_THREAD],
                           int valid_items)
{
    ::rocprim::block_load_direct_warp_striped(
        linear_id, block_iter, items, valid_items
    );
}

template<
    typename T,
    typename Default,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectWarpStriped(int linear_id,
                           InputIteratorT block_iter,
                           T (&items)[ITEMS_PER_THREAD],
                           int valid_items,
                           Default oob_default)
{
    ::rocprim::block_load_direct_warp_striped(
        linear_id, block_iter, items, valid_items, oob_default
    );
}

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_
