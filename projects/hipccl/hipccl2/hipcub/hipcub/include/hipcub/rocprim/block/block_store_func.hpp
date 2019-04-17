/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_

#include "../../config.hpp"

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectBlocked(int linear_id,
                        OutputIteratorT block_iter,
                        T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_store_direct_blocked(
        linear_id, block_iter, items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectBlocked(int linear_id,
                        OutputIteratorT block_iter,
                        T (&items)[ITEMS_PER_THREAD],
                        int valid_items)
{
    ::rocprim::block_store_direct_blocked(
        linear_id, block_iter, items, valid_items
    );
}

template <
    typename T,
    int ITEMS_PER_THREAD
>
HIPCUB_DEVICE inline
void StoreDirectBlockedVectorized(int linear_id,
                                  T* block_iter,
                                  T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_store_direct_blocked_vectorized(
        linear_id, block_iter, items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectStriped(int linear_id,
                        OutputIteratorT block_iter,
                        T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_store_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectStriped(int linear_id,
                        OutputIteratorT block_iter,
                        T (&items)[ITEMS_PER_THREAD],
                        int valid_items)
{
    ::rocprim::block_store_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items, valid_items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectWarpStriped(int linear_id,
                            OutputIteratorT block_iter,
                            T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_store_direct_warp_striped(
        linear_id, block_iter, items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectWarpStriped(int linear_id,
                            OutputIteratorT block_iter,
                            T (&items)[ITEMS_PER_THREAD],
                            int valid_items)
{
    ::rocprim::block_store_direct_warp_striped(
        linear_id, block_iter, items, valid_items
    );
}

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_
