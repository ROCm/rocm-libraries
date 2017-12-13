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

/// \addtogroup collectiveblockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

template<
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_store_direct_blocked(int flat_id, IteratorT block_iter,
                                T (&items)[ItemsPerThread]) [[hc]]
{
    int offset = flat_id * ItemsPerThread;
    IteratorT thread_iter = block_iter + offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        thread_iter[item] = items[item];
    }
}

template<
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_store_direct_blocked(int flat_id, IteratorT block_iter,
                                T (&items)[ItemsPerThread],
                                int valid) [[hc]]
{
    int offset = flat_id * ItemsPerThread;
    IteratorT thread_iter = block_iter + offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        if (item + offset < valid)
        {
            thread_iter[item] = items[item];
        }
    }
}

template<
    class T,
    int ItemsPerThread
>
typename std::enable_if<detail::is_vectorizable<T, ItemsPerThread>()>::type
block_store_direct_blocked_vectorized(int flat_id, T* block_iter,
                                      T (&items)[ItemsPerThread]) [[hc]]
{
    typedef typename detail::match_vector_type<T, ItemsPerThread>::type Vector;
    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(Vector);
    Vector *vectors_ptr = reinterpret_cast<Vector*>(const_cast<T*>(block_iter));

    Vector raw_vector_items[vectors_per_thread];
    T *raw_items = reinterpret_cast<T*>(raw_vector_items);

    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        raw_items[item] = items[item];
    }
    
    block_store_direct_blocked(flat_id, vectors_ptr, raw_vector_items);
}

template<
    class T,
    int ItemsPerThread
>
typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>()>::type
block_store_direct_blocked_vectorized(int flat_id, T* block_iter,
                                      T (&items)[ItemsPerThread]) [[hc]]
{
    block_store_direct_blocked(flat_id, block_iter, items);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group collectiveblockmodule

#endif // ROCPRIM_BLOCK_BLOCK_STORE_HPP_
