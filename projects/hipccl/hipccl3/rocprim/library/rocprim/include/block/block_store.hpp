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

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup collectiveblockmodule
/// @{

/// \brief Stores a blocked arrangement of items in memory to thread block.
///
/// Block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of 
/// \p ItemsPerThread \p items to the thread block.
///
/// \tparam IteratorT - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a flat 1D thread identifier for the calling thread
/// \param block_iter - the input iterator from the thread block to store to
/// \param items - array that data is stored to thread block
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

/// \brief Stores a blocked arrangement of items in memory to thread block,
/// which is guarded by range \p valid.
///
/// Block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of 
/// \p ItemsPerThread \p items to the thread block.
///
/// \tparam IteratorT - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a flat 1D thread identifier for the calling thread
/// \param block_iter - the input iterator from the thread block to store to
/// \param items - array that data is stored to thread block
/// \param valid - maximum range of valid numbers to store
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

/// \brief Stores a blocked arrangement of items in memory to thread block.
///
/// Block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of 
/// \p ItemsPerThread \p items to the thread block.
///
/// The input offset (\p block_iter + offset) must be quad-item aligned.
///
/// The following conditions will prevent vectorization and switch to default
/// block_load_direct_blocked:
/// * \p ItemsPerThread is odd.
/// * The datatype \p T is not a primitive or a HC/HIP vector type (e.g. int2, 
/// int4, etc.
///
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a flat 1D thread identifier for the calling thread
/// \param block_iter - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
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

template<
    unsigned int BlockSize,
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_store_direct_striped(int flat_id, IteratorT block_iter,
                                T (&items)[ItemsPerThread]) [[hc]]
{
    IteratorT thread_iter = block_iter + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
         thread_iter[item * BlockSize] = items[item];
    }
}

template<
    unsigned int BlockSize,
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_store_direct_striped(int flat_id, IteratorT block_iter,
                                T (&items)[ItemsPerThread],
                                int valid) [[hc]]
{
    IteratorT thread_iter = block_iter + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * BlockSize;
        if (flat_id + offset < valid)
        {
             thread_iter[offset] = items[item];
        }
    }
}

/// @}
// end of group collectiveblockmodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_STORE_HPP_
