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

#ifndef ROCPRIM_BLOCK_BLOCK_LOAD_HPP_
#define ROCPRIM_BLOCK_BLOCK_LOAD_HPP_

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

/// \brief Loads a blocked arrangement of items across the thread block to memory.
///
/// Block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of 
/// \p ItemsPerThread into \p items.
///
/// \tparam IteratorT - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a flat 1D thread identifier for the calling thread
/// \param block_iter - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
template<
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_load_direct_blocked(int flat_id, IteratorT block_iter,
                               T (&items)[ItemsPerThread]) [[hc]]
{
    int offset = flat_id * ItemsPerThread;
    IteratorT thread_iter = block_iter + offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item];
    }
}

/// \brief Loads a blocked arrangement of items across the thread block to memory,
/// which is guarded by range \p valid.
///
/// Block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of 
/// \p ItemsPerThread into \p items.
///
/// \tparam IteratorT - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a flat 1D thread identifier for the calling thread
/// \param block_iter - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
/// \param valid - maximum range of valid numbers to load
template<
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_load_direct_blocked(int flat_id, IteratorT block_iter,
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
            items[item] = thread_iter[item];
        }
    }
}

/// \brief Loads a blocked arrangement of items across the thread block to memory,
/// which is guarded by range with a fall-back value for out-of-bound elements.
///
/// Block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of 
/// \p ItemsPerThread into \p items.
///
/// \tparam IteratorT - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
/// \tparam Default - [inferred] The data type of the default value
///
/// \param flat_id - a flat 1D thread identifier for the calling thread
/// \param block_iter - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
/// \param valid - maximum range of valid numbers to load
/// \param out_of_bounds - default value assigned to out-of-bound items
template<
    class IteratorT,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
void block_load_direct_blocked(int flat_id, IteratorT block_iter,
                               T (&items)[ItemsPerThread],
                               int valid, Default out_of_bounds) [[hc]]
{
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
        items[item] = out_of_bounds;

    block_load_direct_blocked(flat_id, block_iter, items, valid);
}

/// \brief Loads a blocked arrangement of items across the thread block to memory.
///
/// Block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of 
/// \p ItemsPerThread into \p items.
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
block_load_direct_blocked_vectorized(int flat_id, T* block_iter,
                                     T (&items)[ItemsPerThread]) [[hc]]
{
    typedef typename detail::match_vector_type<T, ItemsPerThread>::type Vector;
    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(Vector);
    Vector vector_items[vectors_per_thread];

    Vector* vector_ptr = reinterpret_cast<Vector*>(block_iter) +
                         (flat_id * vectors_per_thread);

    #pragma unroll
    for (unsigned int item = 0; item < vectors_per_thread; item++)
    {
        vector_items[item] = *(vector_ptr + item);
    }

    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = *(reinterpret_cast<T*>(vector_items) + item);
    }
}

template<
    class T,
    int ItemsPerThread
>
typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>()>::type
block_load_direct_blocked_vectorized(int flat_id, T* block_iter,
                                     T (&items)[ItemsPerThread]) [[hc]]
{
    block_load_direct_blocked(flat_id, block_iter, items);
}

template<
    unsigned int BlockSize,
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_load_direct_striped(int flat_id, IteratorT block_iter,
                               T (&items)[ItemsPerThread]) [[hc]]
{
    IteratorT thread_iter = block_iter + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item * BlockSize];
    }
}

template<
    unsigned int BlockSize,
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_load_direct_striped(int flat_id, IteratorT block_iter,
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
            items[item] = thread_iter[offset];
        }
    }
}

template<
    unsigned int BlockSize,
    class IteratorT,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
void block_load_direct_striped(int flat_id, IteratorT block_iter,
                               T (&items)[ItemsPerThread],
                               int valid, Default out_of_bounds) [[hc]]
{
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
        items[item] = out_of_bounds;

    block_load_direct_striped<BlockSize>(flat_id, block_iter, items, valid);
}

template<
    unsigned int WarpSize = warp_size(),
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_load_direct_warp_striped(int flat_id, IteratorT block_iter,
                                    T (&items)[ItemsPerThread]) [[hc]]
{
    unsigned int thread_id = detail::logical_lane_id<WarpSize>();
    unsigned int warp_id = flat_id / WarpSize;
    unsigned int warp_offset = warp_id * WarpSize * ItemsPerThread;
    
    IteratorT thread_iter = block_iter + thread_id + warp_offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item * WarpSize];
    }
}

template<
    unsigned int WarpSize = warp_size(),
    class IteratorT,
    class T,
    unsigned int ItemsPerThread
>
void block_load_direct_warp_striped(int flat_id, IteratorT block_iter,
                                    T (&items)[ItemsPerThread],
                                    int valid) [[hc]]
{
    unsigned int thread_id = detail::logical_lane_id<WarpSize>();
    unsigned int warp_id = flat_id / WarpSize;
    unsigned int warp_offset = warp_id * WarpSize * ItemsPerThread;
    
    IteratorT thread_iter = block_iter + thread_id + warp_offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        if (warp_offset + thread_id + (item * WarpSize) < valid)
        {
            items[item] = thread_iter[item * WarpSize];
        }
    }
}

template<
    unsigned int WarpSize = warp_size(),
    class IteratorT,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
void block_load_direct_warp_striped(int flat_id, IteratorT block_iter,
                                    T (&items)[ItemsPerThread],
                                    int valid, Default out_of_bounds) [[hc]]
{
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
        items[item] = out_of_bounds;

    block_load_direct_warp_striped<WarpSize>(flat_id, block_iter, items, valid);
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_LOAD_HPP_
