// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_
#define ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup blockmodule
/// @{

/// \brief Loads data from continuous memory into a blocked arrangement of items
/// across the thread block.
///
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread])
{
    unsigned int offset = flat_id * ItemsPerThread;
    InputIterator thread_iter = block_input + offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item];
    }
}

/// \brief Loads data from continuous memory into a blocked arrangement of items
/// across the thread block, which is guarded by range \p valid.
///
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
/// \param valid - maximum range of valid numbers to load
template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid)
{
    unsigned int offset = flat_id * ItemsPerThread;
    InputIterator thread_iter = block_input + offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        if (item + offset < valid)
        {
            items[item] = thread_iter[item];
        }
    }
}

/// \brief Loads data from continuous memory into a blocked arrangement of items
/// across the thread block, which is guarded by range with a fall-back value
/// for out-of-bound elements.
///
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
/// \tparam Default - [inferred] The data type of the default value
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
/// \param valid - maximum range of valid numbers to load
/// \param out_of_bounds - default value assigned to out-of-bound items
template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
ROCPRIM_DEVICE inline
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid,
                               Default out_of_bounds)
{
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = out_of_bounds;
    }

    block_load_direct_blocked(flat_id, block_input, items, valid);
}

/// \brief Loads data from continuous memory into a blocked arrangement of items
/// across the thread block.
///
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// The input offset (\p block_input + offset) must be quad-item aligned.
///
/// The following conditions will prevent vectorization and switch to default
/// block_load_direct_blocked:
/// * \p ItemsPerThread is odd.
/// * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
/// int4, etc.
///
/// \tparam T - [inferred] the input data type
/// \tparam U - [inferred] the output data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// The type \p T must be such that it can be implicitly converted to \p U.
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
template<
    class T,
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
typename std::enable_if<detail::is_vectorizable<T, ItemsPerThread>()>::type
block_load_direct_blocked_vectorized(unsigned int flat_id,
                                     T* block_input,
                                     U (&items)[ItemsPerThread])
{
    typedef typename detail::match_vector_type<T, ItemsPerThread>::type vector_type;
    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(vector_type);
    vector_type vector_items[vectors_per_thread];

    const vector_type* vector_ptr = reinterpret_cast<const vector_type*>(block_input) +
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
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>()>::type
block_load_direct_blocked_vectorized(unsigned int flat_id,
                                     T* block_input,
                                     U (&items)[ItemsPerThread])
{
    block_load_direct_blocked(flat_id, block_input, items);
}

/// \brief Loads data from continuous memory into a striped arrangement of items
/// across the thread block.
///
/// The striped arrangement is assumed to be (\p BlockSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam BlockSize - the number of threads in a block
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread])
{
    InputIterator thread_iter = block_input + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item * BlockSize];
    }
}

/// \brief Loads data from continuous memory into a striped arrangement of items
/// across the thread block, which is guarded by range \p valid.
///
/// The striped arrangement is assumed to be (\p BlockSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam BlockSize - the number of threads in a block
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
/// \param valid - maximum range of valid numbers to load
template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid)
{
    InputIterator thread_iter = block_input + flat_id;
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

/// \brief Loads data from continuous memory into a striped arrangement of items
/// across the thread block, which is guarded by range with a fall-back value
/// for out-of-bound elements.
///
/// The striped arrangement is assumed to be (\p BlockSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam BlockSize - the number of threads in a block
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
/// \tparam Default - [inferred] The data type of the default value
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
/// \param valid - maximum range of valid numbers to load
/// \param out_of_bounds - default value assigned to out-of-bound items
template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
ROCPRIM_DEVICE inline
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid,
                               Default out_of_bounds)
{
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = out_of_bounds;
    }

    block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block.
///
/// The warp-striped arrangement is assumed to be (\p WarpSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// * The number of threads in the block must be a multiple of \p WarpSize.
/// * The default \p WarpSize is a hardware warpsize and is an optimal value.
/// * \p WarpSize must be a power of two and equal or less than the size of
///   hardware warp.
/// * Using \p WarpSize smaller than hardware warpsize could result in lower
///   performance.
///
/// \tparam WarpSize - [optional] the number of threads in a warp
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
template<
    unsigned int WarpSize = warp_size(),
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
void block_load_direct_warp_striped(unsigned int flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread])
{
    static_assert(detail::is_power_of_two(WarpSize) && WarpSize <= warp_size(),
                 "WarpSize must be a power of two and equal or less"
                 "than the size of hardware warp.");
    unsigned int thread_id = detail::logical_lane_id<WarpSize>();
    unsigned int warp_id = flat_id / WarpSize;
    unsigned int warp_offset = warp_id * WarpSize * ItemsPerThread;

    InputIterator thread_iter = block_input + thread_id + warp_offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item * WarpSize];
    }
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block, which is guarded by range \p valid.
///
/// The warp-striped arrangement is assumed to be (\p WarpSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// * The number of threads in the block must be a multiple of \p WarpSize.
/// * The default \p WarpSize is a hardware warpsize and is an optimal value.
/// * \p WarpSize must be a power of two and equal or less than the size of
///   hardware warp.
/// * Using \p WarpSize smaller than hardware warpsize could result in lower
///   performance.
///
/// \tparam WarpSize - [optional] the number of threads in a warp
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
/// \param valid - maximum range of valid numbers to load
template<
    unsigned int WarpSize = warp_size(),
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE inline
void block_load_direct_warp_striped(unsigned int flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread],
                                    unsigned int valid)
{
    static_assert(detail::is_power_of_two(WarpSize) && WarpSize <= warp_size(),
                 "WarpSize must be a power of two and equal or less"
                 "than the size of hardware warp.");
    unsigned int thread_id = detail::logical_lane_id<WarpSize>();
    unsigned int warp_id = flat_id / WarpSize;
    unsigned int warp_offset = warp_id * WarpSize * ItemsPerThread;

    InputIterator thread_iter = block_input + thread_id + warp_offset;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * WarpSize;
        if (warp_offset + thread_id + offset < valid)
        {
            items[item] = thread_iter[offset];
        }
    }
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block, which is guarded by range with a fall-back value
/// for out-of-bound elements.
///
/// The warp-striped arrangement is assumed to be (\p WarpSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// * The number of threads in the block must be a multiple of \p WarpSize.
/// * The default \p WarpSize is a hardware warpsize and is an optimal value.
/// * \p WarpSize must be a power of two and equal or less than the size of
///   hardware warp.
/// * Using \p WarpSize smaller than hardware warpsize could result in lower
///   performance.
///
/// \tparam WarpSize - [optional] the number of threads in a warp
/// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
/// pointer
/// \tparam T - [inferred] the data type
/// \tparam ItemsPerThread - [inferred] the number of items to be processed by
/// each thread
/// \tparam Default - [inferred] The data type of the default value
///
/// \param flat_id - a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input - the input iterator from the thread block to load from
/// \param items - array that data is loaded to
/// \param valid - maximum range of valid numbers to load
/// \param out_of_bounds - default value assigned to out-of-bound items
template<
    unsigned int WarpSize = warp_size(),
    class InputIterator,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
ROCPRIM_DEVICE inline
void block_load_direct_warp_striped(unsigned int flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread],
                                    unsigned int valid,
                                    Default out_of_bounds)
{
    static_assert(detail::is_power_of_two(WarpSize) && WarpSize <= warp_size(),
                 "WarpSize must be a power of two and equal or less"
                 "than the size of hardware warp.");
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = out_of_bounds;
    }

    block_load_direct_warp_striped<WarpSize>(flat_id, block_input, items, valid);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_
