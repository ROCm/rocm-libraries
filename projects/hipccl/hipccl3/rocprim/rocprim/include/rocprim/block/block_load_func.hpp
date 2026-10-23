// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../functional.hpp"
#include "../intrinsics.hpp"
#include "../types.hpp"
#include "rocprim/intrinsics/arch.hpp"

#include "../thread/thread_load.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Loads data from continuous memory into a blocked arrangement of items
/// across the thread block.
///
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread])
{
    unsigned int offset = flat_id * ItemsPerThread;
    InputIterator thread_iter = block_input + offset;
    ROCPRIM_UNROLL
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
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
/// \param valid maximum range of valid numbers to load
template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid)
{
    unsigned int offset = flat_id * ItemsPerThread;
    InputIterator thread_iter = block_input + offset;
    ROCPRIM_UNROLL
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
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
/// \tparam Default [inferred] The data type of the default value
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
/// \param valid maximum range of valid numbers to load
/// \param out_of_bounds default value assigned to out-of-bound items
template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid,
                               Default out_of_bounds)
{
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = static_cast<T>(out_of_bounds);
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
/// int4, etc.)
///
/// \tparam T [inferred] the input data type
/// \tparam U [inferred] the output data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// The type \p T must be such that it can be implicitly converted to \p U.
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<
    class T,
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto
block_load_direct_blocked_vectorized(unsigned int flat_id,
                                     T* block_input,
                                     U (&items)[ItemsPerThread]) -> typename std::enable_if<detail::is_vectorizable<T, ItemsPerThread>::value>::type
{
    using vector_type = typename detail::match_vector_type<T, ItemsPerThread>::type;
    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(vector_type);
    vector_type vector_items[vectors_per_thread];

    const vector_type* vector_ptr = reinterpret_cast<const vector_type*>(block_input) +
        (flat_id * vectors_per_thread);

    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < vectors_per_thread; item++)
    {
        vector_items[item] = *(vector_ptr + item);
    }

    ROCPRIM_UNROLL
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
ROCPRIM_DEVICE ROCPRIM_INLINE
auto
block_load_direct_blocked_vectorized(unsigned int flat_id,
                                     T* block_input,
                                     U (&items)[ItemsPerThread]) -> typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>::value>::type
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
/// \tparam BlockSize the number of threads in a block
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread])
{
    InputIterator thread_iter = block_input + flat_id;
    ROCPRIM_UNROLL
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
/// \tparam BlockSize the number of threads in a block
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
/// \param valid maximum range of valid numbers to load
template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid)
{
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * BlockSize;
        if (flat_id + offset < valid)
        {
            // Note: Loading data using thread_iter like the other overloads do (thread_iter[offset])
            // doesn't work here for gfx11xx on Windows due to a compiler bug.
            // Temporarily load using the approach below until we have a fix.
            items[item] = block_input[flat_id + offset];
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
/// \tparam BlockSize the number of threads in a block
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
/// \tparam Default [inferred] The data type of the default value
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
/// \param valid maximum range of valid numbers to load
/// \param out_of_bounds default value assigned to out-of-bound items
template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid,
                               Default out_of_bounds)
{
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = out_of_bounds;
    }

    block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block.
///
/// \ingroup blockmodule_warp_load_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// * The number of threads in the block must be a multiple of \p VirtualWaveSize.
/// * \p VirtualWaveSize must be a power of two and equal or less than the size of
///   hardware warp.
/// * Using \p VirtualWaveSize smaller than hardware warpsize could result in lower
///   performance.
///
/// \tparam VirtualWaveSize [optional] the number of threads in a warp
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<unsigned int VirtualWaveSize, class InputIterator, class T, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int  flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread])
{
    static_assert(detail::is_power_of_two(VirtualWaveSize)
                      && VirtualWaveSize <= arch::wavefront::max_size(),
                  "VirtualWaveSize must be a power of two and equal or less "
                  "than the size of hardware warp.");
    assert(VirtualWaveSize <= arch::wavefront::size());

    unsigned int thread_id   = detail::logical_lane_id<VirtualWaveSize>();
    unsigned int warp_id     = flat_id / VirtualWaveSize;
    unsigned int warp_offset = warp_id * VirtualWaveSize * ItemsPerThread;

    InputIterator thread_iter = block_input + thread_id + warp_offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item * VirtualWaveSize];
    }
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block, using the hardware warp size.
///
/// \ingroup blockmodule_warp_load_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<class InputIterator, class T, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int  flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread])
{
    if constexpr(arch::wavefront::min_size() == arch::wavefront::max_size())
    {
        block_load_direct_warp_striped<arch::wavefront::min_size()>(flat_id, block_input, items);
    }
    else
    {
        if(arch::wavefront::size() == ROCPRIM_WARP_SIZE_64)
        {
            block_load_direct_warp_striped<ROCPRIM_WARP_SIZE_64>(flat_id, block_input, items);
        }
        else
        {
            block_load_direct_warp_striped<ROCPRIM_WARP_SIZE_32>(flat_id, block_input, items);
        }
    }
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block, which is guarded by range \p valid.
///
/// \ingroup blockmodule_warp_load_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// * The number of threads in the block must be a multiple of \p VirtualWaveSize.
/// * \p VirtualWaveSize must be a power of two and equal or less than the size of
///   hardware warp.
/// * Using \p VirtualWaveSize smaller than hardware warpsize could result in lower
///   performance.
///
/// \tparam VirtualWaveSize [optional] the number of threads in a warp
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
/// \param valid maximum range of valid numbers to load
template<unsigned int VirtualWaveSize, class InputIterator, class T, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int  flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread],
                                    unsigned int valid)
{
    static_assert(detail::is_power_of_two(VirtualWaveSize)
                      && VirtualWaveSize <= arch::wavefront::max_size(),
                  "VirtualWaveSize must be a power of two and equal or less "
                  "than the size of hardware warp.");
    assert(VirtualWaveSize <= arch::wavefront::size());

    unsigned int thread_id   = detail::logical_lane_id<VirtualWaveSize>();
    unsigned int warp_id     = flat_id / VirtualWaveSize;
    unsigned int warp_offset = warp_id * VirtualWaveSize * ItemsPerThread;

    InputIterator thread_iter = block_input + thread_id + warp_offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * VirtualWaveSize;
        if (warp_offset + thread_id + offset < valid)
        {
            items[item] = thread_iter[offset];
        }
    }
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block, which is guarded by range \p valid, using the hardware warp size.
///
/// \ingroup blockmodule_warp_load_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
/// \param valid maximum range of valid numbers to load
template<class InputIterator, class T, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int  flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread],
                                    unsigned int valid)
{
    if constexpr(arch::wavefront::min_size() == arch::wavefront::max_size())
    {
        block_load_direct_warp_striped<arch::wavefront::min_size()>(flat_id,
                                                                    block_input,
                                                                    items,
                                                                    valid);
    }
    else
    {
        if(arch::wavefront::size() == ROCPRIM_WARP_SIZE_64)
        {
            block_load_direct_warp_striped<ROCPRIM_WARP_SIZE_64>(flat_id,
                                                                 block_input,
                                                                 items,
                                                                 valid);
        }
        else
        {
            block_load_direct_warp_striped<ROCPRIM_WARP_SIZE_32>(flat_id,
                                                                 block_input,
                                                                 items,
                                                                 valid);
        }
    }
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block, which is guarded by range with a fall-back value
/// for out-of-bound elements.
///
/// \ingroup blockmodule_warp_load_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// * The number of threads in the block must be a multiple of \p VirtualWaveSize.
/// * \p VirtualWaveSize must be a power of two and equal or less than the size of
///   hardware warp.
/// * Using \p VirtualWaveSize smaller than hardware warpsize could result in lower
///   performance.
///
/// \tparam VirtualWaveSize [optional] the number of threads in a warp
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
/// \tparam Default [inferred] The data type of the default value
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
/// \param valid maximum range of valid numbers to load
/// \param out_of_bounds default value assigned to out-of-bound items
template<unsigned int VirtualWaveSize,
         class InputIterator,
         class T,
         unsigned int ItemsPerThread,
         class Default>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int  flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread],
                                    unsigned int valid,
                                    Default      out_of_bounds)
{
    static_assert(detail::is_power_of_two(VirtualWaveSize)
                      && VirtualWaveSize <= arch::wavefront::max_size(),
                  "VirtualWaveSize must be a power of two and equal or less "
                  "than the size of hardware warp.");
    assert(VirtualWaveSize <= arch::wavefront::size());

    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = out_of_bounds;
    }

    block_load_direct_warp_striped<VirtualWaveSize>(flat_id, block_input, items, valid);
}

/// \brief Loads data from continuous memory into a warp-striped arrangement of items
/// across the thread block, which is guarded by range with a fall-back value
/// for out-of-bound elements, using the hardware warp size.
///
/// \ingroup blockmodule_warp_load_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// \tparam InputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
/// \tparam Default [inferred] The data type of the default value
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
/// \param valid maximum range of valid numbers to load
/// \param out_of_bounds default value assigned to out-of-bound items
template<class InputIterator, class T, unsigned int ItemsPerThread, class Default>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int  flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread],
                                    unsigned int valid,
                                    Default      out_of_bounds)
{
    if constexpr(arch::wavefront::min_size() == arch::wavefront::max_size())
    {
        block_load_direct_warp_striped<arch::wavefront::min_size()>(flat_id,
                                                                    block_input,
                                                                    items,
                                                                    valid,
                                                                    out_of_bounds);
    }
    else
    {
        if(arch::wavefront::size() == ROCPRIM_WARP_SIZE_64)
        {
            block_load_direct_warp_striped<ROCPRIM_WARP_SIZE_64>(flat_id,
                                                                 block_input,
                                                                 items,
                                                                 valid,
                                                                 out_of_bounds);
        }
        else
        {
            block_load_direct_warp_striped<ROCPRIM_WARP_SIZE_32>(flat_id,
                                                                 block_input,
                                                                 items,
                                                                 valid,
                                                                 out_of_bounds);
        }
    }
}

/// \brief Loads data from continuous memory into a blocked arrangement of items
/// across the thread block.
///
/// \ingroup blockmodule_cast_load_functions
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// The following conditions will prevent casting and switch to default
/// block_store_direct_blocked:
/// * \p ItemsPerThread * sizeof(T) should be a multiple of sizeof(V)
/// * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
/// int4, etc.)
///
/// \tparam V [optional] the type it will be casted to
/// \tparam cache_load_modifier [optional] the type thread load used
/// \tparam VirtualWaveSize [optional] the number of threads in a warp
/// \tparam T [inferred] the output data type
/// \tparam U [inferred] the input data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<class V                      = rocprim::uint128_t,
         cache_load_modifier LoadType = load_default,
         unsigned int        VirtualWaveSize,
         class T,
         class U,
         unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto block_load_direct_blocked_cast(unsigned int flat_id,
                                    T*           block_input,
                                    U (&items)[ItemsPerThread])
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    -> typename std::enable_if<detail::is_vectorizable<T, ItemsPerThread>::value
                               && (ItemsPerThread * sizeof(T)) % sizeof(V) == 0>::type
#endif // DOXYGEN_SHOULD_SKIP_THIS
{
    static_assert(detail::is_power_of_two(VirtualWaveSize)
                      && VirtualWaveSize <= arch::wavefront::max_size(),
                  "VirtualWaveSize must be a power of two and equal or less "
                  "than the size of hardware warp.");
    assert(VirtualWaveSize <= arch::wavefront::size());

    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(V);

    const V* vector_ptr
        = ::rocprim::detail::bit_cast<const V*>(block_input) + flat_id * vectors_per_thread;

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < vectors_per_thread; item++)
    {
        reinterpret_cast<V*>(items)[item] = thread_load<LoadType>(vector_ptr + item);
    }
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class V                      = rocprim::uint128_t,
         cache_load_modifier LoadType = load_default,
         unsigned int        VirtualWaveSize,
         class T,
         class U,
         unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto block_load_direct_blocked_cast(unsigned int flat_id,
                                    T*           block_input,
                                    U (&items)[ItemsPerThread]) ->
    typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>::value
                            || (ItemsPerThread * sizeof(T)) % sizeof(V) != 0>::type
{
    block_load_direct_blocked(flat_id, block_input, items);
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Loads data from continuous memory into a blocked arrangement of items
/// across the thread block, using hardware warp size.
///
/// \ingroup blockmodule_cast_load_functions
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to load a range of
/// \p ItemsPerThread into \p items.
///
/// The following conditions will prevent casting and switch to default
/// block_store_direct_blocked:
/// * \p ItemsPerThread * sizeof(T) should be a multiple of sizeof(V)
/// * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
/// int4, etc.)
///
/// \tparam V [optional] the type it will be casted to
/// \tparam cache_load_modifier [optional] the type thread load used
/// \tparam T [inferred] the output data type
/// \tparam U [inferred] the input data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_input the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<class V                      = rocprim::uint128_t,
         cache_load_modifier LoadType = load_default,
         class T,
         class U,
         unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto block_load_direct_blocked_cast(unsigned int flat_id,
                                    T*           block_input,
                                    U (&items)[ItemsPerThread])
{
    if constexpr(arch::wavefront::min_size() == arch::wavefront::max_size())
    {
        block_load_direct_blocked_cast<V, LoadType, arch::wavefront::min_size()>(flat_id,
                                                                                 block_input,
                                                                                 items);
    }
    else
    {
        if(arch::wavefront::size() == ROCPRIM_WARP_SIZE_64)
        {
            block_load_direct_blocked_cast<V, LoadType, ROCPRIM_WARP_SIZE_64>(flat_id,
                                                                              block_input,
                                                                              items);
        }
        else
        {
            block_load_direct_blocked_cast<V, LoadType, ROCPRIM_WARP_SIZE_32>(flat_id,
                                                                              block_input,
                                                                              items);
        }
    }
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_
