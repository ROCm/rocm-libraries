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

#ifndef ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_
#define ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../functional.hpp"
#include "../intrinsics.hpp"
#include "../types.hpp"
#include "rocprim/intrinsics/arch.hpp"

#include "../thread/thread_store.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Stores a blocked arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory.
///
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// \tparam OutputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to store to
/// \param items array that data is stored to thread block
template<
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_blocked(unsigned int flat_id,
                                OutputIterator block_output,
                                T (&items)[ItemsPerThread])
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    unsigned int offset = flat_id * ItemsPerThread;
    OutputIterator thread_iter = block_output + offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        thread_iter[item] = items[item];
    }
}

/// \brief Stores a blocked arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory, which is guarded by range \p valid.
///
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// \tparam OutputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to store to
/// \param items array that data is stored to thread block
/// \param valid maximum range of valid numbers to store
template<
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_blocked(unsigned int flat_id,
                                OutputIterator block_output,
                                T (&items)[ItemsPerThread],
                                unsigned int valid)
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    unsigned int offset = flat_id * ItemsPerThread;
    OutputIterator thread_iter = block_output + offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        if (item + offset < valid)
        {
            thread_iter[item] = items[item];
        }
    }
}

/// \brief Stores a blocked arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory.
///
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// The input offset (\p block_output + offset) must be quad-item aligned.
///
/// The following conditions will prevent vectorization and switch to default
/// block_load_direct_blocked:
/// * \p ItemsPerThread is odd.
/// * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
/// int4, etc.)
///
/// \tparam T [inferred] the output data type
/// \tparam U [inferred] the input data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// The type \p U must be such that it can be implicitly converted to \p T.
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<
    class T,
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto
block_store_direct_blocked_vectorized(unsigned int flat_id,
                                      T* block_output,
                                      U (&items)[ItemsPerThread]) -> typename std::enable_if<detail::is_vectorizable<T, ItemsPerThread>::value>::type
{
    static_assert(std::is_convertible<U, T>::value,
                  "The type U must be such that it can be implicitly converted to T.");

    using vector_type = typename detail::match_vector_type<T, ItemsPerThread>::type;
    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(vector_type);
    vector_type *vectors_ptr = reinterpret_cast<vector_type*>(const_cast<T*>(block_output));

    vector_type raw_vector_items[vectors_per_thread];
    T *raw_items = reinterpret_cast<T*>(raw_vector_items);

    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        raw_items[item] = items[item];
    }

    block_store_direct_blocked(flat_id, vectors_ptr, raw_vector_items);
}

template<
    class T,
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto
block_store_direct_blocked_vectorized(unsigned int flat_id,
                                      T* block_output,
                                      U (&items)[ItemsPerThread]) -> typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>::value>::type
{
    block_store_direct_blocked(flat_id, block_output, items);
}

/// \brief Stores a striped arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory.
///
/// The striped arrangement is assumed to be (\p BlockSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// \tparam BlockSize the number of threads in a block
/// \tparam OutputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to store to
/// \param items array that data is stored to thread block
template<
    unsigned int BlockSize,
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_striped(unsigned int flat_id,
                                OutputIterator block_output,
                                T (&items)[ItemsPerThread])
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    OutputIterator thread_iter = block_output + flat_id;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
         thread_iter[item * BlockSize] = items[item];
    }
}

/// \brief Stores a striped arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory, which is guarded by range \p valid.
///
/// The striped arrangement is assumed to be (\p BlockSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// \tparam BlockSize the number of threads in a block
/// \tparam OutputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to store to
/// \param items array that data is stored to thread block
/// \param valid maximum range of valid numbers to store
template<
    unsigned int BlockSize,
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_striped(unsigned int flat_id,
                                OutputIterator block_output,
                                T (&items)[ItemsPerThread],
                                unsigned int valid)
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    OutputIterator thread_iter = block_output + flat_id;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * BlockSize;
        if (flat_id + offset < valid)
        {
             thread_iter[offset] = items[item];
        }
    }
}

/// \brief Stores a warp-striped arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory.
///
/// \ingroup blockmodule_warp_store_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// * The number of threads in the block must be a multiple of \p VirtualWaveSize.
/// * \p VirtualWaveSize must be a power of two and equal or less than the size of
///   hardware warp.
/// * Using \p VirtualWaveSize smaller than hardware warpsize could result in lower
///   performance.
///
/// \tparam VirtualWaveSize [optional] the number of threads in a warp
/// \tparam OutputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to store to
/// \param items array that data is stored to thread block
template<unsigned int VirtualWaveSize, class OutputIterator, class T, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_warp_striped(unsigned int   flat_id,
                                     OutputIterator block_output,
                                     T (&items)[ItemsPerThread])
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    static_assert(detail::is_power_of_two(VirtualWaveSize)
                      && VirtualWaveSize <= arch::wavefront::max_size(),
                  "VirtualWaveSize must be a power of two and equal or less "
                  "than the size of hardware warp.");
    unsigned int thread_id   = detail::logical_lane_id<VirtualWaveSize>();
    unsigned int warp_id     = flat_id / VirtualWaveSize;
    unsigned int warp_offset = warp_id * VirtualWaveSize * ItemsPerThread;

    OutputIterator thread_iter = block_output + thread_id + warp_offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        thread_iter[item * VirtualWaveSize] = items[item];
    }
}

/// \brief Stores a warp-striped arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory, using the hardware warp size.
///
/// \ingroup blockmodule_warp_store_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// \tparam OutputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to store to
/// \param items array that data is stored to thread block
template<class OutputIterator, class T, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_warp_striped(unsigned int   flat_id,
                                     OutputIterator block_output,
                                     T (&items)[ItemsPerThread])
{
    if constexpr(arch::wavefront::min_size() == arch::wavefront::max_size())
    {
        block_store_direct_warp_striped<arch::wavefront::min_size()>(flat_id, block_output, items);
    }
    else
    {
        if(arch::wavefront::size() == ROCPRIM_WARP_SIZE_64)
        {
            block_store_direct_warp_striped<ROCPRIM_WARP_SIZE_64>(flat_id, block_output, items);
        }
        else
        {
            block_store_direct_warp_striped<ROCPRIM_WARP_SIZE_32>(flat_id, block_output, items);
        }
    }
}

/// \brief Stores a warp-striped arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory, which is guarded by range \p valid.
///
/// \ingroup blockmodule_warp_store_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// * The number of threads in the block must be a multiple of \p VirtualWaveSize.
/// * \p VirtualWaveSize must be a power of two and equal or less than the size of
///   hardware warp.
/// * Using \p VirtualWaveSize smaller than hardware warpsize could result in lower
///   performance.
///
/// \tparam VirtualWaveSize [optional] the number of threads in a warp
/// \tparam OutputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to store to
/// \param items array that data is stored to thread block
/// \param valid maximum range of valid numbers to store
template<unsigned int VirtualWaveSize, class OutputIterator, class T, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_warp_striped(unsigned int   flat_id,
                                     OutputIterator block_output,
                                     T (&items)[ItemsPerThread],
                                     unsigned int valid)
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    static_assert(detail::is_power_of_two(VirtualWaveSize)
                      && VirtualWaveSize <= arch::wavefront::max_size(),
                  "VirtualWaveSize must be a power of two and equal or less "
                  "than the size of hardware warp.");
    assert(VirtualWaveSize <= arch::wavefront::size());

    unsigned int thread_id   = detail::logical_lane_id<VirtualWaveSize>();
    unsigned int warp_id     = flat_id / VirtualWaveSize;
    unsigned int warp_offset = warp_id * VirtualWaveSize * ItemsPerThread;

    OutputIterator thread_iter = block_output + thread_id + warp_offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * VirtualWaveSize;
        if (warp_offset + thread_id + offset < valid)
        {
            thread_iter[offset] = items[item];
        }
    }
}

/// \brief Stores a warp-striped arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory, which is guarded by range \p valid,
/// using the hardware warp size.
///
/// \ingroup blockmodule_warp_store_functions
/// The warp-striped arrangement is assumed to be (\p VirtualWaveSize * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// \tparam OutputIterator [inferred] an iterator type for input (can be a simple
/// pointer)
/// \tparam T [inferred] the data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to store to
/// \param items array that data is stored to thread block
/// \param valid maximum range of valid numbers to store
template<class OutputIterator, class T, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_warp_striped(unsigned int   flat_id,
                                     OutputIterator block_output,
                                     T (&items)[ItemsPerThread],
                                     unsigned int valid)
{
    if constexpr(arch::wavefront::min_size() == arch::wavefront::max_size())
    {
        block_store_direct_warp_striped<arch::wavefront::min_size()>(flat_id,
                                                                     block_output,
                                                                     items,
                                                                     valid);
    }
    else
    {
        if(arch::wavefront::size() == ROCPRIM_WARP_SIZE_64)
        {
            block_store_direct_warp_striped<ROCPRIM_WARP_SIZE_64>(flat_id,
                                                                  block_output,
                                                                  items,
                                                                  valid);
        }
        else
        {
            block_store_direct_warp_striped<ROCPRIM_WARP_SIZE_32>(flat_id,
                                                                  block_output,
                                                                  items,
                                                                  valid);
        }
    }
}

/// \brief Stores a blocked arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory.
///
/// \ingroup blockmodule_cast_store_functions
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// The following conditions will prevent casting and switch to default
/// block_store_direct_blocked:
/// * \p ItemsPerThread * sizeof(T) should be a multiple of sizeof(V)
/// * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
/// int4, etc.)
///
/// \tparam V [optional] the type it will be casted to
/// \tparam VirtualWaveSize [optional] the number of threads in a warp
/// \tparam T [inferred] the output data type
/// \tparam U [inferred] the input data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<class V = rocprim::uint128_t,
         unsigned int VirtualWaveSize,
         class T,
         class U,
         unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto block_store_direct_blocked_cast(unsigned int flat_id,
                                     T*           block_output,
                                     U (&items)[ItemsPerThread])
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    -> typename std::enable_if<detail::is_vectorizable<T, ItemsPerThread>::value
                               && (ItemsPerThread * sizeof(T)) % sizeof(V) == 0>::type
#endif // DOXYGEN_SHOULD_SKIP_THIS
{
    static_assert(std::is_convertible<U, T>::value,
                  "The type U must be such that it can be implicitly converted to T.");

    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(V);

    V* vector_ptr = ::rocprim::detail::bit_cast<V*>(block_output) + flat_id * vectors_per_thread;

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < vectors_per_thread; item++)
    {
        vector_ptr[item] = *(reinterpret_cast<const V*>(items) + item);
    }
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class V = rocprim::uint128_t,
         unsigned int VirtualWaveSize,
         class T,
         class U,
         unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto block_store_direct_blocked_cast(unsigned int flat_id,
                                     T*           block_output,
                                     U (&items)[ItemsPerThread]) ->
    typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>::value
                            || (ItemsPerThread * sizeof(T)) % sizeof(V) != 0>::type
{
    block_store_direct_blocked(flat_id, block_output, items);
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Stores a blocked arrangement of items from across the thread block
/// into a blocked arrangement on continuous memory, using the hardware warp size.
///
/// \ingroup blockmodule_cast_store_functions
/// The block arrangement is assumed to be (block-threads * \p ItemsPerThread) items
/// across a thread block. Each thread uses a \p flat_id to store a range of
/// \p ItemsPerThread \p items to the thread block.
///
/// The following conditions will prevent casting and switch to default
/// block_load_direct_blocked:
/// * \p ItemsPerThread * sizeof(T) should be a multiple of sizeof(V)
/// * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
/// int4, etc.)
///
/// \tparam V [optional] the type it will be casted to
/// \tparam T [inferred] the output data type
/// \tparam U [inferred] the input data type
/// \tparam ItemsPerThread [inferred] the number of items to be processed by
/// each thread
///
/// \param flat_id a local flat 1D thread id in a block (tile) for the calling thread
/// \param block_output the input iterator from the thread block to load from
/// \param items array that data is loaded to
template<class V = rocprim::uint128_t, class T, class U, unsigned int ItemsPerThread>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto block_store_direct_blocked_cast(unsigned int flat_id,
                                     T*           block_output,
                                     U (&items)[ItemsPerThread])
{

    if constexpr(arch::wavefront::min_size() == arch::wavefront::max_size())
    {
        block_store_direct_blocked_cast<V, arch::wavefront::min_size()>(flat_id,
                                                                        block_output,
                                                                        items);
    }
    else
    {
        if(arch::wavefront::size() == ROCPRIM_WARP_SIZE_64)
        {
            block_store_direct_blocked_cast<V, ROCPRIM_WARP_SIZE_64>(flat_id, block_output, items);
        }
        else
        {
            block_store_direct_blocked_cast<V, ROCPRIM_WARP_SIZE_32>(flat_id, block_output, items);
        }
    }
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_
