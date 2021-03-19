/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
#define ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_reduce_warp_reduce.hpp"
#include "detail/block_reduce_raking_reduce.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE


/// \brief The block_shuffle class is a block level parallel primitive which provides methods
/// for shuffling data partitioned across a block
///
/// \tparam T - the input/output type.
/// \tparam Algorithm - selected scan algorithm, block_scan_algorithm::default_algorithm by default.
/// \tparam BlockSizeY - the number of threads in a block's y dimension, defaults to 1.
/// \tparam BlockSizeZ - the number of threads in a block's z dimension, defaults to 1.
///
/// \par Overview
/// It is commonplace for blocks of threads to rearrange data items between
//  threads.  The BlockShuffle abstraction allows threads to efficiently shift items
//  either (a) up to their successor or (b) down to their predecessor.
/// \endparblock
template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1>
class block_shuffle
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    struct storage_type_
    {
        T prev[BlockSize];
        T next[BlockSize];
    };

    storage_type_* storage;

public:

    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
        using storage_type = detail::raw_storage<storage_type_>;
    #else
        using storage_type = storage_type_; // only for Doxygen
    #endif

    ROCPRIM_DEVICE inline
    block_shuffle()
    {
        ROCPRIM_SHARED_MEMORY storage_type_ shared_storage;
        storage = &shared_storage;
    }

    /// \brief Shuffles data across threads in a block, offseted by the distance value.
    ///
    /// \par A thread with  threadId i receives data from a thread with threadIdx (i-distance), whre distance may be a negative value.
    /// allocated by the method itself.
    /// \par Any shuffle operation with invalid input or output threadIds are not carried out, i.e. threadId < 0 || threadId >= BlockSize.
    ///
    /// \param [in] input - input data to be shuffled to another thread.
    /// \param [out] output - reference to a output value, that receives data from another thread
    /// \param [in] distance - The input threadId + distance = output threadId.
    ROCPRIM_DEVICE inline
    void offset(
      T input,
      T& output,
      int distance =1)
    {
        const size_t flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage->prev[flat_id] = input;
        ::rocprim::syncthreads();

        const int offset_tid = static_cast<int>(flat_id) + distance;
        if ((offset_tid >= 0) && (offset_tid < (int)BlockSize))
        {
            output = storage->prev[static_cast<size_t>(offset_tid)];
        }
    }

    /// \brief Shuffles data across threads in a block, offseted by the distance value.
    ///
    /// \par A thread with  threadId i receives data from a thread with threadIdx (i-distance)%BlockSize, whre distance may be a negative value.
    /// allocated by the method itself.
    /// \par Data is rotated around the block, using (input_threadId + distance) modulous BlockSize to ensure valid threadIds.
    ///
    /// \param [in] input - input data to be shuffled to another thread.
    /// \param [out] output - reference to a output value, that receives data from another thread
    /// \param [in] distance - The input threadId + distance = output threadId.
    ROCPRIM_DEVICE inline
    void rotate(
        T   input,                  ///< [in] The calling thread's input item
        T&  output,                 ///< [out] The \p input item from thread <em>thread</em><sub>(<em>i</em>+<tt>distance></tt>)%<tt><BlockSize></tt></sub> (may be aliased to \p input).  This value is not updated for <em>thread</em><sub>BlockSize-1</sub>
        unsigned int distance = 1)  ///< [in] Offset distance (0 < \p distance < <tt>BlockSize</tt>)
    {
        const size_t flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage->prev[flat_id] = input;
        ::rocprim::syncthreads();

        unsigned int offset = threadIdx.x + distance;
        if (offset >= BlockSize)
            offset -= BlockSize;

        output = storage->prev[offset];
    }


    /// \brief The thread block rotates a blocked arrange of \input items, shifting it up by one item
    ///
    /// \param [in]  input -  The calling thread's input items
    /// \param [out] prev  -  The corresponding predecessor items (may be aliased to \p input).  The item \p prev[0] is not updated for <em>thread</em><sub>0</sub>.
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void up(
        T (&input)[ItemsPerThread],
        T (&prev)[ItemsPerThread])
    {
        const size_t flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage->prev[flat_id] = input[ItemsPerThread -1];
        ::rocprim::syncthreads();
        #pragma unroll
        for (unsigned int i = ItemsPerThread - 1; i > 0; --i)
        {
            prev[i] = input[i - 1];
        }
        if (flat_id > 0)
        {
            prev[0] = storage->prev[flat_id - 1];
        }
    }



    /// \brief The thread block rotates a blocked arrange of \input items, shifting it up by one item
    ///
    /// \param [in]  input - The calling thread's input items
    /// \param [out] prev  - The corresponding predecessor items (may be aliased to \p input).  The item \p prev[0] is not updated for <em>thread</em><sub>0</sub>.
    /// \param [out] block_suffix - The item \p input[ItemsPerThread-1] from <em>thread</em><sub><tt>BlockSize-1</tt></sub>, provided to all threads
    template <int ItemsPerThread>
    ROCPRIM_DEVICE inline void up(
        T (&input)[ItemsPerThread],
        T (&prev)[ItemsPerThread],
        T &block_suffix)
    {
        up(input, prev);
        block_suffix = storage->prev[BlockSize - 1];
    }

    /// \brief The thread block rotates a blocked arrange of \input items, shifting it down by one item
    ///
    /// \param [in]  input -  The calling thread's input items
    /// \param [out] next  -  The corresponding successor items (may be aliased to \p input).  The item \p prev[0] is not updated for <em>thread</em><sub>BlockSize - 1</sub>.
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline void down(
        T (&input)[ItemsPerThread],
        T (&next)[ItemsPerThread])
    {
        const size_t flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage->next[flat_id] = input[0];
        ::rocprim::syncthreads();

        #pragma unroll
        for (unsigned int i = 0; i < (ItemsPerThread - 1); ++i)
        {
          next[i] = input[i + 1];
        }

        if (flat_id <(BlockSize -1))
        {
          next[ItemsPerThread -1] = storage->next[flat_id + 1];
        }
    }


    /// \brief The thread block rotates a blocked arrange of \input items, shifting it down by one item
    ///
    /// \param [in]  input -  The calling thread's input items
    /// \param [out] next  -  The corresponding successor items (may be aliased to \p input).  The item \p prev[0] is not updated for <em>thread</em><sub>BlockSize - 1</sub>.
    /// \param [out] block_prefix -  The item \p input[0] from <em>thread</em><sub><tt>0</tt></sub>, provided to all threads
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline void Down(
        T (&input)[ItemsPerThread],
        T (&next)[ItemsPerThread],
        T &block_prefix)
    {
        Down(input, next);
        block_prefix = storage->next[0];
    }
};


END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
