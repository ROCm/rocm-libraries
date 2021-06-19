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
/// \tparam BlockSizeX - the number of threads in a block's x dimension, it has no defaults value.
/// \tparam BlockSizeY - the number of threads in a block's y dimension, defaults to 1.
/// \tparam BlockSizeZ - the number of threads in a block's z dimension, defaults to 1.
///
/// \par Overview
/// It is commonplace for blocks of threads to rearrange data items between
/// threads.  The BlockShuffle abstraction allows threads to efficiently shift items
/// either (a) up to their successor or (b) down to their predecessor.
/// * Computation can more efficient when:
///   * \p ItemsPerThread is greater than one,
///   * \p T is an arithmetic type,
///   * the number of threads in the block is a multiple of the hardware warp size (see rocprim::warp_size()).
///
/// \par Examples
/// \parblock
/// In the examples shuffle operation is performed on block of 192 threads, each provides
/// one \p int value, result is returned using the same variable as for input.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize block__shuffle_int for int and logical warp of 192 threads
///     using block__shuffle_int = rocprim::block_shuffle<int, 192>;
///     // allocate storage in shared memory
///     __shared__ block_shuffle::storage_type storage;
///
///     int value = ...;
///     // execute block shuffle
///     block__shuffle_int().inclusive_up(
///         value, // input
///         value, // output
///         storage
///     );
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1>
class block_shuffle
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    struct storage_type_
    {
        T prev[BlockSize];
        T next[BlockSize];
    };

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

    /// \brief Shuffles data across threads in a block, offseted by the distance value.
    ///
    /// \par A thread with  threadId i receives data from a thread with threadIdx (i-distance), whre distance may be a negative value.
    /// allocated by the method itself.
    /// \par Any shuffle operation with invalid input or output threadIds are not carried out, i.e. threadId < 0 || threadId >= BlockSize.
    ///
    /// \param [in] input - input data to be shuffled to another thread.
    /// \param [out] output - reference to a output value, that receives data from another thread
    /// \param [in] distance - The input threadId + distance = output threadId.
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block__shuffle_int for int and logical warp of 192 threads
    ///     using block__shuffle_int = rocprim::block_shuffle<int, 192>;
    ///
    ///     int value = ...;
    ///     // execute block shuffle
    ///     block__shuffle_int().offset(
    ///         value, // input
    ///         value  // output
    ///     );
    ///     ...
    /// }
    /// \endcode
    ROCPRIM_DEVICE inline
    void offset(T input,
                T& output,
                int distance = 1)
    {
        offset(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, distance
        );
    }

    ROCPRIM_DEVICE inline
    void offset(const size_t& flat_id,
                T input,
                T& output,
                int distance)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        offset(flat_id, input, output, distance, storage);
    }

    ROCPRIM_DEVICE inline
    void offset(const size_t& flat_id,
                T input,
                T& output,
                int distance,
                storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        storage_.prev[flat_id] = input;

        ::rocprim::syncthreads();

        const int offset_tid = static_cast<int>(flat_id) + distance;
        if ((offset_tid >= 0) && (offset_tid < (int)BlockSize))
        {
            output = storage_.prev[static_cast<size_t>(offset_tid)];
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
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block__shuffle_int for int and logical warp of 192 threads
    ///     using block__shuffle_int = rocprim::block_shuffle<int, 192>;
    ///
    ///     int value = ...;
    ///     // execute block shuffle
    ///     block__shuffle_int().rotate(
    ///         value, // input
    ///         value  // output
    ///     );
    ///     ...
    /// }
    /// \endcode
    ROCPRIM_DEVICE inline
    void rotate(T input,
                T& output,
                unsigned int distance = 1)
    {
        rotate(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, distance
        );
    }

    ROCPRIM_DEVICE inline
    void rotate(const size_t& flat_id,
                T input,
                T& output,
                unsigned int distance)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        rotate(flat_id, input, output, distance, storage);
    }

    ROCPRIM_DEVICE inline
    void rotate(const size_t& flat_id,
                T input,
                T& output,
                unsigned int distance,
                storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        storage_.prev[flat_id] = input;

        ::rocprim::syncthreads();

        unsigned int offset = threadIdx.x + distance;
        if (offset >= BlockSize)
            offset -= BlockSize;

        output = storage_.prev[offset];
    }


    /// \brief The thread block rotates a blocked arrange of \input items,
    /// shifting it up by one item
    ///
    /// \param [in]  input -  The calling thread's input items
    /// \param [out] prev  -  The corresponding predecessor items (may be aliased to \p input).
    /// The item \p prev[0] is not updated for <em>thread</em><sub>0</sub>.
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block__shuffle_int for int and logical warp of 192 threads
    ///     using block__shuffle_int = rocprim::block_shuffle<int, 192>;
    ///
    ///     int value = ...;
    ///     // execute block shuffle
    ///     block__shuffle_int().up(
    ///         value, // input
    ///         value  // output
    ///     );
    ///     ...
    /// }
    /// \endcode
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void up(T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread])
    {
        this->up(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, prev
        );
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void up(const size_t& flat_id,
            T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->up(flat_id, input, prev, storage);
    }


    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void up(const size_t& flat_id,
            T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread],
            storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        storage_.prev[flat_id] = input[ItemsPerThread -1];

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for (unsigned int i = ItemsPerThread - 1; i > 0; --i)
        {
            prev[i] = input[i - 1];
        }

        if (flat_id > 0)
        {
            prev[0] = storage_.prev[flat_id - 1];
        }
    }



    /// \brief The thread block rotates a blocked arrange of \input items,
    /// shifting it up by one item
    ///
    /// \param [in]  input - The calling thread's input items
    /// \param [out] prev  - The corresponding predecessor items (may be aliased to \p input).
    /// The item \p prev[0] is not updated for <em>thread</em><sub>0</sub>.
    /// \param [out] block_suffix - The item \p input[ItemsPerThread-1] from
    /// <em>thread</em><sub><tt>BlockSize-1</tt></sub>, provided to all threads
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void up(T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread],
            T &block_suffix)
    {
        this->up(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, prev, block_suffix
        );
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void up(const size_t& flat_id,
            T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread],
            T &block_suffix)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->up(flat_id, input, prev, block_suffix, storage);
    }

    template <int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void up(const size_t& flat_id,
            T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread],
            T &block_suffix,
            storage_type& storage)
    {
        up(flat_id, input, prev, storage);

        // Update block prefix
        block_suffix = storage->prev[BlockSize - 1];
    }

    /// \brief The thread block rotates a blocked arrange of \input items,
    /// shifting it down by one item
    ///
    /// \param [in]  input -  The calling thread's input items
    /// \param [out] next  -  The corresponding successor items (may be aliased to \p input).
    /// The item \p prev[0] is not updated for <em>thread</em><sub>BlockSize - 1</sub>.
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block__shuffle_int for int and logical warp of 192 threads
    ///     using block__shuffle_int = rocprim::block_shuffle<int, 192>;
    ///
    ///     int value = ...;
    ///     // execute block shuffle
    ///     block__shuffle_int().down(
    ///         value, // input
    ///         value  // output
    ///     );
    ///     ...
    /// }
    /// \endcode
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void down(T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread])
    {
        this->down(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, next
        );
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void down(const size_t& flat_id,
              T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->down(flat_id, input, next, storage);
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void down(const size_t& flat_id,
              T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread],
              storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        storage_.next[flat_id] = input[0];

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for (unsigned int i = 0; i < (ItemsPerThread - 1); ++i)
        {
          next[i] = input[i + 1];
        }

        if (flat_id <(BlockSize -1))
        {
          next[ItemsPerThread -1] = storage_.next[flat_id + 1];
        }
    }

    /// \brief The thread block rotates a blocked arrange of \input items,
    /// shifting it down by one item
    ///
    /// \param [in]  input -  The calling thread's input items
    /// \param [out] next  -  The corresponding successor items (may be aliased to \p input).
    /// The item \p prev[0] is not updated for <em>thread</em><sub>BlockSize - 1</sub>.
    /// \param [out] block_prefix -  The item \p input[0] from <em>thread</em><sub><tt>0</tt></sub>, provided to all threads
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void down(T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread],
              T &block_prefix)
    {
        this->down(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, next, block_prefix
        );
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void down(const size_t& flat_id,
              T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread],
              T &block_prefix)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->down(flat_id, input, next, block_prefix, storage);
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE inline
    void down(const size_t& flat_id,
              T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread],
              T &block_prefix,
              storage_type& storage)
    {
        this->down(flat_id, input, next, storage);

        // Update block prefixstorage_->
        block_prefix = storage->next[0];
    }
};


END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
