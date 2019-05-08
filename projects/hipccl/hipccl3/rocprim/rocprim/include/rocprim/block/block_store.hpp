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

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "block_store_func.hpp"
#include "block_exchange.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief \p block_store_method enumerates the methods available to store a striped arrangement
/// of items into a blocked/striped arrangement on continuous memory
enum class block_store_method
{
    /// A blocked arrangement of items is stored into a blocked arrangement on continuous
    /// memory.
    /// \par Performance Notes:
    /// * Performance decreases with increasing number of items per thread (stride
    /// between reads), because of reduced memory coalescing.
    block_store_direct,

    /// A blocked arrangement of items is stored into a blocked arrangement on continuous
    /// memory using vectorization as an optimization.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, provided that
    /// vectorization requirements are fulfilled. Otherwise, performance will default
    /// to \p block_store_direct.
    /// \par Requirements:
    /// * The output offset (\p block_output) must be quad-item aligned.
    /// * The following conditions will prevent vectorization and switch to default
    /// \p block_store_direct:
    ///   * \p ItemsPerThread is odd.
    ///   * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
    /// int4, etc.
    block_store_vectorize,

    /// A blocked arrangement of items is locally transposed and stored as a striped
    /// arrangement of data on continuous memory.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, regardless of the
    /// number of items per thread.
    /// * Performance may be better compared to \p block_store_direct and
    /// \p block_store_vectorize due to reordering on local memory.
    block_store_transpose,

    /// A blocked arrangement of items is locally transposed and stored as a warp-striped
    /// arrangement of data on continuous memory.
    /// \par Requirements:
    /// * The number of threads in the block must be a multiple of the size of hardware warp.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, regardless of the
    /// number of items per thread.
    /// * Performance may be better compared to \p block_store_direct and
    /// \p block_store_vectorize due to reordering on local memory.
    block_store_warp_transpose,

    /// Defaults to \p block_store_direct
    default_method = block_store_direct
};

/// \brief The \p block_store class is a block level parallel primitive which provides methods
/// for storing an arrangement of items into a blocked/striped arrangement on continous memory.
///
/// \tparam T - the output/output type.
/// \tparam BlockSize - the number of threads in a block.
/// \tparam ItemsPerThread - the number of items to be processed by
/// each thread.
/// \tparam Method - the method to store data.
///
/// \par Overview
/// * The \p block_store class has a number of different methods to store data:
///   * [block_store_direct](\ref ::block_store_method::block_store_direct)
///   * [block_store_vectorize](\ref ::block_store_method::block_store_vectorize)
///   * [block_store_transpose](\ref ::block_store_method::block_store_transpose)
///   * [block_store_warp_transpose](\ref ::block_store_method::block_store_warp_transpose)
///
/// \par Example:
/// \parblock
/// In the examples store operation is performed on block of 128 threads, using type
/// \p int and 8 items per thread.
///
/// \code{.cpp}
/// __global__ void kernel(int * output)
/// {
///     const int offset = hipBlockIdx_x * 128 * 8;
///     int items[8];
///     rocprim::block_store<int, 128, 8, store_method> blockstore;
///     blockstore.store(output + offset, items);
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    block_store_method Method = block_store_method::block_store_direct
>
class block_store
{
private:
    using storage_type_ = typename ::rocprim::detail::empty_storage_type;

public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords \p __shared__. It can be aliased to
    /// an externally allocated memory, or be a part of a union with other storage types
    /// to increase shared memory reusability.
    #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    using storage_type = typename ::rocprim::detail::empty_storage_type;
    #else
    using storage_type = storage_type_; // only for Doxygen
    #endif

    /// \brief Stores an arrangement of items from across the thread block into an
    /// arrangement on continuous memory.
    ///
    /// \tparam OutputIterator - [inferred] an iterator type for output (can be a simple
    /// pointer.
    ///
    /// \param [out] block_output - the output iterator from the thread block to store to.
    /// \param [in] items - array that data is read from.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked(flat_id, block_output, items);
    }

    /// \brief Stores an arrangement of items from across the thread block into an
    /// arrangement on continuous memory, which is guarded by range \p valid.
    ///
    /// \tparam OutputIterator - [inferred] an iterator type for output (can be a simple
    /// pointer.
    ///
    /// \param [out] block_output - the output iterator from the thread block to store to.
    /// \param [in] items - array that data is read from.
    /// \param [in] valid - maximum range of valid numbers to read.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked(flat_id, block_output, items, valid);
    }

    /// \brief Stores an arrangement of items from across the thread block into an
    /// arrangement on continuous memory, using temporary storage.
    ///
    /// \tparam OutputIterator - [inferred] an iterator type for output (can be a simple
    /// pointer.
    ///
    /// \param [out] block_output - the output iterator from the thread block to store to.
    /// \param [in] items - array that data is read from.
    /// \param [in] storage - temporary storage for outputs.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void kernel(...)
    /// {
    ///     int items[8];
    ///     using block_store_int = rocprim::block_store<int, 128, 8>;
    ///     block_store_int bstore;
    ///     __shared__ typename block_store_int::storage_type storage;
    ///     bstore.store(..., items, storage);
    ///     ...
    /// }
    /// \endcode
    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items);
    }

    /// \brief Stores an arrangement of items from across the thread block into an
    /// arrangement on continuous memory, which is guarded by range \p valid,
    /// using temporary storage
    ///
    /// \tparam OutputIterator - [inferred] an iterator type for output (can be a simple
    /// pointer.
    ///
    /// \param [out] block_output - the output iterator from the thread block to store to.
    /// \param [in] items - array that data is read from.
    /// \param [in] valid - maximum range of valid numbers to read.
    /// \param [in] storage - temporary storage for outputs.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void kernel(...)
    /// {
    ///     int items[8];
    ///     using block_store_int = rocprim::block_store<int, 128, 8>;
    ///     block_store_int bstore;
    ///     __shared__ typename block_store_int::storage_type storage;
    ///     bstore.store(..., items, valid, storage);
    ///     ...
    /// }
    /// \endcode
    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items, valid);
    }
};

/// @}
// end of group blockmodule

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_store<T, BlockSize, ItemsPerThread, block_store_method::block_store_vectorize>
{
private:
    using storage_type_ = typename ::rocprim::detail::empty_storage_type;

public:
    #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    using storage_type = typename ::rocprim::detail::empty_storage_type;
    #else
    using storage_type = storage_type_; // only for Doxygen
    #endif

    ROCPRIM_DEVICE inline
    void store(T* block_output,
               T (&items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked_vectorized(flat_id, block_output, items);
    }

    template<class OutputIterator, class U>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               U (&items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_store_direct_blocked(flat_id, block_output, items, valid);
    }

    ROCPRIM_DEVICE inline
    void store(T* block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items);
    }

    template<class OutputIterator, class U>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               U (&items)[ItemsPerThread],
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items, valid);
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_store<T, BlockSize, ItemsPerThread, block_store_method::block_store_transpose>
{
private:
    using block_exchange_type = block_exchange<T, BlockSize, ItemsPerThread>;

public:
    using storage_type = typename block_exchange_type::storage_type;

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_exchange_type().blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_exchange_type().blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items, valid);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_exchange_type().blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_exchange_type().blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items, valid);
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_store<T, BlockSize, ItemsPerThread, block_store_method::block_store_warp_transpose>
{
private:
    using block_exchange_type = block_exchange<T, BlockSize, ItemsPerThread>;

public:
    static_assert(BlockSize % warp_size() == 0,
                 "BlockSize must be a multiple of hardware warpsize");

    using storage_type = typename block_exchange_type::storage_type;

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_exchange_type().blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_exchange_type().blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items, valid);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_exchange_type().blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        block_exchange_type().blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items, valid);
    }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_STORE_HPP_
