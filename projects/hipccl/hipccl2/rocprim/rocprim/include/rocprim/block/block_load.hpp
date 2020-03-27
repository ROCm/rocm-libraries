// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "block_load_func.hpp"
#include "block_exchange.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief \p block_load_method enumerates the methods available to load data
/// from continuous memory into a blocked arrangement of items across the thread block
enum class block_load_method
{
    /// Data from continuous memory is loaded into a blocked arrangement of items.
    /// \par Performance Notes:
    /// * Performance decreases with increasing number of items per thread (stride
    /// between reads), because of reduced memory coalescing.
    block_load_direct,

    /// Data from continuous memory is loaded into a blocked arrangement of items
    /// using vectorization as an optimization.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, provided that
    /// vectorization requirements are fulfilled. Otherwise, performance will default
    /// to \p block_load_direct.
    /// \par Requirements:
    /// * The input offset (\p block_input) must be quad-item aligned.
    /// * The following conditions will prevent vectorization and switch to default
    /// \p block_load_direct:
    ///   * \p ItemsPerThread is odd.
    ///   * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
    /// int4, etc.
    block_load_vectorize,

    /// A striped arrangement of data from continuous memory is locally transposed
    /// into a blocked arrangement of items.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, regardless of the
    /// number of items per thread.
    /// * Performance may be better compared to \p block_load_direct and
    /// \p block_load_vectorize due to reordering on local memory.
    block_load_transpose,

    /// A warp-striped arrangement of data from continuous memory is locally transposed
    /// into a blocked arrangement of items.
    /// \par Requirements:
    /// * The number of threads in the block must be a multiple of the size of hardware warp.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, regardless of the
    /// number of items per thread.
    /// * Performance may be better compared to \p block_load_direct and
    /// \p block_load_vectorize due to reordering on local memory.
    block_load_warp_transpose,

    /// Defaults to \p block_load_direct
    default_method = block_load_direct
};

/// \brief The \p block_load class is a block level parallel primitive which provides methods
/// for loading data from continuous memory into a blocked arrangement of items across the thread
/// block.
///
/// \tparam T - the input/output type.
/// \tparam BlockSize - the number of threads in a block.
/// \tparam ItemsPerThread - the number of items to be processed by
/// each thread.
/// \tparam Method - the method to load data.
///
/// \par Overview
/// * The \p block_load class has a number of different methods to load data:
///   * [block_load_direct](\ref ::block_load_method::block_load_direct)
///   * [block_load_vectorize](\ref ::block_load_method::block_load_vectorize)
///   * [block_load_transpose](\ref ::block_load_method::block_load_transpose)
///   * [block_load_warp_transpose](\ref ::block_load_method::block_load_warp_transpose)
///
/// \par Example:
/// \parblock
/// In the examples load operation is performed on block of 128 threads, using type
/// \p int and 8 items per thread.
///
/// \code{.cpp}
/// __global__ void example_kernel(int * input, ...)
/// {
///     const int offset = hipBlockIdx_x * 128 * 8;
///     int items[8];
///     rocprim::block_load<int, 128, 8, load_method> blockload;
///     blockload.load(input + offset, items);
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    block_load_method Method = block_load_method::block_load_direct,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_load
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

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// thread block.
    ///
    /// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
    /// pointer.
    ///
    /// \param [in] block_input - the input iterator from the thread block to load from.
    /// \param [out] items - array that data is loaded to.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items);
    }

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// thread block, which is guarded by range \p valid.
    ///
    /// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
    /// pointer.
    ///
    /// \param [in] block_input - the input iterator from the thread block to load from.
    /// \param [out] items - array that data is loaded to.
    /// \param [in] valid - maximum range of valid numbers to load.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items, valid);
    }

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// thread block, which is guarded by range with a fall-back value for out-of-bound
    /// elements.
    ///
    /// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
    /// pointer.
    /// \tparam Default - [inferred] The data type of the default value.
    ///
    /// \param [in] block_input - the input iterator from the thread block to load from.
    /// \param [out] items - array that data is loaded to.
    /// \param [in] valid - maximum range of valid numbers to load.
    /// \param [in] out_of_bounds - default value assigned to out-of-bound items.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items, valid,
                                  out_of_bounds);
    }

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// thread block, using temporary storage.
    ///
    /// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
    /// pointer.
    ///
    /// \param [in] block_input - the input iterator from the thread block to load from.
    /// \param [out] items - array that data is loaded to.
    /// \param [in] storage - temporary storage for inputs.
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
    /// __global__ void example_kernel(...)
    /// {
    ///     int items[8];
    ///     using block_load_int = rocprim::block_load<int, 128, 8>;
    ///     block_load_int bload;
    ///     __shared__ typename block_load_int::storage_type storage;
    ///     bload.load(..., items, storage);
    ///     ...
    /// }
    /// \endcode
    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items);
    }

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// thread block, which is guarded by range \p valid, using temporary storage.
    ///
    /// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
    /// pointer
    ///
    /// \param [in] block_input - the input iterator from the thread block to load from.
    /// \param [out] items - array that data is loaded to.
    /// \param [in] valid - maximum range of valid numbers to load.
    /// \param [in] storage - temporary storage for inputs.
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
    /// __global__ void example_kernel(...)
    /// {
    ///     int items[8];
    ///     using block_load_int = rocprim::block_load<int, 128, 8>;
    ///     block_load_int bload;
    ///     tile_static typename block_load_int::storage_type storage;
    ///     bload.load(..., items, valid, storage);
    ///     ...
    /// }
    /// \endcode
    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items, valid);
    }

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// thread block, which is guarded by range with a fall-back value for out-of-bound
    /// elements, using temporary storage.
    ///
    /// \tparam InputIterator - [inferred] an iterator type for input (can be a simple
    /// pointer.
    /// \tparam Default - [inferred] The data type of the default value.
    ///
    /// \param [in] block_input - the input iterator from the thread block to load from.
    /// \param [out] items - array that data is loaded to.
    /// \param [in] valid - maximum range of valid numbers to load.
    /// \param [in] out_of_bounds - default value assigned to out-of-bound items.
    /// \param [in] storage - temporary storage for inputs.
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
    /// __global__ void example_kernel(...)
    /// {
    ///     int items[8];
    ///     using block_load_int = rocprim::block_load<int, 128, 8>;
    ///     block_load_int bload;
    ///     __shared__ typename block_load_int::storage_type storage;
    ///     bload.load(..., items, valid, out_of_bounds, storage);
    ///     ...
    /// }
    /// \endcode
    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items, valid, out_of_bounds);
    }
};

/// @}
// end of group blockmodule

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_load<T, BlockSizeX, ItemsPerThread, block_load_method::block_load_vectorize, BlockSizeY, BlockSizeZ>
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
    void load(T* block_input,
              T (&items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked_vectorized(flat_id, block_input, items);
    }

    template<class InputIterator, class U>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              U (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items, valid);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items, valid,
                                  out_of_bounds);
    }

    ROCPRIM_DEVICE inline
    void load(T* block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        (void) storage;
        load(block_input, items);
    }

    template<class InputIterator, class U>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              U (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items, valid);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items, valid, out_of_bounds);
    }
};

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_load<T, BlockSizeX, ItemsPerThread, block_load_method::block_load_transpose, BlockSizeY, BlockSizeZ>
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

private:
    using block_exchange_type = block_exchange<T, BlockSize, ItemsPerThread>;

public:
    using storage_type = typename block_exchange_type::storage_type;

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid,
                                             out_of_bounds);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid,
                                             out_of_bounds);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }
};

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_load<T, BlockSizeX, ItemsPerThread, block_load_method::block_load_warp_transpose, BlockSizeY, BlockSizeZ>
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
private:
    using block_exchange_type = block_exchange<T, BlockSizeX, ItemsPerThread, BlockSizeY, BlockSizeZ>;

public:
    static_assert(BlockSize % warp_size() == 0,
                 "BlockSize must be a multiple of hardware warpsize");

    using storage_type = typename block_exchange_type::storage_type;

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items, valid);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);

    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items, valid,
                                       out_of_bounds);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items, valid);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items, valid,
                                       out_of_bounds);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_BLOCK_LOAD_HPP_
