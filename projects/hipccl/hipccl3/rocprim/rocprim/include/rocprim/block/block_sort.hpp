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

#ifndef ROCPRIM_BLOCK_BLOCK_SORT_HPP_
#define ROCPRIM_BLOCK_BLOCK_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_sort_bitonic.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for block_sort primitive.
enum class block_sort_algorithm
{
    /// \brief A bitonic sort based algorithm.
    bitonic_sort,
    /// \brief Default block_sort algorithm.
    default_algorithm = bitonic_sort,
};

namespace detail
{

// Selector for block_sort algorithm which gives block sort implementation
// type based on passed block_sort_algorithm enum
template<block_sort_algorithm Algorithm>
struct select_block_sort_impl;

template<>
struct select_block_sort_impl<block_sort_algorithm::bitonic_sort>
{
    template<class Key, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ, class Value>
    using type = block_sort_bitonic<Key, BlockSizeX, BlockSizeY, BlockSizeZ, Value>;
};

} // end namespace detail

/// \brief The block_sort class is a block level parallel primitive which provides
/// methods sorting items (keys or key-value pairs) partitioned across threads in a block
/// using comparison-based sort algorithm.
///
/// \tparam Key - the key type.
/// \tparam BlockSize - the number of threads in a block.
/// \tparam Value - the value type. Default type empty_type indicates
/// a keys-only sort.
/// \tparam Algorithm - selected sort algorithm, block_sort_algorithm::default_algorithm by default.
///
/// \par Overview
/// * Accepts custom compare_functions for sorting across a block.
/// * Performance depends on \p BlockSize.
///   * It is better if \p BlockSize is a power of two.
///   * If \p BlockSize is not a power of two, or when function with \p size overload is used
///     odd-even sort is used instead of bitonic sort, leading to decreased performance.
///
/// \par Examples
/// \parblock
/// In the examples sort is performed on a block of 256 threads, each thread provides
/// one \p int value, results are returned using the same variable as for input.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize block_sort for int, block of 256 threads,
///     // key-only sort
///     using block_sort_int = rocprim::block_sort<int, 256>;
///     // allocate storage in shared memory
///     __shared__ block_sort_int::storage_type storage;
///
///     int input = ...;
///     // execute block sort (ascending)
///     block_sort_int().sort(
///         input,
///         storage
///     );
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class Key,
    unsigned int BlockSizeX,
    class Value = empty_type,
    block_sort_algorithm Algorithm = block_sort_algorithm::default_algorithm,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_sort
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_sort_impl<Algorithm>::template type<Key, BlockSizeX, BlockSizeY, BlockSizeZ, Value>
#endif
{
    using base_type = typename detail::select_block_sort_impl<Algorithm>::template type<Key, BlockSizeX, BlockSizeY, BlockSizeZ, Value>;
public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    using storage_type = typename base_type::storage_type;

    /// \brief Block sort for any data type.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param [in, out] thread_key - reference to a key provided by a thread.
    /// \param [in] compare_function - comparison function object which returns true if the
    /// first argument is is ordered before the second.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, compare_function);
    }

    /// \brief Block sort for any data type.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param [in, out] thread_key - reference to a key provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] compare_function - comparison function object which returns true if the
    /// first argument is is ordered before the second.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples sort is performed on a block of 256 threads, each thread provides
    /// one \p int value, results are returned using the same variable as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_sort for int, block of 256 threads,
    ///     // key-only sort
    ///     using block_sort_int = rocprim::block_sort<int, 256>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_sort_int::storage_type storage;
    ///
    ///     int input = ...;
    ///     // execute block sort (ascending)
    ///     block_sort_int().sort(
    ///         input,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    /// \endparblock
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, storage, compare_function);
    }

    /// \brief Block sort by key for any data type.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param [in, out] thread_key - reference to a key provided by a thread.
    /// \param [in, out] thread_value - reference to a value provided by a thread.
    /// \param [in] compare_function - comparison function object which returns true if the
    /// first argument is is ordered before the second.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, thread_value, compare_function);
    }

    /// \brief Block sort by key for any data type.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param [in, out] thread_key - reference to a key provided by a thread.
    /// \param [in, out] thread_value - reference to a value provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] compare_function - comparison function object which returns true if the
    /// first argument is is ordered before the second.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \parblock
    /// In the examples sort is performed on a block of 256 threads, each thread provides
    /// one \p int key and one \p int value, results are returned using the same variable as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_sort for int, block of 256 threads,
    ///     using block_sort_int = rocprim::block_sort<int, 256, int>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_sort_int::storage_type storage;
    ///
    ///     int key = ...;
    ///     int value = ...;
    ///     // execute block sort (ascending)
    ///     block_sort_int().sort(
    ///         key,
    ///         value,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    /// \endparblock
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, thread_value, storage, compare_function);
    }

    /// \brief Block sort by key for any data type. If \p size is
    /// greater than \p BlockSize, this function does nothing.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param [in, out] thread_key - reference to a key provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] size - custom size of block to be sorted.
    /// \param [in] compare_function - comparison function object which returns true if the
    /// first argument is is ordered before the second.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              storage_type& storage,
              const unsigned int size,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, storage, size, compare_function);
    }

    /// \brief Block sort by key for any data type. If \p size is
    /// greater than \p BlockSize, this function does nothing.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param [in, out] thread_key - reference to a key provided by a thread.
    /// \param [in, out] thread_value - reference to a value provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] size - custom size of block to be sorted.
    /// \param [in] compare_function - comparison function object which returns true if the
    /// first argument is is ordered before the second.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              const unsigned int size,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, thread_value, storage, size, compare_function);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_SORT_HPP_
