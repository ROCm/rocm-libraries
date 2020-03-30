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

#ifndef ROCPRIM_WARP_WARP_SORT_HPP_
#define ROCPRIM_WARP_WARP_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/warp_sort_shuffle.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The warp_sort class provides warp-wide methods for computing a parallel
/// sort of items across thread warps. This class currently implements parallel
/// bitonic sort, and only accepts warp sizes that are powers of two.
///
/// \tparam Key Data type for parameter Key
/// \tparam WarpSize [optional] The number of threads in a warp
/// \tparam Value [optional] Data type for parameter Value. By default, it's empty_type
///
/// \par Overview
/// * \p WarpSize must be power of two.
/// * \p WarpSize must be equal to or less than the size of hardware warp (see
/// rocprim::warp_size()). If it is less, sort is performed separately within groups
/// determined by WarpSize.
/// For example, if \p WarpSize is 4, hardware warp is 64, sort will be performed in logical
/// warps grouped like this: `{ {0, 1, 2, 3}, {4, 5, 6, 7 }, ..., {60, 61, 62, 63} }`
/// (thread is represented here by its id within hardware warp).
/// * Accepts custom compare_functions for sorting across a warp.
/// * Number of threads executing warp_sort's function must be a multiple of \p WarpSize.
///
/// \par Example:
/// \parblock
/// Every thread within the warp uses the warp_sort class by first specializing the
/// warp_sort type, and instantiating an object that will be used to invoke a
/// member function.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
///
///     int value = input[i];
///     rocprim::warp_sort<int, 64> wsort;
///     wsort.sort(value);
///     input[i] = value;
/// }
/// \endcode
///
/// Below is a snippet demonstrating how to pass a custom compare function:
/// \code{.cpp}
/// __device__ bool customCompare(const int& a, const int& b)
/// {
///     return a < b;
/// }
/// ...
/// __global__ void example_kernel(...)
/// {
///     const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
///
///     int value = input[i];
///     rocprim::warp_sort<int, 64> wsort;
///     wsort.sort(value, customCompare);
///     input[i] = value;
/// }
/// \endcode
/// \endparblock
template<
    class Key,
    unsigned int WarpSize = warp_size(),
    class Value = empty_type
>
class warp_sort : detail::warp_sort_shuffle<Key, WarpSize, Value>
{
    typedef typename detail::warp_sort_shuffle<Key, WarpSize, Value> base_type;

    // Check if WarpSize is correct
    static_assert(WarpSize <= warp_size(), "WarpSize can't be greater than hardware warp size.");

public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords \p __shared__. It can be aliased to
    /// an externally allocated memory, or be a part of a union with other storage types
    /// to increase shared memory reusability.
    typedef typename base_type::storage_type storage_type;

    /// \brief Warp sort for any data type.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_key - input/output to pass to other threads
    /// \param compare_function - binary operation function object that will be used for sort.
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

    /// \brief Warp sort for any data type using temporary storage.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_key - input/output to pass to other threads
    /// \param storage - temporary storage for inputs
    /// \param compare_function - binary operation function object that will be used for sort.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     int value = ...;
    ///     using warp_sort_int = rp::warp_sort<int, 64>;
    ///     warp_sort_int wsort;
    ///     __shared__ typename warp_sort_int::storage_type storage;
    ///     wsort.sort(value, storage);
    ///     ...
    /// }
    /// \endcode
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(
            thread_key, storage, compare_function
        );
    }

    /// \brief Warp sort by key for any data type.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_key - input/output key to pass to other threads
    /// \param thread_value - input/output value to pass to other threads
    /// \param compare_function - binary operation function object that will be used for sort.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(
            thread_key, thread_value, compare_function
        );
    }

    /// \brief Warp sort by key for any data type using temporary storage.
    ///
    /// \tparam BinaryFunction - type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_key - input/output key to pass to other threads
    /// \param thread_value - input/output value to pass to other threads
    /// \param storage - temporary storage for inputs
    /// \param compare_function - binary operation function object that will be used for sort.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     int value = ...;
    ///     using warp_sort_int = rp::warp_sort<int, 64>;
    ///     warp_sort_int wsort;
    ///     __shared__ typename warp_sort_int::storage_type storage;
    ///     wsort.sort(key, value, storage);
    ///     ...
    /// }
    /// \endcode
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE inline
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(
            thread_key, thread_value, storage, compare_function
        );
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_SORT_HPP_
