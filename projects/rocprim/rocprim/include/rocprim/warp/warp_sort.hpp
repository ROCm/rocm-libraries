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
/// \tparam VirtualWaveSize [optional] The number of threads in a warp
/// \tparam Value [optional] Data type for parameter Value. By default, it's empty_type
///
/// \par Overview
/// * \p VirtualWaveSize must be power of two.
/// * \p VirtualWaveSize must be equal to or less than the size of hardware warp (see
/// rocprim::arch::wavefront::max_size()). If it is less, sort is performed separately within groups
/// determined by VirtualWaveSize.
/// For example, if \p VirtualWaveSize is 4, hardware warp is 64, sort will be performed in logical
/// warps grouped like this: `{ {0, 1, 2, 3}, {4, 5, 6, 7 }, ..., {60, 61, 62, 63} }`
/// (thread is represented here by its id within hardware warp).
/// * Accepts custom compare_functions for sorting across a warp.
/// * Number of threads executing warp_sort's function must be a multiple of \p VirtualWaveSize.
///
/// \par Stability
/// \p warp_sort is <b>not stable</b>: it doesn't necessarily preserve the relative ordering
/// of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is <b>not guaranteed</b> that \p a will precede \p b as well in the output
/// (ordered) keys.
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
///     const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
///     const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
///
///     int value = input[i];
///     rocprim::warp_sort<int, 64> wsort;
///     wsort.sort(value, customCompare);
///     input[i] = value;
/// }
/// \endcode
/// \endparblock
template<class Key,
         unsigned int VirtualWaveSize           = arch::wavefront::min_size(),
         class Value                            = empty_type,
         arch::wavefront::target TargetWaveSize = arch::wavefront::get_target()>
class warp_sort : detail::warp_sort_shuffle<Key, VirtualWaveSize, Value>
{
    using base_type = typename detail::warp_sort_shuffle<Key, VirtualWaveSize, Value>;

    // Check if VirtualWaveSize is valid for the targets
    static_assert(VirtualWaveSize <= ROCPRIM_MAX_WARP_SIZE,
                  "VirtualWaveSize can't be greater than hardware warp size.");

public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords \p __shared__. It can be aliased to
    /// an externally allocated memory, or be a part of a union with other storage types
    /// to increase shared memory reusability.
    using storage_type = typename base_type::storage_type;

    /// \brief Warp sort for any data type.
    ///
    /// \tparam BinaryFunction type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_key input/output to pass to other threads
    /// \param compare_function binary operation function object that will be used for sort.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& thread_key, BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::sort(thread_key, compare_function);
    }

    /// \brief Warp sort for any data type.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key&, BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)compare_function; // disables unused parameter warning
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return;
    }

    /// \brief Warp sort for any data type.
    ///
    /// \tparam BinaryFunction type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_keys input/output keys to pass to other threads
    /// \param compare_function binary operation function object that will be used for sort.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<unsigned int ItemsPerThread,
             class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::sort(thread_keys, compare_function);
    }

    /// \brief Warp sort for any data type.
    /// Invalid Warp Size
    template<unsigned int ItemsPerThread,
             class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)thread_keys; // disables unused parameter warning
        (void)compare_function; // disables unused parameter warning
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return;
    }

    /// \brief Warp sort for any data type using temporary storage.
    ///
    /// \tparam BinaryFunction type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_key input/output to pass to other threads
    /// \param storage temporary storage for inputs
    /// \param compare_function binary operation function object that will be used for sort.
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
    template<class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key&           thread_key,
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::sort(
            thread_key, storage, compare_function
        );
    }

    /// \brief Warp sort for any data type using temporary storage.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key&, storage_type&, BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)compare_function; // disables unused parameter warning
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return;
    }

    /// \brief Warp sort for any data type using temporary storage.
    ///
    /// \tparam BinaryFunction type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_keys input/output keys to pass to other threads
    /// \param storage temporary storage for inputs
    /// \param compare_function binary operation function object that will be used for sort.
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
    template<unsigned int ItemsPerThread,
             class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::sort(
            thread_keys, storage, compare_function
        );
    }

    /// \brief Warp sort for any data type using temporary storage.
    /// Invalid Warp Size
    template<unsigned int ItemsPerThread,
             class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              storage_type&,
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)thread_keys; // disables unused parameter warning
        (void)compare_function; // disables unused parameter warning
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return;
    }

    /// \brief Warp sort by key for any data type.
    ///
    /// \tparam BinaryFunction type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_key input/output key to pass to other threads
    /// \param thread_value input/output value to pass to other threads
    /// \param compare_function binary operation function object that will be used for sort.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key&           thread_key,
              Value&         thread_value,
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::sort(
            thread_key, thread_value, compare_function
        );
    }

    /// \brief Warp sort by key for any data type.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key&, Value&, BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)compare_function; // disables unused parameter warning
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return;
    }

    /// \brief Warp sort by key for any data type.
    ///
    /// \tparam BinaryFunction type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_keys input/output keys to pass to other threads
    /// \param thread_values input/outputs values to pass to other threads
    /// \param compare_function binary operation function object that will be used for sort.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<unsigned int ItemsPerThread,
             class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::sort(
            thread_keys, thread_values, compare_function
        );
    }

    /// \brief Warp sort by key for any data type.
    /// Invalid Warp Size
    template<unsigned int ItemsPerThread,
             class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)thread_keys; // disables unused parameter warning
        (void)thread_values; // disables unused parameter warning
        (void)compare_function; // disables unused parameter warning
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return;
    }

    /// \brief Warp sort by key for any data type using temporary storage.
    ///
    /// \tparam BinaryFunction type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_key input/output key to pass to other threads
    /// \param thread_value input/output value to pass to other threads
    /// \param storage temporary storage for inputs
    /// \param compare_function binary operation function object that will be used for sort.
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
    template<class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key&           thread_key,
              Value&         thread_value,
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::sort(
            thread_key, thread_value, storage, compare_function
        );
    }

    /// \brief Warp sort by key for any data type using temporary storage.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key&, Value&, storage_type&, BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)compare_function; // disables unused parameter warning
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return;
    }

    /// \brief Warp sort by key for any data type using temporary storage.
    ///
    /// \tparam BinaryFunction type of binary function used for sort. Default type
    /// is rocprim::less<T>.
    ///
    /// \param thread_keys input/output keys to pass to other threads
    /// \param thread_values input/output values to pass to other threads
    /// \param storage temporary storage for inputs
    /// \param compare_function binary operation function object that will be used for sort.
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
    template<unsigned int ItemsPerThread,
             class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::sort(
            thread_keys, thread_values, storage, compare_function
        );
    }

    /// \brief Warp sort by key for any data type using temporary storage.
    /// Invalid Warp Size
    template<unsigned int ItemsPerThread,
             class BinaryFunction                 = ::rocprim::less<Key>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&,
              BinaryFunction compare_function = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)thread_keys; // disables unused parameter warning
        (void)thread_values; // disables unused parameter warning
        (void)compare_function; // disables unused parameter warning
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return;
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_SORT_HPP_
