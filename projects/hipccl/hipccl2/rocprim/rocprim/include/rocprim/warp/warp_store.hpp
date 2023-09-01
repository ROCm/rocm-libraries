// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_WARP_WARP_STORE_HPP_
#define ROCPRIM_WARP_WARP_STORE_HPP_

#include "../config.hpp"
#include "../intrinsics.hpp"
#include "../detail/various.hpp"

#include "warp_exchange.hpp"
#include "../block/block_store_func.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief \p warp_store_method enumerates the methods available to store a blocked/striped
/// arrangement of items into a blocked/striped arrangement in continuous memory
enum class warp_store_method
{
    /// A blocked arrangement of items is stored into a blocked arrangement on continuous
    /// memory.
    /// \par Performance Notes:
    /// * Performance decreases with increasing number of items per thread (stride
    /// between reads), because of reduced memory coalescing.
    warp_store_direct,

    /// A striped arrangement of items is stored into a blocked arrangement on continuous
    /// memory.
    warp_store_striped,

    /// A blocked arrangement of items is stored into a blocked arrangement on continuous
    /// memory using vectorization as an optimization.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, provided that
    /// vectorization requirements are fulfilled. Otherwise, performance will default
    /// to \p warp_store_direct.
    /// \par Requirements:
    /// * The output offset (\p block_output) must be quad-item aligned.
    /// * The following conditions will prevent vectorization and switch to default
    /// \p warp_store_direct:
    ///   * \p ItemsPerThread is odd.
    ///   * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
    /// int4, etc.
    warp_store_vectorize,

    /// A blocked arrangement of items is locally transposed and stored as a striped
    /// arrangement of data on continuous memory.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, regardless of the
    /// number of items per thread.
    /// * Performance may be better compared to \p warp_store_direct and
    /// \p warp_store_vectorize due to reordering on local memory.
    warp_store_transpose,

    /// Defaults to \p warp_store_direct
    default_method = warp_store_direct
};

/// \brief The \p warp_store class is a warp level parallel primitive which provides methods
/// for storing an arrangement of items into a blocked/striped arrangement on continous memory.
///
/// \tparam T - the output/output type.
/// \tparam ItemsPerThread - the number of items to be processed by
/// each thread.
/// \tparam WarpSize - the number of threads in a warp. It must be a divisor of the
/// kernel block size.
/// \tparam Method - the method to store data.
///
/// \par Overview
/// * The \p warp_store class has a number of different methods to store data:
///   * [warp_store_direct](\ref ::warp_store_method::warp_store_direct)
///   * [warp_store_striped](\ref ::warp_store_method::warp_store_striped)
///   * [warp_store_vectorize](\ref ::warp_store_method::warp_store_vectorize)
///   * [warp_store_transpose](\ref ::warp_store_method::warp_store_transpose)
///
/// \par Example:
/// \parblock
/// In the example a store operation is performed on a warp of 8 threads, using type
/// \p int and 4 items per thread.
///
/// \code{.cpp}
/// __global__ void example_kernel(int * output, ...)
/// {
///     constexpr unsigned int threads_per_block = 128;
///     constexpr unsigned int threads_per_warp  =   8;
///     constexpr unsigned int items_per_thread  =   4;
///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
///     const int offset = blockIdx.x * threads_per_block * items_per_thread
///         + warp_id * threads_per_warp * items_per_thread;
///     int items[items_per_thread];
///     rocprim::warp_store<int, items_per_thread, threads_per_warp, load_method> warp_store;
///     warp_store.store(output + offset, items);
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize = ::rocprim::device_warp_size(),
    warp_store_method Method = warp_store_method::warp_store_direct
>
class warp_store
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

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

    /// \brief Stores an arrangement of items from across the warp into an
    /// arrangement on continuous memory.
    ///
    /// \tparam OutputIterator - [inferred] an iterator type for output (can be a simple
    /// pointer.
    ///
    /// \param [out] output - the output iterator to store to.
    /// \param [in] items - array that data is read from.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p OutputIterator
    /// can be dereferenced and then implicitly assigned from \p T.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked(flat_id, output, items);
    }

    /// \brief Stores an arrangement of items from across the warp into an
    /// arrangement on continuous memory, which is guarded by range \p valid,
    /// using temporary storage
    ///
    /// \tparam OutputIterator - [inferred] an iterator type for output (can be a simple
    /// pointer.
    ///
    /// \param [out] output - the output iterator to store to.
    /// \param [in] items - array that data is read from.
    /// \param [in] valid - maximum range of valid numbers to read.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p OutputIterator
    /// can be dereferenced and then implicitly assigned from \p T.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked(flat_id, output, items, valid);
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_store<T, ItemsPerThread, WarpSize, warp_store_method::warp_store_striped>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

public:
    using storage_type = typename ::rocprim::detail::empty_storage_type;

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_warp_striped<WarpSize>(flat_id, output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_warp_striped<WarpSize>(flat_id, output, items, valid);
    }
};

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_store<T, ItemsPerThread, WarpSize, warp_store_method::warp_store_vectorize>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

public:
    using storage_type = typename ::rocprim::detail::empty_storage_type;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(T* output,
               T (&items)[ItemsPerThread],
               storage_type& /*storage*/)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked_vectorized(flat_id, output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked(flat_id, output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked(flat_id, output, items, valid);
    }
};

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_store<T, ItemsPerThread, WarpSize, warp_store_method::warp_store_transpose>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

private:
    using exchange_type = ::rocprim::warp_exchange<T, ItemsPerThread, WarpSize>;

public:
    using storage_type = typename exchange_type::storage_type;

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        exchange_type().blocked_to_striped(items, items, storage);
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_warp_striped<WarpSize>(flat_id, output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        exchange_type().blocked_to_striped(items, items, storage);
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_warp_striped<WarpSize>(flat_id, output, items, valid);
    }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_STORE_HPP_
