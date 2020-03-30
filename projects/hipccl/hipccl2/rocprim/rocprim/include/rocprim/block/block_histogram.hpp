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

#ifndef ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_
#define ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_histogram_atomic.hpp"
#include "detail/block_histogram_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup blockmodule
/// @{

/// \brief Available algorithms for block_histogram primitive.
enum class block_histogram_algorithm
{
    /// Atomic addition is used to update bin count directly.
    /// \par Performance Notes:
    /// * Performance is dependent on hardware implementation of atomic addition.
    /// * Performance may decrease for non-uniform random input distributions
    /// where many concurrent updates may be made to the same bin counter.
    using_atomic,

    /// A two-phase operation is used:-
    /// * Data is sorted using radix-sort.
    /// * "Runs" of same-valued keys are detected using discontinuity; run-lengths
    /// are bin counts.
    /// \par Performance Notes:
    /// * Performance is consistent regardless of sample bin distribution.
    using_sort,

    /// \brief Default block_histogram algorithm.
    default_algorithm = using_atomic,
};

namespace detail
{

// Selector for block_histogram algorithm which gives block histogram implementation
// type based on passed block_histogram_algorithm enum
template<block_histogram_algorithm Algorithm>
struct select_block_histogram_impl;

template<>
struct select_block_histogram_impl<block_histogram_algorithm::using_atomic>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ, unsigned int ItemsPerThread, unsigned int Bins>
    using type = block_histogram_atomic<T, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Bins>;
};

template<>
struct select_block_histogram_impl<block_histogram_algorithm::using_sort>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ, unsigned int ItemsPerThread, unsigned int Bins>
    using type = block_histogram_sort<T, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Bins>;
};

} // end namespace detail

/// \brief The block_histogram class is a block level parallel primitive which provides methods
/// for constructing block-wide histograms from items partitioned across threads in a block.
///
/// \tparam T - the input/output type.
/// \tparam BlockSize - the number of threads in a block.
/// \tparam ItemsPerThread - the number of items to be processed by each thread.
/// \tparam Bins - the number of bins within the histogram.
/// \tparam Algorithm - selected histogram algorithm, block_histogram_algorithm::default_algorithm by default.
///
/// \par Overview
/// * block_histogram has two alternative implementations: \p block_histogram_algorithm::using_atomic
///   and block_histogram_algorithm::using_sort.
///
/// \par Examples
/// \parblock
/// In the examples histogram operation is performed on block of 192 threads, each provides
/// one \p int value, result is returned using the same variable as for input.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize block_histogram for int, logical block of 192 threads,
///     // 2 items per thread and a bin size of 192.
///     using block_histogram_int = rocprim::block_histogram<int, 192, 2, 192>;
///     // allocate storage in shared memory
///     __shared__ block_histogram_int::storage_type storage;
///     __shared__ int hist[192];
///
///     int value[2];
///     ...
///     // execute histogram
///     block_histogram_int().histogram(
///         value, // input
///         hist, // output
///         storage
///     );
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int Bins,
    block_histogram_algorithm Algorithm = block_histogram_algorithm::default_algorithm,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_histogram
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_histogram_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Bins>
#endif
{
    using base_type = typename detail::select_block_histogram_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Bins>;
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
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

    /// \brief Initialize histogram counters to zero.
    ///
    /// \tparam Counter - [inferred] counter type of histogram.
    ///
    /// \param [out] hist - histogram bin count.
    template<class Counter>
    ROCPRIM_DEVICE inline
    void init_histogram(Counter hist[Bins])
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        #pragma unroll
        for(unsigned int offset = 0; offset < Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            if(offset_tid < Bins)
            {
                hist[offset_tid] = Counter();
            }
        }
    }

    /// \brief Update an existing block-wide histogram. Each thread composites an array of
    /// input elements.
    ///
    /// \tparam Counter - [inferred] counter type of histogram.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] hist - histogram bin count.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples histogram operation is performed on block of 192 threads, each provides
    /// one \p int value, result is returned using the same variable as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_histogram for int, logical block of 192 threads,
    ///     // 2 items per thread and a bin size of 192.
    ///     using block_histogram_int = rocprim::block_histogram<int, 192, 2, 192>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_histogram_int::storage_type storage;
    ///     __shared__ int hist[192];
    ///
    ///     int value[2];
    ///     ...
    ///     // initialize histogram
    ///     block_histogram_int().init_histogram(
    ///         hist // output
    ///     );
    ///
    ///     rocprim::syncthreads();
    ///
    ///     // update histogram
    ///     block_histogram_int().composite(
    ///         value, // input
    ///         hist, // output
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    /// \endparblock
    template<class Counter>
    ROCPRIM_DEVICE inline
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        base_type::composite(input, hist, storage);
    }

    /// \overload
    /// \brief Update an existing block-wide histogram. Each thread composites an array of
    /// input elements.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam Counter - [inferred] counter type of histogram.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] hist - histogram bin count.
    template<class Counter>
    ROCPRIM_DEVICE inline
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        base_type::composite(input, hist);
    }

    /// \brief Construct a new block-wide histogram. Each thread contributes an array of
    /// input elements.
    ///
    /// \tparam Counter - [inferred] counter type of histogram.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] hist - histogram bin count.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples histogram operation is performed on block of 192 threads, each provides
    /// one \p int value, result is returned using the same variable as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_histogram for int, logical block of 192 threads,
    ///     // 2 items per thread and a bin size of 192.
    ///     using block_histogram_int = rocprim::block_histogram<int, 192, 2, 192>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_histogram_int::storage_type storage;
    ///     __shared__ int hist[192];
    ///
    ///     int value[2];
    ///     ...
    ///     // execute histogram
    ///     block_histogram_int().histogram(
    ///         value, // input
    ///         hist, // output
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    /// \endparblock
    template<class Counter>
    ROCPRIM_DEVICE inline
    void histogram(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        init_histogram(hist);
        ::rocprim::syncthreads();
        composite(input, hist, storage);
    }

    /// \overload
    /// \brief Construct a new block-wide histogram. Each thread contributes an array of
    /// input elements.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam Counter - [inferred] counter type of histogram.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] hist - histogram bin count.
    template<class Counter>
    ROCPRIM_DEVICE inline
    void histogram(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        init_histogram(hist);
        ::rocprim::syncthreads();
        composite(input, hist);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_
