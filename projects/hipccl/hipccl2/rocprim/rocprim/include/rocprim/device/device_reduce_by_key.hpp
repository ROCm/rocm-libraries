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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_

#include <iterator>
#include <iostream>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "../functional.hpp"

#include "device_reduce_by_key_config.hpp"
#include "detail/device_reduce_by_key.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class KeyCompareFunction
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void fill_unique_counts_kernel(KeysInputIterator keys_input,
                               unsigned int size,
                               unsigned int * unique_counts,
                               KeyCompareFunction key_compare_op,
                               unsigned int blocks_per_full_batch,
                               unsigned int full_batches)
{
    fill_unique_counts<BlockSize, ItemsPerThread>(
        keys_input, size,
        unique_counts,
        key_compare_op,
        blocks_per_full_batch, full_batches
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class UniqueCountOutputIterator
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void scan_unique_counts_kernel(unsigned int * unique_counts,
                               UniqueCountOutputIterator unique_count_output,
                               unsigned int batches)
{
    scan_unique_counts<BlockSize, ItemsPerThread>(unique_counts, unique_count_output, batches);
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class ValuesInputIterator,
    class Result,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class KeyCompareFunction,
    class BinaryFunction
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void reduce_by_key_kernel(KeysInputIterator keys_input,
                          ValuesInputIterator values_input,
                          unsigned int size,
                          const unsigned int * unique_starts,
                          carry_out<Result> * carry_outs,
                          Result * leading_aggregates,
                          UniqueOutputIterator unique_output,
                          AggregatesOutputIterator aggregates_output,
                          KeyCompareFunction key_compare_op,
                          BinaryFunction reduce_op,
                          unsigned int blocks_per_full_batch,
                          unsigned int full_batches)
{
    reduce_by_key<BlockSize, ItemsPerThread>(
        keys_input, values_input, size,
        unique_starts, carry_outs, leading_aggregates,
        unique_output, aggregates_output,
        key_compare_op, reduce_op,
        blocks_per_full_batch, full_batches
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class Result,
    class AggregatesOutputIterator,
    class BinaryFunction
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void scan_and_scatter_carry_outs_kernel(const carry_out<Result> * carry_outs,
                                        const Result * leading_aggregates,
                                        AggregatesOutputIterator aggregates_output,
                                        BinaryFunction reduce_op,
                                        unsigned int batches)
{
    scan_and_scatter_carry_outs<BlockSize, ItemsPerThread>(
        carry_outs, leading_aggregates, aggregates_output,
        reduce_op,
        batches
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        auto error = hipPeekAtLastError(); \
        if(error != hipSuccess) return error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto error = hipStreamSynchronize(stream); \
            if(error != hipSuccess) return error; \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<
    class Config,
    class KeysInputIterator,
    class ValuesInputIterator,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class UniqueCountOutputIterator,
    class BinaryFunction,
    class KeyCompareFunction
>
inline
hipError_t reduce_by_key_impl(void * temporary_storage,
                              size_t& storage_size,
                              KeysInputIterator keys_input,
                              ValuesInputIterator values_input,
                              const unsigned int size,
                              UniqueOutputIterator unique_output,
                              AggregatesOutputIterator aggregates_output,
                              UniqueCountOutputIterator unique_count_output,
                              BinaryFunction reduce_op,
                              KeyCompareFunction key_compare_op,
                              const hipStream_t stream,
                              const bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        typename std::iterator_traits<ValuesInputIterator>::value_type,
        BinaryFunction
    >::type;
    using carry_out_type = carry_out<result_type>;

    using config = default_or_custom_config<
        Config,
        default_reduce_by_key_config<ROCPRIM_TARGET_ARCH, key_type, result_type>
    >;

    constexpr unsigned int items_per_block = config::reduce::block_size * config::reduce::items_per_thread;
    constexpr unsigned int scan_items_per_block = config::scan::block_size * config::scan::items_per_thread;

    const unsigned int blocks = std::max(1u, ::rocprim::detail::ceiling_div(size, items_per_block));
    const unsigned int blocks_per_full_batch = ::rocprim::detail::ceiling_div(blocks, scan_items_per_block);
    const unsigned int full_batches = blocks % scan_items_per_block != 0
        ? blocks % scan_items_per_block
        : scan_items_per_block;
    const unsigned int batches = (blocks_per_full_batch == 1 ? full_batches : scan_items_per_block);

    const size_t unique_counts_bytes = ::rocprim::detail::align_size(batches * sizeof(unsigned int));
    const size_t carry_outs_bytes = ::rocprim::detail::align_size(batches * sizeof(carry_out_type));
    const size_t leading_aggregates_bytes = ::rocprim::detail::align_size(batches * sizeof(result_type));
    if(temporary_storage == nullptr)
    {
        storage_size = unique_counts_bytes + carry_outs_bytes + leading_aggregates_bytes;
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        std::cout << "blocks_per_full_batch " << blocks_per_full_batch << '\n';
        std::cout << "full_batches " << full_batches << '\n';
        std::cout << "batches " << batches << '\n';
        std::cout << "storage_size " << storage_size << '\n';
        hipError_t error = hipStreamSynchronize(stream);
        if(error != hipSuccess) return error;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    unsigned int * unique_counts = reinterpret_cast<unsigned int *>(ptr);
    ptr += unique_counts_bytes;
    carry_out_type * carry_outs = reinterpret_cast<carry_out_type *>(ptr);
    ptr += carry_outs_bytes;
    result_type * leading_aggregates = reinterpret_cast<result_type *>(ptr);

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(fill_unique_counts_kernel<config::reduce::block_size, config::reduce::items_per_thread>),
        dim3(batches), dim3(config::reduce::block_size), 0, stream,
        keys_input, size, unique_counts, key_compare_op,
        blocks_per_full_batch, full_batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("fill_unique_counts", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scan_unique_counts_kernel<config::scan::block_size, config::scan::items_per_thread>),
        dim3(1), dim3(config::scan::block_size), 0, stream,
        unique_counts, unique_count_output,
        batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_unique_counts", config::scan::block_size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(reduce_by_key_kernel<config::reduce::block_size, config::reduce::items_per_thread>),
        dim3(batches), dim3(config::reduce::block_size), 0, stream,
        keys_input, values_input, size,
        const_cast<const unsigned int *>(unique_counts), carry_outs, leading_aggregates,
        unique_output, aggregates_output,
        key_compare_op, reduce_op,
        blocks_per_full_batch, full_batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("reduce_by_key", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scan_and_scatter_carry_outs_kernel<config::scan::block_size, config::scan::items_per_thread>),
        dim3(1), dim3(config::scan::block_size), 0, stream,
        const_cast<const carry_out_type *>(carry_outs), const_cast<const result_type *>(leading_aggregates),
        aggregates_output,
        reduce_op,
        batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_and_scatter_carry_outs", config::scan::block_size, start)

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end of detail namespace

/// \brief Parallel reduce-by-key primitive for device level.
///
/// reduce_by_key function performs a device-wide reduction operation of groups
/// of consecutive values having the same key using binary \p reduce_op operator. The first key of each group
/// is copied to \p unique_output and reduction of the group is written to \p aggregates_output.
/// The total number of group is written to \p unique_count_output.
///
/// \par Overview
/// * Supports non-commutative reduction operators. However, a reduction operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input and \p values_input must have at least \p size elements.
/// * Range specified by \p unique_count_output must have at least 1 element.
/// * Ranges specified by \p unique_output and \p aggregates_output must have at least
/// <tt>*unique_count_output</tt> (i.e. the number of unique keys) elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p reduce_by_key_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam UniqueOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam AggregatesOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam UniqueCountOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for reduction. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p ValuesInputIterator.
/// \tparam KeyCompareFunction - type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - iterator to the first element in the range of keys.
/// \param [in] values_input - iterator to the first element in the range of values to reduce.
/// \param [in] size - number of element in the input range.
/// \param [out] unique_output - iterator to the first element in the output range of unique keys.
/// \param [out] aggregates_output - iterator to the first element in the output range of reductions.
/// \param [out] unique_count_output - iterator to total number of groups.
/// \param [in] reduce_op - binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] key_compare_op - binary operation function object that will be used to determine keys equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is KeyCompareFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful reduction; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level sum operation is performed on an array of
/// integer values and integer keys.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// int * keys_input;           // e.g., [1, 1, 1, 2, 10, 10, 10, 88]
/// int * values_input;         // e.g., [1, 2, 3, 4,  5,  6,  7,  8]
/// int * unique_output;        // empty array of at least 4 elements
/// int * aggregates_output;    // empty array of at least 4 elements
/// int * unique_count_output;  // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::reduce_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input, input_size,
///     unique_output, aggregates_output, unique_count_output
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform reduction
/// rocprim::reduce_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input, input_size,
///     unique_output, aggregates_output, unique_count_output
/// );
/// // unique_output:       [1, 2, 10, 88]
/// // aggregates_output:   [6, 4, 18,  8]
/// // unique_count_output: [4]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class ValuesInputIterator,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class UniqueCountOutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
    class KeyCompareFunction = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
hipError_t reduce_by_key(void * temporary_storage,
                         size_t& storage_size,
                         KeysInputIterator keys_input,
                         ValuesInputIterator values_input,
                         unsigned int size,
                         UniqueOutputIterator unique_output,
                         AggregatesOutputIterator aggregates_output,
                         UniqueCountOutputIterator unique_count_output,
                         BinaryFunction reduce_op = BinaryFunction(),
                         KeyCompareFunction key_compare_op = KeyCompareFunction(),
                         hipStream_t stream = 0,
                         bool debug_synchronous = false)
{
    return detail::reduce_by_key_impl<Config>(
        temporary_storage, storage_size,
        keys_input, values_input, size,
        unique_output, aggregates_output, unique_count_output,
        reduce_op, key_compare_op,
        stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_
