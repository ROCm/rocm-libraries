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

#ifndef ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_

#include <iostream>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/radix_sort.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "device_radix_sort_config.hpp"
#include "detail/device_radix_sort.hpp"

/// \addtogroup devicemodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator
>
__global__
void fill_digit_counts_kernel(KeysInputIterator keys_input,
                              unsigned int size,
                              unsigned int * batch_digit_counts,
                              unsigned int bit,
                              unsigned int current_radix_bits,
                              unsigned int blocks_per_full_batch,
                              unsigned int full_batches)
{
    fill_digit_counts<BlockSize, ItemsPerThread, RadixBits, Descending>(
        keys_input, size,
        batch_digit_counts,
        bit, current_radix_bits,
        blocks_per_full_batch, full_batches
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits
>
__global__
void scan_batches_kernel(unsigned int * batch_digit_counts,
                         unsigned int * digit_counts,
                         unsigned int batches)
{
    scan_batches<BlockSize, ItemsPerThread, RadixBits>(batch_digit_counts, digit_counts, batches);
}

template<unsigned int RadixBits>
__global__
void scan_digits_kernel(unsigned int * digit_counts)
{
    scan_digits<RadixBits>(digit_counts);
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator
>
__global__
void sort_and_scatter_kernel(KeysInputIterator keys_input,
                             KeysOutputIterator keys_output,
                             ValuesInputIterator values_input,
                             ValuesOutputIterator values_output,
                             unsigned int size,
                             const unsigned int * batch_digit_starts,
                             const unsigned int * digit_starts,
                             unsigned int bit,
                             unsigned int current_radix_bits,
                             unsigned int blocks_per_full_batch,
                             unsigned int full_batches)
{
    sort_and_scatter<BlockSize, ItemsPerThread, RadixBits, Descending>(
        keys_input, keys_output, values_input, values_output, size,
        batch_digit_starts, digit_starts,
        bit, current_radix_bits,
        blocks_per_full_batch, full_batches
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
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator
>
inline
hipError_t radix_sort_iteration(KeysInputIterator keys_input,
                                typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                                KeysOutputIterator keys_output,
                                ValuesInputIterator values_input,
                                typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                                ValuesOutputIterator values_output,
                                unsigned int size,
                                unsigned int * batch_digit_counts,
                                unsigned int * digit_counts,
                                bool from_input,
                                bool to_output,
                                unsigned int bit,
                                unsigned int end_bit,
                                unsigned int blocks_per_full_batch,
                                unsigned int full_batches,
                                unsigned int batches,
                                hipStream_t stream,
                                bool debug_synchronous)
{
    constexpr unsigned int radix_size = 1 << RadixBits;

    // Handle cases when (end_bit - bit) is not divisible by RadixBits, i.e. the last
    // iteration has a shorter mask.
    const unsigned int current_radix_bits = ::rocprim::min(RadixBits, end_bit - bit);

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous)
    {
        std::cout << "RadixBits " << RadixBits << '\n';
        std::cout << "bit " << bit << '\n';
        std::cout << "current_radix_bits " << current_radix_bits << '\n';
    }

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    if(from_input)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fill_digit_counts_kernel<
                Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
            >),
            dim3(batches), dim3(Config::sort::block_size), 0, stream,
            keys_input, size,
            batch_digit_counts,
            bit, current_radix_bits,
            blocks_per_full_batch, full_batches
        );
    }
    else
    {
        if(to_output)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(fill_digit_counts_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >),
                dim3(batches), dim3(Config::sort::block_size), 0, stream,
                keys_tmp, size,
                batch_digit_counts,
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(fill_digit_counts_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >),
                dim3(batches), dim3(Config::sort::block_size), 0, stream,
                keys_output, size,
                batch_digit_counts,
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
    }
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("fill_digit_counts", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scan_batches_kernel<Config::scan::block_size, Config::scan::items_per_thread, RadixBits>),
        dim3(radix_size), dim3(Config::scan::block_size), 0, stream,
        batch_digit_counts, digit_counts, batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_batches", radix_size * Config::scan::block_size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scan_digits_kernel<RadixBits>),
        dim3(1), dim3(radix_size), 0, stream,
        digit_counts
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_digits", radix_size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    if(from_input)
    {
        if(to_output)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(sort_and_scatter_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >),
                dim3(batches), dim3(Config::sort::block_size), 0, stream,
                keys_input, keys_output, values_input, values_output, size,
                const_cast<const unsigned int *>(batch_digit_counts),
                const_cast<const unsigned int *>(digit_counts),
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(sort_and_scatter_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >),
                dim3(batches), dim3(Config::sort::block_size), 0, stream,
                keys_input, keys_tmp, values_input, values_tmp, size,
                const_cast<const unsigned int *>(batch_digit_counts),
                const_cast<const unsigned int *>(digit_counts),
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
    }
    else
    {
        if(to_output)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(sort_and_scatter_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >),
                dim3(batches), dim3(Config::sort::block_size), 0, stream,
                keys_tmp, keys_output, values_tmp, values_output, size,
                const_cast<const unsigned int *>(batch_digit_counts),
                const_cast<const unsigned int *>(digit_counts),
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(sort_and_scatter_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >),
                dim3(batches), dim3(Config::sort::block_size), 0, stream,
                keys_output, keys_tmp, values_output, values_tmp, size,
                const_cast<const unsigned int *>(batch_digit_counts),
                const_cast<const unsigned int *>(digit_counts),
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
    }
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("sort_and_scatter", size, start)

    return hipSuccess;
}

template<
    class Config,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator
>
inline
hipError_t radix_sort_impl(void * temporary_storage,
                           size_t& storage_size,
                           KeysInputIterator keys_input,
                           typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           bool& is_result_in_output,
                           unsigned int begin_bit,
                           unsigned int end_bit,
                           hipStream_t stream,
                           bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    static_assert(
        std::is_same<key_type, typename std::iterator_traits<KeysOutputIterator>::value_type>::value,
        "KeysInputIterator and KeysOutputIterator must have the same value_type"
    );
    static_assert(
        std::is_same<value_type, typename std::iterator_traits<ValuesOutputIterator>::value_type>::value,
        "ValuesInputIterator and ValuesOutputIterator must have the same value_type"
    );

    using config = default_or_custom_config<
        Config,
        default_radix_sort_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    constexpr unsigned int max_radix_size = 1 << config::long_radix_bits;

    constexpr unsigned int scan_size = config::scan::block_size * config::scan::items_per_thread;
    constexpr unsigned int sort_size = config::sort::block_size * config::sort::items_per_thread;

    const unsigned int blocks = std::max(1u, ::rocprim::detail::ceiling_div(size, sort_size));
    const unsigned int blocks_per_full_batch = ::rocprim::detail::ceiling_div(blocks, scan_size);
    const unsigned int full_batches = blocks % scan_size != 0
        ? blocks % scan_size
        : scan_size;
    const unsigned int batches = (blocks_per_full_batch == 1 ? full_batches : scan_size);
    const bool with_double_buffer = keys_tmp != nullptr;

    const unsigned int bits = end_bit - begin_bit;
    const unsigned int iterations = ::rocprim::detail::ceiling_div(bits, config::long_radix_bits);
    const unsigned int radix_bits_diff = config::long_radix_bits - config::short_radix_bits;
    const unsigned int short_iterations = radix_bits_diff != 0
        ? ::rocprim::min(iterations, (config::long_radix_bits * iterations - bits) / radix_bits_diff)
        : 0;
    const unsigned int long_iterations = iterations - short_iterations;

    const size_t batch_digit_counts_bytes =
        ::rocprim::detail::align_size(batches * max_radix_size * sizeof(unsigned int));
    const size_t digit_counts_bytes = ::rocprim::detail::align_size(max_radix_size * sizeof(unsigned int));
    const size_t keys_bytes = ::rocprim::detail::align_size(size * sizeof(key_type));
    const size_t values_bytes = with_values ? ::rocprim::detail::align_size(size * sizeof(value_type)) : 0;
    if(temporary_storage == nullptr)
    {
        storage_size = batch_digit_counts_bytes + digit_counts_bytes;
        if(!with_double_buffer)
        {
            storage_size += keys_bytes + values_bytes;
        }
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        std::cout << "blocks_per_full_batch " << blocks_per_full_batch << '\n';
        std::cout << "full_batches " << full_batches << '\n';
        std::cout << "batches " << batches << '\n';
        std::cout << "iterations " << iterations << '\n';
        std::cout << "long_iterations " << long_iterations << '\n';
        std::cout << "short_iterations " << short_iterations << '\n';
        hipError_t error = hipStreamSynchronize(stream);
        if(error != hipSuccess) return error;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    unsigned int * batch_digit_counts = reinterpret_cast<unsigned int *>(ptr);
    ptr += batch_digit_counts_bytes;
    unsigned int * digit_counts = reinterpret_cast<unsigned int *>(ptr);
    ptr += digit_counts_bytes;
   if(!with_double_buffer)
    {
        keys_tmp = reinterpret_cast<key_type *>(ptr);
        ptr += keys_bytes;
        values_tmp = with_values ? reinterpret_cast<value_type *>(ptr) : nullptr;
    }

    bool to_output = with_double_buffer || (iterations - 1) % 2 == 0;
    bool from_input = true;
    if(!with_double_buffer && to_output)
    {
        // Copy input keys and values if necessary (in-place sorting: input and output iterators are equal)
        const bool keys_equal = ::rocprim::detail::are_iterators_equal(keys_input, keys_output);
        const bool values_equal = with_values && ::rocprim::detail::are_iterators_equal(values_input, values_output);
        if(keys_equal || values_equal)
        {
            hipError_t error = ::rocprim::transform(
                keys_input, keys_tmp, size,
                ::rocprim::identity<key_type>(), stream, debug_synchronous
            );
            if(error != hipSuccess) return error;

            if(with_values)
            {
                hipError_t error = ::rocprim::transform(
                    values_input, values_tmp, size,
                    ::rocprim::identity<value_type>(), stream, debug_synchronous
                );
                if(error != hipSuccess) return error;
            }

            from_input = false;
        }
    }

    unsigned int bit = begin_bit;
    for(unsigned int i = 0; i < long_iterations; i++)
    {
        hipError_t error = radix_sort_iteration<config, config::long_radix_bits, Descending>(
            keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output, size,
            batch_digit_counts, digit_counts,
            from_input, to_output,
            bit, end_bit,
            blocks_per_full_batch, full_batches, batches,
            stream, debug_synchronous
        );
        if(error != hipSuccess) return error;

        is_result_in_output = to_output;
        from_input = false;
        to_output = !to_output;
        bit += config::long_radix_bits;
    }
    for(unsigned int i = 0; i < short_iterations; i++)
    {
        hipError_t error = radix_sort_iteration<config, config::short_radix_bits, Descending>(
            keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output, size,
            batch_digit_counts, digit_counts,
            from_input, to_output,
            bit, end_bit,
            blocks_per_full_batch, full_batches, batches,
            stream, debug_synchronous
        );
        if(error != hipSuccess) return error;

        is_result_in_output = to_output;
        from_input = false;
        to_output = !to_output;
        bit += config::short_radix_bits;
    }

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end namespace detail

/// \brief Parallel ascending radix sort primitive for device level.
///
/// \p radix_sort_keys function performs a device-wide radix sort
/// of keys. Function sorts input keys in ascending order.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input and \p keys_output must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed on an array of
/// \p float values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;      // e.g., 8
/// float * input;          // e.g., [0.6, 0.3, 0.65, 0.4, 0.2, 0.08, 1, 0.7]
/// float * output;         // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
/// // keys_output: [0.08, 0.2, 0.3, 0.4, 0.6, 0.65, 0.7, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t radix_sort_keys(void * temporary_storage,
                           size_t& storage_size,
                           KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           unsigned int size,
                           unsigned int begin_bit = 0,
                           unsigned int end_bit = 8 * sizeof(Key),
                           hipStream_t stream = 0,
                           bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool ignored;
    return detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

/// \brief Parallel descending radix sort primitive for device level.
///
/// \p radix_sort_keys_desc function performs a device-wide radix sort
/// of keys. Function sorts input keys in descending order.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input and \p keys_output must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;    // e.g., 8
/// int * input;          // e.g., [6, 3, 5, 4, 2, 8, 1, 7]
/// int * output;         // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
/// // keys_output: [8, 7, 6, 5, 4, 3, 2, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t radix_sort_keys_desc(void * temporary_storage,
                                size_t& storage_size,
                                KeysInputIterator keys_input,
                                KeysOutputIterator keys_output,
                                unsigned int size,
                                unsigned int begin_bit = 0,
                                unsigned int end_bit = 8 * sizeof(Key),
                                hipStream_t stream = 0,
                                bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool ignored;
    return detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

/// \brief Parallel ascending radix sort-by-key primitive for device level.
///
/// \p radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in ascending order of keys.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] values_input - pointer to the first element in the range to sort.
/// \param [out] values_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// unsigned int * keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// unsigned int * keys_output; // empty array of 8 elements
/// double * values_output;     // empty array of 8 elements
///
/// // Keys are in range [0; 8], so we can limit compared bit to bits on indexes
/// // 0, 1, 2, 3, and 4. In order to do this begin_bit is set to 0 and end_bit
/// // is set to 5.
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size, 0, 5
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size, 0, 5
/// );
/// // keys_output:   [ 1,  1, 3, 4,  5,  6, 7,  8]
/// // values_output: [-1, -2, 2, 3, -4, -5, 7, -8]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t radix_sort_pairs(void * temporary_storage,
                            size_t& storage_size,
                            KeysInputIterator keys_input,
                            KeysOutputIterator keys_output,
                            ValuesInputIterator values_input,
                            ValuesOutputIterator values_output,
                            unsigned int size,
                            unsigned int begin_bit = 0,
                            unsigned int end_bit = 8 * sizeof(Key),
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
{
    bool ignored;
    return detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

/// \brief Parallel descending radix sort-by-key primitive for device level.
///
/// \p radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in descending order of keys.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] values_input - pointer to the first element in the range to sort.
/// \param [out] values_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed where input keys are
/// represented by an array of integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;       // e.g., 8
/// int * keys_input;        // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;   // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// int * keys_output;       // empty array of 8 elements
/// double * values_output;  // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size
/// );
/// // keys_output:   [ 8, 7,  6,  5, 4, 3,  1,  1]
/// // values_output: [-8, 7, -5, -4, 3, 2, -1, -2]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t radix_sort_pairs_desc(void * temporary_storage,
                                 size_t& storage_size,
                                 KeysInputIterator keys_input,
                                 KeysOutputIterator keys_output,
                                 ValuesInputIterator values_input,
                                 ValuesOutputIterator values_output,
                                 unsigned int size,
                                 unsigned int begin_bit = 0,
                                 unsigned int end_bit = 8 * sizeof(Key),
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    bool ignored;
    return detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

/// \brief Parallel ascending radix sort primitive for device level.
///
/// \p radix_sort_keys function performs a device-wide radix sort
/// of keys. Function sorts input keys in ascending order.
///
/// \par Overview
/// * The contents of both buffers of \p keys may be altered by the sorting function.
/// * \p current() of \p keys is used as the input.
/// * The function will update \p current() of \p keys to point to the buffer
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam Key - key type. Must be an integral type or a floating-point type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys - reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed on an array of
/// \p float values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;  // e.g., 8
/// float * input;      // e.g., [0.6, 0.3, 0.65, 0.4, 0.2, 0.08, 1, 0.7]
/// float * tmp;        // empty array of 8 elements
/// // Create double-buffer
/// rocprim::double_buffer<float> keys(input, tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size
/// );
/// // keys.current(): [0.08, 0.2, 0.3, 0.4, 0.6, 0.65, 0.7, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class Key
>
inline
hipError_t radix_sort_keys(void * temporary_storage,
                           size_t& storage_size,
                           double_buffer<Key>& keys,
                           unsigned int size,
                           unsigned int begin_bit = 0,
                           unsigned int end_bit = 8 * sizeof(Key),
                           hipStream_t stream = 0,
                           bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool is_result_in_output;
    hipError_t error = detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

/// \brief Parallel descending radix sort primitive for device level.
///
/// \p radix_sort_keys_desc function performs a device-wide radix sort
/// of keys. Function sorts input keys in descending order.
///
/// \par Overview
/// * The contents of both buffers of \p keys may be altered by the sorting function.
/// * \p current() of \p keys is used as the input.
/// * The function will update \p current() of \p keys to point to the buffer
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam Key - key type. Must be an integral type or a floating-point type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys - reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;  // e.g., 8
/// int * input;        // e.g., [6, 3, 5, 4, 2, 8, 1, 7]
/// int * tmp;          // empty array of 8 elements
/// // Create double-buffer
/// rocprim::double_buffer<int> keys(input, tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size
/// );
/// // keys.current(): [8, 7, 6, 5, 4, 3, 2, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class Key
>
inline
hipError_t radix_sort_keys_desc(void * temporary_storage,
                                size_t& storage_size,
                                double_buffer<Key>& keys,
                                unsigned int size,
                                unsigned int begin_bit = 0,
                                unsigned int end_bit = 8 * sizeof(Key),
                                hipStream_t stream = 0,
                                bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool is_result_in_output;
    hipError_t error = detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

/// \brief Parallel ascending radix sort-by-key primitive for device level.
///
/// \p radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in ascending order of keys.
///
/// \par Overview
/// * The contents of both buffers of \p keys and \p values may be altered by the sorting function.
/// * \p current() of \p keys and \p values are used as the input.
/// * The function will update \p current() of \p keys and \p values to point to buffers
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam Key - key type. Must be an integral type or a floating-point type.
/// \tparam Value - value type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys - reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in,out] values - reference to the double-buffer of values, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// unsigned int * keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// unsigned int * keys_tmp;    // empty array of 8 elements
/// double*  values_tmp;        // empty array of 8 elements
/// // Create double-buffers
/// rocprim::double_buffer<unsigned int> keys(keys_input, keys_tmp);
/// rocprim::double_buffer<double> values(values_input, values_tmp);
///
/// // Keys are in range [0; 8], so we can limit compared bit to bits on indexes
/// // 0, 1, 2, 3, and 4. In order to do this begin_bit is set to 0 and end_bit
/// // is set to 5.
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size,
///     0, 5
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size,
///     0, 5
/// );
/// // keys.current():   [ 1,  1, 3, 4,  5,  6, 7,  8]
/// // values.current(): [-1, -2, 2, 3, -4, -5, 7, -8]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class Key,
    class Value
>
inline
hipError_t radix_sort_pairs(void * temporary_storage,
                            size_t& storage_size,
                            double_buffer<Key>& keys,
                            double_buffer<Value>& values,
                            unsigned int size,
                            unsigned int begin_bit = 0,
                            unsigned int end_bit = 8 * sizeof(Key),
                            hipStream_t stream = 0,
                            bool debug_synchronous = false)
{
    bool is_result_in_output;
    hipError_t error = detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

/// \brief Parallel descending radix sort-by-key primitive for device level.
///
/// \p radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in descending order of keys.
///
/// \par Overview
/// * The contents of both buffers of \p keys and \p values may be altered by the sorting function.
/// * \p current() of \p keys and \p values are used as the input.
/// * The function will update \p current() of \p keys and \p values to point to buffers
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p radix_sort_config or
/// a custom class with the same members.
/// \tparam Key - key type. Must be an integral type or a floating-point type.
/// \tparam Value - value type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys - reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in,out] values - reference to the double-buffer of values, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed where input keys are
/// represented by an array of integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;       // e.g., 8
/// int * keys_input;        // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;   // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// int * keys_tmp;          // empty array of 8 elements
/// double * values_tmp;     // empty array of 8 elements
/// // Create double-buffers
/// rocprim::double_buffer<int> keys(keys_input, keys_tmp);
/// rocprim::double_buffer<double> values(values_input, values_tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size
/// );
/// // keys.current():   [ 8, 7,  6,  5, 4, 3,  1,  1]
/// // values.current(): [-8, 7, -5, -4, 3, 2, -1, -2]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class Key,
    class Value
>
inline
hipError_t radix_sort_pairs_desc(void * temporary_storage,
                                 size_t& storage_size,
                                 double_buffer<Key>& keys,
                                 double_buffer<Value>& values,
                                 unsigned int size,
                                 unsigned int begin_bit = 0,
                                 unsigned int end_bit = 8 * sizeof(Key),
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    bool is_result_in_output;
    hipError_t error = detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_
