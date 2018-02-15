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

#ifndef ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HC_HPP_

#include <iostream>
#include <type_traits>
#include <utility>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/radix_sort.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "detail/device_radix_sort.hpp"

/// \addtogroup devicemodule_hc
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#define SYNC(name, start) \
    { \
        if(debug_synchronous) \
        { \
            std::cout << name; \
            acc_view.wait(); \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<bool Descending, class Key, class Value>
inline
void device_radix_sort(void * temporary_storage,
                       size_t& temporary_storage_bytes,
                       const Key * keys_input,
                       Key * keys_output,
                       const Value * values_input,
                       Value * values_output,
                       unsigned int size,
                       unsigned int begin_bit,
                       unsigned int end_bit,
                       hc::accelerator_view& acc_view,
                       bool debug_synchronous)
{
    using bit_key_type = typename ::rocprim::detail::radix_key_codec<Key>::bit_key_type;

    constexpr bool with_values = !std::is_same<Value, ::rocprim::empty_type>::value;

    constexpr unsigned int radix_bits = 8;
    constexpr unsigned int radix_size = 1 << radix_bits;

    constexpr unsigned int scan_block_size = 256;
    constexpr unsigned int scan_items_per_thread = 4;

    constexpr unsigned int sort_block_size = 256;
    constexpr unsigned int sort_items_per_thread = 11;

    constexpr unsigned int scan_size = scan_block_size * scan_items_per_thread;
    constexpr unsigned int sort_size = sort_block_size * sort_items_per_thread;

    const unsigned int blocks = ::rocprim::detail::ceiling_div(static_cast<unsigned int>(size), sort_size);
    const unsigned int blocks_per_full_batch = ::rocprim::detail::ceiling_div(blocks, scan_size);
    const unsigned int full_batches = blocks % scan_size != 0
        ? blocks % scan_size
        : scan_size;
    const unsigned int batches = (blocks_per_full_batch == 1 ? full_batches : scan_size);
    const unsigned int iterations = ::rocprim::detail::ceiling_div(end_bit - begin_bit, radix_bits);

    const size_t batch_digit_counts_bytes = ::rocprim::detail::align_size(batches * radix_size * sizeof(unsigned int));
    const size_t digit_counts_bytes = ::rocprim::detail::align_size(radix_size * sizeof(unsigned int));
    const size_t bit_keys_bytes = ::rocprim::detail::align_size(size * sizeof(bit_key_type));
    const size_t values_bytes = with_values ? ::rocprim::detail::align_size(size * sizeof(Value)) : 0;
    if(temporary_storage == nullptr)
    {
        temporary_storage_bytes = batch_digit_counts_bytes + digit_counts_bytes + bit_keys_bytes + values_bytes;
        return;
    }

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        std::cout << "blocks_per_full_batch " << blocks_per_full_batch << '\n';
        std::cout << "full_batches " << full_batches << '\n';
        std::cout << "batches " << batches << '\n';
        std::cout << "iterations " << iterations << '\n';
        acc_view.wait();
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    unsigned int * batch_digit_counts = reinterpret_cast<unsigned int *>(ptr);
    ptr += batch_digit_counts_bytes;
    unsigned int * digit_counts = reinterpret_cast<unsigned int *>(ptr);
    ptr += digit_counts_bytes;
    bit_key_type * bit_keys0 = reinterpret_cast<bit_key_type *>(ptr);
    ptr += bit_keys_bytes;
    Value * values0 = with_values ? reinterpret_cast<Value *>(ptr) : nullptr;

    bit_key_type * bit_keys1 = reinterpret_cast<bit_key_type *>(keys_output);
    Value * values1 = values_output;

    // Result must be always placed in keys_output and values_output
    if(iterations % 2 == 0)
    {
        std::swap(bit_keys0, bit_keys1);
        std::swap(values0, values1);
    }

    for(unsigned int bit = begin_bit; bit < end_bit; bit += radix_bits)
    {
        // Handle cases when (end_bit - bit) is not divisible by radix_bits, i.e. the last
        // iteration has a shorter mask.
        const unsigned int current_radix_bits = ::rocprim::min(radix_bits, end_bit - bit);

        const bool is_first_iteration = (bit == begin_bit);
        const bool is_last_iteration = (bit + current_radix_bits == end_bit);

        std::chrono::high_resolution_clock::time_point start;

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        if(is_first_iteration)
        {
            hc::parallel_for_each(
                acc_view,
                hc::tiled_extent<1>(batches * sort_block_size, sort_block_size),
                [=](hc::tiled_index<1>) [[hc]]
                {
                    detail::fill_digit_counts<
                        sort_block_size, sort_items_per_thread, radix_bits,
                        Descending
                    >(
                        keys_input, size,
                        batch_digit_counts,
                        bit, current_radix_bits,
                        blocks_per_full_batch, full_batches
                    );
                }
            );
        }
        else
        {
            hc::parallel_for_each(
                acc_view,
                hc::tiled_extent<1>(batches * sort_block_size, sort_block_size),
                [=](hc::tiled_index<1>) [[hc]]
                {
                    detail::fill_digit_counts<
                        sort_block_size, sort_items_per_thread, radix_bits,
                        false
                    >(
                        bit_keys0, size,
                        batch_digit_counts,
                        bit, current_radix_bits,
                        blocks_per_full_batch, full_batches
                    );
                }
            );
        }
        SYNC("fill_digit_counts", start)

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(radix_size * scan_block_size, scan_block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                detail::scan_batches<scan_block_size, scan_items_per_thread, radix_bits>(
                    batch_digit_counts, digit_counts, batches
                );
            }
        );
        SYNC("scan_batches", start)

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(radix_size, radix_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                detail::scan_digits<radix_bits>(digit_counts);
            }
        );
        SYNC("scan_digits", start)

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        if(is_first_iteration && is_last_iteration)
        {
            hc::parallel_for_each(
                acc_view,
                hc::tiled_extent<1>(batches * sort_block_size, sort_block_size),
                [=](hc::tiled_index<1>) [[hc]]
                {
                    detail::sort_and_scatter<
                        sort_block_size, sort_items_per_thread, radix_bits,
                        Descending, Descending
                    >(
                        keys_input, keys_output, values_input, values_output, size,
                        batch_digit_counts, digit_counts,
                        bit, current_radix_bits,
                        blocks_per_full_batch, full_batches
                    );
                }
            );
        }
        else if(is_first_iteration)
        {
            hc::parallel_for_each(
                acc_view,
                hc::tiled_extent<1>(batches * sort_block_size, sort_block_size),
                [=](hc::tiled_index<1>) [[hc]]
                {
                    detail::sort_and_scatter<
                        sort_block_size, sort_items_per_thread, radix_bits,
                        Descending, false
                    >(
                        keys_input, bit_keys1, values_input, values1, size,
                        batch_digit_counts, digit_counts,
                        bit, current_radix_bits,
                        blocks_per_full_batch, full_batches
                    );
                }
            );
        }
        else if(is_last_iteration)
        {
            hc::parallel_for_each(
                acc_view,
                hc::tiled_extent<1>(batches * sort_block_size, sort_block_size),
                [=](hc::tiled_index<1>) [[hc]]
                {
                    detail::sort_and_scatter<
                        sort_block_size, sort_items_per_thread, radix_bits,
                        false, Descending
                    >(
                        bit_keys0, keys_output, values0, values_output, size,
                        batch_digit_counts, digit_counts,
                        bit, current_radix_bits,
                        blocks_per_full_batch, full_batches
                    );
                }
            );
        }
        else
        {
            hc::parallel_for_each(
                acc_view,
                hc::tiled_extent<1>(batches * sort_block_size, sort_block_size),
                [=](hc::tiled_index<1>) [[hc]]
                {
                    detail::sort_and_scatter<
                        sort_block_size, sort_items_per_thread, radix_bits,
                        false, false
                    >(
                        bit_keys0, bit_keys1, values0, values1, size,
                        batch_digit_counts, digit_counts,
                        bit, current_radix_bits,
                        blocks_per_full_batch, full_batches
                    );
                }
            );
        }
        SYNC("sort_and_scatter", start)

        std::swap(bit_keys0, bit_keys1);
        std::swap(values0, values1);
    }
}

#undef SYNC

} // end namespace detail

/// \brief HC parallel ascending radix sort primitive for device level.
///
/// \p device_radix_sort_keys function performs a device-wide radix sort
/// of keys. Function sorts input keys in ascending order.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p temporary_storage_bytes
/// if \p temporary_storage in a null pointer.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 5</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Key - key type. Must be an integral type or a floating-point type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p temporary_storage_bytes and function returns without performing the sort operation.
/// \param [in,out] temporary_storage_bytes - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed on an array of
/// \p float values.
///
/// \code{.cpp}
/// #include <rocprim.hpp>
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;        // e.g., 8
/// hc::array<float> input;   // e.g., [0.6, 0.3, 0.65, 0.4, 0.2, 0.08, 1, 0.7]
/// hc::array<float> output;  // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::device_radix_sort_keys(
///     nullptr, temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(),
///     input_size
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform sort
/// rocprim::device_radix_sort_keys(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(),
///     input_size, 0, 4 * sizeof(float), acc_view
/// );
/// // keys_output: [0.08, 0.2, 0.3, 0.4, 0.6, 0.65, 0.7, 1]
/// \endcode
/// \endparblock
template<class Key>
inline
void device_radix_sort_keys(void * temporary_storage,
                            size_t& temporary_storage_bytes,
                            const Key * keys_input,
                            Key * keys_output,
                            unsigned int size,
                            unsigned int begin_bit = 0,
                            unsigned int end_bit = 8 * sizeof(Key),
                            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                            bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    detail::device_radix_sort<false>(
        temporary_storage, temporary_storage_bytes,
        keys_input, keys_output, values, values, size,
        begin_bit, end_bit,
        acc_view, debug_synchronous
    );
}

/// \brief HC parallel descending radix sort primitive for device level.
///
/// \p device_radix_sort_keys_desc function performs a device-wide radix sort
/// of keys. Function sorts input keys in descending order.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p temporary_storage_bytes
/// if \p temporary_storage in a null pointer.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 5</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Key - key type. Must be an integral type or a floating-point type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p temporary_storage_bytes and function returns without performing the sort operation.
/// \param [in,out] temporary_storage_bytes - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>.
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim.hpp>
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;        // e.g., 8
/// hc::array<int> input;     // e.g., [6, 3, 5, 4, 2, 8, 1, 7]
/// hc::array<int> output;    // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::device_radix_sort_keys_desc(
///     nullptr, temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(),
///     input_size
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform sort
/// rocprim::device_radix_sort_keys_desc(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     input.accelerator_pointer(), output.accelerator_pointer(),
///     input_size, 0, 4 * sizeof(int), acc_view
/// );
/// // keys_output: [8, 7, 6, 5, 4, 3, 2, 1]
/// \endcode
/// \endparblock
template<class Key>
inline
void device_radix_sort_keys_desc(void * temporary_storage,
                                 size_t& temporary_storage_bytes,
                                 const Key * keys_input,
                                 Key * keys_output,
                                 unsigned int size,
                                 unsigned int begin_bit = 0,
                                 unsigned int end_bit = 8 * sizeof(Key),
                                 hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                                 bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    detail::device_radix_sort<true>(
        temporary_storage, temporary_storage_bytes,
        keys_input, keys_output, values, values, size,
        begin_bit, end_bit,
        acc_view, debug_synchronous
    );
}

/// \brief HC parallel ascending radix sort-by-key primitive for device level.
///
/// \p device_radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in ascending order of keys.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p temporary_storage_bytes
/// if \p temporary_storage in a null pointer.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 5</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Key - key type. Must be an integral type or a floating-point type.
/// \tparam Value - value type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p temporary_storage_bytes and function returns without performing the sort operation.
/// \param [in,out] temporary_storage_bytes - reference to a size (in bytes) of \p temporary_storage.
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
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim.hpp>
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;                   // e.g., 8
/// hc::array<unsigned int> keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// hc::array<double> values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// hc::array<unsigned int> keys_output; // empty array of 8 elements
/// hc::array<double> values_output;     // empty array of 8 elements
///
/// // Keys are in range [0; 8], so we can limit compared bit to bits on indexes
/// // 0, 1, 2, 3, and 4. In order to do this \p begin_bit is set to 0 and \p end_bit
/// // is set to 5.
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::device_radix_sort_pairs(
///     nullptr, temporary_storage_size_bytes,
///     keys_input.accelerator_pointer(), keys_output.accelerator_pointer(),
///     values_input.accelerator_pointer(), values_output.accelerator_pointer(),
///     input_size, 0, 5, acc_view
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform sort
/// rocprim::device_radix_sort_pairs(
///     temporary_storage.accelerator_pointer(), temporary_storage_size_bytes,
///     keys_input.accelerator_pointer(), keys_output.accelerator_pointer(),
///     values_input.accelerator_pointer(), values_output.accelerator_pointer(),
///     input_size, 0, 5, acc_view
/// );
/// // keys_output:   [ 1,  1, 3, 4,  5,  6, 7,  8]
/// // values_output: [-1, -2, 2, 3, -4, -5, 7, -8]
/// \endcode
/// \endparblock
template<class Key, class Value>
inline
void device_radix_sort_pairs(void * temporary_storage,
                             size_t& temporary_storage_bytes,
                             const Key * keys_input,
                             Key * keys_output,
                             const Value * values_input,
                             Value * values_output,
                             unsigned int size,
                             unsigned int begin_bit = 0,
                             unsigned int end_bit = 8 * sizeof(Key),
                             hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                             bool debug_synchronous = false)
{
    detail::device_radix_sort<false>(
        temporary_storage, temporary_storage_bytes,
        keys_input, keys_output, values_input, values_output, size,
        begin_bit, end_bit,
        acc_view, debug_synchronous
    );
}

/// \brief HC parallel descending radix sort-by-key primitive for device level.
///
/// \p device_radix_sort_pairs_desc function performs a device-wide radix sort
/// of (key, value) pairs. Function sorts input pairs in descending order of keys.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p temporary_storage_bytes
/// if \p temporary_storage in a null pointer.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 5</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \tparam Key - key type. Must be an integral type or a floating-point type.
/// \tparam Value - value type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p temporary_storage_bytes and function returns without performing the sort operation.
/// \param [in,out] temporary_storage_bytes - reference to a size (in bytes) of \p temporary_storage.
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
/// \param [in] acc_view - [optional] \p hc::accelerator_view object. The default value
/// is \p hc::accelerator().get_default_view() (default view of the default accelerator).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed where input keys are
/// represented by an array of integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim.hpp>
///
/// hc::accelerator_view acc_view = ...;
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;                // e.g., 8
/// hc::array<int> keys_input;        // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// hc::array<double> values_input;   // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// hc::array<int> keys_output;       // empty array of 8 elements
/// hc::array<double> values_output;  // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// // Get required size of the temporary storage
/// rocprim::device_radix_sort_pairs_desc(
///     nullptr, temporary_storage_size_bytes,
///     keys_input.accelerator_pointer(), keys_output.accelerator_pointer(),
///     values_input.accelerator_pointer(), values_output.accelerator_pointer(),
///     input_size, 0, 4 * sizeof(int), acc_view
/// );
///
/// // allocate temporary storage
/// hc::array<char> temporary_storage(temporary_storage_size_bytes, acc_view);
///
/// // perform sort
/// rocprim::device_radix_sort_pairs_desc(
///     temporary_storage, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size, 0, 4 * sizeof(int), acc_view
/// );
/// // keys_output:   [ 8, 7,  6,  5, 4, 3,  1,  1]
/// // values_output: [-8, 7, -5, -4, 3, 2, -1, -2]
/// \endcode
/// \endparblock
template<class Key, class Value>
inline
void device_radix_sort_pairs_desc(void * temporary_storage,
                                  size_t& temporary_storage_bytes,
                                  const Key * keys_input,
                                  Key * keys_output,
                                  const Value * values_input,
                                  Value * values_output,
                                  unsigned int size,
                                  unsigned int begin_bit = 0,
                                  unsigned int end_bit = 8 * sizeof(Key),
                                  hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                                  bool debug_synchronous = false)
{
    detail::device_radix_sort<true>(
        temporary_storage, temporary_storage_bytes,
        keys_input, keys_output, values_input, values_output, size,
        begin_bit, end_bit,
        acc_view, debug_synchronous
    );
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule_hc

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HC_HPP_
