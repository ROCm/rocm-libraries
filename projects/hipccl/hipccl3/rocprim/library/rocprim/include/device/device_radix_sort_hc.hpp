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

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"
#include "../detail/radix_sort.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "detail/device_radix_sort.hpp"

/// \addtogroup collectivedevicemodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

size_t align_size(size_t size)
{
    constexpr size_t alignment = 256;
    return ::rocprim::ceiling_div(size, alignment) * alignment;
}

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
void device_radix_sort(void * temporary_storage,
                       size_t& temporary_storage_bytes,
                       const Key * keys_input,
                       Key * keys_output,
                       const Value * values_input,
                       Value * values_output,
                       size_t size,
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

    const unsigned int blocks = ::rocprim::ceiling_div(static_cast<unsigned int>(size), sort_size);
    const unsigned int blocks_per_full_batch = ::rocprim::ceiling_div(blocks, scan_size);
    const unsigned int full_batches = blocks % scan_size != 0
        ? blocks % scan_size
        : scan_size;
    const unsigned int batches = (blocks_per_full_batch == 1 ? full_batches : scan_size);
    const unsigned int iterations = ::rocprim::ceiling_div(end_bit - begin_bit, radix_bits);

    const size_t batch_digit_counts_bytes = align_size(batches * radix_size * sizeof(unsigned int));
    const size_t digit_counts_bytes = align_size(radix_size * sizeof(unsigned int));
    const size_t bit_keys_bytes = align_size(size * sizeof(bit_key_type));
    const size_t values_bytes = with_values ? align_size(size * sizeof(Value)) : 0;
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

template<class Key>
void device_radix_sort_keys(void * temporary_storage,
                            size_t& temporary_storage_bytes,
                            const Key * keys_input,
                            Key * keys_output,
                            size_t size,
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

template<class Key>
void device_radix_sort_keys_desc(void * temporary_storage,
                                 size_t& temporary_storage_bytes,
                                 const Key * keys_input,
                                 Key * keys_output,
                                 size_t size,
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

template<class Key, class Value>
void device_radix_sort_pairs(void * temporary_storage,
                             size_t& temporary_storage_bytes,
                             const Key * keys_input,
                             Key * keys_output,
                             const Value * values_input,
                             Value * values_output,
                             size_t size,
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

template<class Key, class Value>
void device_radix_sort_pairs_desc(void * temporary_storage,
                                  size_t& temporary_storage_bytes,
                                  const Key * keys_input,
                                  Key * keys_output,
                                  const Value * values_input,
                                  Value * values_output,
                                  size_t size,
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
// end of group collectivedevicemodule

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HC_HPP_
