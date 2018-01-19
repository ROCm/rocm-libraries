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
#include <type_traits>
#include <utility>

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

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

template<unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int RadixBits, bool DescendingIn, class KeyIn>
__global__
void fill_digit_counts_kernel(const KeyIn * keys_input,
                              unsigned int size,
                              unsigned int * batch_digit_counts,
                              unsigned int bit, unsigned int current_radix_bits,
                              unsigned int div_blocks_per_batch, unsigned int rem_blocks_per_batch)
{
    fill_digit_counts<BlockSize, ItemsPerThread, RadixBits, DescendingIn>(
        keys_input, size,
        batch_digit_counts,
        bit, current_radix_bits,
        div_blocks_per_batch, rem_blocks_per_batch
    );
}

template<unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int RadixBits>
__global__
void scan_batches_kernel(unsigned int * batch_digit_counts, unsigned int * digit_counts, unsigned int batches)
{
    scan_batches<BlockSize, ItemsPerThread, RadixBits>(batch_digit_counts, digit_counts, batches);
}

template<unsigned int RadixBits>
__global__
void scan_digits_kernel(unsigned int * digit_counts)
{
    scan_digits<RadixBits>(digit_counts);
}

template<unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int RadixBits, bool DescendingIn, bool DescendingOut, class KeyIn, class KeyOut>
__global__
void sort_and_scatter_kernel(const KeyIn * keys_input,
                             KeyOut * keys_output,
                             unsigned int size,
                             const unsigned int * batch_digit_counts, const unsigned int * digit_counts,
                             unsigned int bit, unsigned int current_radix_bits,
                             unsigned int div_blocks_per_batch, unsigned int rem_blocks_per_batch)
{
    sort_and_scatter<BlockSize, ItemsPerThread, RadixBits, DescendingIn, DescendingOut>(
        keys_input, keys_output, size,
        batch_digit_counts, digit_counts,
        bit, current_radix_bits,
        div_blocks_per_batch, rem_blocks_per_batch
    );
}

template<bool Descending, class Key, class SortedValue>
hipError_t device_radix_sort(void * temporary_storage,
                             size_t& temporary_storage_bytes,
                             const Key * keys_input,
                             Key * keys_output,
                             const SortedValue * values_input,
                             SortedValue * values_output,
                             size_t size,
                             unsigned int begin_bit,
                             unsigned int end_bit,
                             hipStream_t stream,
                             bool debug_synchronous)
{
    using bit_key_type = typename ::rocprim::detail::radix_key_codec<Key>::bit_key_type;

    constexpr unsigned int radix_bits = 8;
    constexpr unsigned int radix_size = 1 << radix_bits;

    constexpr unsigned int scan_block_size = 256;
    constexpr unsigned int scan_items_per_thread = 4;

    constexpr unsigned int sort_block_size = 256;
    constexpr unsigned int sort_items_per_thread = 11;

    constexpr unsigned int scan_size = scan_block_size * scan_items_per_thread;
    constexpr unsigned int sort_size = sort_block_size * sort_items_per_thread;

    const unsigned int blocks = ::rocprim::ceiling_div(static_cast<unsigned int>(size), sort_size);
    const unsigned int div_blocks_per_batch = blocks / scan_size;
    const unsigned int rem_blocks_per_batch = blocks % scan_size;
    const unsigned int batches = (div_blocks_per_batch == 0 ? rem_blocks_per_batch : scan_size);
    const unsigned int iterations = ::rocprim::ceiling_div(end_bit - begin_bit, radix_bits);

    const size_t batch_digit_counts_bytes = scan_size * radix_size * sizeof(unsigned int);
    const size_t digit_counts_bytes = radix_size * sizeof(unsigned int);
    const size_t bit_keys_bytes = size * sizeof(bit_key_type);
    if(temporary_storage == nullptr)
    {
        temporary_storage_bytes = batch_digit_counts_bytes + digit_counts_bytes + bit_keys_bytes;
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "iterations " << iterations << '\n';
        std::cout << "blocks " << blocks << '\n';
        std::cout << "div_blocks_per_batch " << div_blocks_per_batch << '\n';
        std::cout << "rem_blocks_per_batch " << rem_blocks_per_batch << '\n';
    }

    hipError_t error = hipSuccess;

    unsigned int * batch_digit_counts = reinterpret_cast<unsigned int *>(temporary_storage);
    unsigned int * digit_counts = reinterpret_cast<unsigned int *>(
        reinterpret_cast<char *>(batch_digit_counts) + batch_digit_counts_bytes
    );
    bit_key_type * bit_keys0 = reinterpret_cast<bit_key_type *>(
        reinterpret_cast<char *>(digit_counts) + digit_counts_bytes
    );
    bit_key_type * bit_keys1 = reinterpret_cast<bit_key_type *>(keys_output);

    // Result must be placed in keys_output
    if(iterations % 2 == 0)
    {
        std::swap(bit_keys0, bit_keys1);
    }

    for(unsigned int bit = begin_bit; bit < end_bit; bit += radix_bits)
    {
        // Handle cases when (end_bit - bit) is not divisible by radix_bits, i.e. the last
        // iteration has a shorter mask.
        const unsigned int current_radix_bits = ::rocprim::min(radix_bits, end_bit - bit);

        const bool need_encoding = (bit == begin_bit);
        const bool need_decoding = (bit + current_radix_bits == end_bit);

        auto start = std::chrono::high_resolution_clock::now();
        if(need_encoding)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(detail::fill_digit_counts_kernel<sort_block_size, sort_items_per_thread, radix_bits, Descending>),
                dim3(batches), dim3(sort_block_size), 0, stream,
                keys_input, size,
                batch_digit_counts,
                bit, current_radix_bits,
                div_blocks_per_batch, rem_blocks_per_batch
            );
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(detail::fill_digit_counts_kernel<sort_block_size, sort_items_per_thread, radix_bits, false>),
                dim3(batches), dim3(sort_block_size), 0, stream,
                bit_keys0, size,
                batch_digit_counts,
                bit, current_radix_bits,
                div_blocks_per_batch, rem_blocks_per_batch
            );
        }
        if(debug_synchronous)
        {
            std::cout << "fill_digit_counts";
            error = hipStreamSynchronize(stream);
            if(error != hipSuccess) return error;
            auto end = std::chrono::high_resolution_clock::now();
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            std::cout << " " << d.count() * 1000 << "ms" << '\n';
        }

        start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::scan_batches_kernel<scan_block_size, scan_items_per_thread, radix_bits>),
            dim3(radix_size), dim3(scan_block_size), 0, stream,
            batch_digit_counts, digit_counts, batches
        );
        if(debug_synchronous)
        {
            std::cout << "scan_batches";
            error = hipStreamSynchronize(stream);
            if(error != hipSuccess) return error;
            auto end = std::chrono::high_resolution_clock::now();
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            std::cout << " " << d.count() * 1000 << "ms" << '\n';
        }

        start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::scan_digits_kernel<radix_bits>),
            dim3(1), dim3(radix_size), 0, stream,
            digit_counts
        );
        if(debug_synchronous)
        {
            std::cout << "scan_digits";
            error = hipStreamSynchronize(stream);
            if(error != hipSuccess) return error;
            auto end = std::chrono::high_resolution_clock::now();
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            std::cout << " " << d.count() * 1000 << "ms" << '\n';
        }

        start = std::chrono::high_resolution_clock::now();
        if(need_encoding && need_decoding)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(detail::sort_and_scatter_kernel<sort_block_size, sort_items_per_thread, radix_bits, Descending, Descending>),
                dim3(batches), dim3(sort_block_size), 0, stream,
                keys_input, keys_output, size,
                batch_digit_counts, digit_counts,
                bit, current_radix_bits,
                div_blocks_per_batch, rem_blocks_per_batch
            );
        }
        else if(need_encoding)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(detail::sort_and_scatter_kernel<sort_block_size, sort_items_per_thread, radix_bits, Descending, false>),
                dim3(batches), dim3(sort_block_size), 0, stream,
                keys_input, bit_keys1, size,
                batch_digit_counts, digit_counts,
                bit, current_radix_bits,
                div_blocks_per_batch, rem_blocks_per_batch
            );
        }
        else if(need_decoding)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(detail::sort_and_scatter_kernel<sort_block_size, sort_items_per_thread, radix_bits, false, Descending>),
                dim3(batches), dim3(sort_block_size), 0, stream,
                bit_keys0, keys_output, size,
                batch_digit_counts, digit_counts,
                bit, current_radix_bits,
                div_blocks_per_batch, rem_blocks_per_batch
            );
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(detail::sort_and_scatter_kernel<sort_block_size, sort_items_per_thread, radix_bits, false, false>),
                dim3(batches), dim3(sort_block_size), 0, stream,
                bit_keys0, bit_keys1, size,
                batch_digit_counts, digit_counts,
                bit, current_radix_bits,
                div_blocks_per_batch, rem_blocks_per_batch
            );
        }
        if(debug_synchronous)
        {
            std::cout << "sort_and_scatter";
            error = hipStreamSynchronize(stream);
            if(error != hipSuccess) return error;
            auto end = std::chrono::high_resolution_clock::now();
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            std::cout << " " << d.count() * 1000 << "ms" << '\n';
        }

        std::swap(bit_keys0, bit_keys1);
    }

    return error;
}

} // end namespace detail

template<class Key>
hipError_t device_radix_sort_keys(void * temporary_storage,
                                  size_t& temporary_storage_bytes,
                                  const Key * keys_input,
                                  Key * keys_output,
                                  size_t size,
                                  unsigned int begin_bit = 0,
                                  unsigned int end_bit = 8 * sizeof(Key),
                                  hipStream_t stream = 0,
                                  bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    return detail::device_radix_sort<false>(
        temporary_storage, temporary_storage_bytes,
        keys_input, keys_output, values, values, size,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

template<class Key>
hipError_t device_radix_sort_keys_desc(void * temporary_storage,
                                       size_t& temporary_storage_bytes,
                                       const Key * keys_input,
                                       Key * keys_output,
                                       size_t size,
                                       unsigned int begin_bit = 0,
                                       unsigned int end_bit = 8 * sizeof(Key),
                                       hipStream_t stream = 0,
                                       bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    return detail::device_radix_sort<true>(
        temporary_storage, temporary_storage_bytes,
        keys_input, keys_output, values, values, size,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group collectivedevicemodule

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_
