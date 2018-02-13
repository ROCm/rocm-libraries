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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HC_HPP_

#include <iterator>
#include <iostream>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../functional.hpp"

#include "detail/device_reduce_by_key.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#define ROCPRIM_DETAIL_HC_SYNC(name, size, start) \
    { \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            acc_view.wait(); \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<
    class KeysInputIterator,
    class ValuesInputIterator,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class UniqueCountOutputIterator,
    class BinaryFunction,
    class KeyCompareFunction
>
inline
void device_reduce_by_key_impl(void * temporary_storage,
                               size_t& storage_size,
                               KeysInputIterator keys_input,
                               ValuesInputIterator values_input,
                               const size_t size,
                               UniqueOutputIterator unique_output,
                               AggregatesOutputIterator aggregates_output,
                               UniqueCountOutputIterator unique_count_output,
                               BinaryFunction reduce_op,
                               KeyCompareFunction key_compare_op,
                               hc::accelerator_view acc_view,
                               const bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using carry_out_type = carry_out<key_type, value_type>;

    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 7;

    constexpr unsigned int scan_block_size = 256;
    constexpr unsigned int scan_items_per_thread = 7;

    constexpr unsigned int items_per_block = block_size * items_per_thread;
    constexpr unsigned int scan_items_per_block = scan_block_size * scan_items_per_thread;

    const unsigned int blocks = ::rocprim::detail::ceiling_div(static_cast<unsigned int>(size), items_per_block);
    const unsigned int blocks_per_full_batch = ::rocprim::detail::ceiling_div(blocks, scan_items_per_block);
    const unsigned int full_batches = blocks % scan_items_per_block != 0
        ? blocks % scan_items_per_block
        : scan_items_per_block;
    const unsigned int batches = (blocks_per_full_batch == 1 ? full_batches : scan_items_per_block);

    const size_t unique_counts_bytes = ::rocprim::detail::align_size(batches * sizeof(unsigned int));
    const size_t carry_outs_bytes = ::rocprim::detail::align_size(batches * sizeof(carry_out_type));
    if(temporary_storage == nullptr)
    {
        storage_size = unique_counts_bytes + carry_outs_bytes;
        return;
    }

    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "items_per_thread " << items_per_thread << '\n';
        std::cout << "blocks " << blocks << '\n';
        std::cout << "blocks_per_full_batch " << blocks_per_full_batch << '\n';
        std::cout << "full_batches " << full_batches << '\n';
        std::cout << "batches " << batches << '\n';
        std::cout << "storage_size " << storage_size << '\n';
        acc_view.wait();
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    unsigned int * unique_counts = reinterpret_cast<unsigned int *>(ptr);
    ptr += unique_counts_bytes;
    carry_out_type * carry_outs = reinterpret_cast<carry_out_type *>(ptr);

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(batches * block_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            fill_unique_counts<block_size, items_per_thread>(
                keys_input, size, unique_counts, key_compare_op,
                blocks_per_full_batch, full_batches, blocks
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("fill_unique_counts", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(scan_block_size, scan_block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            scan_unique_counts<scan_block_size, scan_items_per_thread>(
                unique_counts, unique_count_output,
                batches
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("scan_unique_counts", scan_block_size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(batches * block_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            reduce_by_key<block_size, items_per_thread>(
                keys_input, values_input, size,
                unique_counts, carry_outs,
                unique_output, aggregates_output,
                key_compare_op, reduce_op,
                blocks_per_full_batch, full_batches, blocks
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("reduce_by_key", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(scan_block_size, scan_block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            scan_and_scatter_carry_outs<scan_block_size, scan_items_per_thread>(
                carry_outs,
                aggregates_output,
                key_compare_op, reduce_op,
                batches
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("scan_and_scatter_carry_outs", scan_block_size, start)
}

#undef ROCPRIM_DETAIL_HC_SYNC

} // end of detail namespace

template<
    class KeysInputIterator,
    class ValuesInputIterator,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class UniqueCountOutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
    class KeyCompareFunction = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
void device_reduce_by_key(void * temporary_storage,
                          size_t& storage_size,
                          KeysInputIterator keys_input,
                          ValuesInputIterator values_input,
                          const size_t size,
                          UniqueOutputIterator unique_output,
                          AggregatesOutputIterator aggregates_output,
                          UniqueCountOutputIterator unique_count_output,
                          BinaryFunction reduce_op = BinaryFunction(),
                          KeyCompareFunction key_compare_op = KeyCompareFunction(),
                          hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                          const bool debug_synchronous = false)
{
    detail::device_reduce_by_key_impl(
        temporary_storage, storage_size,
        keys_input, values_input, size,
        unique_output, aggregates_output, unique_count_output,
        reduce_op, key_compare_op,
        acc_view, debug_synchronous
    );
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HC_HPP_
