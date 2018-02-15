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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HIP_HPP_

#include <iterator>
#include <iostream>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../functional.hpp"

#include "detail/device_reduce_by_key.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class KeyCompareFunction
>
__global__
void fill_unique_counts_kernel(KeysInputIterator keys_input,
                               unsigned int size,
                               unsigned int * unique_counts,
                               KeyCompareFunction key_compare_op,
                               unsigned int blocks_per_full_batch,
                               unsigned int full_batches,
                               unsigned int blocks)
{
    fill_unique_counts<BlockSize, ItemsPerThread>(
        keys_input, size,
        unique_counts,
        key_compare_op,
        blocks_per_full_batch, full_batches, blocks
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class UniqueCountOutputIterator
>
__global__
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
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class CarryOut,
    class KeyCompareFunction,
    class BinaryFunction
>
__global__
void reduce_by_key_kernel(KeysInputIterator keys_input,
                          ValuesInputIterator values_input,
                          unsigned int size,
                          const unsigned int * unique_starts,
                          CarryOut * carry_outs,
                          UniqueOutputIterator unique_output,
                          AggregatesOutputIterator aggregates_output,
                          KeyCompareFunction key_compare_op,
                          BinaryFunction reduce_op,
                          unsigned int blocks_per_full_batch,
                          unsigned int full_batches,
                          unsigned int blocks)
{
    reduce_by_key<BlockSize, ItemsPerThread>(
        keys_input, values_input, size,
        unique_starts, carry_outs,
        unique_output, aggregates_output,
        key_compare_op, reduce_op,
        blocks_per_full_batch, full_batches, blocks
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class AggregatesOutputIterator,
    class CarryOut,
    class KeyCompareFunction,
    class BinaryFunction
>
__global__
void scan_and_scatter_carry_outs_kernel(const CarryOut * carry_outs,
                                        AggregatesOutputIterator aggregates_output,
                                        KeyCompareFunction key_compare_op,
                                        BinaryFunction reduce_op,
                                        unsigned int batches)
{
    scan_and_scatter_carry_outs<BlockSize, ItemsPerThread>(
        carry_outs, aggregates_output,
        key_compare_op, reduce_op,
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
    class KeysInputIterator,
    class ValuesInputIterator,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class UniqueCountOutputIterator,
    class BinaryFunction,
    class KeyCompareFunction
>
inline
hipError_t device_reduce_by_key_impl(void * temporary_storage,
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
        return hipSuccess;
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
        hipError_t error = hipStreamSynchronize(stream);
        if(error != hipSuccess) return error;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    unsigned int * unique_counts = reinterpret_cast<unsigned int *>(ptr);
    ptr += unique_counts_bytes;
    carry_out_type * carry_outs = reinterpret_cast<carry_out_type *>(ptr);

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(fill_unique_counts_kernel<block_size, items_per_thread>),
        dim3(batches), dim3(block_size), 0, stream,
        keys_input, size, unique_counts, key_compare_op,
        blocks_per_full_batch, full_batches, blocks
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("fill_unique_counts", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scan_unique_counts_kernel<scan_block_size, scan_items_per_thread>),
        dim3(1), dim3(scan_block_size), 0, stream,
        unique_counts, unique_count_output,
        batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_unique_counts", scan_block_size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(reduce_by_key_kernel<block_size, items_per_thread>),
        dim3(batches), dim3(block_size), 0, stream,
        keys_input, values_input, size,
        const_cast<const unsigned int *>(unique_counts), carry_outs,
        unique_output, aggregates_output,
        key_compare_op, reduce_op,
        blocks_per_full_batch, full_batches, blocks
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("reduce_by_key", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scan_and_scatter_carry_outs_kernel<scan_block_size, scan_items_per_thread>),
        dim3(1), dim3(scan_block_size), 0, stream,
        const_cast<const carry_out_type *>(carry_outs),
        aggregates_output,
        key_compare_op, reduce_op,
        batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_and_scatter_carry_outs", scan_block_size, start)

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

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
hipError_t device_reduce_by_key(void * temporary_storage,
                                size_t& storage_size,
                                KeysInputIterator keys_input,
                                ValuesInputIterator values_input,
                                const unsigned int size,
                                UniqueOutputIterator unique_output,
                                AggregatesOutputIterator aggregates_output,
                                UniqueCountOutputIterator unique_count_output,
                                BinaryFunction reduce_op = BinaryFunction(),
                                KeyCompareFunction key_compare_op = KeyCompareFunction(),
                                const hipStream_t stream = 0,
                                const bool debug_synchronous = false)
{
    return detail::device_reduce_by_key_impl(
        temporary_storage, storage_size,
        keys_input, values_input, size,
        unique_output, aggregates_output, unique_count_output,
        reduce_op, key_compare_op,
        stream, debug_synchronous
    );
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HIP_HPP_
