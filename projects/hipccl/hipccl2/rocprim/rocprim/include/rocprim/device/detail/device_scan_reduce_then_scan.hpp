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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_REDUCE_THEN_SCAN_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_REDUCE_THEN_SCAN_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_reduce.hpp"


BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Helper functions for performing exclusive or inclusive
// block scan in single_scan.
template<
    bool Exclusive,
    class BlockScan,
    class T,
    unsigned int ItemsPerThread,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
auto single_scan_block_scan(T (&input)[ItemsPerThread],
                            T (&output)[ItemsPerThread],
                            T initial_value,
                            typename BlockScan::storage_type& storage,
                            BinaryFunction scan_op)
    -> typename std::enable_if<Exclusive>::type
{
    BlockScan()
        .exclusive_scan(
            input, // input
            output, // output
            initial_value,
            storage,
            scan_op
        );
}

template<
    bool Exclusive,
    class BlockScan,
    class T,
    unsigned int ItemsPerThread,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
auto single_scan_block_scan(T (&input)[ItemsPerThread],
                            T (&output)[ItemsPerThread],
                            T initial_value,
                            typename BlockScan::storage_type& storage,
                            BinaryFunction scan_op)
    -> typename std::enable_if<!Exclusive>::type
{
    (void) initial_value;
    BlockScan()
        .inclusive_scan(
            input, // input
            output, // output
            storage,
            scan_op
        );
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class ResultType
>
ROCPRIM_DEVICE inline
void single_scan_kernel_impl(InputIterator input,
                             const size_t input_size,
                             ResultType initial_value,
                             OutputIterator output,
                             BinaryFunction scan_op)
{
    constexpr unsigned int block_size = Config::block_size;
    constexpr unsigned int items_per_thread = Config::items_per_thread;

    using result_type = ResultType;

    using block_load_type = ::rocprim::block_load<
        result_type, block_size, items_per_thread,
        Config::block_load_method
    >;
    using block_store_type = ::rocprim::block_store<
        result_type, block_size, items_per_thread,
        Config::block_store_method
    >;
    using block_scan_type = ::rocprim::block_scan<
        result_type, block_size,
        Config::block_scan_method
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
        typename block_scan_type::storage_type scan;
    } storage;

    result_type values[items_per_thread];
    // load input values into values
    block_load_type()
        .load(
            input,
            values,
            input_size,
            *(input),
            storage.load
        );
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    single_scan_block_scan<Exclusive, block_scan_type>(
        values, // input
        values, // output
        initial_value,
        storage.scan,
        scan_op
    );
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    // Save values into output array
    block_store_type()
        .store(
            output,
            values,
            input_size,
            storage.store
        );
}

// Calculates block prefixes that will be used in final_scan
// when performing block scan operations.
template<
    class Config,
    class InputIterator,
    class BinaryFunction,
    class ResultType
>
ROCPRIM_DEVICE inline
void block_reduce_kernel_impl(InputIterator input,
                              BinaryFunction scan_op,
                              ResultType * block_prefixes)
{
    constexpr unsigned int block_size = Config::block_size;
    constexpr unsigned int items_per_thread = Config::items_per_thread;

    using result_type = ResultType;
    using block_reduce_type = ::rocprim::block_reduce<
        result_type, block_size,
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
    using block_load_type = ::rocprim::block_load<
        result_type, block_size, items_per_thread,
        Config::block_load_method
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type load;
        typename block_reduce_type::storage_type reduce;
    } storage;

    const unsigned int flat_id = ::rocprim::flat_block_thread_id();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_thread * block_size;

    // For input values
    result_type values[items_per_thread];
    result_type block_prefix;

    block_load_type()
        .load(
            input + block_offset,
            values,
            storage.load
        );
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    block_reduce_type()
        .reduce(
            values, // input
            block_prefix, // output
            storage.reduce,
            scan_op
        );

    // Save block prefix
    if(flat_id == 0)
    {
        block_prefixes[flat_block_id] = block_prefix;
    }
}

// Helper functions for performing exclusive or inclusive
// block scan operation in final_scan
template<
    bool Exclusive,
    class BlockScan,
    class T,
    unsigned int ItemsPerThread,
    class ResultType,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
auto final_scan_block_scan(const unsigned int flat_block_id,
                           T (&input)[ItemsPerThread],
                           T (&output)[ItemsPerThread],
                           T initial_value,
                           ResultType * block_prefixes,
                           typename BlockScan::storage_type& storage,
                           BinaryFunction scan_op)
    -> typename std::enable_if<Exclusive>::type
{
    if(flat_block_id != 0)
    {
        // Include initial value in block prefix
        initial_value = scan_op(
            initial_value, block_prefixes[flat_block_id - 1]
        );
    }
    BlockScan()
        .exclusive_scan(
            input, // input
            output, // output
            initial_value,
            storage,
            scan_op
        );
}

template<
    bool Exclusive,
    class BlockScan,
    class T,
    unsigned int ItemsPerThread,
    class ResultType,
    class BinaryFunction
>
ROCPRIM_DEVICE inline
auto final_scan_block_scan(const unsigned int flat_block_id,
                           T (&input)[ItemsPerThread],
                           T (&output)[ItemsPerThread],
                           T initial_value,
                           ResultType * block_prefixes,
                           typename BlockScan::storage_type& storage,
                           BinaryFunction scan_op)
    -> typename std::enable_if<!Exclusive>::type
{
    (void) initial_value;
    if(flat_block_id == 0)
    {
        BlockScan()
            .inclusive_scan(
                input, // input
                output, // output
                storage,
                scan_op
            );
    }
    else
    {
        auto block_prefix_op =
            [&block_prefixes, &flat_block_id](const T& /*not used*/)
            {
                return block_prefixes[flat_block_id - 1];
            };
        BlockScan()
            .inclusive_scan(
                input, // input
                output, // output
                storage,
                block_prefix_op,
                scan_op
            );
    }
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class ResultType
>
ROCPRIM_DEVICE inline
void final_scan_kernel_impl(InputIterator input,
                            const size_t input_size,
                            OutputIterator output,
                            const ResultType initial_value,
                            BinaryFunction scan_op,
                            ResultType * block_prefixes)
{
    constexpr unsigned int block_size = Config::block_size;
    constexpr unsigned int items_per_thread = Config::items_per_thread;

    using result_type = ResultType;

    using block_load_type = ::rocprim::block_load<
        result_type, block_size, items_per_thread,
        Config::block_load_method
    >;
    using block_store_type = ::rocprim::block_store<
        result_type, block_size, items_per_thread,
        Config::block_store_method
    >;
    using block_scan_type = ::rocprim::block_scan<
        result_type, block_size,
        Config::block_scan_method
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
        typename block_scan_type::storage_type scan;
    } storage;

    // It's assumed kernel is executed in 1D
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();

    constexpr unsigned int items_per_block = block_size * items_per_thread;
    const unsigned int block_offset = flat_block_id * items_per_block;
    // TODO: number_of_blocks can be calculated on host
    const unsigned int number_of_blocks = (input_size + items_per_block - 1)/items_per_block;

    // For input values
    result_type values[items_per_thread];

    // TODO: valid_in_last_block can be calculated on host
    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);
    // load input values into values
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_type()
            .load(
                input + block_offset,
                values,
                valid_in_last_block,
                *(input + block_offset),
                storage.load
            );
    }
    else
    {
        block_load_type()
            .load(
                input + block_offset,
                values,
                storage.load
            );
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    final_scan_block_scan<Exclusive, block_scan_type>(
        flat_block_id,
        values, // input
        values, // output
        initial_value,
        block_prefixes,
        storage.scan,
        scan_op
    );
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    // Save values into output array
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_store_type()
            .store(
                output + block_offset,
                values,
                valid_in_last_block,
                storage.store
            );
    }
    else
    {
        block_store_type()
            .store(
                output + block_offset,
                values,
                storage.store
            );
    }
}

// Returns size of temporary storage in bytes.
template<class T>
size_t scan_get_temporary_storage_bytes(size_t input_size,
                                        size_t items_per_block)
{
    if(input_size <= items_per_block)
    {
        return 0;
    }
    auto size = (input_size + items_per_block - 1)/(items_per_block);
    return size * sizeof(T) + scan_get_temporary_storage_bytes<T>(size, items_per_block);
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_REDUCE_THEN_SCAN_HPP_
