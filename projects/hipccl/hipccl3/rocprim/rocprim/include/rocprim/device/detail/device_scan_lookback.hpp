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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_LOOKBACK_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_LOOKBACK_HPP_

#include <type_traits>
#include <iterator>

#include "../../detail/various.hpp"
#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"
#include "../../block/block_scan.hpp"

#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"

BEGIN_ROCPRIM_NAMESPACE

// Single pass prefix scan was implemented based on:
// Merrill, D. and Garland, M. Single-pass Parallel Prefix Scan with Decoupled Look-back.
// Technical Report NVR2016-001, NVIDIA Research. Mar. 2016.

namespace detail
{

template<class LookBackScanState>
ROCPRIM_DEVICE inline
void init_lookback_scan_state_kernel_impl(LookBackScanState lookback_scan_state,
                                          const unsigned int number_of_blocks,
                                          ordered_block_id<unsigned int> ordered_block_id)
{
    const unsigned int block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_size = ::rocprim::detail::block_size<0>();
    const unsigned int block_thread_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int id = (block_id * block_size) + block_thread_id;

    // Reset ordered_block_id
    if(id == 0)
    {
        ordered_block_id.reset();
    }
    // Initialize lookback scan status
    lookback_scan_state.initialize_prefix(id, number_of_blocks);
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Exclusive,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class ResultType,
    class LookbackScanState
>
ROCPRIM_DEVICE inline
void lookback_scan_kernel_impl(InputIterator input,
                               OutputIterator output,
                               const size_t size,
                               const ResultType initial_value,
                               BinaryFunction scan_op,
                               LookbackScanState scan_state,
                               const unsigned int number_of_blocks,
                               ordered_block_id<unsigned int> ordered_bid)
{
    // TODO: Enable Exclusive
    (void) initial_value;
    (void) Exclusive;

    using flag_type = typename LookbackScanState::flag_type;
    using result_type = ResultType;
    static_assert(
        std::is_same<result_type, typename LookbackScanState::value_type>::value,
        "value_type of LookbackScanState must be result_type"
    );

    using order_bid_type  = ordered_block_id<unsigned int>;
    using block_load_type = ::rocprim::block_load<
        result_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose
    >;
    using block_store_type = ::rocprim::block_store<
        result_type, BlockSize, ItemsPerThread,
        ::rocprim::block_store_method::block_store_transpose
    >;
    using block_scan_type = ::rocprim::block_scan<
        result_type, BlockSize,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        result_type prefix;
        typename order_bid_type::storage_type ordered_bid;
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
        typename block_scan_type::storage_type scan;
    } storage;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const auto flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();
    const auto flat_block_id = ordered_bid.get(flat_block_thread_id, storage.ordered_bid);
    const unsigned int block_offset = flat_block_id * items_per_block;
    const auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);

    // For input values
    result_type values[ItemsPerThread];

    // load input values into values
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_type()
            .load(
                input + block_offset,
                values,
                valid_in_last_block,
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

    // Scan of block values
    block_scan_type()
        .inclusive_scan(
            values, // input
            values, // output
            storage.scan,
            scan_op
        );

    // First block in grid can store its last value as its complete prefix
    if(flat_block_id == 0)
    {
        if(flat_block_thread_id == BlockSize - 1)
        {
            scan_state.set_complete(flat_block_id, values[ItemsPerThread - 1]);
        }
    }
    // Other blocks set partial, calculate/get their prefixes, set complete
    else
    {
        if(flat_block_thread_id == BlockSize - 1)
        {
            // Set partial
            scan_state.set_partial(flat_block_id, values[ItemsPerThread - 1]);

            // Get prefix
            result_type partial_prefix;
            flag_type flag;
            scan_state.get(flat_block_id - 1, flag, partial_prefix);
            for(unsigned int i = flat_block_id - 2; flag != PREFIX_COMPLETE; i--)
            {
                result_type value;
                scan_state.get(i, flag, value);
                partial_prefix = scan_op(value, partial_prefix);
            }
            storage.prefix = partial_prefix;
        }
        __syncthreads();

        // adding prefix sum of previous blocks
        result_type prefix = storage.prefix;
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values[i] = scan_op(prefix, values[i]);
        }

        // Set complete
        if(flat_block_thread_id == BlockSize - 1)
        {
            scan_state.set_complete(flat_block_id, values[ItemsPerThread - 1]);
        }
    }

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

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_LOOKBACK_HPP_
