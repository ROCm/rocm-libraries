// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "device_scan_common.hpp"
#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"

BEGIN_ROCPRIM_NAMESPACE

// Single pass prefix scan was implemented based on:
// Merrill, D. and Garland, M. Single-pass Parallel Prefix Scan with Decoupled Look-back.
// Technical Report NVR2016-001, NVIDIA Research. Mar. 2016.

namespace detail
{

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class ResultType,
    class LookbackScanState
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void lookback_scan_kernel_impl(InputIterator input,
                               OutputIterator output,
                               const size_t size,
                               ResultType initial_value,
                               BinaryFunction scan_op,
                               LookbackScanState scan_state,
                               const unsigned int number_of_blocks,
                               ordered_block_id<unsigned int> ordered_bid,
                               ResultType * previous_last_element = nullptr,
                               ResultType * new_last_element = nullptr,
                               bool override_first_value = false,
                               bool save_last_value = false)
{
    using result_type = ResultType;
    static_assert(
        std::is_same<result_type, typename LookbackScanState::value_type>::value,
        "value_type of LookbackScanState must be result_type"
    );

    constexpr auto block_size = Config::block_size;
    constexpr auto items_per_thread = Config::items_per_thread;
    constexpr unsigned int items_per_block = block_size * items_per_thread;

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

    using order_bid_type = ordered_block_id<unsigned int>;
    using lookback_scan_prefix_op_type = lookback_scan_prefix_op<
        result_type, BinaryFunction, LookbackScanState
    >;

    ROCPRIM_SHARED_MEMORY struct
    {
        typename order_bid_type::storage_type ordered_bid;
        union
        {
            typename block_load_type::storage_type load;
            typename block_store_type::storage_type store;
            typename block_scan_type::storage_type scan;
        };
    } storage;

    const auto flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();
    const auto flat_block_id = ordered_bid.get(flat_block_thread_id, storage.ordered_bid);
    const unsigned int block_offset = flat_block_id * items_per_block;
    const auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);

    // For input values
    result_type values[items_per_thread];

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

    if(flat_block_id == 0)
    {
        // override_first_value only true when the first chunk already processed
        // and input iterator starts from an offset.
        if(override_first_value)
        {
            if(Exclusive)
                initial_value = scan_op(previous_last_element[0], static_cast<result_type>(*(input-1)));
            else if(flat_block_thread_id == 0)
                values[0] = scan_op(previous_last_element[0], values[0]);
        }

        result_type reduction;
        lookback_block_scan<Exclusive, block_scan_type>(
            values, // input/output
            initial_value,
            reduction,
            storage.scan,
            scan_op
        );

        if(flat_block_thread_id == 0)
        {
            scan_state.set_complete(flat_block_id, reduction);
        }
    }
    else
    {
        // Scan of block values
        auto prefix_op = lookback_scan_prefix_op_type(
            flat_block_id, scan_op, scan_state
        );
        lookback_block_scan<Exclusive, block_scan_type>(
            values, // input/output
            storage.scan,
            prefix_op,
            scan_op
        );
    }
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

        if(save_last_value &&
           (::rocprim::detail::block_thread_id<0>() ==
           (valid_in_last_block - 1) / items_per_thread))
        {
            for(unsigned int i = 0; i < items_per_thread; i++)
            {
                if(i == (valid_in_last_block - 1) % items_per_thread)
                {
                    new_last_element[0] = values[i];
                }
            }
        }
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
