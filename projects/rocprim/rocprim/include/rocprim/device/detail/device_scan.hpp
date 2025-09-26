// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iterator>
#include <type_traits>

#include "../../detail/various.hpp"
#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store.hpp"

#include "../../device/device_scan_config.hpp"

#include "device_scan_common.hpp"
#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"

BEGIN_ROCPRIM_NAMESPACE

// Single pass prefix scan was implemented based on:
// Merrill, D. and Garland, M. Single-pass Parallel Prefix Scan with Decoupled Look-back.
// Technical Report NVR2016-001, NVIDIA Research. Mar. 2016.

namespace detail
{

// Helper functions for performing exclusive or inclusive
// block scan in single_scan.
template<bool Exclusive,
         bool UseInitialValue,
         class BlockScan,
         class T,
         unsigned int ItemsPerThread,
         class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto single_scan_block_scan(T (&input)[ItemsPerThread],
                            T (&output)[ItemsPerThread],
                            T                                 initial_value,
                            typename BlockScan::storage_type& storage,
                            BinaryFunction scan_op) -> typename std::enable_if<Exclusive>::type
{
    BlockScan().exclusive_scan(input, // input
                               output, // output
                               initial_value,
                               storage,
                               scan_op);
}

template<bool Exclusive,
         bool UseInitialValue,
         class BlockScan,
         class T,
         unsigned int ItemsPerThread,
         class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto single_scan_block_scan(T (&input)[ItemsPerThread],
                            T (&output)[ItemsPerThread],
                            T                                 initial_value,
                            typename BlockScan::storage_type& storage,
                            BinaryFunction scan_op) -> typename std::enable_if<!Exclusive>::type
{
    if constexpr(UseInitialValue)
    {
        BlockScan().inclusive_scan(input, // input
                                   initial_value,
                                   output, // output
                                   storage,
                                   scan_op);
    }
    else
    {
        BlockScan().inclusive_scan(input, // input
                                   output, // output
                                   storage,
                                   scan_op);
    }
}

template<class ArchConfig,
         lookback_scan_determinism Determinism,
         bool                      Exclusive,
         bool UseInitialValue,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class AccType,
         class LookbackScanState>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE auto lookback_scan_kernel_impl(InputIterator,
                                                                   OutputIterator,
                                                                   const size_t,
                                                                   AccType,
                                                                   BinaryFunction,
                                                                   LookbackScanState,
                                                                   const unsigned int,
                                                                   AccType* = nullptr,
                                                                   AccType* = nullptr,
                                                                   bool     = false,
                                                                   bool     = false)
    -> std::enable_if_t<!is_lookback_kernel_runnable<LookbackScanState>()>
{
    // No need to build the kernel with sleep on a device that does not require it
}

template<class ArchConfig,
         lookback_scan_determinism Determinism,
         bool                      Exclusive,
         bool UseInitialValue,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class AccType,
         class LookbackScanState>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE auto
    lookback_scan_kernel_impl(InputIterator      input,
                              OutputIterator     output,
                              const size_t       size,
                              AccType            initial_value,
                              BinaryFunction     scan_op,
                              LookbackScanState  scan_state,
                              const unsigned int number_of_blocks,
                              AccType*           previous_last_element = nullptr,
                              AccType*           new_last_element      = nullptr,
                              bool               override_first_value  = false,
                              bool               save_last_value       = false)
        -> std::enable_if_t<is_lookback_kernel_runnable<LookbackScanState>()>
{
    static_assert(std::is_same<AccType, typename LookbackScanState::value_type>::value,
                  "value_type of LookbackScanState must be result_type");
    static constexpr scan_config_params params = ArchConfig::params;

    constexpr auto         block_size       = params.kernel_config.block_size;
    constexpr auto         items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;

    using block_load_type
        = ::rocprim::block_load<AccType, block_size, items_per_thread, params.block_load_method>;
    using block_store_type
        = ::rocprim::block_store<AccType, block_size, items_per_thread, params.block_store_method>;
    using block_scan_type = ::rocprim::block_scan<AccType, block_size, params.block_scan_method>;

    using lookback_scan_prefix_op_type
        = lookback_scan_prefix_op<AccType, BinaryFunction, LookbackScanState, Determinism>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type  load;
        typename block_store_type::storage_type store;
        typename block_scan_type::storage_type  scan;
    } storage;

    const auto         flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();
    const auto         flat_block_id        = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset         = flat_block_id * items_per_block;
    const auto         valid_in_last_block  = size - items_per_block * (number_of_blocks - 1);

    // For input values
    AccType values[items_per_thread];

    // load input values into values
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_type().load(input + block_offset,
                               values,
                               valid_in_last_block,
                               *(input + block_offset),
                               storage.load);
    }
    else
    {
        block_load_type().load(input + block_offset, values, storage.load);
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    if(flat_block_id == 0)
    {
        // override_first_value only true when the first chunk already processed
        // and input iterator starts from an offset.
        if(override_first_value)
        {
            if(Exclusive)
            {
                initial_value
                    = scan_op(previous_last_element[0], static_cast<AccType>(*(input - 1)));
            }
            else if(flat_block_thread_id == 0)
            {
                values[0] = scan_op(previous_last_element[0], values[0]);
            }
        }

        AccType reduction;

        // Since `override_first_value` isn't a constexpr and there's no exclusive block scan
        // overload without an initial_value parameter, the two scan types need separate
        // code paths, this duplicates a bit of code.
        if constexpr(Exclusive)
        {
            lookback_block_scan<Exclusive, block_scan_type>(values, // input/output
                                                            initial_value,
                                                            reduction,
                                                            storage.scan,
                                                            scan_op);

            // Reduction should not contain initial value.
            if(flat_block_thread_id == 0)
            {
                scan_state.set_complete(flat_block_id, reduction);
            }
        }
        else
        {
            if(UseInitialValue && !override_first_value)
            {
                lookback_block_scan<Exclusive, block_scan_type>(values, // input/output
                                                                initial_value,
                                                                reduction,
                                                                storage.scan,
                                                                scan_op);
            }
            else
            {
                // Only use the initial value on the first iteration
                lookback_block_scan<Exclusive, block_scan_type>(values, // input/output
                                                                reduction,
                                                                storage.scan,
                                                                scan_op);
            }

            // Reduction should include initial value. We can avoid block-wide communication
            // communication by letting the thread that has the last element of the scan
            // write it to memory.
            if(flat_block_thread_id == block_size - 1)
            {
                scan_state.set_complete(flat_block_id, values[items_per_thread - 1]);
            }
        }
    }
    else
    {
        // Scan of block values
        auto prefix_op = lookback_scan_prefix_op_type(flat_block_id, scan_op, scan_state);
        lookback_block_scan<Exclusive, block_scan_type>(values, // input/output
                                                        storage.scan,
                                                        prefix_op,
                                                        scan_op);
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    // Save values into output array
    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_store_type().store(output + block_offset, values, valid_in_last_block, storage.store);

        if(save_last_value
           && (::rocprim::detail::block_thread_id<0>()
               == (valid_in_last_block - 1) / items_per_thread))
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
        block_store_type().store(output + block_offset, values, storage.store);
    }
}

} // end of namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_LOOKBACK_HPP_
