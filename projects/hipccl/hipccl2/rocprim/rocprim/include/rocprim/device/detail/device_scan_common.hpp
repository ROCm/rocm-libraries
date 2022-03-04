// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_SCAN_COMMON_HPP_
#define ROCPRIM_DEVICE_SCAN_COMMON_HPP_

#include "../../config.hpp"
#include "../../intrinsics/thread.hpp"

#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"

#include <hip/hip_runtime.h>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    template <typename LookBackScanState>
    ROCPRIM_KERNEL
        __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE) void init_lookback_scan_state_kernel(
            LookBackScanState                             lookback_scan_state,
            const unsigned int                            number_of_blocks,
            ordered_block_id<unsigned int>                ordered_bid,
            unsigned int                                  save_index = 0,
            typename LookBackScanState::value_type* const save_dest  = nullptr)
    {
        const unsigned int block_id        = ::rocprim::detail::block_id<0>();
        const unsigned int block_size      = ::rocprim::detail::block_size<0>();
        const unsigned int block_thread_id = ::rocprim::detail::block_thread_id<0>();
        const unsigned int id              = (block_id * block_size) + block_thread_id;

        // Reset ordered_block_id
        if(id == 0)
        {
            ordered_bid.reset();
        }
        // Save the reduction (i.e. the last prefix) from the previous user of lookback_scan_state
        // If the thread that should reset it is participating then it saves the value before
        // reseting, otherwise the first thread saves it (it won't be reset by any thread).
        if(save_dest != nullptr
           && ((number_of_blocks <= save_index && id == 0) || id == save_index))
        {
            typename LookBackScanState::value_type value;
            typename LookBackScanState::flag_type  dummy_flag;
            lookback_scan_state.get(save_index, dummy_flag, value);

            *save_dest = value;
        }
        // Initialize lookback scan status
        lookback_scan_state.initialize_prefix(id, number_of_blocks);
    }

    template <bool Exclusive,
              class BlockScan,
              class T,
              unsigned int ItemsPerThread,
              class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto
        lookback_block_scan(T (&values)[ItemsPerThread],
                            T /* initial_value */,
                            T&                                reduction,
                            typename BlockScan::storage_type& storage,
                            BinaryFunction scan_op) -> typename std::enable_if<!Exclusive>::type
    {
        BlockScan().inclusive_scan(values, // input
                                   values, // output
                                   reduction,
                                   storage,
                                   scan_op);
    }

    template <bool Exclusive,
              class BlockScan,
              class T,
              unsigned int ItemsPerThread,
              class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto
        lookback_block_scan(T (&values)[ItemsPerThread],
                            T                                 initial_value,
                            T&                                reduction,
                            typename BlockScan::storage_type& storage,
                            BinaryFunction scan_op) -> typename std::enable_if<Exclusive>::type
    {
        BlockScan().exclusive_scan(values, // input
                                   values, // output
                                   initial_value,
                                   reduction,
                                   storage,
                                   scan_op);
        reduction = scan_op(initial_value, reduction);
    }

    template <bool Exclusive,
              class BlockScan,
              class T,
              unsigned int ItemsPerThread,
              class PrefixCallback,
              class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto
        lookback_block_scan(T (&values)[ItemsPerThread],
                            typename BlockScan::storage_type& storage,
                            PrefixCallback&                   prefix_callback_op,
                            BinaryFunction scan_op) -> typename std::enable_if<!Exclusive>::type
    {
        BlockScan().inclusive_scan(values, // input
                                   values, // output
                                   storage,
                                   prefix_callback_op,
                                   scan_op);
    }

    template <bool Exclusive,
              class BlockScan,
              class T,
              unsigned int ItemsPerThread,
              class PrefixCallback,
              class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto
        lookback_block_scan(T (&values)[ItemsPerThread],
                            typename BlockScan::storage_type& storage,
                            PrefixCallback&                   prefix_callback_op,
                            BinaryFunction scan_op) -> typename std::enable_if<Exclusive>::type
    {
        BlockScan().exclusive_scan(values, // input
                                   values, // output
                                   storage,
                                   prefix_callback_op,
                                   scan_op);
    }

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_SCAN_COMMON_HPP_