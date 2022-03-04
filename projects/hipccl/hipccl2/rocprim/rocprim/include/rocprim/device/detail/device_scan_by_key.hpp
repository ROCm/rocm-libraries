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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_BY_KEY_HPP_

#include "device_scan_common.hpp"
#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"

#include "../../block/block_discontinuity.hpp"
#include "../../block/block_load.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store.hpp"
#include "../../config.hpp"
#include "../../detail/binary_op_wrappers.hpp"
#include "../../intrinsics/thread.hpp"
#include "../../types/tuple.hpp"

#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    template <bool Exclusive,
              unsigned int block_size,
              unsigned int items_per_thread,
              typename key_type,
              typename result_type,
              ::rocprim::block_load_method load_keys_method,
              ::rocprim::block_load_method load_values_method>
    struct load_values_flagged
    {
        using block_load_keys
            = ::rocprim::block_load<key_type, block_size, items_per_thread, load_keys_method>;

        using block_discontinuity = ::rocprim::block_discontinuity<key_type, block_size>;
        
        using block_load_values
            = ::rocprim::block_load<result_type, block_size, items_per_thread, load_keys_method>;

        union storage_type {
            struct {
                typename block_load_keys::storage_type     load;
                typename block_discontinuity::storage_type flag;
            } keys;
            typename block_load_values::storage_type load_values;
        };

        // Load flagged values
        // - if the scan is exlusive the last item of each segment (range where the keys compare equal)
        //   is flagged and reset to the initial value. Adding the last item of the range to the 
        //   second to last using `headflag_scan_op_wrapper` will return the initial_value,
        //   which is exactly what should be saved at the start of the next range.
        // - if the scan is inclusive, then the first item of each segment is marked, and it will
        //   restart the scan from that value
        template <typename KeyIterator, typename ValueIterator, typename CompareFunction>
        ROCPRIM_DEVICE void
            load(KeyIterator        keys_input,
                 ValueIterator      values_input,
                 CompareFunction    compare,
                 const result_type  initial_value,
                 const unsigned int flat_block_id,
                 const size_t       starting_block,
                 const size_t       number_of_blocks,
                 const unsigned int flat_thread_id,
                 const size_t       size,
                 rocprim::tuple<result_type, bool> (&wrapped_values)[items_per_thread],
                 storage_type& storage)
        {
            constexpr static unsigned int items_per_block = items_per_thread * block_size;
            const unsigned int            block_offset    = flat_block_id * items_per_block;
            KeyIterator                   block_keys      = keys_input + block_offset;
            ValueIterator                 block_values    = values_input + block_offset;

            key_type    keys[items_per_thread];
            result_type values[items_per_thread];
            bool        flags[items_per_thread];

            auto not_equal
                = [compare](const auto& a, const auto& b) mutable { return !compare(a, b); };

            const auto flag_segment_boundaries = [&]() {
                if(Exclusive)
                {
                    const key_type tile_successor
                        = starting_block + flat_block_id < number_of_blocks - 1
                              ? block_keys[items_per_block]
                              : *block_keys;
                    block_discontinuity {}.flag_tails(
                        flags, tile_successor, keys, not_equal, storage.keys.flag);
                }
                else
                {
                    const key_type tile_predecessor = starting_block + flat_block_id > 0
                                                          ? block_keys[-1]
                                                          : *block_keys;
                    block_discontinuity {}.flag_heads(
                        flags, tile_predecessor, keys, not_equal, storage.keys.flag);
                }
            };

            if(starting_block + flat_block_id < number_of_blocks - 1)
            {
                block_load_keys{}.load(
                    block_keys,
                    keys,
                    storage.keys.load
                );

                flag_segment_boundaries();
                // Reusing shared memory for loading values
                ::rocprim::syncthreads();

                block_load_values{}.load(
                    block_values,
                    values,
                    storage.load_values
                );

                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < items_per_thread; ++i) {
                    rocprim::get<0>(wrapped_values[i])
                        = (Exclusive && flags[i]) ? initial_value : values[i];
                    rocprim::get<1>(wrapped_values[i]) = flags[i];
                }
            }
            else
            {
                const unsigned int valid_in_last_block
                    = static_cast<unsigned int>(size - items_per_block * (number_of_blocks - 1));

                block_load_keys {}.load(
                    block_keys,
                    keys,
                    valid_in_last_block,
                    *block_keys, // Any value is okay, so discontinuity doesn't access undefined items
                    storage.keys.load);

                flag_segment_boundaries();
                // Reusing shared memory for loading values
                ::rocprim::syncthreads();

                block_load_values{}.load(
                    block_values,
                    values,
                    valid_in_last_block,
                    storage.load_values
                );

                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < items_per_thread; ++i) {
                    if(flat_thread_id * items_per_thread + i >= valid_in_last_block) {
                        break;
                    }

                    rocprim::get<0>(wrapped_values[i])
                        = (Exclusive && flags[i]) ? initial_value : values[i];
                    rocprim::get<1>(wrapped_values[i]) = flags[i];
                }
            }
        }
    };

    template <unsigned int block_size,
              unsigned int items_per_thread,
              typename result_type,
              ::rocprim::block_store_method store_method>
    struct unwrap_store
    {
        using block_store_values
            = ::rocprim::block_store<result_type, block_size, items_per_thread, store_method>;

        using storage_type = typename block_store_values::storage_type;

        template <typename OutputIterator>
        ROCPRIM_DEVICE void
            store(OutputIterator     output,
                  const unsigned int flat_block_id,
                  const size_t       starting_block,
                  const size_t       number_of_blocks,
                  const unsigned int flat_thread_id,
                  const size_t       size,
                  const rocprim::tuple<result_type, bool> (&wrapped_values)[items_per_thread],
                  storage_type& storage)
        {
            constexpr static unsigned int items_per_block = items_per_thread * block_size;
            const unsigned int block_offset = flat_block_id * items_per_block;
            OutputIterator block_output = output + block_offset;

            result_type thread_values[items_per_thread];

            if(starting_block + flat_block_id < number_of_blocks - 1)
            {
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < items_per_thread; ++i) {
                    thread_values[i] = rocprim::get<0>(wrapped_values[i]);
                }

                // Reusing shared memory from scan to perform store
                rocprim::syncthreads();

                block_store_values {}.store(block_output, thread_values, storage);
            }
            else
            {
                const unsigned int valid_in_last_block
                    = static_cast<unsigned int>(size - items_per_block * (number_of_blocks - 1));

                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < items_per_thread; ++i) {
                    if(flat_thread_id * items_per_thread + i >= valid_in_last_block) {
                        break;
                    }

                    thread_values[i] = rocprim::get<0>(wrapped_values[i]);
                }

                // Reusing shared memory from scan to perform store
                rocprim::syncthreads();

                block_store_values {}.store(
                    block_output, thread_values, valid_in_last_block, storage);
            }
        }
    };

    template <bool Exclusive,
              typename Config,
              typename KeyInputIterator,
              typename InputIterator,
              typename OutputIterator,
              typename ResultType,
              typename CompareFunction,
              typename BinaryFunction,
              typename LookbackScanState>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void device_scan_by_key_kernel_impl(
        KeyInputIterator                              keys,
        InputIterator                                 values,
        OutputIterator                                output,
        ResultType                                    initial_value,
        const CompareFunction                         compare,
        const BinaryFunction                          scan_op,
        LookbackScanState                             scan_state,
        const size_t                                  size,
        const size_t                                  starting_block,
        const size_t                                  number_of_blocks,
        ordered_block_id<unsigned int>                ordered_bid,
        const rocprim::tuple<ResultType, bool>* const previous_last_value)
    {
        using result_type = ResultType;
        static_assert(std::is_same<rocprim::tuple<ResultType, bool>,
                                   typename LookbackScanState::value_type>::value,
                      "value_type of LookbackScanState must be tuple of result type and flag");

        constexpr auto block_size         = Config::block_size;
        constexpr auto items_per_thread   = Config::items_per_thread;
        constexpr auto load_keys_method   = Config::block_load_method;
        constexpr auto load_values_method = load_keys_method;

        using key_type = typename std::iterator_traits<KeyInputIterator>::value_type;
        using load_flagged = load_values_flagged<Exclusive,
                                                 block_size,
                                                 items_per_thread,
                                                 key_type,
                                                 result_type,
                                                 load_keys_method,
                                                 load_values_method>;

        auto wrapped_op = headflag_scan_op_wrapper<result_type, bool, BinaryFunction>{scan_op};
        using wrapped_type = rocprim::tuple<result_type, bool>;

        using block_scan_type
            = ::rocprim::block_scan<wrapped_type, block_size, Config::block_scan_method>;

        constexpr auto store_method = Config::block_store_method;
        using store_unwrap = unwrap_store<block_size, items_per_thread, result_type, store_method>;

        using order_bid_type = ordered_block_id<unsigned int>;

        ROCPRIM_SHARED_MEMORY union
        {
            struct
            {
                typename load_flagged::storage_type   load;
                typename order_bid_type::storage_type ordered_bid;
            };
            typename block_scan_type::storage_type scan;
            typename store_unwrap::storage_type    store;
        } storage;

        const auto flat_thread_id = ::rocprim::detail::block_thread_id<0>();
        const auto flat_block_id = ordered_bid.get(flat_thread_id, storage.ordered_bid);

        // Load input
        wrapped_type wrapped_values[items_per_thread];
        load_flagged {}.load(keys,
                             values,
                             compare,
                             initial_value,
                             flat_block_id,
                             starting_block,
                             number_of_blocks,
                             flat_thread_id,
                             size,
                             wrapped_values,
                             storage.load);

        // Reusing the storage from load to perform the scan
        ::rocprim::syncthreads();

        // Perform look back scan scan
        if(flat_block_id == 0)
        {
            auto wrapped_initial_value = rocprim::make_tuple(initial_value, false);

            // previous_last_value is used to pass the value from the previous grid, if this is a
            // multi grid launch
            if(previous_last_value != nullptr)
            {
                if(Exclusive) {
                    rocprim::get<0>(wrapped_initial_value) = rocprim::get<0>(*previous_last_value);
                } else if (flat_thread_id == 0) {
                    wrapped_values[0] = wrapped_op(*previous_last_value, wrapped_values[0]);
                }
            }

            wrapped_type reduction;
            lookback_block_scan<Exclusive, block_scan_type>(wrapped_values,
                                                            wrapped_initial_value,
                                                            reduction,
                                                            storage.scan,
                                                            wrapped_op);

            if(flat_thread_id == 0)
            {
                scan_state.set_complete(flat_block_id, reduction);
            }
        }
        else
        {
            auto prefix_op = lookback_scan_prefix_op<wrapped_type,
                                                     decltype(wrapped_op),
                                                     decltype(scan_state)> {
                flat_block_id, wrapped_op, scan_state};

            // Scan of block values
            lookback_block_scan<Exclusive, block_scan_type>(
                wrapped_values,
                storage.scan,
                prefix_op,
                wrapped_op);
        }

        // Store output
        // synchronization is inside the function after unwrapping
        store_unwrap {}.store(output,
                              flat_block_id,
                              starting_block,
                              number_of_blocks,
                              flat_thread_id,
                              size,
                              wrapped_values,
                              storage.store);
    }
} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_BY_KEY_HPP_