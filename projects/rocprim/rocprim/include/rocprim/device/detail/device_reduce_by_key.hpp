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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_BY_KEY_HPP_

#include "device_config_helper.hpp"
#include "lookback_scan_state.hpp"

#include "../../block/block_discontinuity.hpp"
#include "../../block/block_load.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store.hpp"
#include "../../detail/various.hpp"
#include "../../intrinsics/thread.hpp"
#include "../../thread/thread_operators.hpp"

#include "../../config.hpp"
#include "../../type_traits.hpp"

#include <iterator>
#include <type_traits>
#include <utility>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

namespace reduce_by_key
{

template<typename ValueIterator, typename BinaryOp>
using accumulator_type_t
    = ::rocprim::accumulator_t<BinaryOp, ::rocprim::detail::value_type_t<ValueIterator>>;

template<typename AccumulatorType>
using wrapped_type_t = rocprim::tuple<unsigned int, AccumulatorType>;

template<typename AccumulatorType, bool UseSleep = false>
using lookback_scan_state_t
    = detail::lookback_scan_state<wrapped_type_t<AccumulatorType>, UseSleep>;

template<typename KeyType,
         typename AccumulatorType,
         unsigned int      BlockSize,
         unsigned int      ItemsPerThread,
         block_load_method load_keys_method,
         block_load_method load_values_method>
struct load_helper
{
    using block_load_keys = block_load<KeyType, BlockSize, ItemsPerThread, load_keys_method>;
    using block_load_values
        = block_load<AccumulatorType, BlockSize, ItemsPerThread, load_values_method>;

    /// We only need to sync between loading keys & values if BOTH the key and value
    /// loading method require shared memory.
    constexpr static bool requires_inner_sync
        = !std::is_same<typename block_load_keys::storage_type, detail::empty_storage_type>::value
          && !std::is_same<typename block_load_values::storage_type,
                           detail::empty_storage_type>::value;

    union storage_type
    {
        typename block_load_keys::storage_type   keys;
        typename block_load_values::storage_type values;
    };

    template<typename KeyIterator, typename ValueIterator>
    ROCPRIM_DEVICE
    void load_keys_values(KeyIterator        tile_keys,
                          ValueIterator      tile_values,
                          const bool         is_global_last_tile,
                          const unsigned int valid_in_global_last_tile,
                          KeyType (&keys)[ItemsPerThread],
                          AccumulatorType (&values)[ItemsPerThread],
                          storage_type& storage)
    {

        if(!is_global_last_tile)
        {
            block_load_keys{}.load(tile_keys, keys, storage.keys);
            if constexpr(requires_inner_sync)
            {
                ::rocprim::syncthreads();
            }
            block_load_values{}.load(tile_values, values, storage.values);
        }
        else
        {
            block_load_keys{}.load(tile_keys, keys, valid_in_global_last_tile, storage.keys);
            if constexpr(requires_inner_sync)
            {
                ::rocprim::syncthreads();
            }
            block_load_values{}.load(tile_values,
                                     values,
                                     valid_in_global_last_tile,
                                     storage.values);
        }
    }
};

template<typename KeyType, unsigned int BlockSize>
struct discontinuity_helper
{
    using block_discontinuity_type = block_discontinuity<KeyType, BlockSize>;
    using storage_type             = typename block_discontinuity_type::storage_type;

    template<typename KeyIterator, typename CompareFunction, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE
    void flag_heads(KeyIterator tile_keys,
                    const KeyType (&keys)[ItemsPerThread],
                    CompareFunction compare,
                    unsigned int (&head_flags)[ItemsPerThread],
                    const bool    is_global_first_tile,
                    const bool    is_global_last_tile,
                    const size_t  remaining,
                    storage_type& storage)
    {
        if(is_global_last_tile)
        {
            // If it's the last tile globally, the out-of-bound items should not be flagged.
            auto guarded_not_equal
                = ::rocprim::detail::guarded_inequality_wrapper<CompareFunction>(compare,
                                                                                 remaining);

            if(!is_global_first_tile)
            {
                const KeyType tile_predecessor = tile_keys[-1];
                block_discontinuity_type{}.flag_heads(head_flags,
                                                      tile_predecessor,
                                                      keys,
                                                      guarded_not_equal,
                                                      storage);
            }
            else
            {
                block_discontinuity_type{}.flag_heads(head_flags, keys, guarded_not_equal, storage);
            }
        }
        else
        {
            auto not_equal = rocprim::inequality_wrapper<CompareFunction>(compare);

            if(!is_global_first_tile)
            {
                const KeyType tile_predecessor = tile_keys[-1];
                block_discontinuity_type{}.flag_heads(head_flags,
                                                      tile_predecessor,
                                                      keys,
                                                      not_equal,
                                                      storage);
            }
            else
            {
                block_discontinuity_type{}.flag_heads(head_flags, keys, not_equal, storage);
            }
        }
    }
};

template<typename ValueType, unsigned int BlockSize, unsigned int ItemsPerThread>
struct scatter_helper
{
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<ValueType[BlockSize * ItemsPerThread]>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    template<typename ValueIterator, typename Flag, typename ValueFunction, typename IndexFunction>
    ROCPRIM_DEVICE
    void scatter(ValueIterator   tile_values,
                 ValueFunction&& values,
                 const Flag (&is_selected)[ItemsPerThread],
                 IndexFunction&&    block_indices,
                 const unsigned int selected_in_tile,
                 const unsigned int flat_thread_id,
                 storage_type&      storage)
    {
        // Check if all threads in the warp are selecting the same location (selected or rejected)
        uint8_t all_check = 3; // [true, true]
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            all_check &= is_selected[i] ? 1 /* [false, true] */ : 2 /* [true, false] */;
        }
        // all_check:
        //   0: mixed selection
        //   1: all selected
        //   2: all rejected
        //   3: impossible

        // If selected_in_tile >= BlockSize, use shared memory for storing scattered data
        if(selected_in_tile >= BlockSize)
        {
            auto& scatter_storage = storage.get();

            // If all threads in the warp are storing to the same location, we can skip the conditional
            if(all_check == 1)
            {
                // Perform the scatter directly for selected items
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < ItemsPerThread; ++i)
                {
                    scatter_storage[block_indices(i)] = values(i);
                }
            }
            else if(all_check == 2)
            {
                // If all threads reject, we can skip doing any work for the selected items
                // (nothing to store to the scatter storage)
            }
            else
            {
                // Otherwise, process each item individually
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < ItemsPerThread; ++i)
                {
                    if(is_selected[i])
                    {
                        scatter_storage[block_indices(i)] = values(i);
                    }
                }
            }

            ::rocprim::syncthreads();

            // Coalesced write from shared memory to global memory
            for(unsigned int i = flat_thread_id; i < selected_in_tile; i += BlockSize)
            {
                tile_values[i] = scatter_storage[i];
            }
        }
        else
        {
            // When selected_in_tile < BlockSize, store directly to global memory
            if(all_check == 1)
            {
                // Directly store all values since they are all selected
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < ItemsPerThread; ++i)
                {
                    tile_values[block_indices(i)] = values(i);
                }
            }
            else if(all_check == 2)
            {
                // If all are rejected, do nothing
            }
            else
            {
                // Process selected items
                ROCPRIM_UNROLL
                for(unsigned int i = 0; i < ItemsPerThread; ++i)
                {
                    if(is_selected[i])
                    {
                        tile_values[block_indices(i)] = values(i);
                    }
                }
            }
        }
    }
};

template<typename KeyType,
         typename AccumulatorType,
         unsigned int              BlockSize,
         unsigned int              ItemsPerThread,
         block_load_method         load_keys_method,
         block_load_method         load_values_method,
         block_scan_algorithm      scan_algorithm,
         lookback_scan_determinism Determinism>
class tile_helper
{
private:
    using load_type = reduce_by_key::load_helper<KeyType,
                                                 AccumulatorType,
                                                 BlockSize,
                                                 ItemsPerThread,
                                                 load_keys_method,
                                                 load_values_method>;

    using wrapped_type = reduce_by_key::wrapped_type_t<AccumulatorType>;

    using discontinuity_type = reduce_by_key::discontinuity_helper<KeyType, BlockSize>;
    using block_scan_type    = rocprim::block_scan<wrapped_type, BlockSize, scan_algorithm>;
    using prefix_op_factory  = detail::offset_lookback_scan_factory<wrapped_type>;

    using scatter_keys_type = reduce_by_key::scatter_helper<KeyType, BlockSize, ItemsPerThread>;
    using scatter_values_type
        = reduce_by_key::scatter_helper<AccumulatorType, BlockSize, ItemsPerThread>;

public:
    union storage_type
    {
        typename load_type::storage_type load;
        struct
        {
            typename discontinuity_type::storage_type flags;
            typename prefix_op_factory::storage_type  prefix;
            typename block_scan_type::storage_type    scan;
        } scan;
        typename scatter_keys_type::storage_type   scatter_keys;
        typename scatter_values_type::storage_type scatter_values;
    };

    template<typename KeyIterator,
             typename ValueIterator,
             typename UniqueIterator,
             typename ReductionIterator,
             typename UniqueCountIterator,
             typename CompareFunction,
             typename BinaryOp,
             typename LookbackScanState>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
        process_tile(const KeyIterator            tile_keys,
                     const ValueIterator          tile_values,
                     UniqueIterator               unique_keys,
                     ReductionIterator            reductions,
                     UniqueCountIterator          unique_count,
                     BinaryOp                     reduce_op,
                     const CompareFunction        compare,
                     LookbackScanState            scan_state,
                     const unsigned int           tile_id,
                     const std::size_t            starting_tile,
                     const std::size_t            total_number_of_tiles,
                     const std::size_t            size,
                     storage_type&                storage,
                     const std::size_t* const     global_head_count,
                     const AccumulatorType* const previous_accumulated)
    {

        static constexpr unsigned int items_per_tile = BlockSize * ItemsPerThread;

        auto wrapped_op = [&](const wrapped_type& lhs, const wrapped_type& rhs)
        {
            return wrapped_type{rocprim::get<0>(lhs) + rocprim::get<0>(rhs),
                                rocprim::get<0>(rhs) == 0
                                    ? reduce_op(rocprim::get<1>(lhs), rocprim::get<1>(rhs))
                                    : rocprim::get<1>(rhs)};
        };

        const std::size_t global_tile_id = starting_tile + tile_id;
        // first and last tiles across all launches
        const bool is_global_first_tile = global_tile_id == 0;
        const bool is_global_last_tile  = global_tile_id == total_number_of_tiles - 1;
        // first tile in this launch
        const bool is_first_tile = tile_id == 0;

        // When in last tile valid_in_global_last_tile = remaining
        const unsigned int valid_in_global_last_tile
            = static_cast<unsigned int>(size - ((total_number_of_tiles - 1) * items_per_tile));
        const size_t remaining
            = static_cast<size_t>(size - (size_t{global_tile_id} * items_per_tile));

        const unsigned int flat_thread_id = threadIdx.x;

        KeyType         keys[ItemsPerThread];
        AccumulatorType values[ItemsPerThread];

        load_type{}.load_keys_values(tile_keys,
                                     tile_values,
                                     is_global_last_tile,
                                     valid_in_global_last_tile,
                                     keys,
                                     values,
                                     storage.load);
        ::rocprim::syncthreads();

        unsigned int head_flags[ItemsPerThread];
        discontinuity_type{}.flag_heads(tile_keys,
                                        keys,
                                        compare,
                                        head_flags,
                                        is_global_first_tile,
                                        is_global_last_tile,
                                        remaining,
                                        storage.scan.flags);

        wrapped_type wrapped_values[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            rocprim::get<0>(wrapped_values[i]) = head_flags[i];
            rocprim::get<1>(wrapped_values[i]) = values[i];
        }

        unsigned int segment_heads_before   = 0;
        unsigned int segment_heads_in_block = 0;
        wrapped_type reduction;

        // This branch is taken for the first tile in each launch when
        // multiple launches occur due to large indices
        if(is_first_tile)
        {
            wrapped_type initial_value = ::rocprim::make_tuple(0u, values[0] /* dummy value */);

            // previous_accumulated is used to pass the accumulated value from the previous launch
            if(previous_accumulated != nullptr)
            {
                initial_value = ::rocprim::make_tuple(0u, *previous_accumulated);
            }

            block_scan_type{}.exclusive_scan(wrapped_values,
                                             wrapped_values,
                                             initial_value,
                                             reduction,
                                             storage.scan.scan,
                                             wrapped_op);
            // include initial_value in the block reduction
            reduction = wrapped_op(initial_value, reduction);

            if(flat_thread_id == 0)
            {
                scan_state.set_complete(0, reduction);
            }

            segment_heads_in_block = rocprim::get<0>(reduction);
        }
        else
        {
            auto lookback_op
                = detail::lookback_scan_prefix_op<wrapped_type,
                                                  decltype(wrapped_op),
                                                  decltype(scan_state),
                                                  Determinism>{tile_id, wrapped_op, scan_state};

            auto offset_lookback_op = prefix_op_factory::create(lookback_op, storage.scan.prefix);

            block_scan_type{}.exclusive_scan(wrapped_values,
                                             wrapped_values,
                                             storage.scan.scan,
                                             offset_lookback_op,
                                             wrapped_op);
            rocprim::syncthreads();

            segment_heads_before
                = rocprim::get<0>(prefix_op_factory::get_prefix(storage.scan.prefix));
            segment_heads_in_block
                = rocprim::get<0>(prefix_op_factory::get_reduction(storage.scan.prefix));
            reduction = wrapped_op(prefix_op_factory::get_prefix(storage.scan.prefix),
                                   prefix_op_factory::get_reduction(storage.scan.prefix));
        }
        rocprim::syncthreads();

        const std::size_t segment_heads_in_previous_launches
            = global_head_count != nullptr ? *global_head_count : 0u;

        // At this point each item that is flagged as segment head has
        // - The first key of the segment
        // - The number of segments before it (exclusive scan of head_flags)
        // - The reduction of the previous segment
        scatter_keys_type{}.scatter(
            unique_keys + segment_heads_in_previous_launches + segment_heads_before,
            [&keys](unsigned int i) { return keys[i]; },
            head_flags,
            [&](const unsigned int i)
            { return rocprim::get<0>(wrapped_values[i]) - segment_heads_before; },
            segment_heads_in_block,
            flat_thread_id,
            storage.scatter_keys);
        ::rocprim::syncthreads();

        // The first item in the global first tile does not have a reduction
        // The first out of bounds item in the global last tile has the reduction for the last segment
        const unsigned int reductions_in_block
            = segment_heads_in_block - (is_global_first_tile ? 1 : 0)
              + (is_global_last_tile && valid_in_global_last_tile != items_per_tile ? 1 : 0);

        if(is_global_first_tile && flat_thread_id == 0)
        {
            head_flags[0] = 0;
        }
        if(is_global_last_tile && flat_thread_id == valid_in_global_last_tile / ItemsPerThread)
        {
            head_flags[valid_in_global_last_tile - flat_thread_id * ItemsPerThread] = 1;
        }
        scatter_values_type{}.scatter(
            reductions + segment_heads_in_previous_launches + segment_heads_before
                - (!is_global_first_tile ? 1 : 0),
            [&wrapped_values](unsigned int i) { return rocprim::get<1>(wrapped_values[i]); },
            head_flags,
            [&, offset = segment_heads_before + (is_global_first_tile ? 1 : 0)](
                const unsigned int i) { return rocprim::get<0>(wrapped_values[i]) - offset; },
            reductions_in_block,
            flat_thread_id,
            storage.scatter_values);

        if(is_global_last_tile && flat_thread_id == BlockSize - 1)
        {
            const std::size_t total_segment_heads = segment_heads_in_previous_launches
                                                    + segment_heads_before + segment_heads_in_block;
            *unique_count = total_segment_heads;
            if(valid_in_global_last_tile == items_per_tile)
            {
                reductions[total_segment_heads - 1] = rocprim::get<1>(reduction);
            }
        }
    }
};

template<typename ArchConfig,
         lookback_scan_determinism Determinism,
         typename AccumulatorType,
         typename KeyIterator,
         typename ValueIterator,
         typename UniqueIterator,
         typename ReductionIterator,
         typename UniqueCountIterator,
         typename CompareFunction,
         typename BinaryOp,
         typename LookbackScanState>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE auto kernel_impl(KeyIterator,
                                                     ValueIterator,
                                                     const UniqueIterator,
                                                     const ReductionIterator,
                                                     const UniqueCountIterator,
                                                     const BinaryOp,
                                                     const CompareFunction,
                                                     const LookbackScanState,
                                                     const std::size_t,
                                                     const std::size_t,
                                                     const std::size_t,
                                                     const std::size_t* const,
                                                     const AccumulatorType* const)
    -> std::enable_if_t<!is_lookback_kernel_runnable<LookbackScanState>()>
{
    // No need to build the kernel with sleep on a device that does not require it
}

template<typename ArchConfig,
         lookback_scan_determinism Determinism,
         typename AccumulatorType,
         typename KeyIterator,
         typename ValueIterator,
         typename UniqueIterator,
         typename ReductionIterator,
         typename UniqueCountIterator,
         typename CompareFunction,
         typename BinaryOp,
         typename LookbackScanState>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE auto
    kernel_impl(KeyIterator                    keys_input,
                ValueIterator                  values_input,
                const UniqueIterator           unique_keys,
                const ReductionIterator        reductions,
                const UniqueCountIterator      unique_count,
                const BinaryOp                 reduce_op,
                const CompareFunction          compare,
                const LookbackScanState        scan_state,
                const std::size_t              starting_block,
                const std::size_t              total_number_of_blocks,
                const std::size_t              size,
                const std::size_t* const       global_head_count,
                const AccumulatorType* const   previous_accumulated)
        -> std::enable_if_t<is_lookback_kernel_runnable<LookbackScanState>()>
{
    static constexpr reduce_by_key_config_params params = ArchConfig::params;

    static constexpr unsigned int         block_size       = params.kernel_config.block_size;
    static constexpr unsigned int         items_per_thread = params.kernel_config.items_per_thread;
    static constexpr block_load_method    load_keys_method = params.load_keys_method;
    static constexpr block_load_method    load_values_method = params.load_values_method;
    static constexpr block_scan_algorithm scan_algorithm     = params.scan_algorithm;
    static constexpr unsigned int         items_per_block    = block_size * items_per_thread;

    using key_type = ::rocprim::detail::value_type_t<KeyIterator>;

    using tile_processor = tile_helper<key_type,
                                       AccumulatorType,
                                       block_size,
                                       items_per_thread,
                                       load_keys_method,
                                       load_values_method,
                                       scan_algorithm,
                                       Determinism>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename tile_processor::storage_type tile;
    } storage;

    const unsigned int  block_id     = rocprim::flat_block_id<block_size, 1, 1>();
    const unsigned int  block_offset = block_id * items_per_block;
    const KeyIterator   block_keys   = keys_input + block_offset;
    const ValueIterator block_values = values_input + block_offset;

    tile_processor{}.process_tile(block_keys,
                                  block_values,
                                  unique_keys,
                                  reductions,
                                  unique_count,
                                  reduce_op,
                                  compare,
                                  scan_state,
                                  block_id,
                                  starting_block,
                                  total_number_of_blocks,
                                  size,
                                  storage.tile,
                                  global_head_count,
                                  previous_accumulated);
}

} // namespace reduce_by_key

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_BY_KEY_HPP_
