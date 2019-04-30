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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_PARTITION_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_PARTITION_HPP_

#include <type_traits>
#include <iterator>

#include "../../detail/various.hpp"
#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"
#include "../../block/block_scan.hpp"

#include "device_scan_lookback.hpp"
#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class T, class LookbackScanState>
class offset_lookback_scan_prefix_op : public lookback_scan_prefix_op<T, ::rocprim::plus<T>, LookbackScanState>
{
    using base_type = lookback_scan_prefix_op<T, ::rocprim::plus<T>, LookbackScanState>;
    using binary_op_type = ::rocprim::plus<T>;
public:

    struct storage_type
    {
        T block_reduction;
        T exclusive_prefix;
    };

    ROCPRIM_DEVICE inline
    offset_lookback_scan_prefix_op(unsigned int block_id,
                                   LookbackScanState &state,
                                   storage_type& storage)
        : base_type(block_id, binary_op_type(), state), storage_(storage)
    {
    }

    ROCPRIM_DEVICE inline
    ~offset_lookback_scan_prefix_op() = default;

    ROCPRIM_DEVICE inline
    T operator()(T reduction)
    {
        auto prefix = base_type::operator()(reduction);
        if(::rocprim::lane_id() == 0)
        {
            storage_.block_reduction = reduction;
            storage_.exclusive_prefix = prefix;
        }
        return prefix;
    }

    ROCPRIM_DEVICE inline
    T get_reduction() const
    {
        return storage_.block_reduction;
    }

    ROCPRIM_DEVICE inline
    T get_exclusive_prefix() const
    {
        return storage_.exclusive_prefix;
    }

private:
    storage_type& storage_;
};

enum class select_method
{
    flag = 0,
    predicate = 1,
    unique = 2
};

template<
    select_method SelectMethod,
    unsigned int BlockSize,
    class BlockLoadFlagsType,
    class BlockDiscontinuityType,
    class InputIterator,
    class FlagIterator,
    class ValueType,
    unsigned int ItemsPerThread,
    class UnaryPredicate,
    class InequalityOp,
    class StorageType
>
ROCPRIM_DEVICE inline
auto partition_block_load_flags(InputIterator /* block_predecessor */,
                                FlagIterator block_flags,
                                ValueType (&/* values */)[ItemsPerThread],
                                bool (&is_selected)[ItemsPerThread],
                                UnaryPredicate /* predicate */,
                                InequalityOp /* inequality_op */,
                                StorageType& storage,
                                const unsigned int /* block_id */,
                                const unsigned int /* block_thread_id */,
                                const bool is_last_block,
                                const unsigned int valid_in_last_block)
    -> typename std::enable_if<SelectMethod == select_method::flag>::type
{
    if(is_last_block) // last block
    {
        BlockLoadFlagsType()
            .load(
                block_flags,
                is_selected,
                valid_in_last_block,
                false,
                storage.load_flags
            );
    }
    else
    {
        BlockLoadFlagsType()
            .load(
                block_flags,
                is_selected,
                storage.load_flags
            );
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory
}

template<
    select_method SelectMethod,
    unsigned int BlockSize,
    class BlockLoadFlagsType,
    class BlockDiscontinuityType,
    class InputIterator,
    class FlagIterator,
    class ValueType,
    unsigned int ItemsPerThread,
    class UnaryPredicate,
    class InequalityOp,
    class StorageType
>
ROCPRIM_DEVICE inline
auto partition_block_load_flags(InputIterator /* block_predecessor */,
                                FlagIterator /* block_flags */,
                                ValueType (&values)[ItemsPerThread],
                                bool (&is_selected)[ItemsPerThread],
                                UnaryPredicate predicate,
                                InequalityOp /* inequality_op */,
                                StorageType& /* storage */,
                                const unsigned int /* block_id */,
                                const unsigned int block_thread_id,
                                const bool is_last_block,
                                const unsigned int valid_in_last_block)
    -> typename std::enable_if<SelectMethod == select_method::predicate>::type
{
    if(is_last_block) // last block
    {
        const auto offset = block_thread_id * ItemsPerThread;
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if((offset + i) < valid_in_last_block)
            {
                is_selected[i] = predicate(values[i]);
            }
            else
            {
                is_selected[i] = false;
            }
        }
    }
    else
    {
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            is_selected[i] = predicate(values[i]);
        }
    }
}

template<
    select_method SelectMethod,
    unsigned int BlockSize,
    class BlockLoadFlagsType,
    class BlockDiscontinuityType,
    class InputIterator,
    class FlagIterator,
    class ValueType,
    unsigned int ItemsPerThread,
    class UnaryPredicate,
    class InequalityOp,
    class StorageType
>
ROCPRIM_DEVICE inline
auto partition_block_load_flags(InputIterator block_predecessor,
                                FlagIterator /* block_flags */,
                                ValueType (&values)[ItemsPerThread],
                                bool (&is_selected)[ItemsPerThread],
                                UnaryPredicate /* predicate */,
                                InequalityOp inequality_op,
                                StorageType& storage,
                                const unsigned int block_id,
                                const unsigned int block_thread_id,
                                const bool is_last_block,
                                const unsigned int valid_in_last_block)
    -> typename std::enable_if<SelectMethod == select_method::unique>::type
{
    if(block_id > 0)
    {
        const ValueType predecessor = *block_predecessor;
        BlockDiscontinuityType()
            .flag_heads(
                is_selected,
                predecessor,
                values,
                inequality_op,
                storage.discontinuity_values
            );
    }
    else
    {
        BlockDiscontinuityType()
            .flag_heads(
                is_selected,
                values,
                inequality_op,
                storage.discontinuity_values
            );
    }

    // Set is_selected for invalid items to false
    if(is_last_block)
    {
        const auto offset = block_thread_id * ItemsPerThread;
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if((offset + i) >= valid_in_last_block)
            {
                is_selected[i] = false;
            }
        }
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory
}

template<
    bool OnlySelected,
    unsigned int BlockSize,
    class ValueType,
    unsigned int ItemsPerThread,
    class OffsetType,
    class OutputIterator,
    class ScatterStorageType
>
ROCPRIM_DEVICE inline
auto partition_scatter(ValueType (&values)[ItemsPerThread],
                       bool (&is_selected)[ItemsPerThread],
                       OffsetType (&output_indices)[ItemsPerThread],
                       OutputIterator output,
                       const size_t size,
                       const OffsetType selected_prefix,
                       const OffsetType selected_in_block,
                       ScatterStorageType& storage,
                       const unsigned int flat_block_id,
                       const unsigned int flat_block_thread_id,
                       const bool is_last_block,
                       const unsigned int valid_in_last_block)
    -> typename std::enable_if<!OnlySelected>::type
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    // Scatter selected/rejected values to shared memory
    auto scatter_storage = storage.get();
    #pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        unsigned int item_index = (flat_block_thread_id * ItemsPerThread) + i;
        unsigned int selected_item_index = output_indices[i] - selected_prefix;
        unsigned int rejected_item_index = (item_index - selected_item_index) + selected_in_block;
        // index of item in scatter_storage
        unsigned int scatter_index = is_selected[i] ? selected_item_index : rejected_item_index;
        scatter_storage[scatter_index] = values[i];
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    #pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        unsigned int item_index = (i * BlockSize) + flat_block_thread_id;
        unsigned int selected_item_index = item_index;
        unsigned int rejected_item_index = item_index - selected_in_block;
        // number of values rejected in previous blocks
        unsigned int rejected_prefix = (flat_block_id * items_per_block) - selected_prefix;
        // destination index of item scatter_storage[item_index] in output
        OffsetType scatter_index = item_index < selected_in_block
            ? selected_prefix + selected_item_index
            : size - (rejected_prefix + rejected_item_index + 1);

        // last block can store only valid_in_last_block items
        if(!is_last_block || item_index < valid_in_last_block)
        {
            output[scatter_index] = scatter_storage[item_index];
        }
    }
}

template<
    bool OnlySelected,
    unsigned int BlockSize,
    class ValueType,
    unsigned int ItemsPerThread,
    class OffsetType,
    class OutputIterator,
    class ScatterStorageType
>
ROCPRIM_DEVICE inline
auto partition_scatter(ValueType (&values)[ItemsPerThread],
                       bool (&is_selected)[ItemsPerThread],
                       OffsetType (&output_indices)[ItemsPerThread],
                       OutputIterator output,
                       const size_t size,
                       const OffsetType selected_prefix,
                       const OffsetType selected_in_block,
                       ScatterStorageType& storage,
                       const unsigned int flat_block_id,
                       const unsigned int flat_block_thread_id,
                       const bool is_last_block,
                       const unsigned int valid_in_last_block)
    -> typename std::enable_if<OnlySelected>::type
{
    (void) size;
    (void) storage;
    (void) flat_block_id;
    (void) flat_block_thread_id;
    (void) valid_in_last_block;

    if(selected_in_block > BlockSize)
    {
        // Scatter selected values to shared memory
        auto scatter_storage = storage.get();
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            unsigned int scatter_index = output_indices[i] - selected_prefix;
            if(is_selected[i])
            {
                scatter_storage[scatter_index] = values[i];
            }
        }
        ::rocprim::syncthreads(); // sync threads to reuse shared memory

        // Coalesced write from shared memory to global memory
        for(unsigned int i = flat_block_thread_id; i < selected_in_block; i += BlockSize)
        {
            output[selected_prefix + i] = scatter_storage[i];
        }
    }
    else
    {
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(!is_last_block || output_indices[i] < (selected_prefix + selected_in_block))
            {
                if(is_selected[i])
                {
                    output[output_indices[i]] = values[i];
                }
            }
        }
    }
}

template<
    select_method SelectMethod,
    bool OnlySelected,
    class Config,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class UnaryPredicate,
    class InequalityOp,
    class OffsetLookbackScanState
>
ROCPRIM_DEVICE inline
void partition_kernel_impl(InputIterator input,
                           FlagIterator flags,
                           OutputIterator output,
                           SelectedCountOutputIterator selected_count_output,
                           const size_t size,
                           UnaryPredicate predicate,
                           InequalityOp inequality_op,
                           OffsetLookbackScanState offset_scan_state,
                           const unsigned int number_of_blocks,
                           ordered_block_id<unsigned int> ordered_bid)
{
    constexpr auto block_size = Config::block_size;
    constexpr auto items_per_thread = Config::items_per_thread;
    constexpr unsigned int items_per_block = block_size * items_per_thread;

    using offset_type = typename OffsetLookbackScanState::value_type;
    using value_type = typename std::iterator_traits<InputIterator>::value_type;

    // Block primitives
    using block_load_value_type = ::rocprim::block_load<
        value_type, block_size, items_per_thread,
        Config::value_block_load_method
    >;
    using block_load_flag_type = ::rocprim::block_load<
        bool, block_size, items_per_thread,
        Config::value_block_load_method
    >;
    using block_scan_offset_type = ::rocprim::block_scan<
        offset_type, block_size,
        Config::block_scan_method
    >;
    using block_discontinuity_value_type = ::rocprim::block_discontinuity<
        value_type, block_size
    >;
    using order_bid_type = ordered_block_id<unsigned int>;

    // Offset prefix operation type
    using offset_scan_prefix_op_type = offset_lookback_scan_prefix_op<
        offset_type, OffsetLookbackScanState
    >;

    // Memory required for 2-phase scatter
    using exchange_storage_type = value_type[items_per_block];
    using raw_exchange_storage_type = typename detail::raw_storage<exchange_storage_type>;

    ROCPRIM_SHARED_MEMORY struct
    {
        typename order_bid_type::storage_type ordered_bid;
        union
        {
            raw_exchange_storage_type exchange_values;
            typename block_load_value_type::storage_type load_values;
            typename block_load_flag_type::storage_type load_flags;
            typename block_discontinuity_value_type::storage_type discontinuity_values;
            typename block_scan_offset_type::storage_type scan_offsets;
        };
    } storage;

    const auto flat_block_thread_id = ::rocprim::flat_block_thread_id();
    const auto flat_block_id = ordered_bid.get(flat_block_thread_id, storage.ordered_bid);
    const unsigned int block_offset = flat_block_id * items_per_block;
    const auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);

    value_type values[items_per_thread];
    bool is_selected[items_per_thread];
    offset_type output_indices[items_per_thread];

    // Load input values into values
    bool is_last_block = flat_block_id == (number_of_blocks - 1);
    if(is_last_block) // last block
    {
        block_load_value_type()
            .load(
                input + block_offset,
                values,
                valid_in_last_block,
                storage.load_values
            );
    }
    else
    {
        block_load_value_type()
            .load(
                input + block_offset,
                values,
                storage.load_values
            );
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    // Load selection flags into is_selected, generate them using
    // input value and selection predicate, or generate them using
    // block_discontinuity primitive
    partition_block_load_flags<
        SelectMethod, block_size,
        block_load_flag_type, block_discontinuity_value_type
    >(
        input + block_offset - 1,
        flags + block_offset,
        values,
        is_selected,
        predicate,
        inequality_op,
        storage,
        flat_block_id,
        flat_block_thread_id,
        is_last_block,
        valid_in_last_block
    );

    // Convert true/false is_selected flags to 0s and 1s
    #pragma unroll
    for(unsigned int i = 0; i < items_per_thread; i++)
    {
        output_indices[i] = is_selected[i] ? 1 : 0;
    }

    // Number of selected values in previous blocks
    offset_type selected_prefix = 0;
    // Number of selected values in this block
    offset_type selected_in_block = 0;

    // Calculate number of selected values in block and their indices
    if(flat_block_id == 0)
    {
        block_scan_offset_type()
            .exclusive_scan(
                output_indices,
                output_indices,
                offset_type(0), /** initial value */
                selected_in_block,
                storage.scan_offsets,
                ::rocprim::plus<offset_type>()
            );
        if(flat_block_thread_id == 0)
        {
            offset_scan_state.set_complete(flat_block_id, selected_in_block);
        }
        ::rocprim::syncthreads(); // sync threads to reuse shared memory
    }
    // Workaround: Fiji (gfx803) crashes with "Memory access fault by GPU node" on HCC 1.3.18482 (ROCm 2.0)
    // Instead of just `} else {` we use `} syncthreads(); if() {`, because the else-branch can be executed
    // for some unknown reason and 0-th block reads incorrect addresses in lookback_scan_prefix_op::get_prefix.
    ::rocprim::syncthreads();
    if(flat_block_id > 0)
    // end of the workaround
    {
        ROCPRIM_SHARED_MEMORY typename offset_scan_prefix_op_type::storage_type storage_prefix_op;
        auto prefix_op = offset_scan_prefix_op_type(
            flat_block_id,
            offset_scan_state,
            storage_prefix_op
        );
        block_scan_offset_type()
            .exclusive_scan(
                output_indices,
                output_indices,
                storage.scan_offsets,
                prefix_op,
                ::rocprim::plus<offset_type>()
            );
        ::rocprim::syncthreads(); // sync threads to reuse shared memory

        selected_in_block = prefix_op.get_reduction();
        selected_prefix = prefix_op.get_exclusive_prefix();
    }

    // Scatter selected and rejected values
    partition_scatter<OnlySelected, block_size>(
        values, is_selected, output_indices, output, size,
        selected_prefix, selected_in_block, storage.exchange_values,
        flat_block_id, flat_block_thread_id,
        is_last_block, valid_in_last_block
    );

    // Last block in grid stores number of selected values
    if(flat_block_id == (number_of_blocks - 1) && flat_block_thread_id == 0)
    {
        selected_count_output[0] = selected_prefix + selected_in_block;
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_PARTITION_HPP_
