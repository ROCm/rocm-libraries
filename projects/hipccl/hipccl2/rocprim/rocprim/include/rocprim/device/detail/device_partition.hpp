// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../../block/block_discontinuity.hpp"

#include "lookback_scan_state.hpp"
#include "rocprim/type_traits.hpp"
#include "rocprim/types/tuple.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum class select_method
{
    flag = 0,
    predicate = 1,
    unique = 2
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

template<select_method SelectMethod,
         unsigned int  BlockSize,
         class BlockLoadFlagsType,
         class BlockDiscontinuityType,
         class InputIterator,
         class FlagIterator,
         class ValueType,
         unsigned int ItemsPerThread,
         class UnaryPredicate,
         class InequalityOp,
         class StorageType>
ROCPRIM_DEVICE ROCPRIM_INLINE auto
    partition_block_load_flags(InputIterator /* block_predecessor */,
                               FlagIterator block_flags,
                               ValueType (&/* values */)[ItemsPerThread],
                               bool (&is_selected)[ItemsPerThread],
                               UnaryPredicate /* predicate */,
                               InequalityOp /* inequality_op */,
                               StorageType& storage,
                               const bool /* is_first_block */,
                               const unsigned int /* block_thread_id */,
                               const bool         is_global_last_block,
                               const unsigned int valid_in_global_last_block) ->
    typename std::enable_if<SelectMethod == select_method::flag>::type
{
    if(is_global_last_block) // last block
    {
        BlockLoadFlagsType().load(block_flags,
                                  is_selected,
                                  valid_in_global_last_block,
                                  false,
                                  storage.load_flags);
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

template<select_method SelectMethod,
         unsigned int  BlockSize,
         class BlockLoadFlagsType,
         class BlockDiscontinuityType,
         class InputIterator,
         class FlagIterator,
         class ValueType,
         unsigned int ItemsPerThread,
         class UnaryPredicate,
         class InequalityOp,
         class StorageType>
ROCPRIM_DEVICE ROCPRIM_INLINE auto
    partition_block_load_flags(InputIterator /* block_predecessor */,
                               FlagIterator /* block_flags */,
                               ValueType (&values)[ItemsPerThread],
                               bool (&is_selected)[ItemsPerThread],
                               UnaryPredicate predicate,
                               InequalityOp /* inequality_op */,
                               StorageType& /* storage */,
                               const bool /* is_first_block */,
                               const unsigned int block_thread_id,
                               const bool         is_global_last_block,
                               const unsigned int valid_in_global_last_block) ->
    typename std::enable_if<SelectMethod == select_method::predicate>::type
{
    if(is_global_last_block) // last block
    {
        const auto offset = block_thread_id * ItemsPerThread;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if((offset + i) < valid_in_global_last_block)
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
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            is_selected[i] = predicate(values[i]);
        }
    }
}

// This wrapper processes only part of items and flags (valid_count - 1)th item (for tails)
// and (valid_count)th item (for heads), all items after valid_count are unflagged.
template<class InequalityOp>
struct guarded_inequality_op
{
    InequalityOp inequality_op;
    unsigned int valid_count;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    guarded_inequality_op(InequalityOp inequality_op, unsigned int valid_count)
        : inequality_op(inequality_op), valid_count(valid_count)
    {}

    template<class T, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const T& a, const U& b, unsigned int b_index)
    {
        return (b_index < valid_count && inequality_op(a, b));
    }
};

template<select_method SelectMethod,
         unsigned int  BlockSize,
         class BlockLoadFlagsType,
         class BlockDiscontinuityType,
         class InputIterator,
         class FlagIterator,
         class ValueType,
         unsigned int ItemsPerThread,
         class UnaryPredicate,
         class InequalityOp,
         class StorageType>
ROCPRIM_DEVICE ROCPRIM_INLINE auto
    partition_block_load_flags(InputIterator block_predecessor,
                               FlagIterator /* block_flags */,
                               ValueType (&values)[ItemsPerThread],
                               bool (&is_selected)[ItemsPerThread],
                               UnaryPredicate /* predicate */,
                               InequalityOp       inequality_op,
                               StorageType&       storage,
                               const bool         is_first_block,
                               const unsigned int block_thread_id,
                               const bool         is_global_last_block,
                               const unsigned int valid_in_global_last_block) ->
    typename std::enable_if<SelectMethod == select_method::unique>::type
{
    if(is_first_block)
    {
        if(is_global_last_block)
        {
            BlockDiscontinuityType().flag_heads(
                is_selected,
                values,
                guarded_inequality_op<InequalityOp>(inequality_op, valid_in_global_last_block),
                storage.discontinuity_values);
        }
        else
        {
            BlockDiscontinuityType().flag_heads(is_selected,
                                                values,
                                                inequality_op,
                                                storage.discontinuity_values);
        }
    }
    else
    {
        const ValueType predecessor = block_predecessor[0];
        if(is_global_last_block)
        {
            BlockDiscontinuityType().flag_heads(
                is_selected,
                predecessor,
                values,
                guarded_inequality_op<InequalityOp>(inequality_op, valid_in_global_last_block),
                storage.discontinuity_values);
        }
        else
        {
            BlockDiscontinuityType().flag_heads(is_selected,
                                                predecessor,
                                                values,
                                                inequality_op,
                                                storage.discontinuity_values);
        }
    }


    // Set is_selected for invalid items to false
    if(is_global_last_block)
    {
        const auto offset = block_thread_id * ItemsPerThread;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if((offset + i) >= valid_in_global_last_block)
            {
                is_selected[i] = false;
            }
        }
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory
}

template<select_method SelectMethod,
         unsigned int  BlockSize,
         class BlockLoadFlagsType,
         class BlockDiscontinuityType,
         class InputIterator,
         class FlagIterator,
         class ValueType,
         unsigned int ItemsPerThread,
         class FirstUnaryPredicate,
         class SecondUnaryPredicate,
         class InequalityOp,
         class StorageType>
ROCPRIM_DEVICE ROCPRIM_INLINE void
    partition_block_load_flags(InputIterator /*block_predecessor*/,
                               FlagIterator /* block_flags */,
                               ValueType (&values)[ItemsPerThread],
                               bool (&is_selected)[2][ItemsPerThread],
                               FirstUnaryPredicate  select_first_part_op,
                               SecondUnaryPredicate select_second_part_op,
                               InequalityOp /*inequality_op*/,
                               StorageType& /*storage*/,
                               const unsigned int /*block_id*/,
                               const unsigned int block_thread_id,
                               const bool         is_global_last_block,
                               const unsigned int valid_in_global_last_block)
{
    if(is_global_last_block)
    {
        const auto offset = block_thread_id * ItemsPerThread;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if((offset + i) < valid_in_global_last_block)
            {
                is_selected[0][i] = select_first_part_op(values[i]);
                is_selected[1][i] = !is_selected[0][i] && select_second_part_op(values[i]);
            }
            else
            {
                is_selected[0][i] = false;
                is_selected[1][i] = false;
            }
        }
    }
    else
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            is_selected[0][i] = select_first_part_op(values[i]);
            is_selected[1][i] = !is_selected[0][i] && select_second_part_op(values[i]);
        }
    }
}

// two-way partition into one iterator
template<bool         OnlySelected,
         unsigned int BlockSize,
         class ValueType,
         unsigned int ItemsPerThread,
         class OffsetType,
         class SelectType,
         class ScatterStorageType>
ROCPRIM_DEVICE ROCPRIM_INLINE auto
    partition_scatter(ValueType (&values)[ItemsPerThread],
                      bool (&is_selected)[ItemsPerThread],
                      OffsetType (&output_indices)[ItemsPerThread],
                      tuple<SelectType, ::rocprim::empty_type*> output,
                      const size_t                              total_size,
                      const OffsetType                          selected_prefix,
                      const OffsetType                          selected_in_block,
                      ScatterStorageType&                       storage,
                      const unsigned int                        flat_block_id,
                      const unsigned int                        flat_block_thread_id,
                      const bool                                is_global_last_block,
                      const unsigned int                        valid_in_global_last_block,
                      size_t (&prev_selected_count_values)[1],
                      size_t prev_processed) -> typename std::enable_if<!OnlySelected>::type
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    // Scatter selected/rejected values to shared memory
    auto scatter_storage = storage.get();
    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        unsigned int item_index          = (flat_block_thread_id * ItemsPerThread) + i;
        unsigned int selected_item_index = output_indices[i] - selected_prefix;
        unsigned int rejected_item_index = (item_index - selected_item_index) + selected_in_block;
        // index of item in scatter_storage
        unsigned int scatter_index     = is_selected[i] ? selected_item_index : rejected_item_index;
        scatter_storage[scatter_index] = values[i];
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    ValueType reloaded_values[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        const unsigned int item_index = i * BlockSize + flat_block_thread_id;
        reloaded_values[i]            = scatter_storage[item_index];
    }

    const auto calculate_scatter_index = [=](const unsigned int item_index) -> size_t
    {
        const size_t selected_output_index = prev_selected_count_values[0] + selected_prefix;
        const size_t rejected_output_index = total_size + selected_output_index - prev_processed
                                             - flat_block_id * items_per_block + selected_in_block
                                             - 1;
        return item_index < selected_in_block ? selected_output_index + item_index
                                              : rejected_output_index - item_index;
    };
    if(is_global_last_block)
    {
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const unsigned int item_index = i * BlockSize + flat_block_thread_id;
            if(item_index < valid_in_global_last_block)
            {
                get<0>(output)[calculate_scatter_index(item_index)] = reloaded_values[i];
            }
        }
    }
    else
    {
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const unsigned int item_index = i * BlockSize + flat_block_thread_id;
            get<0>(output)[calculate_scatter_index(item_index)] = reloaded_values[i];
        }
    }
}

// two-way partition into two iterators
template<bool         OnlySelected,
         unsigned int BlockSize,
         class ValueType,
         unsigned int ItemsPerThread,
         class OffsetType,
         class SelectType,
         class RejectType,
         class ScatterStorageType>
ROCPRIM_DEVICE ROCPRIM_INLINE auto partition_scatter(ValueType (&values)[ItemsPerThread],
                                                     bool (&is_selected)[ItemsPerThread],
                                                     OffsetType (&output_indices)[ItemsPerThread],
                                                     tuple<SelectType, RejectType> output,
                                                     const size_t /*total_size*/,
                                                     const OffsetType    selected_prefix,
                                                     const OffsetType    selected_in_block,
                                                     ScatterStorageType& storage,
                                                     const unsigned int  flat_block_id,
                                                     const unsigned int  flat_block_thread_id,
                                                     const bool          is_global_last_block,
                                                     const unsigned int  valid_in_global_last_block,
                                                     size_t (&prev_selected_count_values)[1],
                                                     size_t prev_processed) ->
    typename std::enable_if<!OnlySelected>::type
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    // Scatter selected/rejected values to shared memory
    auto scatter_storage = storage.get();
    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        unsigned int item_index          = (flat_block_thread_id * ItemsPerThread) + i;
        unsigned int selected_item_index = output_indices[i] - selected_prefix;
        unsigned int rejected_item_index = (item_index - selected_item_index) + selected_in_block;
        // index of item in scatter_storage
        unsigned int scatter_index     = is_selected[i] ? selected_item_index : rejected_item_index;
        scatter_storage[scatter_index] = values[i];
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    ValueType reloaded_values[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        const unsigned int item_index = i * BlockSize + flat_block_thread_id;
        reloaded_values[i]            = scatter_storage[item_index];
    }

    auto save_to_output = [=](const unsigned int item_index, const unsigned int i)
    {
        const size_t selected_output_index = prev_selected_count_values[0] + selected_prefix;
        const size_t rejected_output_index = prev_processed + flat_block_id * items_per_block
                                             - selected_output_index - selected_in_block;

        if(item_index < selected_in_block)
        {
            get<0>(output)[selected_output_index + item_index] = reloaded_values[i];
        }
        else
        {
            get<1>(output)[rejected_output_index + item_index] = reloaded_values[i];
        }
    };

    if(is_global_last_block)
    {
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const unsigned int item_index = i * BlockSize + flat_block_thread_id;
            if(item_index < valid_in_global_last_block)
            {
                save_to_output(item_index, i);
            }
        }
    }
    else
    {
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const unsigned int item_index = i * BlockSize + flat_block_thread_id;
            save_to_output(item_index, i);
        }
    }
}

// two-way partition, selection only
template<bool         OnlySelected,
         unsigned int BlockSize,
         class ValueType,
         unsigned int ItemsPerThread,
         class OffsetType,
         class SelectType,
         class RejectType,
         class ScatterStorageType>
ROCPRIM_DEVICE ROCPRIM_INLINE auto
    partition_scatter(ValueType (&values)[ItemsPerThread],
                      bool (&is_selected)[ItemsPerThread],
                      OffsetType (&output_indices)[ItemsPerThread],
                      tuple<SelectType, RejectType> output,
                      const size_t /*total_size*/,
                      const OffsetType    selected_prefix,
                      const OffsetType    selected_in_block,
                      ScatterStorageType& storage,
                      const unsigned int /*flat_block_id*/,
                      const unsigned int flat_block_thread_id,
                      const bool         is_global_last_block,
                      const unsigned int /*valid_in_global_last_block*/,
                      size_t (&prev_selected_count_values)[1],
                      size_t /*prev_processed*/) -> typename std::enable_if<OnlySelected>::type
{
    if(selected_in_block > BlockSize)
    {
        // Scatter selected values to shared memory
        auto scatter_storage = storage.get();
        ROCPRIM_UNROLL
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
            get<0>(output)[prev_selected_count_values[0] + selected_prefix + i]
                = scatter_storage[i];
        }
    }
    else
    {
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(!is_global_last_block || output_indices[i] < (selected_prefix + selected_in_block))
            {
                if(is_selected[i])
                {
                    get<0>(output)[prev_selected_count_values[0] + output_indices[i]] = values[i];
                }
            }
        }
    }
}

// three-way partition
template<bool         OnlySelected,
         unsigned int BlockSize,
         class ValueType,
         unsigned int ItemsPerThread,
         class OffsetType,
         class OutputType,
         class ScatterStorageType>
ROCPRIM_DEVICE ROCPRIM_INLINE void partition_scatter(ValueType (&values)[ItemsPerThread],
                                                     bool (&is_selected)[2][ItemsPerThread],
                                                     OffsetType (&output_indices)[ItemsPerThread],
                                                     OutputType output,
                                                     const size_t /*total_size*/,
                                                     const OffsetType    selected_prefix,
                                                     const OffsetType    selected_in_block,
                                                     ScatterStorageType& storage,
                                                     const unsigned int  flat_block_id,
                                                     const unsigned int  flat_block_thread_id,
                                                     const bool          is_global_last_block,
                                                     const unsigned int  valid_in_global_last_block,
                                                     size_t (&prev_selected_count_values)[2],
                                                     size_t prev_processed)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    auto                   scatter_storage = storage.get();
    const size_t first_selected_prefix     = prev_selected_count_values[0] + selected_prefix.x;
    const size_t second_selected_prefix
        = prev_selected_count_values[1] - selected_in_block.x + selected_prefix.y;
    const size_t unselected_prefix = prev_processed - first_selected_prefix - second_selected_prefix
                                     + items_per_block * flat_block_id - 2 * selected_in_block.x
                                     - selected_in_block.y;

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        const unsigned int first_selected_item_index = output_indices[i].x - selected_prefix.x;
        const unsigned int second_selected_item_index = output_indices[i].y - selected_prefix.y
            + selected_in_block.x;
        unsigned int scatter_index{};

        if(is_selected[0][i])
        {
            scatter_index = first_selected_item_index;
        }
        else if(is_selected[1][i])
        {
            scatter_index = second_selected_item_index;
        }
        else
        {
            const unsigned int item_index = (flat_block_thread_id * ItemsPerThread) + i;
            const unsigned int unselected_item_index = (item_index - first_selected_item_index - second_selected_item_index)
                + 2*selected_in_block.x + selected_in_block.y;
            scatter_index = unselected_item_index;
        }
        scatter_storage[scatter_index] = values[i];
    }
    ::rocprim::syncthreads();

    auto save_to_output = [=](const unsigned int item_index) mutable
    {
        if(item_index < selected_in_block.x)
        {
            const size_t first_selected_index    = first_selected_prefix + item_index;
            get<0>(output)[first_selected_index] = scatter_storage[item_index];
        }
        else if(item_index < selected_in_block.x + selected_in_block.y)
        {
            const size_t second_selected_index    = second_selected_prefix + item_index;
            get<1>(output)[second_selected_index] = scatter_storage[item_index];
        }
        else
        {
            const size_t unselected_index    = unselected_prefix + item_index;
            get<2>(output)[unselected_index] = scatter_storage[item_index];
        }
    };

    if(is_global_last_block)
    {
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const unsigned int item_index = (i * BlockSize) + flat_block_thread_id;
            if(item_index < valid_in_global_last_block)
            {
                save_to_output(item_index);
            }
        }
    }
    else
    {
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const unsigned int item_index = (i * BlockSize) + flat_block_thread_id;
            save_to_output(item_index);
        }
    }
}

template<
    unsigned int items_per_thread,
    class offset_type
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void convert_selected_to_indices(offset_type (&output_indices)[items_per_thread],
                                 bool (&is_selected)[items_per_thread])
{
    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < items_per_thread; i++)
    {
        output_indices[i] = is_selected[i] ? 1 : 0;
    }
}

template<
    unsigned int items_per_thread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void convert_selected_to_indices(uint2 (&output_indices)[items_per_thread],
                                 bool (&is_selected)[2][items_per_thread])
{
    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < items_per_thread; i++)
    {
        output_indices[i].x = is_selected[0][i] ? 1 : 0;
        output_indices[i].y = is_selected[1][i] ? 1 : 0;
    }
}

template<class OffsetT>
ROCPRIM_DEVICE ROCPRIM_INLINE void store_selected_count(size_t* selected_count,
                                                        size_t (&prev_selected_count_values)[1],
                                                        const OffsetT selected_prefix,
                                                        const OffsetT selected_in_block)
{
    selected_count[0] = prev_selected_count_values[0] + selected_prefix + selected_in_block;
}

ROCPRIM_DEVICE ROCPRIM_INLINE void store_selected_count(size_t* selected_count,
                                                        size_t (&prev_selected_count_values)[2],
                                                        const uint2 selected_prefix,
                                                        const uint2 selected_in_block)
{
    selected_count[0] = prev_selected_count_values[0] + selected_prefix.x + selected_in_block.x;
    selected_count[1] = prev_selected_count_values[1] + selected_prefix.y + selected_in_block.y;
}

template<unsigned int Size>
ROCPRIM_DEVICE void load_selected_count(const size_t* const prev_selected_count,
                                        size_t (&loaded_values)[Size])
{
    for(unsigned int i = 0; i < Size; ++i)
    {
        loaded_values[i] = prev_selected_count[i];
    }
}

template<select_method SelectMethod,
         bool          OnlySelected,
         class Config,
         class KeyIterator,
         class ValueIterator, // Can be rocprim::empty_type* if key only
         class FlagIterator,
         class OutputKeyIterator,
         class OutputValueIterator,
         class InequalityOp,
         class OffsetLookbackScanState,
         class... UnaryPredicates>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    partition_kernel_impl(KeyIterator             keys_input,
                          ValueIterator           values_input,
                          FlagIterator            flags,
                          OutputKeyIterator       keys_output,
                          OutputValueIterator     values_output,
                          size_t*                 selected_count,
                          size_t*                 prev_selected_count,
                          size_t                  prev_processed,
                          const size_t            total_size,
                          InequalityOp            inequality_op,
                          OffsetLookbackScanState offset_scan_state,
                          const unsigned int      number_of_blocks,
                          UnaryPredicates... predicates)
{
    constexpr auto block_size = Config::block_size;
    constexpr auto items_per_thread = Config::items_per_thread;
    constexpr unsigned int items_per_block = block_size * items_per_thread;

    using offset_type = typename OffsetLookbackScanState::value_type;
    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    using value_type = typename std::iterator_traits<ValueIterator>::value_type;

    // Block primitives
    using block_load_key_type = ::rocprim::block_load<
        key_type, block_size, items_per_thread,
        Config::key_block_load_method
    >;
    using block_load_value_type = ::rocprim::block_load<
        value_type, block_size, items_per_thread,
        Config::value_block_load_method
    >;
    using block_load_flag_type = ::rocprim::block_load<
        bool, block_size, items_per_thread,
        Config::flag_block_load_method
    >;
    using block_scan_offset_type = ::rocprim::block_scan<
        offset_type, block_size,
        Config::block_scan_method
    >;
    using block_discontinuity_key_type = ::rocprim::block_discontinuity<key_type, block_size>;

    // Offset prefix operation type
    using offset_scan_prefix_op_type = offset_lookback_scan_prefix_op<
        offset_type, OffsetLookbackScanState
    >;

    // Memory required for 2-phase scatter
    using exchange_keys_storage_type = key_type[items_per_block];
    using raw_exchange_keys_storage_type = typename detail::raw_storage<exchange_keys_storage_type>;
    using exchange_values_storage_type = value_type[items_per_block];
    using raw_exchange_values_storage_type = typename detail::raw_storage<exchange_values_storage_type>;

    using is_selected_type = std::conditional_t<
        sizeof...(UnaryPredicates) == 1,
        bool[items_per_thread],
        bool[sizeof...(UnaryPredicates)][items_per_thread]>;

    ROCPRIM_SHARED_MEMORY union
    {
        raw_exchange_keys_storage_type                      exchange_keys;
        raw_exchange_values_storage_type                    exchange_values;
        typename block_load_key_type::storage_type          load_keys;
        typename block_load_value_type::storage_type        load_values;
        typename block_load_flag_type::storage_type         load_flags;
        typename block_discontinuity_key_type::storage_type discontinuity_values;
        typename block_scan_offset_type::storage_type       scan_offsets;
    } storage;

    size_t prev_selected_count_values[sizeof...(UnaryPredicates)]{};
    load_selected_count(prev_selected_count, prev_selected_count_values);

    const auto         flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();
    const auto         flat_block_id        = ::rocprim::detail::block_id<0>();
    const auto         block_offset         = flat_block_id * items_per_block;
    const unsigned int valid_in_global_last_block
        = total_size - prev_processed - items_per_block * (number_of_blocks - 1);
    const bool is_last_launch = total_size <= prev_processed + number_of_blocks * items_per_block;
    const bool is_global_last_block = is_last_launch && flat_block_id == (number_of_blocks - 1);

    key_type         keys[items_per_thread];
    is_selected_type is_selected;
    offset_type      output_indices[items_per_thread];

    // Load input values into values
    if(is_global_last_block)
    {
        block_load_key_type().load(keys_input + block_offset,
                                   keys,
                                   valid_in_global_last_block,
                                   storage.load_keys);
    }
    else
    {
        block_load_key_type()
            .load(
                keys_input + block_offset,
                keys,
                storage.load_keys
            );
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    // Load selection flags into is_selected, generate them using
    // input value and selection predicate, or generate them using
    // block_discontinuity primitive
    const bool is_first_block = flat_block_id == 0 && prev_processed == 0;
    partition_block_load_flags<SelectMethod,
                               block_size,
                               block_load_flag_type,
                               block_discontinuity_key_type>(keys_input + block_offset - 1,
                                                             flags + block_offset,
                                                             keys,
                                                             is_selected,
                                                             predicates...,
                                                             inequality_op,
                                                             storage,
                                                             is_first_block,
                                                             flat_block_thread_id,
                                                             is_global_last_block,
                                                             valid_in_global_last_block);

    // Convert true/false is_selected flags to 0s and 1s
    convert_selected_to_indices(output_indices, is_selected);

    // Number of selected values in previous blocks
    offset_type selected_prefix{};
    // Number of selected values in this block
    offset_type selected_in_block{};

    // Calculate number of selected values in block and their indices
    if(flat_block_id == 0)
    {
        block_scan_offset_type()
            .exclusive_scan(
                output_indices,
                output_indices,
                offset_type{}, /** initial value */
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
    else
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
        selected_prefix   = prefix_op.get_prefix();
    }

    // Scatter selected and rejected values
    partition_scatter<OnlySelected, block_size>(keys,
                                                is_selected,
                                                output_indices,
                                                keys_output,
                                                total_size,
                                                selected_prefix,
                                                selected_in_block,
                                                storage.exchange_keys,
                                                flat_block_id,
                                                flat_block_thread_id,
                                                is_global_last_block,
                                                valid_in_global_last_block,
                                                prev_selected_count_values,
                                                prev_processed);

    static constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    if ROCPRIM_IF_CONSTEXPR (with_values) {
        value_type values[items_per_thread];

        ::rocprim::syncthreads(); // sync threads to reuse shared memory
        if(is_global_last_block)
        {
            block_load_value_type().load(values_input + block_offset,
                                         values,
                                         valid_in_global_last_block,
                                         storage.load_values);
        }
        else
        {
            block_load_value_type()
                .load(
                    values_input + block_offset,
                    values,
                    storage.load_values
                );
        }
        ::rocprim::syncthreads(); // sync threads to reuse shared memory

        partition_scatter<OnlySelected, block_size>(values,
                                                    is_selected,
                                                    output_indices,
                                                    values_output,
                                                    total_size,
                                                    selected_prefix,
                                                    selected_in_block,
                                                    storage.exchange_values,
                                                    flat_block_id,
                                                    flat_block_thread_id,
                                                    is_global_last_block,
                                                    valid_in_global_last_block,
                                                    prev_selected_count_values,
                                                    prev_processed);
    }

    // Last block in grid stores number of selected values
    const bool is_last_block = flat_block_id == (number_of_blocks - 1);
    if(is_last_block && flat_block_thread_id == 0)
    {
        store_selected_count(selected_count,
                             prev_selected_count_values,
                             selected_prefix,
                             selected_in_block);
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_PARTITION_HPP_
