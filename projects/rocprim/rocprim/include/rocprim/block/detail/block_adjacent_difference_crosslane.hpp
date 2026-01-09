// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_ADJACENT_DIFFERENCE_CROSSLANE_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_ADJACENT_DIFFERENCE_CROSSLANE_HPP_

#include "block_adjacent_difference_shared_mem.hpp"

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../intrinsics.hpp"
#include "../../intrinsics/thread.hpp"

#include <type_traits>

#include <cassert>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<typename T,
         unsigned int BlockSizeX,
         unsigned int BlockSizeY = 1,
         unsigned int BlockSizeZ = 1,
         bool         UseDPP     = ROCPRIM_DETAIL_USE_DPP>
class block_adjacent_difference_crosslane
{

// Temporary fix: issue with dpp bound_ctrl on Windows, GFX10, GFX11, GFX12 and SPIR-V
// RDNA encounters compile issues in hipCUB and rocThrust.
#if defined(_WIN32) || defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__) \
    || defined(__SPIRV__)
    static constexpr bool bndCtrl = false;
#else
    static constexpr bool bndCtrl = true;
#endif

public:
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int WarpSize  = rocprim::arch::wavefront::min_size();

    struct storage_type
    {
        // Shared memory storage to hold items at the boundaries of warps.
        // Size is calculated based on the number of warps in the block.
        T items[ceiling_div(BlockSize, WarpSize)];
    };

    template<bool         AsFlags,
             bool         Reversed,
             bool         WithTilePredecessor,
             unsigned int ItemsPerThread,
             typename Output,
             typename BinaryFunction>
    ROCPRIM_DEVICE
    void apply_left(const T (&input)[ItemsPerThread],
                    Output (&output)[ItemsPerThread],
                    BinaryFunction op,
                    const T        tile_predecessor_item,
                    storage_type&  storage)
    {
        static constexpr auto as_flags = bool_constant<AsFlags>{};
        static constexpr auto reversed = bool_constant<Reversed>{};

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int lane_id = ::rocprim::lane_id();

        // Save the last item of each thread.
        T last_item = input[ItemsPerThread - 1];

        // If this is the last thread in the warp, save its last item to shared memory.
        // This allows the next warp to access this value as its predecessor.
        if(lane_id == WarpSize - 1)
        {
            storage.items[warp_id] = last_item;
        }

        ROCPRIM_UNROLL
        for(unsigned int i = ItemsPerThread - 1; i > 0; --i)
        {
            output[i] = detail::apply(op,
                                      input[i - 1],
                                      input[i],
                                      flat_id * ItemsPerThread + i,
                                      as_flags,
                                      reversed);
        }
        ::rocprim::syncthreads();

        T predecessor_item;

        predecessor_item = warp_shift1<true>(last_item);

        // Handle the boundary condition: The first lane (lane 0) of a warp cannot
        // get data via shuffle/DPP from within the same warp.
        // It must read the last item of the *previous* warp from shared memory.
        if(lane_id == 0 && flat_id != 0)
        {
            predecessor_item = storage.items[warp_id - 1];
        }

        if constexpr(WithTilePredecessor)
        {
            // If this is the very first thread of the entire tile,
            // use the provided tile predecessor item.
            if(flat_id == 0)
            {
                predecessor_item = tile_predecessor_item;
            }

            output[0] = detail::apply(op,
                                      predecessor_item,
                                      input[0],
                                      flat_id * ItemsPerThread,
                                      as_flags,
                                      reversed);
        }
        else
        {
            if(flat_id != 0)
            {
                output[0] = detail::apply(op,
                                          predecessor_item,
                                          input[0],
                                          flat_id * ItemsPerThread,
                                          as_flags,
                                          reversed);
            }
            else
            {
                output[0] = get_default_item(input, 0, as_flags);
            }
        }
    }

    template<bool         AsFlags,
             bool         Reversed,
             bool         WithTilePredecessor,
             unsigned int ItemsPerThread,
             typename Output,
             typename BinaryFunction>
    ROCPRIM_DEVICE
    void apply_left_partial(const T (&input)[ItemsPerThread],
                            Output (&output)[ItemsPerThread],
                            BinaryFunction     op,
                            const T            tile_predecessor_item,
                            const unsigned int valid_items,
                            storage_type&      storage)
    {
        static constexpr auto as_flags = bool_constant<AsFlags>{};
        static constexpr auto reversed = bool_constant<Reversed>{};

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int lane_id = ::rocprim::lane_id();

        T last_item = input[ItemsPerThread - 1];

        if(lane_id == WarpSize - 1)
        {
            storage.items[warp_id] = last_item;
        }

        ROCPRIM_UNROLL
        for(unsigned int i = ItemsPerThread - 1; i > 0; --i)
        {
            const unsigned int index = flat_id * ItemsPerThread + i;
            output[i]                = get_default_item(input, i, as_flags);

            if(index < valid_items)
            {
                output[i] = detail::apply(op, input[i - 1], input[i], index, as_flags, reversed);
            }
        }
        ::rocprim::syncthreads();

        const unsigned int index = flat_id * ItemsPerThread;

        T predecessor_item;

        predecessor_item = warp_shift1<true>(last_item);

        output[0] = get_default_item(input, 0, as_flags);

        if(index < valid_items)
        {
            if constexpr(WithTilePredecessor)
            {
                if(flat_id == 0)
                {
                    predecessor_item = tile_predecessor_item;
                }

                output[0]
                    = detail::apply(op, predecessor_item, input[0], index, as_flags, reversed);
            }
            else
            {
                if(flat_id != 0)
                {
                    output[0]
                        = detail::apply(op, predecessor_item, input[0], index, as_flags, reversed);
                }
            }
        }
    }

    template<bool         AsFlags,
             bool         Reversed,
             bool         WithTileSuccessor,
             unsigned int ItemsPerThread,
             typename Output,
             typename BinaryFunction>
    ROCPRIM_DEVICE
    void apply_right(const T (&input)[ItemsPerThread],
                     Output (&output)[ItemsPerThread],
                     BinaryFunction op,
                     const T        tile_successor_item,
                     storage_type&  storage)
    {
        static constexpr auto as_flags = bool_constant<AsFlags>{};
        static constexpr auto reversed = bool_constant<Reversed>{};

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int lane_id = ::rocprim::lane_id();

        // Save the first item of each thread.
        T first_item = input[0];

        // If this is the first thread in the warp, save its first item to shared memory.
        // This allows the previous warp to access this value as its successor.
        if(lane_id == 0)
        {
            storage.items[warp_id] = first_item;
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread - 1; ++i)
        {
            output[i] = detail::apply(op,
                                      input[i],
                                      input[i + 1],
                                      flat_id * ItemsPerThread + i + 1,
                                      as_flags,
                                      reversed);
        }
        ::rocprim::syncthreads();

        T successor_item;

        successor_item = warp_shift1<false>(first_item);

        // Handle the boundary condition: The last lane (lane WarpSize - 1) of a warp
        // cannot get data via shuffle/DPP from within the same warp.
        // It must read the first item of the *following* warp from shared memory.
        if(lane_id == WarpSize - 1 && flat_id != BlockSize - 1)
        {
            successor_item = storage.items[warp_id + 1];
        }

        if constexpr(WithTileSuccessor)
        {
            // If this is the very last thread of the entire tile,
            // use the provided tile successor item.
            if(flat_id == BlockSize - 1)
            {
                successor_item = tile_successor_item;
            }

            output[ItemsPerThread - 1] = detail::apply(op,
                                                       input[ItemsPerThread - 1],
                                                       successor_item,
                                                       flat_id * ItemsPerThread + ItemsPerThread,
                                                       as_flags,
                                                       reversed);
        }
        else
        {
            if(flat_id != BlockSize - 1)
            {
                output[ItemsPerThread - 1]
                    = detail::apply(op,
                                    input[ItemsPerThread - 1],
                                    successor_item,
                                    flat_id * ItemsPerThread + ItemsPerThread,
                                    as_flags,
                                    reversed);
            }
            else
            {
                output[ItemsPerThread - 1] = get_default_item(input, ItemsPerThread - 1, as_flags);
            }
        }
    }

    template<bool         AsFlags,
             bool         Reversed,
             unsigned int ItemsPerThread,
             typename Output,
             typename BinaryFunction>
    ROCPRIM_DEVICE
    void apply_right_partial(const T (&input)[ItemsPerThread],
                             Output (&output)[ItemsPerThread],
                             BinaryFunction     op,
                             const unsigned int valid_items,
                             storage_type&      storage)
    {
        static constexpr auto as_flags = bool_constant<AsFlags>{};
        static constexpr auto reversed = bool_constant<Reversed>{};

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int lane_id = ::rocprim::lane_id();

        T first_item = input[0];

        if(lane_id == 0)
        {
            storage.items[warp_id] = first_item;
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread - 1; ++i)
        {
            const unsigned int index = flat_id * ItemsPerThread + i + 1;
            output[i]                = get_default_item(input, i, as_flags);

            if(index < valid_items)
            {
                output[i] = detail::apply(op, input[i], input[i + 1], index, as_flags, reversed);
            }
        }
        ::rocprim::syncthreads();

        T successor_item;

        successor_item = warp_shift1<false>(first_item);

        output[ItemsPerThread - 1] = get_default_item(input, ItemsPerThread - 1, as_flags);

        const unsigned int next_thread_index = flat_id * ItemsPerThread + ItemsPerThread;

        if(next_thread_index < valid_items)
        {
            if(lane_id == WarpSize - 1 && flat_id != BlockSize - 1)
            {
                successor_item = storage.items[warp_id + 1];
            }

            output[ItemsPerThread - 1] = detail::apply(op,
                                                       input[ItemsPerThread - 1],
                                                       successor_item,
                                                       next_thread_index,
                                                       as_flags,
                                                       reversed);
        }
    }

private:
    // Helper function to retrieve the default boundary value when operating in "Flags" mode (AsFlags = true).
    // This is typically used for discontinuity detection (e.g., head flags).
    // The boundary item (the very first item with no predecessor) is by definition the start of a segment,
    // so it returns 1 (true) to flag it.
    template<unsigned int ItemsPerThread>
    ROCPRIM_DEVICE
    int get_default_item(const T (&)[ItemsPerThread],
                         unsigned int /*index*/,
                         bool_constant<true> /*as_flags*/)
    {
        return 1;
    }

    // Helper function to retrieve the default boundary value when operating in standard mode (AsFlags = false).
    // This is used for standard adjacent difference calculations.
    // For the boundary item (which has no predecessor), the standard behavior in this implementation
    // is to return the item itself (effectively input[i] - nothing/identity), preserving the original value.
    template<unsigned int ItemsPerThread>
    ROCPRIM_DEVICE
    T get_default_item(const T (&input)[ItemsPerThread],
                       const unsigned int index,
                       bool_constant<false> /*as_flags*/)
    {
        return input[index];
    }

    // Returns the value from the adjacent lane within the same warp (shift by 1).
    // Prefers DPP when available and allowed (can be faster than shuffle on some archs).
    // Falls back to shuffle if DPP is not suitable or not enabled.
    //
    // Note: This helper only handles intra-warp movement.
    // Cross-warp / tile boundary fixups (lane 0 / lane last) must be handled by the caller.
    template<bool IsUp>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    T warp_shift1(T value)
    {
        if constexpr(UseDPP)
        {
            // We have to know after compiling SPIR-V whether DPP is available.
            // Therefore, this check cannot be done at the C++ constexpr-level.
            if ROCPRIM_AMDGCN_CONSTEXPR(ROCPRIM_HAS_DPP() && !ROCPRIM_HAS_PERMLANE())
            {
                // 0x138: row_shr:1 (gets lane_id - 1)
                // 0x134: row_shl:1 (gets lane_id + 1)
                constexpr unsigned int ctrl = IsUp ? 0x138 : 0x134;

                // row_mask=0xf and bank_mask=0xf enable DPP for all rows/banks (no extra masking).
                return warp_move_dpp<T, ctrl, 0xf, 0xf, bndCtrl>(value);
            }
        }

        // Standard shuffle fallback.
        if constexpr(IsUp)
        {
            return warp_shuffle_up(value, 1, WarpSize);
        }
        else
        {
            return warp_shuffle_down(value, 1, WarpSize);
        }
    }
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_ADJACENT_DIFFERENCE_CROSSLANE_HPP_
