// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Guard tail for the fused GEMM.A2A counter allocation. Four regions share
// the allocation, in this order:
//
//   +0    cursors   MAX_RANKS * 2 * u64   SDMA cursor pair per queue
//   +128  counter2  MAX_RANKS * u32       counter2[dst_rank]
//   +192  counter3  1 * u32
//   +256  counter1  W * tokenTiles * u32  counter1[dst_rank*tokenTiles + j]
//   ...   guard tail
//
// FusedA2AMode=1 shares the cursor region and lays out two regions of its own
// past it:
//
//   +0    cursors   MAX_RANKS * 2 * u64   SDMA cursor pair per queue
//   +128  flag      tokenBlocks*(W-1)*u32 flag[b][queue], rounded to a line
//   ...   counter   batches * u32         counter[iB]
//   ...   guard tail
//
// Each region starts on a 64-byte line; the trailing variable-size region gets
// no tail padding. The three leading regions are sized by FUSED_A2A_MAX_RANKS
// rather than the runtime world size. The guard tail catches an overrun of
// counter1 -- the one region whose index is computed at runtime -- by absorbing
// the write inside the allocation instead of letting it reach whatever
// hipMalloc handed back next.

#include <cstddef>
#include <cstdint>

#include <Tensile/FusedA2AKernArg.hpp> // FUSED_A2A_MAX_RANKS, fusedA2AAlignLine

namespace TensileLite
{
    namespace Client
    {
        // Wide enough that a plausible off-by-one lands inside it, and a
        // whole number of 4-byte counter slots.
        constexpr size_t FUSED_A2A_COUNTER_SENTINEL_BYTES = 64;
        constexpr size_t FUSED_A2A_COUNTER_SENTINEL_WORDS
            = FUSED_A2A_COUNTER_SENTINEL_BYTES / sizeof(uint32_t);

        // [0] reservation, [1] commit. Interleaved per queue so the kernel
        // reaches both from one base and never needs W.
        constexpr size_t FUSED_A2A_CURSORS_PER_QUEUE = 2;

        // Twinned with FUSED_A2A_COUNTER*_OFFSET in Tensile/Components/Signature.py.
        constexpr size_t FUSED_A2A_CURSOR_REGION_BYTES = fusedA2AAlignLine(
            (size_t)FUSED_A2A_MAX_RANKS * FUSED_A2A_CURSORS_PER_QUEUE * sizeof(uint64_t));
        constexpr size_t FUSED_A2A_COUNTER2_OFFSET = FUSED_A2A_CURSOR_REGION_BYTES;
        constexpr size_t FUSED_A2A_COUNTER3_OFFSET = fusedA2AAlignLine(
            FUSED_A2A_COUNTER2_OFFSET + (size_t)FUSED_A2A_MAX_RANKS * sizeof(uint32_t));
        constexpr size_t FUSED_A2A_COUNTER1_OFFSET
            = fusedA2AAlignLine(FUSED_A2A_COUNTER3_OFFSET + sizeof(uint32_t));

        // Live counter bytes. size_t (not uint32) so a large W*tokenTiles cannot
        // wrap and under-allocate. The cursors are inside the per-launch memset
        // range; the kernel raises them back to the hardware write pointer
        // before reserving.
        constexpr size_t fusedA2ACounterPayloadBytes(uint32_t worldSize, uint32_t tokenTiles)
        {
            return FUSED_A2A_COUNTER1_OFFSET
                   + (size_t)worldSize * tokenTiles * sizeof(uint32_t);
        }

        // What to hipMalloc: payload plus the guard tail. The per-launch memset
        // must clear only fusedA2ACounterPayloadBytes, leaving the guard filled.
        constexpr size_t fusedA2ACounterAllocBytes(uint32_t worldSize, uint32_t tokenTiles)
        {
            return fusedA2ACounterPayloadBytes(worldSize, tokenTiles)
                   + FUSED_A2A_COUNTER_SENTINEL_BYTES;
        }

        // Twinned with FUSED_A2A_MODE1_FLAG_OFFSET in Tensile/Components/Signature.py.
        constexpr size_t FUSED_A2A_MODE1_FLAG_OFFSET = FUSED_A2A_CURSOR_REGION_BYTES;

        // The one region reset per launch, sized as the kernel rounds it in
        // a2aElect. worldSize must be >= 1.
        constexpr size_t fusedA2AMode1FlagBytes(uint32_t worldSize, uint32_t tokenBlocks)
        {
            return fusedA2AAlignLine((size_t)tokenBlocks * (worldSize - 1) * sizeof(uint32_t));
        }

        // Live mode-1 bytes. `batches` is the number of distinct iB the kernel
        // can compute, not the world size.
        constexpr size_t
            fusedA2AMode1PayloadBytes(uint32_t worldSize, uint32_t tokenBlocks, uint32_t batches)
        {
            return FUSED_A2A_MODE1_FLAG_OFFSET + fusedA2AMode1FlagBytes(worldSize, tokenBlocks)
                   + (size_t)batches * sizeof(uint32_t);
        }

        constexpr size_t
            fusedA2AMode1AllocBytes(uint32_t worldSize, uint32_t tokenBlocks, uint32_t batches)
        {
            return fusedA2AMode1PayloadBytes(worldSize, tokenBlocks, batches)
                   + FUSED_A2A_COUNTER_SENTINEL_BYTES;
        }

        // Expected guard word at index i: the high half is far from any legal
        // counter value, and the low half varies per word so a shifted
        // overrun is still detected.
        constexpr uint32_t fusedA2ACounterSentinelWord(size_t i)
        {
            return 0xA2A50000u | (uint32_t)(i & 0xFFFFu);
        }

        // Fill the guard tail. `guard` points at the first byte past the
        // payload and must have FUSED_A2A_COUNTER_SENTINEL_WORDS words.
        inline void fusedA2ACounterSentinelFill(uint32_t* guard)
        {
            for(size_t i = 0; i < FUSED_A2A_COUNTER_SENTINEL_WORDS; i++)
                guard[i] = fusedA2ACounterSentinelWord(i);
        }

        // Index of the first corrupted guard word, or -1 if the tail is intact.
        inline int fusedA2ACounterSentinelFirstBad(const uint32_t* guard)
        {
            for(size_t i = 0; i < FUSED_A2A_COUNTER_SENTINEL_WORDS; i++)
            {
                if(guard[i] != fusedA2ACounterSentinelWord(i))
                    return (int)i;
            }
            return -1;
        }
    } // namespace Client
} // namespace TensileLite
