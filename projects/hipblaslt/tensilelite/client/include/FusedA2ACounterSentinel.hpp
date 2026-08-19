// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Guard tail for the fused GEMM.A2A counter allocation. Four regions share
// the allocation: W*tokenTiles first-level counter[dst_rank*tokenTiles+j],
// a W-entry counter2[dst_rank], a single counter3, and the SDMA cursor pairs
// (see FusedA2AClient.cpp counterBytes). The guard tail sits past the cursors
// and catches only an overrun past the top region, by absorbing the write
// inside the allocation instead of letting it reach whatever hipMalloc handed
// back next. An off-by-one in a lower region lands on a live slot above it
// instead -- at the top of counter2's range, that slot is counter3, so the
// failure there is a mis-elected DRAIN owner rather than a loud error.

#include <cstddef>
#include <cstdint>

namespace TensileLite
{
    namespace Client
    {
        // Wide enough that a plausible off-by-one lands inside it, and a
        // whole number of 4-byte counter slots.
        constexpr size_t FUSED_A2A_COUNTER_SENTINEL_BYTES = 64;
        constexpr size_t FUSED_A2A_COUNTER_SENTINEL_WORDS
            = FUSED_A2A_COUNTER_SENTINEL_BYTES / sizeof(uint32_t);

        // 8-aligned: the counts above are an odd or even number of words
        // depending on W and tokenTiles, and s_atomic_umax_x2 needs its operand
        // 2-register aligned.
        constexpr size_t fusedA2ACounterCursorOffset(uint32_t worldSize, uint32_t tokenTiles)
        {
            return (((size_t)worldSize * tokenTiles + worldSize + 1) * sizeof(uint32_t) + 7)
                   & ~(size_t)7;
        }

        // [0] reservation, [1] commit. Interleaved per queue so the kernel
        // reaches both from one base and never needs W.
        constexpr size_t FUSED_A2A_CURSORS_PER_QUEUE = 2;

        // Live counter bytes. size_t (not uint32) so a large W*tokenTiles cannot
        // wrap and under-allocate. The cursors are inside the per-launch memset
        // range; the kernel raises them back to the hardware write pointer
        // before reserving.
        constexpr size_t fusedA2ACounterPayloadBytes(uint32_t worldSize, uint32_t tokenTiles)
        {
            return fusedA2ACounterCursorOffset(worldSize, tokenTiles)
                   + (size_t)worldSize * FUSED_A2A_CURSORS_PER_QUEUE * sizeof(uint64_t);
        }

        // What to hipMalloc: payload plus the guard tail. The per-launch memset
        // must clear only fusedA2ACounterPayloadBytes, leaving the guard filled.
        constexpr size_t fusedA2ACounterAllocBytes(uint32_t worldSize, uint32_t tokenTiles)
        {
            return fusedA2ACounterPayloadBytes(worldSize, tokenTiles)
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
