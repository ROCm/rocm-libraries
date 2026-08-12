// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Guard tail for the fused GEMM.A2A counter allocation. Three levels share
// the allocation: W*tokenTiles first-level counter[dst_rank*tokenTiles+j],
// a W-entry counter2[dst_rank], and a single counter3 (see FusedA2AClient.cpp
// counterBytes). The guard tail sits past counter3 and catches only an
// overrun past that top level, by absorbing the write inside the allocation
// instead of letting it reach whatever hipMalloc handed back next. An
// off-by-one in a lower level lands on a live slot above it instead -- at the
// top of counter2's range, that slot is counter3, so the failure there is a
// mis-elected DRAIN owner rather than a loud error.
//
// These are free functions in a header (rather than TU-private helpers in
// FusedA2AClient.cpp) so tests/FusedA2ACounterSentinel_test.cpp can drive the
// real fill and check.

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

        // Live counter bytes for the three levels above. size_t (not uint32)
        // so a large W*tokenTiles cannot wrap and under-allocate.
        constexpr size_t fusedA2ACounterPayloadBytes(uint32_t worldSize, uint32_t tokenTiles)
        {
            return ((size_t)worldSize * tokenTiles + worldSize + 1) * sizeof(uint32_t);
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
