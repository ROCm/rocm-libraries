// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Guard tail for the fused GEMM.A2A counter allocation.
//
// Three levels share the allocation, indexed kernel-side as
// counter[dst_rank*tokenTiles + WorkGroup1] over W*tokenTiles slots, then a
// W-entry counter2[dst_rank], then a single counter3 at W*tokenTiles + W (see
// FusedA2AClient.cpp where counterBytes is computed). Every index expression
// is derived from grid dimensions, so an off-by-one is the class of bug this
// header exists for.
//
// The guard tail catches only an overrun past the TOP level, and it catches it
// by absorbing the write: the tail is inside the allocation, so the store
// reddens the pattern rather than reaching memory that is not ours. Absent the
// tail, that same store lands in whatever hipMalloc handed back next -- the
// worst silent failure mode on this branch, corrupting unrelated device memory
// while the A2A's own numeric validation still passes.
//
// An off-by-one in a lower level never reaches the tail: it lands on a live
// slot of the level above. At the top of counter2's range that slot is
// counter3, and counter3 is what elects the DRAIN owner, so the failure mode
// there is a mis-elected owner, NOT a loud error.
//
// The detector appends FUSED_A2A_COUNTER_SENTINEL_BYTES past the payload,
// fills it with a known pattern at allocation time, and re-checks it after
// each launch. It is a probe for a whole class of bug, not a fix for a known
// one -- no counter overrun is currently known.
//
// These are free functions in a header (rather than TU-private helpers in
// FusedA2AClient.cpp) so tests/FusedA2ACounterSentinel_test.cpp can drive the
// REAL fill and check. A test that reproduced them could not catch drift in
// the pattern it exists to verify.

#include <cstddef>
#include <cstdint>

namespace TensileLite
{
    namespace Client
    {
        // 64 bytes: wide enough that a plausible off-by-one lands inside it
        // rather than skipping past, and a whole number of 4-byte counter
        // slots so a slot-granular overrun always lands on a word boundary.
        constexpr size_t FUSED_A2A_COUNTER_SENTINEL_BYTES = 64;
        constexpr size_t FUSED_A2A_COUNTER_SENTINEL_WORDS
            = FUSED_A2A_COUNTER_SENTINEL_BYTES / sizeof(uint32_t);

        // Live counter bytes. Three levels share one allocation:
        //   W*tokenTiles  first-level  counter[dst_rank*tokenTiles + j]
        //   W             second-level counter2[dst_rank]
        //   1             third-level  counter3, the grid-wide WG tally that
        //                 elects the single DRAIN owner
        // Computed in size_t (not uint32) so a large W*tokenTiles cannot wrap
        // and under-allocate.
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

        // Expected guard word at index i.
        //
        // Two properties matter, and both are load-bearing:
        //   - the high half is a magic far from any legal counter value (a
        //     counter holds a small tile count, or 0 from the per-launch
        //     memset), so an overrun cannot deposit a value that reads as
        //     intact;
        //   - the low half varies per word, so an overrun landing at a SHIFTED
        //     offset is still detected. With a uniform pattern, a write that
        //     lands one word off leaves every word holding a legal value and
        //     the guard reads intact straight through the corruption.
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

        // Index of the first corrupted guard word, or -1 if the tail is
        // intact. The index (not just a bool) is what tells you how far past
        // the payload the offending write ran.
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
