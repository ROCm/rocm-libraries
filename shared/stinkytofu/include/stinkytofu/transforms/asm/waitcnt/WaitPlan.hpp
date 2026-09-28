/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

// Plain data types describing the result of wait-count planning. A
// WaitInsertionPlan is produced by the WaitDataflow solver (conservative,
// per-consumer waits) and may then be rewritten by WaitPlanOptimizers
// (e.g. ShallowPredPromotion) before the emit phase materialises the IR.

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace stinkytofu {
class BasicBlock;
struct StinkyInstruction;

namespace waitcnt {

/// One logical LDS token retired by a generated wait, together with the direct
/// CFG predecessor through which that token reached the wait.
struct WaitSource {
    int token = 0;
    int tripsBack = -1;  // -1 when the bounded queue lost the exact age
    bool tripAgeSaturated = false;
    int ringSlot = -1;  // -1 when token-to-ring mapping is ambiguous
    std::string predecessor;

    bool operator==(const WaitSource&) const = default;
};

/// One immediate per hardware counter that the emit phase will turn into an
/// s_wait_dscnt / s_wait_loadcnt / s_wait_kmcnt / s_wait_tensorcnt before the
/// anchor. A field of kUnused means "do not emit a wait for this counter".
struct WaitCountSpec {
    static constexpr int kUnused = -1;
    static constexpr size_t kCounterCount = 5;

    int dsCount = kUnused;      // dlcnt -> s_wait_dscnt
    int loadCount = kUnused;    // vlcnt -> s_wait_loadcnt
    int kmCount = kUnused;      // kmcnt -> s_wait_kmcnt
    int tensorCount = kUnused;  // tlcnt -> s_wait_tensorcnt
    int asyncCount = kUnused;   // asynccnt -> s_wait_asynccnt

    // Memory tokens of the tensor_load ops this tensorcnt wait drains (union,
    // sorted-unique). Attached to the emitted s_wait_tensorcnt as MemTokenData so
    // later passes (e.g. TDMLoadWaveSyncPass) can identify the drained wait group.
    // Empty when tensorCount is kUnused or no drained load carries a token.
    std::vector<int> tensorTokens;

    // Sorted-unique operations retired by each counter wait. Populated from
    // the live queues after all plan optimizations, so emitted comments describe
    // the final wait rather than the conservative pre-optimization plan.
    std::array<std::vector<WaitSource>, kCounterCount> sources;

    bool isValid() const {
        return dsCount != kUnused || loadCount != kUnused || kmCount != kUnused ||
               tensorCount != kUnused || asyncCount != kUnused;
    }
};

/// A tail drain to insert immediately before predBB's terminator. Used by
/// the shallow-pred promotion optimizer to pre-drain one CFG path so the
/// merge anchor's wait can stay lenient. Today only the tensor counter
/// supports tail drains; the field is generalised to all three so future
/// optimizers can use the same mechanism.
struct TailDrain {
    BasicBlock* predBB = nullptr;
    WaitCountSpec spec;
};

/// Per-consumer wait spec plus any predecessor tail drains.
///
///   anchorWaits[I]   the s_wait_* immediates to emit before instruction I
///   tailDrains       the s_wait_* immediates to emit before each listed
///                    predecessor's terminator
///
/// Order of entries within each container is the order in which the emit
/// phase will visit them; it MUST be deterministic.
struct WaitInsertionPlan {
    std::unordered_map<StinkyInstruction*, WaitCountSpec> anchorWaits;
    std::vector<TailDrain> tailDrains;
};

}  // namespace waitcnt
}  // namespace stinkytofu
