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

#include <unordered_map>
#include <vector>

namespace stinkytofu {
class BasicBlock;
struct StinkyInstruction;

namespace waitcnt {

/// One immediate per hardware counter that the emit phase will turn into an
/// s_wait_dscnt / s_wait_loadcnt / s_wait_kmcnt / s_wait_tensorcnt before the
/// anchor. A field of kUnused means "do not emit a wait for this counter".
struct WaitCountSpec {
    static constexpr int kUnused = -1;

    int dsCount = kUnused;      // dlcnt -> s_wait_dscnt
    int bufferCount = kUnused;  // vlcnt -> s_wait_loadcnt
    int kmCount = kUnused;      // kmcnt -> s_wait_kmcnt
    int tensorCount = kUnused;  // tlcnt -> s_wait_tensorcnt

    bool isValid() const {
        return dsCount != kUnused || bufferCount != kUnused || kmCount != kUnused ||
               tensorCount != kUnused;
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
