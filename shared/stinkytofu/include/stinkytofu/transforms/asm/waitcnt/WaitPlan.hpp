// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
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
/// s_wait_dscnt / s_wait_loadcnt / s_wait_kmcnt / s_wait_tensorcnt /
/// s_wait_xcnt before the anchor.
///
/// A field of kUnused means "do not emit a wait for this counter".
struct WaitCountSpec {
    static constexpr int kUnused = -1;

    int dsCount = kUnused;      // dlcnt -> s_wait_dscnt
    int bufferCount = kUnused;  // vlcnt -> s_wait_loadcnt
    int kmCount = kUnused;      // kmcnt -> s_wait_kmcnt
    int tensorCount = kUnused;  // tlcnt -> s_wait_tensorcnt
    int xCount = kUnused;       // xcnt -> s_wait_xcnt

    bool isValid() const {
        return dsCount != kUnused || bufferCount != kUnused || kmCount != kUnused ||
               tensorCount != kUnused || xCount != kUnused;
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
