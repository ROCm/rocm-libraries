/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

// Teardown for the def-use analysis: the `GFX::PHI` pseudo instructions and
// the def-use chains recorded on instructions, both built by
// buildUseDefChain(). Nothing here computes an analysis; these functions only
// discard one.
//
// They need discarding by hand because they live in the IR rather than in the
// AnalysisManager, so the pass manager cannot evict them.

#include <cstddef>
#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Function;
class Pass;

/// What discardDefUseAnalysis() got rid of.
struct DefUseAnalysisCleanup {
    size_t removedPhis = 0;
    size_t clearedInstructions = 0;
};

/// Erase every `GFX::PHI` pseudo-instruction from \p function.
///
/// These are analysis artifacts: `insertPhiInstructions()` places them to carry
/// reaching definitions across joins, and they are keyed by physical register
/// rather than by value. They are never emitted.
///
/// This does not repair def-use edges pointing at the removed PHIs, matching
/// the existing rebuild-everything callers. Use
/// discardDefUseAnalysis() when the chains are not being rebuilt.
STINKYTOFU_EXPORT size_t removeAnalysisPhis(Function& function);

/// Clear the def-use edges recorded on every instruction of \p function.
/// Returns the number of instructions visited.
STINKYTOFU_EXPORT size_t clearDefUseChains(Function& function);

/// Discard the whole def-use analysis, leaving only the instruction stream and
/// the CFG. `clearDefUseChains()` covers only the edges half of it.
///
/// Named for neither half on purpose: it removes instructions and clears
/// fields, so borrowing either verb would describe only half of it.
///
/// This is the teardown done at the canonical SSA boundary. Those analyses are
/// keyed by physical register, so they cannot describe SSA values, and leaving
/// them in place invites a consumer to trust stale data.
///
/// Chains are cleared before the PHIs are erased, so no instruction is left
/// pointing at freed memory.
STINKYTOFU_EXPORT DefUseAnalysisCleanup discardDefUseAnalysis(Function& function);

/// Creates a pass that runs discardDefUseAnalysis() on a function.
///
/// Canonical SSA construction reads a function without modifying it, so the
/// cleanup it requires cannot happen inside it and is a pipeline step of its
/// own. Run this immediately before lifting; a leftover `GFX::PHI` makes the
/// lift fail rather than being silently repaired.
///
/// Only instructions are removed. Blocks, edges, and register operands are left
/// alone, so CFG analyses stay valid.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRemoveDefUseAnalysisPass();

}  // namespace stinkytofu
