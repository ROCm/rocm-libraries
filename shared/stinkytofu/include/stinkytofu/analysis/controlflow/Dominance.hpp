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

#include <unordered_map>
#include <vector>

#include "stinkytofu/core/Function.hpp"

namespace stinkytofu {
/// Dominator-tree and dominance-frontier analysis for a Function's CFG.
///
/// Computed via the Cooper-Harvey-Kennedy iterative algorithm:
///   - RPO ordering of basic blocks
///   - Immediate dominator tree (indexed by RPO position)
///   - Dominance frontiers (indexed by RPO position)
///
/// Time: O(N*E) worst case; near-linear for reducible CFGs.
///       N = blocks, E = CFG edges.
struct DominanceInfo {
    static constexpr unsigned kUndef = ~0u;

    /// Basic blocks in reverse post-order.
    std::vector<BasicBlock*> rpo;

    /// Map from BasicBlock* to its RPO index.
    std::unordered_map<const BasicBlock*, unsigned> rpoIndex;

    /// Immediate dominator of each block, indexed by RPO position.
    /// Convention: idom[0] = 0 (entry dominates itself).
    std::vector<unsigned> idom;

    /// Dominance frontier of each block, indexed by RPO position.
    /// df[b] = { j : b dominates a predecessor of j but does not
    ///               strictly dominate j }
    std::vector<std::vector<unsigned>> df;
};

/// Compute dominator tree and dominance frontiers for the given function.
/// Returns an empty DominanceInfo if the function has no blocks.
DominanceInfo computeDominanceInfo(Function& func);

}  // namespace stinkytofu
