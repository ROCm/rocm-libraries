/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/core/Function.hpp"

namespace stinkytofu {
//----------------------------------------------------------------------
// CFG Traversal Utilities
//----------------------------------------------------------------------

// Traverse BasicBlocks in reverse post-order (RPO) starting from entry.
// RPO ensures that for most control flow, a block is visited after
// its predecessors (except for back edges in loops).
//
// This traversal order is optimal for forward dataflow analysis and
// ensures that information naturally flows from definitions to uses.
//
// Example usage:
//   traverseCFGInRPO(function, [&](BasicBlock* bb) {
//       // Process each BasicBlock in RPO order
//       processBlock(bb);
//   });
template <typename Visitor>
void traverseCFGInRPO(Function& func, Visitor&& visitor) {
    if (!func.getEntryBlock()) return;

    // Post-order DFS traversal
    std::vector<BasicBlock*> postOrder;
    std::unordered_set<BasicBlock*> visited;

    std::function<void(BasicBlock*)> dfs = [&](BasicBlock* bb) {
        if (!bb || visited.count(bb)) return;
        visited.insert(bb);

        // Visit successors first (post-order)
        for (BasicBlock* succ : bb->getSuccessors()) {
            dfs(succ);
        }
        postOrder.push_back(bb);
    };

    dfs(func.getEntryBlock());

    // Reverse post-order
    for (auto it = postOrder.rbegin(); it != postOrder.rend(); ++it) {
        visitor(*it);
    }
}

}  // namespace stinkytofu
