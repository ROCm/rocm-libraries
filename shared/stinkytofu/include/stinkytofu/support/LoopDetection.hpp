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

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/core/Function.hpp"

namespace stinkytofu {
/// A natural loop detected from CFG back-edges.
struct Loop {
    BasicBlock* headerBB = nullptr;    // back-edge target (loop entry)
    BasicBlock* latchBB = nullptr;     // BB containing the back-edge branch
    std::vector<BasicBlock*> bodyBBs;  // all BBs in the loop body (includes header)

    bool contains(const BasicBlock* bb) const {
        return std::find(bodyBBs.begin(), bodyBBs.end(), bb) != bodyBBs.end();
    }
};

/// Detect natural loops in a function by finding CFG back-edges.
///
/// A back-edge is an edge from a BB to a dominator (or to itself). We approximate
/// this using DFS: an edge to an already-visited-but-not-finished node is a back-edge.
/// For each back-edge (latch -> header), we collect the loop body by walking predecessors
/// from the latch back to the header.
///
/// Returns one Loop per back-edge found.
inline std::vector<Loop> detectLoops(Function& func) {
    std::vector<Loop> loops;

    if (!func.getEntryBlock()) return loops;

    // DFS coloring: 0 = white (unvisited), 1 = gray (in progress), 2 = black (finished)
    std::unordered_map<BasicBlock*, int> color;
    for (BasicBlock& bb : func) color[&bb] = 0;

    // Back-edges: latch -> header
    std::vector<std::pair<BasicBlock*, BasicBlock*>> backEdges;

    std::function<void(BasicBlock*)> dfs = [&](BasicBlock* bb) {
        color[bb] = 1;  // gray
        for (BasicBlock* succ : bb->getSuccessors()) {
            if (color[succ] == 1) {
                // Edge to a gray node = back-edge
                backEdges.push_back({bb, succ});
            } else if (color[succ] == 0) {
                dfs(succ);
            }
        }
        color[bb] = 2;  // black
    };

    dfs(func.getEntryBlock());

    // For each back-edge, collect the natural loop body by walking predecessors
    // from latch back to header.
    for (auto& [latch, header] : backEdges) {
        Loop loop;
        loop.headerBB = header;
        loop.latchBB = latch;

        std::unordered_set<BasicBlock*> body;
        body.insert(header);

        if (latch != header) {
            body.insert(latch);
            // Walk predecessors from latch to find all BBs that can reach latch
            // without going through header.
            std::vector<BasicBlock*> worklist = {latch};
            while (!worklist.empty()) {
                BasicBlock* curr = worklist.back();
                worklist.pop_back();
                for (BasicBlock* pred : curr->getPredecessors()) {
                    if (body.insert(pred).second) worklist.push_back(pred);
                }
            }
        }

        loop.bodyBBs.assign(body.begin(), body.end());
        loops.push_back(std::move(loop));
    }

    return loops;
}

/// Find which Loop (if any) a given BB belongs to.
/// Returns nullptr if the BB is not part of any detected loop.
inline const Loop* findLoopForBB(const std::vector<Loop>& loops, const BasicBlock* bb) {
    for (const Loop& loop : loops) {
        if (loop.contains(bb)) return &loop;
    }
    return nullptr;
}

}  // namespace stinkytofu
