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

#include <cassert>
#include <iosfwd>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ReadyQueue.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace dag {

using DAGNodeList = std::vector<DAGNode>;

struct RegionDAG {
    DAGNodeList nodes;
    std::vector<std::unordered_set<unsigned>> graph;
    std::unordered_map<StinkyInstruction*, unsigned> instToId;
};

/// True if \p to is reachable from \p from by following \p graph's edges.
/// Used to reject a hard scheduling constraint that would introduce a cycle
/// before it is merged into the region's DAG.
inline bool hasPath(const std::vector<std::unordered_set<unsigned>>& graph, unsigned from,
                    unsigned to) {
    if (from == to) return true;

    std::vector<unsigned> pending{from};
    std::vector<bool> visited(graph.size(), false);
    visited[from] = true;
    while (!pending.empty()) {
        const unsigned current = pending.back();
        pending.pop_back();
        for (unsigned successor : graph[current]) {
            if (successor == to) return true;
            if (!visited[successor]) {
                visited[successor] = true;
                pending.push_back(successor);
            }
        }
    }
    return false;
}

/// Add a non-duplicate DAG edge and update the destination in-degree.
inline void addEdgeById(DAGNode* from, DAGNode* to,
                        std::vector<std::unordered_set<unsigned>>& graph) {
    if (from->id == to->id || graph[from->id].contains(to->id)) return;
    graph[from->id].insert(to->id);
    ++to->inDegree;
}

/// Build RAW/WAR/WAW edges for physical and pseudo registers over \p instructions
/// in program order. Dense node ids match instruction indices.
RegionDAG buildRegisterDependencyDAG(const std::vector<StinkyInstruction*>& instructions);

/// Same as above for an IRList region iterator pair.
RegionDAG buildRegisterDependencyDAG(IRList::iterator regionStart, IRList::iterator regionEnd);

/// Print each DAG node and its successor IDs. \p hardConstraintEdges, if given, marks which
/// edges are scheduler-policy links (not real register dependencies) merged into \p dag by
/// the caller, so debug output can still tell the two apart even though they now share one
/// graph.
void dumpDAGGraph(const RegionDAG& dag, std::ostream& os,
                  const std::set<std::pair<unsigned, unsigned>>& hardConstraintEdges = {});

}  // namespace dag
}  // namespace stinkytofu
