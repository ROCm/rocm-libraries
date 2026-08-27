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
#include <unordered_map>
#include <unordered_set>
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

/// Scheduler-only hard ordering constraints layered over an immutable base DAG.
class HardSchedulingConstraintOverlay {
   public:
    explicit HardSchedulingConstraintOverlay(
        const std::vector<std::unordered_set<unsigned>>& baseGraph)
        : baseGraph_(baseGraph), graph_(baseGraph.size()), inDegree_(baseGraph.size(), 0) {}

    bool tryAdd(unsigned predecessor, unsigned successor) {
        assert(predecessor < graph_.size() && successor < graph_.size());
        if (graph_[predecessor].contains(successor)) return true;
        if (hasPath(successor, predecessor)) return false;
        graph_[predecessor].insert(successor);
        ++inDegree_[successor];
        return true;
    }

    bool isReady(unsigned nodeId, unsigned baseInDegree) const {
        assert(nodeId < inDegree_.size());
        return baseInDegree == 0 && inDegree_[nodeId] == 0;
    }

    const std::unordered_set<unsigned>& successors(unsigned nodeId) const {
        assert(nodeId < graph_.size());
        return graph_[nodeId];
    }

    void satisfyFrom(unsigned predecessor) {
        assert(predecessor < graph_.size());
        for (unsigned successor : graph_[predecessor]) {
            assert(inDegree_[successor] > 0);
            --inDegree_[successor];
        }
    }

    unsigned inDegree(unsigned nodeId) const {
        assert(nodeId < inDegree_.size());
        return inDegree_[nodeId];
    }

   private:
    bool hasPath(unsigned from, unsigned to) const {
        if (from == to) return true;

        std::vector<unsigned> pending{from};
        std::vector<bool> visited(graph_.size(), false);
        visited[from] = true;
        while (!pending.empty()) {
            const unsigned current = pending.back();
            pending.pop_back();
            for (const auto* graph : {&baseGraph_, &graph_}) {
                for (unsigned successor : (*graph)[current]) {
                    if (successor == to) return true;
                    if (!visited[successor]) {
                        visited[successor] = true;
                        pending.push_back(successor);
                    }
                }
            }
        }
        return false;
    }

    const std::vector<std::unordered_set<unsigned>>& baseGraph_;
    std::vector<std::unordered_set<unsigned>> graph_;
    std::vector<unsigned> inDegree_;
};

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

/// Print each DAG node and its successor IDs.
void dumpDAGGraph(const RegionDAG& dag, std::ostream& os);

}  // namespace dag
}  // namespace stinkytofu
