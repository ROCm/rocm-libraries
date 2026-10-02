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

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ReadyQueue.hpp"
#include "RegionDAG.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace dag {

/// Estimated absolute cycle at which each node would start if the unmodified
/// program order were followed exactly: a matrix op spends its whole latency
/// window, anything else its issue cycles. The last entry is the region's
/// estimated length.
///
/// Turns a "must be within N cycles of" requirement into a plain clock number,
/// which is what DAGNode::hazardDeadline and DAGNode::earliestClock are built
/// from.
std::vector<int> computeCumulativeCycles(const RegionDAG& dag);

/// A region whose DAG is ready to hand to a ready queue.
struct PreparedRegion {
    RegionDAG dag;
    /// Which merged edges are policy-injected rather than real register
    /// dependencies. Provenance for debug output only; scheduling does not
    /// consult it, because by then both kinds live in the same graph.
    std::set<std::pair<unsigned, unsigned>> policyEdges;
    std::string bbLabel;
};

/// Populate the DAGNode fields a ready queue's contract requires, hand the
/// region to \p readyQueue, and merge back the orderings it asks for.
///
/// This is the preparation every queue needs, not a scheduling policy: the
/// pre-scans fill in ds-read priority, the per-arch hazard flags and deadlines,
/// and the MSB affinity that the queue reads but cannot compute for itself.
/// Policy that belongs to one caller — the cluster-barrier SCC rule, for
/// instance — stays with that caller and should be applied to \p dag first.
///
/// \p cumCycles must come from computeCumulativeCycles(\p dag), taken before any
/// caller-specific edges are merged so the hazard deadlines measure original
/// program order.
PreparedRegion prepareRegionForScheduling(
    RegionDAG dag, const std::vector<int>& cumCycles, IRList::iterator regionStart,
    IRList::iterator regionEnd, IRList::iterator blockBegin, ReadyQueue& readyQueue,
    const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex);

/// Kahn scheduling of \p region through \p readyQueue, appending to \p scheduled.
///
/// Instructions the queue asks to emit ahead of a pick (v_nop spacers) are
/// appended in order and counted in \p fillerCount. The queue owns any
/// arch/opcode knowledge behind them.
void drainReadyQueue(PreparedRegion& region, ReadyQueue& readyQueue,
                     std::vector<IRBase*>& scheduled, int& fillerCount);

}  // namespace dag
}  // namespace stinkytofu
