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
#include "RegionScheduling.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <unordered_set>

#define DEBUG_TYPE "StinkyDAGSchedulerPass"

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

namespace stinkytofu {
namespace dag {

std::vector<int> computeCumulativeCycles(const RegionDAG& dag) {
    const unsigned regionSize = static_cast<unsigned>(dag.nodes.size());
    std::vector<int> cumCycles(regionSize + 1, 0);
    for (unsigned k = 0; k < regionSize; ++k) {
        const StinkyInstruction* inst = dag.nodes[k].inst;
        cumCycles[k + 1] =
            cumCycles[k] + (isMatrixInstruction(*inst) ? inst->latencyCycles : inst->issueCycles);
    }
    return cumCycles;
}

PreparedRegion prepareRegionForScheduling(
    RegionDAG dag, const std::vector<int>& cumCycles, IRList::iterator regionStart,
    IRList::iterator regionEnd, IRList::iterator blockBegin, ReadyQueue& readyQueue,
    const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex) {
    PreparedRegion prepared;
    prepared.dag = std::move(dag);

    RegionDAG& regionDag = prepared.dag;
    DAGNodeList& dagNodes = regionDag.nodes;
    std::vector<std::unordered_set<unsigned>>& dagGraph = regionDag.graph;
    std::unordered_map<StinkyInstruction*, unsigned>& instToId = regionDag.instToId;
    const unsigned regionSize = static_cast<unsigned>(dagNodes.size());

    if (regionStart != regionEnd) {
        if (BasicBlock* pbb = getStinkyInst(regionStart).getParent())
            prepared.bbLabel = pbb->getLabel();
    }
    if (regionSize == 0) return prepared;

    // Pre-scan: assign dsReadPriority to each ds_read based on WMMA affinity
    // and DsReadOrder config. Lower priority = pick first.
    {
        using DsReadOrder = PassFeatureConfig::DsReadOrder;
        const auto dsOrder =
            readyQueue.getPassContext().getPassFeatureConfig().dagFeatures.dsReadOrder;

        // Collect ds_reads with their affinity and operand type (src register).
        struct DsInfo {
            unsigned idx, affinity, srcReg;
        };
        std::vector<DsInfo> dsReads;

        for (unsigned i = 0; i < regionSize; ++i) {
            if (!isDSRead(*dagNodes[i].inst)) continue;

            unsigned affinity = UINT_MAX;
            // BFS through users, skip PHIs, find earliest WMMA consumer.
            std::vector<StinkyInstruction*> q(dagNodes[i].inst->getUsers().begin(),
                                              dagNodes[i].inst->getUsers().end());
            std::unordered_set<StinkyInstruction*> seen;
            while (!q.empty()) {
                StinkyInstruction* u = q.back();
                q.pop_back();
                if (!seen.insert(u).second) continue;
                if (u->getUnifiedOpcode() == GFX::PHI) {
                    for (auto* pu : u->getUsers()) q.push_back(pu);
                    continue;
                }
                auto it = wmmaIndex.find(u);
                if (it != wmmaIndex.end()) affinity = std::min(affinity, it->second);
            }

            unsigned srcReg = 0;
            for (const StinkyRegister& s : dagNodes[i].inst->getSrcRegs())
                if (s.isRegister()) {
                    srcReg = s.reg.idx;
                    break;
                }

            dsReads.push_back({i, affinity, srcReg});
        }

        // Sort by affinity, then by DAG id.
        std::sort(dsReads.begin(), dsReads.end(), [](const DsInfo& a, const DsInfo& b) {
            return a.affinity != b.affinity ? a.affinity < b.affinity : a.idx < b.idx;
        });

        if (dsOrder == DsReadOrder::ProgramOrder) {
            for (auto& d : dsReads) dagNodes[d.idx].dsReadPriority = d.idx;
        } else {
            // For AscendingCache: find first single-operand affinity group,
            // then zigzag backward through mixed groups.
            // For Ascending: all groups use ascending order.
            std::map<unsigned, std::set<unsigned>> groupSrcRegs;
            for (auto& d : dsReads) groupSrcRegs[d.affinity].insert(d.srcReg);

            // Determine sort direction for mixed groups via look-ahead.
            // Both Ascending and AscendingCache use look-ahead to find the
            // first single-operand group and load the absent operand first.
            // Ascending: all mixed groups use the same direction.
            // AscendingCache: mixed groups zigzag.
            std::map<unsigned, bool> groupAsc;  // affinity → ascending?
            {
                std::vector<unsigned> mixedAffinities;
                for (auto& [aff, regs] : groupSrcRegs)
                    if (regs.size() > 1) mixedAffinities.push_back(aff);

                bool hasSingleOpGroup = (groupSrcRegs.size() > mixedAffinities.size());

                if (dsOrder == DsReadOrder::AscendingCache && !mixedAffinities.empty()) {
                    // AscendingCache: always zigzag for cache reuse.
                    // If single-op anchor exists, work backward from it.
                    // Otherwise, first group ascending, then alternate.
                    bool asc = false;  // last mixed group descending for cache reuse
                    for (int i = (int)mixedAffinities.size() - 1; i >= 0; --i) {
                        groupAsc[mixedAffinities[i]] = asc;
                        asc = !asc;
                    }
                } else if (hasSingleOpGroup && !mixedAffinities.empty()) {
                    // Ascending with single-op anchor: load absent operand first.
                    // All mixed groups use the same direction.
                    bool asc = false;
                    for (int i = (int)mixedAffinities.size() - 1; i >= 0; --i)
                        groupAsc[mixedAffinities[i]] = asc;
                }
                // Ascending without anchor: groupAsc empty → default ascending.
            }

            // Assign priority. Within each group, sort by DAG id
            // (ascending or descending per groupAsc).
            unsigned pri = 0;
            unsigned prevAff = UINT_MAX;
            std::vector<DsInfo*> group;
            auto flushGroup = [&]() {
                if (group.empty()) return;
                bool asc = groupAsc.contains(prevAff) ? groupAsc[prevAff] : true;
                if (!asc) {
                    // Reverse operand type order but keep DAG id order within
                    // each type. Sort by (srcReg descending, idx ascending).
                    std::stable_sort(
                        group.begin(), group.end(),
                        [](const DsInfo* a, const DsInfo* b) { return a->srcReg > b->srcReg; });
                }
                for (auto* d : group) dagNodes[d->idx].dsReadPriority = pri++;
                group.clear();
            };
            for (auto& d : dsReads) {
                if (d.affinity != prevAff) {
                    flushGroup();
                    prevAff = d.affinity;
                }
                group.push_back(&d);
            }
            flushGroup();
        }
    }

    // Pre-scan: flag producers feeding a hazarded consumer, per the arch's hazard rule
    // table (a data-driven table of fixed producer->consumer cycle gaps keyed by register
    // file — e.g. SALU sgpr -> SMEM/tensor_load/VMEM address, VALU vgpr -> VMEM
    // address). Detection per rule: BFS the node's users (skipping PHIs); if a
    // rule.isConsumer user reads a register of rule.regType this node writes, flag it
    // (dagNodes[i].hazardFlags). This half drives the consumer-side gate
    // (CDNA5ReadyQueue::hazardGates_), which blocks the consumer for as long as real
    // intervening instructions are available to pay the wait -- but see
    // DAGNode::hazardDeadline's comment (ReadyQueue.hpp) for the case where they run
    // out and the scheduler's pre-existing "pay the wait via advanceTime, then issue
    // anyway" fallback applies instead.
    //
    // Also computes each flagged producer's hazardDeadline: a throughput heuristic
    // that, when accurate, is what keeps the gate above from ever needing that
    // fallback. Let X = cumCycles[consumerId], the hazarded consumer's estimated
    // absolute cycle (per rule; a producer feeding several consumers, or matching
    // several rules, takes the earliest/tightest deadline over all of them). The
    // deadline is X - rule.distance - producerCost: the gate is stamped only after this
    // producer's own advanceTime has already run (see popNonWmma), so the deadline
    // must reserve that cost too -- using X - rule.distance alone would let the
    // producer start one cost-unit later than it needs to.
    // CDNA5ReadyQueue::decidePromote() forces the producer once its *live* clock_
    // reaches this deadline, not once some proxy node happens to become structurally
    // ready -- clock_ only advances via cycles actually issued, so an unrelated node
    // becoming ready early can't trigger an early force the way a node-based trigger
    // could. Still approximate (X is computed from original program order, which real
    // scheduling may depart from), so it is not a substitute for the gate -- an
    // inaccurate deadline can leave the gate short of real cycles, same as the
    // producer-cost bug this fixed.
    // Same per-arch CDNA5 hazard-rule table the ready queue uses, so the pre-scan's
    // ruleIdx values line up with CDNA5ReadyQueue::hazardGates_ lanes.
    const HWModel& hw = readyQueue.getPassContext().getHWModel();
    for (unsigned i = 0; i < regionSize; ++i) {
        StinkyInstruction* prod = dagNodes[i].inst;
        int bestDeadline = INT_MAX;

        // MSB-affinity tiebreak input (see DAGNode::requiredMsb); -1 = no MSB opinion.
        auto [msbVal, msbHasVgpr] = computeRequiredMsb(prod);
        dagNodes[i].requiredMsb = msbHasVgpr ? msbVal : -1;

        for (int ruleIdx = 0; ruleIdx < hw.hazards.numRules; ++ruleIdx) {
            const HazardRule& rule = hw.hazards.rules[ruleIdx];
            // Def-use discovery only finds write->read pairs; a WAR partner is a later
            // writer and never appears in getUsers(). Those rules are stamped at issue
            // time by the ready queue instead.
            if (rule.dir != HazardDir::WriteThenRead) continue;
            if (!rule.isProducer(*prod)) continue;

            std::unordered_map<uint32_t, int> defKey;
            for (const StinkyRegister& d : prod->getDestRegs()) {
                if (!d.isRegister() || isPseudoReg(d) || d.reg.type != rule.regType) continue;
                for (uint32_t off = 0; off < d.reg.num; ++off)
                    defKey[d.reg.idx + off] = regDepKey(d.reg.type, d.reg.idx + off);
            }
            if (defKey.empty()) continue;

            std::unordered_set<int> hazardKeys;
            unsigned ruleConsumerId = UINT_MAX;
            std::vector<StinkyInstruction*> q(prod->getUsers().begin(), prod->getUsers().end());
            std::unordered_set<StinkyInstruction*> seen;
            while (!q.empty()) {
                StinkyInstruction* u = q.back();
                q.pop_back();
                if (!seen.insert(u).second) continue;
                if (u->getUnifiedOpcode() == GFX::PHI) {
                    for (auto* pu : u->getUsers()) q.push_back(pu);
                    continue;
                }
                if (!rule.isConsumer(*u)) continue;
                bool matchedHere = false;
                for (const StinkyRegister& s : u->getSrcRegs()) {
                    if (!s.isRegister() || isPseudoReg(s) || s.reg.type != rule.regType) continue;
                    for (uint32_t off = 0; off < s.reg.num; ++off) {
                        auto it = defKey.find(s.reg.idx + off);
                        if (it != defKey.end()) {
                            hazardKeys.insert(it->second);
                            matchedHere = true;
                        }
                    }
                }
                if (matchedHere) {
                    auto idIt = instToId.find(u);
                    if (idIt != instToId.end())
                        ruleConsumerId = std::min(ruleConsumerId, idIt->second);
                }
            }
            if (hazardKeys.empty()) continue;
            for (int key : hazardKeys) dagNodes[i].hazardFlags.push_back({ruleIdx, key});
            if (ruleConsumerId != UINT_MAX) {
                // The gap is measured from this producer's own FINISH, not its start
                // (matches the gate: hazardGates_ is stamped to rule.distance only after
                // updateWMMAStatus has already advanced clock_ by the producer's own
                // cost). So the deadline for issuing it must also subtract that cost --
                // otherwise "clock_ >= deadline" would let it start exactly one cycle
                // too late relative to X.
                const int producerCost =
                    isMatrixInstruction(*prod) ? prod->latencyCycles : prod->issueCycles;
                // rule.distance == -1: "hoist as far as possible" mode. Force the deadline
                // to 0 so decidePromote() issues this producer the instant it is free,
                // maximizing its distance from the consumer instead of targeting a fixed gap.
                const int deadline = rule.distance < 0
                                         ? 0
                                         : cumCycles[ruleConsumerId] - rule.distance - producerCost;
                bestDeadline = std::min(bestDeadline, deadline);
            }
        }

        if (!dagNodes[i].hazardFlags.empty()) dagNodes[i].hazardDeadline = bestDeadline;
    }

    // Scheduler-policy ordering requests are merged straight into the register-dependency
    // DAG, same as any other edge. A candidate edge is dropped when it would form a cycle
    // (these orderings are heuristic, not derived from real data dependencies, so
    // contradictory requests across barrier groups are possible).
    std::vector<HardSchedulingConstraint> requestedConstraints;
    const dag::RegionDependencies regionDeps{.dag = regionDag,
                                             .requestedConstraints = requestedConstraints};
    readyQueue.onInitRegion(regionStart, regionEnd, blockBegin, regionDeps);
    // Provenance only (not consulted by scheduling): which merged dagGraph edges are
    // policy-injected rather than real register dependencies, so debug output can still
    // tell them apart now that both live in the same graph.
    std::set<std::pair<unsigned, unsigned>>& mergedHardConstraintEdges = prepared.policyEdges;
    for (const auto& [predecessor, successor] : requestedConstraints) {
        auto predecessorIt = instToId.find(predecessor);
        auto successorIt = instToId.find(successor);
        if (predecessorIt == instToId.end() || successorIt == instToId.end()) continue;

        const unsigned predecessorId = predecessorIt->second;
        const unsigned successorId = successorIt->second;
        if (dagGraph[predecessorId].contains(successorId)) continue;
        if (dag::hasPath(dagGraph, successorId, predecessorId)) {
            PASS_DEBUG(std::cerr << "[DAG hard constraint] skip cycle-forming link "
                                 << predecessorId << " -> " << successorId << "\n");
            continue;
        }
        dagGraph[predecessorId].insert(successorId);
        ++dagNodes[successorId].inDegree;
        mergedHardConstraintEdges.emplace(predecessorId, successorId);
        PASS_DEBUG(std::cerr << "[DAG hard constraint] add link " << predecessorId << " -> "
                             << successorId << "\n");
    }

    PASS_DEBUG(dag::dumpDAGGraph(regionDag, std::cerr, mergedHardConstraintEdges));

    return prepared;
}

void drainReadyQueue(PreparedRegion& region, ReadyQueue& readyQueue,
                     std::vector<IRBase*>& scheduled, int& fillerCount) {
    RegionDAG& regionDag = region.dag;
    DAGNodeList& dagNodes = regionDag.nodes;
    std::vector<std::unordered_set<unsigned>>& dagGraph = regionDag.graph;
    const unsigned regionSize = static_cast<unsigned>(dagNodes.size());
    const std::string& regionBbLabel = region.bbLabel;

    // Kahn's algorithm with stable pick (by original order)

    assert(readyQueue.empty() && "Ready queue must be empty before scheduling a region");

    // Initialize the ready queue with instructions that have in-degree 0.
    for (unsigned i = 0; i < regionSize; ++i) {
        if (dagNodes[i].inDegree == 0) readyQueue.push(&dagNodes[i]);
    }

    // Process the ready queue until it's empty.
    unsigned orderInRegion = 0;
    while (!readyQueue.empty()) {
        // Pop the last instruction from the ready queue.
        DAGNode* currentNode = readyQueue.pickOne();
        ++orderInRegion;

        // Filler instructions the queue emits before this pick; detached so the reorder
        // loop places them in order. The queue owns any arch/opcode knowledge.
        for (StinkyInstruction* filler : readyQueue.takePendingFillerInsts()) {
            PASS_DEBUG(std::cerr << "[DAG drain] emitting filler inst before dagId="
                                 << currentNode->id << "\n");
            scheduled.push_back(filler);
            ++fillerCount;
        }

        if (isBarrier(*currentNode->inst)) {
            PASS_DEBUG(std::cerr << "[DAG schedule] bb=\"" << regionBbLabel << "\" orderInRegion="
                                 << orderInRegion << " dagId=" << currentNode->id
                                 << " movable barrier (position in region schedule)\n";
                       currentNode->inst->dump(std::cerr); std::cerr << "\n");
        }

        // Add the instruction to the scheduled list.
        scheduled.push_back(currentNode->inst);

        // Process all successors of the current node.
        for (unsigned succId : dagGraph[currentNode->id]) {
            DAGNode& succNode = dagNodes[succId];
            succNode.inDegree--;

            // If the successor now has in-degree 0, add it to the ready queue.
            if (succNode.inDegree == 0) {
                readyQueue.push(&succNode);
            }
        }
    }
    assert(orderInRegion == regionSize &&
           "Hard scheduling constraints must not leave unscheduled DAG nodes");
}

}  // namespace dag
}  // namespace stinkytofu
