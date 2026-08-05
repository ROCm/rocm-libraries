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
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"

#include <climits>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"

// Before dag/CDNA*.hpp so PASS_DEBUG inside those headers uses this pass name.
#define DEBUG_TYPE "StinkyDAGSchedulerPass"

#include "dag/CDNA5.hpp"

namespace {
using namespace stinkytofu;

static void dumpDAGGraph(const std::vector<std::unordered_set<unsigned>>& dagGraph,
                         const DAGNodeList& dagNodes) {
    std::cerr << "*** DAG Graph Dump: ***\n";
    for (unsigned i = 0; i < dagGraph.size(); ++i) {
        std::cerr << "Node " << i << ": ";
        dagNodes[i].inst->dump(std::cerr);
        std::cerr << "  successors: ";
        for (unsigned succId : dagGraph[i]) {
            std::cerr << succId << " ";
        }
        std::cerr << "\n";
    }
    std::cerr << "\n\n";
}

// collapseExecMaskedRegions()/expandExecMaskedGroups(): see ExecMaskGrouping.hpp and
// docs/developer/exec-mask-grouping.md.

// A workgroup-scope barrier: a legacy `s_barrier`, or an all-wave (-1) split barrier
// signal/wait. Split barriers with any other id are cluster/expert scope.
static bool isWorkgroupBarrier(const StinkyInstruction& inst) {
    if (!isBarrier(inst)) return false;
    if (isBarrierSignal(inst) || isBarrierWait(inst)) return isSplitBarrierAllWave(inst);
    return true;
}

static bool writesScc(const StinkyInstruction& inst) {
    for (const StinkyRegister& reg : inst.getDestRegs())
        if (reg.isRegister() && reg.reg.type == RegType::SCC) return true;
    return false;
}

static bool readsScc(const StinkyInstruction& inst) {
    for (const StinkyRegister& reg : inst.getSrcRegs())
        if (reg.isRegister() && reg.reg.type == RegType::SCC) return true;
    return false;
}

// A workgroup barrier that guards a tensor_load, i.e. one InsertClusterBarrierPass
// will pick as a handshake anchor. `signal` and `wait` are the same node for a legacy
// single-instruction s_barrier.
struct GuardingBarrier {
    DAGNode* signal = nullptr;
    DAGNode* wait = nullptr;
};

// One SCC value: the def plus every reader of it inside the region, in program order.
// `def` is null when the value was defined in an earlier region (its readers can still
// be moved across a barrier, so it still needs pinning). `liveOut` marks a value with a
// reader outside the region -- the loop terminator, a later region, or a successor.
struct SccChain {
    DAGNode* def = nullptr;
    std::vector<DAGNode*> readers;
    bool liveOut = false;
};

// Mirrors InsertClusterBarrierPass::findPrecedingWorkgroupBarrierSignalInSegment: a
// tensor_load anchors on the nearest preceding workgroup barrier signal. Segment
// boundaries (labels, branches, calls) have side effects, so they already end the
// region -- scanning within the region is the same scan.
static std::vector<GuardingBarrier> collectGuardingBarriers(DAGNodeList& dagNodes) {
    std::vector<GuardingBarrier> guarding;
    GuardingBarrier pending;
    bool recorded = false;

    for (DAGNode& node : dagNodes) {
        const StinkyInstruction& inst = *node.inst;
        if (isWorkgroupBarrier(inst)) {
            if (isBarrierWait(inst)) {
                if (pending.signal != nullptr) pending.wait = &node;
            } else {
                // A signal (or a legacy s_barrier, which is its own wait) opens a pair.
                pending.signal = &node;
                pending.wait = isBarrierSignal(inst) ? nullptr : &node;
                recorded = false;
            }
            continue;
        }
        if (recorded || pending.signal == nullptr || !isTensorLoad(inst)) continue;
        guarding.push_back({pending.signal, pending.wait ? pending.wait : pending.signal});
        recorded = true;
    }
    return guarding;
}

static std::vector<SccChain> collectSccChains(
    DAGNodeList& dagNodes, const std::unordered_map<StinkyInstruction*, unsigned>& instToId) {
    std::vector<SccChain> chains;
    for (DAGNode& node : dagNodes) {
        StinkyInstruction& inst = *node.inst;
        // Read before write: an instruction that does both (s_addc, s_subb) closes the
        // current value and opens the next one.
        if (readsScc(inst)) {
            if (chains.empty()) chains.push_back(SccChain{});
            chains.back().readers.push_back(&node);
        }
        if (!writesScc(inst)) continue;

        SccChain chain;
        chain.def = &node;
        chains.push_back(std::move(chain));
    }

    // Only the region's last SCC def can still be live when the region ends; every
    // earlier one is killed by the def that follows it. Note getUsers() is a flat, per
    // instruction list covering every destination -- `s_sub_u32 s0, s0, 1` writes both
    // an SGPR and SCC, and its SGPR users would otherwise make every such decrement look
    // live-out -- so it has to be narrowed to the users that actually read SCC.
    if (!chains.empty() && chains.back().def != nullptr) {
        SccChain& last = chains.back();
        for (StinkyInstruction* user : last.def->inst->getUsers()) {
            if (!readsScc(*user) || instToId.count(user) != 0) continue;
            last.liveOut = true;
            break;
        }
    }
    return chains;
}

// Nodes reachable from \p start by following DAG edges, i.e. everything that cannot be
// scheduled before it.
static std::vector<char> reachableFrom(unsigned start,
                                       const std::vector<std::unordered_set<unsigned>>& dagGraph) {
    std::vector<char> seen(dagGraph.size(), 0);
    std::vector<unsigned> stack{start};
    seen[start] = 1;
    while (!stack.empty()) {
        const unsigned cur = stack.back();
        stack.pop_back();
        for (unsigned succ : dagGraph[cur]) {
            if (seen[succ]) continue;
            seen[succ] = 1;
            stack.push_back(succ);
        }
    }
    return seen;
}

// --- Special rule: cluster-barrier SCC protection (ClusterBarrier kernels only) ---
//
// InsertClusterBarrierPass runs at kernel scope AFTER this pass. For every tensor_load
// it finds the nearest preceding `s_barrier_signal -1` in the same segment and expands
// a handshake at or before it:
//
//     s_cmp_eq_u32   sgprWaveIdx, 0
//     s_cbranch_scc0 label_skipCBPreSignal_<hash>
//     s_barrier_signal -3
//   label_skipCBPreSignal_<hash>:
//
// The sequence clobbers SCC, so an SCC value whose live range spans that barrier is
// destroyed. The pass can sometimes rematerialize the producer
// (findLiveRestorableSccCmpUpstream), but it gives up as soon as a second SCC reader
// sits in between -- exactly the shape an unrolled loop body produces.
//
// So no def..last-reader range may contain a guarding barrier. Note what that does and
// does not forbid: the chain may still schedule wholly before or wholly after the
// barrier, and may still move as a whole in either direction. Only splitting it is
// illegal. That is a disjunction, which a DAG edge cannot express -- an edge would fix
// one side at graph-build time and cost the freedom to hoist. So the common case is
// enforced dynamically instead: the nodes are tagged here and CDNA5ReadyQueue refuses
// to issue a guarding barrier while a chain is open.
//
// A lock can only work when every reader is free to issue without the barrier going
// first, so a chain with a reader that depends on a guarding barrier is pinned after it
// with an ordinary edge instead. Those edges, and the live-out ones below, always run
// from a lower to a higher program-order id, as does every other edge in the region
// (the RAW/WAR/WAW loop above only ever links an earlier node to the node it is
// visiting), so program order stays a valid topological order and the graph stays
// acyclic.
//
// Barriers that do not guard a tensor_load are left alone: the pass plants nothing
// there, so SCC chains may straddle them at no cost.
static void applyClusterBarrierSccRule(DAGNodeList& dagNodes,
                                       const std::unordered_map<StinkyInstruction*, unsigned>& instToId,
                                       std::vector<std::unordered_set<unsigned>>& dagGraph) {
    const std::vector<GuardingBarrier> guarding = collectGuardingBarriers(dagNodes);
    if (guarding.empty()) return;

    for (const GuardingBarrier& barrier : guarding) {
        barrier.signal->guardingBarrier = true;
        barrier.wait->guardingBarrier = true;
    }

    std::vector<std::vector<char>> reach;
    reach.reserve(guarding.size());
    for (const GuardingBarrier& barrier : guarding)
        reach.push_back(reachableFrom(barrier.signal->id, dagGraph));

    unsigned nextChainId = 0;
    for (const SccChain& chain : collectSccChains(dagNodes, instToId)) {
        // Nothing reads the value inside the region and nothing outside does either:
        // it is dead here, so a clobber cannot hurt it.
        if (chain.readers.empty() && !chain.liveOut) continue;

        DAGNode* first = chain.def ? chain.def : chain.readers.front();
        DAGNode* last = chain.readers.empty() ? first : chain.readers.back();

        // A live-out value is read past the end of the region (the loop terminator, a
        // later region, a successor), so there is no reader here for the queue to close
        // the chain on, and no freedom to preserve either -- that reader is fixed at the
        // region end. Pin the def after every guarding barrier it already follows.
        if (chain.liveOut) {
            for (const GuardingBarrier& barrier : guarding) {
                if (barrier.wait->id >= first->id) continue;
                addEdgeById(barrier.wait, first, dagGraph);
                PASS_DEBUG(std::cerr << "[DAG schedule] cluster-barrier SCC rule: pinned live-out"
                                     << " chain (dagId=" << first->id << ") after barrier wait"
                                     << " (dagId=" << barrier.wait->id << ")\n");
            }
            continue;
        }

        bool alreadySplit = false;
        bool needsLock = false;
        std::vector<const GuardingBarrier*> pinAfter;
        for (size_t i = 0; i < guarding.size(); ++i) {
            const GuardingBarrier& barrier = guarding[i];
            if (barrier.signal->id > first->id && barrier.signal->id < last->id) {
                alreadySplit = true;
                break;
            }
            // The def already depends on the barrier, so the whole chain follows it and
            // there is nothing to keep apart.
            if (reach[i][first->id]) continue;

            bool readerDependsOnBarrier = false;
            for (const DAGNode* reader : chain.readers) {
                if (!reach[i][reader->id]) continue;
                readerDependsOnBarrier = true;
                break;
            }
            if (readerDependsOnBarrier)
                pinAfter.push_back(&barrier);
            else
                needsLock = true;
        }

        if (alreadySplit) {
            // The incoming order already spans the barrier, so the scheduler is not what
            // broke it and no ordering it can pick will put it back together.
            PASS_DEBUG(std::cerr << "[DAG schedule] cluster-barrier SCC rule: chain ["
                                 << first->id << ".." << last->id
                                 << "] already spans a guarding barrier; leaving it to the"
                                    " barrier pass\n");
            continue;
        }

        for (const GuardingBarrier* barrier : pinAfter) {
            if (barrier->wait->id >= first->id) continue;
            addEdgeById(barrier->wait, first, dagGraph);
            PASS_DEBUG(std::cerr << "[DAG schedule] cluster-barrier SCC rule: chain [" << first->id
                                 << ".." << last->id << "] has a reader depending on barrier (dagId="
                                 << barrier->signal->id << "); pinned after it instead of locking\n");
        }

        if (!needsLock) continue;

        const unsigned chainId = ++nextChainId;
        chain.def->sccChainId = chainId;
        chain.def->sccChainDef = true;
        chain.def->sccChainReaders = static_cast<unsigned>(chain.readers.size());
        for (DAGNode* reader : chain.readers) reader->sccChainId = chainId;
        PASS_DEBUG(std::cerr << "[DAG schedule] cluster-barrier SCC rule: chain [" << first->id
                             << ".." << last->id << "] locked as chain " << chainId << " ("
                             << chain.readers.size() << " readers)\n");
    }
}

// --- Region scheduler (does NOT move fences) ---
//
// Build a DAG within a region and perform a stable topological schedule.
// Adds RAW/WAR/WAW deps for physical regs and also respects explicitPreds
// (only when both endpoints are inside the region).
static void scheduleRegionWithMovableSideEffects(
    IRList::iterator regionStart, IRList::iterator regionEnd, IRList::iterator blockBegin,
    std::vector<IRBase*>& scheduled, ReadyQueue& readyQueue,
    const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex, int& fillerCount) {
    if (regionStart == regionEnd) {
        return;  // Empty region, nothing to schedule.
    }

    PASS_DEBUG(std::cerr << "Scheduling region with movable side effects:\n");
    PASS_DEBUG(for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        StinkyInstruction& inst = getStinkyInst(it);
        inst.dump(std::cerr);
    });
    PASS_DEBUG(std::cerr << "\n");

    unsigned regionSize = std::distance(regionStart, regionEnd);

    std::string regionBbLabel;
    if (regionStart != regionEnd) {
        if (BasicBlock* pbb = getStinkyInst(regionStart).getParent())
            regionBbLabel = pbb->getLabel();
    }

    // Map each instruction to an unique id [0..n-1]
    DAGNodeList dagNodes;
    dagNodes.reserve(regionSize);

    unsigned id = 0;
    for (IRList::iterator it = regionStart; it != regionEnd; ++it) {
        dagNodes.emplace_back(&getStinkyInst(it), id++);
    }

    // Reverse lookup for the hazard pre-scan below (find a consumer instruction's id
    // in O(1) instead of rescanning dagNodes per BFS hit).
    std::unordered_map<StinkyInstruction*, unsigned> instToId;
    instToId.reserve(regionSize);
    for (unsigned i = 0; i < regionSize; ++i) instToId[dagNodes[i].inst] = i;

    // Graph
    std::vector<std::unordered_set<unsigned>> dagGraph(regionSize);

    // Track last read/write per physreg inside the region
    /* To ensure correct node dependency, lastRead should track all
     * previous read nodes until the register is overwritten. */
    std::map<StinkyRegister, std::unordered_set<DAGNode*>> lastRead;
    std::map<StinkyRegister, DAGNode*> lastWrite;

    // Build deps graph - same as before for register dependencies
    for (unsigned i = 0; i < dagNodes.size(); ++i) {
        DAGNode& dagNode = dagNodes[i];
        StinkyInstruction& inst = *dagNode.inst;

        // RAW deps:
        // For each source register, add an edge to the last writer of that register.
        for (const StinkyRegister& srcReg : inst.getSrcRegs()) {
            if (!srcReg.isRegister()) continue;

            for (unsigned off = 0; off < srcReg.reg.num; ++off) {
                StinkyRegister reg(srcReg.reg.type, srcReg.reg.idx + off, 1);
                auto itLastWrite = lastWrite.find(reg);
                // Only add edge if the last writer is in the region.
                if (itLastWrite != lastWrite.end()) {
                    DAGNode* lastWriter = itLastWrite->second;
                    addEdgeById(lastWriter, &dagNode, dagGraph);
                }
                // Add node to track the last read of this register
                lastRead[reg].insert(&dagNode);
            }
        }

        // WAW/WAR deps for defs
        for (const StinkyRegister& dstReg : inst.getDestRegs()) {
            if (!dstReg.isRegister()) continue;

            for (unsigned off = 0; off < dstReg.reg.num; ++off) {
                StinkyRegister reg(dstReg.reg.type, dstReg.reg.idx + off, 1);

                // WAW: previous writer of reg must come before this writer
                auto itLastWrite = lastWrite.find(reg);

                // Only add edge if the last writer is in the region.
                if (itLastWrite != lastWrite.end()) {
                    DAGNode* lastWriter = itLastWrite->second;
                    addEdgeById(lastWriter, &dagNode, dagGraph);
                }

                // WAR: previous reader of r must come before this writer
                auto itLastRead = lastRead.find(reg);

                // Only add edge if the last reader is in the region.
                if (itLastRead != lastRead.end()) {
                    for (DAGNode* lastReader : itLastRead->second) {
                        addEdgeById(lastReader, &dagNode, dagGraph);
                    }
                    // Clear last read tracking for this register due to it's overwritten
                    lastRead.erase(reg);
                }

                // track the last write for this register
                lastWrite[reg] = &dagNode;
            }
        }
    }

    if (readyQueue.getPassContext().getPassFeatureConfig().dagFeatures.clusterBarrier)
        applyClusterBarrierSccRule(dagNodes, instToId, dagGraph);

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

    // Prefix sum over the region in original program order: cumCycles[k] = the
    // estimated absolute cycle at which dagNodes[k] would start, if the unmodified
    // program order were followed exactly (WMMA -> latencyCycles, its full co-issue
    // window; otherwise issueCycles). Used below to turn "producer must precede its
    // consumer by N cycles" into a plain deadline number instead of a node to hop
    // before — see DAGNode::hazardDeadline.
    std::vector<int> cumCycles(regionSize + 1, 0);
    for (unsigned k = 0; k < regionSize; ++k) {
        StinkyInstruction* inst = dagNodes[k].inst;
        cumCycles[k + 1] =
            cumCycles[k] + (isMatrixInstruction(*inst) ? inst->latencyCycles : inst->issueCycles);
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
    // deadline is X - rule.cycles - producerCost: the gate is stamped only after this
    // producer's own advanceTime has already run (see popNonWmma), so the deadline
    // must reserve that cost too -- using X - rule.cycles alone would let the
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
                // (matches the gate: hazardGates_ is stamped to rule.cycles only after
                // updateWMMAStatus has already advanced clock_ by the producer's own
                // cost). So the deadline for issuing it must also subtract that cost --
                // otherwise "clock_ >= deadline" would let it start exactly one cycle
                // too late relative to X.
                const int producerCost =
                    isMatrixInstruction(*prod) ? prod->latencyCycles : prod->issueCycles;
                // rule.cycles == -1: "hoist as far as possible" mode. Force the deadline
                // to 0 so decidePromote() issues this producer the instant it is free,
                // maximizing its distance from the consumer instead of targeting a fixed gap.
                const int deadline =
                    rule.cycles < 0 ? 0 : cumCycles[ruleConsumerId] - rule.cycles - producerCost;
                bestDeadline = std::min(bestDeadline, deadline);
            }
        }

        if (!dagNodes[i].hazardFlags.empty()) dagNodes[i].hazardDeadline = bestDeadline;
    }

    PASS_DEBUG(dumpDAGGraph(dagGraph, dagNodes));

    readyQueue.onInitRegion(regionStart, regionEnd, blockBegin);

    // Kahn's algorithm with stable pick (by original order)

    assert(readyQueue.empty() && "Ready queue must be empty before scheduling a region");

    // Initialize the ready queue with instructions that have in-degree 0.
    for (unsigned i = 0; i < regionSize; ++i) {
        if (dagNodes[i].inDegree == 0) {
            readyQueue.push(&dagNodes[i]);
        }
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
}

// Schedule the instructions in the given IRList.
// This will split the instructions into regions based on side-effect instructions
// and schedule each region in a DAG.
//
// In the end, the instructions will be reordered in the block
// to reflect the scheduling order.
static void scheduleInDAG(BasicBlock& bb, ReadyQueue& readyQueue,
                          const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex) {
    PASS_DEBUG(std::cerr << "*** Scheduling Instructions in DAG: ***\n");

    if (bb.empty()) return;

    std::vector<IRBase*> scheduled;
    scheduled.reserve(bb.size());
    // Filler instructions the ready queue emits during this block (detached; attached by
    // the reorder loop). Grows both `scheduled` and the final block, so the size check
    // adds it to bb.size().
    int fillerCount = 0;

    BasicBlock::iterator beginIt = bb.begin();
    BasicBlock::iterator endIt = bb.end();

    readyQueue.onInit(beginIt, endIt);

    BasicBlock::iterator regionStart = beginIt;

    for (BasicBlock::iterator it = beginIt; it != endIt; ++it) {
        IRBase* irNode = it.getNodePtr();
        auto* instPtr = dyn_cast<StinkyInstruction>(irNode);

        if (!instPtr) {
            // Non-instruction IR (e.g. AsmDirective): treat as non-movable
            // side-effect boundary so its position is strictly preserved.
            scheduleRegionWithMovableSideEffects(regionStart, it, beginIt, scheduled, readyQueue,
                                                 wmmaIndex, fillerCount);
            scheduled.push_back(irNode);
            regionStart = std::next(it);
            continue;
        }

        StinkyInstruction& inst = *instPtr;
        if (hasSideEffect(inst)) {
            scheduleRegionWithMovableSideEffects(regionStart, it, beginIt, scheduled, readyQueue,
                                                 wmmaIndex, fillerCount);

            scheduled.push_back(&inst);

            PASS_DEBUG(std::cerr << "Scheduling non-movable side-effect instruction:\n";
                       inst.dump(std::cerr); std::cerr << "\n");

            // Start a new region after the side-effect instruction.
            regionStart = std::next(it);
        }
    }
    // Flush the last region if it has not been flushed yet.
    scheduleRegionWithMovableSideEffects(regionStart, endIt, beginIt, scheduled, readyQueue,
                                         wmmaIndex, fillerCount);

    assert(scheduled.size() == bb.size() + static_cast<size_t>(fillerCount) &&
           "Scheduled instructions size must match original plus filler insts");

    // Now we have a scheduled list of instructions.
    // Reorder the block to reflect the scheduling (move each to end in order). Original
    // instructions already live in bb (remove+append repositions them); filler
    // instructions are detached (no parent) and are only appended.
    for (IRBase* ir : scheduled) {
        if (ir->getParent()) bb.removeIR(ir);
        bb.appendIR(ir);
    }

    readyQueue.onFinishBB();
}

std::unique_ptr<ReadyQueue> chooseReadyQueue(const PassContext& passCtx) {
    if (passCtx.getGemmTileConfig().arch[0] == 12 && passCtx.getGemmTileConfig().arch[1] == 5) {
        PASS_DEBUG(std::cerr << "Using CDNA5ReadyQueue for scheduling\n");
        return std::make_unique<CDNA5ReadyQueue>(passCtx);
    } else {
        PASS_DEBUG(std::cerr << "Using Default ReadyQueue for scheduling\n");
        return std::make_unique<ReadyQueueByDAGid>(passCtx);
    }
}

class StinkyDAGSchedulerPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "StinkyDAGSchedulerPass";
    }

    PassID getPassID() const override {
        return &StinkyDAGSchedulerPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        // Build def-use chains so we can look up cross-BB WMMA consumers
        // of ds_reads for wmmaAffinity annotation.
        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        buildUseDefChain(func, domInfo, true);

        const auto& rpo = AM.getResult<BBIndexAnalysis>(func).rpo;

        // Pre-assign a function-wide index to each WMMA/SWMMA so wmmaAffinity
        // values are comparable across scheduling regions.
        std::unordered_map<StinkyInstruction*, unsigned> wmmaIndex;
        {
            unsigned idx = 0;
            for (auto* bb : rpo) {
                for (auto it = bb->begin(); it != bb->end(); ++it) {
                    auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                    if (!inst) continue;
                    if (isMatrixInstruction(*inst)) wmmaIndex[inst] = idx++;
                }
            }
        }

        const auto& loops = AM.getResult<LoopAnalysis>(func);

        PASS_DEBUG(for (const Loop& loop
                        : loops) {
            std::cerr << "[LoopDetection] Loop: header="
                      << (loop.headerBB ? loop.headerBB->getLabel() : "?")
                      << " latch=" << (loop.latchBB ? loop.latchBB->getLabel() : "?") << "\n";
            for (BasicBlock* bb : loop.bodyBBs) {
                std::cerr << "  body: " << bb->getLabel() << " ->";
                for (BasicBlock* succ : bb->getSuccessors()) std::cerr << " " << succ->getLabel();
                std::cerr << "\n";
            }
        });

        // Cross-BB scheduling state shared across all BBs.
        // Written by all BBs in onFinishBB, read only by loop body BBs in onInit.
        ScheduleAnalysisCache analysisCache;

        // Per-loop ReadyQueue: shared across loop body BBs for loop-specific
        // scheduling state (wmmaNodeCounters, evenly-split config).
        std::map<const Loop*, std::unique_ptr<ReadyQueue>> loopQueues;

        // Map only loop body BBs to their loop — shared queue for loop iterations.
        std::unordered_map<BasicBlock*, const Loop*> bbToLoop;
        for (const Loop& loop : loops) {
            for (BasicBlock* bb : loop.bodyBBs) bbToLoop[bb] = &loop;
        }

        const GfxArchID archId =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);
        const uint32_t wavefrontSize = passCtx.getWavefrontSize();

        auto scheduleBlock = [&](BasicBlock* bb, ReadyQueue& rq) {
            AsmIRBuilder builder(*bb, archId);
            collapseExecMaskedRegions(*bb, builder, wavefrontSize);
            scheduleInDAG(*bb, rq, wmmaIndex);
            expandExecMaskedGroups(*bb);
        };

        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) continue;

            auto it = bbToLoop.find(bb);
            if (it != bbToLoop.end()) {
                const Loop* loop = it->second;
                auto& rq = loopQueues[loop];
                if (!rq) {
                    rq = chooseReadyQueue(passCtx);
                    rq->setLoopContext(loop);
                }
                rq->setAnalysisCache(&analysisCache);
                scheduleBlock(bb, *rq);
            } else {
                auto rq = chooseReadyQueue(passCtx);
                rq->setAnalysisCache(&analysisCache);
                scheduleBlock(bb, *rq);
            }
        }
        return preserveCFGAnalyses();
    }
};

char StinkyDAGSchedulerPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyDAGSchedulerPass() {
    return std::make_unique<StinkyDAGSchedulerPass>();
}
}  // namespace stinkytofu
