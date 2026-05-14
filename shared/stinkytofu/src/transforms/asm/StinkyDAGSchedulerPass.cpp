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

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

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

static bool hasLdsPseudoRegs(const StinkyInstruction& inst) {
    for (const StinkyRegister& r : inst.getSrcRegs())
        if (r.isRegister() && r.reg.type == RegType::LDS) return true;
    for (const StinkyRegister& r : inst.getDestRegs())
        if (r.isRegister() && r.reg.type == RegType::LDS) return true;
    return false;
}

// --- Region scheduler (does NOT move fences) ---
//
// Build a DAG within a region and perform a stable topological schedule.
// Adds RAW/WAR/WAW deps for physical regs and also respects explicitPreds
// (only when both endpoints are inside the region).
static void scheduleRegionWithMovableSideEffects(
    IRList::iterator regionStart, IRList::iterator regionEnd, IRList::iterator blockBegin,
    std::vector<IRBase*>& scheduled, ReadyQueue& readyQueue,
    const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex) {
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
                bool asc = groupAsc.count(prevAff) ? groupAsc[prevAff] : true;
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

static bool hasSideEffect(const StinkyInstruction& inst) {
    if (isGlobalMemStore(inst) || isBranch(inst) || isWaitCnt(inst) || isHasSideEffect(inst)) {
        return true;
    }

    // Barriers and memory ops without LDS pseudo-registers (no MemTokenData
    // assigned) must be treated conservatively as non-movable side effects to
    // preserve strict ordering. When LDS pseudo-regs are present, ordering is
    // enforced by the DAG via def-use edges, so they are safe to schedule.
    if ((isBarrier(inst) || isTensorLoad(inst) || isDSRead(inst) || isDSWrite(inst)) &&
        !hasLdsPseudoRegs(inst)) {
        return true;
    }
    return false;
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
                                                 wmmaIndex);
            scheduled.push_back(irNode);
            regionStart = std::next(it);
            continue;
        }

        StinkyInstruction& inst = *instPtr;
        if (hasSideEffect(inst)) {
            scheduleRegionWithMovableSideEffects(regionStart, it, beginIt, scheduled, readyQueue,
                                                 wmmaIndex);

            scheduled.push_back(&inst);

            PASS_DEBUG(std::cerr << "Scheduling non-movable side-effect instruction:\n";
                       inst.dump(std::cerr); std::cerr << "\n");

            // Start a new region after the side-effect instruction.
            regionStart = std::next(it);
        }
    }
    // Flush the last region if it has not been flushed yet.
    scheduleRegionWithMovableSideEffects(regionStart, endIt, beginIt, scheduled, readyQueue,
                                         wmmaIndex);

    assert(scheduled.size() == bb.size() &&
           "Scheduled instructions size must match original instructions size");

    // Now we have a scheduled list of instructions.
    // Reorder the block to reflect the scheduling (move each to end in order).
    for (IRBase* ir : scheduled) {
        bb.removeIR(ir);
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
                    if (isWMMA(*inst) || isSWMMA(*inst)) wmmaIndex[inst] = idx++;
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
                scheduleInDAG(*bb, *rq, wmmaIndex);
            } else {
                auto rq = chooseReadyQueue(passCtx);
                rq->setAnalysisCache(&analysisCache);
                scheduleInDAG(*bb, *rq, wmmaIndex);
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
