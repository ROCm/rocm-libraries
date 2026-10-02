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
#include <iterator>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/BBIndexAnalysis.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/analysis/asm/Layer2BarrierOverlapAnalysis.hpp"
#include "stinkytofu/analysis/controlflow/DominanceAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/support/CFGTraversal.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"

// Before dag/CDNA*.hpp so PASS_DEBUG inside those headers uses this pass name.
#define DEBUG_TYPE "StinkyDAGSchedulerPass"

#include "dag/ArchReadyQueue.hpp"
#include "dag/RegionDAG.hpp"
#include "dag/RegionScheduling.hpp"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::dag;

// collapseExecMaskedRegions()/expandExecMaskedGroups(): see ExecMaskGrouping.hpp and
// docs/developer/exec-mask-grouping.md.

// A workgroup-scope barrier: a legacy `s_barrier`, or an all-wave (-1) split barrier
// signal/wait. Split barriers with any other id are cluster/expert scope.
static bool isWorkgroupBarrier(const StinkyInstruction& inst) {
    if (!isBarrier(inst)) return false;
    if (isBarrierSignal(inst) || isBarrierWait(inst)) return isSplitBarrierAllWave(inst);
    return true;
}

// SCC is spelled two ways: as a descriptor flag, and as an ordinary operand once
// legalizeImplicitSpecialRegisters has materialized it. Ask about both, so the answer does
// not depend on where this pass sits relative to that one.
static bool writesScc(const StinkyInstruction& inst) {
    if (inst.is(InstFlag::IF_ImplicitWriteSCC)) return true;
    for (const StinkyRegister& reg : inst.getDestRegs())
        if (reg.isRegister() && reg.reg.type == RegType::SCC) return true;
    return false;
}

static bool readsScc(const StinkyInstruction& inst) {
    if (inst.is(InstFlag::IF_ImplicitReadSCC)) return true;
    for (const StinkyRegister& reg : inst.getSrcRegs())
        if (reg.isRegister() && reg.reg.type == RegType::SCC) return true;
    return false;
}

// A workgroup barrier, i.e. a place InsertClusterBarrierPass may expand an SCC-clobbering
// handshake. `signal` and `wait` are the same node for a legacy single-instruction
// s_barrier, and for a half pair whose other end lies in another region.
struct HandshakeBarrier {
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

// Every workgroup barrier in the region, paired up. Segment boundaries (labels, branches,
// calls) have side effects and so already end the region, which is why a scan of the
// region is the same scan InsertClusterBarrierPass makes of the segment.
static std::vector<HandshakeBarrier> collectHandshakeBarriers(DAGNodeList& dagNodes) {
    std::vector<HandshakeBarrier> barriers;
    DAGNode* pendingSignal = nullptr;

    for (DAGNode& node : dagNodes) {
        const StinkyInstruction& inst = *node.inst;
        if (!isWorkgroupBarrier(inst)) continue;
        if (isBarrierWait(inst) && pendingSignal != nullptr) {
            barriers.push_back({pendingSignal, &node});
            pendingSignal = nullptr;
            continue;
        }
        // Anything else closes whatever was open, and a signal left without its wait still
        // stands for itself: the pair closes in another region, and the spot a handshake
        // may be planted at is here either way.
        if (pendingSignal != nullptr) {
            barriers.push_back({pendingSignal, pendingSignal});
            pendingSignal = nullptr;
        }
        if (isBarrierSignal(inst))
            pendingSignal = &node;
        else
            barriers.push_back({&node, &node});
    }
    if (pendingSignal != nullptr) barriers.push_back({pendingSignal, pendingSignal});
    return barriers;
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

// Cluster-barrier SCC rule (ClusterBarrier kernels only). See cluster-barrier.md.
// Scheduler runs before InsertClusterBarrierPass; keeps chains off handshake barriers
// and pins live-out defs after waits. kRule3CrossLoop gates earliestClock on live-out defs.

// kRule3CrossLoop true only.
constexpr int kLiveOutSccDefLeadCycles = 50;

static void applyClusterBarrierSccRule(
    DAGNodeList& dagNodes, const std::unordered_map<StinkyInstruction*, unsigned>& instToId,
    std::vector<std::unordered_set<unsigned>>& dagGraph, int regionCycles) {
    const std::vector<HandshakeBarrier> barriers = collectHandshakeBarriers(dagNodes);
    if (barriers.empty()) return;

    for (const HandshakeBarrier& barrier : barriers) {
        barrier.signal->handshakeBarrier = true;
        barrier.wait->handshakeBarrier = true;
    }

    std::vector<std::vector<char>> reach;
    reach.reserve(barriers.size());
    for (const HandshakeBarrier& barrier : barriers)
        reach.push_back(reachableFrom(barrier.signal->id, dagGraph));

    unsigned nextChainId = 0;
    for (const SccChain& chain : collectSccChains(dagNodes, instToId)) {
        // Nothing reads the value inside the region and nothing outside does either:
        // it is dead here, so a clobber cannot hurt it.
        if (chain.readers.empty() && !chain.liveOut) continue;

        // ClusterBarrier: every SCC use in the region must follow a region SCC writer.
        // Live-in readers alone (def in an earlier region) are not supported.
        if (chain.def == nullptr) {
            STINKY_UNREACHABLE(
                "applyClusterBarrierSccRule: region has SCC reader(s) but no SCC writer");
        }

        DAGNode* first = chain.def;
        DAGNode* last = chain.readers.empty() ? first : chain.readers.back();

        // A live-out value is read past the end of the region (the loop terminator, a
        // later region, a successor), so there is no reader here for the queue to close
        // the chain on, and no freedom to preserve either -- that reader is fixed at the
        // region end. The range therefore reaches from the def to the end of the region,
        // and the only way for no barrier to fall inside it is for the def to follow every
        // barrier in the region -- including the ones it currently comes before.
        //
        // Those are edges that point back up the program order, so they are the one place
        // a cycle could be introduced. Skipping the barriers the def can reach is what
        // rules that out, and the skip costs little: a barrier takes no register operands,
        // so the only way to reach one is through an edge this rule itself added.
        if (chain.liveOut) {
            const std::vector<char> fromDef = reachableFrom(first->id, dagGraph);
            for (const HandshakeBarrier& barrier : barriers) {
                if (fromDef[barrier.wait->id]) continue;
                addEdgeById(barrier.wait, first, dagGraph);
                PASS_DEBUG(std::cerr << "[DAG schedule] cluster-barrier SCC rule: pinned live-out"
                                     << " chain (dagId=" << first->id << ") after barrier wait"
                                     << " (dagId=" << barrier.wait->id << ")\n");
            }
            // kRule3CrossLoop true only: lead ceiling on live-out SCC def (see cluster-barrier.md).
            if (cluster_barrier::kRule3CrossLoop) {
                first->earliestClock = regionCycles - kLiveOutSccDefLeadCycles;
                PASS_DEBUG(std::cerr << "[DAG schedule] cluster-barrier SCC rule: live-out chain"
                                     << " (dagId=" << first->id
                                     << ") held back to clock >= " << first->earliestClock
                                     << " (region " << regionCycles << " cycles)\n");
            }
            continue;
        }

        bool alreadySplit = false;
        bool needsLock = false;
        std::vector<const HandshakeBarrier*> pinAfter;
        for (size_t i = 0; i < barriers.size(); ++i) {
            const HandshakeBarrier& barrier = barriers[i];
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
            PASS_DEBUG(std::cerr << "[DAG schedule] cluster-barrier SCC rule: chain [" << first->id
                                 << ".." << last->id
                                 << "] already spans a barrier; leaving it to the"
                                    " barrier pass\n");
            continue;
        }

        for (const HandshakeBarrier* barrier : pinAfter) {
            if (barrier->wait->id >= first->id) continue;
            addEdgeById(barrier->wait, first, dagGraph);
            PASS_DEBUG(
                std::cerr << "[DAG schedule] cluster-barrier SCC rule: chain [" << first->id << ".."
                          << last->id << "] has a reader depending on barrier (dagId="
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

    // Map each instruction to an unique id [0..n-1] and build register deps.
    dag::RegionDAG regionDag = dag::buildRegisterDependencyDAG(regionStart, regionEnd);
    if (regionDag.nodes.empty()) return;

    // Taken before any policy edge is merged, so the hazard deadlines downstream
    // measure original program order.
    const std::vector<int> cumCycles = dag::computeCumulativeCycles(regionDag);

    // Scheduler-only policy, so it is applied here rather than inside the shared
    // preparation: the repair passes must not inject cluster-barrier SCC chains.
    if (readyQueue.clusterBarrierEnabled())
        applyClusterBarrierSccRule(regionDag.nodes, regionDag.instToId, regionDag.graph,
                                   cumCycles.back());

    dag::PreparedRegion region = dag::prepareRegionForScheduling(
        std::move(regionDag), cumCycles, regionStart, regionEnd, blockBegin, readyQueue, wmmaIndex);
    dag::drainReadyQueue(region, readyQueue, scheduled, fillerCount);
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

// Convert tentative Layer 2 overlap candidates into the published analysis only
// when every ordering needed for that candidate appears in the final schedule.
// This also covers cycle-rejected hard constraints: they authorize a merge only
// if the unconstrained final order happens to satisfy the same safety contract.
Layer2BarrierOverlapAnalysis::Result validateLayer2BarrierOverlaps(
    Function& func, const std::vector<Layer2BarrierOverlapCandidate>& candidates) {
    struct Position {
        const BasicBlock* bb;
        unsigned index;
    };
    std::unordered_map<const StinkyInstruction*, Position> positions;
    for (const BasicBlock& bb : func) {
        unsigned index = 0;
        for (const IRBase& ir : bb) {
            if (const auto* inst = dyn_cast<StinkyInstruction>(&ir))
                positions.emplace(inst, Position{&bb, index++});
        }
    }

    Layer2BarrierOverlapAnalysis::Result result;
    for (const Layer2BarrierOverlapCandidate& candidate : candidates) {
        bool valid = true;
        for (const auto& [predecessor, successor] : candidate.requiredConstraints) {
            auto predecessorIt = positions.find(predecessor);
            auto successorIt = positions.find(successor);
            if (predecessorIt == positions.end() || successorIt == positions.end() ||
                predecessorIt->second.bb != successorIt->second.bb ||
                predecessorIt->second.index >= successorIt->second.index) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;

        for (StinkyInstruction* barrierAfter : candidate.barriersAfter)
            for (StinkyInstruction* barrierBefore : candidate.barriersBefore)
                result.record(barrierAfter, barrierBefore);
    }
    return result;
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
        std::vector<Layer2BarrierOverlapCandidate> layer2BarrierOverlapCandidates;

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
            auto candidates = rq.takeLayer2BarrierOverlapCandidates();
            layer2BarrierOverlapCandidates.insert(layer2BarrierOverlapCandidates.end(),
                                                  std::make_move_iterator(candidates.begin()),
                                                  std::make_move_iterator(candidates.end()));
        };

        for (auto* bb : rpo) {
            if (!passCtx.shouldProcessBasicBlock(*bb)) continue;

            auto it = bbToLoop.find(bb);
            if (it != bbToLoop.end()) {
                const Loop* loop = it->second;
                auto& rq = loopQueues[loop];
                if (!rq) {
                    rq = dag::createArchReadyQueue(passCtx);
                    rq->setLoopContext(loop);
                }
                rq->setAnalysisCache(&analysisCache);
                scheduleBlock(bb, *rq);
            } else {
                auto rq = dag::createArchReadyQueue(passCtx);
                rq->setAnalysisCache(&analysisCache);
                scheduleBlock(bb, *rq);
            }
        }
        AM.getResult<Layer2BarrierOverlapAnalysis>(func) =
            validateLayer2BarrierOverlaps(func, layer2BarrierOverlapCandidates);
        auto preserved = preserveCFGAnalyses();
        preserved.preserve<Layer2BarrierOverlapAnalysis>();
        return preserved;
    }
};

char StinkyDAGSchedulerPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyDAGSchedulerPass() {
    return std::make_unique<StinkyDAGSchedulerPass>();
}
}  // namespace stinkytofu
