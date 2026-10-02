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
#include "stinkytofu/transforms/asm/RepairMatrixCoexecPass.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/WmmaHideBudgetAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/DefUseAnalysisCleanup.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"

// Before dag/*.hpp so PASS_DEBUG inside those headers uses this pass name.
#define DEBUG_TYPE "RepairMatrixCoexecPass"

#include "dag/ArchReadyQueue.hpp"
#include "dag/RegionDAG.hpp"
#include "dag/RegionScheduling.hpp"
#include "dag/WaitAnchors.hpp"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::dag;

/// A run of schedulable instructions with no boundary inside it.
///
/// `instructions` excludes the attached waits, because those are metadata rather
/// than DAG nodes. `first` and `last` bracket the same run in the live block,
/// waits included, which is what the ready queue needs to seed register latency
/// from the block prefix.
struct Segment {
    std::vector<StinkyInstruction*> instructions;
    IRBase* first = nullptr;
    IRBase* last = nullptr;

    bool empty() const {
        return instructions.empty();
    }
    void clear() {
        instructions.clear();
        first = nullptr;
        last = nullptr;
    }
};

/// Where this pass stops, which is everywhere the shared rule stops plus
/// barriers.
///
/// A barrier carrying an LDS token is not a side effect, so the shared rule
/// leaves it schedulable and the DAG orders it by memory token alone. That is
/// right for the scheduler, which decides barrier placement with the whole
/// region in view and the cluster-barrier SCC rule applied. It is wrong here:
/// this pass sees finer segments, applies no SCC rule, and only wants to refill
/// co-issue windows. Relocating a barrier in a double-buffered loop reorders one
/// side of a handshake and hangs the kernel, so barriers are pinned instead.
bool isCoexecRepairBoundary(const StinkyInstruction& inst,
                             const std::unordered_set<StinkyInstruction*>& attachedWaits) {
    return isHardBoundary(inst, attachedWaits) || isBarrier(inst);
}

void emitInstWithWaits(std::vector<IRBase*>& output, StinkyInstruction* inst,
                       const WaitAnchorMap& anchors) {
    if (auto it = anchors.find(inst); it != anchors.end()) {
        for (StinkyInstruction* wait : it->second.waits) output.push_back(wait);
    }
    output.push_back(inst);
}

/// Every StinkyTofu instruction of \p bb in program order, for measurement.
std::vector<StinkyInstruction*> blockInstructions(BasicBlock& bb) {
    std::vector<StinkyInstruction*> out;
    for (IRBase& ir : bb)
        if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) out.push_back(inst);
    return out;
}

/// Replay one segment through \p readyQueue.
///
/// The DAG comes from the wait-free instruction list, while the region handed to
/// the queue is the live IR range, waits included. That split is deliberate: the
/// queue only reads the range to decay register latencies over the block prefix,
/// and a wait writes no register, so it can see the real instruction stream
/// without being schedulable itself.
std::vector<StinkyInstruction*> repairSegment(
    const Segment& segment, const WaitAnchorMap& anchors, BasicBlock& bb, ReadyQueue& readyQueue,
    const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex, int& fillerCount) {
    if (segment.empty()) return {};

    RegionDAG dag = buildRegisterDependencyDAG(segment.instructions);

    // Taken before the edges below are merged, so hazard deadlines downstream
    // measure original program order.
    const std::vector<int> cumCycles = computeCumulativeCycles(dag);

    // The constraints the hardware rules do not express: counter order keeps each
    // wait immediate counting the operations it was computed for, and the pin
    // edges hold work whose only value is its position.
    addSyntheticOrderEdges(dag, segment.instructions, anchors);

    // An attached wait is kept out of the DAG but re-emitted in front of its
    // anchor, where it still costs an issue cycle. Telling the queue about that
    // cycle is what stops it filling the window right up to the last cycle and
    // then having the wait push the anchor past the close.
    for (DAGNode& node : dag.nodes) {
        auto it = anchors.find(node.inst);
        if (it == anchors.end()) continue;
        for (StinkyInstruction* wait : it->second.waits) node.preIssueCycles += wait->issueCycles;
    }

    // blockBegin is the real start of the block, not of the segment: the queue
    // decays register latencies over everything ahead of the region, so a load
    // issued earlier in the block still gates a consumer inside it.
    PreparedRegion region = prepareRegionForScheduling(
        std::move(dag), cumCycles, IRList::iterator(segment.first),
        IRList::iterator(segment.last->getNext()), bb.begin(), readyQueue, wmmaIndex);

    std::vector<IRBase*> scheduled;
    scheduled.reserve(segment.instructions.size());
    drainReadyQueue(region, readyQueue, scheduled, fillerCount);

    // Drop the v_nop spacers the queue asks for, rather than splicing freshly
    // created nodes into a block this pass does not own: it runs inside a region
    // adaptor, on IR extracted into a temporary function and spliced back, so
    // every instruction it emits must already belong to the block.
    //
    // Nothing is lost by dropping them. InsertCoexecHazardPass runs later over
    // final IR, counts the spacers actually present, and tops up the shortfall,
    // so the spacing this queue wanted still arrives -- just from the pass that
    // owns inserting it. fillerCount is reported so a caller can see it happened.
    std::vector<StinkyInstruction*> out;
    out.reserve(scheduled.size());
    for (IRBase* ir : scheduled) {
        if (ir->getParent() == nullptr) continue;  // queue-created spacer
        out.push_back(cast<StinkyInstruction>(ir));
    }

    // Checked at runtime, not by assert: release builds compile asserts out, and
    // a segment that comes back short silently drops instructions from the block,
    // which corrupts the module and crashes somewhere else entirely. The usual
    // cause is a cycle among the edges added above, which leaves the queue unable
    // to drain every node.
    if (out.size() != segment.instructions.size()) {
        report_fatal_error("RepairMatrixCoexecPass: segment scheduled " +
                           std::to_string(out.size()) + " of " +
                           std::to_string(segment.instructions.size()) +
                           " instructions; the added ordering edges are probably cyclic");
    }
    return out;
}

void repairBlock(BasicBlock& bb, const PassContext& passCtx,
                 const std::unordered_map<StinkyInstruction*, unsigned>& wmmaIndex) {
    const WaitAnchorMap anchors = discoverWaitAnchors(bb);
    // Without a wait-anchored matrix op there is nothing for this pass to repair.
    if (anchors.empty()) return;

    PASS_DEBUG(dumpMatrixCoexecOccupancy(measureMatrixCoexecOccupancy(blockInstructions(bb)),
                                          "before", std::cerr));

    const std::unordered_set<StinkyInstruction*> attachedWaits = collectAttachedWaits(anchors);

    // This pass refills windows in a schedule that already met the region's
    // targets, so per-window capacity is the authority and nothing is gained by
    // exceeding it.
    ArchReadyQueueOptions queueOptions;
    queueOptions.fullWindowOverridesHideBudget = true;
    std::unique_ptr<ReadyQueue> readyQueue = createArchReadyQueue(passCtx, queueOptions);
    readyQueue->onInit(bb.begin(), bb.end());

    std::vector<IRBase*> output;
    output.reserve(bb.size());
    int fillerCount = 0;

    Segment segment;
    segment.instructions.reserve(bb.size());

    auto flushSegment = [&]() {
        if (segment.empty()) {
            segment.clear();
            return;
        }
        for (StinkyInstruction* inst :
             repairSegment(segment, anchors, bb, *readyQueue, wmmaIndex, fillerCount))
            emitInstWithWaits(output, inst, anchors);
        segment.clear();
    };

    for (IRBase& ir : bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) {
            flushSegment();
            output.push_back(&ir);
            continue;
        }

        auto* inst = cast<StinkyInstruction>(&ir);
        // Attached waits travel with their anchor, so they are neither segment
        // members nor boundaries.
        if (attachedWaits.count(inst) != 0) continue;

        if (isCoexecRepairBoundary(*inst, attachedWaits)) {
            flushSegment();
            output.push_back(inst);
            continue;
        }

        if (segment.first == nullptr) segment.first = &ir;
        segment.last = &ir;
        segment.instructions.push_back(inst);
    }
    flushSegment();

    readyQueue->onFinishBB();

    if (output.size() != bb.size()) {
        report_fatal_error("RepairMatrixCoexecPass: rebuilt block has " +
                           std::to_string(output.size()) + " instructions, expected " +
                           std::to_string(bb.size()));
    }
    PASS_DEBUG(if (fillerCount > 0) std::cerr
               << "[RepairMatrixCoexec] dropped " << fillerCount
               << " queue spacer(s); InsertCoexecHazardPass will supply the spacing\n");

    for (IRBase* ir : output) {
        bb.removeIR(ir);
        bb.appendIR(ir);
    }

    PASS_DEBUG(dumpMatrixCoexecOccupancy(measureMatrixCoexecOccupancy(blockInstructions(bb)),
                                          "after", std::cerr));
}

class RepairMatrixCoexecPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "RepairMatrixCoexecPass";
    }

    PassID getPassID() const override {
        return &RepairMatrixCoexecPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        // Rebuilt here rather than inherited. The scheduler builds these chains,
        // but StinkyWaitCntInsertionPass then erases the PHI pseudo-instructions
        // without first unlinking them, so every chain reaching this pass can name
        // freed memory -- and prepareRegionForScheduling walks getUsers() to find
        // each hazard's nearest consumer. Rebuilding also makes the deadlines this
        // pass computes the same ones the scheduler computed, which is the point.
        const auto& domInfo = AM.getResult<DominanceAnalysis>(func);
        buildUseDefChain(func, domInfo, /*clearExisting=*/true);

        // Function-wide, so the ds-read affinity one segment computes is
        // comparable with every other segment's: a per-segment index would make
        // "the first matrix op that consumes this load" mean something different
        // in each one.
        //
        // Walked directly rather than through BBIndexAnalysis. All the index needs
        // is an order consistent across segments, and this pass runs inside a
        // region adaptor on a filtered block set, where depending on a cached CFG
        // traversal buys nothing and adds a way to be wrong.
        std::unordered_map<StinkyInstruction*, unsigned> wmmaIndex;
        {
            unsigned idx = 0;
            for (BasicBlock& bb : func)
                for (IRBase& ir : bb)
                    if (auto* inst = dyn_cast<StinkyInstruction>(&ir))
                        if (isMatrixInstruction(*inst)) wmmaIndex[inst] = idx++;
        }

        const GfxArchID archId =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);
        const uint32_t wavefrontSize = passCtx.getWavefrontSize();

        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            // The DAG does not model the exec mask, so a narrow-exec span is
            // collapsed to one opaque node that isHardBoundary then refuses to
            // schedule through. See ExecMaskGrouping.hpp.
            AsmIRBuilder builder(bb, archId);
            collapseExecMaskedRegions(bb, builder, wavefrontSize);
            repairBlock(bb, passCtx, wmmaIndex);
            expandExecMaskedGroups(bb);
        }

        // Hand the IR back in the shape it arrived: the rebuild above re-inserted
        // the PHIs that ran before this pass had already stripped, and nothing
        // downstream strips them a second time. Clears the chains first, so the
        // next pass cannot inherit the dangling ones this pass just worked around.
        discardDefUseAnalysis(func);
        return PreservedAnalyses::none();
    }
};

char RepairMatrixCoexecPass::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createRepairMatrixCoexecPass() {
    return std::make_unique<RepairMatrixCoexecPass>();
}
}  // namespace stinkytofu
