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

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/asm/HazardGapAnalysisPass.hpp"
#include "stinkytofu/analysis/asm/WmmaHideBudgetAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

// Before dag/*.hpp so PASS_DEBUG inside those headers uses this pass name.
#define DEBUG_TYPE "RepairMatrixCoexecPass"

#include "dag/MatrixCoexecReadyQueue.hpp"
#include "dag/RegionDAG.hpp"
#include "dag/WaitAnchors.hpp"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::dag;

/// Where this pass stops: everywhere the shared rule stops, plus barriers.
///
/// A barrier carrying an LDS token is not a side effect, so the shared rule
/// leaves it schedulable. The scheduler can place it safely, seeing the whole
/// region and applying the cluster-barrier SCC rule; this pass does neither.
/// Moving a barrier in a double-buffered loop reorders one side of a handshake
/// and hangs the kernel, so barriers are boundaries here.
bool isCoexecRepairBoundary(const StinkyInstruction& inst,
                            const std::unordered_set<StinkyInstruction*>& attachedWaits) {
    return isHardBoundary(inst, attachedWaits) || isBarrier(inst);
}

/// One piece of a block: a run the pass may reorder, or one IR object it may
/// not -- a boundary, or IR that is not an instruction.
///
/// A run excludes its attached waits, which are metadata rather than DAG nodes
/// and are re-emitted in front of their anchors.
struct BlockPiece {
    std::vector<StinkyInstruction*> segment;
    IRBase* fixed = nullptr;
};

std::vector<BlockPiece> cutBlock(BasicBlock& bb,
                                 const std::unordered_set<StinkyInstruction*>& attachedWaits) {
    std::vector<BlockPiece> pieces;
    BlockPiece run;
    auto flushRun = [&]() {
        if (!run.segment.empty()) pieces.push_back(std::move(run));
        run = BlockPiece();
    };
    for (IRBase& ir : bb) {
        auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst == nullptr || ir.getType() != IRBase::IRType::StinkyTofu) {
            flushRun();
            pieces.push_back({{}, &ir});
            continue;
        }
        // Attached waits travel with their anchor, so they are neither segment
        // members nor boundaries.
        if (attachedWaits.count(inst) != 0) continue;
        if (isCoexecRepairBoundary(*inst, attachedWaits)) {
            flushRun();
            pieces.push_back({{}, inst});
            continue;
        }
        run.segment.push_back(inst);
    }
    flushRun();
    return pieces;
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

std::vector<IRBase*> blockOrder(BasicBlock& bb) {
    std::vector<IRBase*> out;
    out.reserve(bb.size());
    for (IRBase& ir : bb) out.push_back(&ir);
    return out;
}

std::vector<const StinkyInstruction*> instructionsOf(const std::vector<IRBase*>& irs) {
    std::vector<const StinkyInstruction*> out;
    out.reserve(irs.size());
    for (IRBase* ir : irs)
        if (auto* inst = dyn_cast<StinkyInstruction>(ir)) out.push_back(inst);
    return out;
}

/// Order one segment through \p queue, on its register DAG plus the edges that
/// keep wait immediates correct and the matrix/memory skeleton in place.
std::vector<StinkyInstruction*> scheduleSegment(const std::vector<StinkyInstruction*>& segment,
                                                const WaitAnchorMap& anchors,
                                                MatrixCoexecReadyQueue& queue) {
    RegionDAG dag = buildRegisterDependencyDAG(segment);
    addSyntheticOrderEdges(dag, segment, anchors);
    queue.beginSegment(dag);

    std::vector<StinkyInstruction*> out;
    out.reserve(segment.size());
    for (DAGNode& node : dag.nodes)
        if (node.inDegree == 0) queue.push(&node);
    while (!queue.empty()) {
        DAGNode* node = queue.pickOne();
        out.push_back(node->inst);
        for (unsigned succ : dag.graph[node->id])
            if (--dag.nodes[succ].inDegree == 0) queue.push(&dag.nodes[succ]);
    }

    // A runtime check, not an assert: in a release build a short segment would
    // silently drop instructions. The usual cause is a cycle among the edges
    // added above.
    if (out.size() != segment.size()) {
        report_fatal_error("RepairMatrixCoexecPass: segment scheduled " +
                           std::to_string(out.size()) + " of " + std::to_string(segment.size()) +
                           " instructions; the added ordering edges are probably cyclic");
    }
    return out;
}

struct BlockLayout {
    std::vector<IRBase*> order;
    int deferrals = 0;
    int hoists = 0;
    int predictedSwitches = 0;
};

/// Lay \p pieces out through one queue, so a window and the bank state carry
/// across the boundaries between segments. Segments marked in \p keep stay in
/// input order but still go through the queue's accounting, which the segments
/// after them are decided against.
BlockLayout layOutBlock(const std::vector<BlockPiece>& pieces, const std::vector<bool>& keep,
                        const WaitAnchorMap& anchors, const PassContext& passCtx) {
    MatrixCoexecReadyQueue queue(passCtx, anchors);
    BlockLayout layout;
    for (size_t i = 0; i < pieces.size(); ++i) {
        const BlockPiece& piece = pieces[i];
        if (piece.fixed != nullptr) {
            layout.order.push_back(piece.fixed);
            if (auto* inst = dyn_cast<StinkyInstruction>(piece.fixed)) queue.advancePast(*inst);
            continue;
        }
        std::vector<StinkyInstruction*> order;
        if (keep[i]) {
            order = piece.segment;
            for (StinkyInstruction* inst : order) queue.advancePast(*inst);
        } else {
            order = scheduleSegment(piece.segment, anchors, queue);
        }
        for (StinkyInstruction* inst : order) emitInstWithWaits(layout.order, inst, anchors);
    }
    layout.deferrals = queue.policy().deferrals();
    layout.hoists = queue.policy().hoists();
    layout.predictedSwitches = queue.policy().predictedSwitches();
    return layout;
}

/// Segments holding an end of a hazard pair whose gap \p output leaves shorter
/// than both \p input's and the rule's distance.
///
/// Both orders pair each consumer with the same producer: the DAG keeps every
/// register dependency, so the latest writer of each source is unchanged. And
/// keeping both ends' segments in input order restores the gap exactly, because
/// nothing else can cross into or out of the span between them.
std::vector<size_t> segmentsShorteningHazards(
    const std::vector<const StinkyInstruction*>& input,
    const std::vector<const StinkyInstruction*>& output,
    const std::unordered_map<const StinkyInstruction*, size_t>& pieceOf) {
    using PairKey = std::tuple<int, const StinkyInstruction*, const StinkyInstruction*>;
    std::map<PairKey, int> floorOf;
    for (const HazardGap& pair : measureHazardGaps(input))
        floorOf[{pair.ruleIdx, input[pair.producer], input[pair.consumer]}] =
            std::min(pair.gap, kCdna5HazardRules[pair.ruleIdx].distance);

    std::vector<size_t> offenders;
    for (const HazardGap& pair : measureHazardGaps(output)) {
        const PairKey key{pair.ruleIdx, output[pair.producer], output[pair.consumer]};
        auto it = floorOf.find(key);
        const int floor =
            it != floorOf.end() ? it->second : kCdna5HazardRules[pair.ruleIdx].distance;
        if (pair.gap >= floor) continue;
        for (const StinkyInstruction* end : {output[pair.producer], output[pair.consumer]})
            if (auto piece = pieceOf.find(end); piece != pieceOf.end())
                offenders.push_back(piece->second);
    }
    return offenders;
}

/// Upper bound on repairBlock's rounds. Each round only takes moves its model
/// scores strictly better, so the order settles well before this; the cap only
/// bounds a case where it would not.
constexpr int kMaxRepairRounds = 8;

/// One round of repair over \p bb, against the order it is in now. Returns
/// whether that order changed.
bool repairRound(BasicBlock& bb, const WaitAnchorMap& anchors, const PassContext& passCtx) {
    const std::unordered_set<StinkyInstruction*> attachedWaits = collectAttachedWaits(anchors);
    const std::vector<BlockPiece> pieces = cutBlock(bb, attachedWaits);

    std::unordered_map<const StinkyInstruction*, size_t> pieceOf;
    for (size_t i = 0; i < pieces.size(); ++i) {
        for (StinkyInstruction* inst : pieces[i].segment) {
            pieceOf[inst] = i;
            if (auto it = anchors.find(inst); it != anchors.end())
                for (StinkyInstruction* wait : it->second.waits) pieceOf[wait] = i;
        }
    }

    const std::vector<IRBase*> current = blockOrder(bb);
    const std::vector<const StinkyInstruction*> input = instructionsOf(current);

    std::vector<bool> keep(pieces.size(), false);
    BlockLayout layout = layOutBlock(pieces, keep, anchors, passCtx);
    int kept = 0;
    for (;;) {
        bool newlyKept = false;
        for (size_t piece :
             segmentsShorteningHazards(input, instructionsOf(layout.order), pieceOf)) {
            if (keep[piece]) continue;
            keep[piece] = true;
            newlyKept = true;
            ++kept;
        }
        if (!newlyKept) break;
        layout = layOutBlock(pieces, keep, anchors, passCtx);
    }

    if (layout.order.size() != current.size()) {
        report_fatal_error("RepairMatrixCoexecPass: rebuilt block has " +
                           std::to_string(layout.order.size()) + " instructions, expected " +
                           std::to_string(current.size()));
    }
    PASS_DEBUG(std::cerr << "[RepairMatrixCoexec] deferred " << layout.deferrals << ", hoisted "
                         << layout.hoists << ", kept " << kept
                         << " segment(s) in input order for hazard gaps, predicted "
                         << layout.predictedSwitches << " bank switch(es)\n");

    if (layout.order == current) return false;
    for (IRBase* ir : layout.order) {
        bb.removeIR(ir);
        bb.appendIR(ir);
    }
    return true;
}

void repairBlock(BasicBlock& bb, const PassContext& passCtx) {
    const WaitAnchorMap anchors = discoverWaitAnchors(bb);
    // Without a wait-anchored matrix op there is nothing for this pass to repair.
    if (anchors.empty()) return;

    PASS_DEBUG(dumpMatrixCoexecOccupancy(measureMatrixCoexecOccupancy(blockInstructions(bb)),
                                         "before", std::cerr));

    // Repeat until the order settles, which also makes the pass idempotent. Each
    // round plans against the order the last one left: once work has left a
    // window, the work in front of it can follow. A round never shortens a
    // hazard gap below its input's or the rule's, so neither do all of them.
    for (int round = 0; round < kMaxRepairRounds && repairRound(bb, anchors, passCtx); ++round) {
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

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
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
            repairBlock(bb, passCtx);
            expandExecMaskedGroups(bb);
        }
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
