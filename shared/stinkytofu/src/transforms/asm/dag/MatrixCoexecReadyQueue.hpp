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
#include <climits>
#include <iostream>
#include <vector>

#include "ReadyQueue.hpp"
#include "RegionDAG.hpp"
#include "WaitAnchors.hpp"
#include "stinkytofu/analysis/asm/CoexecWindow.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

namespace stinkytofu {
namespace dag {

/// Issue cycles of what InsertVgprMsbPass inserts. Zero when the module or the
/// arch does not switch banks.
struct VgprMsbSwitchCost {
    int switchCycles = 0;
    int nopCycles = 0;
};

/// Decides the order of each segment for RepairMatrixCoexecPass.
///
/// An empty window between independent matrix ops costs nothing: the matrix
/// pipe runs at full rate. What costs cycles is work that does not fit in front
/// of the next matrix op, which delays it, and a dependent pair, which
/// InsertCoexecHazardPass pads with v_nops. So the scheduler's order is kept,
/// except in two ways:
///   - movable work, meaning anything outside the matrix/memory skeleton, may
///     move past the matrix op that closes its window, to go first in the next
///     one. It keeps its order relative to other movable work, never moves past
///     a matrix op that depends on it, and moves past at most one matrix op;
///   - a window between dependent matrix ops takes VALU work from just after
///     the second one, in place of the spacers.
///
/// planCuts decides how much leaves each window, for the whole segment at once,
/// because moving work out of a full window only helps if the windows after it
/// have room.
///
/// Time is counted in the cycles really spent before the next matrix op: the
/// matrix op's timeline (CoexecWindow), the waits re-emitted in front of the
/// next one, and the bank switches InsertVgprMsbPass will add. Switches are
/// charged for every instruction, fixed ones included, so a window's room is
/// what is left after them.
class MatrixCoexecPickPolicy {
   public:
    /// How much movable work one window may pass on. A window holds a few
    /// cycles, so a longer carry cannot land anywhere; the cap bounds the plan.
    static constexpr unsigned kMaxCarried = 8;

    MatrixCoexecPickPolicy(const WaitAnchorMap& anchors, VgprMsbSwitchCost msbCost)
        : anchors_(anchors), msbCost_(msbCost) {}

    /// Take the next segment, in input order, and plan its windows.
    void beginSegment(const RegionDAG& dag) {
        const unsigned n = static_cast<unsigned>(dag.nodes.size());
        segment_.assign(n, nullptr);
        scheduled_.assign(n, false);
        movableIndex_.assign(n, kFixed);
        movableBefore_.assign(n + 1, 0);
        movableIds_.clear();
        matrixIds_.clear();
        for (const DAGNode& node : dag.nodes) {
            segment_[node.id] = node.inst;
            if (isMatrixInstruction(*node.inst)) matrixIds_.push_back(node.id);
        }
        for (unsigned id = 0; id < n; ++id) {
            if (!isMatrixMemorySkeleton(*segment_[id])) {
                movableIndex_[id] = static_cast<unsigned>(movableIds_.size());
                movableIds_.push_back(id);
            }
            movableBefore_[id + 1] = static_cast<unsigned>(movableIds_.size());
        }
        nextMatrix_ = 0;
        cuts_ = planCuts(dag);
    }

    DAGNode* select(const ReadySetByDAGid& matrixQueue, const ReadySetByDAGid& otherQueue) {
        if (nextMatrix_ == matrixIds_.size())
            return otherQueue.empty() ? nullptr : otherQueue.top();

        // Follow the plan for the window the next matrix op closes, ready or
        // not: it usually still waits on the window's DS loads, and the work
        // planned past it must not slip in ahead meanwhile.
        const unsigned closer = matrixIds_[nextMatrix_];
        const unsigned cut = cuts_[nextMatrix_];
        for (DAGNode* node : otherQueue) {
            if (node->id > closer) break;
            if (movableIndex_[node->id] == kFixed || movableIndex_[node->id] < cut) return node;
        }
        // The lowest planned id is always ready, so this only guards progress.
        if (matrixQueue.empty()) return otherQueue.top();

        DAGNode* matrix = matrixQueue.top();
        // Whatever is still unscheduled ahead of the matrix op goes past it.
        const auto moved = static_cast<int>(
            std::count(scheduled_.begin(), scheduled_.begin() + matrix->id, false));
        if (moved != 0) {
            deferrals_ += moved;
            PASS_DEBUG(std::cerr << "[MatrixCoexecReadyQueue] defer " << moved
                                 << " past matrix dagId=" << matrix->id << "\n");
            return matrix;
        }
        if (openMatrix_ != nullptr) {
            if (DAGNode* hoist = findHoist(otherQueue, *matrix)) {
                ++hoists_;
                PASS_DEBUG(std::cerr << "[MatrixCoexecReadyQueue] hoist dagId=" << hoist->id
                                     << " ahead of dependent matrix dagId=" << matrix->id << "\n");
                return hoist;
            }
        }
        return matrix;
    }

    void onPicked(const DAGNode& node) {
        scheduled_[node.id] = true;
        // The skeleton chain issues matrix ops in input order.
        if (isMatrixInstruction(*node.inst)) ++nextMatrix_;
        advancePast(*node.inst);
    }

    /// Account for \p inst issuing next, whether picked here or emitted around
    /// the segments: a boundary, or a segment kept in input order.
    void advancePast(const StinkyInstruction& inst) {
        if (isLabel(inst)) {
            // Control can arrive here from elsewhere, so no window carries over.
            switchCycles(inst, msb_, &predictedSwitches_);
            window_ = CoexecWindow();
            openMatrix_ = nullptr;
            return;
        }
        if (isBarrier(inst)) {
            issue(window_, msb_, inst, &predictedSwitches_);
            // Every wave has to arrive first, which outlasts any shadow.
            window_.close();
            return;
        }
        if (!isMatrixInstruction(inst)) {
            issue(window_, msb_, inst, &predictedSwitches_);
            return;
        }
        for (const StinkyInstruction* wait : attachedWaits(inst))
            issue(window_, msb_, *wait, &predictedSwitches_);
        window_.advance(switchCycles(inst, msb_, &predictedSwitches_));
        window_.open(inst);
        openMatrix_ = &inst;
    }

    /// Movable instructions moved past a matrix op.
    int deferrals() const {
        return deferrals_;
    }
    /// VALU instructions hoisted into a dependent window.
    int hoists() const {
        return hoists_;
    }
    /// s_set_vgpr_msb InsertVgprMsbPass will insert in what has gone past.
    int predictedSwitches() const {
        return predictedSwitches_;
    }

   private:
    static constexpr unsigned kFixed = UINT_MAX;

    const WaitAnchorMap& anchors_;
    const VgprMsbSwitchCost msbCost_;
    CoexecWindow window_;
    const StinkyInstruction* openMatrix_ = nullptr;
    int msb_ = VgprMsbState::NOT_REQUIRED;

    std::vector<StinkyInstruction*> segment_;
    std::vector<bool> scheduled_;
    std::vector<unsigned> matrixIds_;
    /// Index into matrixIds_ of the next matrix op to issue.
    unsigned nextMatrix_ = 0;
    /// Position of each instruction among the movable ones, or kFixed.
    std::vector<unsigned> movableIndex_;
    /// Movable instructions in input order, and how many come before each id.
    std::vector<unsigned> movableIds_;
    std::vector<unsigned> movableBefore_;
    /// Per matrix op: movable work below this movable index goes ahead of it.
    std::vector<unsigned> cuts_;

    int deferrals_ = 0;
    int hoists_ = 0;
    int predictedSwitches_ = 0;

    const std::vector<StinkyInstruction*>& attachedWaits(const StinkyInstruction& matrix) const {
        static const std::vector<StinkyInstruction*> kNone;
        auto it = anchors_.find(const_cast<StinkyInstruction*>(&matrix));
        return it == anchors_.end() ? kNone : it->second.waits;
    }

    /// Cycles of what InsertVgprMsbPass will put in front of \p inst, moving
    /// \p msb past it. That pass sees an exec-mask group's members, so they are
    /// what is counted.
    int switchCycles(const StinkyInstruction& inst, int& msb, int* switches = nullptr) const {
        if (isExecMaskGroup(inst)) {
            int cycles = 0;
            if (const auto* group = inst.getModifier<ExecGroupData>())
                for (const StinkyInstruction* member : group->children)
                    cycles += switchCycles(*member, msb, switches);
            return cycles;
        }
        const int inserted = vgprMsbInsertionsBefore(inst, msb);
        if (inserted == 0) return 0;
        if (switches != nullptr) ++*switches;
        return msbCost_.switchCycles + (inserted == 2 ? msbCost_.nopCycles : 0);
    }

    /// Cycles \p inst spends from \p window's position, its bank switch
    /// included, moving \p window and \p msb past it. A blocked cycle the
    /// timeline rolls over is not counted: the hardware spends it, not \p inst.
    int issue(CoexecWindow& window, int& msb, const StinkyInstruction& inst,
              int* switches = nullptr) const {
        const int switchCost = switchCycles(inst, msb, switches);
        window.advance(switchCost);
        const int cost =
            fillsCoexecSlot(inst) ? window.valuAdvanceCycles(inst.issueCycles) : inst.issueCycles;
        window.advance(cost);
        return switchCost + cost;
    }

    /// Whether \p inst, followed by \p matrix's waits and bank switch, fits in
    /// the free part of the open window.
    bool fits(const StinkyInstruction& inst, const StinkyInstruction& matrix) const {
        CoexecWindow window = window_;
        int msb = msb_;
        const int room = window.freeSpace();
        int need = issue(window, msb, inst);
        for (const StinkyInstruction* wait : attachedWaits(matrix))
            need += issue(window, msb, *wait);
        need += switchCycles(matrix, msb);
        return need <= room;
    }

    /// For each matrix op, one past the movable index of its latest movable
    /// predecessor: the lowest cut that keeps everything it depends on ahead.
    std::vector<unsigned> earliestCuts(const RegionDAG& dag) const {
        const unsigned n = static_cast<unsigned>(segment_.size());
        std::vector<std::vector<unsigned>> preds(n);
        for (unsigned from = 0; from < n; ++from)
            for (unsigned to : dag.graph[from]) preds[to].push_back(from);

        std::vector<unsigned> cuts;
        std::vector<unsigned> seenBy(n, UINT_MAX);
        std::vector<unsigned> work;
        for (unsigned k = 0; k < matrixIds_.size(); ++k) {
            unsigned cut = 0;
            work.assign(1, matrixIds_[k]);
            while (!work.empty()) {
                const unsigned id = work.back();
                work.pop_back();
                for (unsigned pred : preds[id]) {
                    if (seenBy[pred] == k) continue;
                    seenBy[pred] = k;
                    if (movableIndex_[pred] != kFixed) cut = std::max(cut, movableIndex_[pred] + 1);
                    work.push_back(pred);
                }
            }
            cuts.push_back(cut);
        }
        return cuts;
    }

    /// Simulated time of window \p w, from the matrix op that opens it to the
    /// one that closes it, or to the segment's end for the last window. Window 0
    /// is the one already open when the segment starts.
    ///
    /// The window holds the movable work its predecessor passed on, from movable
    /// index \p from, then its own work in input order without the movable work
    /// from index \p cut on, which it passes on in turn.
    int windowTime(unsigned w, unsigned from, unsigned cut) const {
        CoexecWindow window;
        int msb = VgprMsbState::NOT_REQUIRED;
        int start = 0;
        if (w == 0) {
            window = window_;
            msb = msb_;
            start = window.position();
        } else {
            const StinkyInstruction& opener = *segment_[matrixIds_[w - 1]];
            switchCycles(opener, msb);
            window.open(opener);
        }
        const unsigned ownBegin = w == 0 ? 0 : matrixIds_[w - 1] + 1;
        const unsigned ownEnd =
            w < matrixIds_.size() ? matrixIds_[w] : static_cast<unsigned>(segment_.size());

        for (unsigned m = from; m < movableBefore_[ownBegin]; ++m)
            issue(window, msb, *segment_[movableIds_[m]]);
        for (unsigned id = ownBegin; id < ownEnd; ++id)
            if (movableIndex_[id] == kFixed || movableIndex_[id] < cut)
                issue(window, msb, *segment_[id]);
        if (w < matrixIds_.size()) {
            const StinkyInstruction& closer = *segment_[matrixIds_[w]];
            for (const StinkyInstruction* wait : attachedWaits(closer)) issue(window, msb, *wait);
            window.advance(switchCycles(closer, msb));
        }
        return std::max(window.latency(), window.position()) - start;
    }

    /// For each window, the movable index from which its own movable work goes
    /// past the matrix op that closes it, chosen to minimize the segment's
    /// simulated time.
    ///
    /// A window holds the tail its predecessor passed on and its own work
    /// without the tail it passes on, so its time depends only on those two
    /// cuts. That makes the choice a dynamic program over the windows, with the
    /// predecessor's cut as the state. Ties keep more work in place.
    std::vector<unsigned> planCuts(const RegionDAG& dag) const {
        const unsigned windows = static_cast<unsigned>(matrixIds_.size());
        if (windows == 0) return {};
        const unsigned total = static_cast<unsigned>(movableIds_.size());
        const std::vector<unsigned> earliest = earliestCuts(dag);

        // Window w cuts in [lo[w], hi[w]], where hi keeps everything. Only its
        // own work may leave, at most kMaxCarried of it.
        std::vector<unsigned> lo(windows), hi(windows);
        for (unsigned w = 0; w < windows; ++w) {
            hi[w] = movableBefore_[matrixIds_[w]];
            const unsigned own = w == 0 ? 0 : movableBefore_[matrixIds_[w - 1] + 1];
            const unsigned capped = hi[w] > kMaxCarried ? hi[w] - kMaxCarried : 0;
            // Before the first matrix op there is no window to move work out of.
            lo[w] = w == 0 && openMatrix_ == nullptr ? hi[w] : std::max({own, capped, earliest[w]});
        }

        // after[c - lo[w]]: least time from window w + 1 on, when window w cuts at c.
        std::vector<int> after(hi[windows - 1] - lo[windows - 1] + 1);
        for (unsigned c = lo[windows - 1]; c <= hi[windows - 1]; ++c)
            after[c - lo[windows - 1]] = windowTime(windows, c, total);

        std::vector<std::vector<unsigned>> bestCut(windows);
        for (unsigned w = windows; w-- > 0;) {
            const unsigned fromLo = w == 0 ? 0 : lo[w - 1];
            const unsigned fromHi = w == 0 ? 0 : hi[w - 1];
            std::vector<int> best(fromHi - fromLo + 1, INT_MAX);
            bestCut[w].assign(fromHi - fromLo + 1, hi[w]);
            for (unsigned from = fromLo; from <= fromHi; ++from) {
                for (unsigned cut = hi[w] + 1; cut-- > lo[w];) {
                    const int time = windowTime(w, from, cut) + after[cut - lo[w]];
                    if (time < best[from - fromLo]) {
                        best[from - fromLo] = time;
                        bestCut[w][from - fromLo] = cut;
                    }
                }
            }
            after = std::move(best);
        }

        std::vector<unsigned> cuts(windows);
        unsigned from = 0;
        for (unsigned w = 0; w < windows; ++w)
            from = cuts[w] = bestCut[w][from - (w == 0 ? 0 : lo[w - 1])];
        return cuts;
    }

    /// VALU work from just after \p matrix that fits ahead of it, when \p matrix
    /// depends on the open matrix op and would otherwise get v_nop spacers.
    DAGNode* findHoist(const ReadySetByDAGid& otherQueue, const DAGNode& matrix) const {
        if (!wmmaToWmmaCoexecOverlap(*openMatrix_, *matrix.inst)) return nullptr;
        const unsigned limit =
            nextMatrix_ + 1 < matrixIds_.size() ? matrixIds_[nextMatrix_ + 1] : UINT_MAX;
        for (DAGNode* node : otherQueue) {
            if (node->id <= matrix.id) continue;
            if (node->id >= limit) break;
            if (!fillsCoexecSlot(*node->inst)) continue;
            // One touching the open matrix op's registers needs spacers itself.
            if (wmmaToValuCoexecOverlap(*openMatrix_, *node->inst)) continue;
            if (fits(*node->inst, *matrix.inst)) return node;
        }
        return nullptr;
    }
};

/// The ready queue RepairMatrixCoexecPass schedules a block with: matrix ops
/// and everything else in two sets ordered by original id, and the policy above
/// choosing between them. One queue spans the block, so a window and the bank
/// state carry across segment boundaries.
class MatrixCoexecReadyQueue : public ReadyQueue {
   public:
    MatrixCoexecReadyQueue(const PassContext& passCtx, const WaitAnchorMap& anchors)
        : ReadyQueue(passCtx), policy_(anchors, vgprMsbSwitchCost(passCtx)) {}

    void beginSegment(const RegionDAG& dag) {
        policy_.beginSegment(dag);
    }

    void advancePast(const StinkyInstruction& inst) {
        policy_.advancePast(inst);
    }

    const MatrixCoexecPickPolicy& policy() const {
        return policy_;
    }

    void push(DAGNode* node) override {
        if (isMatrixInstruction(*node->inst))
            matrixQueue_.push(node);
        else
            otherQueue_.push(node);
    }

    DAGNode* pickOne() override {
        DAGNode* node = policy_.select(matrixQueue_, otherQueue_);
        if (isMatrixInstruction(*node->inst))
            matrixQueue_.erase(node);
        else
            otherQueue_.erase(node);
        policy_.onPicked(*node);
        return node;
    }

    bool empty() const override {
        return matrixQueue_.empty() && otherQueue_.empty();
    }

   private:
    ReadySetByDAGid matrixQueue_;
    ReadySetByDAGid otherQueue_;
    MatrixCoexecPickPolicy policy_;

    static VgprMsbSwitchCost vgprMsbSwitchCost(const PassContext& passCtx) {
        if (passCtx.getAsmCapsConfig().vgprMsbMode == VgprMsbMode::None) return {};
        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        const HwInstDesc* msb = getMCIDByUOp(GFX::s_set_vgpr_msb, archId);
        const HwInstDesc* nop = getMCIDByUOp(GFX::s_nop, archId);
        return {msb != nullptr ? msb->issue : 0, nop != nullptr ? nop->issue : 0};
    }
};

}  // namespace dag
}  // namespace stinkytofu
