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
#include "stinkytofu/analysis/asm/WmmaHideBudgetAnalysis.hpp"

#include <algorithm>
#include <sstream>

#include "../../transforms/asm/dag/RegionDAG.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

namespace stinkytofu {
namespace {

// The scheduler honours blockedScaleMask unconditionally (pickOneFromWMMA), so the
// budget reads it unconditionally too and describes the scheduler that will actually
// run rather than a hypothetical one.
uint16_t blockedScaleMaskOf(const StinkyInstruction& inst) {
    const HwInstDesc* desc = inst.getHwInstDesc();
    return desc != nullptr ? desc->blockedScaleMask : 0;
}

// Issue cycles the window of \p inst can hide work in: everything after its own issue
// slot, less the cycles its blockedScaleMask reserves.
int wmmaHideCapacityCycles(const StinkyInstruction& inst) {
    const int latency = inst.latencyCycles;
    const int issue = std::max(1, inst.issueCycles);
    if (latency <= issue) return 0;
    const uint16_t blocked = blockedScaleMaskOf(inst);
    int cycles = 0;
    for (int pos = issue; pos < latency; ++pos)
        if (!isBlockedWindowCycle(pos, latency, blocked)) ++cycles;
    return cycles;
}

// VALU-pipe slots the same window offers: co-issue bits that are not also blocked.
// Reads the per-instruction coIssueWindow (matrix-format overrides already resolved),
// the same value the pick paths gate on -- not the unresolved descriptor field.
int wmmaHideCapacityValu(const StinkyInstruction& inst, uint16_t blocked) {
    const int latency = inst.latencyCycles;
    constexpr int kCoIssueBits = static_cast<int>(sizeof(inst.coIssueWindow) * 8);
    int slots = 0;
    for (int pos = 0; pos < latency && pos < kCoIssueBits; ++pos) {
        if (((inst.coIssueWindow >> pos) & 1u) == 0u) continue;
        if (!isBlockedWindowCycle(pos, latency, blocked)) ++slots;
    }
    return slots;
}

}  // namespace

int RegionHideBudget::windowsPastSlot() const {
    int n = 0;
    for (const WmmaWindowBudget& w : windows)
        if (w.mustIssuePastSlot()) ++n;
    return n;
}

// Only windows that HAD co-issue slots and lost every one of them to blockedScaleMask.
// A matrix op that simply declares no co-issue window (v_wmma_f32_16x16x4_f32 carries
// coIssueWindow 0x0000) also has capacityValu == 0, but nothing was blocked there and
// saying LD_SCALE took its slots would be false.
int RegionHideBudget::windowsWithValuBlockedOut() const {
    int n = 0;
    for (const WmmaWindowBudget& w : windows) {
        if (w.capacityValu != 0 || w.wmma == nullptr) continue;
        if (wmmaHideCapacityValu(*w.wmma, /*blocked=*/0) > 0) ++n;
    }
    return n;
}

// One pass over the region DAG, no IR mutation.
//
// Demand counts VALU at its issueCycles even though a VALU inside a window can cost more
// (computeValuAdvanceCycles walks to the next co-issue bit). That makes demand a
// deliberate LOWER bound, so an overrun is only ever demanded where one is owed beyond
// doubt. Barriers are left out -- they are not window fillers (updateWMMAStatus charges a
// barrier its full latency), and placing them is the barrier-threshold work, not this --
// as are pseudo nodes, which never reach an issue pipe.
RegionHideBudget analyzeWmmaHideBudget(const dag::RegionDAG& regionDag) {
    RegionHideBudget budget;
    const unsigned n = static_cast<unsigned>(regionDag.nodes.size());
    if (n == 0) return budget;

    // (1) Number the matrix ops in program order. Those are the windows.
    std::vector<int> wmmaOrder(n, -1);
    for (unsigned i = 0; i < n; ++i) {
        StinkyInstruction* inst = regionDag.nodes[i].inst;
        if (!isMatrixInstruction(*inst)) continue;
        wmmaOrder[i] = budget.numWindows();
        budget.windowIndex[inst] = budget.numWindows();
        budget.windows.push_back({inst, wmmaHideCapacityCycles(*inst),
                                  wmmaHideCapacityValu(*inst, blockedScaleMaskOf(*inst)), 0});
    }
    const int numWindows = budget.numWindows();
    if (numWindows == 0) return budget;

    // (2) Deadline per node: the earliest WMMA that transitively depends on it, so the
    // last window it can still hide in is deadline-1. RegionDAG ids are program indices
    // and its edges run strictly forward, so one reverse sweep settles every node --
    // every successor is already final by the time we read it.
    const int kNoDeadline = numWindows;
    std::vector<int> deadline(n, kNoDeadline);
    for (unsigned i = n; i-- > 0;) {
        int d = wmmaOrder[i] >= 0 ? wmmaOrder[i] : kNoDeadline;
        for (unsigned succ : regionDag.graph[i]) d = std::min(d, deadline[succ]);
        deadline[i] = d;
    }

    // (3) Charge the issue cycles of each filler to the deadline it inherited.
    std::vector<int> demandAt(static_cast<size_t>(numWindows), 0);
    for (unsigned i = 0; i < n; ++i) {
        StinkyInstruction* inst = regionDag.nodes[i].inst;
        if (wmmaOrder[i] >= 0 || isPseudoInst(inst) || isBarrier(*inst)) continue;
        const int cycles = std::max(1, inst->issueCycles);
        if (deadline[i] >= kNoDeadline) {
            budget.floatingCycles += cycles;
            continue;
        }
        budget.deadlinedCycles += cycles;
        demandAt[static_cast<size_t>(deadline[i])] += cycles;
    }
    budget.prologueCycles = demandAt[0];

    // (4) Walk the deadlines in order and hand each window the overrun a later WMMA
    // forces on it. Everything due before WMMA i has only windows 0..i-1 to hide in; when
    // that shadow runs short the shortfall is granted to window i-1, the latest one that
    // can still meet the deadline. Prologue work (deadline 0) is excluded -- it precedes
    // every window, so no window can be blamed for it.
    int cumDemand = 0, cumCapacity = 0, granted = 0;
    for (int i = 1; i < numWindows; ++i) {
        cumDemand += demandAt[static_cast<size_t>(i)];
        cumCapacity += budget.windows[static_cast<size_t>(i - 1)].capacityCycles;
        const int need = cumDemand - cumCapacity - granted;
        if (need > 0) {
            budget.windows[static_cast<size_t>(i - 1)].extraIssue += need;
            granted += need;
        }
    }
    return budget;
}

void reportWmmaHideBudget(const PassContext& passCtx, const RegionHideBudget& budget) {
    const char* const kRemarkPass = "StinkyDAGScheduler";
    if (budget.numWindows() == 0) return;

    if (const int pastSlot = budget.windowsPastSlot(); pastSlot > 0) {
        // Name the windows, so a kernel author can find them, but keep the line bounded
        // on a region with many of them.
        constexpr int kMaxListed = 6;
        std::ostringstream oss;
        oss << pastSlot << " of " << budget.numWindows()
            << " WMMA windows must issue past their slot (";
        int listed = 0;
        for (int i = 0; i < budget.numWindows(); ++i) {
            const WmmaWindowBudget& w = budget.windows[static_cast<size_t>(i)];
            if (!w.mustIssuePastSlot()) continue;
            if (listed == kMaxListed) {
                oss << ", ...";
                break;
            }
            if (listed++ > 0) oss << ", ";
            oss << "#" << i << " +" << w.extraIssue << " over " << w.capacityCycles;
        }
        oss << " issue cycles); work a later WMMA depends on does not fit the shadow "
               "before it";
        emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kRemarkPass, "WmmaWindowPastSlot",
                             oss.str()});
    }

    if (const int noValu = budget.windowsWithValuBlockedOut(); noValu > 0) {
        std::ostringstream oss;
        oss << noValu << " of " << budget.numWindows()
            << " WMMA windows lost every VALU co-issue slot they had to the LD_SCALE cycle "
               "of a scale pair, so no VALU can be hidden in them at all";
        emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kRemarkPass, "NoValuCoIssueSlot",
                             oss.str()});
    }
}

}  // namespace stinkytofu
