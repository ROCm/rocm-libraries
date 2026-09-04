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
#include <iostream>
#include <sstream>

#include "../../transforms/asm/dag/RegionDAG.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"

#define DEBUG_TYPE "WmmaHideBudgetAnalysis"

namespace stinkytofu {
namespace {

// VALU-pipe slots the same window offers: co-issue bits that are not also
// blocked. Reads the per-instruction coIssueWindow (matrix-format overrides
// already resolved), the same value the pick paths gate on -- not the
// unresolved descriptor field.
int wmmaHideCapacityValu(const StinkyInstruction &inst, uint16_t blocked) {
  const int latency = inst.latencyCycles;
  constexpr int kCoIssueBits = static_cast<int>(sizeof(inst.coIssueWindow) * 8);
  int slots = 0;
  for (int pos = 0; pos < latency && pos < kCoIssueBits; ++pos) {
    if (((inst.coIssueWindow >> pos) & 1u) == 0u)
      continue;
    if (!isBlockedWindowCycle(pos, latency, blocked))
      ++slots;
  }
  return slots;
}

} // namespace

int RegionHideBudget::windowsPastSlot() const {
  int n = 0;
  for (const WmmaWindowBudget &w : windows)
    if (w.mustIssuePastSlot())
      ++n;
  return n;
}

// Only windows that HAD co-issue slots and lost every one of them to
// blockedScaleMask. A matrix op that simply declares no co-issue window
// (v_wmma_f32_16x16x4_f32 carries coIssueWindow 0x0000) also has capacityValu
// == 0, but nothing was blocked there and saying LD_SCALE took its slots would
// be false.
int RegionHideBudget::windowsWithValuBlockedOut() const {
  int n = 0;
  for (const WmmaWindowBudget &w : windows) {
    if (w.capacityValu != 0 || w.wmma == nullptr)
      continue;
    if (wmmaHideCapacityValu(*w.wmma, /*blocked=*/0) > 0)
      ++n;
  }
  return n;
}

// The previous prefix-density implementation was intentionally removed. This
// entry point now receives the barrier analysis computed by
// CDNA5ReadyQueue::onInitRegion and keeps it as the input for the next budget
// policy.
RegionHideBudget
analyzeWmmaHideBudget(const dag::RegionDAG &regionDag,
                      const std::vector<WmmaHideBudgetBarrierInfo> &barriers,
                      int wmmaHideBudgetBase) {
  RegionHideBudget budget;
  budget.barriers = barriers;
  budget.extraIssueByWmmaIndex = !barriers.empty();
  budget.wmmaHideBudgetBase = std::max(0, wmmaHideBudgetBase);

  // Step 1: count every DAG instruction. This is deliberately a raw partition
  // of the node list: barriers and pseudos are non-WMMA here and can be
  // filtered by later policy steps when their scheduling role is considered.
  for (const dag::DAGNode &node : regionDag.nodes) {
    if (isMatrixInstruction(*node.inst)) {
      ++budget.wmmaInstructionCount;
    } else {
      ++budget.nonWmmaInstructionCount;
      if (isDSRead(*node.inst))
        ++budget.dsLoadInstructionCount;
      else
        ++budget.nonDsLoadInstructionCount;
    }
  }

  // Initial policy: one window per WMMA, each starting with an extra-issue
  // budget of 0. Preserve program/DAG-node order and build the instruction
  // lookup at the same time.
  budget.windows.reserve(static_cast<size_t>(budget.wmmaInstructionCount));
  for (const dag::DAGNode &node : regionDag.nodes) {
    if (!isMatrixInstruction(*node.inst))
      continue;
    WmmaWindowBudget window;
    window.wmma = node.inst;
    window.extraIssue = 0;
    budget.windowIndex[node.inst] = budget.numWindows();
    budget.windows.push_back(window);
  }

  // Step 2: distribute every Before barrier's DS loads over the WMMA windows at
  // and after its final threshold. Accumulate contributions from all Before
  // barriers. Quotient/remainder distribution keeps each barrier's total
  // contribution exactly N.
  for (const WmmaHideBudgetBarrierInfo &info : barriers) {
    if (info.position != WmmaHideBudgetBarrierPosition::Before)
      continue;

    const int begin = std::clamp(info.threshold, 0, budget.numWindows());
    const int span = budget.numWindows() - begin;
    const int dsLoads = std::max(0, info.dsLoadCount);
    if (span == 0 || dsLoads == 0) {
      PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis before] barrier="
                           << info.barrier << " threshold=" << info.threshold
                           << " begin=" << begin << " span=" << span
                           << " dsLoadCount=" << dsLoads << " action=skip\n");
      continue;
    }

    const int perWindow = dsLoads / span;
    const int remainder = dsLoads % span;
    for (int i = begin; i < budget.numWindows(); ++i) {
      const int offset = i - begin;
      const int contribution = perWindow + (offset < remainder ? 1 : 0);
      budget.windows[static_cast<size_t>(i)].extraIssue += contribution;
    }
    PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis before] barrier="
                         << info.barrier << " threshold=" << info.threshold
                         << " begin=" << begin << " end=" << budget.numWindows()
                         << " dsLoadCount=" << dsLoads << " perWindow="
                         << perWindow << " remainder=" << remainder << "\n");
  }

  // Step 3: distribute every After barrier's DS loads from window 0 up to the
  // smaller of its required WMMA span and final threshold. Contributions
  // accumulate across all After barriers and on top of the Before-barrier
  // contributions above.
  for (const WmmaHideBudgetBarrierInfo &info : barriers) {
    if (info.position != WmmaHideBudgetBarrierPosition::After)
      continue;

    const int end = std::clamp(std::min(info.dsLoadWmmaNeeded, info.threshold),
                               0, budget.numWindows());
    const int dsLoads = std::max(0, info.dsLoadCount);
    if (end == 0 || dsLoads == 0) {
      PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis after] barrier="
                           << info.barrier << " threshold=" << info.threshold
                           << " wmmaNeeded=" << info.dsLoadWmmaNeeded
                           << " end=" << end << " dsLoadCount=" << dsLoads
                           << " action=skip\n");
      continue;
    }

    const int perWindow = dsLoads / end;
    const int remainder = dsLoads % end;
    for (int i = 0; i < end; ++i) {
      const int contribution = perWindow + (i < remainder ? 1 : 0);
      budget.windows[static_cast<size_t>(i)].extraIssue += contribution;
    }
    PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis after] barrier="
                         << info.barrier << " threshold=" << info.threshold
                         << " wmmaNeeded=" << info.dsLoadWmmaNeeded
                         << " begin=0" << " end=" << end << " dsLoadCount="
                         << dsLoads << " perWindow=" << perWindow
                         << " remainder=" << remainder << "\n");
  }

  // Step 4: place remaining non-DS-load instructions in the first 50% of WMMA
  // windows. Walk top-down first and fill each window to wmmaHideBudgetBase.
  // Only after every front-half window reaches the base do we distribute any
  // remainder evenly.
  const int frontHalfEnd = (budget.numWindows() + 1) / 2;
  int remainingNonDs = budget.nonDsLoadInstructionCount;
  if (frontHalfEnd > 0 && remainingNonDs > 0) {
    for (int i = 0; i < frontHalfEnd && remainingNonDs > 0; ++i) {
      WmmaWindowBudget &window = budget.windows[static_cast<size_t>(i)];
      const int deficit =
          std::max(0, budget.wmmaHideBudgetBase - window.extraIssue);
      const int contribution = std::min(deficit, remainingNonDs);
      window.extraIssue += contribution;
      remainingNonDs -= contribution;
      PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis non-ds fill] index=" << i
                           << " deficit=" << deficit << " contribution="
                           << contribution << " budget=" << window.extraIssue
                           << " remaining=" << remainingNonDs << "\n");
    }
  }

  if (frontHalfEnd > 0 && remainingNonDs > 0) {
    const int perWindow = remainingNonDs / frontHalfEnd;
    const int remainder = remainingNonDs % frontHalfEnd;
    for (int i = 0; i < frontHalfEnd; ++i) {
      const int contribution = perWindow + (i < remainder ? 1 : 0);
      budget.windows[static_cast<size_t>(i)].extraIssue += contribution;
    }
    PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis non-ds spread] begin=0"
                         << " end=" << frontHalfEnd << " remainingBeforeSpread="
                         << remainingNonDs << " perWindow=" << perWindow
                         << " remainder=" << remainder << "\n");
    remainingNonDs = 0;
  }
  PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis non-ds] end=" << frontHalfEnd
                       << " nonDsLoadCount=" << budget.nonDsLoadInstructionCount
                       << " remaining=" << remainingNonDs << "\n");

  PASS_DEBUG(
      std::cerr << "[WmmaHideBudgetAnalysis] dagNodes="
                << regionDag.nodes.size()
                << " wmmaInstructions=" << budget.wmmaInstructionCount
                << " nonWmmaInstructions=" << budget.nonWmmaInstructionCount
                << " dsLoadInstructions=" << budget.dsLoadInstructionCount
                << " nonDsLoadInstructions=" << budget.nonDsLoadInstructionCount
                << " wmmaHideBudgetBase=" << budget.wmmaHideBudgetBase
                << " barriers=" << barriers.size() << "\n");
  for (const WmmaHideBudgetBarrierInfo &info : barriers) {
    PASS_DEBUG(
        std::cerr << "[WmmaHideBudgetAnalysis barrier] barrier=" << info.barrier
                  << " position="
                  << (info.position == WmmaHideBudgetBarrierPosition::After
                          ? "after"
                          : "before")
                  << " threshold=" << info.threshold
                  << " dsLoadCount=" << info.dsLoadCount
                  << " dsLoadWmmaNeeded=" << info.dsLoadWmmaNeeded << "\n");
  }
  for (int i = 0; i < budget.numWindows(); ++i) {
    PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis window] index=" << i
                         << " extraIssue="
                         << budget.windows[static_cast<size_t>(i)].extraIssue
                         << "\n");
  }
  return budget;
}

void reportWmmaHideBudget(const PassContext &passCtx,
                          const RegionHideBudget &budget) {
  const char *const kRemarkPass = "StinkyDAGScheduler";
  if (budget.numWindows() == 0)
    return;

  if (const int pastSlot = budget.windowsPastSlot(); pastSlot > 0) {
    // Name the windows, so a kernel author can find them, but keep the line
    // bounded on a region with many of them.
    constexpr int kMaxListed = 6;
    std::ostringstream oss;
    oss << pastSlot << " of " << budget.numWindows()
        << " WMMA windows must issue past their slot (";
    int listed = 0;
    for (int i = 0; i < budget.numWindows(); ++i) {
      const WmmaWindowBudget &w = budget.windows[static_cast<size_t>(i)];
      if (!w.mustIssuePastSlot())
        continue;
      if (listed == kMaxListed) {
        oss << ", ...";
        break;
      }
      if (listed++ > 0)
        oss << ", ";
      oss << "#" << i << " +" << w.extraIssue << " over " << w.capacityCycles;
    }
    oss << " issue cycles); work a later WMMA depends on does not fit the "
           "shadow "
           "before it";
    emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kRemarkPass,
                         "WmmaWindowPastSlot", oss.str()});
  }

  if (const int noValu = budget.windowsWithValuBlockedOut(); noValu > 0) {
    std::ostringstream oss;
    oss << noValu << " of " << budget.numWindows()
        << " WMMA windows lost every VALU co-issue slot they had to the "
           "LD_SCALE cycle "
           "of a scale pair, so no VALU can be hidden in them at all";
    emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kRemarkPass,
                         "NoValuCoIssueSlot", oss.str()});
  }
}

} // namespace stinkytofu
