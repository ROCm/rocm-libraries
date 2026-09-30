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
#include <cmath>
#include <iostream>

#include "../../transforms/asm/dag/RegionDAG.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "WmmaHideBudgetAnalysis"

namespace stinkytofu {

namespace {

int computeThrottleOnlyWindowsNeeded(int dsLoadCount, const DsLoadBudgetConfig& config) {
    if (dsLoadCount <= 0 || config.dsReadPerCap <= 0) return 0;

    const int perCap = config.dsReadPerCap;
    int windowsNeeded = (dsLoadCount + perCap - 1) / perCap;
    if (config.dsReadQueueDepth <= 0 || dsLoadCount <= config.dsReadQueueDepth ||
        config.dsReadThrottleLatency <= 0 || config.wmmaLatency <= 0)
        return windowsNeeded;

    const int overflowCount = dsLoadCount - config.dsReadQueueDepth;
    const int transitionCount =
        std::min(overflowCount, std::max(0, config.dsReadThrottleTransitionEntries));
    const int fullThrottleCount = overflowCount - transitionCount;
    const float cyclesPerDs = static_cast<float>(config.dsReadThrottleLatency) /
                              static_cast<float>(config.dsReadQueueDepth);
    const float transitionFactor =
        static_cast<float>(std::clamp(config.dsReadThrottleTransitionFactor, 0.0, 1.0));
    const float cyclesNeeded =
        cyclesPerDs * (transitionFactor * transitionCount + fullThrottleCount);
    const float baseWindows =
        static_cast<float>(config.dsReadQueueDepth + perCap - 1) / static_cast<float>(perCap);
    const float latencyWindows = cyclesNeeded / static_cast<float>(config.wmmaLatency);
    return static_cast<int>(std::ceil(baseWindows + latencyWindows));
}

}  // namespace

std::vector<int> computeDsLoadWmmaWindowDistribution(int dsLoadCount,
                                                     const DsLoadBudgetConfig& config) {
    std::vector<int> distribution;
    if (dsLoadCount <= 0 || config.dsReadPerCap <= 0) return distribution;

    const int perCap = config.dsReadPerCap;
    const int depth = std::max(0, config.dsReadQueueDepth);
    const int initialLoads = depth > 0 ? std::min(dsLoadCount, depth) : dsLoadCount;
    int remaining = dsLoadCount;
    for (int initialRemaining = initialLoads; initialRemaining > 0;) {
        const int contribution = std::min(perCap, initialRemaining);
        distribution.push_back(contribution);
        initialRemaining -= contribution;
        remaining -= contribution;
    }

    const bool hasThrottle =
        depth > 0 && config.dsReadThrottleLatency > 0 && config.wmmaLatency > 0;
    for (int overflowIndex = 0; remaining > 0; ++overflowIndex, --remaining) {
        int target = hasThrottle
                         ? computeThrottleOnlyWindowsNeeded(depth + overflowIndex + 1, config) - 1
                         : static_cast<int>(distribution.size());
        target = std::max(0, target);
        while (target < static_cast<int>(distribution.size()) &&
               distribution[static_cast<size_t>(target)] >= perCap)
            ++target;
        if (target >= static_cast<int>(distribution.size()))
            distribution.resize(static_cast<size_t>(target) + 1, 0);
        ++distribution[static_cast<size_t>(target)];
    }
    return distribution;
}

int computeDsLoadWmmaWindowsNeeded(int dsLoadCount, const DsLoadBudgetConfig& config) {
    return static_cast<int>(computeDsLoadWmmaWindowDistribution(dsLoadCount, config).size());
}

// The previous prefix-density implementation was intentionally removed. This
// entry point now receives the barrier analysis computed by
// CDNA5ReadyQueue::onInitRegion and keeps it as the input for the next budget
// policy.
RegionHideBudget analyzeWmmaHideBudget(const dag::RegionDAG& regionDag,
                                       const std::vector<WmmaHideBudgetBarrierInfo>& barriers,
                                       int wmmaHideBudgetBase,
                                       const DsLoadBudgetConfig& dsLoadConfig) {
    RegionHideBudget budget;
    budget.barriers = barriers;
    budget.issueBudgetByWmmaIndex = !barriers.empty();
    budget.wmmaHideBudgetBase = std::max(0, wmmaHideBudgetBase);

    // Step 1: count every DAG instruction. This deliberately matches
    // CDNA5ReadyQueue::rememberPick(): each picked non-WMMA node contributes one,
    // including barriers and pseudo instructions, regardless of issueCycles.
    for (const dag::DAGNode& node : regionDag.nodes) {
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

    // Initial policy: one window per WMMA, each starting with an instruction
    // budget of 0. Preserve program/DAG-node order and build the instruction
    // lookup at the same time.
    budget.windows.reserve(static_cast<size_t>(budget.wmmaInstructionCount));
    for (const dag::DAGNode& node : regionDag.nodes) {
        if (!isMatrixInstruction(*node.inst)) continue;
        WmmaWindowBudget window;
        window.wmma = node.inst;
        budget.windowIndex[node.inst] = budget.numWindows();
        budget.windows.push_back(window);
    }

    auto addDsBudget = [&](int window, int count) {
        if (count <= 0) return;
        WmmaWindowBudget& target = budget.windows[static_cast<size_t>(window)];
        target.dsLoadBudget += count;
        target.issueBudget += count;
    };

    auto distributeEvenly = [&](int begin, int end, int dsLoads) {
        const int span = end - begin;
        if (span <= 0 || dsLoads <= 0) return;
        const int perWindow = dsLoads / span;
        const int remainder = dsLoads % span;
        for (int i = begin; i < end; ++i) {
            const int contribution = perWindow + (i - begin < remainder ? 1 : 0);
            addDsBudget(i, contribution);
        }
    };

    // Use the same hard-cap/throttle model as computeWmmaWindowsNeeded. If a
    // barrier threshold leaves fewer legal windows than the model requests,
    // preserve the total DS demand in the final legal window.
    auto distributeByThrottle = [&](int begin, int end, int dsLoads) {
        const int span = end - begin;
        if (span <= 0 || dsLoads <= 0) return;
        if (dsLoadConfig.dsReadPerCap <= 0) {
            distributeEvenly(begin, end, dsLoads);
            return;
        }

        const std::vector<int> distribution =
            computeDsLoadWmmaWindowDistribution(dsLoads, dsLoadConfig);
        for (int i = 0; i < static_cast<int>(distribution.size()); ++i)
            addDsBudget(begin + std::min(i, span - 1), distribution[static_cast<size_t>(i)]);
    };

    // Step 2: distribute every Before barrier's DS loads over the WMMA windows at
    // and after its final threshold. Overlapping groups preserve the existing
    // even policy; non-overlapping groups use the throttle-shaped policy.
    for (const WmmaHideBudgetBarrierInfo& info : barriers) {
        if (info.position != WmmaHideBudgetBarrierPosition::Before) continue;

        const int begin = std::clamp(info.threshold, 0, budget.numWindows());
        const int span = budget.numWindows() - begin;
        const int dsLoads = std::max(0, info.dsLoadCount);
        if (span == 0 || dsLoads == 0) {
            PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis before] barrier=" << info.barrier
                                 << " threshold=" << info.threshold << " begin=" << begin
                                 << " span=" << span << " dsLoadCount=" << dsLoads
                                 << " action=skip\n");
            continue;
        }

        if (info.overlap)
            distributeEvenly(begin, budget.numWindows(), dsLoads);
        else
            distributeByThrottle(begin, budget.numWindows(), dsLoads);
        PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis before] barrier=" << info.barrier
                             << " threshold=" << info.threshold << " begin=" << begin
                             << " end=" << budget.numWindows() << " dsLoadCount=" << dsLoads
                             << " overlap=" << info.overlap
                             << " policy=" << (info.overlap ? "even" : "throttle") << "\n");
    }

    // Step 3: distribute every After barrier's DS loads from window 0 up to the
    // smaller of its required WMMA span and final threshold, using the same
    // overlap-dependent policy as Before groups.
    for (const WmmaHideBudgetBarrierInfo& info : barriers) {
        if (info.position != WmmaHideBudgetBarrierPosition::After) continue;

        const int end =
            std::clamp(std::min(info.dsLoadWmmaNeeded, info.threshold), 0, budget.numWindows());
        const int dsLoads = std::max(0, info.dsLoadCount);
        if (end == 0 || dsLoads == 0) {
            PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis after] barrier=" << info.barrier
                                 << " threshold=" << info.threshold
                                 << " wmmaNeeded=" << info.dsLoadWmmaNeeded << " end=" << end
                                 << " dsLoadCount=" << dsLoads << " action=skip\n");
            continue;
        }

        if (info.overlap)
            distributeEvenly(0, end, dsLoads);
        else
            distributeByThrottle(0, end, dsLoads);
        PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis after] barrier=" << info.barrier
                             << " threshold=" << info.threshold << " wmmaNeeded="
                             << info.dsLoadWmmaNeeded << " begin=0" << " end=" << end
                             << " dsLoadCount=" << dsLoads << " overlap=" << info.overlap
                             << " policy=" << (info.overlap ? "even" : "throttle") << "\n");
    }

    // Step 4: place remaining non-DS-load instructions in the first 50% of WMMA
    // windows. Walk top-down first and fill each window to wmmaHideBudgetBase.
    // Only after every front-half window reaches the base do we distribute any
    // remainder evenly.
    const int frontHalfEnd = (budget.numWindows() + 1) / 2;
    int remainingNonDs = budget.nonDsLoadInstructionCount;
    if (frontHalfEnd > 0 && remainingNonDs > 0) {
        for (int i = 0; i < frontHalfEnd && remainingNonDs > 0; ++i) {
            WmmaWindowBudget& window = budget.windows[static_cast<size_t>(i)];
            const int deficit = std::max(0, budget.wmmaHideBudgetBase - window.issueBudget);
            const int contribution = std::min(deficit, remainingNonDs);
            window.issueBudget += contribution;
            remainingNonDs -= contribution;
            PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis non-ds fill] index=" << i
                                 << " deficit=" << deficit << " contribution=" << contribution
                                 << " budget=" << window.issueBudget
                                 << " remaining=" << remainingNonDs << "\n");
        }
    }

    if (frontHalfEnd > 0 && remainingNonDs > 0) {
        const int perWindow = remainingNonDs / frontHalfEnd;
        const int remainder = remainingNonDs % frontHalfEnd;
        for (int i = 0; i < frontHalfEnd; ++i) {
            const int contribution = perWindow + (i < remainder ? 1 : 0);
            budget.windows[static_cast<size_t>(i)].issueBudget += contribution;
        }
        PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis non-ds spread] begin=0" << " end="
                             << frontHalfEnd << " remainingBeforeSpread=" << remainingNonDs
                             << " perWindow=" << perWindow << " remainder=" << remainder << "\n");
        remainingNonDs = 0;
    }
    PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis non-ds] end=" << frontHalfEnd
                         << " nonDsLoadCount=" << budget.nonDsLoadInstructionCount
                         << " remaining=" << remainingNonDs << "\n");

    PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis] dagNodes=" << regionDag.nodes.size()
                         << " wmmaInstructions=" << budget.wmmaInstructionCount
                         << " nonWmmaInstructions=" << budget.nonWmmaInstructionCount
                         << " dsLoadInstructions=" << budget.dsLoadInstructionCount
                         << " nonDsLoadInstructions=" << budget.nonDsLoadInstructionCount
                         << " wmmaHideBudgetBase=" << budget.wmmaHideBudgetBase
                         << " barriers=" << barriers.size() << "\n");
    for (const WmmaHideBudgetBarrierInfo& info : barriers) {
        PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis barrier] barrier=" << info.barrier
                             << " position="
                             << (info.position == WmmaHideBudgetBarrierPosition::After ? "after"
                                                                                       : "before")
                             << " threshold=" << info.threshold << " dsLoadCount="
                             << info.dsLoadCount << " dsLoadWmmaNeeded=" << info.dsLoadWmmaNeeded
                             << " overlap=" << info.overlap << "\n");
    }
    for (int i = 0; i < budget.numWindows(); ++i) {
        PASS_DEBUG(std::cerr << "[WmmaHideBudgetAnalysis window] index=" << i << " issueBudget="
                             << budget.windows[static_cast<size_t>(i)].issueBudget
                             << " dsLoadBudget="
                             << budget.windows[static_cast<size_t>(i)].dsLoadBudget << "\n");
    }
    return budget;
}

}  // namespace stinkytofu
