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
#include "stinkytofu/transforms/asm/waitcnt/WmmaRunWaitMerge.hpp"

#include <algorithm>
#include <vector>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace waitcnt {

namespace {

int minUsed(int a, int b) {
    if (a == WaitCountSpec::kUnused) return b;
    if (b == WaitCountSpec::kUnused) return a;
    return std::min(a, b);
}

void mergeInto(WaitCountSpec& into, const WaitCountSpec& from) {
    into.dsCount = minUsed(into.dsCount, from.dsCount);
    into.loadCount = minUsed(into.loadCount, from.loadCount);
    into.kmCount = minUsed(into.kmCount, from.kmCount);
    into.tensorCount = minUsed(into.tensorCount, from.tensorCount);
    into.asyncCount = minUsed(into.asyncCount, from.asyncCount);
    into.tensorTokens.insert(into.tensorTokens.end(), from.tensorTokens.begin(),
                             from.tensorTokens.end());
    std::sort(into.tensorTokens.begin(), into.tensorTokens.end());
    into.tensorTokens.erase(std::unique(into.tensorTokens.begin(), into.tensorTokens.end()),
                            into.tensorTokens.end());
}

// Move every wait planned inside \p run onto its first instruction.
void mergeRun(WaitInsertionPlan& plan, const std::vector<StinkyInstruction*>& run) {
    if (run.size() < 2) return;
    WaitCountSpec merged;
    bool any = false;
    for (StinkyInstruction* inst : run) {
        auto it = plan.anchorWaits.find(inst);
        if (it == plan.anchorWaits.end()) continue;
        mergeInto(merged, it->second);
        any = true;
        plan.anchorWaits.erase(it);
    }
    if (any) plan.anchorWaits[run.front()] = merged;
}

}  // namespace

void WmmaRunWaitMerge::rewrite(WaitInsertionPlan& plan, const DataflowResult&, Function& func) {
    for (BasicBlock& bb : func) {
        std::vector<StinkyInstruction*> run;
        for (IRBase& ir : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            if (isMatrixInstruction(*inst)) {
                run.push_back(inst);
                continue;
            }
            mergeRun(plan, run);
            run.clear();
        }
        mergeRun(plan, run);
    }
}

}  // namespace waitcnt
}  // namespace stinkytofu
