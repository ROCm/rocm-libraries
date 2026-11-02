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
#include "stinkytofu/transforms/asm/waitcnt/WmmaGroupDsWaitMerge.hpp"

#include <algorithm>
#include <vector>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

namespace stinkytofu {
namespace waitcnt {

namespace {

void mergeGroup(WaitInsertionPlan& plan, const std::vector<StinkyInstruction*>& group,
                int dsCount) {
    plan.anchorWaits[group.front()].dsCount = dsCount;
    for (size_t i = 1; i < group.size(); ++i) {
        auto it = plan.anchorWaits.find(group[i]);
        it->second.dsCount = WaitCountSpec::kUnused;
        if (!it->second.isValid()) plan.anchorWaits.erase(it);
    }
}

}  // namespace

void WmmaGroupDsWaitMerge::rewrite(WaitInsertionPlan& plan, const DataflowResult&, Function& func) {
    if (groupSize_ < 2) return;
    for (BasicBlock& bb : func) {
        std::vector<StinkyInstruction*> group;
        int merged = WaitCountSpec::kUnused;
        int dsOpsSinceFirst = 0;
        for (IRBase& ir : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            if (!isMatrixInstruction(*inst)) {
                if (classifyMemOp(*inst) == CK_DS) ++dsOpsSinceFirst;
                continue;
            }
            auto it = plan.anchorWaits.find(inst);
            if (it == plan.anchorWaits.end() || it->second.dsCount == WaitCountSpec::kUnused) {
                group.clear();
                continue;
            }
            const int dsCount = it->second.dsCount;
            if (group.empty()) {
                merged = dsCount;
                dsOpsSinceFirst = 0;
            } else {
                merged = std::min(merged, std::max(0, dsCount - dsOpsSinceFirst));
            }
            group.push_back(inst);
            if (static_cast<int>(group.size()) == groupSize_) {
                mergeGroup(plan, group, merged);
                group.clear();
            }
        }
    }
}

}  // namespace waitcnt
}  // namespace stinkytofu
