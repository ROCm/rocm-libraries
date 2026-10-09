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
#include "stinkytofu/transforms/asm/waitcnt/RoleSplitDrain.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace waitcnt {

namespace {

bool isRingPublishAnchor(const StinkyInstruction& inst) {
    const auto* ring = inst.getModifier<LdsRingData>();
    return ring != nullptr &&
           std::any_of(ring->accesses.begin(), ring->accesses.end(), [](const LdsRingAccess& a) {
               return a.kind == LdsRingAccessKind::Publish;
           });
}

void addTensorTailDrain(WaitInsertionPlan& plan, BasicBlock* pred, int wait) {
    for (TailDrain& drain : plan.tailDrains) {
        if (drain.predBB != pred) continue;
        if (drain.spec.tensorCount == WaitCountSpec::kUnused || wait < drain.spec.tensorCount)
            drain.spec.tensorCount = wait;
        return;
    }
    TailDrain drain;
    drain.predBB = pred;
    drain.spec.tensorCount = wait;
    plan.tailDrains.push_back(drain);
}

}  // namespace

void RoleSplitDrain::rewrite(WaitInsertionPlan& plan, const DataflowResult& /*dfr*/,
                             Function& func) {
    for (BasicBlock& bb : func) {
        const auto& preds = bb.getPredecessors();
        if (preds.size() < 2) continue;

        // Only the block's first tensor event sees the predecessors' tensor
        // queues unchanged.
        StinkyInstruction* anchor = nullptr;
        WaitCountSpec* spec = nullptr;
        for (IRBase& ir : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            auto it = plan.anchorWaits.find(inst);
            const bool waitsOnTensor =
                it != plan.anchorWaits.end() && it->second.tensorCount != WaitCountSpec::kUnused;
            if (!waitsOnTensor && classifyMemOp(*inst) != CK_Tensor &&
                !inst->is(InstFlag::IF_WaitTensorCnt) && inst->getModifier<LdsRingData>() == nullptr)
                continue;
            if (waitsOnTensor && isRingPublishAnchor(*inst)) {
                anchor = inst;
                spec = &it->second;
            }
            break;
        }
        if (anchor == nullptr) continue;

        std::vector<std::pair<BasicBlock*, int>> perPred;
        bool differs = false;
        for (BasicBlock* pred : preds) {
            // A drain at the tail of `pred` must not also fire on its other edges.
            if (pred == &bb || pred->getSuccessors().size() != 1) {
                perPred.clear();
                break;
            }
            const int wait = df.requiredWaitViaPred(anchor, pred, CK_Tensor);
            if (!perPred.empty() && wait != perPred.front().second) differs = true;
            perPred.emplace_back(pred, wait);
        }
        if (perPred.size() != preds.size() || !differs) continue;

        for (const auto& [pred, wait] : perPred) {
            if (wait != WaitCountSpec::kUnused) addTensorTailDrain(plan, pred, wait);
        }
        spec->tensorCount = WaitCountSpec::kUnused;
    }
}

}  // namespace waitcnt
}  // namespace stinkytofu
