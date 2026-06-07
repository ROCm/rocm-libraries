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
#include "stinkytofu/transforms/asm/waitcnt/ShallowPredPromotion.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace waitcnt {

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

namespace {

CounterKind classifyMemOp(const StinkyInstruction& inst) {
    if (isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst)) return CK_DS;
    if (isGlobalMemLoad(inst) || isGlobalMemStore(inst)) return CK_Buffer;
    if (isTensorLoad(inst)) return CK_Tensor;
    return CK_Count;
}

bool isPhi(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::PHI;
}

bool isOnSamePipeline(const StinkyInstruction& a, const StinkyInstruction& b) {
    return classifyMemOp(a) == CK_DS && classifyMemOp(b) == CK_DS;
}

/// Collect the concrete memops on counter `c` that `consumer` transitively
/// depends on, flattening PHI sources to their incoming defs.
std::unordered_set<StinkyInstruction*> collectMemOpDeps(StinkyInstruction* consumer,
                                                        CounterKind c) {
    std::unordered_set<StinkyInstruction*> result;
    std::unordered_set<StinkyInstruction*> seenPhi;
    std::function<void(StinkyInstruction*)> recurse = [&](StinkyInstruction* x) {
        for (StinkyInstruction* src : x->getSources()) {
            if (src == nullptr) continue;
            if (isPhi(*src)) {
                if (seenPhi.insert(src).second) recurse(src);
                continue;
            }
            if (classifyMemOp(*src) != c) continue;
            if (isOnSamePipeline(*src, *consumer)) continue;
            result.insert(src);
        }
    };
    recurse(consumer);
    return result;
}

/// Set / tighten a specific counter field in `spec`.
void setCounter(WaitCountSpec& spec, CounterKind c, int w) {
    switch (c) {
        case CK_DS:
            spec.dsCount = w;
            break;
        case CK_Buffer:
            spec.bufferCount = w;
            break;
        case CK_Tensor:
            spec.tensorCount = w;
            break;
        default:
            break;
    }
}

int getCounter(const WaitCountSpec& spec, CounterKind c) {
    switch (c) {
        case CK_DS:
            return spec.dsCount;
        case CK_Buffer:
            return spec.bufferCount;
        case CK_Tensor:
            return spec.tensorCount;
        default:
            return WaitCountSpec::kUnused;
    }
}

/// Compute the strictest required wait on counter `c` if only path
/// through `pred` were live. `deps` is the consumer's dep set on `c`.
/// Returns -1 when no dep is in flight on this path.
int perPredRequiredWait(const DataflowResult& dfr, BasicBlock* pred, CounterKind c,
                        const std::unordered_set<StinkyInstruction*>& deps) {
    auto it = dfr.exitState.find(pred);
    if (it == dfr.exitState.end()) return -1;
    int best = -1;
    for (const auto& q : it->second.queues[c]) {
        for (StinkyInstruction* d : deps) {
            int n = q.countFrom(d);
            if (n > 0) {
                int w = n - 1;
                if (best < 0 || w < best) best = w;  // strictest dep on this path
            }
        }
    }
    return best;
}

}  // namespace

// ---------------------------------------------------------------------------
// ShallowPredPromotion
// ---------------------------------------------------------------------------

void ShallowPredPromotion::rewrite(WaitInsertionPlan& plan, const DataflowResult& dfr,
                                   Function& /*func*/) {
    // Accumulate per-pred per-counter tail drains. For each pred/counter
    // pair we keep the STRICTEST (smallest) requested immediate so every
    // requesting anchor is satisfied.
    std::unordered_map<BasicBlock*, std::unordered_map<int, int>> tailMin;

    for (auto& kv : plan.anchorWaits) {
        StinkyInstruction* consumer = kv.first;
        WaitCountSpec& spec = kv.second;
        if (consumer == nullptr) continue;
        BasicBlock* bb = consumer->getParent();
        if (bb == nullptr) continue;

        const auto& preds = bb->getPredecessors();
        if (preds.size() < 2) continue;  // not a join

        for (int ci = 0; ci < CK_Count; ++ci) {
            CounterKind c = static_cast<CounterKind>(ci);
            int curWait = getCounter(spec, c);
            if (curWait == WaitCountSpec::kUnused) continue;

            auto deps = collectMemOpDeps(consumer, c);
            if (deps.empty()) continue;

            struct PathInfo {
                BasicBlock* pred;
                int w;
                bool correctable;
            };
            std::vector<PathInfo> infos;
            infos.reserve(preds.size());

            for (BasicBlock* p : preds) {
                int w = perPredRequiredWait(dfr, p, c, deps);
                if (w < 0) continue;  // unconstrained on this path
                // Correctable: not a self-pred, and pred has only this
                // block as successor (else the tail drain would also fire
                // on pred's other successors).
                bool correctable = (p != bb && p->getSuccessors().size() == 1);
                infos.push_back({p, w, correctable});
            }
            if (infos.empty()) continue;

            int maxCorrectable = std::numeric_limits<int>::min();
            int minUncorrectable = std::numeric_limits<int>::max();
            for (const auto& info : infos) {
                if (info.correctable) {
                    if (info.w > maxCorrectable) maxCorrectable = info.w;
                } else {
                    if (info.w < minUncorrectable) minUncorrectable = info.w;
                }
            }

            int anchorWait;
            if (maxCorrectable == std::numeric_limits<int>::min()) {
                anchorWait = minUncorrectable;
            } else if (minUncorrectable == std::numeric_limits<int>::max()) {
                anchorWait = maxCorrectable;
            } else {
                anchorWait = std::min(maxCorrectable, minUncorrectable);
            }

            if (anchorWait <= curWait) continue;  // no relaxation possible

            // Record corrective drains for correctable preds whose strictest
            // wait is below the new (relaxed) anchor wait.
            for (const auto& info : infos) {
                if (!info.correctable) continue;
                if (info.w >= anchorWait) continue;
                auto& m = tailMin[info.pred];
                auto mit = m.find(ci);
                if (mit == m.end() || info.w < mit->second) m[ci] = info.w;
            }

            setCounter(spec, c, anchorWait);
        }
    }

    // Materialise tail drains into the plan, deterministically ordered.
    plan.tailDrains.clear();
    std::vector<BasicBlock*> orderedPreds;
    orderedPreds.reserve(tailMin.size());
    for (const auto& kv : tailMin) orderedPreds.push_back(kv.first);
    std::sort(orderedPreds.begin(), orderedPreds.end(),
              [](BasicBlock* a, BasicBlock* b) { return a < b; });

    for (BasicBlock* pred : orderedPreds) {
        const auto& m = tailMin.at(pred);
        TailDrain td;
        td.predBB = pred;
        for (const auto& kv : m) setCounter(td.spec, static_cast<CounterKind>(kv.first), kv.second);
        if (td.spec.isValid()) plan.tailDrains.push_back(td);
    }
}

}  // namespace waitcnt
}  // namespace stinkytofu
