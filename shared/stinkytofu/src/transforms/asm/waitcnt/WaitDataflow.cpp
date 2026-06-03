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
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

#include <algorithm>
#include <iostream>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "WaitDataflow"

namespace stinkytofu {
namespace waitcnt {

// ---------------------------------------------------------------------------
// Classification helpers
// ---------------------------------------------------------------------------

namespace {

CounterKind classifyMemOp(const StinkyInstruction& inst) {
    if (isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst)) return CK_DS;
    if (isGlobalMemLoad(inst) || isGlobalMemStore(inst)) return CK_Buffer;
    if (isTensorLoad(inst)) return CK_Tensor;
    return CK_Count;  // sentinel: not tracked
}

bool isPhi(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::PHI;
}

bool isTensorAnchor(const StinkyInstruction& inst) {
    return isBarrier(inst) || isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst);
}

bool isLdsWriterAnchor(const StinkyInstruction& inst) {
    return isTensorLoad(inst) || isDSWrite(inst);
}

bool isOnSamePipeline(const StinkyInstruction& a, const StinkyInstruction& b) {
    auto aK = classifyMemOp(a);
    auto bK = classifyMemOp(b);
    // Only DS-vs-DS share a hardware FIFO today.
    return aK == CK_DS && bK == CK_DS;
}

}  // namespace

// ---------------------------------------------------------------------------
// CounterQueue / DataflowState
// ---------------------------------------------------------------------------

int CounterQueue::countFrom(StinkyInstruction* op) const {
    auto it = std::find(ops.begin(), ops.end(), op);
    if (it == ops.end()) return 0;
    return static_cast<int>(std::distance(it, ops.end()));
}

void DataflowState::clear() {
    for (auto& q : queues) q.ops.clear();
    phiSummaries.clear();
}

bool DataflowState::operator==(const DataflowState& other) const {
    for (int c = 0; c < CK_Count; ++c) {
        if (!(queues[c] == other.queues[c])) return false;
    }
    if (phiSummaries.size() != other.phiSummaries.size()) return false;
    for (const auto& kv : phiSummaries) {
        auto it = other.phiSummaries.find(kv.first);
        if (it == other.phiSummaries.end()) return false;
        if (!(kv.second == it->second)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// WaitDataflow
// ---------------------------------------------------------------------------

WaitDataflow::WaitDataflow(Function& func, const DominanceInfo& /*domInfo*/,
                           const std::vector<BasicBlock*>& rpo)
    : func_(func), rpo_(rpo) {
    // Default cap: max(8, 2*N) with a ceiling of 64. Tunable via setIterationCap.
    const unsigned n = static_cast<unsigned>(rpo_.size());
    iterationCap_ = std::min<unsigned>(64u, std::max<unsigned>(8u, 2u * n));
}

DataflowState WaitDataflow::mergeFromPredecessors(BasicBlock& bb) const {
    DataflowState merged;
    const auto& preds = bb.getPredecessors();
    if (preds.empty()) return merged;

    // Step 1: ordered intersection of per-counter queues across predecessors.
    // We keep the order from the first pred that has stored exit state.
    BasicBlock* anchorPred = nullptr;
    for (BasicBlock* p : preds) {
        if (result_.exitState.count(p)) {
            anchorPred = p;
            break;
        }
    }
    if (anchorPred == nullptr) return merged;

    const DataflowState& anchorState = result_.exitState.at(anchorPred);
    for (int c = 0; c < CK_Count; ++c) {
        for (StinkyInstruction* op : anchorState.queues[c].ops) {
            bool inAll = true;
            for (BasicBlock* p : preds) {
                if (p == anchorPred) continue;
                auto it = result_.exitState.find(p);
                if (it == result_.exitState.end()) {
                    // Unprocessed pred -> treat as empty: op is not in flight
                    // on every path, so drop from merge.
                    inAll = false;
                    break;
                }
                const auto& q = it->second.queues[c].ops;
                if (std::find(q.begin(), q.end(), op) == q.end()) {
                    inAll = false;
                    break;
                }
            }
            if (inAll) merged.queues[c].ops.push_back(op);
        }
    }

    // Step 2: build PhiSummary for each PHI in @p bb. The PHI's sources are
    // ordered to match bb.getPredecessors(). For each source we look up its
    // contribution per counter:
    //   - memop: use the predecessor's exit-queue position (countFrom - 1).
    //   - PHI:   look up that PHI's summary in the pred's exit state.
    //   - else:  ignored (VALU or undefined).
    // We take MAX across paths -- the strictest constraint must hold on
    // every path through the consumer.
    for (IRBase& ir : bb) {
        auto* phi = dyn_cast<StinkyInstruction>(&ir);
        if (phi == nullptr) continue;
        if (!isPhi(*phi)) break;  // PHIs are clustered at block top

        const auto& srcs = phi->getSources();
        PhiSummary summary;
        for (size_t j = 0; j < preds.size() && j < srcs.size(); ++j) {
            StinkyInstruction* src = srcs[j];
            if (src == nullptr) continue;
            BasicBlock* p = preds[j];
            auto it = result_.exitState.find(p);
            if (it == result_.exitState.end()) continue;

            auto recordWait = [&](CounterKind c, int w) {
                if (w < 0) return;
                if (summary.waits[c] == WaitCountSpec::kUnused || w > summary.waits[c]) {
                    summary.waits[c] = w;
                }
            };

            if (isPhi(*src)) {
                auto sit = it->second.phiSummaries.find(src);
                if (sit != it->second.phiSummaries.end()) {
                    for (int c = 0; c < CK_Count; ++c) {
                        recordWait(static_cast<CounterKind>(c), sit->second.waits[c]);
                    }
                }
                continue;
            }

            CounterKind c = classifyMemOp(*src);
            if (c == CK_Count) continue;
            int n = it->second.queues[c].countFrom(src);
            if (n > 0) recordWait(c, n - 1);
        }
        merged.phiSummaries[phi] = summary;
    }

    return merged;
}

// Per-counter local bookkeeping during a block walk. Mirrors the redundancy
// elision logic from the old pass: if the previously emitted wait + the
// number of new ops issued since is already tight enough for the new
// requirement, suppress the emit.
namespace {
struct CounterEmitState {
    int lastEmittedWait = WaitCountSpec::kUnused;
    int opsSinceLastWait = 0;

    void recordNewOp() {
        ++opsSinceLastWait;
    }
    bool needsNewWait(int required) const {
        return lastEmittedWait == WaitCountSpec::kUnused ||
               lastEmittedWait + opsSinceLastWait > required;
    }
    void recordEmittedWait(int v) {
        lastEmittedWait = v;
        opsSinceLastWait = 0;
    }
};
}  // namespace

void WaitDataflow::transferBlock(BasicBlock& bb, DataflowState& state) {
    auto& plan = emitPlan_[&bb];
    plan.clear();

    CounterEmitState emit[CK_Count];

    for (IRBase& ir : bb) {
        auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst == nullptr) continue;
        if (isPhi(*inst)) continue;  // PhiSummary already computed in merge

        // Required wait per counter for this consumer. -1 = no constraint.
        int required[CK_Count] = {WaitCountSpec::kUnused, WaitCountSpec::kUnused,
                                  WaitCountSpec::kUnused};
        auto recordRequired = [&](CounterKind c, int w) {
            if (w < 0) return;
            if (required[c] == WaitCountSpec::kUnused || w < required[c]) {
                // We want the STRICTEST per source = smallest w (closest to
                // tail). Across multiple constrained sources on the same
                // counter, the smallest w is what guarantees safety for all.
                required[c] = w;
            }
        };

        // Walk UD edges. Each source either is a memop on some counter, is
        // a PHI (use its summary), or is something we don't track.
        for (StinkyInstruction* src : inst->getSources()) {
            if (src == nullptr) continue;

            if (isPhi(*src)) {
                auto it = state.phiSummaries.find(src);
                if (it == state.phiSummaries.end()) continue;
                for (int c = 0; c < CK_Count; ++c) {
                    recordRequired(static_cast<CounterKind>(c), it->second.waits[c]);
                }
                continue;
            }

            CounterKind c = classifyMemOp(*src);
            if (c == CK_Count) continue;

            // Same-pipeline FIFO: the hardware orders these implicitly, no
            // synthetic wait needed.
            if (isOnSamePipeline(*src, *inst)) continue;

            int n = state.queues[c].countFrom(src);
            if (n > 0) recordRequired(c, n - 1);
        }

        // Conservative fallback: tensor-wait anchor without MemTokenData
        // cannot be proved disjoint from any pending tensor load -> force 0.
        if (isTensorAnchor(*inst) && inst->getModifier<MemTokenData>() == nullptr &&
            !state.queues[CK_Tensor].ops.empty()) {
            required[CK_Tensor] = 0;
        }
        // Same fallback for an LDS writer that lacks MemTokenData vs. any
        // pending DS op on a non-same pipeline. We approximate "non-same
        // pipeline" by "queue non-empty AND writer is not a ds_write" (the
        // only same-pipeline writer case today).
        if (isLdsWriterAnchor(*inst) && inst->getModifier<MemTokenData>() == nullptr &&
            !state.queues[CK_DS].ops.empty() && !isDSWrite(*inst)) {
            required[CK_DS] = 0;
        }
        // Barrier vs. pending DS ops: drain when either side lacks tokens
        // or when token sets overlap. The pseudo-reg UD chain catches the
        // overlap case (barriers have a pseudo-reg src per memtoken) but
        // not the missing-token case, so handle it here.
        if (isBarrier(*inst) && !state.queues[CK_DS].ops.empty()) {
            bool needs = inst->getModifier<MemTokenData>() == nullptr;
            if (!needs) {
                for (StinkyInstruction* op : state.queues[CK_DS].ops) {
                    if (op->getModifier<MemTokenData>() == nullptr) {
                        needs = true;
                        break;
                    }
                }
            }
            if (needs) required[CK_DS] = 0;
        }

        // Decide what to emit, applying redundancy elision.
        WaitCountSpec spec;
        for (int c = 0; c < CK_Count; ++c) {
            if (required[c] == WaitCountSpec::kUnused) continue;
            if (!emit[c].needsNewWait(required[c])) continue;

            switch (c) {
                case CK_DS:     spec.dsCount     = required[c]; break;
                case CK_Buffer: spec.bufferCount = required[c]; break;
                case CK_Tensor: spec.tensorCount = required[c]; break;
                default: break;
            }
            emit[c].recordEmittedWait(required[c]);

            // A wait drains the queue down to the kept tail.
            auto& q = state.queues[c].ops;
            if (required[c] <= 0) {
                q.clear();
            } else if (static_cast<int>(q.size()) > required[c]) {
                q.erase(q.begin(), q.end() - required[c]);
            }
        }
        if (spec.isValid()) plan.emplace_back(inst, spec);

        // Append this instruction to its counter queue (after the wait, so
        // the wait's snapshot of the queue excludes its own consumer --
        // matches hardware semantics: the wait runs before the issue).
        CounterKind self = classifyMemOp(*inst);
        if (self != CK_Count) {
            state.queues[self].ops.push_back(inst);
            emit[self].recordNewOp();
        }
    }
}

bool WaitDataflow::solve() {
    capHit_ = false;
    result_.entryState.clear();
    result_.exitState.clear();
    emitPlan_.clear();

    // Seed every block with empty state so lookups during iteration always
    // succeed (an empty entry is the lattice bottom).
    for (BasicBlock* bb : rpo_) {
        result_.entryState[bb] = DataflowState();
        result_.exitState[bb] = DataflowState();
    }

    for (unsigned iter = 0; iter < iterationCap_; ++iter) {
        bool changed = false;
        for (BasicBlock* bb : rpo_) {
            DataflowState entry = mergeFromPredecessors(*bb);
            DataflowState working = entry;
            transferBlock(*bb, working);

            if (!(result_.exitState[bb] == working)) {
                result_.exitState[bb] = std::move(working);
                changed = true;
            }
            result_.entryState[bb] = std::move(entry);
        }
        if (!changed) return true;
    }

    // Iteration cap hit. Mark for conservative emission downstream.
    capHit_ = true;
    std::cerr << "[WaitDataflow] iteration cap " << iterationCap_
              << " hit; falling back to s_wait_* 0 at every anchor.\n";
    return false;
}

WaitInsertionPlan WaitDataflow::materializePlan() const {
    WaitInsertionPlan plan;

    if (capHit_) {
        // Conservative fallback: every anchor we recorded gets a
        // s_wait_* 0 for every counter it touched in any iteration.
        for (const auto& kv : emitPlan_) {
            for (const auto& entry : kv.second) {
                WaitCountSpec spec;
                if (entry.second.dsCount != WaitCountSpec::kUnused) spec.dsCount = 0;
                if (entry.second.bufferCount != WaitCountSpec::kUnused) spec.bufferCount = 0;
                if (entry.second.tensorCount != WaitCountSpec::kUnused) spec.tensorCount = 0;
                if (spec.isValid()) plan.anchorWaits[entry.first] = spec;
            }
        }
        return plan;
    }

    for (const auto& kv : emitPlan_) {
        for (const auto& entry : kv.second) {
            // Multiple iterations of the dataflow can produce different
            // emit plans for the same instruction; the LAST iteration's
            // plan is the fixed-point plan, and that is what we have here
            // because emitPlan_ is overwritten each pass. Merge per-counter
            // entries: a later (looser) emit might leave a counter unused
            // but an earlier (tighter) emit may have set it. We use the
            // entry from the last pass directly.
            plan.anchorWaits[entry.first] = entry.second;
        }
    }
    return plan;
}

}  // namespace waitcnt
}  // namespace stinkytofu
