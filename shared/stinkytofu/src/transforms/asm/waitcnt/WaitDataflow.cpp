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
#include <unordered_set>

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
    return CK_Count;
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
    return classifyMemOp(a) == CK_DS && classifyMemOp(b) == CK_DS;
}

}  // namespace

// ---------------------------------------------------------------------------
// PerPredQueue / DataflowState
// ---------------------------------------------------------------------------

int PerPredQueue::countFrom(StinkyInstruction* op) const {
    auto it = std::find(ops.begin(), ops.end(), op);
    if (it == ops.end()) return 0;
    return static_cast<int>(std::distance(it, ops.end()));
}

void DataflowState::clear() {
    for (auto& v : queues) v.clear();
    phiSummaries.clear();
}

bool DataflowState::operator==(const DataflowState& other) const {
    for (int c = 0; c < CK_Count; ++c) {
        if (queues[c] != other.queues[c]) return false;
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
    : func(func), rpo(rpo) {
    const unsigned n = static_cast<unsigned>(rpo.size());
    iterationCap = std::min<unsigned>(64u, std::max<unsigned>(8u, 2u * n));
}

DataflowState WaitDataflow::mergeFromPredecessors(BasicBlock& bb) const {
    DataflowState entry;
    const auto& preds = bb.getPredecessors();

    // Seed one PerPredQueue per pred per counter from that pred's collapsed
    // exit state. Self-preds (back-edges) are seeded too: at fixed point
    // the back-edge's exit is the loop body's true exit, which is what the
    // header should see.
    for (BasicBlock* p : preds) {
        auto it = result.exitState.find(p);
        if (it == result.exitState.end()) continue;
        const auto& predState = it->second;
        for (int c = 0; c < CK_Count; ++c) {
            // Pred's exit is a single collapsed PerPredQueue per counter.
            // Carry it into bb tagged with the actual predecessor.
            for (const auto& predQ : predState.queues[c]) {
                PerPredQueue q;
                q.pred = p;
                q.ops = predQ.ops;
                entry.queues[c].push_back(std::move(q));
            }
        }
    }

    // Build PhiSummary for each PHI by walking incoming sources against the
    // matching pred's exit state.
    for (IRBase& ir : bb) {
        auto* phi = dyn_cast<StinkyInstruction>(&ir);
        if (phi == nullptr) continue;
        if (!isPhi(*phi)) break;

        const auto& srcs = phi->getSources();
        PhiSummary summary;
        auto recordWait = [&](CounterKind c, int w) {
            if (w < 0) return;
            if (summary.waits[c] == WaitCountSpec::kUnused || w > summary.waits[c]) {
                summary.waits[c] = w;
            }
        };

        for (size_t j = 0; j < preds.size() && j < srcs.size(); ++j) {
            StinkyInstruction* src = srcs[j];
            if (src == nullptr) continue;
            auto pit = result.exitState.find(preds[j]);
            if (pit == result.exitState.end()) continue;
            const auto& predState = pit->second;

            if (isPhi(*src)) {
                auto sit = predState.phiSummaries.find(src);
                if (sit != predState.phiSummaries.end()) {
                    for (int c = 0; c < CK_Count; ++c) {
                        recordWait(static_cast<CounterKind>(c), sit->second.waits[c]);
                    }
                }
                continue;
            }

            CounterKind c = classifyMemOp(*src);
            if (c == CK_Count) continue;
            // Pred has one collapsed queue per counter.
            for (const auto& q : predState.queues[c]) {
                int n = q.countFrom(src);
                if (n > 0) {
                    recordWait(c, n - 1);
                    break;
                }
            }
        }
        entry.phiSummaries[phi] = summary;
    }

    return entry;
}

// Per-counter local bookkeeping during a block walk. Mirrors the redundancy
// elision logic from the old pass: if the previously emitted wait plus the
// number of new ops issued since is already tight enough, suppress the
// new emit.
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

// Trim every per-pred queue in a counter to keep at most @p keep tail ops.
void trimQueues(std::vector<PerPredQueue>& qs, int keep) {
    for (auto& q : qs) {
        if (keep <= 0) {
            q.ops.clear();
        } else if (static_cast<int>(q.ops.size()) > keep) {
            q.ops.erase(q.ops.begin(), q.ops.end() - keep);
        }
    }
}

// Append a local in-block memop to every per-pred queue. Local ops are in
// flight on every CFG path through this block, so they join every path's
// tail. If no per-pred queue exists yet, create a synthetic one
// (pred == nullptr) so the in-block prefix is still tracked.
void appendToAllPaths(std::vector<PerPredQueue>& qs, StinkyInstruction* op) {
    if (qs.empty()) qs.push_back(PerPredQueue{});
    for (auto& q : qs) q.ops.push_back(op);
}

// Collapse per-pred queues for a counter into a single union queue tagged
// with @p selfBlock. Order is first-occurrence across paths, walking
// per-pred queues in their existing order. This is what successors will
// seed their own per-pred queue from.
void collapseToExitView(std::vector<PerPredQueue>& qs, BasicBlock* selfBlock) {
    std::deque<StinkyInstruction*> u;
    std::unordered_set<StinkyInstruction*> seen;
    for (const auto& q : qs) {
        for (StinkyInstruction* op : q.ops) {
            if (seen.insert(op).second) u.push_back(op);
        }
    }
    qs.clear();
    PerPredQueue out;
    out.pred = selfBlock;
    out.ops = std::move(u);
    qs.push_back(std::move(out));
}

}  // namespace

void WaitDataflow::transferBlock(BasicBlock& bb, DataflowState& state) {
    auto& plan = emitPlan[&bb];
    plan.clear();

    CounterEmitState emit[CK_Count];

    for (IRBase& ir : bb) {
        auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst == nullptr) continue;
        if (isPhi(*inst)) continue;  // PhiSummary already computed in merge

        // Required wait per counter. -1 = no constraint yet.
        int required[CK_Count] = {WaitCountSpec::kUnused, WaitCountSpec::kUnused,
                                  WaitCountSpec::kUnused};

        // Tighten required[c] = min(required[c], w). The min across deps on
        // the same counter is what's safe: it drains the closest-to-tail
        // dep, which is the most permissive wait that still satisfies it.
        auto tightenRequired = [&](CounterKind c, int w) {
            if (w < 0) return;
            if (required[c] == WaitCountSpec::kUnused || w < required[c]) required[c] = w;
        };

        // Per-counter lookup of a single dep's strictest wait across all
        // per-pred queues: max over preds where the dep is in flight.
        auto perDepWait = [&](CounterKind c, StinkyInstruction* dep) -> int {
            int best = -1;
            for (const auto& q : state.queues[c]) {
                int n = q.countFrom(dep);
                if (n > 0) {
                    int w = n - 1;
                    if (w > best) best = w;
                }
            }
            return best;
        };

        for (StinkyInstruction* src : inst->getSources()) {
            if (src == nullptr) continue;

            if (isPhi(*src)) {
                auto it = state.phiSummaries.find(src);
                if (it == state.phiSummaries.end()) continue;
                for (int c = 0; c < CK_Count; ++c) {
                    tightenRequired(static_cast<CounterKind>(c), it->second.waits[c]);
                }
                continue;
            }

            CounterKind c = classifyMemOp(*src);
            if (c == CK_Count) continue;
            if (isOnSamePipeline(*src, *inst)) continue;

            int w = perDepWait(c, src);
            if (w >= 0) tightenRequired(c, w);
        }

        auto anyOpInFlight = [&](CounterKind c) {
            for (const auto& q : state.queues[c]) {
                if (!q.ops.empty()) return true;
            }
            return false;
        };

        // Conservative MemTokenData fallbacks. The pseudo-reg UD chain can
        // only express deps the implicit-dependency pass tagged; an
        // untagged anchor or producer means we can't prove disjointness,
        // so we force the matching counter to 0.
        if (isTensorAnchor(*inst) && inst->getModifier<MemTokenData>() == nullptr &&
            anyOpInFlight(CK_Tensor)) {
            required[CK_Tensor] = 0;
        }
        if (isLdsWriterAnchor(*inst) && inst->getModifier<MemTokenData>() == nullptr &&
            anyOpInFlight(CK_DS) && !isDSWrite(*inst)) {
            required[CK_DS] = 0;
        }
        if (isBarrier(*inst) && anyOpInFlight(CK_DS)) {
            bool needs = inst->getModifier<MemTokenData>() == nullptr;
            if (!needs) {
                for (const auto& q : state.queues[CK_DS]) {
                    for (StinkyInstruction* op : q.ops) {
                        if (op->getModifier<MemTokenData>() == nullptr) {
                            needs = true;
                            break;
                        }
                    }
                    if (needs) break;
                }
            }
            if (needs) required[CK_DS] = 0;
        }

        // Decide what to emit (apply redundancy elision) and trim per-pred
        // queues accordingly.
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
            trimQueues(state.queues[c], required[c]);
        }
        if (spec.isValid()) plan.emplace_back(inst, spec);

        // Append self to its counter queue (after the wait, so the wait's
        // snapshot of the queue excludes its own consumer).
        CounterKind self = classifyMemOp(*inst);
        if (self != CK_Count) {
            appendToAllPaths(state.queues[self], inst);
            emit[self].recordNewOp();
        }
    }

    // Collapse per-pred queues into a single union representative for
    // successors.
    for (int c = 0; c < CK_Count; ++c) {
        collapseToExitView(state.queues[c], &bb);
    }
}

bool WaitDataflow::solve() {
    capHit = false;
    result.entryState.clear();
    result.exitState.clear();
    emitPlan.clear();

    // Seed every block with empty state so lookups during iteration always
    // succeed (an empty state is the lattice bottom).
    for (BasicBlock* bb : rpo) {
        result.entryState[bb] = DataflowState();
        result.exitState[bb] = DataflowState();
    }

    for (unsigned iter = 0; iter < iterationCap; ++iter) {
        bool changed = false;
        for (BasicBlock* bb : rpo) {
            DataflowState entry = mergeFromPredecessors(*bb);
            DataflowState working = entry;
            transferBlock(*bb, working);

            if (!(result.exitState[bb] == working)) {
                result.exitState[bb] = std::move(working);
                changed = true;
            }
            result.entryState[bb] = std::move(entry);
        }
        if (!changed) return true;
    }

    capHit = true;
    std::cerr << "[WaitDataflow] iteration cap " << iterationCap
              << " hit; falling back to s_wait_* 0 at every anchor.\n";
    return false;
}

WaitInsertionPlan WaitDataflow::materializePlan() const {
    WaitInsertionPlan plan;

    if (capHit) {
        for (const auto& kv : emitPlan) {
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

    for (const auto& kv : emitPlan) {
        for (const auto& entry : kv.second) {
            plan.anchorWaits[entry.first] = entry.second;
        }
    }
    return plan;
}

}  // namespace waitcnt
}  // namespace stinkytofu
