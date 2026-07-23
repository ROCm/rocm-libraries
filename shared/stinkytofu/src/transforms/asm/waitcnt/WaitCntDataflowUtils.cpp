// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/transforms/asm/waitcnt/WaitCntDataflowUtils.hpp"

#include <algorithm>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
namespace waitcnt {
namespace utils {

bool isPhi(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::PHI;
}

bool isTensorAnchorInst(const StinkyInstruction& inst) {
    return isBarrier(inst) || isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst);
}

bool isLdsWriterAnchorInst(const StinkyInstruction& inst) {
    return isTensorLoad(inst) || isDSWrite(inst);
}

bool isOnSamePipeline(const StinkyInstruction& a, const StinkyInstruction& b) {
    return classifyMemOp(a) == CK_DS && classifyMemOp(b) == CK_DS;
}

bool hasTokenOverlap(const std::vector<int>& a, const std::vector<int>& b) {
    for (int t : a) {
        if (std::find(b.begin(), b.end(), t) != b.end()) return true;
    }
    return false;
}

void trimQueues(std::vector<PerPredQueue>& qs, int keep) {
    for (auto& q : qs) {
        if (keep <= 0) {
            q.ops.clear();
        } else if (static_cast<int>(q.ops.size()) > keep) {
            q.ops.erase(q.ops.begin(), q.ops.end() - keep);
        }
    }
}

bool appendToAllPaths(std::vector<PerPredQueue>& qs, StinkyInstruction* op) {
    if (qs.empty()) qs.push_back(PerPredQueue{});
    bool dropped = false;
    for (auto& q : qs) {
        q.ops.push_back(op);
        while (q.ops.size() > kMaxInFlight) {
            q.ops.pop_front();
            dropped = true;
        }
    }
    return dropped;
}

int phiCurrentQueueWait(StinkyInstruction* phi, CounterKind c, const DataflowState& state,
                        std::unordered_set<StinkyInstruction*>& seen) {
    if (!seen.insert(phi).second) return WaitCountSpec::kUnused;
    int best = WaitCountSpec::kUnused;
    auto tighten = [&](int w) {
        if (w < 0) return;
        if (best == WaitCountSpec::kUnused || w < best) best = w;
    };
    for (StinkyInstruction* src : phi->getSources()) {
        if (src == nullptr) continue;
        if (isPhi(*src)) {
            tighten(phiCurrentQueueWait(src, c, state, seen));
            continue;
        }
        if (classifyMemOp(*src) != c) continue;
        for (const auto& q : state.queues[c]) {
            int n = q.countFrom(src);
            if (n > 0) tighten(n - 1);
        }
    }
    return best;
}

void computeRequiredWaits(StinkyInstruction* inst, const DataflowState& state,
                          const std::array<RawWaitPredicate, CK_Count>& rawNeedsWait,
                          int required[CK_Count]) {
    for (int c = 0; c < CK_Count; ++c) required[c] = WaitCountSpec::kUnused;

    auto tightenRequired = [&](CounterKind c, int w) {
        if (w < 0) return;
        if (required[c] == WaitCountSpec::kUnused || w < required[c]) required[c] = w;
    };

    for (StinkyInstruction* src : inst->getSources()) {
        if (src == nullptr) continue;

        if (isPhi(*src)) {
            for (int c = 0; c < CK_Count; ++c) {
                if (!rawNeedsWait[c](*inst)) continue;
                std::unordered_set<StinkyInstruction*> seen;
                int w = phiCurrentQueueWait(src, static_cast<CounterKind>(c), state, seen);
                tightenRequired(static_cast<CounterKind>(c), w);
            }
            continue;
        }

        CounterKind c = classifyMemOp(*src);
        if (c == CK_Count) continue;
        if (!rawNeedsWait[c](*inst)) continue;

        for (const auto& q : state.queues[c]) {
            int n = q.countFrom(src);
            if (n > 0) tightenRequired(c, n - 1);
        }
    }

    auto anyOpInFlight = [&](CounterKind c) {
        for (const auto& q : state.queues[c]) {
            if (!q.ops.empty()) return true;
        }
        return false;
    };

    auto scanDsAntiDeps = [&](const StinkyInstruction& anchor, const std::vector<int>& anchorTokens,
                              bool barrierMode) {
        for (const auto& q : state.queues[CK_DS]) {
            const int qsize = static_cast<int>(q.ops.size());
            for (int idx = 0; idx < qsize; ++idx) {
                StinkyInstruction* op = q.ops[idx];
                if (op == inst) continue;
                if (!barrierMode && !isDSRead(*op) && !isDSAtomic(*op)) continue;
                if (isOnSamePipeline(anchor, *op)) continue;
                const auto* opTokens = op->getModifier<MemTokenData>();
                bool overlap =
                    (opTokens == nullptr) || hasTokenOverlap(opTokens->tokens, anchorTokens);
                if (!overlap) continue;
                tightenRequired(CK_DS, qsize - idx - 1);
            }
        }
    };

    if (isLdsWriterAnchorInst(*inst)) {
        const auto* tk = inst->getModifier<MemTokenData>();
        if (tk != nullptr) scanDsAntiDeps(*inst, tk->tokens, /*barrierMode=*/false);
    }
    if (isBarrier(*inst)) {
        const auto* tk = inst->getModifier<MemTokenData>();
        if (tk != nullptr) scanDsAntiDeps(*inst, tk->tokens, /*barrierMode=*/true);
    }

    if (isTensorAnchorInst(*inst) && inst->getModifier<MemTokenData>() != nullptr) {
        for (const auto& q : state.queues[CK_Tensor]) {
            const int qsize = static_cast<int>(q.ops.size());
            for (int idx = 0; idx < qsize; ++idx) {
                StinkyInstruction* op = q.ops[idx];
                if (op == inst) continue;
                if (op->getModifier<MemTokenData>() == nullptr) {
                    tightenRequired(CK_Tensor, qsize - idx - 1);
                }
            }
        }
    }

    if (isTensorAnchorInst(*inst) && inst->getModifier<MemTokenData>() == nullptr &&
        anyOpInFlight(CK_Tensor)) {
        required[CK_Tensor] = 0;
    }
    if (isLdsWriterAnchorInst(*inst) && inst->getModifier<MemTokenData>() == nullptr &&
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
}

void seedQueuesFromPredecessors(
    BasicBlock& bb, const std::unordered_map<const BasicBlock*, DataflowState>& exitState,
    DataflowState& entry) {
    for (BasicBlock* p : bb.getPredecessors()) {
        auto it = exitState.find(p);
        if (it == exitState.end()) continue;
        const DataflowState& predState = it->second;
        for (int c = 0; c < CK_Count; ++c) {
            for (const auto& predQ : predState.queues[c]) {
                bool dup = false;
                for (const auto& existing : entry.queues[c]) {
                    if (existing.pred == p && existing.ops == predQ.ops) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    PerPredQueue q;
                    q.pred = p;
                    q.ops = predQ.ops;
                    entry.queues[c].push_back(std::move(q));
                }
            }
        }
    }
}

}  // namespace utils
}  // namespace waitcnt
}  // namespace stinkytofu
