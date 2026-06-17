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
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "WaitDataflow"

namespace stinkytofu {
namespace waitcnt {

// ---------------------------------------------------------------------------
// Per-counter policy
//
// Everything that is *specific to one hardware counter* lives in this table
// so the dataflow transfer (transferBlock) stays counter-agnostic. To add a
// counter, or to change WHEN a counter drains, edit this table -- not the
// transfer loop.
// ---------------------------------------------------------------------------
struct CounterPolicy {
    bool (*isProducer)(const StinkyInstruction&);
    bool (*rawNeedsWait)(const StinkyInstruction&);
};

static const CounterPolicy& defaultCounterPolicy(CounterKind c) {
    // Indexed by CounterKind; row order MUST match the enum. Captureless
    // lambdas decay to plain function pointers, so this stays a constant
    // table with no per-call allocation.
    static const CounterPolicy kPolicies[CK_Count] = {
        // CK_DS: ds_read / ds_write / ds_atomic; every consumer drains.
        {[](const StinkyInstruction& i) { return isDSRead(i) || isDSWrite(i) || isDSAtomic(i); },
         [](const StinkyInstruction&) { return true; }},
        // CK_Buffer: vector global/buffer load+store; every consumer drains.
        {[](const StinkyInstruction& i) { return isBufferMemLoad(i) || isBufferMemStore(i); },
         [](const StinkyInstruction&) { return true; }},
        // CK_KM: SMRD scalar loads (s_load_*); every consumer drains.
        {[](const StinkyInstruction& i) { return isSMemLoad(i); },
         [](const StinkyInstruction&) { return true; }},
        // CK_Tensor: tensor_load_to_lds; every consumer drains.
        {[](const StinkyInstruction& i) { return isTensorLoad(i); },
         [](const StinkyInstruction& i) { return true; }},
    };
    return kPolicies[c];
}

CounterKind classifyMemOp(const StinkyInstruction& inst) {
    for (int c = 0; c < CK_Count; ++c) {
        if (defaultCounterPolicy(static_cast<CounterKind>(c)).isProducer(inst)) {
            return static_cast<CounterKind>(c);
        }
    }
    return CK_Count;
}

namespace {

constexpr size_t kMaxInFlight = 64;

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

bool hasTokenOverlap(const std::vector<int>& a, const std::vector<int>& b) {
    for (int t : a) {
        if (std::find(b.begin(), b.end(), t) != b.end()) return true;
    }
    return false;
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

WaitDataflow::WaitDataflow(Function& /*func*/, const DominanceInfo& /*domInfo*/,
                           const std::vector<BasicBlock*>& rpo)
    : rpo(rpo) {
    const unsigned n = static_cast<unsigned>(rpo.size());
    // A self-loop that never drains a counter fills its per-pred queue one op
    // per fixed-point iteration up to kMaxInFlight before the lattice
    // stabilises, so the cap floor must clear that window for the fixed point
    // to be reached. The solver breaks early on convergence, so this only
    // affects genuinely slow / non-convergent inputs.
    const unsigned floor = static_cast<unsigned>(kMaxInFlight) + 8u;
    iterationCap = std::min<unsigned>(256u, std::max<unsigned>(floor, 2u * n));

    // Seed each counter's RAW-wait constraint with the built-in default
    // from the policy table; callers may override via setRawNeedsWait().
    for (int c = 0; c < CK_Count; ++c) {
        rawNeedsWait[c] = defaultCounterPolicy(static_cast<CounterKind>(c)).rawNeedsWait;
    }
}

void WaitDataflow::setRawNeedsWait(CounterKind c, RawWaitPredicate pred) {
    rawNeedsWait[c] = pred ? std::move(pred) : defaultCounterPolicy(c).rawNeedsWait;
}

DataflowState WaitDataflow::mergeFromPredecessors(BasicBlock& bb) const {
    return mergeFromPredecessors(bb, result.exitState);
}

DataflowState WaitDataflow::mergeFromPredecessors(
    BasicBlock& bb, const std::unordered_map<const BasicBlock*, DataflowState>& exitState) const {
    DataflowState entry;
    const auto& preds = bb.getPredecessors();

    // Seed one PerPredQueue per pred per counter from each pred's exit
    // queues, retagged so the optimizer can identify the direct pred for
    // tail drains. Self-preds (back-edges) are seeded too: at fixed point
    // the back-edge's exit is the loop body's true exit, which is what
    // the header should see.
    //
    // Also forward each pred's PhiSummary table -- PHI summaries live with
    // their defining block but must reach every downstream consumer. If
    // the same PHI is summarised differently on different paths (transient
    // during fixed-point iteration), keep the strictest (min) wait per
    // counter so the consumer stays safe.
    for (BasicBlock* p : preds) {
        auto it = exitState.find(p);
        if (it == exitState.end()) continue;
        const auto& predState = it->second;
        for (int c = 0; c < CK_Count; ++c) {
            for (const auto& predQ : predState.queues[c]) {
                PerPredQueue q;
                q.pred = p;
                q.ops = predQ.ops;
                // Dedup identical (pred, ops) queues. A back-edge otherwise
                // re-copies the same per-pred queue on every fixed-point
                // iteration: the predecessor's exit already contains the
                // queues it inherited from this block last round, so the
                // queue COUNT grows by one each iteration and the state
                // never stabilises (hitting the iteration cap and forcing
                // the conservative s_wait_* 0 fallback). Identical queues
                // yield identical countFrom() results, so collapsing them
                // is loss-free and restores convergence.
                bool dup = false;
                for (const auto& existing : entry.queues[c]) {
                    if (existing.pred == q.pred && existing.ops == q.ops) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) entry.queues[c].push_back(std::move(q));
            }
        }
        for (const auto& kv : predState.phiSummaries) {
            auto [sit, inserted] = entry.phiSummaries.emplace(kv.first, kv.second);
            if (!inserted) {
                for (int c = 0; c < CK_Count; ++c) {
                    int a = sit->second.waits[c];
                    int b = kv.second.waits[c];
                    if (b < 0) continue;
                    if (a < 0 || b < a) sit->second.waits[c] = b;
                }
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
        // Per counter, the PHI's strictest wait is the smallest (countFrom
        // - 1) across all constrained incoming paths. A consumer that
        // reads the PHI must drain on every path that carries a memop, so
        // the shallowest constrained path defines the bound.
        auto recordWait = [&](CounterKind c, int w) {
            if (w < 0) return;
            if (summary.waits[c] == WaitCountSpec::kUnused || w < summary.waits[c]) {
                summary.waits[c] = w;
            }
        };

        for (size_t j = 0; j < preds.size() && j < srcs.size(); ++j) {
            StinkyInstruction* src = srcs[j];
            if (src == nullptr) continue;
            auto pit = exitState.find(preds[j]);
            if (pit == exitState.end()) continue;
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

// Trim every per-pred queue in a counter to keep at most `keep` tail ops.
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
// (pred == nullptr) so the in-block prefix is still tracked. The queue is
// capped at kMaxInFlight so an undrained counter cannot grow it forever.
// Returns true if the cap had to drop an op -- i.e. the queue exceeded the
// hardware in-flight window -- so the caller can flag the overflow for an
// end-of-solve diagnostic.
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

// Human-readable name for a counter, for diagnostics.
const char* counterName(CounterKind c) {
    switch (c) {
        case CK_DS:
            return "ds (dscnt)";
        case CK_Buffer:
            return "buffer (loadcnt/storecnt)";
        case CK_KM:
            return "scalar (kmcnt)";
        case CK_Tensor:
            return "tensor (tlcnt)";
        default:
            return "?";
    }
}

int getCounterField(const WaitCountSpec& spec, CounterKind c) {
    switch (c) {
        case CK_DS:
            return spec.dsCount;
        case CK_Buffer:
            return spec.bufferCount;
        case CK_KM:
            return spec.kmCount;
        case CK_Tensor:
            return spec.tensorCount;
        default:
            return WaitCountSpec::kUnused;
    }
}

void setCounterField(WaitCountSpec& spec, CounterKind c, int w) {
    switch (c) {
        case CK_DS:
            spec.dsCount = w;
            break;
        case CK_Buffer:
            spec.bufferCount = w;
            break;
        case CK_KM:
            spec.kmCount = w;
            break;
        case CK_Tensor:
            spec.tensorCount = w;
            break;
        default:
            break;
    }
}

// Trim only the per-pred queues tagged with `pred` on counter `c`.
void trimPredQueues(std::vector<PerPredQueue>& qs, BasicBlock* pred, int keep) {
    for (auto& q : qs) {
        if (q.pred != pred) continue;
        if (keep <= 0) {
            q.ops.clear();
        } else if (static_cast<int>(q.ops.size()) > keep) {
            q.ops.erase(q.ops.begin(), q.ops.end() - keep);
        }
    }
}

const WaitCountSpec* findTailDrainSpec(const WaitInsertionPlan& plan, BasicBlock* predBB) {
    for (const TailDrain& td : plan.tailDrains) {
        if (td.predBB == predBB && td.spec.isValid()) return &td.spec;
    }
    return nullptr;
}

DataflowState adjustedEntry(BasicBlock& bb, const WaitInsertionPlan& plan,
                            const DataflowState& rawEntry) {
    DataflowState state = rawEntry;
    for (BasicBlock* pred : bb.getPredecessors()) {
        const WaitCountSpec* td = findTailDrainSpec(plan, pred);
        if (td == nullptr) continue;
        for (int c = 0; c < CK_Count; ++c) {
            int w = getCounterField(*td, static_cast<CounterKind>(c));
            if (w == WaitCountSpec::kUnused) continue;
            trimPredQueues(state.queues[c], pred, w);
        }
    }
    return state;
}

void restoreTensorState(DataflowState& state, const DataflowState& frozen) {
    state.queues[CK_Tensor] = frozen.queues[CK_Tensor];
    for (auto& kv : state.phiSummaries) {
        auto it = frozen.phiSummaries.find(kv.first);
        kv.second.waits[CK_Tensor] = (it == frozen.phiSummaries.end())
                                         ? WaitCountSpec::kUnused
                                         : it->second.waits[CK_Tensor];
    }
}

WaitCountSpec mergePlanAndComputed(const WaitInsertionPlan& plan, StinkyInstruction* inst,
                                   const int computed[CK_Count], CounterEmitState emit[CK_Count]) {
    WaitCountSpec applySpec;
    auto pit = plan.anchorWaits.find(inst);
    const bool inPlan = pit != plan.anchorWaits.end();

    for (int ci = 0; ci < CK_Count; ++ci) {
        CounterKind c = static_cast<CounterKind>(ci);
        int planned = inPlan ? getCounterField(pit->second, c) : WaitCountSpec::kUnused;
        int comp = computed[ci];

        // The optimizer's planned wait is a FLOOR (we must emit at least
        // that strong a drain), but the freshly recomputed requirement may
        // be STRICTER (a smaller count drains deeper). Take the strictest
        // (smallest) of the two so a relaxed planned value can never mask a
        // tighter residual the consumer actually needs -- otherwise the
        // drain slips to a later instruction and the first consumer of a
        // freshly produced operand runs unguarded.
        int w = WaitCountSpec::kUnused;
        if (planned != WaitCountSpec::kUnused) w = planned;
        if (comp != WaitCountSpec::kUnused && (w == WaitCountSpec::kUnused || comp < w)) w = comp;

        if (w == WaitCountSpec::kUnused) continue;
        if (!emit[c].needsNewWait(w)) continue;

        setCounterField(applySpec, c, w);
        emit[c].recordEmittedWait(w);
    }
    return applySpec;
}

int phiCurrentQueueWait(StinkyInstruction* phi, CounterKind c, const DataflowState& state,
                        std::unordered_set<StinkyInstruction*>& seen);

// Compute per-counter required waits for `inst` against the live `state`.
void computeRequiredWaits(StinkyInstruction* inst, DataflowState& state,
                          const std::array<WaitDataflow::RawWaitPredicate, CK_Count>& rawNeedsWait,
                          int required[CK_Count]) {
    // Required wait per counter. -1 = no constraint yet.
    for (int c = 0; c < CK_Count; ++c) required[c] = WaitCountSpec::kUnused;

    // Tighten required[c] = min(required[c], w). The min across deps on
    // the same counter is what's safe: it drains the closest-to-tail
    // dep, which is the most permissive wait that still satisfies it.
    auto tightenRequired = [&](CounterKind c, int w) {
        if (w < 0) return;
        if (required[c] == WaitCountSpec::kUnused || w < required[c]) required[c] = w;
    };

    // For each src dep on counter `c` that appears in some per-pred
    // queue, contribute its (countFrom - 1) wait via tightenRequired.
    // The final required[c] is min over all (dep, pred) hits because
    // the emitted wait must drain on every constrained path.
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
        // No same-pipeline filter here: an SSA RAW edge (e.g. ds_store
        // consuming ds_load's vreg output) needs the wait even though
        // both live on the same hardware FIFO. Same-pipeline only
        // skips ANTI-deps; see scanDsAntiDeps below.

        // Per-counter emit constraint (e.g. tlcnt only drains at a
        // barrier). Overridable via WaitDataflow::setRawNeedsWait();
        // defaults come from defaultCounterPolicy().
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

    // WAR-on-LDS / barrier ordering: the SSA def-use chain captures
    // RAW (consumer's src == producer) but NOT anti-dependencies. An
    // LDS writer must wait for prior LDS readers on the same token,
    // and a barrier must wait for any prior DS op on a matching
    // token. Scan per-pred DS queues for token overlap and treat each
    // hit as an extra DS dep that flows through tightenRequired.
    //
    // Same-pipeline pairs (ds_write writer vs ds_read reader) are
    // skipped: the DS FIFO orders them in hardware.
    //
    // Conservative fallbacks live below: if either side lacks
    // MemTokenData we cannot prove disjointness and force wait 0.
    auto scanDsAntiDeps = [&](const StinkyInstruction& anchor, const std::vector<int>& anchorTokens,
                              bool barrierMode) {
        for (const auto& q : state.queues[CK_DS]) {
            const int qsize = static_cast<int>(q.ops.size());
            for (int idx = 0; idx < qsize; ++idx) {
                StinkyInstruction* op = q.ops[idx];
                if (op == inst) continue;
                // Barrier guards every DS op on a matching token; LDS
                // writer guards only readers/atomics.
                if (!barrierMode && !isDSRead(*op) && !isDSAtomic(*op)) continue;
                if (isOnSamePipeline(anchor, *op)) continue;
                auto* opTokens = op->getModifier<MemTokenData>();
                bool overlap =
                    (opTokens == nullptr) || hasTokenOverlap(opTokens->tokens, anchorTokens);
                if (!overlap) continue;
                tightenRequired(CK_DS, qsize - idx - 1);
            }
        }
    };

    if (isLdsWriterAnchor(*inst)) {
        const auto* tk = inst->getModifier<MemTokenData>();
        if (tk != nullptr) scanDsAntiDeps(*inst, tk->tokens, /*barrierMode=*/false);
    }
    if (isBarrier(*inst)) {
        const auto* tk = inst->getModifier<MemTokenData>();
        if (tk != nullptr) scanDsAntiDeps(*inst, tk->tokens, /*barrierMode=*/true);
    }

    // Tensor-side conservative scan: any tensor_load_to_lds in flight
    // that lacks MemTokenData cannot be proven disjoint from a tensor
    // anchor, so treat it as an extra dep. Tagged overlaps are already
    // covered by the SSA UD chain through LDS<token> pseudo-regs.
    if (isTensorAnchor(*inst) && inst->getModifier<MemTokenData>() != nullptr) {
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

    // Conservative MemTokenData fallbacks. An untagged anchor or
    // untagged producer means we cannot prove disjointness, so we
    // force the matching counter to 0.
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
}

// Tightest wait for counter `c` from a (possibly nested) PHI consumer source.
// Recurses through the PHI inputs to the leaf memops and scans the LIVE per-pred
// queues for each leaf via countFrom() (countFrom == 1 => tail => wait 0). We
// scan the live queue rather than the frozen PhiSummary depth so intervening
// ops are counted, keeping the pipeline full. A leaf that has already drained
// out of the queue contributes no wait. `seen` guards against PHI cycles.
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

}  // namespace

void WaitDataflow::transferBlock(BasicBlock& bb, DataflowState& state) {
    auto& plan = emitPlan[&bb];
    plan.clear();

    CounterEmitState emit[CK_Count];

    for (IRBase& ir : bb) {
        auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst == nullptr) continue;
        if (isPhi(*inst)) continue;  // PhiSummary already computed in merge

        int required[CK_Count];
        computeRequiredWaits(inst, state, rawNeedsWait, required);

        // Decide what to emit (apply redundancy elision) and trim per-pred
        // queues accordingly.
        WaitCountSpec spec;
        for (int c = 0; c < CK_Count; ++c) {
            if (required[c] == WaitCountSpec::kUnused) continue;
            if (!emit[c].needsNewWait(required[c])) continue;

            switch (c) {
                case CK_DS:
                    spec.dsCount = required[c];
                    break;
                case CK_Buffer:
                    spec.bufferCount = required[c];
                    break;
                case CK_KM:
                    spec.kmCount = required[c];
                    break;
                case CK_Tensor:
                    spec.tensorCount = required[c];
                    break;
                default:
                    break;
            }
            emit[c].recordEmittedWait(required[c]);
            trimQueues(state.queues[c], required[c]);
        }
        if (spec.isValid()) plan.emplace_back(inst, spec);

        // Append self to its counter queue (after the wait, so the wait's
        // snapshot of the queue excludes its own consumer).
        CounterKind self = classifyMemOp(*inst);
        if (self != CK_Count) {
            if (appendToAllPaths(state.queues[self], inst)) {
                // Counter issued past its hardware in-flight window without
                // draining; the oldest provably-complete op was dropped.
                overflowSites.emplace(&bb, self);
            }
            emit[self].recordNewOp();
        }
    }

    // Intentionally do NOT collapse per-pred queues at exit: a single
    // union queue would lose per-pred position info and force downstream
    // consumers to compute strictly conservative (over-deep) waits. Each
    // successor's mergeFromPredecessors copies all queues across,
    // retagging them as via-this-pred. The per-pred queue length cap
    // (kMaxInFlight, applied in appendToAllPaths) is what bounds the
    // lattice in place of the old exit-collapse, so a self-loop with an
    // undrained counter still converges.
}

void WaitDataflow::reportCounterOverflow() const {
    // One line per (block, counter) that overflowed the in-flight window in
    // the converged transfer pass, emitted in deterministic RPO order.
    // Non-fatal: the dropped ops are provably complete, so the plan stays
    // correct, but a counter pinned at the full window is a strong hint of a
    // missing drain (real RAW hazard) or dead async loads.
    for (BasicBlock* bb : rpo) {
        for (int c = 0; c < CK_Count; ++c) {
            if (overflowSites.find({bb, static_cast<CounterKind>(c)}) == overflowSites.end())
                continue;
            std::cerr
                << "[WaitDataflow] warning: block '" << bb->getLabel() << "' overflowed the "
                << counterName(static_cast<CounterKind>(c))
                << " in-flight window (kMaxInFlight=" << kMaxInFlight
                << "): the counter was issued past its hardware window without draining, so "
                   "the oldest provably-complete op(s) were dropped. Confirm a drain (barrier "
                   "/ wait) is not required here.\n";
        }
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
        // Cleared each sweep so that, at the fixed point, overflowSites holds
        // exactly the steady-state overflows (the converged sweep re-runs
        // every block's transfer and re-detects any sustained overflow).
        overflowSites.clear();
        for (BasicBlock* bb : rpo) {
            DataflowState entry = mergeFromPredecessors(*bb);
            if (!loopCarriedTokenDepsEnabled && iter > 0) {
                restoreTensorState(entry, result.entryState[bb]);
            }
            DataflowState working = entry;
            transferBlock(*bb, working);
            if (!loopCarriedTokenDepsEnabled && iter > 0) {
                restoreTensorState(working, result.exitState[bb]);
            }

            PASS_DEBUG({
                for (int c = 0; c < CK_Count; ++c) {
                    if (working.queues[c].empty()) continue;
                    size_t tot = 0;
                    size_t maxLen = 0;
                    for (const auto& q : working.queues[c]) {
                        tot += q.ops.size();
                        maxLen = std::max(maxLen, q.ops.size());
                    }
                    std::cerr << "[WaitDataflow] iter=" << iter << " bb=" << bb << " counter=" << c
                              << " nQueues=" << working.queues[c].size() << " totOps=" << tot
                              << " maxLen=" << maxLen << "\n";
                }
            });

            if (!(result.exitState[bb] == working)) {
                result.exitState[bb] = std::move(working);
                changed = true;
            }
            result.entryState[bb] = std::move(entry);
        }
        if (!changed) {
            // Fixed point reached: surface any counter that overflowed its
            // hardware in-flight window (issued past the cap without draining).
            reportCounterOverflow();
            PASS_DEBUG({ std::cerr << "[WaitDataflow] converged in " << iter << " iterations\n"; });
            return true;
        }
    }

    capHit = true;
    std::cerr << "[WaitDataflow] iteration cap " << iterationCap
              << " hit; falling back to s_wait_* 0 at every anchor.\n";
    return false;
}

void WaitDataflow::finalizePlan(WaitInsertionPlan& plan) const {
    // The conservative solver trims queues with PRE-optimizer wait values.
    // A WaitPlanOptimizer (e.g. ShallowPredPromotion) may then RELAX an
    // anchor's wait -- which leaves additional same-counter ops in flight
    // past that anchor than the conservative drain did. Those extra ops
    // must reach the successors so a downstream consumer can re-derive the
    // wait it now needs.
    //
    // Replaying each block in isolation from the solver's entry state
    // cannot see this: the solver's residual was computed with the
    // (deeper) conservative waits, so a relaxed dominating anchor would
    // silently drop the wait a dominated consumer requires (e.g. a tensor
    // load whose only producer is a conditional pred, consumed by a loop
    // header barrier). Instead, re-run the forward dataflow to a fixed
    // point using the FINAL plan's waits to drive the per-counter trims,
    // so the post-optimizer residual propagates across the CFG.
    //
    // The optimizer's anchor waits are the floor we must still emit;
    // finalize only ADDS waits that the relaxed residual now requires and
    // DROPS waits made redundant. Snapshot them before we rebuild.
    const WaitInsertionPlan optimizerPlan = plan;

    std::unordered_map<const BasicBlock*, DataflowState> finalEntry;
    std::unordered_map<const BasicBlock*, DataflowState> finalExit;
    std::unordered_map<StinkyInstruction*, WaitCountSpec> newAnchors;

    for (unsigned iter = 0; iter < iterationCap; ++iter) {
        bool changed = false;
        newAnchors.clear();

        for (BasicBlock* bb : rpo) {
            // Entry = merge of recomputed predecessor exits (back-edges
            // start at bottom and tighten over iterations), then apply the
            // optimizer's predecessor tail drains.
            DataflowState state = mergeFromPredecessors(*bb, finalExit);
            state = adjustedEntry(*bb, optimizerPlan, state);
            if (!loopCarriedTokenDepsEnabled && iter > 0) {
                auto eit = finalEntry.find(bb);
                if (eit != finalEntry.end()) restoreTensorState(state, eit->second);
            }
            finalEntry[bb] = state;
            CounterEmitState emit[CK_Count];

            for (IRBase& ir : *bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (inst == nullptr) continue;
                if (isPhi(*inst)) continue;

                int computed[CK_Count];
                computeRequiredWaits(inst, state, rawNeedsWait, computed);

                // Emit the optimizer's planned wait where present (floor),
                // else the freshly recomputed requirement.
                WaitCountSpec applySpec = mergePlanAndComputed(optimizerPlan, inst, computed, emit);

                for (int c = 0; c < CK_Count; ++c) {
                    int w = getCounterField(applySpec, static_cast<CounterKind>(c));
                    if (w == WaitCountSpec::kUnused) continue;
                    trimQueues(state.queues[c], w);
                }

                if (applySpec.isValid()) newAnchors[inst] = applySpec;

                CounterKind self = classifyMemOp(*inst);
                if (self != CK_Count) {
                    appendToAllPaths(state.queues[self], inst);
                    emit[self].recordNewOp();
                }
            }

            auto it = finalExit.find(bb);
            if (!loopCarriedTokenDepsEnabled && iter > 0 && it != finalExit.end()) {
                restoreTensorState(state, it->second);
            }
            if (it == finalExit.end() || !(it->second == state)) {
                finalExit[bb] = std::move(state);
                changed = true;
            }
        }

        if (!changed) break;
    }

    plan.anchorWaits = std::move(newAnchors);
}

WaitInsertionPlan WaitDataflow::materializePlan() const {
    WaitInsertionPlan plan;

    if (capHit) {
        for (const auto& kv : emitPlan) {
            for (const auto& entry : kv.second) {
                WaitCountSpec spec;
                if (entry.second.dsCount != WaitCountSpec::kUnused) spec.dsCount = 0;
                if (entry.second.bufferCount != WaitCountSpec::kUnused) spec.bufferCount = 0;
                if (entry.second.kmCount != WaitCountSpec::kUnused) spec.kmCount = 0;
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
