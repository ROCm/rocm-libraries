// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/TDMLoadWaveSyncPass.hpp"

#include <algorithm>
#include <deque>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "TDMLoadWaveSyncPass"

namespace {
using namespace stinkytofu;

/// Tokens of a tensor_load's MemTokenData ({} if none). Distinct token vectors are
/// distinct TDM wait groups.
std::vector<int> tdmTokens(const StinkyInstruction& inst) {
    const auto* mt = inst.getModifier<MemTokenData>();
    return mt ? mt->tokens : std::vector<int>{};
}

/// Sorted-unique union of two token vectors.
std::vector<int> unionTokens(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out(a);
    out.insert(out.end(), b.begin(), b.end());
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

/// True if any token of `toks` is in `s`.
bool anyTokenIn(const std::vector<int>& toks, const std::vector<int>& s) {
    for (int t : toks)
        if (std::find(s.begin(), s.end(), t) != s.end()) return true;
    return false;
}

/// Barrier comment naming the wait's drained set, e.g. "TDM wait-group barrier [0,2]".
std::string barrierTag(const StinkyInstruction* waitInst) {
    std::string tag = "TDM wait-group barrier";
    if (waitInst != nullptr) {
        if (const auto* mt = waitInst->getModifier<MemTokenData>()) {
            tag += " [";
            for (size_t i = 0; i < mt->tokens.size(); ++i) {
                if (i > 0) tag += ",";
                tag += std::to_string(mt->tokens[i]);
            }
            tag += "]";
        }
    }
    return tag;
}

/// Append `tag` to a pre-existing barrier's comment (idempotent) so the emitted asm
/// shows it already covers a split this pass would otherwise insert.
void tagExistingWaveGroupBarrier(StinkyInstruction* barrier, const std::string& tag) {
    if (auto* c = barrier->getModifier<CommentData>()) {
        if (c->comment.find("TDM wait-group barrier") != std::string::npos) return;
        c->comment += "; ";
        c->comment += tag;
    } else {
        barrier->addModifier<CommentData>(CommentData{tag});
    }
}

/// One open deferrable group on the backward flow. The anchor (map key in
/// `Frontier`) is the earliest deferrable load of the group and the barrier site;
/// this holds the rest of the group's state.
struct DeferEntry {
    std::vector<int> deferSeen;  // union of the group's deferrable tokens
    bool barrierInGap = false;   // a barrier already sits between the group and the boundary
    StinkyInstruction* gapBarrier = nullptr;  // wait-half of that barrier, for tagging
};

/// The backward-dataflow state: the set of still-open deferrable groups, keyed by
/// barrier-anchor position. Multiple entries arise when sibling CFG paths reach a
/// block with distinct anchors (each may need its own barrier); a deferrable load
/// common to those paths collapses them to one.
using Frontier = std::map<StinkyInstruction*, DeferEntry>;

/// Merge `b` into `a` (union of groups). Same anchor -> union tokens; suppress the
/// barrier only if EVERY contributing path already has one (AND of barrierInGap).
/// Returns true if `a` grew, so the worklist knows to re-process.
bool mergeFrontier(Frontier& a, const Frontier& b) {
    bool changed = false;
    for (const auto& [anchor, e] : b) {
        auto it = a.find(anchor);
        if (it == a.end()) {
            a.emplace(anchor, e);
            changed = true;
            continue;
        }
        std::vector<int> u = unionTokens(it->second.deferSeen, e.deferSeen);
        if (u != it->second.deferSeen) {
            it->second.deferSeen = std::move(u);
            changed = true;
        }
        const bool nb = it->second.barrierInGap && e.barrierInGap;
        if (nb != it->second.barrierInGap) {
            it->second.barrierInGap = nb;
            changed = true;
        }
        // Keep a witness barrier only while the gap stays synchronized on all paths.
        if (!nb) {
            it->second.gapBarrier = nullptr;
        } else if (it->second.gapBarrier == nullptr) {
            it->second.gapBarrier = e.gapBarrier;
        }
    }
    return changed;
}

/// Per-path backward-scan state. `frontier` is the open deferrable groups. `workingS`
/// is the still-urgent subset of the trigger's drained set S: a load is urgent iff its
/// token is in `workingS`, and crossing a wait removes that wait's drained tokens from
/// `workingS` (those loads are already drained on this path, so they can no longer be
/// co-resident with a deferrable load below). A token in the original S but no longer
/// in `workingS` is neither urgent nor deferrable — it is ignored.
struct ScanState {
    Frontier frontier;
    std::vector<int> workingS;
};

/// Merge `b` into `a`: union frontiers and union `workingS` (a token stays urgent if
/// undrained on ANY incoming path). Returns true if `a` grew (either component), so the
/// worklist re-processes.
bool mergeScanState(ScanState& a, const ScanState& b) {
    bool changed = mergeFrontier(a.frontier, b.frontier);
    std::vector<int> w = unionTokens(a.workingS, b.workingS);
    if (w != a.workingS) {
        a.workingS = std::move(w);
        changed = true;
    }
    return changed;
}

class TDMLoadWaveSyncPass : public StinkyInstPass {
   public:
    static char ID;

    TDMLoadWaveSyncPass() = default;

    const char* getName() const override {
        return "TDMLoadWaveSyncPass";
    }

    PassID getPassID() const override {
        return &TDMLoadWaveSyncPass::ID;
    }

    /// Result of scanning one block body backward.
    struct BodyResult {
        ScanState residual;       // state at the block top (propagate to preds)
        bool terminated = false;  // the flow ended in this block (do not propagate)
    };

    /// Scan instructions [0, endIdx) of `bb` in reverse (backward) from `st`. `S` is the
    /// trigger's full drained set (fixed); `st.workingS` is the still-urgent subset that
    /// shrinks as waits are crossed. Classify each tensor_load by its tokens:
    ///   - token in `st.workingS`         -> URGENT: it is the boundary. Plant a barrier
    ///     before every open group's anchor (unless one already sits in the gap) and end.
    ///   - token not in `S`               -> DEFERRABLE: collapse the open groups into one
    ///     anchored here; its tokens join deferSeen and barrierInGap resets.
    ///   - token in `S` but not workingS  -> already drained on this path; ignore it.
    /// Crossing an s_wait_tensorcnt removes its drained tokens from `workingS`; when
    /// `workingS` empties, nothing further back can be urgent so the flow ends. A barrier
    /// with groups open marks barrierInGap and witnesses its wait half.
    /// Instructions are gathered forward into a vector and walked by index (intrusive-list
    /// end() is not safely decrementable).
    ///
    /// `out` collects (anchor, tokens) for barriers to insert; `tagsOut` collects
    /// (anchor, existing barrier) for pre-existing barriers already at a split. Both
    /// are keyed by anchor so a tag whose anchor also got an insertion can be dropped.
    /// `anchorWait` records the wait that motivated each anchor, for the barrier comment.
    BodyResult scanBody(BasicBlock& bb, size_t endIdx, StinkyInstruction* waitInst,
                        const std::vector<int>& S, ScanState st,
                        std::vector<std::pair<StinkyInstruction*, std::vector<int>>>& out,
                        std::vector<std::pair<StinkyInstruction*, StinkyInstruction*>>& tagsOut,
                        std::unordered_map<StinkyInstruction*, StinkyInstruction*>& anchorWait) {
        Frontier& F = st.frontier;
        std::vector<StinkyInstruction*> insts;
        insts.reserve(endIdx);
        size_t seen = 0;
        for (auto it = bb.begin(); it != bb.end() && seen < endIdx; ++it, ++seen) {
            insts.push_back(dyn_cast<StinkyInstruction>(it.getNodePtr()));
        }
        for (size_t i = insts.size(); i-- > 0;) {
            StinkyInstruction* inst = insts[i];
            if (inst == nullptr) continue;
            if (!isTensorLoad(*inst)) {
                // Crossing a wait drains its tokens: remove them from workingS. Once
                // workingS empties, no urgent load remains further back, so end the flow.
                if (inst->is(InstFlag::IF_WaitTensorCnt)) {
                    const std::vector<int> ws = drainedSetForWait(inst);
                    if (!ws.empty()) {
                        std::vector<int> next;
                        for (int t : st.workingS)
                            if (std::find(ws.begin(), ws.end(), t) == ws.end()) next.push_back(t);
                        st.workingS = std::move(next);
                        if (st.workingS.empty()) return {ScanState{}, true};
                    }
                }
                if (!F.empty() && isBarrier(*inst)) {
                    // Witness the wait half (backward-first); the classic comment is there.
                    StinkyInstruction* waitHalf = isBarrierWait(*inst) ? inst : nullptr;
                    for (auto& [anchor, e] : F) {
                        e.barrierInGap = true;
                        if (waitHalf) e.gapBarrier = waitHalf;
                    }
                }
                continue;
            }
            std::vector<int> toks = tdmTokens(*inst);
            if (anyTokenIn(toks, st.workingS)) {  // URGENT: boundary
                for (auto& [anchor, e] : F) {
                    if (!e.barrierInGap)
                        out.emplace_back(anchor, unionTokens(toks, e.deferSeen));
                    else if (e.gapBarrier != nullptr)
                        tagsOut.emplace_back(anchor, e.gapBarrier);
                    else
                        continue;
                    anchorWait.emplace(anchor, waitInst);  // for the comment's S
                }
                return {ScanState{}, true};
            }
            if (anyTokenIn(toks, S)) continue;  // in S but already drained on this path
            // DEFERRABLE: collapse the open groups into one anchored here.
            std::vector<int> merged = toks;
            for (auto& [anchor, e] : F) merged = unionTokens(merged, e.deferSeen);
            F.clear();
            F.emplace(inst, DeferEntry{std::move(merged), false, nullptr});
        }
        return {std::move(st), false};
    }

    /// The drained (urgent) token set S for a tensorcnt wait.
    ///
    /// Primary source: the wait's OWN MemTokenData, which StinkyWaitCntInsertionPass
    /// attaches — the exact union of the tokens of every tensor_load the wait
    /// drains, computed there from the live use-def / FIFO state. This is the
    /// authoritative drained set and covers multi-group waits correctly.
    ///
    /// A wait with no MemTokenData (e.g. a manually-inserted source-template fence,
    /// often a full-drain s_wait_tensorcnt 0) yields {} — the scan inserts nothing for
    /// it. Such fences have no reliable drained set to recover: guessing from the next
    /// consumer's token would misclassify loads and plant barriers at bogus boundaries.
    static std::vector<int> drainedSetForWait(StinkyInstruction* waitInst) {
        if (const auto* wmt = waitInst->getModifier<MemTokenData>()) return wmt->tokens;
        return {};
    }

    static size_t blockSize(BasicBlock& bb) {
        size_t n = 0;
        for (auto it = bb.begin(); it != bb.end(); ++it) ++n;
        return n;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        // With a single wave there is nothing to separate across waves, so a
        // workgroup barrier is pure overhead. TensileLite also gates the param on
        // NumWaves>1; guard here too so the pass is self-consistent however invoked.
        if (passCtx.getGemmTileConfig().NumWaves <= 1) return preserveCFGAnalyses();

        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        size_t inserted = 0;
        // (anchor load, union tokens) collected across the whole function; the
        // anchor may live in a different block than the triggering wait (the
        // urgent→deferrable transition can be in a CFG predecessor).
        std::vector<std::pair<StinkyInstruction*, std::vector<int>>> pending;
        // (anchor load, existing barrier) for splits already synchronized by a
        // pre-existing barrier — no insertion, just a comment tag. Keyed by anchor so
        // a tag can be dropped if the same anchor also got an insertion.
        std::vector<std::pair<StinkyInstruction*, StinkyInstruction*>> pendingTags;
        // anchor -> the wait that motivated a barrier there, so the emitted comment can
        // show the wait's drained set S (read from the wait's memtoken at emit time).
        // Waits that resolve to one anchor share the same S, so first-wins is fine.
        std::unordered_map<StinkyInstruction*, StinkyInstruction*> anchorWait;

        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            // Each s_wait_tensorcnt is a trigger: its drained token set splits the
            // preceding tensor_loads into a waited-for (urgent) group and a
            // not-yet-waited (deferrable) group. A backward dataflow from the wait
            // finds the urgent/deferrable boundary on every reaching path and plants
            // a barrier there. The frontier merges (unions) at CFG joins and reaches
            // a fixpoint, so a block is scanned a bounded number of times — no path
            // enumeration — and the two groups may straddle a loop back-edge (double
            // prefetch). Mutations are deferred until after every wait is scanned so
            // an inserted barrier can never perturb a later scan.
            size_t idx = 0;
            for (auto waitIt = bb.begin(); waitIt != bb.end(); ++waitIt, ++idx) {
                auto* waitInst = dyn_cast<StinkyInstruction>(waitIt.getNodePtr());
                if (waitInst == nullptr) continue;
                if (!waitInst->is(InstFlag::IF_WaitTensorCnt)) continue;

                // S = the tokens this wait drains. Loads with a token in S are urgent
                // (waited-for); the rest are deferrable.
                std::vector<int> drained = drainedSetForWait(waitInst);
                if (drained.empty()) continue;  // no classifiable urgent set

                // Seed: scan this block above the wait, starting with workingS = S. If
                // the boundary is here the flow terminates; otherwise propagate to preds.
                BodyResult seed = scanBody(bb, idx, waitInst, drained, ScanState{{}, drained},
                                           pending, pendingTags, anchorWait);
                if (seed.terminated) continue;

                // Backward worklist over blocks. bot[B] is the ScanState arriving at
                // B's bottom (merged across successors toward the wait); a block is
                // (re)queued only when that state grows, which bounds the work by the
                // lattice height (monotone union) — the loop back-edge included.
                // Predecessors are scanned without re-checking shouldProcessBasicBlock
                // (only the trigger block is filtered), so a barrier can land in a
                // predecessor the filter would exclude — assumes a kernel-scope filter
                // that accepts all blocks, as the backend registers this pass with.
                std::map<BasicBlock*, ScanState> bot;
                std::deque<BasicBlock*> worklist;
                auto propagate = [&](BasicBlock& from, const ScanState& residual) {
                    for (BasicBlock* pred : from.getPredecessors()) {
                        const bool created = bot.find(pred) == bot.end();
                        const bool grew = mergeScanState(bot[pred], residual);
                        if (created || grew) worklist.push_back(pred);
                    }
                };
                propagate(bb, seed.residual);
                while (!worklist.empty()) {
                    BasicBlock* B = worklist.front();
                    worklist.pop_front();
                    BodyResult r = scanBody(*B, blockSize(*B), waitInst, drained, bot[B], pending,
                                            pendingTags, anchorWait);
                    if (r.terminated) continue;
                    propagate(*B, r.residual);
                }
            }
        }

        // Emit the split pair (s_barrier_signal -1 / s_barrier_wait -1) directly, not
        // a monolithic s_barrier: barrier legalization runs only at conversion time,
        // before this pass, so a bare s_barrier here would reach emission unsplit.
        // Mirrors legalizeBarrier / InsertClusterBarrierPass.
        const HwInstDesc* signalDesc = getMCIDByUOp(GFX::s_barrier_signal, archId);
        const HwInstDesc* waitDesc = getMCIDByUOp(GFX::s_barrier_wait, archId);
        assert(signalDesc && waitDesc &&
               "split-barrier opcodes are not supported on this architecture");

        // Multiple scan paths can reach the same anchor with different urgent tokens;
        // one barrier sits there, so union their token sets (not first-wins) — it must
        // cover every contributing path. Distinct boundaries have distinct anchors.
        std::vector<StinkyInstruction*> anchorOrder;  // deterministic emit order
        std::unordered_map<StinkyInstruction*, std::vector<int>> mergedTokens;
        for (auto& [anchor, toks] : pending) {
            auto it = mergedTokens.find(anchor);
            if (it == mergedTokens.end()) {
                mergedTokens.emplace(anchor, toks);
                anchorOrder.push_back(anchor);
            } else {
                it->second = unionTokens(it->second, toks);
            }
        }

        for (StinkyInstruction* anchor : anchorOrder) {
            const std::vector<int>& toks = mergedTokens[anchor];
            BasicBlock* anchorBB = anchor->getParent();
            AsmIRBuilder irBuilder(*anchorBB, archId);
            StinkyInstruction* signalInst = irBuilder.create(signalDesc, anchor);
            signalInst->addSrcReg(StinkyRegister(-1));  // -1 = workgroup barrier
            signalInst->addModifier<MemTokenData>(MemTokenData{toks});
            StinkyInstruction* waitInst = irBuilder.create(waitDesc, anchor);
            waitInst->addSrcReg(StinkyRegister(-1));  // -1 = workgroup barrier
            waitInst->addModifier<MemTokenData>(MemTokenData{toks});
            auto wi = anchorWait.find(anchor);
            waitInst->addModifier<CommentData>(
                CommentData{barrierTag(wi == anchorWait.end() ? nullptr : wi->second)});
            ++inserted;
        }

        // Tag pre-existing barriers already at a split, so the emitted asm distinguishes
        // them from the barriers we insert (and from plain barriers, which are untagged).
        // Skip any anchor that also got an insertion — the insertion wins. Comment-only;
        // no CFG change.
        std::unordered_map<StinkyInstruction*, bool> tagged;
        for (auto& [anchor, barrier] : pendingTags) {
            if (mergedTokens.count(anchor)) continue;  // that split got a real insertion
            if (tagged.count(barrier)) continue;       // one barrier, tag once
            auto wi = anchorWait.find(anchor);
            tagExistingWaveGroupBarrier(barrier,
                                        barrierTag(wi == anchorWait.end() ? nullptr : wi->second));
            tagged.emplace(barrier, true);
        }

        PASS_DEBUG(std::cerr << "[TDMLoadWaveSyncPass] inserted_barriers=" << inserted
                             << " tagged_barriers=" << tagged.size() << "\n");
        // Only comment tags and no inserted barriers -> the CFG is untouched; keep
        // downstream analyses. (Tagging mutates a comment modifier, not the CFG.)
        if (inserted == 0) return preserveCFGAnalyses();
        return PreservedAnalyses::none();
    }
};

char TDMLoadWaveSyncPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createTDMLoadWaveSyncPass() {
    return std::make_unique<TDMLoadWaveSyncPass>();
}
}  // namespace stinkytofu
