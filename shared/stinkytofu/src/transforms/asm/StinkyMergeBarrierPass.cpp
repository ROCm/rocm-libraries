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
#include "stinkytofu/transforms/asm/StinkyMergeBarrierPass.hpp"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/analysis/asm/Layer2BarrierOverlapAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/LoopDetection.hpp"

#define DEBUG_TYPE "StinkyMergeBarrierPass"

// StinkyMergeBarrierPass
// ======================
// Runs immediately after StinkyDAGSchedulerPass. Within loop bodies it looks
// for two (or more) barrier groups that Layer 2 identified as a directional
// overlapping pair and the scheduler placed only a few cycles apart. It fuses
// them into a single group carrying the union of their memory tokens, dropping
// the redundant second signal/wait pair.
//
// A legal "barrier group" is exactly one adjacent s_barrier_signal /
// s_barrier_wait pair that shares the same non-empty LDS token set, e.g.:
//   s_barrier_signal -1   // token 0
//   s_barrier_wait   -1   // token 0
// Anything else (lone signal/wait, signal/.../wait with filler between, mismatched
// tokens) is ignored. Two consecutive legal groups G1 (tokens T1) and G2 (tokens
// T2) may merge only when Layer2BarrierOverlapAnalysis contains G1 -> G2,
// T1 ≠ T2, and the modeled cycle-distance is below the configured threshold.
//
// The merge never moves instructions: it only unions G2's tokens into G1's
// barriers and drops G2's redundant signal/wait pair. That is correct precisely
// when nothing between the two groups is ordered against the merged barrier — no
// other barrier and no producer/consumer of a merged token (T1 ∪ T2) sits in
// between. When such an instruction exists the merge is skipped, because keeping
// the barrier in place would move it to the wrong side of that dependency.

namespace {
using namespace stinkytofu;

// CDNA5 (Gfx1250) default merge distance, in cycles. Used when
// dagFeatures.mergeBarrierThreshold holds the sentinel value (0). Explicit
// non-sentinel config wins. Mirrors the kCdna5* tunables in dag/CDNA5.hpp.
constexpr int kCdna5MergeBarrierThreshold = 11;

// An adjacent s_barrier_signal / s_barrier_wait pair with one shared token set.
struct BarrierGroup {
    std::vector<StinkyInstruction*> barriers;  // [signal, wait]
    std::unordered_set<uint32_t> tokens;
    IRList::iterator firstIt;  // signal
    IRList::iterator lastIt;   // wait
};

using Layer2OverlapResult = Layer2BarrierOverlapAnalysis::Result;

bool containsBarrier(const BarrierGroup& group, const StinkyInstruction* barrier) {
    return std::find(group.barriers.begin(), group.barriers.end(), barrier) != group.barriers.end();
}

// Layer 2 records a directional after->before relation. The merge candidate
// must preserve that direction in final program order.
bool isLayer2OverlapPair(const BarrierGroup& earlier, const BarrierGroup& later,
                         const Layer2OverlapResult& overlaps) {
    for (StinkyInstruction* barrierAfter : earlier.barriers)
        for (StinkyInstruction* barrierBefore : later.barriers)
            if (overlaps.contains(barrierAfter, barrierBefore)) return true;
    return false;
}

// A successful merge makes \p surviving and \p removed one logical barrier group.
// Redirect the removed group's scheduler-approved incoming/outgoing relations to
// that logical survivor. This intentionally derives only the transitive chain
// eligibility needed after an approved merge; it does not authorize an unrelated
// pair that was absent from the scheduler's validated relation graph.
void transferLayer2OverlapPairs(Layer2OverlapResult& overlaps, const BarrierGroup& surviving,
                                const BarrierGroup& removed) {
    std::vector<Layer2BarrierOverlapAnalysis::BarrierPair> additions;
    for (const auto& [barrierAfter, barrierBefore] : overlaps.pairs()) {
        const bool afterRemoved = containsBarrier(removed, barrierAfter);
        const bool beforeRemoved = containsBarrier(removed, barrierBefore);
        if (afterRemoved && !beforeRemoved)
            for (StinkyInstruction* survivor : surviving.barriers)
                additions.emplace_back(survivor, barrierBefore);
        if (beforeRemoved && !afterRemoved)
            for (StinkyInstruction* survivor : surviving.barriers)
                additions.emplace_back(barrierAfter, survivor);
    }
    for (const auto& [barrierAfter, barrierBefore] : additions)
        overlaps.record(barrierAfter, barrierBefore);
}

// LDS memory-token ids attached to a barrier (from the pseudo LDS registers
// planted by StinkyBuildImplicitDependencyPass; barriers carry them on both
// src and dest, so scanning dest is sufficient).
std::unordered_set<uint32_t> barrierTokenSet(const StinkyInstruction& inst) {
    std::unordered_set<uint32_t> tokens;
    for (const StinkyRegister& r : inst.getDestRegs())
        if (isPseudoReg(r) && r.reg.type == RegType::LDS) tokens.insert(r.reg.idx);
    return tokens;
}

// True if \p inst is ordered against a barrier guarding any token in \p tokens,
// i.e. it produces or consumes one of those LDS tokens (an LDS pseudo operand on
// src or dest whose index is in the set). Such an instruction has a fixed
// side relative to the barrier and must not be jumped over by a merge.
bool touchesTokens(const StinkyInstruction& inst, const std::unordered_set<uint32_t>& tokens) {
    for (const StinkyRegister& r : inst.getSrcRegs())
        if (isPseudoReg(r) && r.reg.type == RegType::LDS && tokens.count(r.reg.idx)) return true;
    for (const StinkyRegister& r : inst.getDestRegs())
        if (isPseudoReg(r) && r.reg.type == RegType::LDS && tokens.count(r.reg.idx)) return true;
    return false;
}

// Collect legal barrier groups in program order: only adjacent
// s_barrier_signal + s_barrier_wait with identical non-empty token sets.
std::vector<BarrierGroup> collectBarrierGroups(BasicBlock& bb) {
    std::vector<BarrierGroup> groups;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* signal = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (signal == nullptr || !isBarrierSignal(*signal)) continue;

        std::unordered_set<uint32_t> tokens = barrierTokenSet(*signal);
        if (tokens.empty()) continue;

        auto waitIt = std::next(it);
        if (waitIt == bb.end()) continue;
        auto* wait = dyn_cast<StinkyInstruction>(waitIt.getNodePtr());
        if (wait == nullptr || !isBarrierWait(*wait)) continue;
        if (barrierTokenSet(*wait) != tokens) continue;

        groups.push_back({{signal, wait}, std::move(tokens), it, waitIt});
        it = waitIt;  // for-loop ++it advances past the wait
    }
    return groups;
}

// Sum of issueCycles over the instructions strictly between \p afterIt and
// \p beforeIt (both exclusive). Non-StinkyInstruction IR contributes nothing.
int cycleDistance(IRList::iterator afterIt, IRList::iterator beforeIt) {
    int cycles = 0;
    for (auto it = std::next(afterIt); it != beforeIt; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isMatrixInstruction(*inst))
            cycles += inst->latencyCycles;
        else
            cycles += inst->issueCycles;
    }
    return cycles;
}

// Add \p tokenId as an LDS pseudo register to \p regs if not already present.
void addUniqueLds(std::vector<StinkyRegister>& regs, uint32_t tokenId, bool /*isDest*/) {
    for (const StinkyRegister& r : regs)
        if (r.isRegister() && r.reg.type == RegType::LDS && r.reg.idx == tokenId) return;
    regs.push_back(StinkyRegister(RegType::LDS, tokenId, 1));
}

// Fold \p extraTokens into a barrier's MemTokenData modifier and LDS pseudo
// src/dest registers (deduplicated), so the merged barrier guards both groups.
void addTokensToBarrier(StinkyInstruction* barrier, const std::unordered_set<uint32_t>& extra) {
    // MemTokenData modifier.
    if (auto* mt = barrier->getModifier<MemTokenData>()) {
        for (uint32_t t : extra) {
            if (std::find(mt->tokens.begin(), mt->tokens.end(), static_cast<int>(t)) ==
                mt->tokens.end())
                mt->tokens.push_back(static_cast<int>(t));
        }
    }

    // LDS pseudo registers (barriers carry each token on both src and dest).
    std::vector<StinkyRegister> srcs = barrier->getSrcRegs();
    std::vector<StinkyRegister> dsts = barrier->getDestRegs();
    for (uint32_t t : extra) {
        addUniqueLds(srcs, t, /*isDest=*/false);
        addUniqueLds(dsts, t, /*isDest=*/true);
    }
    barrier->setSrcRegs(srcs);
    barrier->setDestRegs(dsts);
}

// Overwrite barrier comment with merged-token summary:
// "merged barrier, sync LDS0, sync LDS1, ..."
void setMergedBarrierComment(StinkyInstruction* barrier,
                             const std::unordered_set<uint32_t>& mergedTokens) {
    std::vector<uint32_t> sortedTokens(mergedTokens.begin(), mergedTokens.end());
    std::sort(sortedTokens.begin(), sortedTokens.end());

    std::string comment = "merged barrier";
    for (uint32_t token : sortedTokens) {
        comment += ", sync LDS";
        comment += std::to_string(token);
    }

    if (auto* c = barrier->getModifier<CommentData>()) {
        c->comment = comment;
    } else {
        barrier->addModifier<CommentData>(CommentData{comment});
    }
}

// Attempt to merge the two consecutive groups g1 (earlier) and g2 (later) inside
// \p bb. Returns true on success (IR mutated). \p threshold is in cycles.
bool tryMergePair(BasicBlock& bb, const BarrierGroup& g1, const BarrierGroup& g2, int threshold,
                  Layer2OverlapResult& overlaps) {
    if (!isLayer2OverlapPair(g1, g2, overlaps)) return false;

    // Only distinct token sets are merge candidates. Same-token consecutive
    // legal groups are successive syncs of one token and must both remain.
    if (g1.tokens == g2.tokens) return false;

    const int dist = cycleDistance(g1.lastIt, g2.firstIt);
    if (dist >= threshold) return false;

    // Merged barrier will guard the union of both token sets.
    std::unordered_set<uint32_t> mergedTokens = g1.tokens;
    mergedTokens.insert(g2.tokens.begin(), g2.tokens.end());

    // Merge without moving instructions. This is only correct when nothing
    // strictly between the two groups is ordered against the merged barrier:
    // no other barrier, and no producer/consumer of a merged token. Otherwise
    // folding G2 back onto G1's position would cross that dependency — bail.
    for (auto it = std::next(g1.lastIt); it != g2.firstIt; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isBarrier(*inst)) return false;
        if (touchesTokens(*inst, mergedTokens)) return false;
    }

    // Safe: fold g2's tokens into every barrier of g1, then drop g2's redundant
    // signal/wait pair. No instruction is moved.
    for (StinkyInstruction* barrier : g1.barriers) {
        addTokensToBarrier(barrier, g2.tokens);
        setMergedBarrierComment(barrier, mergedTokens);
    }
    transferLayer2OverlapPairs(overlaps, g1, g2);
    for (StinkyInstruction* barrier : g2.barriers) bb.removeIR(barrier);

    return true;
}

// Repeatedly merge mergeable adjacent barrier-group pairs in \p bb until a fixed
// point. Chained close groups collapse into one multi-token group.
void mergeBarriersInBlock(BasicBlock& bb, int threshold, Layer2OverlapResult& overlaps) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<BarrierGroup> groups = collectBarrierGroups(bb);
        for (size_t i = 0; i + 1 < groups.size(); ++i) {
            if (tryMergePair(bb, groups[i], groups[i + 1], threshold, overlaps)) {
                changed = true;
                break;  // group layout changed; rebuild before continuing
            }
        }
    }
}

class StinkyMergeBarrierPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "StinkyMergeBarrierPass";
    }

    PassID getPassID() const override {
        return &StinkyMergeBarrierPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        // CDNA5 (Gfx1250) barrier-token scheduling only. Match chooseReadyQueue().
        if (passCtx.getGemmTileConfig().arch[0] != 12 || passCtx.getGemmTileConfig().arch[1] != 5)
            return preserveCFGAnalyses();

        const int cfg = passCtx.getPassFeatureConfig().dagFeatures.mergeBarrierThreshold;
        const int threshold = cfg > 0 ? cfg : kCdna5MergeBarrierThreshold;

        if (threshold == -1) {
            return preserveCFGAnalyses();
        }

        // Copy because chained merges redirect relations away from removed barriers.
        Layer2OverlapResult layer2Overlaps = AM.getResult<Layer2BarrierOverlapAnalysis>(func);

        // Only touch loop-body basic blocks — the request targets the loop
        // interior, where the scheduler emits the repeated barrier groups.
        const auto& loops = AM.getResult<LoopAnalysis>(func);
        std::unordered_set<BasicBlock*> loopBodyBBs;
        for (const Loop& loop : loops)
            for (BasicBlock* bb : loop.bodyBBs) loopBodyBBs.insert(bb);

        // Walk blocks in program order (deterministic) and process the ones
        // that belong to a loop body.
        for (BasicBlock& bb : func) {
            if (!loopBodyBBs.count(&bb)) continue;
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            mergeBarriersInBlock(bb, threshold, layer2Overlaps);
        }
        return preserveCFGAnalyses();
    }
};

char StinkyMergeBarrierPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyMergeBarrierPass() {
    return std::make_unique<StinkyMergeBarrierPass>();
}
}  // namespace stinkytofu
