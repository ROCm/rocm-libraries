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
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/LoopDetection.hpp"

#define DEBUG_TYPE "StinkyMergeBarrierPass"

// StinkyMergeBarrierPass
// ======================
// Runs immediately after StinkyDAGSchedulerPass. Within loop bodies it looks
// for two (or more) barrier groups that the scheduler placed only a few cycles
// apart and fuses them into a single group carrying the union of their memory
// tokens, dropping the redundant second signal/wait pair.
//
// A "barrier group" is a maximal run of consecutive barrier instructions
// (typically an s_barrier_signal / s_barrier_wait pair) that all share the same
// LDS memory-token set. Two consecutive groups G1 (tokens T1) and G2 (tokens T2)
// are merged when the modeled cycle-distance (sum of issueCycles of the
// instructions strictly between them) is below the configured threshold.
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

// A maximal run of consecutive barrier instructions that share one token set.
struct BarrierGroup {
    std::vector<StinkyInstruction*> barriers;
    std::unordered_set<uint32_t> tokens;
    IRList::iterator firstIt;  // iterator of the first barrier in the run
    IRList::iterator lastIt;   // iterator of the last barrier in the run
};

// LDS memory-token ids attached to a barrier (from the pseudo LDS registers
// planted by StinkyBuildImplicitDependencyPass; barriers carry them on both
// src and dest, so scanning dest is sufficient).
std::unordered_set<uint32_t> barrierTokenSet(const StinkyInstruction& inst) {
    std::unordered_set<uint32_t> tokens;
    for (const StinkyRegister& r : inst.getDestRegs())
        if (isPseudoReg(r) && r.reg.type == RegType::LDS) tokens.insert(r.reg.idx);
    return tokens;
}

// Collect the barrier groups of a block in program order.
std::vector<BarrierGroup> collectBarrierGroups(BasicBlock& bb) {
    std::vector<BarrierGroup> groups;
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr || !isBarrier(*inst)) continue;

        std::unordered_set<uint32_t> tokens = barrierTokenSet(*inst);
        if (tokens.empty()) continue;  // only token-carrying barriers participate

        if (!groups.empty() && groups.back().tokens == tokens &&
            std::next(groups.back().lastIt) == it) {
            groups.back().barriers.push_back(inst);
            groups.back().lastIt = it;
        } else {
            groups.push_back({{inst}, std::move(tokens), it, it});
        }
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
        cycles += inst->issueCycles;
    }
    return cycles;
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

// Attempt to merge the two consecutive groups g1 (earlier) and g2 (later) inside
// \p bb. Returns true on success (IR mutated). \p threshold is in cycles.
bool tryMergePair(BasicBlock& bb, const BarrierGroup& g1, const BarrierGroup& g2, int threshold) {
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
    for (StinkyInstruction* barrier : g1.barriers) addTokensToBarrier(barrier, g2.tokens);
    for (StinkyInstruction* barrier : g2.barriers) bb.removeIR(barrier);

    return true;
}

// Repeatedly merge mergeable adjacent barrier-group pairs in \p bb until a fixed
// point. Chained close groups collapse into one multi-token group.
void mergeBarriersInBlock(BasicBlock& bb, int threshold) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<BarrierGroup> groups = collectBarrierGroups(bb);
        for (size_t i = 0; i + 1 < groups.size(); ++i) {
            if (tryMergePair(bb, groups[i], groups[i + 1], threshold)) {
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
            mergeBarriersInBlock(bb, threshold);
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
