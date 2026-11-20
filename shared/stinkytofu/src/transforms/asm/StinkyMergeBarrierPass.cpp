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

#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/LoopAnalysis.hpp"
#include "stinkytofu/analysis/asm/Layer2BarrierOverlapAnalysis.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/LoopDetection.hpp"

#define DEBUG_TYPE "StinkyMergeBarrierPass"

// StinkyMergeBarrierPass
// ======================
// Runs immediately after StinkyDAGSchedulerPass. Within loop bodies it looks
// for two barrier groups that Layer 2 identified as a directional overlapping
// pair and the scheduler placed with no matrix instruction between them. It
// moves only G2's s_barrier_signal to immediately after G1's s_barrier_wait.
// G2's s_barrier_wait stays where it is. Neither group's memory tokens change.
//
// Both workgroup groups use barrier id -1, and only one signal of that id may
// be in flight. Placing G2's signal after G1's wait closes that id first.
// A signal that is already the next instruction after G1's wait is left alone.
//
// A legal "barrier group" is exactly one adjacent s_barrier_signal /
// s_barrier_wait pair that shares the same non-empty LDS token set, e.g.:
//   s_barrier_signal -1   // token 0
//   s_barrier_wait   -1   // token 0
// Anything else (lone signal/wait, signal/.../wait with filler between, mismatched
// tokens) is ignored. Two consecutive legal groups G1 (tokens T1) and G2 (tokens
// T2) are candidates only when Layer2BarrierOverlapAnalysis contains G1 -> G2,
// T1 ≠ T2, and no matrix instruction is issued between them. Issuing a WMMA
// occupies the WMMA queue long enough to hide the later barrier, so cycle
// distance is not a gate.
//
// Ordinary instructions are not moved. A producer or consumer of T2 between
// the groups would be crossed by the moved signal, and another barrier
// between them would too, so those pairs are left untouched.

namespace {
using namespace stinkytofu;

// An adjacent s_barrier_signal / s_barrier_wait pair with one shared token set.
struct BarrierGroup {
    std::vector<StinkyInstruction*> barriers;  // [signal, wait]
    std::unordered_set<uint32_t> tokens;
    IRList::iterator firstIt;  // signal
    IRList::iterator lastIt;   // wait
};

using Layer2OverlapResult = Layer2BarrierOverlapAnalysis::Result;

// Layer 2 records a directional after->before relation. The candidate must
// preserve that direction in final program order.
bool isLayer2OverlapPair(const BarrierGroup& earlier, const BarrierGroup& later,
                         const Layer2OverlapResult& overlaps) {
    for (StinkyInstruction* barrierAfter : earlier.barriers)
        for (StinkyInstruction* barrierBefore : later.barriers)
            if (overlaps.contains(barrierAfter, barrierBefore)) return true;
    return false;
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

// Move g2's signal to just after g1's wait. Tokens and g2's wait stay put.
// Returns true when the IR changed.
bool tryMergePair(BasicBlock& bb, const BarrierGroup& g1, const BarrierGroup& g2,
                  Layer2OverlapResult& overlaps) {
    if (!isLayer2OverlapPair(g1, g2, overlaps)) return false;

    // Only distinct token sets are candidates. Same-token consecutive legal
    // groups are successive syncs of one token and must both remain.
    if (g1.tokens == g2.tokens) return false;

    StinkyInstruction* laterSignal = g2.barriers.front();
    if (laterSignal == nullptr || !isBarrierSignal(*laterSignal)) return false;
    // Already as early as the shared barrier id allows.
    if (std::next(g1.lastIt).getNodePtr() == laterSignal) return false;

    // The signal moves earlier across whatever sits between the groups. An
    // instruction there may touch G1's tokens. It must not touch G2's tokens,
    // and another barrier must not sit between them. A matrix instruction
    // between them also blocks the hoist: one WMMA, including scaled WMMA,
    // fills the WMMA queue long enough to hide G2.
    for (auto it = std::next(g1.lastIt); it != g2.firstIt; ++it) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst == nullptr) continue;
        if (isBarrier(*inst)) return false;
        if (isMatrixInstruction(*inst)) return false;
        if (touchesTokens(*inst, g2.tokens)) return false;
    }

    bb.removeIR(laterSignal);
    bb.insertIR(std::next(g1.lastIt), laterSignal);
    return true;
}

// Hoist every eligible later signal. A hoisted signal is no longer adjacent to
// its wait, so that group is not a candidate again.
void mergeBarriersInBlock(BasicBlock& bb, Layer2OverlapResult& overlaps) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<BarrierGroup> groups = collectBarrierGroups(bb);
        for (size_t i = 0; i + 1 < groups.size(); ++i) {
            if (tryMergePair(bb, groups[i], groups[i + 1], overlaps)) {
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
            mergeBarriersInBlock(bb, layer2Overlaps);
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
