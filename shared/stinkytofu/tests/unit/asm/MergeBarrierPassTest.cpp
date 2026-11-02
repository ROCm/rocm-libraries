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
// Unit tests for StinkyMergeBarrierPass.
//
// The pass runs after StinkyDAGSchedulerPass and fuses Layer 2 overlapping
// barrier groups when no matrix instruction is issued between them. Cycle
// distance is not a gate. These tests build barrier groups directly in a
// self-loop body block and publish the scheduler analysis result.
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/Layer2BarrierOverlapAnalysis.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyMergeBarrierPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

class MergeBarrierPassTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    GemmTileConfig config;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;

    void SetUp() override {
        config.arch[0] = 12;
        config.arch[1] = 5;
        config.arch[2] = 0;
        func = std::make_unique<Function>("merge_barrier_test");
        setFunctionArch(*func, arch);
        // Self-loop so LoopAnalysis marks this block as a loop body.
        bb = func->createBasicBlock("loop_body");
        bb->addSuccessor(bb);
        registerAllAnalyses(am);
    }

    void TearDown() override {
        func.reset();
        bb = nullptr;
    }

    StinkyInstruction* createSignal(int token) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_barrier_signal, arch));
        inst->addSrcReg(StinkyRegister(-1));  // all-wave
        inst->addModifier<MemTokenData>(MemTokenData{std::vector<int>{token}});
        return inst;
    }

    StinkyInstruction* createWait(int token) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_barrier_wait, arch));
        inst->addSrcReg(StinkyRegister(-1));  // all-wave
        inst->addModifier<MemTokenData>(MemTokenData{std::vector<int>{token}});
        return inst;
    }

    // Scaled WMMA used by mxf8. It carries IF_MXWMMA, not only IF_WMMA, and
    // isMatrixInstruction must still see it as a matrix issue.
    StinkyInstruction* createScaleWmma() {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_scale_f32_16x16x128_f8f6f4, arch);
        if (desc == nullptr) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("v", 0, 8));
        inst->addSrcReg(StinkyRegister("v", 16, 8));
        inst->addSrcReg(StinkyRegister("v", 16, 8));
        inst->addSrcReg(StinkyRegister("v", 0, 8));
        return inst;
    }

    std::vector<std::pair<StinkyInstruction*, StinkyInstruction*>> legalBarrierGroups() {
        std::vector<std::pair<StinkyInstruction*, StinkyInstruction*>> groups;
        for (auto it = bb->begin(); it != bb->end(); ++it) {
            auto* signal = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (signal == nullptr || !isBarrierSignal(*signal)) continue;
            auto waitIt = std::next(it);
            if (waitIt == bb->end()) continue;
            auto* wait = dyn_cast<StinkyInstruction>(waitIt.getNodePtr());
            if (wait != nullptr && isBarrierWait(*wait)) groups.emplace_back(signal, wait);
        }
        return groups;
    }

    // Run BuildImplicitDependency (to attach LDS pseudo tokens to barriers) then
    // MergeBarrier. mergeThreshold is accepted so existing call sites stay
    // stable; the pass no longer consults it. Most tests publish the forward
    // Layer 2 relation for each consecutive group; dedicated gate tests can
    // omit or reverse it.
    void runPasses(int mergeThreshold, bool publishOverlap = true, bool reverseOverlap = false) {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        PassFeatureConfig pfc;
        pfc.loopConfig.unrollGemm = true;
        pfc.dagFeatures.mergeBarrierThreshold = mergeThreshold;
        ctx.setPassFeatureConfig(pfc);

        auto implicitDep = createStinkyBuildImplicitDependencyPass();
        implicitDep->run(*func, ctx, am);

        if (publishOverlap) {
            Layer2BarrierOverlapAnalysis::Result overlaps;
            auto groups = legalBarrierGroups();
            for (size_t i = 0; i + 1 < groups.size(); ++i) {
                if (reverseOverlap)
                    overlaps.record(groups[i + 1].first, groups[i].first);
                else
                    overlaps.record(groups[i].first, groups[i + 1].first);
            }
            am.getResult<Layer2BarrierOverlapAnalysis>(*func) = std::move(overlaps);
        }

        auto mergePass = createStinkyMergeBarrierPass();
        mergePass->run(*func, ctx, am);
    }

    int countByMnemonic(const std::string& mnem) const {
        int count = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            const char* m = inst->getHwInstDesc() ? inst->getHwInstDesc()->mnemonic : nullptr;
            if (m && std::string(m) == mnem) count++;
        }
        return count;
    }

    int getPos(const StinkyInstruction* target) const {
        int pos = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (cast<StinkyInstruction>(&ir) == target) return pos;
            ++pos;
        }
        return -1;
    }

    // Sorted unique memory-token ids on a barrier's MemTokenData modifier.
    static std::vector<int> tokensOf(const StinkyInstruction* inst) {
        std::vector<int> t;
        if (const auto* mt = inst->getModifier<MemTokenData>()) t = mt->tokens;
        std::sort(t.begin(), t.end());
        t.erase(std::unique(t.begin(), t.end()), t.end());
        return t;
    }

    // First barrier of the given mnemonic in program order.
    StinkyInstruction* firstBarrier(const std::string& mnem) const {
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            auto* inst = const_cast<StinkyInstruction*>(cast<StinkyInstruction>(&ir));
            const char* m = inst->getHwInstDesc() ? inst->getHwInstDesc()->mnemonic : nullptr;
            if (m && std::string(m) == mnem) return inst;
        }
        return nullptr;
    }
};

// Two directly-adjacent barrier groups merge, collapsing to a single
// signal/wait pair carrying both tokens.
TEST_F(MergeBarrierPassTest, AdjacentGroupsMerge) {
    createSignal(0);
    createWait(0);
    createSignal(1);
    createWait(1);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);

    runPasses(/*mergeThreshold=*/0);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);

    // Surviving barriers carry the union of both token sets.
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0, 1}));
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_wait")), (std::vector<int>{0, 1}));
}

TEST_F(MergeBarrierPassTest, MissingLayer2OverlapPairPreventsMerge) {
    createSignal(0);
    createWait(0);
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/100000, /*publishOverlap=*/false);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);
}

TEST_F(MergeBarrierPassTest, ReverseLayer2OverlapPairPreventsMerge) {
    createSignal(0);
    createWait(0);
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/100000, /*publishOverlap=*/true,
              /*reverseOverlap=*/true);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);
}

// Non-matrix instructions between the groups do not hide a barrier, so the
// groups still merge even when their issue cycles exceed the old threshold.
TEST_F(MergeBarrierPassTest, NonMatrixInstructionsBetweenStillMerge) {
    createSignal(0);
    createWait(0);
    int cycles = 0;
    for (int i = 0; i < 20; ++i) {
        StinkyInstruction* filler =
            createVAddInBlock(bb, arch, /*dest=*/40 + i, /*s0=*/0, /*s1=*/1);
        cycles += filler->issueCycles;
    }
    createSignal(1);
    createWait(1);

    ASSERT_GT(cycles, 11);
    // The old cycle gate would reject this. A tiny threshold must not matter.
    runPasses(/*mergeThreshold=*/1);
    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0, 1}));
}

// One issued WMMA is enough queued latency to hide the later barrier.
TEST_F(MergeBarrierPassTest, MatrixInstructionBetweenPreventsMerge) {
    createSignal(0);
    createWait(0);
    StinkyInstruction* wmma = createScaleWmma();
    ASSERT_NE(wmma, nullptr);
    ASSERT_TRUE(isMatrixInstruction(*wmma));
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/100000);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);
    EXPECT_EQ(countByMnemonic("v_wmma_scale_f32_16x16x128_f8f6f4"), 1);
}

// A producer of the later group's token sits before that barrier. Folding the
// later barrier back would put the producer after it, so the merge is skipped.
TEST_F(MergeBarrierPassTest, LaterTokenProducerBetweenPreventsMerge) {
    createSignal(0);
    createWait(0);
    // Producer of token 1 (writes LDS token 1).
    createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/16, /*memTokens=*/{1});
    createSignal(1);
    createWait(1);

    // Large threshold: distance is not the gate here — the dependency is.
    runPasses(/*mergeThreshold=*/100000);

    // No merge: both barrier groups remain intact and the producer stays put.
    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);
    EXPECT_EQ(countByMnemonic("tensor_load_to_lds"), 1);
}

// The earlier barrier does not move. A producer of its token may stay between
// the groups while the later barrier folds back onto it.
TEST_F(MergeBarrierPassTest, EarlierTokenProducerBetweenStillMerges) {
    createSignal(0);
    createWait(0);
    createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/16, /*memTokens=*/{0});
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/100000);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);
    EXPECT_EQ(countByMnemonic("tensor_load_to_lds"), 1);
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0, 1}));
}

// signal / filler / wait is NOT a legal group (must be adjacent signal+wait),
// so it neither self-merges nor participates in a merge with another token.
TEST_F(MergeBarrierPassTest, SplitSameTokenSignalWaitNotMerged) {
    createSignal(0);
    createVAddInBlock(bb, arch, /*dest=*/40, /*s0=*/0, /*s1=*/1);
    createVAddInBlock(bb, arch, /*dest=*/41, /*s0=*/2, /*s1=*/3);
    createWait(0);

    runPasses(/*mergeThreshold=*/100000);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0}));
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_wait")), (std::vector<int>{0}));
}

// Split (illegal) LDS0 pair must not merge with a later legal LDS1 pair.
TEST_F(MergeBarrierPassTest, SplitPairDoesNotMergeWithOtherToken) {
    createSignal(0);
    createVAddInBlock(bb, arch, /*dest=*/40, /*s0=*/0, /*s1=*/1);
    createWait(0);
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/100000);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0}));
}

// Canonical merge pattern: adjacent signal+wait (T0), filler, adjacent
// signal+wait (T1) with T0 ≠ T1.
TEST_F(MergeBarrierPassTest, LegalDifferentTokenGroupsMerge) {
    createSignal(0);
    createWait(0);
    createVAddInBlock(bb, arch, /*dest=*/40, /*s0=*/0, /*s1=*/1);
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/100000);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0, 1}));
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_wait")), (std::vector<int>{0, 1}));
}

// An instruction between the groups that touches an *unrelated* token (not in
// either group's set) does not constrain the merged barrier, so the merge still
// happens.
TEST_F(MergeBarrierPassTest, UnrelatedTokenBetweenStillMerges) {
    createSignal(0);
    createWait(0);
    // Producer of token 5 — outside the merged set {0, 1}.
    createTensorLoadInBlock(bb, arch, /*s0=*/8, /*s1=*/16, /*memTokens=*/{5});
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/100000);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);
    EXPECT_EQ(countByMnemonic("tensor_load_to_lds"), 1);
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0, 1}));
}

// Three back-to-back groups collapse into one multi-token group (chained merge).
TEST_F(MergeBarrierPassTest, ChainedGroupsCollapse) {
    createSignal(0);
    createWait(0);
    createSignal(1);
    createWait(1);
    createSignal(2);
    createWait(2);

    runPasses(/*mergeThreshold=*/0);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_wait")), (std::vector<int>{0, 1, 2}));
}

// Barrier groups outside any loop body are left untouched — the pass only
// rewrites the loop interior.
TEST_F(MergeBarrierPassTest, NonLoopBlockUntouched) {
    // A fresh block with no back-edge is not part of any detected loop.
    BasicBlock* plain = func->createBasicBlock("plain");
    bb = plain;  // redirect the createSignal/createWait/count helpers to it

    createSignal(0);
    createWait(0);
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/0);

    // Adjacent groups, but not in a loop body → no merge.
    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);
}
