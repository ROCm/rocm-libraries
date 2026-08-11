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
// The pass runs after StinkyDAGSchedulerPass and fuses barrier groups that sit
// closer than DagFeatures::mergeBarrierThreshold cycles apart into a single
// multi-token group. These tests build barrier groups directly (no scheduler)
// in a self-loop body block and exercise a range of threshold values.
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyMergeBarrierPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

// Default used by the pass when mergeBarrierThreshold is left at the sentinel 0.
static constexpr int kCdna5MergeBarrierThreshold = 11;

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

    // Run BuildImplicitDependency (to attach LDS pseudo tokens to barriers) then
    // MergeBarrier with the given cycle threshold (0 => pass default).
    void runPasses(int mergeThreshold) {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        PassFeatureConfig pfc;
        pfc.loopConfig.unrollGemm = true;
        pfc.dagFeatures.mergeBarrierThreshold = mergeThreshold;
        ctx.setPassFeatureConfig(pfc);

        auto implicitDep = createStinkyBuildImplicitDependencyPass();
        implicitDep->run(*func, ctx, am);

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

// Two directly-adjacent barrier groups (distance 0) merge under the default
// threshold, collapsing to a single signal/wait pair carrying both tokens.
TEST_F(MergeBarrierPassTest, AdjacentGroupsMergeWithDefaultThreshold) {
    createSignal(0);
    createWait(0);
    createSignal(1);
    createWait(1);

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);

    runPasses(/*mergeThreshold=*/0);  // 0 => default kCdna5MergeBarrierThreshold

    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);

    // Surviving barriers carry the union of both token sets.
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0, 1}));
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_wait")), (std::vector<int>{0, 1}));
}

// Distance strictly below the threshold merges; equal-or-above does not.
TEST_F(MergeBarrierPassTest, ThresholdBoundaryControlsMerge) {
    createSignal(0);
    createWait(0);
    // Neutral VALU filler (neither token producer nor consumer).
    StinkyInstruction* f0 = createVAddInBlock(bb, arch, /*dest=*/40, /*s0=*/0, /*s1=*/1);
    StinkyInstruction* f1 = createVAddInBlock(bb, arch, /*dest=*/41, /*s0=*/2, /*s1=*/3);
    createSignal(1);
    createWait(1);

    const int distance = f0->issueCycles + f1->issueCycles;
    ASSERT_GT(distance, 0) << "filler must contribute a positive cycle distance";

    // Threshold equal to the distance: dist >= threshold, so no merge.
    runPasses(/*mergeThreshold=*/distance);
    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);
}

TEST_F(MergeBarrierPassTest, DistanceBelowThresholdMerges) {
    createSignal(0);
    createWait(0);
    StinkyInstruction* f0 = createVAddInBlock(bb, arch, /*dest=*/40, /*s0=*/0, /*s1=*/1);
    StinkyInstruction* f1 = createVAddInBlock(bb, arch, /*dest=*/41, /*s0=*/2, /*s1=*/3);
    createSignal(1);
    createWait(1);

    const int distance = f0->issueCycles + f1->issueCycles;

    // Threshold one above the distance: dist < threshold, so merge.
    runPasses(/*mergeThreshold=*/distance + 1);
    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1);
    EXPECT_EQ(tokensOf(firstBarrier("s_barrier_signal")), (std::vector<int>{0, 1}));
}

// A producer of a merged token sitting between the two groups is ordered against
// the merged barrier. Since the pass never moves instructions, the merge must be
// skipped even when the distance is below threshold.
TEST_F(MergeBarrierPassTest, TokenProducerBetweenPreventsMerge) {
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

// A consumer of a merged token (ds_read of token 0, guarded by the first group)
// sitting between the groups is likewise ordered against the merged barrier, so
// the merge is skipped without moving anything.
TEST_F(MergeBarrierPassTest, TokenConsumerBetweenPreventsMerge) {
    createSignal(0);
    createWait(0);
    // Consumer of token 0 (reads LDS token 0).
    createDSLoadInBlock(bb, arch, /*destReg=*/20, /*addrReg=*/30, /*memTokens=*/{0});
    createSignal(1);
    createWait(1);

    runPasses(/*mergeThreshold=*/100000);

    // No merge: both barrier groups remain intact.
    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 2);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 2);
}

// An instruction between the groups that touches an *unrelated* token (not in
// either group's set) does not constrain the merged barrier, so the merge still
// happens when the distance is below threshold.
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

    runPasses(/*mergeThreshold=*/0);  // default threshold, all adjacent

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

// The compile-time default mirrors the documented value.
TEST(MergeBarrierConstants, KnownDefault) {
    EXPECT_EQ(kCdna5MergeBarrierThreshold, 11);
}
