/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include <gtest/gtest.h>

#include <memory>

#include "PhiTestFixtures.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/DefUseAnalysisCleanup.hpp"
#include "stinkytofu/transforms/asm/PhiPlacement.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

size_t countPhis(const Function& func) {
    size_t phis = 0;
    for (const BasicBlock& bb : func) {
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst != nullptr && inst->getUnifiedOpcode() == GFX::PHI) ++phis;
        }
    }
    return phis;
}

bool anyChains(const Function& func) {
    for (const BasicBlock& bb : func) {
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst == nullptr) continue;
            if (!inst->getSources().empty() || !inst->getUsers().empty()) return true;
        }
    }
    return false;
}

class DefUseAnalysisCleanupTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
    }

    std::unique_ptr<Function> func;
};

}  // namespace

TEST_F(DefUseAnalysisCleanupTest, CleanupOfAnUntouchedFunctionRemovesNothing) {
    setFunctionArch(*func, kArch);
    BasicBlock* entry = func->createBasicBlock("entry");
    createVAddInBlock(entry, kArch, 2, 0, 1);

    const DefUseAnalysisCleanup cleanup = discardDefUseAnalysis(*func);

    EXPECT_EQ(cleanup.removedPhis, 0u);
    EXPECT_EQ(cleanup.clearedInstructions, 1u);
    EXPECT_EQ(func->size(), 1u);
}

TEST_F(DefUseAnalysisCleanupTest, RemovesAnalysisPhisAndChainsAfterBuildUseDefChain) {
    buildIteratedDFCfg(*func, kArch);
    buildUseDefChain(*func, /*clearExisting=*/false);

    ASSERT_GT(countPhis(*func), 0u);
    ASSERT_TRUE(anyChains(*func));

    const DefUseAnalysisCleanup cleanup = discardDefUseAnalysis(*func);

    EXPECT_GT(cleanup.removedPhis, 0u);
    EXPECT_EQ(countPhis(*func), 0u);
    EXPECT_FALSE(anyChains(*func));
}

TEST_F(DefUseAnalysisCleanupTest, PassClearsAnalysesAndLetsLiftingSucceed) {
    // Together these two passes are the canonical SSA boundary: the cleanup is a
    // mutation, so it cannot live inside construction, and lifting rejects a
    // function that still carries analysis PHIs.
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);
    buildUseDefChain(*func, /*clearExisting=*/false);
    ASSERT_GT(countPhis(*func), 0u);
    ASSERT_TRUE(anyChains(*func));

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.addPass(createRemoveDefUseAnalysisPass());
    pm.addPass(createLiftAsmRegistersToSSAPass());
    pm.run(*func);

    EXPECT_EQ(countPhis(*func), 0u);
    EXPECT_FALSE(anyChains(*func));
    EXPECT_TRUE(cfg.hUse->getSources().empty());
    // The merges the analysis PHIs approximated are block arguments now.
    EXPECT_TRUE(func->hasAttachedSSA());
    size_t blockArgs = 0;
    for (const BasicBlock& bb : *func) blockArgs += bb.ssaArguments().size();
    EXPECT_GT(blockArgs, 0u);
}

TEST_F(DefUseAnalysisCleanupTest, CleanupKeepsBlocksEdgesAndRealInstructions) {
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);
    buildUseDefChain(*func, /*clearExisting=*/false);

    const size_t blocksBefore = func->size();
    const size_t successorsBefore = cfg.entry->getSuccessors().size();

    discardDefUseAnalysis(*func);

    EXPECT_EQ(func->size(), blocksBefore);
    EXPECT_EQ(cfg.entry->getSuccessors().size(), successorsBefore);
    // The real instructions the fixture created are still addressable.
    EXPECT_EQ(cfg.entryDef->getUnifiedOpcode(), GFX::v_add_f32);
    EXPECT_EQ(cfg.hUse->getUnifiedOpcode(), GFX::v_add_f32);
}

TEST_F(DefUseAnalysisCleanupTest, CleanupIsIdempotent) {
    buildIteratedDFCfg(*func, kArch);
    buildUseDefChain(*func, /*clearExisting=*/false);

    discardDefUseAnalysis(*func);
    const DefUseAnalysisCleanup second = discardDefUseAnalysis(*func);

    EXPECT_EQ(second.removedPhis, 0u);
    EXPECT_EQ(countPhis(*func), 0u);
    EXPECT_FALSE(anyChains(*func));
}

TEST_F(DefUseAnalysisCleanupTest, RemoveAnalysisPhisCountsWhatItErased) {
    buildIteratedDFCfg(*func, kArch);
    insertPhiInstructions(*func, /*clearExisting=*/false);

    const size_t before = countPhis(*func);
    ASSERT_GT(before, 0u);

    EXPECT_EQ(removeAnalysisPhis(*func), before);
    EXPECT_EQ(countPhis(*func), 0u);
}

TEST_F(DefUseAnalysisCleanupTest, ClearDefUseChainsVisitsEveryInstruction) {
    setFunctionArch(*func, kArch);
    BasicBlock* entry = func->createBasicBlock("entry");
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 1);
    buildUseDefChain(*func, /*clearExisting=*/false);
    ASSERT_TRUE(anyChains(*func));

    EXPECT_EQ(clearDefUseChains(*func), 2u);
    EXPECT_FALSE(anyChains(*func));
}

TEST_F(DefUseAnalysisCleanupTest, ChainsCanBeRebuiltAfterCleanup) {
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);
    buildUseDefChain(*func, /*clearExisting=*/false);
    discardDefUseAnalysis(*func);

    // The existing analyses remain usable on a cleaned function.
    buildUseDefChain(*func, /*clearExisting=*/false);

    EXPECT_GT(countPhis(*func), 0u);
    EXPECT_FALSE(cfg.hUse->getSources().empty());
}
