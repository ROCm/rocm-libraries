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
#include <gtest/gtest.h>

#include <algorithm>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class SetpcSwappcCfgTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
    AnalysisManager am;

    void SetUp() override {
        func = std::make_unique<Function>("setpc_swappc_cfg_test");
        setFunctionArch(*func, arch);
        entry = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    /// Build a fresh PassManager and run only the CFG builder over \p func.
    void runCFGBuilder() {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(func->getGemmTileConfig());
        pm.addPass(createCFGBuilderPass());
        pm.run(*func);
    }

    StinkyInstruction* createSetpc(BasicBlock* bb, int srcSGPR) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_setpc_b64, arch));
        inst->addSrcReg(StinkyRegister("s", srcSGPR, 2));
        return inst;
    }

    StinkyInstruction* createSwappc(BasicBlock* bb, int dstSGPR, int srcSGPR) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_swappc_b64, arch));
        inst->addDestReg(StinkyRegister("s", dstSGPR, 2));
        inst->addSrcReg(StinkyRegister("s", srcSGPR, 2));
        return inst;
    }

    StinkyInstruction* createNop(BasicBlock* bb) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_nop, arch));
        inst->addSrcReg(StinkyRegister(0));
        return inst;
    }

    StinkyInstruction* createLabelInst(BasicBlock* bb, const std::string& name) {
        AsmIRBuilder builder(*bb, arch);
        return builder.createLabel(name);
    }

    BasicBlock* findBlock(const std::string& label) {
        for (BasicBlock& bb : *func) {
            if (bb.getLabel() == label) return &bb;
        }
        return nullptr;
    }
};

// After Commit 3, s_setpc_b64 is classified as IF_Branch + IF_IndirectBranch.
// With no LabelData/CallTargetData modifier, it is an unknown indirect branch:
// no intra-Function successor, no fall-through. The block AFTER the setpc must
// therefore be unreachable (no predecessor).
TEST_F(SetpcSwappcCfgTest, SetpcWithoutMetadataHasNoSuccessorAndDoesNotFallThrough) {
    createSetpc(entry, /*srcSGPR=*/0);
    createLabelInst(entry, "after_setpc");
    createNop(entry);

    runCFGBuilder();

    BasicBlock* setpcBlock = func->getEntryBlock();
    BasicBlock* afterBlock = findBlock("after_setpc");
    ASSERT_NE(setpcBlock, nullptr);
    ASSERT_NE(afterBlock, nullptr);
    ASSERT_NE(setpcBlock, afterBlock) << "CFG builder must split at the label";

    EXPECT_TRUE(setpcBlock->getSuccessors().empty())
        << "Indirect branch with no metadata has no statically-known successor.";
    EXPECT_TRUE(afterBlock->getPredecessors().empty())
        << "Block after s_setpc_b64 must be unreachable: setpc is unconditional "
           "and has no fall-through.";
}

// s_swappc_b64 is a call (IF_Branch + IF_Call + IF_IndirectBranch). Even though
// its callee target is not visible inside this Function, control returns to the
// instruction after the call; the local CFG must keep the fall-through edge.
TEST_F(SetpcSwappcCfgTest, SwappcFallsThroughToNextBlock) {
    createSwappc(entry, /*dstSGPR=*/2, /*srcSGPR=*/0);
    createLabelInst(entry, "after_call");
    createNop(entry);

    runCFGBuilder();

    BasicBlock* callBlock = func->getEntryBlock();
    BasicBlock* afterBlock = findBlock("after_call");
    ASSERT_NE(callBlock, nullptr);
    ASSERT_NE(afterBlock, nullptr);
    ASSERT_NE(callBlock, afterBlock);

    // The callee target lives in another Function (modelled later by
    // CallGraphAnalysis), so the local successor list does NOT include it.
    EXPECT_TRUE(callBlock->getSuccessors().empty() ||
                std::find(callBlock->getSuccessors().begin(), callBlock->getSuccessors().end(),
                          afterBlock) != callBlock->getSuccessors().end())
        << "Call block may only have a fall-through successor (no inter-Function edges).";

    const auto& succs = callBlock->getSuccessors();
    ASSERT_EQ(succs.size(), 1u) << "Call must keep exactly one fall-through CFG successor.";
    EXPECT_EQ(succs.front(), afterBlock);

    const auto& preds = afterBlock->getPredecessors();
    ASSERT_EQ(preds.size(), 1u);
    EXPECT_EQ(preds.front(), callBlock);
}

// A normal s_branch must still build a single successor edge to its labelled
// target. Sanity check that the new fall-through rule did not regress
// unconditional-branch handling.
TEST_F(SetpcSwappcCfgTest, UnconditionalBranchStillNoFallThrough) {
    AsmIRBuilder builder(*entry, arch);
    StinkyInstruction* br = builder.create(getMCIDByUOp(GFX::s_branch, arch));
    br->addSrcReg(StinkyRegister(std::string("target")));
    createLabelInst(entry, "fall_block");
    createNop(entry);
    createLabelInst(entry, "target");
    createNop(entry);

    runCFGBuilder();

    BasicBlock* brBlock = func->getEntryBlock();
    BasicBlock* fallBlock = findBlock("fall_block");
    BasicBlock* targetBlock = findBlock("target");
    ASSERT_NE(brBlock, nullptr);
    ASSERT_NE(fallBlock, nullptr);
    ASSERT_NE(targetBlock, nullptr);

    const auto& succs = brBlock->getSuccessors();
    ASSERT_EQ(succs.size(), 1u);
    EXPECT_EQ(succs.front(), targetBlock);

    EXPECT_TRUE(fallBlock->getPredecessors().empty())
        << "Unconditional branch must not fall through to the next block.";
}

}  // namespace
