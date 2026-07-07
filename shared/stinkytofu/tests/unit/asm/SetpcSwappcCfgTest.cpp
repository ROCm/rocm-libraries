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

    void SetUp() override {
        func = std::make_unique<Function>("setpc_swappc_cfg_test");
        setFunctionArch(*func, arch);
        entry = func->createBasicBlock("entry");
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

// when the rocisa->StinkyTofu converter encounters an SSetPCB64 with non-empty
// longBranchLabel, it stamps a LabelData{...} modifier
// on the s_setpc_b64. This test verifies the StinkyTofu-side behaviour the
// converter relies on: an s_setpc_b64 carrying LabelData{X} must produce a CFG
// edge to the basic block labelled X, with no fall-through and no other
// successors. It exercises the contract directly without running
// LongBranchLoweringPass, isolating the converter's stamp-LabelData
// responsibility.
TEST_F(SetpcSwappcCfgTest, SetpcWithLabelDataYieldsDirectCFGEdge) {
    StinkyInstruction* setpc = createSetpc(entry, /*srcSGPR=*/62);
    /// Mirror what ToStinkyTofuUtils.cpp::legalizeInstruction does for
    /// rocisa::SSetPCB64 with longBranchLabel="label_target".
    setpc->addModifier<LabelData>(LabelData{"label_target"});
    createLabelInst(entry, "fall_through");
    createNop(entry);
    createLabelInst(entry, "label_target");
    createNop(entry);

    runCFGBuilder();

    BasicBlock* setpcBlock = func->getEntryBlock();
    BasicBlock* fallBlock = findBlock("fall_through");
    BasicBlock* targetBlock = findBlock("label_target");
    ASSERT_NE(setpcBlock, nullptr);
    ASSERT_NE(fallBlock, nullptr);
    ASSERT_NE(targetBlock, nullptr);

    const auto& succs = setpcBlock->getSuccessors();
    ASSERT_EQ(succs.size(), 1u)
        << "Converter-stamped LabelData must produce exactly one CFG successor.";
    EXPECT_EQ(succs.front(), targetBlock);

    EXPECT_TRUE(fallBlock->getPredecessors().empty())
        << "Block immediately after the s_setpc_b64 must remain unreachable.";

    const auto& preds = targetBlock->getPredecessors();
    EXPECT_NE(std::find(preds.begin(), preds.end(), setpcBlock), preds.end())
        << "Target block must list the s_setpc_b64 source block as a predecessor.";
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
