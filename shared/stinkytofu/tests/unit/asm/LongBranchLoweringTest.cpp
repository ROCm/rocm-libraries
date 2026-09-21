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
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/LongBranchLoweringPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class LongBranchLoweringTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;

    void SetUp() override {
        func = std::make_unique<Function>("long_branch_lowering_test");
        setFunctionArch(*func, arch);
        entry = func->createBasicBlock("entry");
    }

    /// Run only the LongBranchLoweringPass (no CFG split).
    void runLowering() {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(func->getGemmTileConfig());
        pm.addPass(createLongBranchLoweringPass());
        pm.run(*func);
    }

    /// Run lowering then CFG builder (the production order).
    void runLoweringThenCFG() {
        PassManager pm;
        registerAllAnalyses(pm.getAnalysisManager());
        pm.setGemmTileConfig(func->getGemmTileConfig());
        pm.addPass(createLongBranchLoweringPass());
        pm.addPass(createCFGBuilderPass());
        pm.run(*func);
    }

    StinkyInstruction* createGetpc(BasicBlock* bb, int dstSGPR) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_getpc_b64, arch));
        inst->addDestReg(StinkyRegister("s", dstSGPR, 2));
        return inst;
    }

    StinkyInstruction* createAddI32Label(BasicBlock* bb, int dstSGPR, const std::string& label,
                                         int offset) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_add_i32, arch));
        inst->addDestReg(StinkyRegister("s", dstSGPR, 1));
        inst->addSrcReg(StinkyRegister(label));
        inst->addSrcReg(StinkyRegister(offset));
        return inst;
    }

    StinkyInstruction* createAddU32(BasicBlock* bb, int dstSGPR, int src0SGPR, int src1SGPR) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_add_u32, arch));
        inst->addDestReg(StinkyRegister("s", dstSGPR, 1));
        inst->addSrcReg(StinkyRegister("s", src0SGPR, 1));
        inst->addSrcReg(StinkyRegister("s", src1SGPR, 1));
        return inst;
    }

    StinkyInstruction* createAddCU32_zero(BasicBlock* bb, int dstSGPR, int src0SGPR) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_addc_u32, arch));
        inst->addDestReg(StinkyRegister("s", dstSGPR, 1));
        inst->addSrcReg(StinkyRegister("s", src0SGPR, 1));
        inst->addSrcReg(StinkyRegister(0));
        return inst;
    }

    StinkyInstruction* createSubU32(BasicBlock* bb, int dstSGPR, int src0SGPR, int src1SGPR) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_sub_u32, arch));
        inst->addDestReg(StinkyRegister("s", dstSGPR, 1));
        inst->addSrcReg(StinkyRegister("s", src0SGPR, 1));
        inst->addSrcReg(StinkyRegister("s", src1SGPR, 1));
        return inst;
    }

    StinkyInstruction* createSubBU32_zero(BasicBlock* bb, int dstSGPR, int src0SGPR) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_subb_u32, arch));
        inst->addDestReg(StinkyRegister("s", dstSGPR, 1));
        inst->addSrcReg(StinkyRegister("s", src0SGPR, 1));
        inst->addSrcReg(StinkyRegister(0));
        return inst;
    }

    StinkyInstruction* createAbsI32(BasicBlock* bb, int dstSGPR, int srcSGPR) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_abs_i32, arch));
        inst->addDestReg(StinkyRegister("s", dstSGPR, 1));
        inst->addSrcReg(StinkyRegister("s", srcSGPR, 1));
        return inst;
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

    /// Find the (single) s_setpc_b64 in this function.
    StinkyInstruction* findSetpc() {
        for (BasicBlock& bb : *func) {
            for (auto& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (inst && inst->getUnifiedOpcode() == GFX::s_setpc_b64) return inst;
            }
        }
        return nullptr;
    }
};

// Positive-arm rocisa idiom (the most common form, used by SLongBranchPositive
// and the positive arm of SLongBranch). The pass should recover the label and
// stamp it on the s_setpc_b64.
TEST_F(LongBranchLoweringTest, RecoversPositiveLongBranchLabel) {
    createGetpc(entry, /*dstSGPR=*/62);
    createAddI32Label(entry, /*dstSGPR=*/64, "label_PrefetchEnd", /*offset=*/4);
    createAddU32(entry, /*dst=*/62, /*src0=*/62, /*src1=*/64);
    createAddCU32_zero(entry, /*dst=*/63, /*src0=*/63);
    StinkyInstruction* setpc = createSetpc(entry, /*srcSGPR=*/62);

    runLowering();

    const auto* lbl = setpc->getModifier<LabelData>();
    ASSERT_NE(lbl, nullptr) << "Long-branch idiom should be recognized and "
                               "annotated with LabelData.";
    EXPECT_EQ(lbl->label, "label_PrefetchEnd");

    auto targets = getBranchTargets(*setpc);
    ASSERT_EQ(targets.size(), 1u);
    EXPECT_EQ(targets[0], "label_PrefetchEnd");
}

// Negative-arm rocisa idiom: s_abs_i32 + s_sub_u32 + s_subb_u32 between the
// s_add_i32 anchor and the terminating s_setpc_b64. The pass must walk past
// s_abs_i32 and recognize s_sub_u32 / s_subb_u32 as the lo/hi accumulators.
TEST_F(LongBranchLoweringTest, RecoversNegativeLongBranchLabel) {
    createGetpc(entry, /*dstSGPR=*/62);
    createAddI32Label(entry, /*dstSGPR=*/64, "label_LoopBegin", /*offset=*/4);
    createAbsI32(entry, /*dstSGPR=*/64, /*srcSGPR=*/64);
    createSubU32(entry, /*dst=*/62, /*src0=*/62, /*src1=*/64);
    createSubBU32_zero(entry, /*dst=*/63, /*src0=*/63);
    StinkyInstruction* setpc = createSetpc(entry, /*srcSGPR=*/62);

    runLowering();

    const auto* lbl = setpc->getModifier<LabelData>();
    ASSERT_NE(lbl, nullptr);
    EXPECT_EQ(lbl->label, "label_LoopBegin");
}

// A bare s_setpc_b64 with no surrounding idiom must NOT get a label stamped:
// the recovered information is unsound and would create a phantom CFG edge.
TEST_F(LongBranchLoweringTest, BareSetpcIsNotMisidentified) {
    StinkyInstruction* setpc = createSetpc(entry, /*srcSGPR=*/62);
    createNop(entry);

    runLowering();

    EXPECT_EQ(setpc->getModifier<LabelData>(), nullptr)
        << "An s_setpc_b64 with no def chain must remain an opaque indirect "
           "branch; we must not invent a target label.";
}

// If the instruction already carries a LabelData (e.g. the converter
// recognized the rocisa SSetPCB64 long-branch hint) the pass must leave it
// alone.
TEST_F(LongBranchLoweringTest, ExistingLabelDataIsPreserved) {
    StinkyInstruction* setpc = createSetpc(entry, /*srcSGPR=*/62);
    setpc->addModifier(LabelData{"explicit_label"});

    // Add the idiom anchored at a different label so that, if the pass
    // mistakenly overwrote, we'd notice.
    BasicBlock* dummyBlock = func->createBasicBlock("dummy");
    createGetpc(dummyBlock, /*dstSGPR=*/62);
    createAddI32Label(dummyBlock, /*dstSGPR=*/64, "should_not_use_this", /*offset=*/4);
    createAddU32(dummyBlock, /*dst=*/62, /*src0=*/62, /*src1=*/64);
    createAddCU32_zero(dummyBlock, /*dst=*/63, /*src0=*/63);

    runLowering();

    const auto* lbl = setpc->getModifier<LabelData>();
    ASSERT_NE(lbl, nullptr);
    EXPECT_EQ(lbl->label, "explicit_label");
}

// End-to-end: after LongBranchLoweringPass + CFGBuilderPass, the CFG sees
// the long-branch as a direct edge into the labelled target, and the block
// immediately after the s_setpc_b64 is unreachable (no fall-through).
TEST_F(LongBranchLoweringTest, CFGEdgeIsBuiltToRecoveredLabel) {
    createGetpc(entry, /*dstSGPR=*/62);
    createAddI32Label(entry, /*dstSGPR=*/64, "label_after", /*offset=*/4);
    createAddU32(entry, /*dst=*/62, /*src0=*/62, /*src1=*/64);
    createAddCU32_zero(entry, /*dst=*/63, /*src0=*/63);
    createSetpc(entry, /*srcSGPR=*/62);
    createLabelInst(entry, "fall_through_block");
    createNop(entry);
    createLabelInst(entry, "label_after");
    createNop(entry);

    runLoweringThenCFG();

    BasicBlock* setpcBlock = func->getEntryBlock();
    BasicBlock* fallBlock = findBlock("fall_through_block");
    BasicBlock* targetBlock = findBlock("label_after");
    ASSERT_NE(setpcBlock, nullptr);
    ASSERT_NE(fallBlock, nullptr);
    ASSERT_NE(targetBlock, nullptr);

    const auto& succs = setpcBlock->getSuccessors();
    ASSERT_EQ(succs.size(), 1u) << "Long-branch must have exactly one CFG successor "
                                   "after lowering.";
    EXPECT_EQ(succs.front(), targetBlock);

    EXPECT_TRUE(fallBlock->getPredecessors().empty())
        << "Block immediately after the s_setpc_b64 must remain unreachable.";

    const auto& preds = targetBlock->getPredecessors();
    EXPECT_NE(std::find(preds.begin(), preds.end(), setpcBlock), preds.end())
        << "Target block must list the long-branch source block as a predecessor.";
}

}  // namespace
