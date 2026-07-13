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

#include "TestHelpers.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

int countStinkyInstructions(const BasicBlock& bb) {
    int count = 0;
    for (const IRBase& ir : bb)
        if (ir.getType() == IRBase::IRType::StinkyTofu) count++;
    return count;
}

std::vector<StinkyInstruction*> collectInstructions(BasicBlock& bb) {
    std::vector<StinkyInstruction*> insts;
    for (IRBase& ir : bb)
        if (ir.getType() == IRBase::IRType::StinkyTofu)
            insts.push_back(cast<StinkyInstruction>(&ir));
    return insts;
}

}  // namespace

class ExecMaskGroupingTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;

    void SetUp() override {
        func = std::make_unique<Function>("exec_mask_grouping_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }

    StinkyInstruction* createExecNarrow(int srcSgpr) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, arch));
        inst->addDestReg(StinkyRegister::getEXECRegister(32));
        inst->addSrcReg(StinkyRegister("s", srcSgpr, 1));
        return inst;
    }

    StinkyInstruction* createExecReset() {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, arch));
        inst->addDestReg(StinkyRegister::getEXECRegister(32));
        inst->addSrcReg(StinkyRegister(-1));
        return inst;
    }
};

TEST_F(ExecMaskGroupingTest, CollapseAndExpand_RoundTripsSpanExactly) {
    StinkyInstruction* before = createVAddInBlock(bb, arch, 40, 41, 42);
    StinkyInstruction* narrow = createExecNarrow(10);
    StinkyInstruction* g1 = createVAddInBlock(bb, arch, 0, 1, 2);
    StinkyInstruction* g2 = createVAddInBlock(bb, arch, 3, 4, 5);
    StinkyInstruction* g3 = createVAddInBlock(bb, arch, 6, 7, 8);
    StinkyInstruction* reset = createExecReset();
    StinkyInstruction* after = createVAddInBlock(bb, arch, 50, 51, 52);

    const std::vector<StinkyInstruction*> original = {before, narrow, g1, g2, g3, reset, after};
    const int totalIssue = narrow->issueCycles + g1->issueCycles + g2->issueCycles +
                           g3->issueCycles + reset->issueCycles;
    const int totalLatency = narrow->latencyCycles + g1->latencyCycles + g2->latencyCycles +
                             g3->latencyCycles + reset->latencyCycles;

    AsmIRBuilder builder(*bb, arch);
    collapseExecMaskedRegions(*bb, builder, /*wavefrontSize=*/32);

    ASSERT_EQ(countStinkyInstructions(*bb), 3);
    std::vector<StinkyInstruction*> collapsed = collectInstructions(*bb);
    EXPECT_EQ(collapsed[0], before);
    EXPECT_EQ(collapsed[2], after);

    StinkyInstruction* group = collapsed[1];
    EXPECT_EQ(group->getUnifiedOpcode(), GFX::EXEC_GROUP);
    EXPECT_TRUE(isExecMaskGroup(*group));
    EXPECT_EQ(group->issueCycles, totalIssue);
    EXPECT_EQ(group->latencyCycles, totalLatency);

    auto hasDest = [&](RegType type, uint32_t idx) {
        for (const StinkyRegister& d : group->getDestRegs())
            if (d.isRegister() && d.reg.type == type && d.reg.idx == idx) return true;
        return false;
    };
    EXPECT_TRUE(hasDest(RegType::EXEC_LO, 0));
    EXPECT_TRUE(hasDest(RegType::V, 0));
    EXPECT_TRUE(hasDest(RegType::V, 3));
    EXPECT_TRUE(hasDest(RegType::V, 6));

    auto* groupData = group->getModifier<ExecGroupData>();
    ASSERT_NE(groupData, nullptr);
    const std::vector<StinkyInstruction*> expectedChildren = {narrow, g1, g2, g3, reset};
    EXPECT_EQ(groupData->children, expectedChildren);

    expandExecMaskedGroups(*bb);

    EXPECT_EQ(collectInstructions(*bb), original);
    for (StinkyInstruction* inst : collectInstructions(*bb))
        EXPECT_NE(inst->getUnifiedOpcode(), GFX::EXEC_GROUP);
}

TEST_F(ExecMaskGroupingTest, CollapseExecMaskedRegions_HandlesNesting) {
    StinkyInstruction* outerNarrow = createExecNarrow(10);
    StinkyInstruction* innerNarrow = createExecNarrow(11);
    StinkyInstruction* guarded = createVAddInBlock(bb, arch, 0, 1, 2);
    StinkyInstruction* innerReset = createExecReset();
    StinkyInstruction* guarded2 = createVAddInBlock(bb, arch, 3, 4, 5);
    StinkyInstruction* outerReset = createExecReset();

    AsmIRBuilder builder(*bb, arch);
    collapseExecMaskedRegions(*bb, builder, 32);

    ASSERT_EQ(countStinkyInstructions(*bb), 1);
    StinkyInstruction* group = collectInstructions(*bb)[0];
    EXPECT_EQ(group->getUnifiedOpcode(), GFX::EXEC_GROUP);

    auto* groupData = group->getModifier<ExecGroupData>();
    ASSERT_NE(groupData, nullptr);
    const std::vector<StinkyInstruction*> expectedChildren = {outerNarrow, innerNarrow, guarded,
                                                              innerReset,  guarded2,    outerReset};
    EXPECT_EQ(groupData->children, expectedChildren);
}

TEST_F(ExecMaskGroupingTest, CollapseExecMaskedRegions_LeavesUnmatchedNarrowUngrouped) {
    createExecNarrow(10);
    createVAddInBlock(bb, arch, 0, 1, 2);

    const int n = countStinkyInstructions(*bb);
    AsmIRBuilder builder(*bb, arch);
    collapseExecMaskedRegions(*bb, builder, 32);

    EXPECT_EQ(countStinkyInstructions(*bb), n);
    for (StinkyInstruction* inst : collectInstructions(*bb))
        EXPECT_NE(inst->getUnifiedOpcode(), GFX::EXEC_GROUP);
}

TEST_F(ExecMaskGroupingTest, CollapseExecMaskedRegions_MultipleSiblingSpans) {
    createExecNarrow(10);
    createVAddInBlock(bb, arch, 0, 1, 2);
    createExecReset();

    createVAddInBlock(bb, arch, 20, 21, 22);

    createExecNarrow(12);
    createVAddInBlock(bb, arch, 3, 4, 5);
    createExecReset();

    AsmIRBuilder builder(*bb, arch);
    collapseExecMaskedRegions(*bb, builder, 32);

    ASSERT_EQ(countStinkyInstructions(*bb), 3);
    std::vector<StinkyInstruction*> insts = collectInstructions(*bb);
    EXPECT_EQ(insts[0]->getUnifiedOpcode(), GFX::EXEC_GROUP);
    EXPECT_EQ(insts[1]->getUnifiedOpcode(), GFX::v_add_f32);
    EXPECT_EQ(insts[2]->getUnifiedOpcode(), GFX::EXEC_GROUP);
    EXPECT_NE(insts[0], insts[2]);
}

// s_mov_b64 exec, -1 must be recognized as a full-mask reset even in wave32.
TEST_F(ExecMaskGroupingTest, CollapseExecMaskedRegions_B64ResetClosesWave32Span) {
    StinkyInstruction* narrow = createExecNarrow(10);
    StinkyInstruction* guarded = createVAddInBlock(bb, arch, 0, 1, 2);

    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* b64reset = builder.create(getMCIDByUOp(GFX::s_mov_b64, arch));
    b64reset->addDestReg(StinkyRegister(RegType::EXEC, 0, 2));
    b64reset->addSrcReg(StinkyRegister(-1));

    collapseExecMaskedRegions(*bb, builder, /*wavefrontSize=*/32);

    ASSERT_EQ(countStinkyInstructions(*bb), 1);
    StinkyInstruction* group = collectInstructions(*bb)[0];
    EXPECT_EQ(group->getUnifiedOpcode(), GFX::EXEC_GROUP);
    auto* groupData = group->getModifier<ExecGroupData>();
    ASSERT_NE(groupData, nullptr);
    EXPECT_EQ(groupData->children, (std::vector<StinkyInstruction*>{narrow, guarded, b64reset}));
}

// Any exec write that isn't a literal-(-1) reset opens a span, including non-mov
// instructions like s_or_b32.
TEST_F(ExecMaskGroupingTest, CollapseExecMaskedRegions_NonMovExecWriteOpensSpan) {
    AsmIRBuilder builder(*bb, arch);

    StinkyInstruction* orNarrow = builder.create(getMCIDByUOp(GFX::s_or_b32, arch));
    orNarrow->addDestReg(StinkyRegister::getEXECRegister(32));
    orNarrow->addSrcReg(StinkyRegister::getEXECRegister(32));
    orNarrow->addSrcReg(StinkyRegister("s", 5, 1));

    StinkyInstruction* guarded = createVAddInBlock(bb, arch, 0, 1, 2);
    StinkyInstruction* reset = createExecReset();

    collapseExecMaskedRegions(*bb, builder, /*wavefrontSize=*/32);

    ASSERT_EQ(countStinkyInstructions(*bb), 1);
    StinkyInstruction* group = collectInstructions(*bb)[0];
    EXPECT_EQ(group->getUnifiedOpcode(), GFX::EXEC_GROUP);
    auto* groupData = group->getModifier<ExecGroupData>();
    ASSERT_NE(groupData, nullptr);
    EXPECT_EQ(groupData->children, (std::vector<StinkyInstruction*>{orNarrow, guarded, reset}));
}
