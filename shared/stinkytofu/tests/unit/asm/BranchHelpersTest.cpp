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
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class BranchHelpersTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;

    void SetUp() override {
        func = std::make_unique<Function>("branch_helpers_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }
};

// s_branch is an unconditional direct branch and exposes its label via the
// legacy LiteralString src0 fallback path.
TEST_F(BranchHelpersTest, SBranchIsUnconditionalDirectBranch) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_branch, arch));
    inst->addSrcReg(StinkyRegister(std::string("label_foo")));

    EXPECT_TRUE(isBranch(*inst));
    EXPECT_FALSE(isConditionalBranch(*inst));
    EXPECT_TRUE(isUnconditionalBranch(*inst));
    EXPECT_FALSE(isCall(*inst));
    EXPECT_FALSE(isReturn(*inst));
    EXPECT_FALSE(isIndirectBranch(*inst));

    auto targets = getBranchTargets(*inst);
    ASSERT_EQ(targets.size(), 1u);
    EXPECT_EQ(targets[0], "label_foo");
    EXPECT_EQ(getBranchTarget(*inst), "label_foo");
}

// s_cbranch_scc1 is a conditional branch.
TEST_F(BranchHelpersTest, SCbranchIsConditionalBranch) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cbranch_scc1, arch));
    inst->addSrcReg(StinkyRegister(std::string("label_loop")));

    EXPECT_TRUE(isBranch(*inst));
    EXPECT_TRUE(isConditionalBranch(*inst));
    EXPECT_FALSE(isUnconditionalBranch(*inst));
    EXPECT_FALSE(isCall(*inst));
    EXPECT_FALSE(isReturn(*inst));

    auto targets = getBranchTargets(*inst);
    ASSERT_EQ(targets.size(), 1u);
    EXPECT_EQ(targets[0], "label_loop");
}

// LabelData modifier takes precedence over the LiteralString src0 fallback.
TEST_F(BranchHelpersTest, LabelDataModifierIsPreferred) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_branch, arch));
    inst->addSrcReg(StinkyRegister(std::string("legacy_label")));
    inst->addModifier<LabelData>(LabelData{"preferred_label"});

    auto targets = getBranchTargets(*inst);
    ASSERT_EQ(targets.size(), 1u);
    EXPECT_EQ(targets[0], "preferred_label");
}

// A non-branch instruction yields no targets and is not a call/return/indirect.
TEST_F(BranchHelpersTest, NonBranchHasNoTargets) {
    StinkyInstruction* inst = createVAddInBlock(bb, arch, 0, 1, 2);

    EXPECT_FALSE(isBranch(*inst));
    EXPECT_FALSE(isConditionalBranch(*inst));
    EXPECT_FALSE(isCall(*inst));
    EXPECT_FALSE(isReturn(*inst));
    EXPECT_FALSE(isIndirectBranch(*inst));
    EXPECT_TRUE(getBranchTargets(*inst).empty());
    EXPECT_EQ(getBranchTarget(*inst), "");
}

}  // namespace
