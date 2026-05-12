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

#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class CallReturnModifiersTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;

    void SetUp() override {
        func = std::make_unique<Function>("call_return_modifiers_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }

    /// Build a branch carrying \p label as its LiteralString src0 so we can
    /// confirm the call/return modifiers override that fallback. We use
    /// s_branch as a generic branch carrier; the modifiers' effect on
    /// getBranchTargets / isReturn does not depend on the opcode.
    StinkyInstruction* makeBranchWithLegacyLabel(const std::string& label) {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_branch, arch));
        inst->addSrcReg(StinkyRegister(label));
        return inst;
    }
};

// CallTargetData masks the legacy LiteralString src0 fallback: a swappc-style
// call does not have an intra-Function successor; the call edge lives in the
// CallGraphAnalysis instead.
TEST_F(CallReturnModifiersTest, CallTargetDataMasksIntraFunctionSuccessor) {
    StinkyInstruction* inst = makeBranchWithLegacyLabel("would_be_target");

    inst->addModifier<CallTargetData>(
        CallTargetData{std::vector<std::string>{"activation_relu_gwvw1", "activation_gelu_gwvw1"}});

    const auto* cd = inst->getModifier<CallTargetData>();
    ASSERT_NE(cd, nullptr);
    ASSERT_EQ(cd->calleeNames.size(), 2u);
    EXPECT_EQ(cd->calleeNames[0], "activation_relu_gwvw1");
    EXPECT_EQ(cd->calleeNames[1], "activation_gelu_gwvw1");

    EXPECT_TRUE(isBranch(*inst));
    EXPECT_TRUE(getBranchTargets(*inst).empty())
        << "Calls have no intra-Function successor; their targets are tracked "
           "by CallGraphAnalysis instead.";
    EXPECT_EQ(getBranchTarget(*inst), "");
    EXPECT_FALSE(isReturn(*inst));
}

// ReturnTerminatorData also masks intra-Function successors *and* makes
// isReturn() true even when the IF_Return hardware flag is not set.
TEST_F(CallReturnModifiersTest, ReturnTerminatorDataMasksAndMarksReturn) {
    StinkyInstruction* inst = makeBranchWithLegacyLabel("would_be_target");

    EXPECT_FALSE(isReturn(*inst));

    inst->addModifier<ReturnTerminatorData>(ReturnTerminatorData{});

    EXPECT_TRUE(isBranch(*inst));
    EXPECT_TRUE(isReturn(*inst));
    EXPECT_TRUE(getBranchTargets(*inst).empty())
        << "Returns have no intra-Function successor; their return-edge is "
           "modelled in the call graph.";
    EXPECT_EQ(getBranchTarget(*inst), "");
}

// CallTargetData / ReturnTerminatorData take priority over a LabelData
// modifier: the converter may stamp both for documentation, but the call/return
// nature wins for CFG-edge purposes.
TEST_F(CallReturnModifiersTest, CallReturnModifiersOverrideLabelData) {
    StinkyInstruction* call = makeBranchWithLegacyLabel("legacy");
    call->addModifier<LabelData>(LabelData{"some_label"});
    call->addModifier<CallTargetData>(CallTargetData{std::vector<std::string>{"callee"}});
    EXPECT_TRUE(getBranchTargets(*call).empty());

    StinkyInstruction* ret = makeBranchWithLegacyLabel("legacy");
    ret->addModifier<LabelData>(LabelData{"some_label"});
    ret->addModifier<ReturnTerminatorData>(ReturnTerminatorData{});
    EXPECT_TRUE(getBranchTargets(*ret).empty());
    EXPECT_TRUE(isReturn(*ret));
}

// classof / dyn_cast wiring works for both new modifier types.
TEST_F(CallReturnModifiersTest, ModifierDynCastWorks) {
    StinkyInstruction* inst = makeBranchWithLegacyLabel("legacy");
    inst->addModifier<CallTargetData>(CallTargetData{std::vector<std::string>{"a", "b"}});
    inst->addModifier<ReturnTerminatorData>(ReturnTerminatorData{});

    bool sawCall = false, sawReturn = false;
    for (const std::unique_ptr<Modifier>& mod : inst->getModifiers()) {
        if (auto* cd = dyn_cast<CallTargetData>(mod.get())) {
            sawCall = true;
            EXPECT_EQ(cd->calleeNames.size(), 2u);
        }
        if (dyn_cast<ReturnTerminatorData>(mod.get())) {
            sawReturn = true;
        }
    }
    EXPECT_TRUE(sawCall);
    EXPECT_TRUE(sawReturn);
}

// Modifier clone() (used when copying instructions) preserves the new types.
TEST_F(CallReturnModifiersTest, ModifierCloneIsDeep) {
    CallTargetData original{std::vector<std::string>{"foo", "bar"}};
    std::unique_ptr<Modifier> copy = original.clone();
    auto* cloned = dyn_cast<CallTargetData>(copy.get());
    ASSERT_NE(cloned, nullptr);
    ASSERT_EQ(cloned->calleeNames.size(), 2u);
    EXPECT_EQ(cloned->calleeNames[0], "foo");
    EXPECT_EQ(cloned->calleeNames[1], "bar");

    ReturnTerminatorData rt{};
    std::unique_ptr<Modifier> rtCopy = rt.clone();
    EXPECT_NE(dyn_cast<ReturnTerminatorData>(rtCopy.get()), nullptr);
}

}  // namespace
