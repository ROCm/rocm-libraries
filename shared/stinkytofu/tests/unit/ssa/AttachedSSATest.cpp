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
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/AttachedSSAVerifier.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

class AttachedSSATest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
    }

    std::unique_ptr<Function> func;
};

TEST_F(AttachedSSATest, PreLiftInstructionsHaveNoAttachedSSA) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);

    EXPECT_FALSE(func->hasAttachedSSA());
    EXPECT_FALSE(add->hasAttachedSSA());
    EXPECT_EQ(add->getNumSrcRegs(), 2u);
    EXPECT_EQ(add->getSrcReg(0).reg.idx, 0u);
    EXPECT_TRUE(verifyAttachedSSA(*func).ok());
}

TEST_F(AttachedSSATest, AttachResultsAndOperandsMaintainsUseLists) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* def = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* use = createVAddInBlock(entry, kArch, 3, 2, 4);

    StinkySSAValue* src0 = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* src1 = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* result = func->ssaArena().createRegister(RegType::V);
    src0->setPhysicalBinding({RegType::V, 0, 1, 0, false, false, false});

    AttachedSSA defSSA;
    defSSA.results.push_back(result);
    defSSA.operands.push_back(makeSSAValueOperand(src0));
    defSSA.operands.push_back(makeSSAValueOperand(src1));
    def->attachSSA(std::move(defSSA));

    AttachedSSA useSSA;
    useSSA.results.push_back(func->ssaArena().createRegister(RegType::V));
    useSSA.operands.push_back(makeSSAValueOperand(result));
    useSSA.operands.push_back(makeSSAValueOperand(src1));
    use->attachSSA(std::move(useSSA));

    EXPECT_TRUE(def->hasAttachedSSA());
    EXPECT_EQ(result->defOp(), def);
    EXPECT_EQ(result->resultIndex(), 0);
    EXPECT_EQ(result->useCount(), 1u);
    EXPECT_EQ(src1->useCount(), 2u);
    EXPECT_EQ(use->getSSAOperandValue(0), result);
    EXPECT_TRUE(verifyAttachedSSA(*func).ok()) << verifyAttachedSSA(*func).toString();
}

TEST_F(AttachedSSATest, SetOperandValueRewiresUseLists) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* defA = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* defB = createVAddInBlock(entry, kArch, 4, 5, 6);
    StinkyInstruction* use = createVAddInBlock(entry, kArch, 3, 2, 1);

    StinkySSAValue* a = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* b = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* other = func->ssaArena().createRegister(RegType::V);

    AttachedSSA aSSA;
    aSSA.results.push_back(a);
    aSSA.operands.push_back(makeSSAValueOperand(other));
    aSSA.operands.push_back(makeSSAValueOperand(other));
    defA->attachSSA(std::move(aSSA));

    AttachedSSA bSSA;
    bSSA.results.push_back(b);
    bSSA.operands.push_back(makeSSAValueOperand(other));
    bSSA.operands.push_back(makeSSAValueOperand(other));
    defB->attachSSA(std::move(bSSA));

    AttachedSSA useSSA;
    useSSA.results.push_back(func->ssaArena().createRegister(RegType::V));
    useSSA.operands.push_back(makeSSAValueOperand(b));
    useSSA.operands.push_back(makeSSAValueOperand(other));
    use->attachSSA(std::move(useSSA));

    EXPECT_EQ(use->getSSAOperandValue(0), b);
    use->setSSAOperandValue(0, a);
    EXPECT_EQ(use->getSSAOperandValue(0), a);
    EXPECT_EQ(a->useCount(), 1u);
    EXPECT_EQ(b->useCount(), 0u);
    EXPECT_TRUE(verifyAttachedSSA(*func).ok()) << verifyAttachedSSA(*func).toString();
}

TEST_F(AttachedSSATest, ReplaceAllUsesWithUpdatesEveryUse) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* def = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* use0 = createVAddInBlock(entry, kArch, 3, 2, 1);
    StinkyInstruction* use1 = createVAddInBlock(entry, kArch, 4, 2, 1);

    StinkySSAValue* oldV = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* newV = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* other = func->ssaArena().createRegister(RegType::V);

    AttachedSSA defSSA;
    defSSA.results.push_back(oldV);
    defSSA.operands.push_back(makeSSAValueOperand(other));
    defSSA.operands.push_back(makeSSAValueOperand(other));
    def->attachSSA(std::move(defSSA));

    AttachedSSA u0;
    u0.results.push_back(func->ssaArena().createRegister(RegType::V));
    u0.operands.push_back(makeSSAValueOperand(oldV));
    u0.operands.push_back(makeSSAValueOperand(other));
    use0->attachSSA(std::move(u0));

    AttachedSSA u1;
    u1.results.push_back(func->ssaArena().createRegister(RegType::V));
    u1.operands.push_back(makeSSAValueOperand(oldV));
    u1.operands.push_back(makeSSAValueOperand(other));
    use1->attachSSA(std::move(u1));

    oldV->replaceAllUsesWith(newV);
    EXPECT_TRUE(oldV->useEmpty());
    EXPECT_EQ(newV->useCount(), 2u);
    EXPECT_EQ(use0->getSSAOperandValue(0), newV);
    EXPECT_EQ(use1->getSSAOperandValue(0), newV);
    EXPECT_TRUE(verifyAttachedSSA(*func).ok()) << verifyAttachedSSA(*func).toString();
}

TEST_F(AttachedSSATest, CloneDoesNotCopyAttachedSSA) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* def = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkySSAValue* result = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* src = func->ssaArena().createRegister(RegType::V);

    AttachedSSA ssa;
    ssa.results.push_back(result);
    ssa.operands.push_back(makeSSAValueOperand(src));
    ssa.operands.push_back(makeSSAValueOperand(src));
    def->attachSSA(std::move(ssa));

    StinkyInstruction* cloned = def->clone();
    EXPECT_FALSE(cloned->hasAttachedSSA());
    EXPECT_EQ(cloned->getNumSrcRegs(), def->getNumSrcRegs());
    entry->appendIR(cloned);
}

TEST_F(AttachedSSATest, BlockArgumentsRecordPhiIncomingUses) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkySSAValue* fromLeft = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* fromRight = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* phi = func->ssaArena().createBlockArgument(RegType::V);

    join->addSSAArgument(phi);
    join->setSSAArgumentIncoming(0, left, fromLeft);
    join->setSSAArgumentIncoming(0, right, fromRight);

    EXPECT_EQ(phi->kind(), StinkySSAValue::Kind::BlockArgument);
    EXPECT_EQ(fromLeft->useCount(), 1u);
    EXPECT_EQ(fromRight->useCount(), 1u);
    EXPECT_EQ(join->ssaArguments()[0].incoming[0].predecessor, left);
    EXPECT_TRUE(verifyAttachedSSA(*func).ok()) << verifyAttachedSSA(*func).toString();
}

TEST_F(AttachedSSATest, VerifierDetectsNullValueOperand) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* def = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkySSAValue* result = func->ssaArena().createRegister(RegType::V);

    AttachedSSA ssa;
    ssa.results.push_back(result);
    ssa.operands.push_back(makeSSAValueOperand(nullptr));
    ssa.operands.push_back(makeSSAValueOperand(nullptr));
    def->attachSSA(std::move(ssa));

    const AttachedSSAVerificationResult verified = verifyAttachedSSA(*func);
    EXPECT_FALSE(verified.ok());
    EXPECT_NE(verified.toString().find("value operand is null"), std::string::npos)
        << verified.toString();
}

TEST_F(AttachedSSATest, VerifierDetectsABlockArgumentThatIsNotOne) {
    BasicBlock* join = func->createBasicBlock("join");

    // Kind is what separates a merge from an instruction result, so a Register
    // value standing in as a block argument has to be caught.
    join->addSSAArgument(func->ssaArena().createRegister(RegType::V));

    const AttachedSSAVerificationResult verified = verifyAttachedSSA(*func);
    EXPECT_FALSE(verified.ok());
    EXPECT_NE(verified.toString().find("is not a block argument"), std::string::npos)
        << verified.toString();
}

TEST_F(AttachedSSATest, VerifierDetectsABlockArgumentWithADefiningInstruction) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* def = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkySSAValue* arg = func->ssaArena().createBlockArgument(RegType::V);

    AttachedSSA ssa;
    ssa.results.push_back(arg);
    ssa.operands.push_back(makeSSAValueOperand(func->ssaArena().createRegister(RegType::V)));
    ssa.operands.push_back(makeSSAValueOperand(func->ssaArena().createRegister(RegType::V)));
    def->attachSSA(std::move(ssa));
    entry->addSSAArgument(arg);

    const AttachedSSAVerificationResult verified = verifyAttachedSSA(*func);
    EXPECT_FALSE(verified.ok());
    EXPECT_NE(verified.toString().find("has a defining instruction"), std::string::npos)
        << verified.toString();
}

TEST_F(AttachedSSATest, VerifierDetectsIncomingFromANonPredecessor) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* elsewhere = func->createBasicBlock("elsewhere");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(left, join);

    StinkySSAValue* phi = func->ssaArena().createBlockArgument(RegType::V);
    join->addSSAArgument(phi);
    join->setSSAArgumentIncoming(0, left, func->ssaArena().createRegister(RegType::V));
    // An edge that does not exist cannot carry a value.
    join->setSSAArgumentIncoming(0, elsewhere, func->ssaArena().createRegister(RegType::V));

    const AttachedSSAVerificationResult verified = verifyAttachedSSA(*func);
    EXPECT_FALSE(verified.ok());
    EXPECT_NE(verified.toString().find("predecessor is not a CFG predecessor"), std::string::npos)
        << verified.toString();
}

TEST_F(AttachedSSATest, VerifierDiagnosticsAreDeterministic) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* first = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* second = createVAddInBlock(entry, kArch, 3, 2, 1);

    for (StinkyInstruction* inst : {first, second}) {
        AttachedSSA ssa;
        ssa.results.push_back(func->ssaArena().createRegister(RegType::V));
        ssa.operands.push_back(makeSSAValueOperand(nullptr));
        ssa.operands.push_back(makeSSAValueOperand(nullptr));
        inst->attachSSA(std::move(ssa));
    }

    // Diagnostics walk the function rather than a hash container, so a broken
    // function reports the same text every run.
    const std::string once = verifyAttachedSSA(*func).toString();
    EXPECT_EQ(verifyAttachedSSA(*func).toString(), once);
    EXPECT_NE(once.find("#0 operand0"), std::string::npos) << once;
    EXPECT_LT(once.find("#0 operand0"), once.find("#1 operand0")) << once;
}

TEST_F(AttachedSSATest, ClearAttachedSSAUnbindsValues) {
    BasicBlock* entry = func->createBasicBlock("entry");
    StinkyInstruction* def = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkySSAValue* result = func->ssaArena().createRegister(RegType::V);
    StinkySSAValue* src = func->ssaArena().createRegister(RegType::V);

    AttachedSSA ssa;
    ssa.results.push_back(result);
    ssa.operands.push_back(makeSSAValueOperand(src));
    ssa.operands.push_back(makeSSAValueOperand(src));
    def->attachSSA(std::move(ssa));

    def->clearAttachedSSA();
    EXPECT_FALSE(def->hasAttachedSSA());
    EXPECT_EQ(result->defOp(), nullptr);
    EXPECT_TRUE(src->useEmpty());

    func->clearAttachedSSA();
    EXPECT_EQ(func->ssaArena().valueCount(), 0u);
}

}  // namespace
