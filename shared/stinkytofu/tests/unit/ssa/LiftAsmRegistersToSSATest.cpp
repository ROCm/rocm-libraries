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

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "AttachedSSATestUtils.hpp"
#include "PhiTestFixtures.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/ssa/AttachedSSAVerifier.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

/// SSA form of the whole function, the artifact a dump would produce.
std::string ssaIR(const Function& function) {
    AsmPrinterOptions options;
    options.ssaForm = true;
    return toString(function, options);
}

class LiftAsmRegistersToSSATest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
    }

    /// Lifts and requires success.
    void lift(const LiftAsmRegistersToSSAOptions& options = {}) {
        Expected<LiftAttachedSSAResult> result = liftAsmRegistersToAttachedSSA(*func, options);
        EXPECT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
        const AttachedSSAVerificationResult verification = verifyAttachedSSA(*func);
        EXPECT_TRUE(verification.ok()) << verification.toString();
    }

    /// Lifts and requires failure, returning the diagnostic.
    std::string liftError(const LiftAsmRegistersToSSAOptions& options = {}) {
        Expected<LiftAttachedSSAResult> result = liftAsmRegistersToAttachedSSA(*func, options);
        EXPECT_TRUE(result.hasError());
        return result.hasError() ? result.getError() : std::string{};
    }

    size_t valueCount() const {
        return func->ssaArena().valueCount();
    }

    /// Value with dense ID \p id, which lift assigns in creation order.
    StinkySSAValue* value(uint32_t id) const {
        return func->ssaArena().get(id);
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
};

}  // namespace

TEST_F(LiftAsmRegistersToSSATest, EmptyFunctionAttachesNothing) {
    Function empty("empty");
    Expected<LiftAttachedSSAResult> result = liftAsmRegistersToAttachedSSA(empty);
    ASSERT_TRUE(result.hasValue()) << result.getError();
    EXPECT_EQ(result->valueCount, 0u);
    EXPECT_EQ(result->blockArgumentCount, 0u);
    EXPECT_FALSE(empty.hasAttachedSSA());
}

TEST_F(LiftAsmRegistersToSSATest, EmptyBlockAttachesNothing) {
    lift();
    EXPECT_EQ(valueCount(), 0u);
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(LiftAsmRegistersToSSATest, StraightLineProducesVerifiedSSA) {
    StinkyInstruction* add = createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);

    lift();

    ASSERT_EQ(valueCount(), 3u);
    // Live-ins are block arguments of the entry, created in register order.
    ASSERT_EQ(entry->ssaArguments().size(), 2u);
    EXPECT_EQ(entry->ssaArguments()[0].value, value(1));
    EXPECT_EQ(entry->ssaArguments()[1].value, value(2));
    EXPECT_EQ(value(1)->kind(), StinkySSAValue::Kind::BlockArgument);
    EXPECT_EQ(bindingKeyOf(value(1)).idx, 0u);
    EXPECT_EQ(bindingKeyOf(value(2)).idx, 1u);

    EXPECT_EQ(value(3)->kind(), StinkySSAValue::Kind::Register);
    EXPECT_EQ(value(3)->defOp(), add);
    EXPECT_EQ(bindingKeyOf(value(3)).idx, 2u);
}

TEST_F(LiftAsmRegistersToSSATest, LiftedDumpIsExact) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 0);

    lift();

    EXPECT_EQ(ssaIR(*func),
              "st.func @kernel() {\n"
              "  ^entry(%1:v, %2:v):\n"
              "    %3:v = \"st.v_add_f32\"(%1:v, %2:v) "
              "{ issueCycles = 1, latencyCycles = 5 }\n"
              "    %4:v = \"st.v_add_f32\"(%3:v, %1:v) "
              "{ issueCycles = 1, latencyCycles = 5 }\n"
              "}\n");
}

TEST_F(LiftAsmRegistersToSSATest, RepeatedDefinitionsOfOneRegisterBecomeDistinctValues) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* last = createVAddInBlock(entry, kArch, 3, 2, 2);

    lift();

    // v0, v1 live-ins, then three definitions; the last reads the second v2.
    ASSERT_EQ(valueCount(), 5u);
    EXPECT_EQ(bindingKeyOf(value(3)).idx, 2u);
    EXPECT_EQ(bindingKeyOf(value(4)).idx, 2u);
    EXPECT_NE(value(3), value(4));
    EXPECT_EQ(ssaSourceValue(*last, 0), value(4));
    EXPECT_EQ(ssaSourceValue(*last, 1), value(4));
}

TEST_F(LiftAsmRegistersToSSATest, OneValueUsedTwiceRecordsTwoUses) {
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 0);

    lift();

    ASSERT_EQ(valueCount(), 2u);
    // Uses are per operand and never deduplicated, so reading one value through
    // two operands records two of them.
    EXPECT_EQ(value(1)->useCount(), 2u);
    EXPECT_EQ(ssaSourceValue(*add, 0), value(1));
    EXPECT_EQ(ssaSourceValue(*add, 1), value(1));
}

TEST_F(LiftAsmRegistersToSSATest, OneLiveInIsSharedByEveryReadOfThatUnit) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 0, 1);

    lift();

    EXPECT_EQ(value(1)->useCount(), 2u);
    EXPECT_EQ(value(2)->useCount(), 2u);
    // Two live-ins plus two definitions, not four live-ins.
    EXPECT_EQ(valueCount(), 4u);
}

TEST_F(LiftAsmRegistersToSSATest, ReadModifyWriteReadsTheOldValueAndDefinesANewOne) {
    // v2 = v_add_f32 v2, v1 reads the incoming v2 and defines a new one.
    StinkyInstruction* add = createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/2, /*src1=*/1);

    lift();

    // Live-ins v1 and v2 are created first, in register order, then the result.
    ASSERT_EQ(valueCount(), 3u);
    StinkySSAValue* incoming = value(2);
    EXPECT_EQ(incoming->kind(), StinkySSAValue::Kind::BlockArgument);
    EXPECT_EQ(bindingKeyOf(incoming).idx, 2u);
    EXPECT_EQ(incoming->useCount(), 1u);
    EXPECT_EQ(ssaSourceValue(*add, 0), incoming);

    StinkySSAValue* defined = value(3);
    EXPECT_EQ(defined->kind(), StinkySSAValue::Kind::Register);
    EXPECT_EQ(bindingKeyOf(defined).idx, 2u);
    EXPECT_TRUE(defined->useEmpty());
}

TEST_F(LiftAsmRegistersToSSATest, MultiDwordRangeBindsOneValuePerDword) {
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);

    lift();

    ASSERT_EQ(valueCount(), 5u);
    const std::vector<StinkySSAValue*> units = ssaDestUnits(*load, 0);
    ASSERT_EQ(units.size(), 4u);
    for (unsigned unit = 0; unit < 4; ++unit) {
        EXPECT_EQ(units[unit]->kind(), StinkySSAValue::Kind::Register);
        EXPECT_EQ(units[unit]->defOp(), load);
        EXPECT_EQ(units[unit]->resultIndex(), unit);
        EXPECT_EQ(bindingKeyOf(units[unit]).idx, 4u + unit);
    }
}

TEST_F(LiftAsmRegistersToSSATest, PartialUseOfAWideDefinitionKeepsUnitIdentity) {
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    StinkyInstruction* store = createDSWriteInBlock(entry, kArch, /*addr=*/0, /*data=*/4);

    lift();

    const std::vector<StinkySSAValue*> loaded = ssaDestUnits(*load, 0);
    ASSERT_EQ(loaded.size(), 4u);
    EXPECT_EQ(ssaSourceUnits(*store, 1), (std::vector<StinkySSAValue*>{loaded[0], loaded[1]}));
    // v6 and v7 are defined but never read, so they gain no uses.
    EXPECT_TRUE(loaded[2]->useEmpty());
    EXPECT_TRUE(loaded[3]->useEmpty());
}

TEST_F(LiftAsmRegistersToSSATest, LiteralOperandsBindNothing) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 2, 1));
    mov->addSrcReg(StinkyRegister(7));

    lift();

    ASSERT_EQ(valueCount(), 1u);
    // The operand slot still exists, carrying the literal payload rather than a
    // value, so operand indices keep lining up with srcRegs.
    ASSERT_EQ(mov->getNumSSAOperands(), 1u);
    EXPECT_EQ(mov->getSSAOperand(0)->kind(), StinkyOpOperand::Kind::LiteralInt);
    EXPECT_TRUE(ssaSourceUnits(*mov, 0).empty());
}

TEST_F(LiftAsmRegistersToSSATest, SpecialRegistersAreNotLifted) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* cmp = builder.create(getMCIDByUOp(GFX::v_cmp_eq_u32, kArch));
    cmp->addDestReg(StinkyRegister::getSCCRegister());
    cmp->addSrcReg(StinkyRegister("v", 0, 1));
    cmp->addSrcReg(StinkyRegister("v", 1, 1));

    lift();

    EXPECT_EQ(valueCount(), 2u);
    EXPECT_EQ(cmp->getNumSSAResults(), 0u);
    EXPECT_TRUE(ssaDestUnits(*cmp, 0).empty());
    // SCC keeps its physical spelling in an SSA dump.
    EXPECT_TRUE(contains(ssaIR(*func), "SCC0 = \"st.v_cmp_eq_u32\"(%1:v, %2:v)")) << ssaIR(*func);
}

TEST_F(LiftAsmRegistersToSSATest, InstructionsWithoutAllocatableOperandsAreStillAttached) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* nop = builder.create(getMCIDByUOp(GFX::s_nop, kArch));

    lift();

    // Attachment is what says "this instruction was lifted", so it happens even
    // when there is nothing to bind.
    EXPECT_TRUE(nop->hasAttachedSSA());
    EXPECT_EQ(nop->getNumSSAResults(), 0u);
}

TEST_F(LiftAsmRegistersToSSATest, LabelsAreSkippedButStillConsumeADiagnosticIndex) {
    AsmIRBuilder builder(*entry, kArch);
    builder.createLabel("top");
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 3, 1));
    mov->addSrcReg(StinkyRegister("a", 4, 1));

    // Instruction indices count every StinkyInstruction, labels included, so the
    // add is #1 even though the label carries no dataflow.
    EXPECT_TRUE(contains(liftError(), "@kernel #1 src0")) << liftError();
}

TEST_F(LiftAsmRegistersToSSATest, StrictModeRejectsInferredLiveIns) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    LiftAsmRegistersToSSAOptions options;
    options.allowInferredLiveIns = false;
    const std::string error = liftError(options);
    EXPECT_TRUE(contains(error, "@kernel #0 src0: reads v0 with no reaching definition")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, StrictModeAcceptsFullyDefinedCode) {
    // Every read is defined earlier in the block, so no live-in is needed.
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister(1));
    createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/0);

    LiftAsmRegistersToSSAOptions options;
    options.allowInferredLiveIns = false;
    lift(options);

    EXPECT_EQ(valueCount(), 2u);
    EXPECT_TRUE(entry->ssaArguments().empty());
    for (StinkySSAValue* v : func->ssaArena().values())
        EXPECT_EQ(v->kind(), StinkySSAValue::Kind::Register);
}

TEST_F(LiftAsmRegistersToSSATest, RejectsUnreachableBlocks) {
    func->createBasicBlock("orphan");
    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "^orphan is unreachable from the entry")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsEntryBlockThatIsALoopHeader) {
    // A live-in arriving at a loop header has no predecessor edge to merge on.
    func->addEdge(entry, entry);
    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "the entry must not be a loop header")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsTemplateVirtualRegisters) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister::Virtual(0));
    mov->addSrcReg(StinkyRegister("v", 1, 1));

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "unresolved template virtual register")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, LiftsSgprOperands) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    lift();

    ASSERT_EQ(valueCount(), 2u);
    EXPECT_EQ(value(1)->kind(), StinkySSAValue::Kind::BlockArgument);
    EXPECT_EQ(value(1)->type().regType, RegType::S);
    EXPECT_EQ(bindingKeyOf(value(1)).idx, 4u);
    EXPECT_EQ(value(2)->type().regType, RegType::V);
    // The class travels with the value, so a dump names it.
    EXPECT_TRUE(contains(ssaIR(*func), "^entry(%1:s):")) << ssaIR(*func);
}

TEST_F(LiftAsmRegistersToSSATest, ScalarAndVectorRegistersWithTheSameIndexAreDistinct) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 4, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    lift();

    // s4 is a live-in and v4 is defined here; the shared index must not merge
    // them, because the register key carries the class.
    ASSERT_EQ(valueCount(), 2u);
    EXPECT_EQ(bindingKeyOf(value(1)).type, RegType::S);
    EXPECT_EQ(bindingKeyOf(value(2)).type, RegType::V);
    EXPECT_EQ(bindingKeyOf(value(1)).idx, bindingKeyOf(value(2)).idx);
}

TEST_F(LiftAsmRegistersToSSATest, LiftsWideScalarRangesPerDword) {
    // tensor_load_to_lds reads a 4-SGPR base and an 8-SGPR descriptor.
    StinkyInstruction* load = createTensorLoadInBlock(entry, kArch, /*src0=*/8, /*src1=*/16);

    lift();

    const std::vector<StinkySSAValue*> base = ssaSourceUnits(*load, 0);
    const std::vector<StinkySSAValue*> descriptor = ssaSourceUnits(*load, 1);
    ASSERT_EQ(base.size(), 4u);
    EXPECT_EQ(descriptor.size(), 8u);

    for (size_t unit = 0; unit < base.size(); ++unit) {
        EXPECT_EQ(bindingKeyOf(base[unit]).type, RegType::S);
        EXPECT_EQ(bindingKeyOf(base[unit]).idx, 8u + unit);
    }
}

TEST_F(LiftAsmRegistersToSSATest, RejectsAccumulatorOperands) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* wmma = builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kArch));
    wmma->addDestReg(StinkyRegister("a", 10, 8));
    wmma->addSrcReg(StinkyRegister("v", 20, 8));
    wmma->addSrcReg(StinkyRegister("v", 30, 8));
    wmma->addSrcReg(StinkyRegister("a", 10, 8));

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "register class 'a' is not lifted yet")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsAnalysisPhis) {
    AsmIRBuilder builder(*entry, kArch);
    builder.createPhi(RegType::V, 2);

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "analysis PHIs must be removed")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsTrue16HalfOperands) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("v", 1, 1));
    mov->addModifier<True16Modifiers>(
        True16Modifiers(HighBitSel::HIGH, HighBitSel::NONE, {HighBitSel::NONE}));

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "True16 half operands")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, RejectsOneInstructionDefiningAUnitTwice) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* instruction = builder.create(getMCIDByUOp(GFX::v_add_f32, kArch));
    instruction->addDestReg(StinkyRegister("v", 2, 1));
    instruction->addDestReg(StinkyRegister("v", 2, 1));
    instruction->addSrcReg(StinkyRegister("v", 0, 1));

    const std::string error = liftError();
    EXPECT_TRUE(contains(error, "defines v2 more than once")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, DiagnosticsNameTheFunctionAndInstruction) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 3, 1));
    mov->addSrcReg(StinkyRegister("a", 4, 1));

    EXPECT_EQ(liftError(),
              "@kernel #1 src0: register class 'a' is not lifted yet; "
              "VGPRs and SGPRs are supported");
}

TEST_F(LiftAsmRegistersToSSATest, KernelPreflightSpotsCallSitesInAnyFunction) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    Function callee("callee");
    setFunctionArch(callee, kArch);
    BasicBlock* calleeEntry = callee.createBasicBlock("entry");
    createVAddInBlock(calleeEntry, kArch, 3, 0, 1);

    const std::vector<const Function*> kernel{func.get(), &callee};
    EXPECT_FALSE(kernelHasCallSites(kernel));

    // A call anywhere in the kernel disqualifies the whole thing, not just the
    // function that contains it.
    AsmIRBuilder builder(*calleeEntry, kArch);
    StinkyInstruction* call = builder.create(getMCIDByUOp(GFX::s_swappc_b64, kArch));
    ASSERT_TRUE(isCall(*call));

    EXPECT_TRUE(kernelHasCallSites(kernel));
    EXPECT_TRUE(kernelHasCallSites({&callee}));
}

TEST_F(LiftAsmRegistersToSSATest, RepeatedLiftsProduceIdenticalSSA) {
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createDSWriteInBlock(entry, kArch, 0, 4);
    createVAddInBlock(entry, kArch, 8, 4, 5);

    lift();
    const std::string first = ssaIR(*func);
    lift();

    // Value IDs are what an AllocationResult is keyed by, so re-lifting an
    // unchanged function has to reproduce them exactly.
    EXPECT_EQ(ssaIR(*func), first);
}

TEST_F(LiftAsmRegistersToSSATest, RelaftingReplacesTheEarlierValues) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    lift();
    const size_t afterFirst = valueCount();
    lift();

    // Lifting clears before it builds, so values do not accumulate.
    EXPECT_EQ(valueCount(), afterFirst);
}

TEST_F(LiftAsmRegistersToSSATest, FailureLeavesNothingAttached) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("second");

    const std::string error = liftError();

    // Construction is atomic: a rejected function keeps no partial SSA, so a
    // consumer cannot mistake it for a lifted one.
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(func->hasAttachedSSA());
    EXPECT_EQ(valueCount(), 0u);
    EXPECT_TRUE(entry->ssaArguments().empty());
}

TEST_F(LiftAsmRegistersToSSATest, FailureClearsSSAFromAnEarlierLift) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();
    ASSERT_TRUE(func->hasAttachedSSA());

    // Making the function unsupported must not leave SSA describing the version
    // of the function that was lifted earlier.
    func->createBasicBlock("orphan");
    EXPECT_FALSE(liftError().empty());
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(LiftAsmRegistersToSSATest, DoesNotRewritePhysicalOperands) {
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 8, 4, 5);
    const std::string before = toString(*func);

    lift();

    // The instruction stream keeps its physical spelling; SSA is attached beside
    // it, which is what lets legacy replay and diagnostics still work.
    EXPECT_EQ(toString(*func), before);
}

// ---------------------------------------------------------------------------
// Register ranges
// ---------------------------------------------------------------------------

TEST_F(LiftAsmRegistersToSSATest, RangeUnitsKeepOperandOrderAndConsecutiveOrigins) {
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, /*dest=*/8, /*addr=*/0);

    lift();

    const std::vector<StinkySSAValue*> units = ssaDestUnits(*load, 0);
    ASSERT_EQ(units.size(), 4u);
    EXPECT_EQ(bindingIndicesOf(units), (std::vector<unsigned>{8, 9, 10, 11}));
    for (size_t unit = 0; unit < units.size(); ++unit) EXPECT_EQ(units[unit]->resultIndex(), unit);
}

TEST_F(LiftAsmRegistersToSSATest, OverlappingSourceRangesShareTheOverlappingUnits) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);

    // src0 = v[4:5], src1 = v[5:6]: v5 is read through both operands.
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* store = builder.create(getMCIDByUOp(GFX::ds_store_b64, kArch));
    store->addSrcReg(StinkyRegister("v", 4, 2));
    store->addSrcReg(StinkyRegister("v", 5, 2));

    lift();

    const std::vector<StinkySSAValue*> first = ssaSourceUnits(*store, 0);
    const std::vector<StinkySSAValue*> second = ssaSourceUnits(*store, 1);
    ASSERT_EQ(first.size(), 2u);
    ASSERT_EQ(second.size(), 2u);
    EXPECT_EQ(bindingIndicesOf(first), (std::vector<unsigned>{4, 5}));
    EXPECT_EQ(bindingIndicesOf(second), (std::vector<unsigned>{5, 6}));
    // One value per physical unit, so the shared v5 appears in both operands.
    EXPECT_EQ(first[1], second[0]);
    EXPECT_EQ(first[1]->useCount(), 2u);
}

TEST_F(LiftAsmRegistersToSSATest, DisjointRangesStayIndependent) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    createDsReadB128InBlock(entry, kArch, /*dest=*/8, /*addr=*/0);

    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* store = builder.create(getMCIDByUOp(GFX::ds_store_b64, kArch));
    store->addSrcReg(StinkyRegister("v", 4, 2));
    store->addSrcReg(StinkyRegister("v", 8, 2));

    lift();

    EXPECT_EQ(bindingIndicesOf(ssaSourceUnits(*store, 0)), (std::vector<unsigned>{4, 5}));
    EXPECT_EQ(bindingIndicesOf(ssaSourceUnits(*store, 1)), (std::vector<unsigned>{8, 9}));
}

TEST_F(LiftAsmRegistersToSSATest, PartialRedefinitionOfARangeOnlyReplacesThoseUnits) {
    StinkyInstruction* wide = createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    // Overwrite only v4.
    StinkyInstruction* narrow = createVAddInBlock(entry, kArch, /*dest=*/4, /*src0=*/20, 21);

    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* consumer = builder.create(getMCIDByUOp(GFX::ds_store_b64, kArch));
    consumer->addSrcReg(StinkyRegister("v", 0, 1));
    consumer->addSrcReg(StinkyRegister("v", 4, 4));

    lift();

    const std::vector<StinkySSAValue*> wideUnits = ssaDestUnits(*wide, 0);
    const std::vector<StinkySSAValue*> consumed = ssaSourceUnits(*consumer, 1);
    ASSERT_EQ(wideUnits.size(), 4u);
    ASSERT_EQ(consumed.size(), 4u);

    // v4 comes from the narrow redefinition; v5 to v7 still come from the load.
    EXPECT_EQ(consumed[0], ssaDefinedValue(*narrow));
    EXPECT_NE(consumed[0], wideUnits[0]);
    EXPECT_EQ(consumed[1], wideUnits[1]);
    EXPECT_EQ(consumed[2], wideUnits[2]);
    EXPECT_EQ(consumed[3], wideUnits[3]);
}

TEST_F(LiftAsmRegistersToSSATest, WideReadModifyWriteAccumulatorChainsThroughValues) {
    // v[0:7] = wmma(v[8:15], v[16:23], v[0:7]) twice: the accumulator is read
    // and written by each instruction, so the chain must thread through values.
    AsmIRBuilder builder(*entry, kArch);
    std::vector<StinkyInstruction*> wmmas;
    for (int i = 0; i < 2; ++i) {
        StinkyInstruction* wmma =
            builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kArch));
        wmma->addDestReg(StinkyRegister("v", 0, 8));
        wmma->addSrcReg(StinkyRegister("v", 8, 8));
        wmma->addSrcReg(StinkyRegister("v", 16, 8));
        wmma->addSrcReg(StinkyRegister("v", 0, 8));
        wmmas.push_back(wmma);
    }

    lift();

    const std::vector<StinkySSAValue*> firstRead = ssaSourceUnits(*wmmas[0], 2);
    const std::vector<StinkySSAValue*> firstDef = ssaDestUnits(*wmmas[0], 0);
    const std::vector<StinkySSAValue*> secondRead = ssaSourceUnits(*wmmas[1], 2);
    const std::vector<StinkySSAValue*> secondDef = ssaDestUnits(*wmmas[1], 0);

    ASSERT_EQ(firstRead.size(), 8u);
    ASSERT_EQ(secondDef.size(), 8u);
    for (size_t unit = 0; unit < 8; ++unit) {
        // Each instruction reads the previous value and defines a new one.
        EXPECT_EQ(firstRead[unit]->kind(), StinkySSAValue::Kind::BlockArgument);
        EXPECT_NE(firstRead[unit], firstDef[unit]);
        EXPECT_EQ(secondRead[unit], firstDef[unit]);
        EXPECT_NE(secondRead[unit], secondDef[unit]);
        // The tie stays observable: read and written units share one register.
        EXPECT_EQ(bindingKeyOf(firstDef[unit]).idx, bindingKeyOf(firstRead[unit]).idx);
    }
}

TEST_F(LiftAsmRegistersToSSATest, DestinationOverlappingItsOwnSourceReadsTheOldUnits) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);

    // v[4:5] = op(v[5:6]): v5 is both read and written.
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* shifted = builder.create(getMCIDByUOp(GFX::v_lshlrev_b64, kArch));
    shifted->addDestReg(StinkyRegister("v", 4, 2));
    shifted->addSrcReg(StinkyRegister("v", 5, 2));

    lift();

    const std::vector<StinkySSAValue*> read = ssaSourceUnits(*shifted, 0);
    const std::vector<StinkySSAValue*> written = ssaDestUnits(*shifted, 0);
    ASSERT_EQ(read.size(), 2u);
    ASSERT_EQ(written.size(), 2u);
    EXPECT_EQ(bindingIndicesOf(read), (std::vector<unsigned>{5, 6}));
    EXPECT_EQ(bindingIndicesOf(written), (std::vector<unsigned>{4, 5}));
    // The v5 that is read is the incoming value, not the one defined here.
    EXPECT_NE(read[0], written[1]);
}

// ---------------------------------------------------------------------------
// Lift scope: which register classes become SSA, and what the rest look like
// ---------------------------------------------------------------------------

namespace {

/// v<dest> = v_mov_b32 s<src>: one operand in each class, so a scope decision is
/// visible on a single instruction.
StinkyInstruction* createVMovFromSgpr(BasicBlock* bb, int destVgpr, int srcSgpr) {
    AsmIRBuilder builder(*bb, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", destVgpr, 1));
    mov->addSrcReg(StinkyRegister("s", srcSgpr, 1));
    return mov;
}

bool anyOfClass(const Function& function, RegType regClass) {
    for (StinkySSAValue* value : function.ssaArena().values()) {
        if (value != nullptr && value->type().regType == regClass) return true;
    }
    return false;
}

LiftAsmRegistersToSSAOptions scopedTo(RegType regClass) {
    LiftAsmRegistersToSSAOptions options;
    options.classes = RegClassSet::only(regClass);
    return options;
}

}  // namespace

TEST_F(LiftAsmRegistersToSSATest, ScopingToSgprLeavesVectorOperandsPhysical) {
    createVMovFromSgpr(entry, /*destVgpr=*/0, /*srcSgpr=*/4);

    lift(scopedTo(RegType::S));

    // Only the scalar operand became a value; the vector destination defines
    // nothing and keeps its physical spelling.
    EXPECT_TRUE(anyOfClass(*func, RegType::S));
    EXPECT_FALSE(anyOfClass(*func, RegType::V));
    EXPECT_TRUE(contains(ssaIR(*func), "v0 = \"st.v_mov_b32\"(%1:s)")) << ssaIR(*func);
}

TEST_F(LiftAsmRegistersToSSATest, ScopingToVgprLeavesScalarOperandsPhysical) {
    createVMovFromSgpr(entry, /*destVgpr=*/0, /*srcSgpr=*/4);

    lift(scopedTo(RegType::V));

    EXPECT_TRUE(anyOfClass(*func, RegType::V));
    EXPECT_FALSE(anyOfClass(*func, RegType::S));
    EXPECT_TRUE(contains(ssaIR(*func), "%1:v = \"st.v_mov_b32\"(s4)")) << ssaIR(*func);
}

TEST_F(LiftAsmRegistersToSSATest, AnOutOfScopeClassIsNotAnError) {
    // The distinction that makes scoping work: a class the lifter cannot model is
    // an error, but a class deliberately left physical is simply skipped. Lifting
    // a VGPR-only function for SGPRs is a legal no-op, not a rejection.
    createVAddInBlock(entry, kArch, 2, 0, 1);

    lift(scopedTo(RegType::S));

    EXPECT_EQ(valueCount(), 0u);
    EXPECT_FALSE(anyOfClass(*func, RegType::V));
}

TEST_F(LiftAsmRegistersToSSATest, ATrue16HalfOutsideTheLiftScopeIsNotAnError) {
    // The selector names a vector operand and this lift covers only scalars, so
    // the instruction keeps both registers and its modifier verbatim. Rejecting
    // it would discard the whole function over an operand nothing touches.
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("v", 1, 1));
    mov->addModifier<True16Modifiers>(
        True16Modifiers(HighBitSel::HIGH, HighBitSel::NONE, {HighBitSel::NONE}));

    lift(scopedTo(RegType::S));

    EXPECT_EQ(valueCount(), 0u);
    ASSERT_NE(mov->getModifier<True16Modifiers>(), nullptr);
    EXPECT_EQ(mov->getModifier<True16Modifiers>()->getDst0(), HighBitSel::HIGH);
}

TEST_F(LiftAsmRegistersToSSATest, ATrue16HalfOnAnInScopeOperandStillRejects) {
    // Same scope, but a lifted operand now shares the instruction with the
    // selector, so the lift cannot leave it physical and must decline. This is
    // what keeps the guard scoped rather than simply gone.
    StinkyInstruction* mov = createVMovFromSgpr(entry, /*destVgpr=*/0, /*srcSgpr=*/4);
    mov->addModifier<True16Modifiers>(
        True16Modifiers(HighBitSel::NONE, HighBitSel::NONE, {HighBitSel::LOW}));

    const std::string error = liftError(scopedTo(RegType::S));
    EXPECT_TRUE(contains(error, "True16 half operands")) << error;
    EXPECT_TRUE(contains(error, "src0 is in the lift scope")) << error;
}

TEST_F(LiftAsmRegistersToSSATest, AClassTheLifterCannotModelIsStillAnError) {
    // Scoping selects among the classes lifting can model; it does not widen them.
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("a", 1, 1));

    LiftAsmRegistersToSSAOptions options;
    options.classes = RegClassSet::all().add(RegType::A);

    EXPECT_TRUE(contains(liftError(options), "register class 'a' is not lifted yet"));
}

TEST_F(LiftAsmRegistersToSSATest, AnEmptyScopeIsRejected) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    LiftAsmRegistersToSSAOptions options;
    options.classes = {};

    EXPECT_TRUE(contains(liftError(options), "no register classes to lift"));
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(LiftAsmRegistersToSSATest, ScopeIsRecordedOnTheArena) {
    createVMovFromSgpr(entry, /*destVgpr=*/0, /*srcSgpr=*/4);

    lift(scopedTo(RegType::S));

    // Every walker that steps operands beside AttachedSSA reads this, because the
    // shape fingerprint cannot distinguish two scopes over one program.
    EXPECT_EQ(func->ssaArena().liftedClasses(), RegClassSet::only(RegType::S));
    EXPECT_EQ(func->ssaArena().liftedClasses().toString(), "s");

    // Clearing drops the claim, so an unlifted function does not appear scoped.
    func->clearAttachedSSA();
    EXPECT_EQ(func->ssaArena().liftedClasses(), RegClassSet::all());
}

// ---------------------------------------------------------------------------
// Control flow: block-argument placement and dominator-tree renaming
// ---------------------------------------------------------------------------

namespace {

class LiftCfgTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
    }

    void lift() {
        Expected<LiftAttachedSSAResult> result = liftAsmRegistersToAttachedSSA(*func);
        ASSERT_TRUE(result.hasValue()) << result.getError();
        const AttachedSSAVerificationResult verification = verifyAttachedSSA(*func);
        ASSERT_TRUE(verification.ok()) << verification.toString();
    }

    std::unique_ptr<Function> func;
};

}  // namespace

TEST_F(LiftCfgTest, DiamondPlacesOneArgumentAtTheJoin) {
    BasicBlock* entry = func->createBasicBlock("entry");
    setFunctionArch(*func, kArch);
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkyInstruction* leftDef = createVAddInBlock(left, kArch, 5, 20, 21);
    StinkyInstruction* rightDef = createVAddInBlock(right, kArch, 5, 22, 23);
    StinkyInstruction* use = createVAddInBlock(join, kArch, 6, 5, 5);

    lift();

    const SSABlockArgument* arg = vgprArgumentFor(*join, 5);
    ASSERT_NE(arg, nullptr);
    EXPECT_EQ(mergeArgumentCount(*join), 1u);
    ASSERT_EQ(arg->incoming.size(), 2u);
    EXPECT_EQ(incomingValueFrom(*arg, left), ssaDefinedValue(*leftDef));
    EXPECT_EQ(incomingValueFrom(*arg, right), ssaDefinedValue(*rightDef));
    EXPECT_EQ(ssaSourceValue(*use, 0), arg->value);
}

TEST_F(LiftCfgTest, ValueDefinedInADominatorNeedsNoArgument) {
    BasicBlock* entry = func->createBasicBlock("entry");
    setFunctionArch(*func, kArch);
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkyInstruction* def = createVAddInBlock(entry, kArch, 5, 20, 21);
    StinkyInstruction* use = createVAddInBlock(join, kArch, 6, 5, 5);

    lift();

    EXPECT_EQ(mergeArgumentCount(*join), 0u);
    EXPECT_EQ(ssaSourceValue(*use, 0), ssaDefinedValue(*def));
}

TEST_F(LiftCfgTest, DefinedButNeverReadNeedsNoArgument) {
    DeadRegCfg cfg = buildDeadRegCfg(*func, kArch);

    lift();

    // v0 merges at C but is never read, so pruning places no argument at all.
    EXPECT_EQ(mergeArgumentCount(*func), 0u);
    EXPECT_NE(ssaDefinedValue(*cfg.aDef), nullptr);
}

TEST_F(LiftCfgTest, IteratedDominanceFrontierPlacesArgumentsAtBothJoins) {
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);

    lift();

    const SSABlockArgument* gArg = vgprArgumentFor(*cfg.G, 0);
    const SSABlockArgument* hArg = vgprArgumentFor(*cfg.H, 0);
    ASSERT_NE(gArg, nullptr);
    ASSERT_NE(hArg, nullptr);

    EXPECT_EQ(ssaSourceValue(*cfg.gUse, 0), gArg->value);
    EXPECT_EQ(ssaSourceValue(*cfg.hUse, 0), hArg->value);

    // H merges the value coming through G with the one from entry via D.
    ASSERT_EQ(hArg->incoming.size(), 2u);
    EXPECT_EQ(incomingValueFrom(*hArg, cfg.D), ssaDefinedValue(*cfg.entryDef));
    EXPECT_EQ(incomingValueFrom(*hArg, cfg.G), gArg->value);
}

TEST_F(LiftCfgTest, LastDefinitionInABlockWins) {
    RedefSameBlockCfg cfg = buildRedefSameBlockCfg(*func, kArch);

    lift();

    const SSABlockArgument* arg = vgprArgumentFor(*cfg.C, 0);
    ASSERT_NE(arg, nullptr);
    ASSERT_EQ(arg->incoming.size(), 2u);
    EXPECT_EQ(incomingValueFrom(*arg, cfg.A), ssaDefinedValue(*cfg.aDef2));
    EXPECT_NE(incomingValueFrom(*arg, cfg.A), ssaDefinedValue(*cfg.aDef1));
    EXPECT_EQ(incomingValueFrom(*arg, cfg.B), ssaDefinedValue(*cfg.bDef));
}

TEST_F(LiftCfgTest, ChainOfDiamondsPlacesOneArgumentPerJoin) {
    ChainOfDiamondsCfg cfg = buildChainOfDiamondsCfg(*func, kArch);

    lift();

    ASSERT_NE(vgprArgumentFor(*cfg.C, 0), nullptr);
    ASSERT_NE(vgprArgumentFor(*cfg.F, 0), nullptr);
    ASSERT_NE(vgprArgumentFor(*cfg.I, 0), nullptr);
    EXPECT_EQ(ssaSourceValue(*cfg.iUse, 0), vgprArgumentFor(*cfg.I, 0)->value);
}

TEST_F(LiftCfgTest, LoopHeaderArgumentsMergeEntryAndBackEdge) {
    NestedLoopCfg cfg = buildNestedLoopCfg(*func, kArch);

    lift();

    const SSABlockArgument* aArg = vgprArgumentFor(*cfg.A, 0);
    const SSABlockArgument* bArg = vgprArgumentFor(*cfg.B, 0);
    ASSERT_NE(aArg, nullptr);
    ASSERT_NE(bArg, nullptr);

    // Outer header merges the entry definition with the value from the latch.
    ASSERT_EQ(aArg->incoming.size(), 2u);
    EXPECT_EQ(incomingValueFrom(*aArg, cfg.entry), ssaDefinedValue(*cfg.entryDef));
    EXPECT_NE(incomingValueFrom(*aArg, cfg.D), nullptr);

    // Inner header merges the outer header's value with the inner latch.
    ASSERT_EQ(bArg->incoming.size(), 2u);
    EXPECT_EQ(incomingValueFrom(*bArg, cfg.A), aArg->value);
    EXPECT_EQ(incomingValueFrom(*bArg, cfg.C), ssaDefinedValue(*cfg.cDef));
    EXPECT_EQ(ssaSourceValue(*cfg.bUse, 0), bArg->value);
}

TEST_F(LiftCfgTest, SelfLoopProducesASelfReferentialArgument) {
    SelfLoopJoinCfg cfg = buildSelfLoopJoinCfg(*func, kArch);

    lift();

    const SSABlockArgument* arg = vgprArgumentFor(*cfg.C, 0);
    ASSERT_NE(arg, nullptr);
    ASSERT_EQ(arg->incoming.size(), 3u);

    // The self edge carries the argument's own value: nothing in C redefines v0.
    EXPECT_EQ(incomingValueFrom(*arg, cfg.C), arg->value);
    EXPECT_EQ(ssaSourceValue(*cfg.cUse, 0), arg->value);
    EXPECT_EQ(ssaSourceValue(*cfg.dUse, 0), arg->value);
}

TEST_F(LiftCfgTest, IrreducibleCfgProducesMutuallyReferentialArguments) {
    IrreducibleCfg cfg = buildIrreducibleCfg(*func, kArch);

    lift();

    const SSABlockArgument* cArg = vgprArgumentFor(*cfg.C, 0);
    const SSABlockArgument* dArg = vgprArgumentFor(*cfg.D, 0);
    const SSABlockArgument* eArg = vgprArgumentFor(*cfg.E, 0);
    ASSERT_NE(cArg, nullptr);
    ASSERT_NE(dArg, nullptr);
    ASSERT_NE(eArg, nullptr);

    // C and D feed each other around the irreducible cycle.
    EXPECT_EQ(incomingValueFrom(*cArg, cfg.A), ssaDefinedValue(*cfg.aDef));
    EXPECT_EQ(incomingValueFrom(*cArg, cfg.D), dArg->value);
    EXPECT_EQ(incomingValueFrom(*dArg, cfg.B), ssaDefinedValue(*cfg.bDef));
    EXPECT_EQ(incomingValueFrom(*dArg, cfg.C), cArg->value);
    EXPECT_EQ(ssaSourceValue(*cfg.eUse, 0), eArg->value);
}

TEST_F(LiftCfgTest, MultipleRegistersMergeAtOneJoin) {
    MultiRegJoinCfg cfg = buildMultiRegJoinCfg(*func, kArch);

    lift();

    ASSERT_NE(vgprArgumentFor(*cfg.C, 0), nullptr);
    ASSERT_NE(vgprArgumentFor(*cfg.C, 1), nullptr);
    EXPECT_EQ(mergeArgumentCount(*cfg.C), 2u);
}

TEST_F(LiftCfgTest, PartialRedefinitionOfAWideRegisterOnlyMergesThatDword) {
    WideRegPartialRedefCfg cfg = buildWideRegPartialRedefCfg(*func, kArch);

    lift();

    // Only v0 is redefined, so only v0 merges; v1 to v3 flow from the entry.
    ASSERT_NE(vgprArgumentFor(*cfg.G, 0), nullptr);
    ASSERT_NE(vgprArgumentFor(*cfg.H, 0), nullptr);
    EXPECT_EQ(mergeArgumentCount(*cfg.G), 1u);
    EXPECT_EQ(mergeArgumentCount(*cfg.H), 1u);

    EXPECT_EQ(ssaSourceValue(*cfg.gUse, 0), vgprArgumentFor(*cfg.G, 0)->value);
    // v2 still comes straight from the wide entry load.
    EXPECT_EQ(ssaSourceValue(*cfg.gUse, 1), ssaDefinedValue(*cfg.entryWideDef, /*unit=*/2));
    EXPECT_EQ(ssaSourceValue(*cfg.hUse, 1), ssaDefinedValue(*cfg.entryWideDef, /*unit=*/1));
}

TEST_F(LiftCfgTest, RepeatedLiftsOfACfgProduceIdenticalSSA) {
    buildIteratedDFCfg(*func, kArch);

    AsmPrinterOptions options;
    options.ssaForm = true;

    lift();
    const std::string first = toString(*func, options);
    lift();

    EXPECT_EQ(toString(*func, options), first);
}

TEST_F(LiftCfgTest, DuplicatePredecessorEdgesFillEverySlot) {
    BasicBlock* entry = func->createBasicBlock("entry");
    setFunctionArch(*func, kArch);
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, join);
    // The same edge twice, as a branch whose target is also the fallthrough.
    func->addEdge(left, join);
    func->addEdge(left, join);

    createVAddInBlock(entry, kArch, 5, 20, 21);
    StinkyInstruction* leftDef = createVAddInBlock(left, kArch, 5, 22, 23);
    StinkyInstruction* use = createVAddInBlock(join, kArch, 6, 5, 5);

    lift();

    const SSABlockArgument* arg = vgprArgumentFor(*join, 5);
    ASSERT_NE(arg, nullptr);
    // One slot per predecessor edge, so the duplicated edge gets two.
    ASSERT_EQ(arg->incoming.size(), 3u);
    EXPECT_EQ(incomingValuesFrom(*arg, left),
              (std::vector<StinkySSAValue*>{ssaDefinedValue(*leftDef), ssaDefinedValue(*leftDef)}));
    EXPECT_EQ(ssaSourceValue(*use, 0), arg->value);
}

// ---------------------------------------------------------------------------
// Pass wrapper
// ---------------------------------------------------------------------------

namespace {

class LiftAsmRegistersToSSAPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    void runPass(const LiftAsmRegistersToSSAOptions& options = {}) {
        auto pass = createLiftAsmRegistersToSSAPass(options);
        pass->run(*func, passCtx, am);
    }

    /// Runs the pass with remarks on and returns what it wrote to stderr.
    std::string runPassCapturingRemarks(bool remarksEnabled = true) {
        passCtx.setRemarksEnabled(remarksEnabled);
        std::ostringstream captured;
        std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
        runPass();
        std::cerr.rdbuf(previous);
        return captured.str();
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
    PassContext passCtx;
    AnalysisManager am;
};

}  // namespace

TEST_F(LiftAsmRegistersToSSAPassTest, HasNameAndStableID) {
    auto first = createLiftAsmRegistersToSSAPass();
    auto second = createLiftAsmRegistersToSSAPass();

    ASSERT_NE(first, nullptr);
    EXPECT_STREQ(first->getName(), "Lift Asm Registers to SSA");
    EXPECT_EQ(first->getPassID(), second->getPassID());
}

TEST_F(LiftAsmRegistersToSSAPassTest, AttachesVerifiedSSAOnSuccess) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    runPass();

    ASSERT_TRUE(func->hasAttachedSSA());
    EXPECT_EQ(func->ssaArena().valueCount(), 3u);
    const AttachedSSAVerificationResult verification = verifyAttachedSSA(*func);
    EXPECT_TRUE(verification.ok()) << verification.toString();
}

TEST_F(LiftAsmRegistersToSSAPassTest, RunsThroughThePassManager) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.addPass(createLiftAsmRegistersToSSAPass());
    pm.run(*func);

    // Attached SSA is IR, not analysis-manager state, so it survives the pass
    // manager's post-pass invalidation and a later consumer can still read it.
    ASSERT_TRUE(func->hasAttachedSSA());
    EXPECT_TRUE(verifyAttachedSSA(*func).ok());
}

TEST_F(LiftAsmRegistersToSSAPassTest, DoesNotRewritePhysicalOperands) {
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 8, 4, 5);
    const std::string before = toString(*func);

    runPass();

    ASSERT_TRUE(func->hasAttachedSSA());
    EXPECT_EQ(toString(*func), before);
}

TEST_F(LiftAsmRegistersToSSAPassTest, UnsupportedFunctionIsLeftWithoutSSA) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("orphan");

    const std::string remarks = runPassCapturingRemarks();

    EXPECT_FALSE(func->hasAttachedSSA());
    // The reason is reported rather than dropped, so the pipeline learns why
    // there is no SSA instead of just finding none.
    EXPECT_TRUE(contains(remarks, "^orphan is unreachable from the entry")) << remarks;
}

TEST_F(LiftAsmRegistersToSSAPassTest, FailureClearsAnEarlierLift) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    runPass();
    ASSERT_TRUE(func->hasAttachedSSA());

    // Keeping the earlier SSA would hand a consumer values describing a function
    // that has since changed.
    func->createBasicBlock("orphan");
    runPass();

    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(LiftAsmRegistersToSSAPassTest, RerunRebuildsEquivalentSSA) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 0);

    AsmPrinterOptions options;
    options.ssaForm = true;

    runPass();
    ASSERT_TRUE(func->hasAttachedSSA());
    const std::string first = toString(*func, options);

    runPass();
    ASSERT_TRUE(func->hasAttachedSSA());
    EXPECT_EQ(toString(*func, options), first);
}

TEST_F(LiftAsmRegistersToSSAPassTest, RefusesToRunWhenBlockFilteringExcludesABlock) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::byLabels({"somewhere_else"}));

    const std::string remarks = runPassCapturingRemarks();

    EXPECT_FALSE(func->hasAttachedSSA());
    EXPECT_TRUE(contains(remarks, "basic-block filtering excludes")) << remarks;
}

TEST_F(LiftAsmRegistersToSSAPassTest, RunsWhenBlockFilteringIncludesEveryBlock) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::byLabels({"entry"}));

    runPass();

    EXPECT_TRUE(func->hasAttachedSSA());
}

TEST_F(LiftAsmRegistersToSSAPassTest, ForwardsOptionsToTheLifter) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setRemarksEnabled(true);

    LiftAsmRegistersToSSAOptions options;
    options.allowInferredLiveIns = false;
    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    runPass(options);
    std::cerr.rdbuf(previous);

    EXPECT_FALSE(func->hasAttachedSSA());
    EXPECT_TRUE(contains(captured.str(), "no reaching definition")) << captured.str();
}

TEST_F(LiftAsmRegistersToSSAPassTest, ReportsWhyAFunctionWasNotLifted) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("orphan");

    const std::string remarks = runPassCapturingRemarks();

    EXPECT_TRUE(contains(remarks, "missed: LiftAsmRegistersToSSA")) << remarks;
    EXPECT_TRUE(contains(remarks, "^orphan is unreachable from the entry")) << remarks;
}

TEST_F(LiftAsmRegistersToSSAPassTest, ReportsWhatWasLifted) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    const std::string remarks = runPassCapturingRemarks();

    EXPECT_TRUE(contains(remarks, "remark: LiftAsmRegistersToSSA")) << remarks;
    EXPECT_TRUE(contains(remarks, "@kernel: lifted 3 SSA value(s) and 2 block argument(s)"))
        << remarks;
}

TEST_F(LiftAsmRegistersToSSAPassTest, StaysQuietWhenRemarksAreDisabled) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("second");

    const std::string remarks = runPassCapturingRemarks(/*remarksEnabled=*/false);

    EXPECT_TRUE(remarks.empty()) << remarks;
}

TEST_F(LiftAsmRegistersToSSAPassTest, LiftsAFunctionThatStillCarriesDefUseChains) {
    // Only analysis PHIs are rejected, because only they are visible in the
    // instruction stream. Stale chains on a straight-line function are invisible
    // to lifting, which is why the pipeline is expected to clear them first.
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 1);
    buildUseDefChain(*func, /*clearExisting=*/false);

    runPass();

    ASSERT_TRUE(func->hasAttachedSSA());
    EXPECT_TRUE(verifyAttachedSSA(*func).ok());
}

TEST_F(LiftAsmRegistersToSSAPassTest, RefusesAFunctionThatStillCarriesAnalysisPhis) {
    // The pass does not clean up after other analyses: it reads the function
    // without modifying it, so leftover PHIs are reported and nothing attaches.
    // RemoveDefUseAnalysisPass is what clears them.
    Function cfgFunc("cfg");
    buildIteratedDFCfg(cfgFunc, kArch);
    buildUseDefChain(cfgFunc, /*clearExisting=*/false);

    const auto countAnalysisPhis = [](const Function& function) {
        size_t phis = 0;
        for (const BasicBlock& bb : function) {
            for (const IRBase& ir : bb) {
                const auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (inst != nullptr && inst->getUnifiedOpcode() == GFX::PHI) ++phis;
            }
        }
        return phis;
    };
    const size_t analysisPhis = countAnalysisPhis(cfgFunc);
    ASSERT_GT(analysisPhis, 0u);

    passCtx.setRemarksEnabled(true);
    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createLiftAsmRegistersToSSAPass()->run(cfgFunc, passCtx, am);
    std::cerr.rdbuf(previous);

    EXPECT_FALSE(cfgFunc.hasAttachedSSA());
    EXPECT_TRUE(contains(captured.str(), "analysis PHIs must be removed")) << captured.str();
    // The stream is left exactly as it was, PHIs included.
    EXPECT_EQ(countAnalysisPhis(cfgFunc), analysisPhis);
}

TEST_F(LiftAsmRegistersToSSAPassTest, PreservesCFGAnalyses) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    auto pass = createLiftAsmRegistersToSSAPass();
    const PreservedAnalyses preserved = pass->run(*func, passCtx, am);

    EXPECT_TRUE(preserved.isPreserved<DominanceAnalysis>());
    EXPECT_TRUE(preserved.isPreserved<BBIndexAnalysis>());
    EXPECT_TRUE(preserved.isPreserved<LoopAnalysis>());
}
