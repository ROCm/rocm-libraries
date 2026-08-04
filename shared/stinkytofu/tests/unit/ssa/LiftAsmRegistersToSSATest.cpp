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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>

#include "CanonicalSSATestUtils.hpp"
#include "PhiTestFixtures.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/serialization/ssa/CanonicalSSAPrinter.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/ssa/DumpCanonicalSSAPass.hpp"
#include "stinkytofu/transforms/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class LiftAsmRegistersToSSATest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
    }

    /// Lifts and requires success, returning the graph.
    CanonicalSSA lift(const LiftAsmRegistersToSSAOptions& options = {}) {
        Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func, options);
        EXPECT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
        if (!result.hasValue()) return CanonicalSSA{};
        return std::move(*result);
    }

    /// Lifts and requires failure, returning the diagnostic.
    std::string liftError(const LiftAsmRegistersToSSAOptions& options = {}) {
        Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func, options);
        EXPECT_TRUE(result.hasError());
        return result.hasError() ? result.getError() : std::string{};
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
};

}  // namespace

TEST_F(LiftAsmRegistersToSSATest, EmptyFunctionYieldsEmptyGraph) {
    Function empty("empty");
    Expected<CanonicalSSA> result = liftAsmRegistersToSSA(empty);
    ASSERT_TRUE(result.hasValue()) << result.getError();
    EXPECT_TRUE(result->empty());
}

TEST_F(LiftAsmRegistersToSSATest, EmptyBlockYieldsEmptyGraph) {
    EXPECT_TRUE(lift().empty());
}

TEST_F(LiftAsmRegistersToSSATest, StraightLineProducesVerifiedSSA) {
    createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);

    const CanonicalSSA ssa = lift();
    EXPECT_EQ(ssa.valueCount(), 3u);
    EXPECT_EQ(ssa.value(1).kind, SSAValueKind::LiveIn);
    EXPECT_EQ(ssa.value(1).origin.idx, 0u);
    EXPECT_EQ(ssa.value(2).kind, SSAValueKind::LiveIn);
    EXPECT_EQ(ssa.value(2).origin.idx, 1u);
    EXPECT_EQ(ssa.value(3).kind, SSAValueKind::InstructionDef);
    EXPECT_EQ(ssa.value(3).origin.idx, 2u);
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

TEST_F(LiftAsmRegistersToSSATest, LiftedDumpIsExact) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 0);

    const CanonicalSSA ssa = lift();
    EXPECT_EQ(canonicalSSAToString(*func, ssa),
              "ssa.func @kernel {\n"
              "  initial_values:\n"
              "    %1:v = livein { origin = v0 }\n"
              "    %2:v = livein { origin = v1 }\n"
              "  ^entry:\n"
              "    %3:v = \"st.v_add_f32\"(src0 = [%1:v], src1 = [%2:v]) "
              "{ inst = #0, origin = [v2] }\n"
              "      // physical: v2 = \"st.v_add_f32\"(v0, v1)\n"
              "    %4:v = \"st.v_add_f32\"(src0 = [%3:v], src1 = [%1:v]) "
              "{ inst = #1, origin = [v3] }\n"
              "      // physical: v3 = \"st.v_add_f32\"(v2, v0)\n"
              "}\n");
}

TEST_F(LiftAsmRegistersToSSATest, RepeatedDefinitionsOfOneRegisterBecomeDistinctValues) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 2);

    const CanonicalSSA ssa = lift();
    // v0, v1 live-ins, then three definitions; the last reads the second v2.
    ASSERT_EQ(ssa.valueCount(), 5u);
    EXPECT_EQ(ssa.value(3).origin.idx, 2u);
    EXPECT_EQ(ssa.value(4).origin.idx, 2u);
    EXPECT_NE(ssa.value(3).id, ssa.value(4).id);

    const std::string text = canonicalSSAToString(*func, ssa);
    EXPECT_TRUE(contains(text, "%5:v = \"st.v_add_f32\"(src0 = [%4:v], src1 = [%4:v])")) << text;
}

TEST_F(LiftAsmRegistersToSSATest, OneValueUsedTwiceRecordsTwoUses) {
    createVAddInBlock(entry, kArch, 2, 0, 0);

    const CanonicalSSA ssa = lift();
    ASSERT_EQ(ssa.valueCount(), 2u);
    const SSAValue& liveIn = ssa.value(1);
    ASSERT_EQ(liveIn.uses.size(), 2u);
    EXPECT_EQ(liveIn.uses[0].operand, 0u);
    EXPECT_EQ(liveIn.uses[1].operand, 1u);
}

TEST_F(LiftAsmRegistersToSSATest, OneLiveInIsSharedByEveryReadOfThatUnit) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 0, 1);

    const CanonicalSSA ssa = lift();
    EXPECT_EQ(ssa.value(1).uses.size(), 2u);
    EXPECT_EQ(ssa.value(2).uses.size(), 2u);
    // Two live-ins plus two definitions, not four live-ins.
    EXPECT_EQ(ssa.valueCount(), 4u);
}

TEST_F(LiftAsmRegistersToSSATest, ReadModifyWriteReadsTheOldValueAndDefinesANewOne) {
    // v2 = v_add_f32 v2, v1 reads the incoming v2 and defines a new one.
    createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/2, /*src1=*/1);

    const CanonicalSSA ssa = lift();
    // Live-ins v1 and v2 are created first, in register order, then the result.
    ASSERT_EQ(ssa.valueCount(), 3u);
    const SSAValueID incoming = 2;
    EXPECT_EQ(ssa.value(incoming).kind, SSAValueKind::LiveIn);
    EXPECT_EQ(ssa.value(incoming).origin.idx, 2u);
    ASSERT_EQ(ssa.value(incoming).uses.size(), 1u);
    EXPECT_EQ(ssa.value(incoming).uses[0].operand, 0u);

    const SSAValueID defined = 3;
    EXPECT_EQ(ssa.value(defined).kind, SSAValueKind::InstructionDef);
    EXPECT_EQ(ssa.value(defined).origin.idx, 2u);
    EXPECT_TRUE(ssa.value(defined).uses.empty());
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

TEST_F(LiftAsmRegistersToSSATest, MultiDwordRangeBindsOneValuePerDword) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);

    const CanonicalSSA ssa = lift();
    ASSERT_EQ(ssa.valueCount(), 5u);
    for (unsigned unit = 0; unit < 4; ++unit) {
        const SSAValue& value = ssa.value(2 + unit);
        EXPECT_EQ(value.kind, SSAValueKind::InstructionDef);
        EXPECT_EQ(value.origin.idx, 4u + unit);
        EXPECT_EQ(value.definingUnit, unit);
    }
}

TEST_F(LiftAsmRegistersToSSATest, PartialUseOfAWideDefinitionKeepsUnitIdentity) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    createDSWriteInBlock(entry, kArch, /*addr=*/0, /*data=*/4);

    const CanonicalSSA ssa = lift();
    const std::string text = canonicalSSAToString(*func, ssa);
    EXPECT_TRUE(contains(text, "%2:v, %3:v, %4:v, %5:v = \"st.ds_load_b128\"")) << text;
    EXPECT_TRUE(contains(text, "\"st.ds_store_b64\"(src0 = [%1:v], src1 = [%2:v, %3:v])")) << text;
    // v6 and v7 are defined but never read, so they gain no uses.
    EXPECT_TRUE(ssa.value(4).uses.empty());
    EXPECT_TRUE(ssa.value(5).uses.empty());
}

TEST_F(LiftAsmRegistersToSSATest, LiteralOperandsBindNothing) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 2, 1));
    mov->addSrcReg(StinkyRegister(7));

    const CanonicalSSA ssa = lift();
    ASSERT_EQ(ssa.valueCount(), 1u);
    const SSAInstructionInfo* info = ssa.findInstructionInfo(*mov);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->sources.size(), 1u);
    EXPECT_TRUE(info->sources[0].units.empty());
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

TEST_F(LiftAsmRegistersToSSATest, SpecialRegistersAreNotLifted) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* cmp = builder.create(getMCIDByUOp(GFX::v_cmp_eq_u32, kArch));
    cmp->addDestReg(StinkyRegister::getSCCRegister());
    cmp->addSrcReg(StinkyRegister("v", 0, 1));
    cmp->addSrcReg(StinkyRegister("v", 1, 1));

    const CanonicalSSA ssa = lift();
    const SSAInstructionInfo* info = ssa.findInstructionInfo(*cmp);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->destinations.size(), 1u);
    EXPECT_TRUE(info->destinations[0].units.empty());
    EXPECT_EQ(ssa.valueCount(), 2u);
}

TEST_F(LiftAsmRegistersToSSATest, InstructionsWithoutAllocatableOperandsAreStillBound) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* nop = builder.create(getMCIDByUOp(GFX::s_nop, kArch));

    const CanonicalSSA ssa = lift();
    EXPECT_NE(ssa.findInstructionInfo(*nop), nullptr);
    EXPECT_FALSE(contains(canonicalSSAToString(*func, ssa), "unmapped"));
}

TEST_F(LiftAsmRegistersToSSATest, LabelsAreSkippedButStillConsumeAnIndex) {
    AsmIRBuilder builder(*entry, kArch);
    builder.createLabel("top");
    createVAddInBlock(entry, kArch, 2, 0, 1);

    const CanonicalSSA ssa = lift();
    EXPECT_TRUE(contains(canonicalSSAToString(*func, ssa), "{ inst = #1, origin = [v2] }"));
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
    Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func, options);
    ASSERT_TRUE(result.hasValue()) << result.getError();

    EXPECT_EQ(result->valueCount(), 2u);
    for (const SSAValue& value : result->values())
        EXPECT_EQ(value.kind, SSAValueKind::InstructionDef);
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

    const CanonicalSSA ssa = lift();
    ASSERT_EQ(ssa.valueCount(), 2u);
    EXPECT_EQ(ssa.value(1).kind, SSAValueKind::LiveIn);
    EXPECT_EQ(ssa.value(1).origin.type, RegType::S);
    EXPECT_EQ(ssa.value(1).origin.idx, 4u);
    EXPECT_EQ(ssa.value(2).origin.type, RegType::V);
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

TEST_F(LiftAsmRegistersToSSATest, ScalarAndVectorRegistersWithTheSameIndexAreDistinct) {
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 4, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    const CanonicalSSA ssa = lift();
    // s4 is a live-in and v4 is defined here; the shared index must not merge
    // them, because the register key carries the class.
    ASSERT_EQ(ssa.valueCount(), 2u);
    EXPECT_EQ(ssa.value(1).origin.type, RegType::S);
    EXPECT_EQ(ssa.value(2).origin.type, RegType::V);
    EXPECT_EQ(ssa.value(1).origin.idx, ssa.value(2).origin.idx);
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

TEST_F(LiftAsmRegistersToSSATest, LiftsWideScalarRangesPerDword) {
    // tensor_load_to_lds reads a 4-SGPR base and an 8-SGPR descriptor.
    StinkyInstruction* load = createTensorLoadInBlock(entry, kArch, /*src0=*/8, /*src1=*/16);

    const CanonicalSSA ssa = lift();
    const SSAInstructionInfo* info = ssa.findInstructionInfo(*load);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->sources.size(), 2u);
    EXPECT_EQ(info->sources[0].units.size(), 4u);
    EXPECT_EQ(info->sources[1].units.size(), 8u);

    for (size_t unit = 0; unit < 4; ++unit) {
        const SSAValue& value = ssa.value(info->sources[0].units[unit]);
        EXPECT_EQ(value.origin.type, RegType::S);
        EXPECT_EQ(value.origin.idx, 8u + unit);
    }
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
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

    const std::string error = liftError();
    EXPECT_EQ(error,
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

TEST_F(LiftAsmRegistersToSSATest, RepeatedLiftsProduceIdenticalGraphs) {
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createDSWriteInBlock(entry, kArch, 0, 4);
    createVAddInBlock(entry, kArch, 8, 4, 5);

    CanonicalSSAPrinterOptions options;
    options.printUses = true;

    const CanonicalSSA first = lift();
    const CanonicalSSA second = lift();
    EXPECT_EQ(canonicalSSAToString(*func, first, options),
              canonicalSSAToString(*func, second, options));
}

TEST_F(LiftAsmRegistersToSSATest, ResultCanBeSeededIntoTheAnalysisCache) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func);
    ASSERT_TRUE(result.hasValue()) << result.getError();

    AnalysisManager manager;
    registerAllAnalyses(manager);
    manager.insertResult<CanonicalSSAAnalysis>(std::move(result));

    const auto* cached = manager.getCachedResult<CanonicalSSAAnalysis>();
    ASSERT_NE(cached, nullptr);
    ASSERT_TRUE(cached->hasValue()) << cached->getError();
    EXPECT_TRUE(verifyCanonicalSSA(*func, **cached).ok());
}

TEST_F(LiftAsmRegistersToSSATest, FailureLeavesNoGraphBehind) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("second");

    Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func);

    // Construction is atomic: an error carries no partial graph, so a caller
    // cannot seed one by mistake.
    ASSERT_TRUE(result.hasError());
    EXPECT_FALSE(result.getError().empty());
}

// ---------------------------------------------------------------------------
// Register ranges
// ---------------------------------------------------------------------------

namespace {

/// Ordered SSA units bound to a source operand.
std::vector<SSAValueID> sourceUnits(const CanonicalSSA& ssa, const StinkyInstruction& instruction,
                                    size_t operand) {
    const SSAInstructionInfo* info = ssa.findInstructionInfo(instruction);
    if (info == nullptr || operand >= info->sources.size()) return {};
    return info->sources[operand].units;
}

std::vector<SSAValueID> destUnits(const CanonicalSSA& ssa, const StinkyInstruction& instruction,
                                  size_t operand) {
    const SSAInstructionInfo* info = ssa.findInstructionInfo(instruction);
    if (info == nullptr || operand >= info->destinations.size()) return {};
    return info->destinations[operand].units;
}

/// Physical origins of the units bound to \p units, for grouping checks.
std::vector<unsigned> originsOf(const CanonicalSSA& ssa, const std::vector<SSAValueID>& units) {
    std::vector<unsigned> origins;
    origins.reserve(units.size());
    for (SSAValueID id : units) origins.push_back(ssa.value(id).origin.idx);
    return origins;
}

}  // namespace

TEST_F(LiftAsmRegistersToSSATest, RangeUnitsKeepOperandOrderAndConsecutiveOrigins) {
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, /*dest=*/8, /*addr=*/0);

    const CanonicalSSA ssa = lift();
    const std::vector<SSAValueID> units = destUnits(ssa, *load, 0);

    ASSERT_EQ(units.size(), 4u);
    EXPECT_EQ(originsOf(ssa, units), (std::vector<unsigned>{8, 9, 10, 11}));
    for (size_t unit = 0; unit < units.size(); ++unit)
        EXPECT_EQ(ssa.value(units[unit]).definingUnit, unit);
}

TEST_F(LiftAsmRegistersToSSATest, OverlappingSourceRangesShareTheOverlappingUnits) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);

    // src0 = v[4:5], src1 = v[5:6]: v5 is read through both operands.
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* store = builder.create(getMCIDByUOp(GFX::ds_store_b64, kArch));
    store->addSrcReg(StinkyRegister("v", 4, 2));
    store->addSrcReg(StinkyRegister("v", 5, 2));

    const CanonicalSSA ssa = lift();
    const std::vector<SSAValueID> first = sourceUnits(ssa, *store, 0);
    const std::vector<SSAValueID> second = sourceUnits(ssa, *store, 1);

    ASSERT_EQ(first.size(), 2u);
    ASSERT_EQ(second.size(), 2u);
    EXPECT_EQ(originsOf(ssa, first), (std::vector<unsigned>{4, 5}));
    EXPECT_EQ(originsOf(ssa, second), (std::vector<unsigned>{5, 6}));
    // One value per physical unit, so the shared v5 appears in both operands.
    EXPECT_EQ(first[1], second[0]);
    EXPECT_EQ(ssa.value(first[1]).uses.size(), 2u);
}

TEST_F(LiftAsmRegistersToSSATest, DisjointRangesStayIndependent) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    createDsReadB128InBlock(entry, kArch, /*dest=*/8, /*addr=*/0);

    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* store = builder.create(getMCIDByUOp(GFX::ds_store_b64, kArch));
    store->addSrcReg(StinkyRegister("v", 4, 2));
    store->addSrcReg(StinkyRegister("v", 8, 2));

    const CanonicalSSA ssa = lift();

    EXPECT_EQ(originsOf(ssa, sourceUnits(ssa, *store, 0)), (std::vector<unsigned>{4, 5}));
    EXPECT_EQ(originsOf(ssa, sourceUnits(ssa, *store, 1)), (std::vector<unsigned>{8, 9}));
}

TEST_F(LiftAsmRegistersToSSATest, PartialRedefinitionOfARangeOnlyReplacesThoseUnits) {
    StinkyInstruction* wide = createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    // Overwrite only v4.
    StinkyInstruction* narrow = createVAddInBlock(entry, kArch, /*dest=*/4, /*src0=*/20, 21);

    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* consumer = builder.create(getMCIDByUOp(GFX::ds_store_b64, kArch));
    consumer->addSrcReg(StinkyRegister("v", 0, 1));
    consumer->addSrcReg(StinkyRegister("v", 4, 4));

    const CanonicalSSA ssa = lift();
    const std::vector<SSAValueID> wideUnits = destUnits(ssa, *wide, 0);
    const std::vector<SSAValueID> consumed = sourceUnits(ssa, *consumer, 1);

    ASSERT_EQ(consumed.size(), 4u);
    // v4 comes from the narrow redefinition; v5 to v7 still come from the load.
    EXPECT_EQ(consumed[0], destUnits(ssa, *narrow, 0).front());
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

    const CanonicalSSA ssa = lift();

    const std::vector<SSAValueID> firstRead = sourceUnits(ssa, *wmmas[0], 2);
    const std::vector<SSAValueID> firstDef = destUnits(ssa, *wmmas[0], 0);
    const std::vector<SSAValueID> secondRead = sourceUnits(ssa, *wmmas[1], 2);
    const std::vector<SSAValueID> secondDef = destUnits(ssa, *wmmas[1], 0);

    ASSERT_EQ(firstRead.size(), 8u);
    ASSERT_EQ(secondDef.size(), 8u);
    for (size_t unit = 0; unit < 8; ++unit) {
        // Each instruction reads the previous value and defines a new one.
        EXPECT_EQ(ssa.value(firstRead[unit]).kind, SSAValueKind::LiveIn);
        EXPECT_NE(firstRead[unit], firstDef[unit]);
        EXPECT_EQ(secondRead[unit], firstDef[unit]);
        EXPECT_NE(secondRead[unit], secondDef[unit]);
        // The tie stays observable: read and written units share one origin.
        EXPECT_EQ(ssa.value(firstDef[unit]).origin.idx, ssa.value(firstRead[unit]).origin.idx);
    }
}

TEST_F(LiftAsmRegistersToSSATest, DestinationOverlappingItsOwnSourceReadsTheOldUnits) {
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);

    // v[4:5] = op(v[5:6]): v5 is both read and written.
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* shifted = builder.create(getMCIDByUOp(GFX::v_lshlrev_b64, kArch));
    shifted->addDestReg(StinkyRegister("v", 4, 2));
    shifted->addSrcReg(StinkyRegister("v", 5, 2));

    const CanonicalSSA ssa = lift();
    const std::vector<SSAValueID> read = sourceUnits(ssa, *shifted, 0);
    const std::vector<SSAValueID> written = destUnits(ssa, *shifted, 0);

    ASSERT_EQ(read.size(), 2u);
    ASSERT_EQ(written.size(), 2u);
    EXPECT_EQ(originsOf(ssa, read), (std::vector<unsigned>{5, 6}));
    EXPECT_EQ(originsOf(ssa, written), (std::vector<unsigned>{4, 5}));
    // The v5 that is read is the incoming value, not the one defined here.
    EXPECT_NE(read[0], written[1]);
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());
}

// ---------------------------------------------------------------------------
// Control flow: PHI placement and dominator-tree renaming
// ---------------------------------------------------------------------------

namespace {

class LiftCfgTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
    }

    CanonicalSSA lift() {
        Expected<CanonicalSSA> result = liftAsmRegistersToSSA(*func);
        EXPECT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
        if (!result.hasValue()) return CanonicalSSA{};

        // Dominance-aware verification is the real check for these CFGs.
        const DominanceInfo dominance = computeDominanceInfo(*func);
        const CanonicalSSAVerificationResult verification =
            verifyCanonicalSSA(*func, *result, dominance);
        EXPECT_TRUE(verification.ok()) << verification.toString();
        return std::move(*result);
    }

    /// PHI for \p vgpr in \p block, or null when none was placed.
    static const SSAPhi* phiFor(const CanonicalSSA& ssa, const BasicBlock& block, unsigned vgpr) {
        for (SSAPhiID id : ssa.phisForBlock(block)) {
            if (ssa.phi(id).origin.idx == vgpr) return &ssa.phi(id);
        }
        return nullptr;
    }

    static size_t phiCountIn(const CanonicalSSA& ssa, const BasicBlock& block) {
        return ssa.phisForBlock(block).size();
    }

    /// SSA value bound to source operand \p operand of \p instruction.
    static SSAValueID sourceValue(const CanonicalSSA& ssa, const StinkyInstruction& instruction,
                                  size_t operand) {
        const SSAInstructionInfo* info = ssa.findInstructionInfo(instruction);
        if (info == nullptr || operand >= info->sources.size()) return kInvalidSSAValueID;
        const std::vector<SSAValueID>& units = info->sources[operand].units;
        return units.empty() ? kInvalidSSAValueID : units.front();
    }

    static SSAValueID definedValue(const CanonicalSSA& ssa, const StinkyInstruction& instruction,
                                   size_t unit = 0) {
        const SSAInstructionInfo* info = ssa.findInstructionInfo(instruction);
        if (info == nullptr || info->destinations.empty()) return kInvalidSSAValueID;
        const std::vector<SSAValueID>& units = info->destinations.front().units;
        return unit < units.size() ? units[unit] : kInvalidSSAValueID;
    }

    std::unique_ptr<Function> func;
};

}  // namespace

TEST_F(LiftCfgTest, DiamondPlacesOnePhiAtTheJoin) {
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

    const CanonicalSSA ssa = lift();

    const SSAPhi* phi = phiFor(ssa, *join, 5);
    ASSERT_NE(phi, nullptr);
    EXPECT_EQ(phiCountIn(ssa, *join), 1u);
    ASSERT_EQ(phi->incoming.size(), 2u);
    EXPECT_EQ(phi->incoming[0].predecessor, left);
    EXPECT_EQ(phi->incoming[0].value, definedValue(ssa, *leftDef));
    EXPECT_EQ(phi->incoming[1].predecessor, right);
    EXPECT_EQ(phi->incoming[1].value, definedValue(ssa, *rightDef));
    EXPECT_EQ(sourceValue(ssa, *use, 0), phi->result);
}

TEST_F(LiftCfgTest, ValueDefinedInADominatorNeedsNoPhi) {
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

    const CanonicalSSA ssa = lift();

    EXPECT_EQ(phiCountIn(ssa, *join), 0u);
    EXPECT_EQ(sourceValue(ssa, *use, 0), definedValue(ssa, *def));
}

TEST_F(LiftCfgTest, DefinedButNeverReadNeedsNoPhi) {
    DeadRegCfg cfg = buildDeadRegCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    // v0 merges at C but is never read, so pruning places no PHI at all.
    EXPECT_EQ(ssa.phiCount(), 0u);
    EXPECT_NE(definedValue(ssa, *cfg.aDef), kInvalidSSAValueID);
}

TEST_F(LiftCfgTest, IteratedDominanceFrontierPlacesPhisAtBothJoins) {
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    const SSAPhi* gPhi = phiFor(ssa, *cfg.G, 0);
    const SSAPhi* hPhi = phiFor(ssa, *cfg.H, 0);
    ASSERT_NE(gPhi, nullptr);
    ASSERT_NE(hPhi, nullptr);

    EXPECT_EQ(sourceValue(ssa, *cfg.gUse, 0), gPhi->result);
    EXPECT_EQ(sourceValue(ssa, *cfg.hUse, 0), hPhi->result);

    // H merges the value coming through G with the one from entry via D.
    ASSERT_EQ(hPhi->incoming.size(), 2u);
    EXPECT_EQ(hPhi->incoming[0].value, definedValue(ssa, *cfg.entryDef));
    EXPECT_EQ(hPhi->incoming[1].value, gPhi->result);
}

TEST_F(LiftCfgTest, LastDefinitionInABlockWins) {
    RedefSameBlockCfg cfg = buildRedefSameBlockCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    const SSAPhi* phi = phiFor(ssa, *cfg.C, 0);
    ASSERT_NE(phi, nullptr);
    ASSERT_EQ(phi->incoming.size(), 2u);
    EXPECT_EQ(phi->incoming[0].value, definedValue(ssa, *cfg.aDef2));
    EXPECT_NE(phi->incoming[0].value, definedValue(ssa, *cfg.aDef1));
    EXPECT_EQ(phi->incoming[1].value, definedValue(ssa, *cfg.bDef));
}

TEST_F(LiftCfgTest, ChainOfDiamondsPlacesOnePhiPerJoin) {
    ChainOfDiamondsCfg cfg = buildChainOfDiamondsCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    ASSERT_NE(phiFor(ssa, *cfg.C, 0), nullptr);
    ASSERT_NE(phiFor(ssa, *cfg.F, 0), nullptr);
    ASSERT_NE(phiFor(ssa, *cfg.I, 0), nullptr);
    EXPECT_EQ(sourceValue(ssa, *cfg.iUse, 0), phiFor(ssa, *cfg.I, 0)->result);
}

TEST_F(LiftCfgTest, LoopHeaderPhisMergeEntryAndBackEdge) {
    NestedLoopCfg cfg = buildNestedLoopCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    const SSAPhi* aPhi = phiFor(ssa, *cfg.A, 0);
    const SSAPhi* bPhi = phiFor(ssa, *cfg.B, 0);
    ASSERT_NE(aPhi, nullptr);
    ASSERT_NE(bPhi, nullptr);

    // Outer header merges the entry definition with the value from the latch.
    ASSERT_EQ(aPhi->incoming.size(), 2u);
    EXPECT_EQ(aPhi->incoming[0].predecessor, cfg.entry);
    EXPECT_EQ(aPhi->incoming[0].value, definedValue(ssa, *cfg.entryDef));
    EXPECT_EQ(aPhi->incoming[1].predecessor, cfg.D);

    // Inner header merges the outer header's value with the inner latch.
    ASSERT_EQ(bPhi->incoming.size(), 2u);
    EXPECT_EQ(bPhi->incoming[0].value, aPhi->result);
    EXPECT_EQ(bPhi->incoming[1].value, definedValue(ssa, *cfg.cDef));
    EXPECT_EQ(sourceValue(ssa, *cfg.bUse, 0), bPhi->result);
}

TEST_F(LiftCfgTest, SelfLoopProducesASelfReferentialPhi) {
    SelfLoopJoinCfg cfg = buildSelfLoopJoinCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    const SSAPhi* phi = phiFor(ssa, *cfg.C, 0);
    ASSERT_NE(phi, nullptr);
    ASSERT_EQ(phi->incoming.size(), 3u);

    // The self edge carries the PHI's own result: nothing in C redefines v0.
    bool sawSelfEdge = false;
    for (const SSAPhiIncoming& incoming : phi->incoming) {
        if (incoming.predecessor != cfg.C) continue;
        sawSelfEdge = true;
        EXPECT_EQ(incoming.value, phi->result);
    }
    EXPECT_TRUE(sawSelfEdge);
    EXPECT_EQ(sourceValue(ssa, *cfg.cUse, 0), phi->result);
    EXPECT_EQ(sourceValue(ssa, *cfg.dUse, 0), phi->result);
}

TEST_F(LiftCfgTest, IrreducibleCfgProducesMutuallyReferentialPhis) {
    IrreducibleCfg cfg = buildIrreducibleCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    const SSAPhi* cPhi = phiFor(ssa, *cfg.C, 0);
    const SSAPhi* dPhi = phiFor(ssa, *cfg.D, 0);
    const SSAPhi* ePhi = phiFor(ssa, *cfg.E, 0);
    ASSERT_NE(cPhi, nullptr);
    ASSERT_NE(dPhi, nullptr);
    ASSERT_NE(ePhi, nullptr);

    // C and D feed each other around the irreducible cycle.
    EXPECT_EQ(cPhi->incoming[0].value, definedValue(ssa, *cfg.aDef));
    EXPECT_EQ(cPhi->incoming[1].value, dPhi->result);
    EXPECT_EQ(dPhi->incoming[0].value, definedValue(ssa, *cfg.bDef));
    EXPECT_EQ(dPhi->incoming[1].value, cPhi->result);
    EXPECT_EQ(sourceValue(ssa, *cfg.eUse, 0), ePhi->result);
}

TEST_F(LiftCfgTest, MultipleRegistersMergeAtOneJoin) {
    MultiRegJoinCfg cfg = buildMultiRegJoinCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    ASSERT_NE(phiFor(ssa, *cfg.C, 0), nullptr);
    ASSERT_NE(phiFor(ssa, *cfg.C, 1), nullptr);
    EXPECT_EQ(phiCountIn(ssa, *cfg.C), 2u);
}

TEST_F(LiftCfgTest, PartialRedefinitionOfAWideRegisterOnlyMergesThatDword) {
    WideRegPartialRedefCfg cfg = buildWideRegPartialRedefCfg(*func, kArch);

    const CanonicalSSA ssa = lift();

    // Only v0 is redefined, so only v0 merges; v1 to v3 flow from the entry.
    ASSERT_NE(phiFor(ssa, *cfg.G, 0), nullptr);
    ASSERT_NE(phiFor(ssa, *cfg.H, 0), nullptr);
    EXPECT_EQ(phiCountIn(ssa, *cfg.G), 1u);
    EXPECT_EQ(phiCountIn(ssa, *cfg.H), 1u);

    EXPECT_EQ(sourceValue(ssa, *cfg.gUse, 0), phiFor(ssa, *cfg.G, 0)->result);
    // v2 still comes straight from the wide entry load.
    EXPECT_EQ(sourceValue(ssa, *cfg.gUse, 1), definedValue(ssa, *cfg.entryWideDef, /*unit=*/2));
    EXPECT_EQ(sourceValue(ssa, *cfg.hUse, 1), definedValue(ssa, *cfg.entryWideDef, /*unit=*/1));
}

TEST_F(LiftCfgTest, RepeatedLiftsOfACfgProduceIdenticalDumps) {
    buildIteratedDFCfg(*func, kArch);

    CanonicalSSAPrinterOptions options;
    options.printUses = true;

    const CanonicalSSA first = lift();
    const CanonicalSSA second = lift();
    EXPECT_EQ(canonicalSSAToString(*func, first, options),
              canonicalSSAToString(*func, second, options));
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

    const CanonicalSSA ssa = lift();

    const SSAPhi* phi = phiFor(ssa, *join, 5);
    ASSERT_NE(phi, nullptr);
    ASSERT_EQ(phi->incoming.size(), 3u);
    for (const SSAPhiIncoming& incoming : phi->incoming) {
        EXPECT_NE(incoming.value, kInvalidSSAValueID);
        if (incoming.predecessor == left) {
            EXPECT_EQ(incoming.value, definedValue(ssa, *leftDef));
        }
    }
    EXPECT_EQ(sourceValue(ssa, *use, 0), phi->result);
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

    /// The graph the pass seeded into the analysis cache, or null when it did
    /// not lift one. Deliberately the cached result: getResult() would lift on
    /// demand and so could not tell whether the pass had done anything.
    const CanonicalSSA* graph(const AnalysisManager& manager) const {
        const auto* cached = manager.getCachedResult<CanonicalSSAAnalysis>();
        return cached != nullptr && cached->hasValue() ? &**cached : nullptr;
    }

    const CanonicalSSA* graph() const {
        return graph(am);
    }

    /// Why the pass declined to lift. Empty when it lifted, "<none>" when it
    /// recorded nothing at all.
    std::string notLiftedReason() const {
        const auto* cached = am.getCachedResult<CanonicalSSAAnalysis>();
        if (cached == nullptr) return "<none>";
        return cached->hasError() ? cached->getError() : std::string{};
    }

    /// Physical instruction stream, used to prove the pass rewrites nothing.
    std::string physicalIR() const {
        std::ostringstream out;
        AsmPrinter printer(out);
        printer.print(*func);
        return out.str();
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

TEST_F(LiftAsmRegistersToSSAPassTest, SeedsAVerifiedGraphOnSuccess) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    runPass();

    const CanonicalSSA* ssa = graph();
    ASSERT_NE(ssa, nullptr) << notLiftedReason();
    EXPECT_EQ(ssa->valueCount(), 3u);
    EXPECT_TRUE(verifyCanonicalSSA(*func, *ssa).ok());
}

TEST_F(LiftAsmRegistersToSSAPassTest, RunsThroughThePassManager) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.addPass(createLiftAsmRegistersToSSAPass());
    pm.run(*func);

    // The pass preserves its own result, so it survives the pass manager's
    // post-pass invalidation and a later consumer can still read it.
    const CanonicalSSA* ssa = graph(pm.getAnalysisManager());
    ASSERT_NE(ssa, nullptr);
    EXPECT_TRUE(verifyCanonicalSSA(*func, *ssa).ok());
}

TEST_F(LiftAsmRegistersToSSAPassTest, DoesNotRewritePhysicalOperands) {
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 8, 4, 5);
    const std::string before = physicalIR();

    runPass();

    ASSERT_NE(graph(), nullptr) << notLiftedReason();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(LiftAsmRegistersToSSAPassTest, UnsupportedFunctionIsLeftWithoutSSA) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("orphan");

    runPass();

    EXPECT_EQ(graph(), nullptr);
    // The reason is recorded rather than dropped, so a consumer asking for the
    // graph learns why there is none instead of just finding nothing.
    EXPECT_TRUE(contains(notLiftedReason(), "^orphan is unreachable from the entry"))
        << notLiftedReason();
}

TEST_F(LiftAsmRegistersToSSAPassTest, FailureReplacesAnEarlierGraph) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    runPass();
    ASSERT_NE(graph(), nullptr);

    // Make the function unsupported, then run again: leaving the earlier result
    // cached would hand a consumer SSA describing a function that has changed.
    func->createBasicBlock("orphan");
    runPass();

    EXPECT_EQ(graph(), nullptr);
    EXPECT_FALSE(notLiftedReason().empty());
}

TEST_F(LiftAsmRegistersToSSAPassTest, RerunRebuildsAnEquivalentGraph) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 0);

    runPass();
    ASSERT_NE(graph(), nullptr) << notLiftedReason();
    const std::string first = canonicalSSAToString(*func, *graph());

    runPass();
    ASSERT_NE(graph(), nullptr);
    EXPECT_EQ(canonicalSSAToString(*func, *graph()), first);
}

TEST_F(LiftAsmRegistersToSSAPassTest, RefusesToRunWhenBlockFilteringExcludesABlock) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::byLabels({"somewhere_else"}));

    runPass();

    EXPECT_EQ(graph(), nullptr);
    EXPECT_TRUE(contains(notLiftedReason(), "basic-block filtering excludes")) << notLiftedReason();
}

TEST_F(LiftAsmRegistersToSSAPassTest, RunsWhenBlockFilteringIncludesEveryBlock) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setBasicBlockFilter(BasicBlockFilterBuilder::byLabels({"entry"}));

    runPass();

    EXPECT_NE(graph(), nullptr) << notLiftedReason();
}

TEST_F(LiftAsmRegistersToSSAPassTest, ForwardsOptionsToTheLifter) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    LiftAsmRegistersToSSAOptions options;
    options.allowInferredLiveIns = false;
    runPass(options);

    EXPECT_EQ(graph(), nullptr);
    EXPECT_TRUE(contains(notLiftedReason(), "no reaching definition")) << notLiftedReason();
}

TEST_F(LiftAsmRegistersToSSAPassTest, ReportsWhyAFunctionWasNotLifted) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("orphan");
    passCtx.setRemarksEnabled(true);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    runPass();
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "missed: LiftAsmRegistersToSSA")) << text;
    EXPECT_TRUE(contains(text, "^orphan is unreachable from the entry")) << text;
}

TEST_F(LiftAsmRegistersToSSAPassTest, ReportsWhatWasLifted) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    passCtx.setRemarksEnabled(true);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    runPass();
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "remark: LiftAsmRegistersToSSA")) << text;
    EXPECT_TRUE(contains(text, "@kernel: lifted 3 SSA value(s) and 0 phi(s)")) << text;
}

TEST_F(LiftAsmRegistersToSSAPassTest, StaysQuietWhenRemarksAreDisabled) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    func->createBasicBlock("second");

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    runPass();
    std::cerr.rdbuf(previous);

    EXPECT_TRUE(captured.str().empty()) << captured.str();
}

TEST_F(LiftAsmRegistersToSSAPassTest, LiftsAFunctionThatStillCarriesDefUseChains) {
    // Only analysis PHIs are rejected, because only they are visible in the
    // instruction stream. Stale chains on a straight-line function are invisible
    // to lifting, which is why the pipeline is expected to clear them first.
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 1);
    buildUseDefChain(*func, /*clearExisting=*/false);

    runPass();

    ASSERT_NE(graph(), nullptr) << notLiftedReason();
    EXPECT_TRUE(verifyCanonicalSSA(*func, *graph()).ok());
}

TEST_F(LiftAsmRegistersToSSAPassTest, RefusesAFunctionThatStillCarriesAnalysisPhis) {
    // The pass does not clean up after other analyses: it reads the function
    // without modifying it, so leftover PHIs are reported and nothing attaches.
    // RemoveDefUseAnalysisPass is what clears them.
    Function cfgFunc("cfg");
    buildIteratedDFCfg(cfgFunc, kArch);
    buildUseDefChain(cfgFunc, /*clearExisting=*/false);

    size_t analysisPhis = 0;
    for (const BasicBlock& bb : cfgFunc) {
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst != nullptr && inst->getUnifiedOpcode() == GFX::PHI) ++analysisPhis;
        }
    }
    ASSERT_GT(analysisPhis, 0u);

    auto pass = createLiftAsmRegistersToSSAPass();
    pass->run(cfgFunc, passCtx, am);

    EXPECT_EQ(graph(), nullptr);
    EXPECT_TRUE(contains(notLiftedReason(), "analysis PHIs must be removed")) << notLiftedReason();
    // The stream is left exactly as it was, PHIs included.
    size_t phisAfter = 0;
    for (const BasicBlock& bb : cfgFunc) {
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst != nullptr && inst->getUnifiedOpcode() == GFX::PHI) ++phisAfter;
        }
    }
    EXPECT_EQ(phisAfter, analysisPhis);
}

TEST_F(LiftAsmRegistersToSSAPassTest, DumpPassPrintsTheAttachedSidecar) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    runPass();
    ASSERT_NE(graph(), nullptr) << notLiftedReason();

    std::ostringstream captured;
    std::streambuf* previous = std::cout.rdbuf(captured.rdbuf());
    createDumpCanonicalSSAPass()->run(*func, passCtx, am);
    std::cout.rdbuf(previous);

    EXPECT_EQ(captured.str(), canonicalSSAToString(*func, *graph()));
}

TEST_F(LiftAsmRegistersToSSAPassTest, DumpPassLeavesTheFunctionUnchanged) {
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 8, 4, 5);
    runPass();
    const std::string before = physicalIR();

    std::ostringstream captured;
    std::streambuf* previous = std::cout.rdbuf(captured.rdbuf());
    const PreservedAnalyses preserved = createDumpCanonicalSSAPass()->run(*func, passCtx, am);
    std::cout.rdbuf(previous);

    EXPECT_EQ(physicalIR(), before);
    EXPECT_NE(graph(), nullptr);
    EXPECT_TRUE(preserved.areAllPreserved());
}

TEST_F(LiftAsmRegistersToSSAPassTest, DumpPassReportsAMissingSidecar) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    std::ostringstream captured;
    std::ostringstream capturedErr;
    std::streambuf* previousOut = std::cout.rdbuf(captured.rdbuf());
    std::streambuf* previousErr = std::cerr.rdbuf(capturedErr.rdbuf());
    createDumpCanonicalSSAPass()->run(*func, passCtx, am);
    std::cout.rdbuf(previousOut);
    std::cerr.rdbuf(previousErr);

    EXPECT_TRUE(captured.str().empty()) << captured.str();
    EXPECT_TRUE(contains(capturedErr.str(), "has no canonical SSA")) << capturedErr.str();
}

TEST_F(LiftAsmRegistersToSSAPassTest, DumpPassCanPrintAPlaceholderInstead) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    DumpCanonicalSSAConfig config;
    config.requireCanonicalSSA = false;

    std::ostringstream captured;
    std::streambuf* previous = std::cout.rdbuf(captured.rdbuf());
    createDumpCanonicalSSAPass(config)->run(*func, passCtx, am);
    std::cout.rdbuf(previous);

    EXPECT_TRUE(contains(captured.str(), "<no canonical SSA attached>")) << captured.str();
}

TEST_F(LiftAsmRegistersToSSAPassTest, DumpPassWritesToAFileWhenAsked) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    runPass();

    const std::string path =
        std::filesystem::temp_directory_path() / "stinkytofu_dump_canonical_ssa_test.ssa";
    DumpCanonicalSSAConfig config;
    config.outputPath = path;
    createDumpCanonicalSSAPass(config)->run(*func, passCtx, am);

    std::ifstream file(path);
    ASSERT_TRUE(file.is_open());
    const std::string contents((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
    file.close();
    std::filesystem::remove(path);

    ASSERT_NE(graph(), nullptr) << notLiftedReason();
    EXPECT_EQ(contents, canonicalSSAToString(*func, *graph()));
}

TEST_F(LiftAsmRegistersToSSAPassTest, DumpPassReportsVerificationFailures) {
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);

    // Seed a deliberately broken graph: the use record mirroring src0 is
    // missing, which the verifier must catch before the dump is trusted.
    CanonicalSSABuilder builder;
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    bindInstruction(builder, *add, {{in0}, {in1}});
    builder.value(in0).uses.clear();
    am.insertResult<CanonicalSSAAnalysis>(Expected<CanonicalSSA>(builder.take()));

    std::ostringstream captured;
    std::streambuf* previous = std::cout.rdbuf(captured.rdbuf());
    createDumpCanonicalSSAPass()->run(*func, passCtx, am);
    std::cout.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "// canonical SSA verification failed:")) << text;
    EXPECT_TRUE(contains(text, "records this slot 0 time(s)")) << text;
    // The dump still follows, so a malformed graph stays inspectable.
    EXPECT_TRUE(contains(text, "ssa.func @kernel")) << text;
}

TEST_F(LiftAsmRegistersToSSAPassTest, PreservesCFGAnalyses) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    auto pass = createLiftAsmRegistersToSSAPass();
    const PreservedAnalyses preserved = pass->run(*func, passCtx, am);

    EXPECT_TRUE(preserved.isPreserved<DominanceAnalysis>());
    EXPECT_TRUE(preserved.isPreserved<BBIndexAnalysis>());
    EXPECT_TRUE(preserved.isPreserved<LoopAnalysis>());
}
