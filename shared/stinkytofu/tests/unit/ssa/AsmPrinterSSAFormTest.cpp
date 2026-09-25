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

#include "AttachedSSATestUtils.hpp"
#include "PhiTestFixtures.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class AsmPrinterSSAFormTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
    }

    BasicBlock* makeEntry() {
        setFunctionArch(*func, kArch);
        return func->createBasicBlock("entry");
    }

    void lift() {
        Expected<LiftAttachedSSAResult> result = liftAsmRegistersToAttachedSSA(*func);
        ASSERT_TRUE(result.hasValue()) << result.getError();
    }

    std::string physical() const {
        return toString(*func);
    }

    std::string ssa() const {
        AsmPrinterOptions options;
        options.ssaForm = true;
        return toString(*func, options);
    }

    std::unique_ptr<Function> func;
};

}  // namespace

TEST_F(AsmPrinterSSAFormTest, DefaultOptionsPrintThePhysicalProgram) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    const std::string beforeLift = physical();

    lift();

    // The toggle is opt-in, so attaching SSA must not change what the default
    // printer emits. Everything downstream of lift still reads physical text.
    EXPECT_EQ(physical(), beforeLift);
    EXPECT_TRUE(contains(physical(), "v2 = \"st.v_add_f32\"(v0, v1)")) << physical();
}

TEST_F(AsmPrinterSSAFormTest, UnliftedFunctionPrintsIdenticallyInBothForms) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);

    // Nothing is attached, so ssaForm has nothing to substitute and falls back
    // to the physical spelling operand by operand.
    EXPECT_EQ(ssa(), physical());
}

TEST_F(AsmPrinterSSAFormTest, StraightLineDumpIsExact) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 0);

    lift();

    EXPECT_EQ(ssa(),
              "st.func @kernel() {\n"
              "  ^entry(%1:v, %2:v):\n"
              "    %3:v = \"st.v_add_f32\"(%1:v, %2:v) "
              "{ issueCycles = 1, latencyCycles = 5 }\n"
              "    %4:v = \"st.v_add_f32\"(%3:v, %1:v) "
              "{ issueCycles = 1, latencyCycles = 5 }\n"
              "}\n");
}

TEST_F(AsmPrinterSSAFormTest, LiveInsAppearAsEntryBlockArguments) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);

    lift();

    EXPECT_TRUE(contains(ssa(), "^entry(%1:v, %2:v):")) << ssa();
    // A live-in arrives without an edge, so it gets no phi line.
    EXPECT_FALSE(contains(ssa(), "= phi(")) << ssa();
}

TEST_F(AsmPrinterSSAFormTest, RepeatedDefinitionsPrintDistinctValues) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 2, 0, 1);

    lift();

    // Both instructions write v2, and the dump has to show two values rather
    // than one register written twice.
    EXPECT_TRUE(contains(ssa(), "%3:v = \"st.v_add_f32\"(%1:v, %2:v)")) << ssa();
    EXPECT_TRUE(contains(ssa(), "%4:v = \"st.v_add_f32\"(%1:v, %2:v)")) << ssa();
}

TEST_F(AsmPrinterSSAFormTest, RangeOperandsAreBracketedPerDword) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    createDSWriteInBlock(entry, kArch, /*addr=*/0, /*data=*/4);

    lift();

    // Brackets keep the four units of one operand distinguishable from four
    // separate operands.
    EXPECT_TRUE(contains(ssa(), "[%2:v, %3:v, %4:v, %5:v] = \"st.ds_load_b128\"(%1:v)")) << ssa();
    EXPECT_TRUE(contains(ssa(), "\"st.ds_store_b64\"(%1:v, [%2:v, %3:v])")) << ssa();
}

TEST_F(AsmPrinterSSAFormTest, PartialRedefinitionKeepsEveryUnitVisible) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    createVAddInBlock(entry, kArch, /*dest=*/4, /*src0=*/0, /*src1=*/0);

    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* store = builder.create(getMCIDByUOp(GFX::ds_store_b128, kArch));
    store->addSrcReg(StinkyRegister("v", 0, 1));
    store->addSrcReg(StinkyRegister("v", 4, 4));

    lift();

    // The wide read mixes the redefined v4 with the three surviving units.
    EXPECT_TRUE(contains(ssa(), "\"st.ds_store_b128\"(%1:v, [%6:v, %3:v, %4:v, %5:v])")) << ssa();
}

TEST_F(AsmPrinterSSAFormTest, NonLiftedOperandsKeepTheirPhysicalSpelling) {
    BasicBlock* entry = makeEntry();
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 2, 1));
    mov->addSrcReg(StinkyRegister(7));

    StinkyInstruction* cmp = builder.create(getMCIDByUOp(GFX::v_cmp_eq_u32, kArch));
    cmp->addDestReg(StinkyRegister::getSCCRegister());
    cmp->addSrcReg(StinkyRegister("v", 0, 1));
    cmp->addSrcReg(StinkyRegister("v", 1, 1));

    lift();

    // Literals and special registers were never lifted, so there is no value to
    // print for them and the physical form is the honest spelling. v0 and v1 are
    // live-ins, so the mov's destination is the third value created.
    EXPECT_TRUE(contains(ssa(), "%3:v = \"st.v_mov_b32\"(7)")) << ssa();
    EXPECT_TRUE(contains(ssa(), "SCC0 = \"st.v_cmp_eq_u32\"(%1:v, %2:v)")) << ssa();
}

TEST_F(AsmPrinterSSAFormTest, ScalarValuesPrintTheirRegisterClass) {
    BasicBlock* entry = makeEntry();
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    lift();

    EXPECT_TRUE(contains(ssa(), "^entry(%1:s):")) << ssa();
    EXPECT_TRUE(contains(ssa(), "%2:v = \"st.v_mov_b32\"(%1:s)")) << ssa();
}

TEST_F(AsmPrinterSSAFormTest, MergePrintsAPhiLineInPredecessorOrder) {
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);
    ASSERT_NE(cfg.G, nullptr);

    lift();

    const SSABlockArgument* arg = vgprArgumentFor(*cfg.G, 0);
    ASSERT_NE(arg, nullptr);

    // Lift fills incoming as its dominator walk reaches each predecessor; the
    // dump orders them by the block's predecessor list so it stays readable and
    // stable if that walk changes.
    const std::string text = ssa();
    ASSERT_EQ(cfg.G->getPredecessors().size(), 2u);
    const std::string expected = "%" + std::to_string(arg->value->valueId()) + ":v = phi(^" +
                                 cfg.G->getPredecessors()[0]->getLabel() + ": ";
    EXPECT_TRUE(contains(text, expected)) << text;
    EXPECT_TRUE(contains(text, ", ^" + cfg.G->getPredecessors()[1]->getLabel() + ": ")) << text;
}

TEST_F(AsmPrinterSSAFormTest, MergeAppearsBothInTheHeaderAndAsAPhiLine) {
    BasicBlock* entry = makeEntry();
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkyInstruction* leftDef = createVAddInBlock(left, kArch, 5, 20, 21);
    StinkyInstruction* rightDef = createVAddInBlock(right, kArch, 5, 22, 23);
    createVAddInBlock(join, kArch, 6, 5, 5);

    lift();

    const SSABlockArgument* arg = vgprArgumentFor(*join, 5);
    ASSERT_NE(arg, nullptr);
    const std::string merged = "%" + std::to_string(arg->value->valueId()) + ":v";
    const std::string fromLeft = "%" + std::to_string(ssaDefinedValue(*leftDef)->valueId()) + ":v";
    const std::string fromRight =
        "%" + std::to_string(ssaDefinedValue(*rightDef)->valueId()) + ":v";

    EXPECT_TRUE(contains(ssa(), "^join(" + merged + "):")) << ssa();
    EXPECT_TRUE(
        contains(ssa(), merged + " = phi(^left: " + fromLeft + ", ^right: " + fromRight + ")"))
        << ssa();
}

TEST_F(AsmPrinterSSAFormTest, UnlabelledBlocksFallBackToPositionalNames) {
    setFunctionArch(*func, kArch);
    BasicBlock* entry = func->createBasicBlock("");
    createVAddInBlock(entry, kArch, 2, 0, 1);

    lift();

    EXPECT_TRUE(contains(ssa(), "^bb0(%1:v, %2:v):")) << ssa();
}

TEST_F(AsmPrinterSSAFormTest, OutputIsByteIdenticalAcrossRepeatedPrints) {
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);
    ASSERT_NE(cfg.entry, nullptr);

    lift();

    EXPECT_EQ(ssa(), ssa());
}

TEST_F(AsmPrinterSSAFormTest, PrintsPhysicalAgainAfterAttachedSSAIsCleared) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    const std::string beforeLift = physical();

    lift();
    ASSERT_TRUE(contains(ssa(), "%3:v"));
    func->clearAttachedSSA();

    // Destruction clears attached SSA, so an ssaForm dump afterwards is just the
    // physical program: there is no stale value identity left to print.
    EXPECT_EQ(ssa(), beforeLift);
}
