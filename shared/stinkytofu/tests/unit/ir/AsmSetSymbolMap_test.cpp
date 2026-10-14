// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A register operand carries both a symbolic name like `vgprTmp` and a numeric
// index. The IR identifies registers by index, so when the two disagree the
// operand quietly reads the wrong register. resolveSymbolicOperands fixes that,
// resetting each index to whatever its name resolves to.
//
// Names resolve through `.set` directives, and the same name can be redefined
// partway through a function:
//
//     .set vgprTmp, 10
//     v_add_f32 v[vgprTmp], ...    // index 10
//     .set vgprTmp, 20
//     v_add_f32 v[vgprTmp], ...    // index 20
//
// Each operand therefore resolves against the bindings in force where it sits,
// the same way the assembler would read it. That ordering is the main thing
// these tests check, which is why they interleave directives and instructions
// by hand rather than parsing a .stir file.
//
// Two kinds of operand are left alone: virtual registers, which have no real
// index yet, and names with no `.set` in scope, which say nothing about the
// index they are attached to.

#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

void appendSetDirective(BasicBlock* bb, const std::string& symbol, const std::string& value) {
    AsmDirective* directive = IRBase::createIR<AsmDirective>();
    directive->kind = AsmDirectiveKind::SET;
    directive->name = ".set";
    directive->symbol = symbol;
    directive->value = value;
    bb->appendIR(directive);
}

StinkyRegister namedReg(const char* type, uint32_t idx, uint16_t num, const std::string& name) {
    StinkyRegister reg(type, idx, num);
    reg.setSymbolicName(name);
    return reg;
}

// A named operand that is still a template placeholder rather than an index.
StinkyRegister virtualNamedReg(const std::string& name) {
    StinkyRegister reg("v", 0, 1);
    reg.reg.idx |= StinkyRegister::kVirtualBit;
    reg.setSymbolicName(name);
    return reg;
}

// v_add_f32 dest, src0, src1 appended to the end of the block.
StinkyInstruction* addVAdd(BasicBlock* bb, const StinkyRegister& dest, const StinkyRegister& src0,
                           const StinkyRegister& src1) {
    AsmIRBuilder builder(*bb, kArch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_add_f32, kArch));
    inst->addDestReg(dest);
    inst->addSrcReg(src0);
    inst->addSrcReg(src1);
    return inst;
}

// ds_load_b128 dest[4], v0 — a 4-wide destination, for range-width resolution.
StinkyInstruction* addDsLoadB128(BasicBlock* bb, const StinkyRegister& dest) {
    AsmIRBuilder builder(*bb, kArch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::ds_load_b128, kArch));
    inst->addDestReg(dest);
    inst->addSrcReg(StinkyRegister("v", 0, 1));
    return inst;
}

class AsmSetSymbolMapTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    std::unique_ptr<Function> func;
    std::vector<SymbolicOperandFix> fixes;
};

TEST_F(AsmSetSymbolMapTest, RewritesIndexToMatchName) {
    BasicBlock* bb = block("entry");
    appendSetDirective(bb, "vgprTmp", "20");
    StinkyInstruction* add = addVAdd(bb, namedReg("v", 0, 1, "vgprTmp"), vgpr(1), vgpr(2));

    EXPECT_EQ(resolveSymbolicOperands(*func, fixes), 1u);
    EXPECT_EQ(add->getDestRegs()[0].reg.idx, 20u);
    ASSERT_EQ(fixes.size(), 1u);
    EXPECT_EQ(fixes[0].symbol, "vgprTmp");
    EXPECT_EQ(fixes[0].fromIdx, 0u);
    EXPECT_EQ(fixes[0].toIdx, 20u);
}

TEST_F(AsmSetSymbolMapTest, LeavesAlreadyCorrectIndexAlone) {
    BasicBlock* bb = block("entry");
    appendSetDirective(bb, "vgprTmp", "20");
    StinkyInstruction* add = addVAdd(bb, namedReg("v", 20, 1, "vgprTmp"), vgpr(1), vgpr(2));

    // Agreement is not a correction, so it must not be reported as one.
    EXPECT_EQ(resolveSymbolicOperands(*func, fixes), 0u);
    EXPECT_EQ(add->getDestRegs()[0].reg.idx, 20u);
    EXPECT_TRUE(fixes.empty());
}

TEST_F(AsmSetSymbolMapTest, RewritesBothDestinationAndSourceOperands) {
    BasicBlock* bb = block("entry");
    appendSetDirective(bb, "vgprOut", "30");
    appendSetDirective(bb, "vgprIn", "31");
    StinkyInstruction* add =
        addVAdd(bb, namedReg("v", 0, 1, "vgprOut"), namedReg("v", 1, 1, "vgprIn"), vgpr(2));

    EXPECT_EQ(resolveSymbolicOperands(*func, fixes), 2u);
    EXPECT_EQ(add->getDestRegs()[0].reg.idx, 30u);
    EXPECT_EQ(add->getSrcRegs()[0].reg.idx, 31u);
}

TEST_F(AsmSetSymbolMapTest, EachOperandUsesTheBindingInForceWhereItAppears) {
    BasicBlock* bb = block("entry");
    appendSetDirective(bb, "vgprTmp", "10");
    StinkyInstruction* first = addVAdd(bb, namedReg("v", 0, 1, "vgprTmp"), vgpr(1), vgpr(2));
    appendSetDirective(bb, "vgprTmp", "20");
    StinkyInstruction* second = addVAdd(bb, namedReg("v", 0, 1, "vgprTmp"), vgpr(1), vgpr(2));

    // A symbol redefined mid-function reads the way the assembler would, so the
    // same name resolves to a different index at each operand.
    EXPECT_EQ(resolveSymbolicOperands(*func, fixes), 2u);
    EXPECT_EQ(first->getDestRegs()[0].reg.idx, 10u);
    EXPECT_EQ(second->getDestRegs()[0].reg.idx, 20u);
}

TEST_F(AsmSetSymbolMapTest, LeavesUnresolvableNameAlone) {
    BasicBlock* bb = block("entry");
    StinkyInstruction* add = addVAdd(bb, namedReg("v", 7, 1, "vgprNeverDefined"), vgpr(1), vgpr(2));

    // An unresolvable name says nothing about the index.
    EXPECT_EQ(resolveSymbolicOperands(*func, fixes), 0u);
    EXPECT_EQ(add->getDestRegs()[0].reg.idx, 7u);
    EXPECT_TRUE(fixes.empty());
}

TEST_F(AsmSetSymbolMapTest, IgnoresVirtualRegisters) {
    BasicBlock* bb = block("entry");
    appendSetDirective(bb, "vgprTmp", "20");
    StinkyInstruction* add = addVAdd(bb, virtualNamedReg("vgprTmp"), vgpr(1), vgpr(2));

    // A virtual register is not an index yet, so a name cannot contradict it.
    EXPECT_EQ(resolveSymbolicOperands(*func, fixes), 0u);
    EXPECT_TRUE(add->getDestRegs()[0].isVirtualReg());
}

TEST_F(AsmSetSymbolMapTest, ResolvesRangeNameAgainstTheOperandWidth) {
    BasicBlock* bb = block("entry");
    appendSetDirective(bb, "vgprQuad", "16");
    // Resolution is width-sensitive, so the operand's own `num` has to reach
    // resolveNamedIndex: a four-wide name only resolves on a four-wide operand.
    StinkyInstruction* load = addDsLoadB128(bb, namedReg("v", 0, 4, "vgprQuad+0:vgprQuad+3"));

    EXPECT_EQ(resolveSymbolicOperands(*func, fixes), 1u);
    EXPECT_EQ(load->getDestRegs()[0].reg.idx, 16u);
}

}  // namespace
