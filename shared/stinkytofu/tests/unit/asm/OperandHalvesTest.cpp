// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// forEachVgprHalf indexes True16 half selectors by PRINTED operand position, the way rocisa
// builds them and StinkyAsmEmitter prints them. Every case checks the helper against the
// emitted text, so the two conventions cannot drift apart again (#7844 counted VGPRs only,
// #10052 made true16 generic by printed position).

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/OperandHalves.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"

using namespace stinkytofu;

namespace {

using Half = std::pair<unsigned, HighBitSel>;

class OperandHalvesTest : public ::testing::Test {
   protected:
    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
        gemmConfig.arch = {12, 5, 0};
        func.setGemmTileConfig(gemmConfig);
        bb = func.createBasicBlock("entry");
        builder = std::make_unique<AsmIRBuilder>(*bb, arch);
    }

    StinkyInstruction* create(const std::string& mnemonic) {
        const HwInstDesc* desc = getMCIDByIsaOp(getMnemonicToIsaOpcode(mnemonic, arch), arch);
        EXPECT_NE(desc, nullptr) << mnemonic;
        return desc ? builder->create(desc) : nullptr;
    }

    static std::vector<Half> halves(const StinkyInstruction& inst, OperandSide side) {
        std::vector<Half> out;
        forEachVgprHalf(inst, side, [&](unsigned idx, HighBitSel half) { out.emplace_back(idx, half); });
        return out;
    }

    static std::string emitted(const StinkyInstruction& inst) {
        StinkyAsmEmitter emitter;
        return emitter.emit(inst);
    }

    Function func;
    BasicBlock* bb = nullptr;
    GemmTileConfig gemmConfig;
    GfxArchID arch;
    std::unique_ptr<AsmIRBuilder> builder;
};

// v_add_f16 v1.l, s0, v2.h: the SGPR takes printed slot 0, so v2 is slot 1 (HIGH), not slot 0.
TEST_F(OperandHalvesTest, SgprBeforeVgprSourceKeepsItsPrintedSlot) {
    StinkyInstruction* inst = create("v_add_f16");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));
    inst->addModifier<True16Modifiers>(
        True16Modifiers(HighBitSel::LOW, HighBitSel::NONE, {HighBitSel::NONE, HighBitSel::HIGH}));

    EXPECT_EQ(halves(*inst, OperandSide::Src), (std::vector<Half>{{2, HighBitSel::HIGH}}));
    EXPECT_EQ(halves(*inst, OperandSide::Dst), (std::vector<Half>{{1, HighBitSel::LOW}}));
    const std::string asmText = emitted(*inst);
    EXPECT_NE(asmText.find("v1.l"), std::string::npos) << asmText;
    EXPECT_NE(asmText.find("v2.h"), std::string::npos) << asmText;
}

// A literal constant takes a printed slot too.
TEST_F(OperandHalvesTest, ConstantBeforeVgprSourceKeepsItsPrintedSlot) {
    StinkyInstruction* inst = create("v_add_f16");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister(1.0));
    inst->addSrcReg(StinkyRegister("v", 2, 1));
    inst->addModifier<True16Modifiers>(
        True16Modifiers(HighBitSel::HIGH, HighBitSel::NONE, {HighBitSel::NONE, HighBitSel::LOW}));

    EXPECT_EQ(halves(*inst, OperandSide::Src), (std::vector<Half>{{2, HighBitSel::LOW}}));
    EXPECT_EQ(halves(*inst, OperandSide::Dst), (std::vector<Half>{{1, HighBitSel::HIGH}}));
    const std::string asmText = emitted(*inst);
    EXPECT_NE(asmText.find("v2.l"), std::string::npos) << asmText;
}

// v_fma_f16 v1.l, v2.l, s3, v4.h: two VGPR sources around an SGPR keep their own halves.
TEST_F(OperandHalvesTest, SgprBetweenVgprSourcesKeepsBothHalves) {
    StinkyInstruction* inst = create("v_fma_f16");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));
    inst->addSrcReg(StinkyRegister("s", 3, 1));
    inst->addSrcReg(StinkyRegister("v", 4, 1));
    inst->addModifier<True16Modifiers>(True16Modifiers(
        HighBitSel::LOW, HighBitSel::NONE, {HighBitSel::LOW, HighBitSel::NONE, HighBitSel::HIGH}));

    EXPECT_EQ(halves(*inst, OperandSide::Src),
              (std::vector<Half>{{2, HighBitSel::LOW}, {4, HighBitSel::HIGH}}));
    const std::string asmText = emitted(*inst);
    EXPECT_NE(asmText.find("v2.l"), std::string::npos) << asmText;
    EXPECT_NE(asmText.find("v4.h"), std::string::npos) << asmText;
    EXPECT_EQ(asmText.find("s3."), std::string::npos) << asmText;
}

// A pseudo destination is not printed, so the VGPR after it is still printed destination 0.
TEST_F(OperandHalvesTest, PseudoDestinationTakesNoSlot) {
    StinkyInstruction* inst = create("v_mov_b32");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister(RegType::LDS, 0, 1));
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    inst->addModifier<True16Modifiers>(
        True16Modifiers(HighBitSel::HIGH, HighBitSel::NONE, {HighBitSel::NONE}));

    EXPECT_EQ(halves(*inst, OperandSide::Dst), (std::vector<Half>{{0, HighBitSel::HIGH}}));
    EXPECT_EQ(halves(*inst, OperandSide::Src), (std::vector<Half>{{1, HighBitSel::NONE}}));
}

// The read-modify-write duplicate the IR appends after the printed sources is visited as a
// whole-register read (no selector exists for it).
TEST_F(OperandHalvesTest, ReadModifyWriteDuplicateIsVisitedWhole) {
    StinkyInstruction* inst = create("v_fmac_f16");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));
    inst->addSrcReg(StinkyRegister("v", 3, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));  // RMW duplicate of the destination
    inst->addModifier<True16Modifiers>(
        True16Modifiers(HighBitSel::LOW, HighBitSel::NONE, {HighBitSel::LOW, HighBitSel::HIGH}));

    EXPECT_EQ(halves(*inst, OperandSide::Src),
              (std::vector<Half>{{2, HighBitSel::LOW}, {3, HighBitSel::HIGH}, {1, HighBitSel::NONE}}));
}

// A multi-VGPR operand reports every VGPR with the operand's half.
TEST_F(OperandHalvesTest, RangesExpandPerVgpr) {
    StinkyInstruction* inst = create("v_mov_b32");
    ASSERT_NE(inst, nullptr);
    inst->addDestReg(StinkyRegister("v", 4, 2));
    inst->addSrcReg(StinkyRegister("v", 8, 2));

    EXPECT_EQ(halves(*inst, OperandSide::Dst),
              (std::vector<Half>{{4, HighBitSel::NONE}, {5, HighBitSel::NONE}}));
    EXPECT_EQ(halves(*inst, OperandSide::Src),
              (std::vector<Half>{{8, HighBitSel::NONE}, {9, HighBitSel::NONE}}));
}

}  // namespace
