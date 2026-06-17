/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * Unit tests: instruction byte costing (VALU promotion + literal extra), shared
 * with AccumulateInstructionSizePass via InstructionSizeCosting.
 * ************************************************************************ */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/transforms/asm/InstructionSizeCosting.hpp"

using namespace stinkytofu;

namespace {
StinkyRegister litInt(int32_t v) {
    StinkyRegister r;
    r.dataType = StinkyRegister::Type::LiteralInt;
    r.literalInt = v;
    return r;
}

StinkyRegister litStr(const char* s) {
    StinkyRegister r;
    r.dataType = StinkyRegister::Type::LiteralString;
    r.literalValue.assign(s);
    return r;
}
}  // namespace

class InstructionSizeCostingTest : public ::testing::Test {
   protected:
    GfxArchID arch{};
    std::unique_ptr<Function> func;
    BasicBlock* bb{};

    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
        func = std::make_unique<Function>("instruction_size_test");
        bb = func->createBasicBlock("entry");
    }

    AsmIRBuilder makeBuilder() {
        return AsmIRBuilder(*bb, arch);
    }
};

// ---------------------------------------------------------------------------
// .align / label alignment padding (paddingBytesForCodeAlignment)
// ---------------------------------------------------------------------------

TEST(InstructionSizeCosting_Padding, Align16_FromOffset8_Uses8Bytes) {
    // Matches two 4B NOPs in disassembly: (16 - 8) % 16 = 8
    EXPECT_EQ(paddingBytesForCodeAlignment(8, 16), 8);
    EXPECT_EQ(paddingBytesForCodeAlignment(0, 16), 0);
    EXPECT_EQ(paddingBytesForCodeAlignment(16, 16), 0);
    EXPECT_EQ(paddingBytesForCodeAlignment(0, 1), 0);
    EXPECT_EQ(paddingBytesForCodeAlignment(4, 0), 0);
}

// ---------------------------------------------------------------------------
// VOP promotion (getEffectiveBaseSizeInBytes)
// ---------------------------------------------------------------------------

TEST_F(InstructionSizeCostingTest, VCvtBf16_NoVop3pModifier_LowVgpr_Stays4ByteBase) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f32_bf16, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 199, 1));
    inst->addSrcReg(StinkyRegister("v", 6, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 4);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, VCvtBf16_OpSel0_ModifierForces8Bytes_EvenIfLowVgpr) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f32_bf16, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 199, 1));
    inst->addSrcReg(StinkyRegister("v", 6, 1));
    inst->addModifier(VOP3PModifiers({0}, {}, {}));

    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(getLiteralExtraBytes(*inst), 0);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCvtBf16_OpSel10_ForcesVop3_8Bytes) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f32_bf16, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 199, 1));
    inst->addSrcReg(StinkyRegister("v", 254, 1));
    inst->addModifier(VOP3PModifiers({1, 0}, {}, {}));

    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCvtBf16_Src255_Mod256_Logical255_PromotesTo8Bytes) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f32_bf16, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 199, 1));
    inst->addSrcReg(StinkyRegister("v", 255, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCvtF16F32_OpSel10_ForcesVop3_EvenIfLowVgpr) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f16_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));
    inst->addModifier(VOP3PModifiers({1, 0}, {}, {}));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCvtF16F32_V127_Stays4ByteBase) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f16_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 127, 1));
    inst->addSrcReg(StinkyRegister("v", 127, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 4);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, VCvtF16F32_V128_PromotesTo8Bytes) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f16_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 128, 1));
    inst->addSrcReg(StinkyRegister("v", 128, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCvtF16F32_Src256_Mod256_Logical0_Stays4ByteBase) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f16_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 256, 1));
    inst->addSrcReg(StinkyRegister("v", 256, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 4);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, VCvtF16F32_Src383_Mod256_Logical127_Stays4ByteBase) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f16_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 383, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 4);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, VCvtF16F32_Src384_Mod256_Logical128_PromotesTo8Bytes) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f16_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 384, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCvtF32F16_V132_PromotesTo8Bytes) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f32_f16, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 132, 1));
    inst->addSrcReg(StinkyRegister("v", 132, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCvtF32F16_V12_Stays4ByteBase) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f32_f16, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 12, 1));
    inst->addSrcReg(StinkyRegister("v", 12, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 4);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, VCvtF16F32_Dst128_Src127_OnlySrcCounts_Stays4ByteBase) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f16_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 128, 1));
    inst->addSrcReg(StinkyRegister("v", 127, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 4);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, VCvtF16F32_Dst127_Src128_OnlySrcCounts_PromotesTo8Bytes) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cvt_f16_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 127, 1));
    inst->addSrcReg(StinkyRegister("v", 128, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCndmask_LastSrcVcc_Stays4ByteBase) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cndmask_b32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));
    inst->addSrcReg(StinkyRegister(RegType::VCC, 0, 1));

    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 4);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, VCndmask_LastSrcNotVcc_PromotesTo8Bytes) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cndmask_b32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    inst->addSrcReg(StinkyRegister("v", 2, 1));
    inst->addSrcReg(StinkyRegister("v", 3, 1));

    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VAddCoCi_LastSrcVcc_4LastNotVcc_8) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_add_co_ci_u32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* i = b.create(d);
    i->addDestReg(StinkyRegister("v", 0, 1));
    i->addDestReg(StinkyRegister(RegType::VCC, 0, 1));
    i->addSrcReg(StinkyRegister("v", 1, 1));
    i->addSrcReg(StinkyRegister("v", 2, 1));
    i->addSrcReg(StinkyRegister(RegType::VCC, 0, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*i), 4);

    StinkyInstruction* j = b.create(d);
    j->addDestReg(StinkyRegister("v", 0, 1));
    j->addDestReg(StinkyRegister(RegType::VCC, 0, 1));
    j->addSrcReg(StinkyRegister("v", 1, 1));
    j->addSrcReg(StinkyRegister("v", 2, 1));
    j->addSrcReg(StinkyRegister("v", 3, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*j), 8);
}

TEST_F(InstructionSizeCostingTest, VAddF32_Src1NotVgpr_PromotesTo8) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_add_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    StinkyRegister c;
    c.dataType = StinkyRegister::Type::LiteralDouble;
    c.literalDouble = 1.0;  // not VGPR
    inst->addSrcReg(c);

    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(getLiteralExtraBytes(*inst), 0);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

TEST_F(InstructionSizeCostingTest, VCmp_DestVcc_Base4) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cmp_lt_u32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister(RegType::VCC, 0, 1));
    inst->addSrcReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, VCmp_DestNotVcc_Promotes8) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_cmp_lt_u32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister(RegType::S, 0, 1));
    inst->addSrcReg(StinkyRegister("v", 0, 1));
    inst->addSrcReg(StinkyRegister("v", 1, 1));
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
}

// Gfx1250: v_mul_lo_u32 uses VOP3_2SRC (HwInstDesc::encoding = 64 bits).
// hardwareEncodingBytes is encoding/8 when encoding > 0; else 4 (see
// InstructionSizeCosting.hpp).
TEST_F(InstructionSizeCostingTest, VMulLoU32_TableEncoding64_HardwareEncodingBytes8) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_mul_lo_u32, arch);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->encoding, 64u);
    EXPECT_EQ(d->microcode, MicrocodeFormat::MC_VOP3);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("v", 4, 1));
    inst->addSrcReg(StinkyRegister("s", 19, 1));
    inst->addSrcReg(StinkyRegister("v", 4, 1));
    EXPECT_EQ(hardwareEncodingBytes(*inst), 8);
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 8);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 8);
}

// ENC_VOP3PX3 / v_wmma_scale16_* : 128-bit (16 B) instruction word; MCID must
// not size as 192-bit.
TEST_F(InstructionSizeCostingTest, VWmmaScale16_MC_VOP3PX3_Total16Bytes_NoLiteralTail) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_wmma_scale16_f32_16x16x128_f8f6f4, arch);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->microcode, MicrocodeFormat::MC_VOP3PX3);
    EXPECT_EQ(d->encoding, 128u);
    StinkyInstruction* inst = b.create(d);
    EXPECT_EQ(hardwareEncodingBytes(*inst), 16);
    EXPECT_EQ(getEffectiveBaseSizeInBytes(*inst), 16);
    EXPECT_EQ(getLiteralExtraBytes(*inst), 0);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 16);
}

// ---------------------------------------------------------------------------
// Literal / fixed format (getLiteralExtraBytes)
// ---------------------------------------------------------------------------

TEST_F(InstructionSizeCostingTest, SMemLoad_LiteralExtraZero) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_load_dword, arch);
    ASSERT_NE(d, nullptr);
    (void)b.create(d);
    StinkyInstruction& inst = getStinkyInst(bb->begin());
    EXPECT_EQ(getLiteralExtraBytes(inst), 0);
}

TEST_F(InstructionSizeCostingTest, SMovkI32_Simm16Inline_Minus128_NoLiteralExtra) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_movk_i32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister(RegType::S, 8, 1));
    inst->addSrcReg(litInt(-128));
    EXPECT_EQ(hardwareEncodingBytes(*inst), 4);
    EXPECT_EQ(getLiteralExtraBytes(*inst), 0);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, SMovkI32_SopkFormat_NoLiteralExtra_LargeImmediate) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_movk_i32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister(RegType::S, 8, 1));
    inst->addSrcReg(litInt(70000));
    // SOPK is modeled as a fixed 32-bit encoding (simm16 in-word); no
    // literal-pool add-on.
    EXPECT_EQ(getLiteralExtraBytes(*inst), 0);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, SMovkI32_HexFF80_InlineNoLiteralExtra) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_movk_i32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister(RegType::S, 8, 1));
    inst->addSrcReg(litStr("0xff80"));
    EXPECT_EQ(getLiteralExtraBytes(*inst), 0);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, BufferLoadB32_MUBUF_Extra0_Total12) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::buffer_load_b32, arch);
    ASSERT_NE(d, nullptr);
    (void)b.create(d);
    StinkyInstruction& inst = getStinkyInst(bb->begin());
    EXPECT_EQ(hardwareEncodingBytes(inst), 12);
    EXPECT_EQ(getLiteralExtraBytes(inst), 0);
    EXPECT_EQ(totalInstructionEncodingBytes(inst, nullptr, 0, nullptr), 12);
}

TEST_F(InstructionSizeCostingTest, SWaitcnt_Sopp_Extra0) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_waitcnt, arch);
    ASSERT_NE(d, nullptr);
    (void)b.create(d);
    StinkyInstruction& inst = getStinkyInst(bb->begin());
    EXPECT_EQ(getLiteralExtraBytes(inst), 0);
}

TEST_F(InstructionSizeCostingTest, SMovB32_LiteralInt100_Plus4) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_mov_b32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(litInt(100));

    EXPECT_EQ(getLiteralExtraBytes(*inst), 4);
    EXPECT_EQ(totalInstructionEncodingBytes(*inst), hardwareEncodingBytes(*inst) + 4);
}

TEST_F(InstructionSizeCostingTest, SMovB32_ShortInt42_NoLiteralExtra) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_mov_b32, arch);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(litInt(42));

    EXPECT_EQ(getLiteralExtraBytes(*inst), 0);
}

TEST_F(InstructionSizeCostingTest, BufferOOB_String_Plus4) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_mov_b32, arch);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(litStr("BufferOOB"));

    EXPECT_EQ(getLiteralExtraBytes(*inst), 4);
}

TEST_F(InstructionSizeCostingTest, LabelString_AddrFromMap) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_mov_b32, arch);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(litStr("label_foo"));

    std::unordered_map<std::string, int64_t> m;
    m["label_foo"] = 100;
    EXPECT_EQ(getLiteralExtraBytes(*inst, &m, 0, nullptr), 4);

    std::unordered_map<std::string, int64_t> m2;
    m2["label_foo"] = 8;
    EXPECT_EQ(getLiteralExtraBytes(*inst, &m2, 0, nullptr), 0);
}

// VALU *_f32: hex `0x........` is float32 bits — same literal-extra as decimal
// `LiteralDouble`.
TEST_F(InstructionSizeCostingTest, VMulF32_Hex40800000_MatchesLiteralDouble4_0) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::v_mul_f32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* instHex = b.create(d);
    instHex->addDestReg(StinkyRegister("v", 14, 1));
    instHex->addSrcReg(litStr("0x40800000"));
    instHex->addSrcReg(StinkyRegister("v", 14, 1));

    StinkyInstruction* instFloat = b.create(d);
    instFloat->addDestReg(StinkyRegister("v", 14, 1));
    StinkyRegister c;
    c.dataType = StinkyRegister::Type::LiteralDouble;
    c.literalDouble = 4.0;
    instFloat->addSrcReg(c);
    instFloat->addSrcReg(StinkyRegister("v", 14, 1));

    EXPECT_EQ(getLiteralExtraBytes(*instHex), 0);
    EXPECT_EQ(getLiteralExtraBytes(*instFloat), 0);
    EXPECT_EQ(getLiteralExtraBytes(*instHex), getLiteralExtraBytes(*instFloat));
}

// SALU: same hex token keeps integer-style non-short literal (+4), not
// float-bit reinterpret.
TEST_F(InstructionSizeCostingTest, SMovB32_Hex40800000_StillPlus4_NotValuF32Rule) {
    auto b = makeBuilder();
    const HwInstDesc* d = getMCIDByUOp(GFX::s_mov_b32, arch);
    ASSERT_NE(d, nullptr);
    StinkyInstruction* inst = b.create(d);
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(litStr("0x40800000"));

    EXPECT_EQ(getLiteralExtraBytes(*inst), 4);
}
