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

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;

class HwInstDescTest : public ::testing::Test {
   protected:
    GfxArchID arch;

    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
    }

    const HwInstDesc* getDescByMnemonic(const std::string& mnemonic) {
        uint16_t isaOp = getMnemonicToIsaOpcode(mnemonic, arch);
        if (isaOp == GFX::INVALID) return nullptr;
        return getMCIDByIsaOp(isaOp, arch);
    }
};

// ---------------------------------------------------------------------------
// VOP2: v_add_f32 — basic vector ALU, 32-bit encoding, promotable to VOP3
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, VOP2_VAddF32_Microcode) {
    auto* desc = getDescByMnemonic("v_add_f32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_VOP2);
}

TEST_F(HwInstDescTest, VOP2_VAddF32_Unit) {
    auto* desc = getDescByMnemonic("v_add_f32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->unit, ExecUnit::VALU);
}

TEST_F(HwInstDescTest, VOP2_VAddF32_Flags) {
    auto* desc = getDescByMnemonic("v_add_f32");
    ASSERT_NE(desc, nullptr);
    EXPECT_TRUE(desc->has(IF_VALU));
    EXPECT_TRUE(desc->has(IF_Commutative));
    EXPECT_FALSE(desc->has(IF_SALU));
}

TEST_F(HwInstDescTest, VOP2_VAddF32_OperandFields) {
    auto* desc = getDescByMnemonic("v_add_f32");
    ASSERT_NE(desc, nullptr);

    auto fields = desc->operandFields;
    ASSERT_EQ(fields.size(), 3u);

    // D0: {D0, vdst, vgpr, 32}
    EXPECT_TRUE(fields[0].isDest);
    EXPECT_EQ(fields[0].encodeField, EncodeField::vdst);
    EXPECT_EQ(fields[0].fieldType, FieldType::vgpr);
    EXPECT_EQ(fields[0].fieldSizeBits, 32u);

    // S0: {S0, src0, src, 32}
    EXPECT_FALSE(fields[1].isDest);
    EXPECT_EQ(fields[1].encodeField, EncodeField::src0);
    EXPECT_EQ(fields[1].fieldType, FieldType::src);
    EXPECT_EQ(fields[1].fieldSizeBits, 32u);

    // S1: {S1, vsrc1, vgpr, 32}
    EXPECT_FALSE(fields[2].isDest);
    EXPECT_EQ(fields[2].encodeField, EncodeField::vsrc1);
    EXPECT_EQ(fields[2].fieldType, FieldType::vgpr);
    EXPECT_EQ(fields[2].fieldSizeBits, 32u);
}

TEST_F(HwInstDescTest, VOP2_VAddF32_PromotedFormat) {
    auto* desc = getDescByMnemonic("v_add_f32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->promotedFormat, MicrocodeFormat::MC_VOP3);
}

TEST_F(HwInstDescTest, VOP2_VAddF32_PromotedFields) {
    auto* desc = getDescByMnemonic("v_add_f32");
    ASSERT_NE(desc, nullptr);

    auto pf = desc->promotedFields;
    ASSERT_EQ(pf.size(), 3u);

    // In promoted VOP3 encoding, S1 is relaxed from vgpr to src
    EXPECT_EQ(pf[2].encodeField, EncodeField::src1);
    EXPECT_EQ(pf[2].fieldType, FieldType::src);
}

// ---------------------------------------------------------------------------
// VOP3: v_fma_f32 — 64-bit encoding, no promotion
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, VOP3_VFmaF32) {
    auto* desc = getDescByMnemonic("v_fma_f32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_VOP3);
    EXPECT_EQ(desc->unit, ExecUnit::VALU);
    EXPECT_TRUE(desc->has(IF_VALU));
    EXPECT_EQ(desc->promotedFormat, MicrocodeFormat::NONE);
}

// ---------------------------------------------------------------------------
// SOP2: s_add_u32 — scalar ALU, 32-bit encoding
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, SOP2_SAddU32) {
    auto* desc = getDescByMnemonic("s_add_u32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_SOP2);
    EXPECT_EQ(desc->unit, ExecUnit::SALU);
    EXPECT_TRUE(desc->has(IF_SALU));
    EXPECT_FALSE(desc->has(IF_VALU));

    auto fields = desc->operandFields;
    ASSERT_EQ(fields.size(), 3u);

    // D0: {D0, sdst, sdst, 32}
    EXPECT_TRUE(fields[0].isDest);
    EXPECT_EQ(fields[0].encodeField, EncodeField::sdst);
    EXPECT_EQ(fields[0].fieldType, FieldType::sdst);

    // S0: {S0, ssrc0, ssrc, 32}
    EXPECT_EQ(fields[1].encodeField, EncodeField::ssrc0);
    EXPECT_EQ(fields[1].fieldType, FieldType::ssrc);

    // S1: {S1, ssrc1, ssrc, 32}
    EXPECT_EQ(fields[2].encodeField, EncodeField::ssrc1);
    EXPECT_EQ(fields[2].fieldType, FieldType::ssrc);
}

// ---------------------------------------------------------------------------
// MUBUF_LOAD: buffer_load_b32 — buffer memory, 96-bit encoding
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, MUBUF_BufferLoadB32) {
    auto* desc = getDescByMnemonic("buffer_load_b32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_VBUFFER);
    EXPECT_EQ(desc->unit, ExecUnit::BufferMemory);
    EXPECT_TRUE(desc->has(IF_MUBUFLoad));

    auto fields = desc->operandFields;
    ASSERT_EQ(fields.size(), 4u);

    // D0: {D0, vdata, vgpr, 32}
    EXPECT_TRUE(fields[0].isDest);
    EXPECT_EQ(fields[0].encodeField, EncodeField::vdata);
    EXPECT_EQ(fields[0].fieldType, FieldType::vgpr);
    EXPECT_EQ(fields[0].fieldSizeBits, 32u);

    // S1: {S1, rsrc, sreg, 128, RSRC}
    EXPECT_EQ(fields[2].encodeField, EncodeField::rsrc);
    EXPECT_EQ(fields[2].fieldType, FieldType::sreg);
    EXPECT_EQ(fields[2].fieldSizeBits, 128u);
}

// ---------------------------------------------------------------------------
// MUBUF_LOAD per-instruction override: buffer_load_b64 — D0 size 32 -> 64
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, MUBUF_BufferLoadB64_FieldOverride) {
    auto* desc = getDescByMnemonic("buffer_load_b64");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_VBUFFER);

    auto fields = desc->operandFields;
    ASSERT_GE(fields.size(), 4u);

    // D0 size overridden from format default (32) to 64
    EXPECT_TRUE(fields[0].isDest);
    EXPECT_EQ(fields[0].encodeField, EncodeField::vdata);
    EXPECT_EQ(fields[0].fieldSizeBits, 64u);

    // S0 unchanged from format
    EXPECT_EQ(fields[1].encodeField, EncodeField::vaddr);
    EXPECT_EQ(fields[1].fieldSizeBits, 32u);
}

// ---------------------------------------------------------------------------
// FLAT_LOAD: flat_load_b32 — global memory, 96-bit encoding
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, FLAT_FlatLoadB32) {
    auto* desc = getDescByMnemonic("flat_load_b32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_VFLAT);
    EXPECT_EQ(desc->unit, ExecUnit::GlobalMemory);
    EXPECT_TRUE(desc->has(IF_FLATLoad));
}

// ---------------------------------------------------------------------------
// DS: ds_load_b32 — LDS memory, 64-bit encoding
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, DS_DsLoadB32) {
    auto* desc = getDescByMnemonic("ds_load_b32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_VDS);
    EXPECT_EQ(desc->unit, ExecUnit::LDS);
    EXPECT_TRUE(desc->has(IF_DSRead));
}

// ---------------------------------------------------------------------------
// SMEM: s_load_b32 — scalar memory, 64-bit encoding
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, SMRD_SLoadB32) {
    auto* desc = getDescByMnemonic("s_load_b32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_SMEM);
    EXPECT_EQ(desc->unit, ExecUnit::ScalarMemory);
    EXPECT_TRUE(desc->has(IF_SMemLoad));
}

// ---------------------------------------------------------------------------
// WMMA: v_wmma_f32_16x16x16_f16 — matrix unit, VOP3P encoding
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, WMMA_F32_16x16x16_F16) {
    auto* desc = getDescByMnemonic("v_wmma_f32_16x16x16_f16");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_VOP3P);
    EXPECT_EQ(desc->unit, ExecUnit::MatrixUnit);
    EXPECT_TRUE(desc->has(IF_WMMA));

    auto fields = desc->operandFields;
    ASSERT_GE(fields.size(), 3u);

    // D0: 256-bit destination (8 VGPRs)
    EXPECT_TRUE(fields[0].isDest);
    EXPECT_EQ(fields[0].encodeField, EncodeField::vdst);
    EXPECT_EQ(fields[0].fieldSizeBits, 256u);
}

// ---------------------------------------------------------------------------
// SOPP_BRANCH: s_branch — branch unit, label operand
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, SOPP_SBranch) {
    auto* desc = getDescByMnemonic("s_branch");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_SOPP);
    EXPECT_EQ(desc->unit, ExecUnit::BranchUnit);
    EXPECT_TRUE(desc->has(IF_Branch));

    auto fields = desc->operandFields;
    ASSERT_EQ(fields.size(), 1u);
    EXPECT_EQ(fields[0].encodeField, EncodeField::simm16);
    EXPECT_EQ(fields[0].fieldType, FieldType::label);
}

// ---------------------------------------------------------------------------
// VOPC: v_cmp_eq_f32 — compare, promotable to VOP3
// ---------------------------------------------------------------------------
TEST_F(HwInstDescTest, VOPC_VCmpEqF32) {
    auto* desc = getDescByMnemonic("v_cmp_eq_f32");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->microcode, MicrocodeFormat::MC_VOPC);
    EXPECT_EQ(desc->unit, ExecUnit::VALU);
    EXPECT_TRUE(desc->has(IF_VALU));
    EXPECT_EQ(desc->promotedFormat, MicrocodeFormat::MC_VOP3);

    auto fields = desc->operandFields;
    ASSERT_EQ(fields.size(), 3u);

    // D0: vcc (64-bit)
    EXPECT_TRUE(fields[0].isDest);
    EXPECT_EQ(fields[0].encodeField, EncodeField::vdst);
    EXPECT_EQ(fields[0].fieldType, FieldType::vcc);
    EXPECT_EQ(fields[0].fieldSizeBits, 64u);

    // In promoted encoding, D0 becomes sreg
    auto pf = desc->promotedFields;
    ASSERT_EQ(pf.size(), 3u);
    EXPECT_EQ(pf[0].fieldType, FieldType::sreg);
}
