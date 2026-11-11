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

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

// =============================================================================
// Read-Write Operand Tests
//
// A destination that also acts as a source must appear in both destRegs and
// srcRegs, and the verifier checks that. Two things make a destination one of
// those, and readsDestination answers for both:
//   - HwInstDesc marks the operand field isReadWrite (RW), a property of the
//     opcode: s_cmov_b32, v_swap_b32, v_cvt_sr_*, the buffer atomics;
//   - the write is narrower than the register and op_sel names the half it
//     lands in, so the other half survives. A property of the instance, which
//     is why these cases carry an op_sel.
//
// Each instruction has:
//   - a "valid" test  : register present in both lists  → verifier passes
//   - an "invalid" test: register missing from srcRegs   → verifier catches it
// =============================================================================

class ReadWriteOperandTest : public ::testing::Test {
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

    /// Build a single-instruction function and run the read-write verifier.
    /// Returns the verifier error string (empty = pass).
    ///
    /// \p opSel, when given, is attached as the instruction's op_sel. A narrow
    /// destination is read-write only where op_sel names the half it writes, so
    /// a case about one of those has to say which half.
    std::string verifyRW(const std::string& mnemonic, const std::vector<StinkyRegister>& destRegs,
                         const std::vector<StinkyRegister>& srcRegs,
                         const std::vector<int>& opSel = {}) {
        Function func("test");
        setFunctionArch(func, arch);
        BasicBlock* bb = func.createBasicBlock("entry");

        const HwInstDesc* desc = getDescByMnemonic(mnemonic);
        if (!desc) return "Unknown mnemonic: " + mnemonic;

        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(desc);
        for (const auto& r : destRegs) inst->addDestReg(r);
        for (const auto& r : srcRegs) inst->addSrcReg(r);
        if (!opSel.empty()) inst->addModifier<VOP3PModifiers>(VOP3PModifiers(opSel));

        AsmVerifierConfig config;
        config.checkRegisterWidths = false;
        config.checkReadWriteOperands = true;
        return validateStinkyIR(func, config);
    }

    /// Same, with every category enabled, over a whole list of instructions.
    std::string verifyAll(const std::vector<std::tuple<std::string, std::vector<StinkyRegister>,
                                                       std::vector<StinkyRegister>>>& insts) {
        Function func("test");
        setFunctionArch(func, arch);
        BasicBlock* bb = func.createBasicBlock("entry");

        AsmIRBuilder builder(*bb, arch);
        for (const auto& [mnemonic, destRegs, srcRegs] : insts) {
            const HwInstDesc* desc = getDescByMnemonic(mnemonic);
            if (!desc) return "Unknown mnemonic: " + mnemonic;
            StinkyInstruction* inst = builder.create(desc);
            for (const auto& r : destRegs) inst->addDestReg(r);
            for (const auto& r : srcRegs) inst->addSrcReg(r);
        }

        AsmVerifierConfig config;
        config.checkRegisterWidths = true;
        config.checkReadWriteOperands = true;
        return validateStinkyIR(func, config);
    }
};

// ---------------------------------------------------------------------------
// m0 is encoded in the scalar operand space, so an sdst field accepts it.
// ---------------------------------------------------------------------------
TEST_F(ReadWriteOperandTest, SMovB32_M0DestIsAcceptedAsScalar) {
    std::string error = verifyAll({{"s_mov_b32", {StinkyRegister("m", 0, 1)}, {sgpr(2)}}});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, SMovB32_VgprDestIsStillRejected) {
    std::string error = verifyAll({{"s_mov_b32", {vgpr(1)}, {sgpr(2)}}});
    EXPECT_NE(error.find("Register width validation failed"), std::string::npos) << error;
}

// ---------------------------------------------------------------------------
// s_cmov_b32  —  D0 is RW: if (SCC) s0 = s1; else s0 = s0
// HW fields: {D0, sdst, sdst, 32, RW}, {S0, ssrc0, ssrc, 32}
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, SCMovB32_Valid) {
    std::string error = verifyRW("s_cmov_b32", {sgpr(1)}, {sgpr(2), sgpr(1)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, SCMovB32_MissingDstInSrc) {
    std::string error = verifyRW("s_cmov_b32", {sgpr(1)}, {sgpr(2)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// s_cmov_b64  —  D0 is RW (64-bit)
// HW fields: {D0, sdst, sdst, 64, RW}, {S0, ssrc0, ssrc, 64}
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, SCMovB64_Valid) {
    std::string error = verifyRW("s_cmov_b64", {sgpr(2, 2)}, {sgpr(4, 2), sgpr(2, 2)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, SCMovB64_MissingDstInSrc) {
    std::string error = verifyRW("s_cmov_b64", {sgpr(2, 2)}, {sgpr(4, 2)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// v_fmac_f32  —  D0 is RW: dst = src0 * src1 + dst
// HW fields: {D0, vdst, vgpr, 32, RW}, {S0, src0, src, 32}, {S1, vsrc1, vgpr, 32}
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, VFmacF32_Valid) {
    std::string error = verifyRW("v_fmac_f32", {vgpr(0)}, {vgpr(1), vgpr(2), vgpr(0)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, VFmacF32_MissingDstInSrc) {
    std::string error = verifyRW("v_fmac_f32", {vgpr(0)}, {vgpr(1), vgpr(2)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// v_cvt_sr_fp8_f32  —  D0 is RW: stochastic round, packs into existing reg
// HW fields: {D0, vdst, vgpr, 8, RW}, {S0, src0, src, 32}, {S1, src1, src, 32}
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, VCvtSrFp8F32_Valid) {
    std::string error = verifyRW("v_cvt_sr_fp8_f32", {vgpr(0)}, {vgpr(1), vgpr(2), vgpr(0)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, VCvtSrFp8F32_MissingDstInSrc) {
    std::string error = verifyRW("v_cvt_sr_fp8_f32", {vgpr(0)}, {vgpr(1), vgpr(2)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// v_cvt_pk_fp8_f32  —  read-write by instance, not by opcode. The table gives a
// 16-bit D0 and op_sel gives the half it writes, so the other half keeps what
// was already there and a four-value pack is two of these writing one register.
// HW fields: {D0, vdst, vgpr, 16}, {S0, src0, src, 32}, {S1, src1, src, 32}
// op_sel for a two-source VOP3 is [src0, src1, dst].
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, VCvtPkFp8F32_HighHalfValid) {
    std::string error =
        verifyRW("v_cvt_pk_fp8_f32", {vgpr(0)}, {vgpr(1), vgpr(2), vgpr(0)}, {0, 0, 1});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, VCvtPkFp8F32_HighHalfMissingDstInSrc) {
    std::string error = verifyRW("v_cvt_pk_fp8_f32", {vgpr(0)}, {vgpr(1), vgpr(2)}, {0, 0, 1});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

TEST_F(ReadWriteOperandTest, VCvtPkFp8F32_LowHalfMissingDstInSrc) {
    // The low half keeps the high one, so it is read-write as well.
    std::string error = verifyRW("v_cvt_pk_fp8_f32", {vgpr(0)}, {vgpr(1), vgpr(2)}, {0, 0, 0});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

TEST_F(ReadWriteOperandTest, VCvtPkFp8F32_NoOpSelNeedsNoDstInSrc) {
    // Naming no half is the producer saying it writes the whole register, and
    // the table's flag is off, so nothing here reads the destination.
    std::string error = verifyRW("v_cvt_pk_fp8_f32", {vgpr(0)}, {vgpr(1), vgpr(2)});
    EXPECT_TRUE(error.empty()) << error;
}

// ---------------------------------------------------------------------------
// v_cvt_pk_bf8_f32  —  same pattern as the FP8 variant
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, VCvtPkBf8F32_HighHalfValid) {
    std::string error =
        verifyRW("v_cvt_pk_bf8_f32", {vgpr(0)}, {vgpr(1), vgpr(2), vgpr(0)}, {0, 0, 1});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, VCvtPkBf8F32_HighHalfMissingDstInSrc) {
    std::string error = verifyRW("v_cvt_pk_bf8_f32", {vgpr(0)}, {vgpr(1), vgpr(2)}, {0, 0, 1});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// The pack converts stay out of the table's read-write set: which half survives
// is chosen per instance, and the table describes the encoding.
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, PackConvertsAreNotTableReadWrite) {
    for (const char* mnemonic : {"v_cvt_pk_fp8_f32", "v_cvt_pk_bf8_f32"}) {
        const HwInstDesc* desc = getDescByMnemonic(mnemonic);
        ASSERT_NE(desc, nullptr) << mnemonic;
        ASSERT_FALSE(desc->operandFields.empty()) << mnemonic;
        EXPECT_TRUE(desc->operandFields[0].isDest) << mnemonic;
        EXPECT_EQ(desc->operandFields[0].fieldSizeBits, 16) << mnemonic;
        EXPECT_FALSE(desc->operandFields[0].isReadWrite) << mnemonic;
    }
}

// ---------------------------------------------------------------------------
// v_add_f16 also declares a 16-bit destination, and no producer names a half for
// it, so it is a whole-register write and reads nothing back.
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, VAddF16_NeedsNoDstInSrc) {
    std::string error = verifyRW("v_add_f16", {vgpr(0)}, {vgpr(1), vgpr(2)});
    EXPECT_TRUE(error.empty()) << error;
}

// ---------------------------------------------------------------------------
// v_cvt_sr_bf8_f32  —  D0 is RW (same pattern as FP8 variant)
// HW fields: {D0, vdst, vgpr, 8, RW}, {S0, src0, src, 32}, {S1, src1, src, 32}
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, VCvtSrBf8F32_Valid) {
    std::string error = verifyRW("v_cvt_sr_bf8_f32", {vgpr(0)}, {vgpr(1), vgpr(2), vgpr(0)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, VCvtSrBf8F32_MissingDstInSrc) {
    std::string error = verifyRW("v_cvt_sr_bf8_f32", {vgpr(0)}, {vgpr(1), vgpr(2)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// buffer_atomic_add_f32  —  D0 (vdata) is RW: atomic read-modify-write
// MUBUF_ATOMIC fields: {D0, vdata, vgpr, 32, RW}, {S0, vaddr, vgpr, 32},
//                       {S1, rsrc, sreg, 128}, {S2, soffset, sreg_m0, 32}
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, BufferAtomicAddF32_Valid) {
    std::string error =
        verifyRW("buffer_atomic_add_f32", {vgpr(0)}, {vgpr(0), vgpr(1), sgpr(0, 4), sgpr(4)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, BufferAtomicAddF32_MissingDstInSrc) {
    std::string error =
        verifyRW("buffer_atomic_add_f32", {vgpr(0)}, {vgpr(1), sgpr(0, 4), sgpr(4)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// buffer_atomic_cmpswap_b32  —  D0 (vdata) is RW, size overridden to 64
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, BufferAtomicCmpswapB32_Valid) {
    std::string error = verifyRW("buffer_atomic_cmpswap_b32", {vgpr(0, 2)},
                                 {vgpr(0, 2), vgpr(2), sgpr(0, 4), sgpr(4)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, BufferAtomicCmpswapB32_MissingDstInSrc) {
    std::string error =
        verifyRW("buffer_atomic_cmpswap_b32", {vgpr(0, 2)}, {vgpr(2), sgpr(0, 4), sgpr(4)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// buffer_atomic_cmpswap_b64  —  D0 (vdata) is RW, size overridden to 128
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, BufferAtomicCmpswapB64_Valid) {
    std::string error = verifyRW("buffer_atomic_cmpswap_b64", {vgpr(0, 4)},
                                 {vgpr(0, 4), vgpr(4), sgpr(0, 4), sgpr(4)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, BufferAtomicCmpswapB64_MissingDstInSrc) {
    std::string error =
        verifyRW("buffer_atomic_cmpswap_b64", {vgpr(0, 4)}, {vgpr(4), sgpr(0, 4), sgpr(4)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// v_swmmac_f32_16x16x64_f16  —  D0 (acc) is RW: acc += A * B
// SWMMA fields: {D0, vdst, vgpr, 256, RW}, {S0, src0, src_vgpr, 256},
//               {S1, src1, src_vgpr, 512*}, {S2, src2, src_vgpr, 32}
//  * S1 overridden to 512 for this instruction
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, SWMMA_F32_16x16x64_F16_Valid) {
    std::string error = verifyRW("v_swmmac_f32_16x16x64_f16", {vgpr(0, 8)},
                                 {vgpr(8, 8), vgpr(16, 16), vgpr(32), vgpr(0, 8)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, SWMMA_F32_16x16x64_F16_MissingAccInSrc) {
    std::string error =
        verifyRW("v_swmmac_f32_16x16x64_f16", {vgpr(0, 8)}, {vgpr(8, 8), vgpr(16, 16), vgpr(32)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}

// ---------------------------------------------------------------------------
// v_swap_b32  —  Both D0 and D1 are RW: v_swap_b32 v0, v1
// HW fields: {D0, vdst, vgpr, 32, RW}, {D1, src0, src_vgpr, 32, RW}
// ---------------------------------------------------------------------------

TEST_F(ReadWriteOperandTest, VSwapB32_Valid) {
    std::string error = verifyRW("v_swap_b32", {vgpr(0), vgpr(1)}, {vgpr(0), vgpr(1)});
    EXPECT_TRUE(error.empty()) << error;
}

TEST_F(ReadWriteOperandTest, VSwapB32_MissingFirstRegInSrc) {
    std::string error = verifyRW("v_swap_b32", {vgpr(0), vgpr(1)}, {vgpr(1)});
    EXPECT_NE(error.find("Read-write"), std::string::npos) << "Expected RW error, got: " << error;
}
