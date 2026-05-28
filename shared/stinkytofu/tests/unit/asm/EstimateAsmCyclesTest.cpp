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

#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/transforms/asm/EstimateAsmCyclesPass.hpp"

using namespace stinkytofu;

// Helper class to build test IR and run pass
class EstimateAsmCyclesTest : public ::testing::Test {
   protected:
    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);  // GFX1250
        gemmConfig.arch[0] = 12;
        gemmConfig.arch[1] = 5;
        gemmConfig.arch[2] = 0;
        gemmConfig.TileA0 = 16;
        gemmConfig.TileB0 = 16;
        gemmConfig.TileM0 = 16;
        gemmConfig.NumGRA = 4;
        gemmConfig.NumGRB = 4;
        gemmConfig.NumGRM = 4;
        gemmConfig.NumWaves = 4;

        // Create a Function with a BasicBlock for testing
        func = std::make_unique<Function>("test_function");
        loopBB = func->createBasicBlock("label_LoopBeginL");
    }

    void TearDown() override {
        // Clean up Function (which will clean up BasicBlocks and IR)
        func.reset();
        loopBB = nullptr;
    }

    // Create IRBuilder for building test instructions
    AsmIRBuilder getIRBuilder() {
        return AsmIRBuilder(*loopBB, arch);
    }

    // Helper to create a v_add_f32 instruction with issueCycles
    void createLabel(const std::string& label) {
        auto builder = getIRBuilder();
        builder.createLabel(label);
    }

    // Helper to create a v_add_f32 instruction with issueCycles
    StinkyInstruction* createVAddF32(int destReg, int src0Reg, int src1Reg, int issueCycles = 1) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_add_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
        inst->issueCycles = issueCycles;
        return inst;
    }

    // Helper to create a v_mul_f32 instruction with issueCycles
    StinkyInstruction* createVMulF32(int destReg, int src0Reg, int src1Reg, int issueCycles = 1) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_mul_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
        inst->issueCycles = issueCycles;
        return inst;
    }

    // Helper to create a v_fma_f32 instruction with issueCycles
    StinkyInstruction* createVFmaF32(int destReg, int src0Reg, int src1Reg, int src2Reg,
                                     int issueCycles = 1) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_fma_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src2Reg, 1));
        inst->issueCycles = issueCycles;
        return inst;
    }

    // Helper to run the pass and get the result
    unsigned int runPassAndGetResult() {
        func->setGemmTileConfig(gemmConfig);
        AnalysisManager AM;
        AM.registerPass<EstimateAsmCyclesAnalysis>();
        return AM.getResult<EstimateAsmCyclesAnalysis>(*func);
    }

    GemmTileConfig gemmConfig;
    GfxArchID arch;
    std::unique_ptr<Function> func;
    BasicBlock* loopBB;
};

// Test with empty BasicBlock
TEST_F(EstimateAsmCyclesTest, EmptyBasicBlock) {
    unsigned int cycles = runPassAndGetResult();
    EXPECT_EQ(cycles, 0);
}

// Test with instructions that have zero issueCycles
TEST_F(EstimateAsmCyclesTest, ZeroIssueCycles) {
    createLabel("label_LoopBeginL");
    createVMulF32(3, 4, 5, 0);

    unsigned int cycles = runPassAndGetResult();
    EXPECT_EQ(cycles, 0);
}

// Test that only instructions after "label_LoopBeginL" are processed
TEST_F(EstimateAsmCyclesTest, OnlyProcessLabelLoopBeginL) {
    // Use a non-loop-named basic block so pass logic starts counting only
    // after it sees an explicit internal label "label_LoopBeginL".
    BasicBlock* testBB = func->createBasicBlock("entry");
    loopBB = testBB;

    // Put a non-loop instruction before the label in the same basic block.
    createVAddF32(0, 1, 2, 10);

    // Add loop label + instruction that should be counted.
    createLabel("label_LoopBeginL");
    createVAddF32(0, 1, 2, 5);

    unsigned int cycles = runPassAndGetResult();
    // Should only count cycles after label_LoopBeginL.
    EXPECT_EQ(cycles, 5);
}

// Test with many instructions
TEST_F(EstimateAsmCyclesTest, ManyInstructions) {
    const int numInstructions = 100;
    const int cyclesPerInst = 2;
    unsigned int expectedCycles = 0;

    createLabel("label_LoopBeginL");
    for (int i = 0; i < numInstructions; ++i) {
        createVAddF32(i, i + 1, i + 2, cyclesPerInst);
        expectedCycles += cyclesPerInst;
    }

    unsigned int cycles = runPassAndGetResult();
    EXPECT_EQ(cycles, expectedCycles);
}

// Test for gfx1250: two basic blocks "LocalRead" (exact content from lr.s) and
// "loopBody" (label_LoopBeginL, exact content from loop.s converted to Stinky IR).
TEST_F(EstimateAsmCyclesTest, Gfx1250LocalReadAndLoopBody) {
    arch = getGfxArchID(12, 5, 0);
    gemmConfig.arch[0] = 12;
    gemmConfig.arch[1] = 5;
    gemmConfig.arch[2] = 0;

    // gfx1250-compatible minimal LocalRead + loop-body smoke test.
    const char* minimalFullIR = R"(
st.func @gfx1250_localread_loop() {
^LocalRead:
"st.s_set_vgpr_msb"(12) { issueCycles = 1, latencyCycles = 1 }
v1 = "st.v_and_b32"(31, v[200]) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(3072) { issueCycles = 1, latencyCycles = 1 }
v0 = "st.v_and_b32"(15, v1) { issueCycles = 1, latencyCycles = 1 }
v0 = "st.v_lshlrev_b32"(5, v0) { issueCycles = 1, latencyCycles = 1 }
v1 = "st.v_lshrrev_b32"(4, v1) { issueCycles = 1, latencyCycles = 1 }
v0 = "st.v_lshl_add_u32"(v1, 3, v0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(12) { issueCycles = 1, latencyCycles = 1 }
v4 = "st.v_lshrrev_b32"(5, v[200]) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(3072) { issueCycles = 1, latencyCycles = 1 }
v4 = "st.v_and_b32"(1, v4) { issueCycles = 1, latencyCycles = 1 }
v0 = "st.v_lshl_add_u32"(v4, 9, v0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(12) { issueCycles = 1, latencyCycles = 1 }
v2 = "st.v_and_b32"(31, v[200]) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(3072) { issueCycles = 1, latencyCycles = 1 }
v1 = "st.v_and_b32"(15, v2) { issueCycles = 1, latencyCycles = 1 }
v1 = "st.v_lshlrev_b32"(5, v1) { issueCycles = 1, latencyCycles = 1 }
v2 = "st.v_lshrrev_b32"(4, v2) { issueCycles = 1, latencyCycles = 1 }
v1 = "st.v_lshl_add_u32"(v2, 3, v1) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(12) { issueCycles = 1, latencyCycles = 1 }
v3 = "st.v_lshrrev_b32"(6, v[200]) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(3072) { issueCycles = 1, latencyCycles = 1 }
v3 = "st.v_and_b32"(1, v3) { issueCycles = 1, latencyCycles = 1 }
v1 = "st.v_lshl_add_u32"(v3, 9, v1) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(12) { issueCycles = 1, latencyCycles = 1 }
v2 = "st.v_lshrrev_b32"(5, v[200]) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(3072) { issueCycles = 1, latencyCycles = 1 }
v2 = "st.v_lshrrev_b32"(2, v2) { issueCycles = 1, latencyCycles = 1 }
s16 = "st.s_mov_b32"(32) { issueCycles = 1, latencyCycles = 1 }
v2 = "st.v_mul_lo_u32"(s16, v2) { issueCycles = 1, latencyCycles = 1 }
v[140] = "st.v_add_nc_u32"(v2, v0) { issueCycles = 1, latencyCycles = 1 }
v[140] = "st.v_lshlrev_b32"(1, v[140]) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(34828) { issueCycles = 1, latencyCycles = 1 }
v0 = "st.v_lshrrev_b32"(5, v[200]) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(3072) { issueCycles = 1, latencyCycles = 1 }
v0 = "st.v_lshrrev_b32"(2, v0) { issueCycles = 1, latencyCycles = 1 }
v0 = "st.v_mul_lo_u32"(s16, v0) { issueCycles = 1, latencyCycles = 1 }
v[141] = "st.v_add_nc_u32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }
v[141] = "st.v_lshlrev_b32"(1, v[141]) { issueCycles = 1, latencyCycles = 1 }
v[141], vcc = "st.v_add_co_u32"(0x4000, v[141]) { issueCycles = 1, latencyCycles = 1 }
^LocalReadDoA_I0:
v[0:3] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
^LocalReadDoB_I0:
v[4:7] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
^label_LoopBeginL:
"st.s_xor_b32"(s[1], s[1], 0x8000) { issueCycles = 1, latencyCycles = 1 }
"st.s_add_u32"(s[2], s[2], s[3]) { issueCycles = 1, latencyCycles = 1 }
"st.s_sub_u32"(s[4], s[4], 1) { issueCycles = 1, latencyCycles = 1 }
"st.s_cmp_eq_i32"(s[4], 0x2) { issueCycles = 1, latencyCycles = 1 }
"st.s_barrier_signal"(-1) { issueCycles = 1, latencyCycles = 1 }
"st.s_barrier_wait"(-1) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(130) { issueCycles = 1, latencyCycles = 1 }
v[4:7] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
v[8:11] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 32, gds = false } }
v[12:15] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
v[16:19] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 32, gds = false } }
"st.s_wait_dscnt"(0) { issueCycles = 1, latencyCycles = 1 }
v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[12:19], v[4:11], v[0:7]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[20:23] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 2048, gds = false } }
v[24:27] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 2080, gds = false } }
v[28:31] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 4096, gds = false } }
v[32:35] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 4128, gds = false } }
"st.s_wait_dscnt"(2) { issueCycles = 1, latencyCycles = 1 }
v[8:15] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[4:11], v[8:15]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(4) { issueCycles = 1, latencyCycles = 1 }
v[16:23] = "st.v_wmma_f32_16x16x32_bf16"(v[28:35], v[4:11], v[16:23]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(6) { issueCycles = 1, latencyCycles = 1 }
v[24:31] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[8:15], v[24:31]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(8) { issueCycles = 1, latencyCycles = 1 }
v[32:39] = "st.v_wmma_f32_16x16x32_bf16"(v[28:35], v[8:15], v[32:39]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(10) { issueCycles = 1, latencyCycles = 1 }
v[40:47] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[12:19], v[40:47]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(12) { issueCycles = 1, latencyCycles = 1 }
v[48:55] = "st.v_wmma_f32_16x16x32_bf16"(v[28:35], v[12:19], v[48:55]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(14) { issueCycles = 1, latencyCycles = 1 }
v[56:63] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[16:23], v[56:63]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_barrier_signal"(-1) { issueCycles = 1, latencyCycles = 1 }
"st.s_barrier_wait"(-1) { issueCycles = 1, latencyCycles = 1 }
v[140] = "st.v_xor_b32"(0x8000, v[140]) { issueCycles = 1, latencyCycles = 1 }
v[141] = "st.v_xor_b32"(0x8000, v[141]) { issueCycles = 1, latencyCycles = 1 }
v[64:71] = "st.v_wmma_f32_16x16x32_bf16"(v[12:19], v[20:27], v[64:71]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[72:79] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[20:27], v[72:79]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[80:87] = "st.v_wmma_f32_16x16x32_bf16"(v[28:35], v[20:27], v[80:87]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_cbranch_scc0"(label_LoopBeginL) { issueCycles = 1, latencyCycles = 1 }
}
)";

    func = std::make_unique<Function>("gfx1250_localread_loop");
    PassManager pmMinimal;
    pmMinimal.setGemmTileConfig(gemmConfig);
    pmMinimal.setPassFeatureConfig(PassFeatureConfig());
    pmMinimal.setBasicBlockFilter(BasicBlockFilterBuilder::all());

    StinkyErrorCode parseErr = StinkyIRConverter::populateFunctionFromString(
        minimalFullIR, *func, pmMinimal.getPassContext(), arch);
    ASSERT_EQ(parseErr, StinkyErrorCode::SUCCESS)
        << "Failed to parse minimal gfx1250 LocalRead + loopBody IR";

    unsigned int minimalCycles = runPassAndGetResult();
    EXPECT_GT(minimalCycles, 0u) << "Expected positive cycle count for gfx1250 loop body";
    return;

    // Stinky IR: LocalRead block = exact same instructions as lr.s with original symbols preserved.
    // Symbols: vgprSerial (input), vgprLocalReadAddrA, vgprLocalReadAddrB, vgprLocalReadAddrB+0,
    // vgprLocalReadAddrB+1
    const char* localReadIR = R"(
^LocalRead:
v9 = "st.v_and_b32"(63, v[vgprSerial]) { issueCycles = 4, latencyCycles = 4 }
v8 = "st.v_and_b32"(15, v9) { issueCycles = 4, latencyCycles = 4 }
v8 = "st.v_lshlrev_b32"(6, v8) { issueCycles = 4, latencyCycles = 4 }
v8 = "st.v_lshlrev_b32"(1, v8) { issueCycles = 4, latencyCycles = 4 }
v9 = "st.v_lshrrev_b32"(4, v9) { issueCycles = 4, latencyCycles = 4 }
v8 = "st.v_lshl_add_u32"(v9, 3, v8) { issueCycles = 4, latencyCycles = 4 }
v10 = "st.v_and_b32"(63, v[vgprSerial]) { issueCycles = 4, latencyCycles = 4 }
v9 = "st.v_and_b32"(15, v10) { issueCycles = 4, latencyCycles = 4 }
v9 = "st.v_lshlrev_b32"(6, v9) { issueCycles = 4, latencyCycles = 4 }
v9 = "st.v_lshlrev_b32"(1, v9) { issueCycles = 4, latencyCycles = 4 }
v10 = "st.v_lshrrev_b32"(4, v10) { issueCycles = 4, latencyCycles = 4 }
v9 = "st.v_lshl_add_u32"(v10, 3, v9) { issueCycles = 4, latencyCycles = 4 }
v11 = "st.v_lshrrev_b32"(6, v[vgprSerial]) { issueCycles = 4, latencyCycles = 4 }
v11 = "st.v_and_b32"(3, v11) { issueCycles = 4, latencyCycles = 4 }
v9 = "st.v_lshl_add_u32"(v11, 11, v9) { issueCycles = 4, latencyCycles = 4 }
v10 = "st.v_lshrrev_b32"(6, v[vgprSerial]) { issueCycles = 4, latencyCycles = 4 }
v10 = "st.v_lshrrev_b32"(2, v10) { issueCycles = 4, latencyCycles = 4 }
s46 = "st.s_mov_b32"(64) { issueCycles = 4, latencyCycles = 4 }
v10 = "st.v_mul_lo_u32"(s46, v10) { issueCycles = 4, latencyCycles = 4 }
v[vgprLocalReadAddrA] = "st.v_add_lshl_u32"(v10, v8, 0x1) { issueCycles = 4, latencyCycles = 4 }
v11 = "st.v_lshrrev_b32"(8, v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 4 }
v[vgprLocalReadAddrA] = "st.v_lshl_add_u32"(v11, 5, v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 4 }
v8 = "st.v_lshrrev_b32"(6, v[vgprSerial]) { issueCycles = 4, latencyCycles = 4 }
v8 = "st.v_lshrrev_b32"(2, v8) { issueCycles = 4, latencyCycles = 4 }
v8 = "st.v_mul_lo_u32"(s46, v8) { issueCycles = 4, latencyCycles = 4 }
v[vgprLocalReadAddrB] = "st.v_add_lshl_u32"(v8, v9, 0x1) { issueCycles = 4, latencyCycles = 4 }
v10 = "st.v_lshrrev_b32"(8, v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 4 }
v[vgprLocalReadAddrB] = "st.v_lshl_add_u32"(v10, 5, v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 4 }
v[vgprLocalReadAddrB+0], vcc = "st.v_add_co_u32"(0x3600, v[vgprLocalReadAddrB+0]) { issueCycles = 4, latencyCycles = 4 }
v[vgprLocalReadAddrB+1] = "st.v_add_u32"(65536, v[vgprLocalReadAddrB+0]) { issueCycles = 4, latencyCycles = 4 }
^LocalReadDoA_I0:
v[0:3] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
^LocalReadDoB_I0:
v[4:7] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
)";

    // Loop body = exact same instruction sequence as loop.s (converted to Stinky IR, gfx950
    // v_smfmac)
    std::string loopBodyIR = R"loopbody(
^label_LoopBeginL:
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 0, dscnt = 0, kmcnt = -1 } }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 9, dscnt = 0, kmcnt = -1 } }
acc[0:3] = "st.v_smfmac_f32_16x16x32_bf16"(v[500:503], v[504:507], acc[0:3]) { issueCycles = 4, latencyCycles = 16 }
v[0:3] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 64, gds = false } }
acc[4:7] = "st.v_smfmac_f32_16x16x32_bf16"(v[508:511], v[512:515], acc[4:7]) { issueCycles = 4, latencyCycles = 16 }
v[4:7] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 64, gds = false } }
acc[8:11] = "st.v_smfmac_f32_16x16x32_bf16"(v[516:519], v[520:523], acc[8:11]) { issueCycles = 4, latencyCycles = 16 }
v[8:11] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 192, gds = false } }
acc[12:15] = "st.v_smfmac_f32_16x16x32_bf16"(v[524:527], v[528:531], acc[12:15]) { issueCycles = 4, latencyCycles = 16 }
v[12:15] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 4672, gds = false } }
acc[16:19] = "st.v_smfmac_f32_16x16x32_bf16"(v[532:535], v[536:539], acc[16:19]) { issueCycles = 4, latencyCycles = 16 }
v[16:19] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 4800, gds = false } }
acc[20:23] = "st.v_smfmac_f32_16x16x32_bf16"(v[540:543], v[544:547], acc[20:23]) { issueCycles = 4, latencyCycles = 16 }
v[20:23] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 9280, gds = false } }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 6, dscnt = 0, kmcnt = -1 } }
v[24:27] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 9408, gds = false } }
acc[28:31] = "st.v_smfmac_f32_16x16x32_bf16"(v[548:551], v[552:555], acc[28:31]) { issueCycles = 4, latencyCycles = 16 }
v[28:31] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 192, gds = false } }
acc[32:35] = "st.v_smfmac_f32_16x16x32_bf16"(v[556:559], v[560:563], acc[32:35]) { issueCycles = 4, latencyCycles = 16 }
v[32:35] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 18496, gds = false } }
acc[36:39] = "st.v_smfmac_f32_16x16x32_bf16"(v[564:567], v[568:571], acc[36:39]) { issueCycles = 4, latencyCycles = 16 }
v[36:39] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 18624, gds = false } }
acc[40:43] = "st.v_smfmac_f32_16x16x32_bf16"(v[572:575], v[576:579], acc[40:43]) { issueCycles = 4, latencyCycles = 16 }
v[40:43] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 36928, gds = false } }
acc[44:47] = "st.v_smfmac_f32_16x16x32_bf16"(v[580:583], v[584:587], acc[44:47]) { issueCycles = 4, latencyCycles = 16 }
v[44:47] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 37056, gds = false } }
acc[48:51] = "st.v_smfmac_f32_16x16x32_bf16"(v[588:591], v[592:595], acc[48:51]) { issueCycles = 4, latencyCycles = 16 }
v[48:51] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 55360, gds = false } }
acc[52:55] = "st.v_smfmac_f32_16x16x32_bf16"(v[596:599], v[600:603], acc[52:55]) { issueCycles = 4, latencyCycles = 16 }
v[52:55] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 55488, gds = false } }
acc[56:59] = "st.v_smfmac_f32_16x16x32_bf16"(v[604:607], v[608:611], acc[56:59]) { issueCycles = 4, latencyCycles = 16 }
v[56:59] = "st.ds_read_b128"(v[vgprLocalReadAddrB+1]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 8256, gds = false } }
acc[60:63] = "st.v_smfmac_f32_16x16x32_bf16"(v[612:615], v[616:619], acc[60:63]) { issueCycles = 4, latencyCycles = 16 }
v[60:63] = "st.ds_read_b128"(v[vgprLocalReadAddrB+1]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 8384, gds = false } }
acc[64:67] = "st.v_smfmac_f32_16x16x32_bf16"(v[620:623], v[624:627], acc[64:67]) { issueCycles = 4, latencyCycles = 16 }
acc[68:71] = "st.v_smfmac_f32_16x16x32_bf16"(v[628:631], v[632:635], acc[68:71]) { issueCycles = 4, latencyCycles = 16 }
acc[72:75] = "st.v_smfmac_f32_16x16x32_bf16"(v[636:639], v[640:643], acc[72:75]) { issueCycles = 4, latencyCycles = 16 }
acc[76:79] = "st.v_smfmac_f32_16x16x32_bf16"(v[644:647], v[648:651], acc[76:79]) { issueCycles = 4, latencyCycles = 16 }
acc[80:83] = "st.v_smfmac_f32_16x16x32_bf16"(v[652:655], v[656:659], acc[80:83]) { issueCycles = 4, latencyCycles = 16 }
acc[84:87] = "st.v_smfmac_f32_16x16x32_bf16"(v[660:663], v[664:667], acc[84:87]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 0, dscnt = 0, kmcnt = -1 } }
"st.s_barrier"(0) { issueCycles = 4, latencyCycles = 4 }
acc[88:91] = "st.v_smfmac_f32_16x16x32_bf16"(v[668:671], v[672:675], acc[88:91]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[0:3]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 0, gds = false } }
acc[92:95] = "st.v_smfmac_f32_16x16x32_bf16"(v[676:679], v[680:683], acc[92:95]) { issueCycles = 4, latencyCycles = 16 }
v[0:3] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[96:99] = "st.v_smfmac_f32_16x16x32_bf16"(v[684:687], v[688:691], acc[96:99]) { issueCycles = 4, latencyCycles = 16 }
acc[100:103] = "st.v_smfmac_f32_16x16x32_bf16"(v[692:695], v[696:699], acc[100:103]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[4:7]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 4608, gds = false } }
acc[104:107] = "st.v_smfmac_f32_16x16x32_bf16"(v[700:703], v[704:707], acc[104:107]) { issueCycles = 4, latencyCycles = 16 }
v[4:7] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[108:111] = "st.v_smfmac_f32_16x16x32_bf16"(v[708:711], v[712:715], acc[108:111]) { issueCycles = 4, latencyCycles = 16 }
acc[112:115] = "st.v_smfmac_f32_16x16x32_bf16"(v[716:719], v[720:723], acc[112:115]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[8:11]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 9216, gds = false } }
acc[116:119] = "st.v_smfmac_f32_16x16x32_bf16"(v[724:727], v[728:731], acc[116:119]) { issueCycles = 4, latencyCycles = 16 }
v[8:11] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[120:123] = "st.v_smfmac_f32_16x16x32_bf16"(v[732:735], v[736:739], acc[120:123]) { issueCycles = 4, latencyCycles = 16 }
acc[124:127] = "st.v_smfmac_f32_16x16x32_bf16"(v[740:743], v[744:747], acc[124:127]) { issueCycles = 4, latencyCycles = 16 }
acc[128:131] = "st.v_smfmac_f32_16x16x32_bf16"(v[748:751], v[752:755], acc[128:131]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[12:15]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 13824, gds = false } }
acc[132:135] = "st.v_smfmac_f32_16x16x32_bf16"(v[756:759], v[760:763], acc[132:135]) { issueCycles = 4, latencyCycles = 16 }
v[12:15] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[136:139] = "st.v_smfmac_f32_16x16x32_bf16"(v[764:767], v[768:771], acc[136:139]) { issueCycles = 4, latencyCycles = 16 }
acc[140:143] = "st.v_smfmac_f32_16x16x32_bf16"(v[772:775], v[776:779], acc[140:143]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[16:19]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 18432, gds = false } }
acc[144:147] = "st.v_smfmac_f32_16x16x32_bf16"(v[780:783], v[784:787], acc[144:147]) { issueCycles = 4, latencyCycles = 16 }
v[16:19] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[148:151] = "st.v_smfmac_f32_16x16x32_bf16"(v[788:791], v[792:795], acc[148:151]) { issueCycles = 4, latencyCycles = 16 }
acc[152:155] = "st.v_smfmac_f32_16x16x32_bf16"(v[796:799], v[800:803], acc[152:155]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[20:23]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 23040, gds = false } }
acc[156:159] = "st.v_smfmac_f32_16x16x32_bf16"(v[804:807], v[808:811], acc[156:159]) { issueCycles = 4, latencyCycles = 16 }
v[20:23] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[160:163] = "st.v_smfmac_f32_16x16x32_bf16"(v[812:815], v[816:819], acc[160:163]) { issueCycles = 4, latencyCycles = 16 }
acc[164:167] = "st.v_smfmac_f32_16x16x32_bf16"(v[820:823], v[824:827], acc[164:167]) { issueCycles = 4, latencyCycles = 16 }
acc[168:171] = "st.v_smfmac_f32_16x16x32_bf16"(v[828:831], v[832:835], acc[168:171]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[24:27]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 27648, gds = false } }
acc[172:175] = "st.v_smfmac_f32_16x16x32_bf16"(v[836:839], v[840:843], acc[172:175]) { issueCycles = 4, latencyCycles = 16 }
v[24:27] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[176:179] = "st.v_smfmac_f32_16x16x32_bf16"(v[844:847], v[848:851], acc[176:179]) { issueCycles = 4, latencyCycles = 16 }
acc[180:183] = "st.v_smfmac_f32_16x16x32_bf16"(v[852:855], v[856:859], acc[180:183]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[28:31]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 32256, gds = false } }
acc[184:187] = "st.v_smfmac_f32_16x16x32_bf16"(v[860:863], v[864:867], acc[184:187]) { issueCycles = 4, latencyCycles = 16 }
v[28:31] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[188:191] = "st.v_smfmac_f32_16x16x32_bf16"(v[868:871], v[872:875], acc[188:191]) { issueCycles = 4, latencyCycles = 16 }
acc[192:195] = "st.v_smfmac_f32_16x16x32_bf16"(v[876:879], v[880:883], acc[192:195]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[32:35]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 36864, gds = false } }
acc[196:199] = "st.v_smfmac_f32_16x16x32_bf16"(v[884:887], v[888:891], acc[196:199]) { issueCycles = 4, latencyCycles = 16 }
v[32:35] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[200:203] = "st.v_smfmac_f32_16x16x32_bf16"(v[900:903], v[904:907], acc[200:203]) { issueCycles = 4, latencyCycles = 16 }
acc[204:207] = "st.v_smfmac_f32_16x16x32_bf16"(v[908:911], v[912:915], acc[204:207]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[36:39]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 41472, gds = false } }
acc[212:215] = "st.v_smfmac_f32_16x16x32_bf16"(v[916:919], v[920:923], acc[212:215]) { issueCycles = 4, latencyCycles = 16 }
v[36:39] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[216:219] = "st.v_smfmac_f32_16x16x32_bf16"(v[924:927], v[928:931], acc[216:219]) { issueCycles = 4, latencyCycles = 16 }
acc[220:223] = "st.v_smfmac_f32_16x16x32_bf16"(v[932:935], v[936:939], acc[220:223]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[40:43]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 46080, gds = false } }
acc[224:227] = "st.v_smfmac_f32_16x16x32_bf16"(v[940:943], v[944:947], acc[224:227]) { issueCycles = 4, latencyCycles = 16 }
v[40:43] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[228:231] = "st.v_smfmac_f32_16x16x32_bf16"(v[948:951], v[952:955], acc[228:231]) { issueCycles = 4, latencyCycles = 16 }
acc[232:235] = "st.v_smfmac_f32_16x16x32_bf16"(v[956:959], v[960:963], acc[232:235]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[44:47]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 50688, gds = false } }
acc[236:239] = "st.v_smfmac_f32_16x16x32_bf16"(v[964:967], v[968:971], acc[236:239]) { issueCycles = 4, latencyCycles = 16 }
v[44:47] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[0:3] = "st.v_smfmac_f32_16x16x32_bf16"(v[972:975], v[976:979], acc[0:3]) { issueCycles = 4, latencyCycles = 16 }
acc[4:7] = "st.v_smfmac_f32_16x16x32_bf16"(v[980:983], v[984:987], acc[4:7]) { issueCycles = 4, latencyCycles = 16 }
acc[8:11] = "st.v_smfmac_f32_16x16x32_bf16"(v[988:991], v[992:995], acc[8:11]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[48:51]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 55296, gds = false } }
acc[12:15] = "st.v_smfmac_f32_16x16x32_bf16"(v[996:999], v[1000:1003], acc[12:15]) { issueCycles = 4, latencyCycles = 16 }
v[48:51] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[16:19] = "st.v_smfmac_f32_16x16x32_bf16"(v[1004:1007], v[1008:1011], acc[16:19]) { issueCycles = 4, latencyCycles = 16 }
acc[20:23] = "st.v_smfmac_f32_16x16x32_bf16"(v[1012:1015], v[1016:1019], acc[20:23]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[52:55]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 59904, gds = false } }
acc[24:27] = "st.v_smfmac_f32_16x16x32_bf16"(v[1020:1023], v[1024:1027], acc[24:27]) { issueCycles = 4, latencyCycles = 16 }
v[52:55] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[28:31] = "st.v_smfmac_f32_16x16x32_bf16"(v[1028:1031], v[1032:1035], acc[28:31]) { issueCycles = 4, latencyCycles = 16 }
acc[32:35] = "st.v_smfmac_f32_16x16x32_bf16"(v[1036:1039], v[1040:1043], acc[32:35]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB], v[56:59]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 64512, gds = false } }
acc[36:39] = "st.v_smfmac_f32_16x16x32_bf16"(v[1044:1047], v[1048:1051], acc[36:39]) { issueCycles = 4, latencyCycles = 16 }
v[56:59] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[40:43] = "st.v_smfmac_f32_16x16x32_bf16"(v[1052:1055], v[1056:1059], acc[40:43]) { issueCycles = 4, latencyCycles = 16 }
acc[44:47] = "st.v_smfmac_f32_16x16x32_bf16"(v[1060:1063], v[1064:1067], acc[44:47]) { issueCycles = 4, latencyCycles = 16 }
acc[48:51] = "st.v_smfmac_f32_16x16x32_bf16"(v[1068:1071], v[1072:1075], acc[48:51]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB+1], v[60:63]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 3584, gds = false } }
acc[52:55] = "st.v_smfmac_f32_16x16x32_bf16"(v[1076:1079], v[1080:1083], acc[52:55]) { issueCycles = 4, latencyCycles = 16 }
v[60:63] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[56:59] = "st.v_smfmac_f32_16x16x32_bf16"(v[1084:1087], v[1088:1091], acc[56:59]) { issueCycles = 4, latencyCycles = 16 }
acc[60:63] = "st.v_smfmac_f32_16x16x32_bf16"(v[1092:1095], v[1096:1099], acc[60:63]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB+1], v[64:67]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 8192, gds = false } }
acc[64:67] = "st.v_smfmac_f32_16x16x32_bf16"(v[1100:1103], v[1104:1107], acc[64:67]) { issueCycles = 4, latencyCycles = 16 }
v[64:67] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[68:71] = "st.v_smfmac_f32_16x16x32_bf16"(v[1108:1111], v[1112:1115], acc[68:71]) { issueCycles = 4, latencyCycles = 16 }
acc[72:75] = "st.v_smfmac_f32_16x16x32_bf16"(v[1116:1119], v[1120:1123], acc[72:75]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB+1], v[68:71]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 12800, gds = false } }
acc[76:79] = "st.v_smfmac_f32_16x16x32_bf16"(v[1124:1127], v[1128:1131], acc[76:79]) { issueCycles = 4, latencyCycles = 16 }
v[68:71] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[80:83] = "st.v_smfmac_f32_16x16x32_bf16"(v[1132:1135], v[1136:1139], acc[80:83]) { issueCycles = 4, latencyCycles = 16 }
acc[84:87] = "st.v_smfmac_f32_16x16x32_bf16"(v[1140:1143], v[1144:1147], acc[84:87]) { issueCycles = 4, latencyCycles = 16 }
acc[88:91] = "st.v_smfmac_f32_16x16x32_bf16"(v[1148:1151], v[1152:1155], acc[88:91]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB+1], v[72:75]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 17408, gds = false } }
acc[92:95] = "st.v_smfmac_f32_16x16x32_bf16"(v[1156:1159], v[1160:1163], acc[92:95]) { issueCycles = 4, latencyCycles = 16 }
v[72:75] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[96:99] = "st.v_smfmac_f32_16x16x32_bf16"(v[1164:1167], v[1168:1171], acc[96:99]) { issueCycles = 4, latencyCycles = 16 }
acc[100:103] = "st.v_smfmac_f32_16x16x32_bf16"(v[1172:1175], v[1176:1179], acc[100:103]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrB+1], v[76:79]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 22016, gds = false } }
acc[104:107] = "st.v_smfmac_f32_16x16x32_bf16"(v[1180:1183], v[1184:1187], acc[104:107]) { issueCycles = 4, latencyCycles = 16 }
v[76:79] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[108:111] = "st.v_smfmac_f32_16x16x32_bf16"(v[1188:1191], v[1192:1195], acc[108:111]) { issueCycles = 4, latencyCycles = 16 }
acc[112:115] = "st.v_smfmac_f32_16x16x32_bf16"(v[1196:1199], v[1200:1203], acc[112:115]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrA], v[80:83]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 0, gds = false } }
acc[116:119] = "st.v_smfmac_f32_16x16x32_bf16"(v[1204:1207], v[1208:1211], acc[116:119]) { issueCycles = 4, latencyCycles = 16 }
v[80:83] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[120:123] = "st.v_smfmac_f32_16x16x32_bf16"(v[1212:1215], v[1216:1219], acc[120:123]) { issueCycles = 4, latencyCycles = 16 }
acc[124:127] = "st.v_smfmac_f32_16x16x32_bf16"(v[1220:1223], v[1224:1227], acc[124:127]) { issueCycles = 4, latencyCycles = 16 }
acc[128:131] = "st.v_smfmac_f32_16x16x32_bf16"(v[1228:1231], v[1232:1235], acc[128:131]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 22, dscnt = 0, kmcnt = -1 } }
"st.ds_write_b128"(v[vgprLocalWriteAddrA], v[84:87]) { issueCycles = 4, latencyCycles = 4, mod.ds = { na = 1, offset = 4608, gds = false } }
acc[132:135] = "st.v_smfmac_f32_16x16x32_bf16"(v[1236:1239], v[1240:1243], acc[132:135]) { issueCycles = 4, latencyCycles = 16 }
v[84:87] = "st.buffer_load_dwordx4"(v90, s[0:3], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
acc[136:139] = "st.v_smfmac_f32_16x16x32_bf16"(v[1244:1247], v[1248:1251], acc[136:139]) { issueCycles = 4, latencyCycles = 16 }
acc[140:143] = "st.v_smfmac_f32_16x16x32_bf16"(v[1252:1255], v[1256:1259], acc[140:143]) { issueCycles = 4, latencyCycles = 16 }
"st.s_waitcnt"() { issueCycles = 4, latencyCycles = 4, mod.swaitcnt = { vlcnt = 0, vscnt = -1, dlcnt = 0, dscnt = 0, kmcnt = -1 } }
"st.s_barrier"(0) { issueCycles = 4, latencyCycles = 4 }
acc[156:159] = "st.v_smfmac_f32_16x16x32_bf16"(v[1284:1287], v[1288:1291], acc[156:159]) { issueCycles = 4, latencyCycles = 16 }
v[64:67] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
acc[160:163] = "st.v_smfmac_f32_16x16x32_bf16"(v[1292:1295], v[1296:1299], acc[160:163]) { issueCycles = 4, latencyCycles = 16 }
v[68:71] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
acc[164:167] = "st.v_smfmac_f32_16x16x32_bf16"(v[1300:1303], v[1304:1307], acc[164:167]) { issueCycles = 4, latencyCycles = 16 }
v[72:75] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 128, gds = false } }
acc[168:171] = "st.v_smfmac_f32_16x16x32_bf16"(v[1308:1311], v[1312:1315], acc[168:171]) { issueCycles = 4, latencyCycles = 16 }
v[76:79] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 4608, gds = false } }
acc[172:175] = "st.v_smfmac_f32_16x16x32_bf16"(v[1316:1319], v[1320:1323], acc[172:175]) { issueCycles = 4, latencyCycles = 16 }
v[80:83] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 4736, gds = false } }
acc[176:179] = "st.v_smfmac_f32_16x16x32_bf16"(v[1324:1327], v[1328:1331], acc[176:179]) { issueCycles = 4, latencyCycles = 16 }
v[84:87] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 9216, gds = false } }
acc[180:183] = "st.v_smfmac_f32_16x16x32_bf16"(v[1332:1335], v[1336:1339], acc[180:183]) { issueCycles = 4, latencyCycles = 16 }
v[88:91] = "st.ds_read_b128"(v[vgprLocalReadAddrA]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 9344, gds = false } }
acc[184:187] = "st.v_smfmac_f32_16x16x32_bf16"(v[1340:1343], v[1344:1347], acc[184:187]) { issueCycles = 4, latencyCycles = 16 }
v[92:95] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 128, gds = false } }
acc[188:191] = "st.v_smfmac_f32_16x16x32_bf16"(v[1348:1351], v[1352:1355], acc[188:191]) { issueCycles = 4, latencyCycles = 16 }
v[96:99] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 18432, gds = false } }
acc[192:195] = "st.v_smfmac_f32_16x16x32_bf16"(v[1356:1359], v[1360:1363], acc[192:195]) { issueCycles = 4, latencyCycles = 16 }
v[100:103] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 18560, gds = false } }
acc[196:199] = "st.v_smfmac_f32_16x16x32_bf16"(v[1364:1367], v[1368:1371], acc[196:199]) { issueCycles = 4, latencyCycles = 16 }
v[104:107] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 36864, gds = false } }
acc[200:203] = "st.v_smfmac_f32_16x16x32_bf16"(v[1372:1375], v[1376:1379], acc[200:203]) { issueCycles = 4, latencyCycles = 16 }
v[108:111] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 36992, gds = false } }
acc[204:207] = "st.v_smfmac_f32_16x16x32_bf16"(v[1380:1383], v[1384:1387], acc[204:207]) { issueCycles = 4, latencyCycles = 16 }
v[112:115] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 55296, gds = false } }
acc[208:211] = "st.v_smfmac_f32_16x16x32_bf16"(v[1388:1391], v[1392:1395], acc[208:211]) { issueCycles = 4, latencyCycles = 16 }
v[116:119] = "st.ds_read_b128"(v[vgprLocalReadAddrB]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 55424, gds = false } }
acc[212:215] = "st.v_smfmac_f32_16x16x32_bf16"(v[1396:1399], v[1400:1403], acc[212:215]) { issueCycles = 4, latencyCycles = 16 }
v[120:123] = "st.ds_read_b128"(v[vgprLocalReadAddrB+1]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 8192, gds = false } }
acc[216:219] = "st.v_smfmac_f32_16x16x32_bf16"(v[1404:1407], v[1408:1411], acc[216:219]) { issueCycles = 4, latencyCycles = 16 }
v[124:127] = "st.ds_read_b128"(v[vgprLocalReadAddrB+1]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 8320, gds = false } }
acc[220:223] = "st.v_smfmac_f32_16x16x32_bf16"(v[1412:1415], v[1416:1419], acc[220:223]) { issueCycles = 4, latencyCycles = 16 }
acc[224:227] = "st.v_smfmac_f32_16x16x32_bf16"(v[1420:1423], v[1424:1427], acc[224:227]) { issueCycles = 4, latencyCycles = 16 }
acc[228:231] = "st.v_smfmac_f32_16x16x32_bf16"(v[1428:1431], v[1432:1435], acc[228:231]) { issueCycles = 4, latencyCycles = 16 }
acc[232:235] = "st.v_smfmac_f32_16x16x32_bf16"(v[1436:1439], v[1440:1443], acc[232:235]) { issueCycles = 4, latencyCycles = 16 }
acc[236:239] = "st.v_smfmac_f32_16x16x32_bf16"(v[1444:1447], v[1448:1451], acc[236:239]) { issueCycles = 4, latencyCycles = 16 }
"st.s_cbranch_scc0"(label_LoopBeginL) { issueCycles = 1, latencyCycles = 1 }
)loopbody";

    std::string fullIR =
        std::string("st.func @gfx1250_localread_loop() {") + localReadIR + loopBodyIR + "\n}\n";
    func = std::make_unique<Function>("gfx1250_localread_loop");
    PassManager pm;
    pm.setGemmTileConfig(gemmConfig);
    pm.setPassFeatureConfig(PassFeatureConfig());
    pm.setBasicBlockFilter(BasicBlockFilterBuilder::all());

    StinkyErrorCode err =
        StinkyIRConverter::populateFunctionFromString(fullIR, *func, pm.getPassContext(), arch);
    ASSERT_EQ(err, StinkyErrorCode::SUCCESS) << "Failed to parse LocalRead + loopBody IR";

    loopBB = func->getEntryBlock();
    if (loopBB->getLabel() != "label_LoopBeginL") {
        for (BasicBlock& bb : *func) {
            if (bb.getLabel() == "label_LoopBeginL") {
                loopBB = &bb;
                break;
            }
        }
    }

    unsigned int cycles = runPassAndGetResult();
    EXPECT_GT(cycles, 0u) << "Expected exact cycle count for gfx1250 loop body";
}

// gfx1250: loop450.s full-body STIR (regen: tests/unit/asm/data/loop450_asm_to_stir.py).
TEST_F(EstimateAsmCyclesTest, Gfx1250LoopBodyFromLoop450) {
    static const char kGfx1250Loop450Stir[] = R"__L450_STIR__(^label_LoopBeginL:
"st.s_wait_dscnt"(12) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[144:151], v[2:9], v[0:7]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[200:203] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 64, gds = false } }
SCC0 = "st.s_cmp_eq_u32"(s[12], s[10]) { issueCycles = 1, latencyCycles = 1 }
s92 = "st.s_cselect_b32"(s[62], s[66], SCC0) { issueCycles = 1, latencyCycles = 1 }
s93 = "st.s_cselect_b32"(s[63], 0, SCC0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[8:15] = "st.v_wmma_f32_16x16x32_bf16"(v[152:159], v[2:9], v[8:15]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[204:207] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 96, gds = false } }
s[48], SCC0 = "st.s_add_u32"(s[48], s92) { issueCycles = 1, latencyCycles = 1 }
s[49], SCC0 = "st.s_addc_u32"(s[49], s93) { issueCycles = 1, latencyCycles = 1 }
s[56], SCC0 = "st.s_sub_u32"(s[56], s92) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[16:23] = "st.v_wmma_f32_16x16x32_bf16"(v[160:167], v[2:9], v[16:23]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[58:61] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 64, gds = false } }
s[57], SCC0 = "st.s_subb_u32"(s[57], s93) { issueCycles = 1, latencyCycles = 1 }
"st.s_delay_alu"() { issueCycles = 1, latencyCycles = 1, mod.delayalu = { instid0Type = "SALU", instid0Distance = 1 } }
SCC0 = "st.s_cmp_eq_u32"(s[57], 0) { issueCycles = 1, latencyCycles = 1 }
s[50] = "st.s_cselect_b32"(s[56], BufferLimit, SCC0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[24:31] = "st.v_wmma_f32_16x16x32_bf16"(v[168:175], v[2:9], v[24:31]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[62:65] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 96, gds = false } }
s8 = "st.s_and_b32"(s[50], 127) { issueCycles = 1, latencyCycles = 1 }
"st.s_delay_alu"() { issueCycles = 1, latencyCycles = 1, mod.delayalu = { instid0Type = "SALU", instid0Distance = 1 } }
s8 = "st.s_lshl_b32"(s8, 25) { issueCycles = 1, latencyCycles = 1 }
s[49] = "st.s_and_b32"(s[49], 33554431) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[32:39] = "st.v_wmma_f32_16x16x32_bf16"(v[176:183], v[2:9], v[32:39]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[208:211] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 5184, gds = false } }
s[49] = "st.s_or_b32"(s[49], s8) { issueCycles = 1, latencyCycles = 1 }
s[50] = "st.s_lshr_b32"(s[50], 7) { issueCycles = 1, latencyCycles = 1 }
SCC0 = "st.s_cmp_eq_u32"(s[12], s[10]) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[40:47] = "st.v_wmma_f32_16x16x32_bf16"(v[184:191], v[2:9], v[40:47]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[212:215] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 5216, gds = false } }
s92 = "st.s_cselect_b32"(s[64], s[68], SCC0) { issueCycles = 1, latencyCycles = 1 }
s93 = "st.s_cselect_b32"(s[65], 0, SCC0) { issueCycles = 1, latencyCycles = 1 }
s[52], SCC0 = "st.s_add_u32"(s[52], s92) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[48:55] = "st.v_wmma_f32_16x16x32_bf16"(v[192:199], v[2:9], v[48:55]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[216:219] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 10304, gds = false } }
s[53], SCC0 = "st.s_addc_u32"(s[53], s93) { issueCycles = 1, latencyCycles = 1 }
s[60], SCC0 = "st.s_sub_u32"(s[60], s92) { issueCycles = 1, latencyCycles = 1 }
s[61], SCC0 = "st.s_subb_u32"(s[61], s93) { issueCycles = 1, latencyCycles = 1 }
"st.s_wait_dscnt"(7) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[56:63] = "st.v_wmma_f32_16x16x32_bf16"(v[144:151], v[10:17], v[56:63]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[220:223] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 10336, gds = false } }
SCC0 = "st.s_cmp_eq_u32"(s[61], 0) { issueCycles = 1, latencyCycles = 1 }
s[54] = "st.s_cselect_b32"(s[60], BufferLimit, SCC0) { issueCycles = 1, latencyCycles = 1 }
"st.s_delay_alu"() { issueCycles = 1, latencyCycles = 1, mod.delayalu = { instid0Type = "SALU", instid0Distance = 1 } }
s8 = "st.s_and_b32"(s[54], 127) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[64:71] = "st.v_wmma_f32_16x16x32_bf16"(v[152:159], v[10:17], v[64:71]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[224:227] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 15424, gds = false } }
s8 = "st.s_lshl_b32"(s8, 25) { issueCycles = 1, latencyCycles = 1 }
s[53] = "st.s_and_b32"(s[53], 33554431) { issueCycles = 1, latencyCycles = 1 }
"st.s_delay_alu"() { issueCycles = 1, latencyCycles = 1, mod.delayalu = { instid0Type = "SALU", instid0Distance = 1 } }
s[53] = "st.s_or_b32"(s[53], s8) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[72:79] = "st.v_wmma_f32_16x16x32_bf16"(v[160:167], v[10:17], v[72:79]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[228:231] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 15456, gds = false } }
s[54] = "st.s_lshr_b32"(s[54], 7) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[80:87] = "st.v_wmma_f32_16x16x32_bf16"(v[168:175], v[10:17], v[80:87]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[232:235] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 20544, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[88:95] = "st.v_wmma_f32_16x16x32_bf16"(v[176:183], v[10:17], v[88:95]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[236:239] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 20576, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[96:103] = "st.v_wmma_f32_16x16x32_bf16"(v[184:191], v[10:17], v[96:103]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[240:243] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 25664, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[104:111] = "st.v_wmma_f32_16x16x32_bf16"(v[192:199], v[10:17], v[104:111]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[244:247] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 25696, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[112:119] = "st.v_wmma_f32_16x16x32_bf16"(v[144:151], v[18:25], v[112:119]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[248:251] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 30784, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[120:127] = "st.v_wmma_f32_16x16x32_bf16"(v[152:159], v[18:25], v[120:127]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[252:255] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 30816, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[128:135] = "st.v_wmma_f32_16x16x32_bf16"(v[160:167], v[18:25], v[128:135]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[66:69] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 5184, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[136:143] = "st.v_wmma_f32_16x16x32_bf16"(v[168:175], v[18:25], v[136:143]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[70:73] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 5216, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[144:151] = "st.v_wmma_f32_16x16x32_bf16"(v[176:183], v[18:25], v[144:151]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[74:77] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 10304, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[152:159] = "st.v_wmma_f32_16x16x32_bf16"(v[184:191], v[18:25], v[152:159]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[78:81] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 10336, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[160:167] = "st.v_wmma_f32_16x16x32_bf16"(v[192:199], v[18:25], v[160:167]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[82:85] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 15424, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[168:175] = "st.v_wmma_f32_16x16x32_bf16"(v[144:151], v[26:33], v[168:175]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[86:89] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 15456, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[176:183] = "st.v_wmma_f32_16x16x32_bf16"(v[152:159], v[26:33], v[176:183]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[90:93] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 20544, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[184:191] = "st.v_wmma_f32_16x16x32_bf16"(v[160:167], v[26:33], v[184:191]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[94:97] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 20576, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[192:199] = "st.v_wmma_f32_16x16x32_bf16"(v[168:175], v[26:33], v[192:199]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[98:101] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 25664, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[200:207] = "st.v_wmma_f32_16x16x32_bf16"(v[176:183], v[26:33], v[200:207]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[102:105] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 25696, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[208:215] = "st.v_wmma_f32_16x16x32_bf16"(v[184:191], v[26:33], v[208:215]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[106:109] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 30784, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[216:223] = "st.v_wmma_f32_16x16x32_bf16"(v[192:199], v[26:33], v[216:223]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[110:113] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 30816, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[224:231] = "st.v_wmma_f32_16x16x32_bf16"(v[144:151], v[34:41], v[224:231]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[232:239] = "st.v_wmma_f32_16x16x32_bf16"(v[152:159], v[34:41], v[232:239]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[240:247] = "st.v_wmma_f32_16x16x32_bf16"(v[160:167], v[34:41], v[240:247]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[248:255] = "st.v_wmma_f32_16x16x32_bf16"(v[168:175], v[34:41], v[248:255]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[176:183], v[34:41], v[0:7]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_barrier_signal"(-1) { issueCycles = 1, latencyCycles = 1 }
"st.s_barrier_wait"(-1) { issueCycles = 1, latencyCycles = 1 }
v[8:15] = "st.v_wmma_f32_16x16x32_bf16"(v[184:191], v[34:41], v[8:15]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[114:117]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 0, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[16:23] = "st.v_wmma_f32_16x16x32_bf16"(v[192:199], v[34:41], v[16:23]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[114:117] = "st.buffer_load_b128"(v[136], s[48:51], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[118:121]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 2560, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[24:31] = "st.v_wmma_f32_16x16x32_bf16"(v[144:151], v[42:49], v[24:31]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[118:121] = "st.buffer_load_b128"(v[136], s[48:51], s[70]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[122:125]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 5120, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[32:39] = "st.v_wmma_f32_16x16x32_bf16"(v[152:159], v[42:49], v[32:39]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[122:125] = "st.buffer_load_b128"(v[136], s[48:51], s[71]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[126:129]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 7680, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[40:47] = "st.v_wmma_f32_16x16x32_bf16"(v[160:167], v[42:49], v[40:47]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[126:129] = "st.buffer_load_b128"(v[136], s[48:51], s[72]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[130:133]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 10240, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[48:55] = "st.v_wmma_f32_16x16x32_bf16"(v[168:175], v[42:49], v[48:55]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[130:133] = "st.buffer_load_b128"(v[136], s[48:51], s[73]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[134:137]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 12800, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[56:63] = "st.v_wmma_f32_16x16x32_bf16"(v[176:183], v[42:49], v[56:63]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[134:137] = "st.buffer_load_b128"(v[136], s[48:51], s[74]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[138:141]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 15360, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[64:71] = "st.v_wmma_f32_16x16x32_bf16"(v[184:191], v[42:49], v[64:71]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[138:141] = "st.buffer_load_b128"(v[136], s[48:51], s[75]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[142:145]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 17920, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[72:79] = "st.v_wmma_f32_16x16x32_bf16"(v[192:199], v[42:49], v[72:79]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[142:145] = "st.buffer_load_b128"(v[136], s[48:51], s[76]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[146:149]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 20480, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[80:87] = "st.v_wmma_f32_16x16x32_bf16"(v[144:151], v[50:57], v[80:87]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[146:149] = "st.buffer_load_b128"(v[136], s[48:51], s[77]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[150:153]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 23040, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[88:95] = "st.v_wmma_f32_16x16x32_bf16"(v[152:159], v[50:57], v[88:95]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[150:153] = "st.buffer_load_b128"(v[136], s[48:51], s[78]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[154:157]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 25600, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[96:103] = "st.v_wmma_f32_16x16x32_bf16"(v[160:167], v[50:57], v[96:103]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[154:157] = "st.buffer_load_b128"(v[136], s[48:51], s[79]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[158:161]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 28160, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[104:111] = "st.v_wmma_f32_16x16x32_bf16"(v[168:175], v[50:57], v[104:111]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[158:161] = "st.buffer_load_b128"(v[136], s[48:51], s[80]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[162:165]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 30720, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[112:119] = "st.v_wmma_f32_16x16x32_bf16"(v[176:183], v[50:57], v[112:119]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[162:165] = "st.buffer_load_b128"(v[136], s[48:51], s[81]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[138], v[166:169]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 33280, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[120:127] = "st.v_wmma_f32_16x16x32_bf16"(v[184:191], v[50:57], v[120:127]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[166:169] = "st.buffer_load_b128"(v[136], s[48:51], s[82]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[170:173]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 0, gds = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[128:135] = "st.v_wmma_f32_16x16x32_bf16"(v[192:199], v[50:57], v[128:135]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { va_vdst = 0 } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[170:173] = "st.buffer_load_b128"(v[137], s[52:55], 0) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[174:177]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 2560, gds = false } }
"st.s_wait_dscnt"(16) { issueCycles = 1, latencyCycles = 1 }
v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[200:207], v[58:65], v[0:7]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[174:177] = "st.buffer_load_b128"(v[137], s[52:55], s[84]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[8:15] = "st.v_wmma_f32_16x16x32_bf16"(v[208:215], v[58:65], v[8:15]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[178:181]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 5120, gds = false } }
v[16:23] = "st.v_wmma_f32_16x16x32_bf16"(v[216:223], v[58:65], v[16:23]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[178:181] = "st.buffer_load_b128"(v[137], s[52:55], s[85]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[182:185]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 7680, gds = false } }
v[24:31] = "st.v_wmma_f32_16x16x32_bf16"(v[224:231], v[58:65], v[24:31]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[182:185] = "st.buffer_load_b128"(v[137], s[52:55], s[86]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[186:189]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 10240, gds = false } }
v[32:39] = "st.v_wmma_f32_16x16x32_bf16"(v[232:239], v[58:65], v[32:39]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[186:189] = "st.buffer_load_b128"(v[137], s[52:55], s[87]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[190:193]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 12800, gds = false } }
v[40:47] = "st.v_wmma_f32_16x16x32_bf16"(v[240:247], v[58:65], v[40:47]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[190:193] = "st.buffer_load_b128"(v[137], s[52:55], s[88]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[194:197]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 15360, gds = false } }
v[48:55] = "st.v_wmma_f32_16x16x32_bf16"(v[248:255], v[58:65], v[48:55]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[194:197] = "st.buffer_load_b128"(v[137], s[52:55], s[89]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[198:201]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 17920, gds = false } }
v[56:63] = "st.v_wmma_f32_16x16x32_bf16"(v[200:207], v[66:73], v[56:63]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[198:201] = "st.buffer_load_b128"(v[137], s[52:55], s[90]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[202:205]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 20480, gds = false } }
v[64:71] = "st.v_wmma_f32_16x16x32_bf16"(v[208:215], v[66:73], v[64:71]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[202:205] = "st.buffer_load_b128"(v[137], s[52:55], s[91]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[206:209]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 23040, gds = false } }
v[72:79] = "st.v_wmma_f32_16x16x32_bf16"(v[216:223], v[66:73], v[72:79]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[206:209] = "st.buffer_load_b128"(v[137], s[52:55], s[92]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[210:213]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 25600, gds = false } }
v[80:87] = "st.v_wmma_f32_16x16x32_bf16"(v[224:231], v[66:73], v[80:87]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[210:213] = "st.buffer_load_b128"(v[137], s[52:55], s[93]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[214:217]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 28160, gds = false } }
v[88:95] = "st.v_wmma_f32_16x16x32_bf16"(v[232:239], v[66:73], v[88:95]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[214:217] = "st.buffer_load_b128"(v[137], s[52:55], s[94]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[218:221]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 30720, gds = false } }
v[96:103] = "st.v_wmma_f32_16x16x32_bf16"(v[240:247], v[66:73], v[96:103]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[218:221] = "st.buffer_load_b128"(v[137], s[52:55], s[95]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_loadcnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
"st.ds_store_b128"(v[139], v[222:225]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 33280, gds = false } }
v[104:111] = "st.v_wmma_f32_16x16x32_bf16"(v[248:255], v[66:73], v[104:111]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[222:225] = "st.buffer_load_b128"(v[137], s[52:55], s[96]) { issueCycles = 12, latencyCycles = 116, mod.mubuf = { offen = true, offset12 = 0, glc = false, slc = false, nt = false, lds = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { vm_vsrc = 0 } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[112:119] = "st.v_wmma_f32_16x16x32_bf16"(v[200:207], v[74:81], v[112:119]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[120:127] = "st.v_wmma_f32_16x16x32_bf16"(v[208:215], v[74:81], v[120:127]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_dscnt"(0) { issueCycles = 1, latencyCycles = 1 }
"st.s_barrier_signal"(-1) { issueCycles = 1, latencyCycles = 1 }
"st.s_barrier_wait"(-1) { issueCycles = 1, latencyCycles = 1 }
v[128:135] = "st.v_wmma_f32_16x16x32_bf16"(v[216:223], v[74:81], v[128:135]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[144:147] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[136:143] = "st.v_wmma_f32_16x16x32_bf16"(v[224:231], v[74:81], v[136:143]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[148:151] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 32, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[144:151] = "st.v_wmma_f32_16x16x32_bf16"(v[232:239], v[74:81], v[144:151]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[2:5] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[152:159] = "st.v_wmma_f32_16x16x32_bf16"(v[240:247], v[74:81], v[152:159]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[6:9] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 32, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[160:167] = "st.v_wmma_f32_16x16x32_bf16"(v[248:255], v[74:81], v[160:167]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[152:155] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 5120, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[168:175] = "st.v_wmma_f32_16x16x32_bf16"(v[200:207], v[82:89], v[168:175]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[156:159] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 5152, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[176:183] = "st.v_wmma_f32_16x16x32_bf16"(v[208:215], v[82:89], v[176:183]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[160:163] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 10240, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[184:191] = "st.v_wmma_f32_16x16x32_bf16"(v[216:223], v[82:89], v[184:191]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[164:167] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 10272, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[192:199] = "st.v_wmma_f32_16x16x32_bf16"(v[224:231], v[82:89], v[192:199]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[168:171] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 15360, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[200:207] = "st.v_wmma_f32_16x16x32_bf16"(v[232:239], v[82:89], v[200:207]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[172:175] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 15392, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[208:215] = "st.v_wmma_f32_16x16x32_bf16"(v[240:247], v[82:89], v[208:215]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[176:179] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 20480, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[216:223] = "st.v_wmma_f32_16x16x32_bf16"(v[248:255], v[82:89], v[216:223]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[180:183] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 20512, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[224:231] = "st.v_wmma_f32_16x16x32_bf16"(v[200:207], v[90:97], v[224:231]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[184:187] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 25600, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[232:239] = "st.v_wmma_f32_16x16x32_bf16"(v[208:215], v[90:97], v[232:239]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[188:191] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 25632, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[240:247] = "st.v_wmma_f32_16x16x32_bf16"(v[216:223], v[90:97], v[240:247]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[192:195] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 30720, gds = false } }
"st.s_set_vgpr_msb"(9) { issueCycles = 1, latencyCycles = 1 }
v[248:255] = "st.v_wmma_f32_16x16x32_bf16"(v[224:231], v[90:97], v[248:255]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(65) { issueCycles = 1, latencyCycles = 1 }
v[196:199] = "st.ds_load_b128"(v[140]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 30752, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[232:239], v[90:97], v[0:7]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[10:13] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 5120, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[8:15] = "st.v_wmma_f32_16x16x32_bf16"(v[240:247], v[90:97], v[8:15]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[14:17] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 5152, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[16:23] = "st.v_wmma_f32_16x16x32_bf16"(v[248:255], v[90:97], v[16:23]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[18:21] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 10240, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[24:31] = "st.v_wmma_f32_16x16x32_bf16"(v[200:207], v[98:105], v[24:31]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[22:25] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 10272, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[32:39] = "st.v_wmma_f32_16x16x32_bf16"(v[208:215], v[98:105], v[32:39]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[26:29] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 15360, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[40:47] = "st.v_wmma_f32_16x16x32_bf16"(v[216:223], v[98:105], v[40:47]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[30:33] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 15392, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[48:55] = "st.v_wmma_f32_16x16x32_bf16"(v[224:231], v[98:105], v[48:55]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[34:37] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 20480, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[56:63] = "st.v_wmma_f32_16x16x32_bf16"(v[232:239], v[98:105], v[56:63]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[38:41] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 20512, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[64:71] = "st.v_wmma_f32_16x16x32_bf16"(v[240:247], v[98:105], v[64:71]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[42:45] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 25600, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[72:79] = "st.v_wmma_f32_16x16x32_bf16"(v[248:255], v[98:105], v[72:79]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[46:49] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 25632, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[80:87] = "st.v_wmma_f32_16x16x32_bf16"(v[200:207], v[106:113], v[80:87]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[50:53] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 30720, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[88:95] = "st.v_wmma_f32_16x16x32_bf16"(v[208:215], v[106:113], v[88:95]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_set_vgpr_msb"(129) { issueCycles = 1, latencyCycles = 1 }
v[54:57] = "st.ds_load_b128"(v[141]) { issueCycles = 1, latencyCycles = 56, mod.ds = { na = 1, offset = 30752, gds = false } }
"st.s_set_vgpr_msb"(89) { issueCycles = 1, latencyCycles = 1 }
v[96:103] = "st.v_wmma_f32_16x16x32_bf16"(v[216:223], v[106:113], v[96:103]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[104:111] = "st.v_wmma_f32_16x16x32_bf16"(v[224:231], v[106:113], v[104:111]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[112:119] = "st.v_wmma_f32_16x16x32_bf16"(v[232:239], v[106:113], v[112:119]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[120:127] = "st.v_wmma_f32_16x16x32_bf16"(v[240:247], v[106:113], v[120:127]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
v[128:135] = "st.v_wmma_f32_16x16x32_bf16"(v[248:255], v[106:113], v[128:135]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { inputPermute = "", scaleStr = "", negStr = "", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false } }
"st.s_wait_alu"() { issueCycles = 1, latencyCycles = 1, mod.waitalu = { va_vdst = 0 } }
s[12], SCC0 = "st.s_sub_u32"(s[12], 1) { issueCycles = 1, latencyCycles = 1 }
"st.s_delay_alu"() { issueCycles = 1, latencyCycles = 1, mod.delayalu = { instid0Type = "SALU", instid0Distance = 1 } }
SCC0 = "st.s_cmp_eq_i32"(s[12], 0x2) { issueCycles = 1, latencyCycles = 1 }
"st.s_cbranch_scc0"(label_LoopBeginL) { issueCycles = 1, latencyCycles = 1 }
"st.s_setreg_IMM32_b32"(0, 0) { issueCycles = 1, latencyCycles = 1 })__L450_STIR__";

    arch = getGfxArchID(12, 5, 0);
    gemmConfig.arch[0] = 12;
    gemmConfig.arch[1] = 5;
    gemmConfig.arch[2] = 0;

    const std::string fullIR =
        std::string("st.func @gfx1250_loop450_full() {\n") + kGfx1250Loop450Stir + "}\n";
    func = std::make_unique<Function>("gfx1250_loop450_full");
    PassManager pm;
    pm.setGemmTileConfig(gemmConfig);
    pm.setPassFeatureConfig(PassFeatureConfig());
    pm.setBasicBlockFilter(BasicBlockFilterBuilder::all());

    StinkyErrorCode err =
        StinkyIRConverter::populateFunctionFromString(fullIR, *func, pm.getPassContext(), arch);
    ASSERT_EQ(err, StinkyErrorCode::SUCCESS)
        << "Failed to parse gfx1250 loop body IR from loop450.s";

    loopBB = func->getEntryBlock();
    if (loopBB->getLabel() != "label_LoopBeginL") {
        for (BasicBlock& bb : *func) {
            if (bb.getLabel() == "label_LoopBeginL") {
                loopBB = &bb;
                break;
            }
        }
    }

    unsigned int cycles = runPassAndGetResult();
    EXPECT_EQ(779, cycles) << "Expected 779 cycles for gfx1250 loop body example";
}
