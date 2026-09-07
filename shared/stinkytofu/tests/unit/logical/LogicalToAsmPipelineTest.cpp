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

#include <memory>
#include <sstream>

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/ir/logical/LogicalToFunctionConverter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/transforms/logical/CompositeInstructionLoweringPass.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

using namespace stinkytofu;

/**
 * Test the complete pipeline from high-level IR to assembly string:
 * 1. Create high-level IR instructions (LogicalInstruction with shared_ptr)
 * 2. Convert PyLogicalModule to Function using LogicalToFunctionConverter
 * 3. Run logical lowering passes using unified PassManager
 * 4. Emit assembly string
 */
class IRToAsmPipelineTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Helper registers
        using namespace stinkytofu::test;
        v0 = vgpr(0);
        v1 = vgpr(1);
        v2 = vgpr(2);
        v3 = vgpr(3);
    }

    StinkyRegister v0, v1, v2, v3;
};

TEST_F(IRToAsmPipelineTest, SimpleVectorALU) {
    Function func("kernel");
    BasicBlock* entryBB = func.createBasicBlock("entry");

    LogicalInstruction* vadd = VAddF32(v0, v1, v2, std::nullopt, std::nullopt, "add two floats");
    LogicalInstruction* vmul = VMulF32(v0, v0, v3, std::nullopt, std::nullopt, "multiply result");
    entryBB->appendIR(static_cast<IRBase*>(vadd));
    entryBB->appendIR(static_cast<IRBase*>(vmul));

    PassManager pm;

    // Verify initial setup
    size_t instCount = 0;
    for (BasicBlock& bb : func) {
        instCount += bb.size();
    }
    EXPECT_EQ(instCount, 2) << "Should have 2 logical instructions in IRList";

    // Step 3: Set architecture config and run logical lowering passes
    GemmTileConfig config;
    config.arch = {12, 5, 0};  // Gfx1250
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    // Expand composite instructions
    pm.addPass(createCompositeInstructionLoweringPass());

    // Lower LogicalInstruction -> StinkyInstruction
    pm.addPass(createToStinkyAsmPass());

    pm.run(func);

    // Verify lowering (all instructions should now be StinkyInstruction)
    instCount = 0;
    size_t asmInstCount = 0;
    for (BasicBlock& bb : func) {
        for (IRBase& ir : bb) {
            instCount++;
            if (ir.getType() == IRBase::IRType::StinkyTofu) {
                asmInstCount++;
            }
        }
    }
    EXPECT_EQ(instCount, 2) << "Should still have 2 instructions";
    EXPECT_EQ(asmInstCount, 2) << "All instructions should be StinkyInstruction";

    // TODO: Step 4: Emit assembly string and verify output
}

/**
 * True16Modifiers set on a LogicalInstruction must survive logical->asm
 * lowering and render as the .l/.h operand suffix. This is the channel the
 * pure-Python rocisa_stinkytofu_adaptor uses to carry t16 half-selects (op_sel
 * would bypass stinkytofu's true16-aware SSA/wait passes).
 */
TEST_F(IRToAsmPipelineTest, True16HalfSelectSurvivesLowering) {
    using H = HighBitSel;
    Function func("kernel");
    BasicBlock* entryBB = func.createBasicBlock("entry");

    // Inject an already-derived modifier; deriving it from tagged operands is
    // tested elsewhere (adaptor _apply_true16 / attachTrue16ModifiersFromOperands).
    auto add = [&](LogicalInstruction* inst, H dst0, std::vector<H> srcs) {
        inst->true16 = True16Modifiers(dst0, H::NONE, std::move(srcs));
        entryBB->appendIR(static_cast<IRBase*>(inst));
    };

    // Binary f16 ALU (dst, src0, src1) -- HIGH and LOW both exercised.
    add(VAddF16(v0, v1, v2), H::HIGH, {H::HIGH, H::HIGH});
    add(VMulF16(v0, v1, v2), H::HIGH, {H::HIGH, H::HIGH});
    add(VMaxF16(v0, v1, v2), H::LOW, {H::LOW, H::LOW});
    add(VMinF16(v0, v1, v2), H::LOW, {H::LOW, H::LOW});
    // Unary f16 transcendentals (dst, src0).
    add(VExpF16(v0, v1), H::HIGH, {H::HIGH});
    add(VRcpF16(v0, v1), H::LOW, {H::LOW});
    // Ternary fma (dst, src0, src1, src2).
    add(VFmaF16(v0, v1, v2, v3), H::HIGH, {H::HIGH, H::HIGH, H::HIGH});
    // f16 compares (mask dst has no half; both srcs do).
    add(VCmpGTF16(v0, v1, v2), H::NONE, {H::HIGH, H::HIGH});
    add(VCmpGEF16(v0, v1, v2), H::NONE, {H::LOW, H::LOW});
    // b16 select (src2 is VCC -> no half) and b16 shift (shift amount is src0).
    add(VCndMaskB16(v0, v1, v2, v3), H::HIGH, {H::HIGH, H::HIGH});
    add(VLShiftLeftB16(v0, v1, v2), H::HIGH, {H::NONE, H::HIGH});
    // Conversions: half rides on whichever operand carries the 16-bit value.
    add(VCvtF16toF32(v0, v1), H::NONE, {H::HIGH});     // f16 src -> f32 dst
    add(VCvtF32toF16(v0, v1), H::HIGH, {H::NONE});     // f32 src -> f16 dst
    add(VCvtPkFP8toF32(v0, v1), H::NONE, {H::HIGH});   // packed-fp8 src
    add(VCvtPkBF8toF32(v0, v1), H::NONE, {H::HIGH});   // packed-bf8 src
    add(PVCvtBF16toFP32(v0, v1), H::NONE, {H::HIGH});  // bf16 src

    PassManager pm;
    GemmTileConfig config;
    config.arch = {12, 5, 0};  // Gfx1250
    config.TileA0 = 16;
    config.TileB0 = 16;
    config.TileM0 = 16;
    config.NumGRA = 4;
    config.NumGRB = 4;
    config.NumGRM = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);
    pm.addPass(createCompositeInstructionLoweringPass());
    pm.addPass(createToStinkyAsmPass());
    pm.run(func);

    StinkyAsmEmitter emitter;
    std::string asmText;
    for (BasicBlock& bb : func) {
        for (IRBase& ir : bb) {
            if (ir.getType() == IRBase::IRType::StinkyTofu) {
                asmText += emitter.emit(static_cast<StinkyInstruction&>(ir)) + "\n";
            }
        }
    }

    // Every expected line must appear verbatim; a dropped half-select would
    // silently strip the .l/.h suffix (the exact bug this guards against).
    const std::vector<std::string> expected = {
        "v_add_f16 v0.h, v1.h, v2.h",
        "v_mul_f16 v0.h, v1.h, v2.h",
        "v_max_f16 v0.l, v1.l, v2.l",
        "v_min_f16 v0.l, v1.l, v2.l",
        "v_exp_f16 v0.h, v1.h",
        "v_rcp_f16 v0.l, v1.l",
        "v_fma_f16 v0.h, v1.h, v2.h, v3.h",
        "v_cmp_gt_f16 v0, v1.h, v2.h",
        "v_cmp_ge_f16 v0, v1.l, v2.l",
        "v_cndmask_b16 v0.h, v1.h, v2.h",
        "v_lshlrev_b16 v0.h, v1, v2.h",
        "v_cvt_f32_f16 v0, v1.h",
        "v_cvt_f16_f32 v0.h, v1",
        "v_cvt_pk_f32_fp8 v0, v1.h",
        "v_cvt_pk_f32_bf8 v0, v1.h",
        "v_cvt_f32_bf16 v0, v1.h",
    };
    for (const std::string& want : expected) {
        EXPECT_NE(asmText.find(want), std::string::npos)
            << "true16 half-select missing/dropped; expected line:\n  " << want
            << "\nfull assembly:\n"
            << asmText;
    }
}
