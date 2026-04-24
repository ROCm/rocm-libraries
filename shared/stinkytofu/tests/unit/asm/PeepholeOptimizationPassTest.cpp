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

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/PeepholeOptimizationPass.hpp"

using namespace stinkytofu;

// Test fixture for PeepholeOptimizationPass
class PeepholeOptimizationPassTest : public ::testing::Test {
   protected:
    GemmTileConfig gemmConfig;
    GfxArchID arch;
    std::unique_ptr<Function> func;
    BasicBlock* bb;
    std::unique_ptr<Pass> peepholePass;
    AnalysisManager am;

    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);  // GFX1250
        gemmConfig.arch[0] = 12;
        gemmConfig.arch[1] = 5;
        gemmConfig.arch[2] = 0;

        // Create a Function with a single BasicBlock for testing
        func = std::make_unique<Function>("test_peephole");
        func->setGemmTileConfig(gemmConfig);
        bb = func->createBasicBlock("entry");

        // Create the peephole optimization pass
        peepholePass = createPeepholeOptimizationPass();
        registerAllAnalyses(am);
    }

    void TearDown() override {
        func.reset();
        bb = nullptr;
        peepholePass.reset();
    }

    // Create IRBuilder for building test instructions
    AsmIRBuilder getIRBuilder() {
        return AsmIRBuilder(*bb, arch);
    }

    // Helper to create v_fma_f32 instruction
    // v_fma_f32 dest, a, b, c  =>  dest = a * b + c
    StinkyInstruction* createVFmaF32(int destReg, int aReg, int bReg, int cReg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_fma_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", aReg, 1));
        inst->addSrcReg(StinkyRegister("v", bReg, 1));
        inst->addSrcReg(StinkyRegister("v", cReg, 1));
        return inst;
    }

    // Helper to create v_fma_f32 with constant addend
    // v_fma_f32 dest, a, b, #const  =>  dest = a * b + const
    StinkyInstruction* createVFmaF32WithConst(int destReg, int aReg, int bReg, float constVal) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_fma_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", aReg, 1));
        inst->addSrcReg(StinkyRegister("v", bReg, 1));

        // Add constant operand
        StinkyRegister constReg;
        constReg.dataType = StinkyRegister::Type::LiteralDouble;
        constReg.literalDouble = constVal;
        inst->addSrcReg(constReg);
        return inst;
    }

    // Helper to create v_fma_f16 with constant addend
    StinkyInstruction* createVFmaF16WithConst(int destReg, int aReg, int bReg, float constVal) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_fma_f16, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", aReg, 1));
        inst->addSrcReg(StinkyRegister("v", bReg, 1));

        StinkyRegister constReg;
        constReg.dataType = StinkyRegister::Type::LiteralDouble;
        constReg.literalDouble = constVal;
        inst->addSrcReg(constReg);
        return inst;
    }

    // Helper to create v_add_f32 instruction
    StinkyInstruction* createVAddF32(int destReg, int src0Reg, int src1Reg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_add_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
        return inst;
    }

    // Helper to create v_add_f32 with constant
    StinkyInstruction* createVAddF32WithConst(int destReg, float constVal, int srcReg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_add_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));

        StinkyRegister constRegister;
        constRegister.dataType = StinkyRegister::Type::LiteralDouble;
        constRegister.literalDouble = constVal;
        inst->addSrcReg(constRegister);
        inst->addSrcReg(StinkyRegister("v", srcReg, 1));
        return inst;
    }

    // Helper to create v_add_f16 with constant
    StinkyInstruction* createVAddF16WithConst(int destReg, float constVal, int srcReg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_add_f16, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));

        StinkyRegister constRegister;
        constRegister.dataType = StinkyRegister::Type::LiteralDouble;
        constRegister.literalDouble = constVal;
        inst->addSrcReg(constRegister);
        inst->addSrcReg(StinkyRegister("v", srcReg, 1));
        return inst;
    }

    // Helper to create v_add_f32 with swapped operands (register first, constant second)
    // Useful for testing commutative pattern matching
    StinkyInstruction* createVAddF32WithConstSwapped(int destReg, int srcReg, float constVal) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_add_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", srcReg, 1));

        StinkyRegister constRegister;
        constRegister.dataType = StinkyRegister::Type::LiteralDouble;
        constRegister.literalDouble = constVal;
        inst->addSrcReg(constRegister);
        return inst;
    }

    // Helper to create v_mul_f32 instruction
    StinkyInstruction* createVMulF32(int destReg, int src0Reg, int src1Reg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_mul_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
        return inst;
    }

    // Helper to create v_mul_f32 with constant operand
    StinkyInstruction* createVMulF32WithConst(int destReg, float constVal, int srcReg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_mul_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));

        StinkyRegister constRegister;
        constRegister.dataType = StinkyRegister::Type::LiteralDouble;
        constRegister.literalDouble = constVal;
        inst->addSrcReg(constRegister);
        inst->addSrcReg(StinkyRegister("v", srcReg, 1));
        return inst;
    }

    // Helper to create v_mul_f16
    StinkyInstruction* createVMulF16(int destReg, int src0Reg, int src1Reg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_mul_f16, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", src0Reg, 1));
        inst->addSrcReg(StinkyRegister("v", src1Reg, 1));
        return inst;
    }

    // Helper to create v_mul_f16 with constant operand
    StinkyInstruction* createVMulF16WithConst(int destReg, float constVal, int srcReg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_mul_f16, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));

        StinkyRegister constRegister;
        constRegister.dataType = StinkyRegister::Type::LiteralDouble;
        constRegister.literalDouble = constVal;
        inst->addSrcReg(constRegister);
        inst->addSrcReg(StinkyRegister("v", srcReg, 1));
        return inst;
    }

    // Helper to create v_fma_f16 (base version)
    StinkyInstruction* createVFmaF16(int destReg, int aReg, int bReg, int cReg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_fma_f16, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", aReg, 1));
        inst->addSrcReg(StinkyRegister("v", bReg, 1));
        inst->addSrcReg(StinkyRegister("v", cReg, 1));
        return inst;
    }

    // Helper to create v_mov_b32
    StinkyInstruction* createVMovB32(int destReg, int srcReg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_mov_b32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", srcReg, 1));
        return inst;
    }

    // Count instructions in the IRList
    size_t countInstructions() {
        size_t count = 0;
        for (auto& inst : *bb) {
            (void)inst;  // Unused, just counting
            count++;
        }
        return count;
    }

    // Run the peephole pass
    void runPass() {
        PassContext ctx;
        ctx.setGemmTileConfig(gemmConfig);

        // Build use-def chains before running the pass (standalone test mode)
        // In production, OptimizationPipeline builds this once at the start
        buildUseDefChain(*func, true);

        peepholePass->run(*func, ctx, am);
    }
};

// ============================================================================
// Pattern 1: Add+FMA Fusion (F32) - In-place variant
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, AddFMAFusion_F32_InPlace) {
    // Create pattern:
    //   v_fma_f32 v0, v1, v2, 1.0     // v0 = v1 * v2 + 1.0
    //   v_add_f32 v0, 1.0, v0         // v0 = 1.0 + v0
    // Expected after optimization:
    //   v_fma_f32 v0, v1, v2, 2.0     // v0 = v1 * v2 + 2.0

    createVFmaF32WithConst(0, 1, 2, 1.0f);
    createVAddF32WithConst(0, 1.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // After fusion, should have only 1 instruction
    EXPECT_EQ(countInstructions(), 1);

    // The remaining instruction should be v_fma_f32 with constant 2.0
    auto& inst = *bb->begin();
    auto* fma = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(fma, nullptr);
    EXPECT_EQ(fma->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f32));
    EXPECT_EQ(fma->getDestRegs()[0].reg.idx, 0u);
    EXPECT_EQ(fma->getSrcRegs()[0].reg.idx, 1u);
    EXPECT_EQ(fma->getSrcRegs()[1].reg.idx, 2u);
    EXPECT_EQ(fma->getSrcRegs()[2].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_FLOAT_EQ(fma->getSrcRegs()[2].literalDouble, 2.0f);
}

TEST_F(PeepholeOptimizationPassTest, AddFMAFusion_F32_NonInPlace) {
    // Create pattern:
    //   v_fma_f32 v0, v1, v2, 1.5     // v0 = v1 * v2 + 1.5
    //   v_add_f32 v3, 0.5, v0         // v3 = 0.5 + v0
    // Expected after optimization:
    //   v_fma_f32 v3, v1, v2, 2.0     // v3 = v1 * v2 + 2.0

    createVFmaF32WithConst(0, 1, 2, 1.5f);
    createVAddF32WithConst(3, 0.5f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(fma, nullptr);
    EXPECT_EQ(fma->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f32));
    EXPECT_EQ(fma->getDestRegs()[0].reg.idx, 3u);  // Destination changed to v3
    EXPECT_EQ(fma->getSrcRegs()[0].reg.idx, 1u);
    EXPECT_EQ(fma->getSrcRegs()[1].reg.idx, 2u);
    EXPECT_EQ(fma->getSrcRegs()[2].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_FLOAT_EQ(fma->getSrcRegs()[2].literalDouble, 2.0f);
}

// ============================================================================
// Pattern 2: Add+FMA Fusion (F16)
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, AddFMAFusion_F16_InPlace) {
    // Create pattern:
    //   v_fma_f16 v10, v11, v12, 0.5   // v10 = v11 * v12 + 0.5
    //   v_add_f16 v10, 0.5, v10        // v10 = 0.5 + v10
    // Expected after optimization:
    //   v_fma_f16 v10, v11, v12, 1.0   // v10 = v11 * v12 + 1.0

    createVFmaF16WithConst(10, 11, 12, 0.5f);
    createVAddF16WithConst(10, 0.5f, 10);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(fma, nullptr);
    EXPECT_EQ(fma->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f16));
    EXPECT_EQ(fma->getDestRegs()[0].reg.idx, 10u);
    EXPECT_EQ(fma->getSrcRegs()[2].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_FLOAT_EQ(fma->getSrcRegs()[2].literalDouble, 1.0f);
}

TEST_F(PeepholeOptimizationPassTest, AddFMAFusion_F16_NonInPlace) {
    // Create pattern:
    //   v_fma_f16 v5, v6, v7, 2.0     // v5 = v6 * v7 + 2.0
    //   v_add_f16 v8, 3.0, v5         // v8 = 3.0 + v5
    // Expected after optimization:
    //   v_fma_f16 v8, v6, v7, 5.0     // v8 = v6 * v7 + 5.0

    createVFmaF16WithConst(5, 6, 7, 2.0f);
    createVAddF16WithConst(8, 3.0f, 5);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(fma, nullptr);
    EXPECT_EQ(fma->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f16));
    EXPECT_EQ(fma->getDestRegs()[0].reg.idx, 8u);
    EXPECT_EQ(fma->getSrcRegs()[2].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_FLOAT_EQ(fma->getSrcRegs()[2].literalDouble, 5.0f);
}

// ============================================================================
// Negative Tests: Pattern Should NOT Match
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, NoFusion_FMAResultHasMultipleUses) {
    // Create pattern:
    //   v_fma_f32 v0, v1, v2, 1.0
    //   v_add_f32 v3, 1.0, v0
    //   v_mul_f32 v4, v0, v0       // v0 used again!
    // Should NOT fuse because v0 has multiple uses

    createVFmaF32WithConst(0, 1, 2, 1.0f);
    createVAddF32WithConst(3, 1.0f, 0);
    createVMulF32(4, 0, 0);

    ASSERT_EQ(countInstructions(), 3);

    runPass();

    // Should still have 3 instructions
    EXPECT_EQ(countInstructions(), 3);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_FMAConstantMissing) {
    // Create pattern:
    //   v_fma_f32 v0, v1, v2, v3     // Uses register v3, not constant
    //   v_add_f32 v0, 1.0, v0
    // Should NOT fuse because FMA doesn't have constant addend

    createVFmaF32(0, 1, 2, 3);  // All registers, no constant
    createVAddF32WithConst(0, 1.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should still have 2 instructions
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_AddConstantMissing) {
    // Create pattern:
    //   v_fma_f32 v0, v1, v2, 1.0
    //   v_add_f32 v0, v3, v0         // Uses register v3, not constant
    // Should NOT fuse because ADD doesn't have constant addend

    createVFmaF32WithConst(0, 1, 2, 1.0f);
    createVAddF32(0, 3, 0);  // All registers, no constant

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should still have 2 instructions
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_WrongOrder) {
    // Create pattern with wrong order:
    //   v_add_f32 v0, 1.0, v1        // ADD comes first
    //   v_fma_f32 v1, v2, v3, 1.0    // FMA comes second
    // Should NOT fuse because ADD comes before FMA

    createVAddF32WithConst(0, 1.0f, 1);
    createVFmaF32WithConst(1, 2, 3, 1.0f);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should still have 2 instructions
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, Fusion_NonConsecutive) {
    // Create pattern with intervening instruction:
    //   v_fma_f32 v0, v1, v2, 1.0
    //   v_mul_f32 v5, v6, v7         // Intervening instruction (doesn't touch v0)
    //   v_add_f32 v0, 1.0, v0
    //
    // This SHOULD fuse! The intervening v_mul doesn't affect v0, so it's safe.
    // Modern peephole optimization doesn't require strict consecutiveness -
    // it uses def-use analysis to determine safety.

    createVFmaF32WithConst(0, 1, 2, 1.0f);
    createVMulF32(5, 6, 7);
    createVAddF32WithConst(0, 1.0f, 0);

    ASSERT_EQ(countInstructions(), 3);

    runPass();

    // After fusion: v_fma_f32 + v_mul_f32 (ADD removed)
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_InterveningDefOverwrites) {
    // Create pattern with intervening instruction that OVERWRITES the FMA result:
    //   v_fma_f32 v0, v1, v2, 1.0    // Defines v0
    //   v_mul_f32 v0, v6, v7         // Overwrites v0!
    //   v_add_f32 v3, 1.0, v0        // Uses v0 (from v_mul, not v_fma!)
    //
    // This should NOT fuse because the ADD uses the v_mul result, not the v_fma result

    createVFmaF32WithConst(0, 1, 2, 1.0f);
    createVMulF32(0, 6, 7);  // Overwrites v0!
    createVAddF32WithConst(3, 1.0f, 0);

    ASSERT_EQ(countInstructions(), 3);

    runPass();

    // No fusion should occur
    EXPECT_EQ(countInstructions(), 3);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, EmptyIRList) {
    // Empty IR list should not crash
    ASSERT_EQ(countInstructions(), 0);

    runPass();

    EXPECT_EQ(countInstructions(), 0);
}

TEST_F(PeepholeOptimizationPassTest, SingleInstruction) {
    // Single instruction should not crash
    createVFmaF32WithConst(0, 1, 2, 1.0f);

    ASSERT_EQ(countInstructions(), 1);

    runPass();

    EXPECT_EQ(countInstructions(), 1);
}

TEST_F(PeepholeOptimizationPassTest, NegativeConstants) {
    // Test with negative constants
    //   v_fma_f32 v0, v1, v2, -1.0
    //   v_add_f32 v0, -2.0, v0
    // Expected: v_fma_f32 v0, v1, v2, -3.0

    createVFmaF32WithConst(0, 1, 2, -1.0f);
    createVAddF32WithConst(0, -2.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(fma, nullptr);
    EXPECT_EQ(fma->getSrcRegs()[2].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_FLOAT_EQ(fma->getSrcRegs()[2].literalDouble, -3.0f);
}

TEST_F(PeepholeOptimizationPassTest, ZeroConstants) {
    // Test with zero constants
    //   v_fma_f32 v0, v1, v2, 0.0
    //   v_add_f32 v0, 0.0, v0
    // Expected: v_fma_f32 v0, v1, v2, 0.0

    createVFmaF32WithConst(0, 1, 2, 0.0f);
    createVAddF32WithConst(0, 0.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(fma, nullptr);
    EXPECT_EQ(fma->getSrcRegs()[2].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_FLOAT_EQ(fma->getSrcRegs()[2].literalDouble, 0.0f);
}

// ============================================================================
// Multiple Patterns in Same Block
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, MultipleFusionOpportunities) {
    // Create two independent fusion opportunities:
    //   v_fma_f32 v0, v1, v2, 1.0
    //   v_add_f32 v0, 1.0, v0
    //   v_fma_f32 v10, v11, v12, 2.0
    //   v_add_f32 v10, 3.0, v10
    // Expected: Both should fuse

    createVFmaF32WithConst(0, 1, 2, 1.0f);
    createVAddF32WithConst(0, 1.0f, 0);
    createVFmaF32WithConst(10, 11, 12, 2.0f);
    createVAddF32WithConst(10, 3.0f, 10);

    ASSERT_EQ(countInstructions(), 4);

    runPass();

    // Should have 2 instructions after both fusions
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, MixedF32AndF16) {
    // Create one F32 and one F16 fusion:
    //   v_fma_f32 v0, v1, v2, 1.0
    //   v_add_f32 v0, 1.0, v0
    //   v_fma_f16 v10, v11, v12, 0.5
    //   v_add_f16 v10, 0.5, v10
    // Expected: Both should fuse

    createVFmaF32WithConst(0, 1, 2, 1.0f);
    createVAddF32WithConst(0, 1.0f, 0);
    createVFmaF16WithConst(10, 11, 12, 0.5f);
    createVAddF16WithConst(10, 0.5f, 10);

    ASSERT_EQ(countInstructions(), 4);

    runPass();

    EXPECT_EQ(countInstructions(), 2);
}

// ============================================================================
// Register Reuse Tests
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, Fusion_WithRegisterReuse) {
    // Test the fix for def-use analysis with register reuse (GitHub issue #XXX)
    //
    // This tests the scenario from GELU where the same register is redefined
    // multiple times:
    //   v_fma_f32 v10, v1, v2, 1.0      // def #1 at pos 0
    //   v_add_f32 v10, 1.0, v10         // def #2 at pos 1, uses def #1
    //   v_mul_f32 v10, v3, v10          // def #3 at pos 2, uses def #2 (NOT def #1!)
    //   v_mul_f32 v5, v4, v10           // uses def #3 (NOT def #1!)
    //
    // The def-use analysis must recognize that def #1 has only ONE use (at pos 1),
    // not THREE uses. Instructions at pos 2 and 3 use later definitions of v10.
    //
    // Expected: FMA+ADD fusion should succeed because def #1 has exactly one use.

    createVFmaF32WithConst(10, 1, 2, 1.0f);  // pos 0: v10 = v1*v2 + 1.0
    createVAddF32WithConst(10, 1.0f, 10);    // pos 1: v10 = 1.0 + v10  (uses pos 0's v10)
    createVMulF32(10, 3, 10);  // pos 2: v10 = v3 * v10   (uses pos 1's v10, redefines)
    createVMulF32(5, 4, 10);   // pos 3: v5  = v4 * v10   (uses pos 2's v10)

    ASSERT_EQ(countInstructions(), 4);

    runPass();

    // After fusion: FMA+ADD should fuse into one instruction
    // Remaining: fused FMA (pos 0), v_mul (pos 2), v_mul (pos 3) = 3 instructions
    EXPECT_EQ(countInstructions(), 3);

    // Verify the fused instruction has the correct constant
    auto it = bb->begin();
    auto* fma = static_cast<StinkyInstruction*>(&*it);
    EXPECT_EQ(fma->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f32));
    EXPECT_EQ(fma->getDestRegs()[0].reg.idx, 10u);
    EXPECT_FLOAT_EQ(fma->getSrcRegs()[2].literalDouble, 2.0f);  // 1.0 + 1.0 = 2.0
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_RegisterReuseWithMultipleUsesBeforeRedef) {
    // Test that fusion DOESN'T happen when the definition truly has multiple uses
    // before being redefined:
    //   v_fma_f32 v10, v1, v2, 1.0      // def #1
    //   v_mul_f32 v5, v10, v10          // use #1 of def #1
    //   v_add_f32 v6, 1.0, v10          // use #2 of def #1 (multiple uses!)
    //   v_mul_f32 v10, v3, v4           // redefines v10
    //
    // The FMA result has TWO uses before being redefined, so fusion should NOT happen.

    createVFmaF32WithConst(10, 1, 2, 1.0f);  // def v10
    createVMulF32(5, 10, 10);                // use v10 twice
    createVAddF32WithConst(6, 1.0f, 10);     // use v10 again (3rd use total)
    createVMulF32(10, 3, 4);                 // redefine v10

    ASSERT_EQ(countInstructions(), 4);

    runPass();

    // No fusion should occur - v10 from FMA has 3 uses before redefinition
    EXPECT_EQ(countInstructions(), 4);
}

// ============================================================================
// Integration Test with PassManager
// ============================================================================

class PeepholePassManagerTest : public ::testing::Test, public stinkytofu::PassManager {
   protected:
    stinkytofu::Function func{"kernel"};
    GemmTileConfig gemmConfig;
    GfxArchID arch;
    BasicBlock* bb;

    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);  // GFX1250
        gemmConfig.arch[0] = 12;
        gemmConfig.arch[1] = 5;
        gemmConfig.arch[2] = 0;

        func.setGemmTileConfig(gemmConfig);
        bb = func.createBasicBlock("entry");

        setGemmTileConfig(gemmConfig);
        registerAllAnalyses(getAnalysisManager());
    }

    void TearDown() override {
        bb = nullptr;
    }

    AsmIRBuilder getIRBuilder() {
        return AsmIRBuilder(*bb, arch);
    }

    StinkyInstruction* createVFmaF32WithConst(int destReg, int aReg, int bReg, float constVal) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_fma_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));
        inst->addSrcReg(StinkyRegister("v", aReg, 1));
        inst->addSrcReg(StinkyRegister("v", bReg, 1));

        StinkyRegister constReg;
        constReg.dataType = StinkyRegister::Type::LiteralDouble;
        constReg.literalDouble = constVal;
        inst->addSrcReg(constReg);
        return inst;
    }

    StinkyInstruction* createVAddF32WithConst(int destReg, float constVal, int srcReg) {
        auto builder = getIRBuilder();
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_add_f32, arch));

        inst->addDestReg(StinkyRegister("v", destReg, 1));

        StinkyRegister constRegister;
        constRegister.dataType = StinkyRegister::Type::LiteralDouble;
        constRegister.literalDouble = constVal;
        inst->addSrcReg(constRegister);
        inst->addSrcReg(StinkyRegister("v", srcReg, 1));
        return inst;
    }

    size_t countInstructions() {
        size_t count = 0;
        for (auto& inst : *bb) {
            (void)inst;
            count++;
        }
        return count;
    }
};

//===----------------------------------------------------------------------===//
// Test: Commutative Pattern Matching
//===----------------------------------------------------------------------===//

TEST_F(PeepholeOptimizationPassTest, Fusion_CommutativeOperandOrder) {
    // Test that v_add_f32 fusion works when operands are swapped
    // Pattern expects: v_add_f32 $dst, $const, $reg
    // This test has:   v_add_f32 $dst, $reg, $const (swapped!)
    // Should still match because v_add_f32 is commutative

    createVFmaF32WithConst(10, 1, 2, 1.0f);       // v10 = v1*v2 + 1.0
    createVAddF32WithConstSwapped(11, 10, 1.0f);  // v11 = v10 + 1.0 (operands swapped!)

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should fuse: v_fma_f32 v11, v1, v2, 2.0
    EXPECT_EQ(countInstructions(), 1);

    // Verify the FMA has the correct folded constant
    auto& irList = *bb;
    auto it = irList.begin();
    ASSERT_NE(it, irList.end());

    StinkyInstruction* fma = cast<StinkyInstruction>(&(*it));
    EXPECT_EQ(fma->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f32));

    // Check destination
    ASSERT_EQ(fma->getDestRegs().size(), 1);
    EXPECT_EQ(fma->getDestRegs()[0].reg.idx, 11u);

    // Check constant is folded: 1.0 + 1.0 = 2.0
    ASSERT_EQ(fma->getSrcRegs().size(), 3);
    EXPECT_TRUE(fma->getSrcRegs()[2].dataType == StinkyRegister::Type::LiteralDouble);
    EXPECT_DOUBLE_EQ(fma->getSrcRegs()[2].literalDouble, 2.0);
}

TEST_F(PeepholeOptimizationPassTest, Fusion_CommutativeInPlace) {
    // Test commutative matching with in-place operation
    // v_fma_f32 v10, v1, v2, 1.0
    // v_add_f32 v10, v10, 1.0  (register first, constant second - swapped!)

    createVFmaF32WithConst(10, 1, 2, 1.0f);       // v10 = v1*v2 + 1.0
    createVAddF32WithConstSwapped(10, 10, 1.0f);  // v10 = v10 + 1.0 (swapped, in-place!)

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should fuse: v_fma_f32 v10, v1, v2, 2.0
    EXPECT_EQ(countInstructions(), 1);

    auto& irList = *bb;
    auto it = irList.begin();
    ASSERT_NE(it, irList.end());

    StinkyInstruction* fma = cast<StinkyInstruction>(&(*it));
    EXPECT_EQ(fma->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f32));

    // Check destination is v10
    ASSERT_EQ(fma->getDestRegs().size(), 1);
    EXPECT_EQ(fma->getDestRegs()[0].reg.idx, 10u);

    // Check constant is folded
    ASSERT_EQ(fma->getSrcRegs().size(), 3);
    EXPECT_DOUBLE_EQ(fma->getSrcRegs()[2].literalDouble, 2.0);
}

//===----------------------------------------------------------------------===//
// Test: MUL+MUL Fusion
//===----------------------------------------------------------------------===//

TEST_F(PeepholeOptimizationPassTest, MULMULFusion_F32_Basic) {
    // Test basic MUL+MUL fusion with constant folding
    // v_mul_f32 v0, 2.0, v1     // v0 = 2.0 * v1
    // v_mul_f32 v2, 3.0, v0     // v2 = 3.0 * v0
    // => v_mul_f32 v2, 6.0, v1  // v2 = 6.0 * v1 (folded: 2.0 * 3.0 = 6.0)

    createVMulF32WithConst(0, 2.0f, 1);  // v0 = 2.0 * v1
    createVMulF32WithConst(2, 3.0f, 0);  // v2 = 3.0 * v0

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // After fusion, should have only 1 instruction
    EXPECT_EQ(countInstructions(), 1);

    // Verify the result: v_mul_f32 v2, 6.0, v1
    auto& inst = *bb->begin();
    auto* mul = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(mul, nullptr);
    EXPECT_EQ(mul->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_mul_f32));
    EXPECT_EQ(mul->getDestRegs()[0].reg.idx, 2u);  // Destination is v2
    EXPECT_EQ(mul->getSrcRegs()[1].reg.idx, 1u);   // Source is v1
    EXPECT_EQ(mul->getSrcRegs()[0].dataType, StinkyRegister::Type::LiteralDouble);
    EXPECT_FLOAT_EQ(mul->getSrcRegs()[0].literalDouble, 6.0f);  // 2.0 * 3.0 = 6.0
}

TEST_F(PeepholeOptimizationPassTest, MULMULFusion_F32_InPlace) {
    // Test in-place MUL+MUL fusion
    // v_mul_f32 v0, 2.0, v1     // v0 = 2.0 * v1
    // v_mul_f32 v0, 3.0, v0     // v0 = 3.0 * v0 (in-place)
    // => v_mul_f32 v0, 6.0, v1  // v0 = 6.0 * v1

    createVMulF32WithConst(0, 2.0f, 1);  // v0 = 2.0 * v1
    createVMulF32WithConst(0, 3.0f, 0);  // v0 = 3.0 * v0 (in-place)

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* mul = static_cast<StinkyInstruction*>(&inst);
    EXPECT_EQ(mul->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_mul_f32));
    EXPECT_EQ(mul->getDestRegs()[0].reg.idx, 0u);  // Destination is v0
    EXPECT_EQ(mul->getSrcRegs()[1].reg.idx, 1u);   // Source is v1
    EXPECT_FLOAT_EQ(mul->getSrcRegs()[0].literalDouble, 6.0f);
}

TEST_F(PeepholeOptimizationPassTest, MULMULFusion_F16_Basic) {
    // Test MUL+MUL fusion for F16
    // v_mul_f16 v0, 2.5, v1     // v0 = 2.5 * v1
    // v_mul_f16 v2, 4.0, v0     // v2 = 4.0 * v0
    // => v_mul_f16 v2, 10.0, v1 // v2 = 10.0 * v1 (folded: 2.5 * 4.0 = 10.0)

    createVMulF16WithConst(0, 2.5f, 1);  // v0 = 2.5 * v1
    createVMulF16WithConst(2, 4.0f, 0);  // v2 = 4.0 * v0

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* mul = static_cast<StinkyInstruction*>(&inst);
    EXPECT_EQ(mul->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_mul_f16));
    EXPECT_EQ(mul->getDestRegs()[0].reg.idx, 2u);
    EXPECT_EQ(mul->getSrcRegs()[1].reg.idx, 1u);
    EXPECT_FLOAT_EQ(mul->getSrcRegs()[0].literalDouble, 10.0f);  // 2.5 * 4.0 = 10.0
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_MUL_MultipleUses) {
    // First MUL result has multiple uses - should NOT fuse
    // v_mul_f32 v0, 2.0, v1     // v0 = 2.0 * v1
    // v_mul_f32 v2, 3.0, v0     // v2 = 3.0 * v0
    // v_mul_f32 v3, 4.0, v0     // v3 = 4.0 * v0 (v0 used again!)

    createVMulF32WithConst(0, 2.0f, 1);  // v0 = 2.0 * v1
    createVMulF32WithConst(2, 3.0f, 0);  // v2 = 3.0 * v0
    createVMulF32WithConst(3, 4.0f, 0);  // v3 = 4.0 * v0 (v0 has 2 uses)

    ASSERT_EQ(countInstructions(), 3);

    runPass();

    // Should NOT fuse because v0 has multiple uses
    EXPECT_EQ(countInstructions(), 3);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_MUL_MissingConstant) {
    // Second MUL doesn't have a constant - should NOT fuse
    // v_mul_f32 v0, 2.0, v1     // v0 = 2.0 * v1
    // v_mul_f32 v2, v3, v0      // v2 = v3 * v0 (no constant!)

    createVMulF32WithConst(0, 2.0f, 1);  // v0 = 2.0 * v1
    createVMulF32(2, 3, 0);              // v2 = v3 * v0 (no constant)

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should NOT fuse because second MUL doesn't have constant
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, MULMULFusion_WithZero) {
    // Test constant folding with zero
    // v_mul_f32 v0, 5.0, v1     // v0 = 5.0 * v1
    // v_mul_f32 v2, 0.0, v0     // v2 = 0.0 * v0
    // => v_mul_f32 v2, 0.0, v1  // v2 = 0.0 * v1 (folded: 5.0 * 0.0 = 0.0)

    createVMulF32WithConst(0, 5.0f, 1);  // v0 = 5.0 * v1
    createVMulF32WithConst(2, 0.0f, 0);  // v2 = 0.0 * v0

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* mul = static_cast<StinkyInstruction*>(&inst);
    EXPECT_FLOAT_EQ(mul->getSrcRegs()[0].literalDouble, 0.0f);  // 5.0 * 0.0 = 0.0
}

TEST_F(PeepholeOptimizationPassTest, MULMULFusion_Negative) {
    // Test constant folding with negative values
    // v_mul_f32 v0, -2.0, v1    // v0 = -2.0 * v1
    // v_mul_f32 v2, 3.0, v0     // v2 = 3.0 * v0
    // => v_mul_f32 v2, -6.0, v1 // v2 = -6.0 * v1 (folded: -2.0 * 3.0 = -6.0)

    createVMulF32WithConst(0, -2.0f, 1);  // v0 = -2.0 * v1
    createVMulF32WithConst(2, 3.0f, 0);   // v2 = 3.0 * v0

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* mul = static_cast<StinkyInstruction*>(&inst);
    EXPECT_FLOAT_EQ(mul->getSrcRegs()[0].literalDouble, -6.0f);  // -2.0 * 3.0 = -6.0
}

//===----------------------------------------------------------------------===//
// Test: MUL+FMA Fusion
//===----------------------------------------------------------------------===//

TEST_F(PeepholeOptimizationPassTest, MULFMAFusion_F32_Basic) {
    // Pattern: v_mul followed by v_fma where MUL result is used in FMA
    // v_mul_f32 v0, 2.0, v1    // v0 = 2.0 * v1
    // v_fma_f32 v2, 3.0, v0, v3 // v2 = 3.0 * v0 + v3 = 3.0 * (2.0 * v1) + v3
    // => v_fma_f32 v2, 6.0, v1, v3 // v2 = 6.0 * v1 + v3 (folded: 2.0 * 3.0 = 6.0)

    createVMulF32WithConst(0, 2.0f, 1);  // v0 = 2.0 * v1

    // Create v_fma_f32 v2, 3.0, v0, v3
    auto* fma = createVFmaF32(2, 10, 0, 3);  // Base FMA: v2 = v10 * v0 + v3
    {
        auto regs = fma->getSrcRegs();
        regs[0] = StinkyRegister(3.0);
        fma->setSrcRegs(regs);
    }  // Change to: v2 = 3.0 * v0 + v3

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should fuse into single FMA with folded constant
    EXPECT_EQ(countInstructions(), 1);

    auto& irList = *bb;
    auto it = irList.begin();
    ASSERT_NE(it, irList.end());

    StinkyInstruction* fma_result = cast<StinkyInstruction>(&(*it));
    EXPECT_EQ(fma_result->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f32));

    // Check destination is v2
    ASSERT_EQ(fma_result->getDestRegs().size(), 1);
    EXPECT_EQ(fma_result->getDestRegs()[0].reg.idx, 2u);

    // Check constant is folded: 2.0 * 3.0 = 6.0
    ASSERT_EQ(fma_result->getSrcRegs().size(), 3);
    EXPECT_TRUE(fma_result->getSrcRegs()[0].dataType == StinkyRegister::Type::LiteralDouble);
    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[0].literalDouble, 6.0f);

    // Check that v1 is the operand (from original MUL)
    EXPECT_EQ(fma_result->getSrcRegs()[1].reg.idx, 1u);

    // Check that v3 is preserved as the addend
    EXPECT_EQ(fma_result->getSrcRegs()[2].reg.idx, 3u);
}

TEST_F(PeepholeOptimizationPassTest, MULFMAFusion_F32_Commutative_MulOperand) {
    // Test commutative matching: MUL result used in different position of FMA multiply
    // v_mul_f32 v0, 2.0, v1    // v0 = 2.0 * v1
    // v_fma_f32 v2, v0, 3.0, v3 // v2 = v0 * 3.0 + v3 (MUL result in first position!)
    // Should still match because FMA multiply operands are commutative

    createVMulF32WithConst(0, 2.0f, 1);  // v0 = 2.0 * v1

    // Create v_fma_f32 v2, v0, 3.0, v3
    auto* fma = createVFmaF32(2, 0, 10, 3);  // v2 = v0 * v10 + v3
    {
        auto regs = fma->getSrcRegs();
        regs[1] = StinkyRegister(3.0);
        fma->setSrcRegs(regs);
    }  // Change to: v2 = v0 * 3.0 + v3

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should fuse
    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma_result = static_cast<StinkyInstruction*>(&inst);

    // Check constant is folded: 2.0 * 3.0 = 6.0
    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[0].literalDouble, 6.0f);
}

// NOTE: Commutative matching only works for the ROOT instruction being matched,
// not for producer instructions found via def-use chains. This matches LLVM behavior.
// Test removed: MULFMAFusion_F32_Commutative_MulConstant

TEST_F(PeepholeOptimizationPassTest, MULFMAFusion_F16_Basic) {
    // Test F16 variant
    // v_mul_f16 v0, 2.0, v1
    // v_fma_f16 v2, 3.0, v0, v3
    // => v_fma_f16 v2, 6.0, v1, v3

    createVMulF16WithConst(0, 2.0f, 1);  // v0 = 2.0 * v1

    auto* fma = createVFmaF16(2, 10, 0, 3);  // v2 = v10 * v0 + v3
    {
        auto regs = fma->getSrcRegs();
        regs[0] = StinkyRegister(3.0);
        fma->setSrcRegs(regs);
    }  // Change: v2 = 3.0 * v0 + v3

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma_result = static_cast<StinkyInstruction*>(&inst);

    EXPECT_EQ(fma_result->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_fma_f16));
    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[0].literalDouble, 6.0f);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_MULFMA_MultipleUses) {
    // MUL result has multiple uses - should NOT fuse
    // v_mul_f32 v0, 2.0, v1
    // v_fma_f32 v2, 3.0, v0, v3
    // v_add_f32 v4, v0, v5        // v0 used again!

    createVMulF32WithConst(0, 2.0f, 1);

    auto* fma = createVFmaF32(2, 10, 0, 3);
    {
        auto regs = fma->getSrcRegs();
        regs[0] = StinkyRegister(3.0);
        fma->setSrcRegs(regs);
    }

    // Extra use of v0
    createVAddF32(4, 0, 5);

    ASSERT_EQ(countInstructions(), 3);

    runPass();

    // Should NOT fuse - still 3 instructions
    EXPECT_EQ(countInstructions(), 3);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_MULFMA_MissingConstant) {
    // MUL doesn't have constant - should NOT fuse
    // v_mul_f32 v0, v1, v2        // No constant!
    // v_fma_f32 v3, 3.0, v0, v4

    createVMulF32(0, 1, 2);  // v0 = v1 * v2 (no constant)

    auto* fma = createVFmaF32(3, 10, 0, 4);
    {
        auto regs = fma->getSrcRegs();
        regs[0] = StinkyRegister(3.0);
        fma->setSrcRegs(regs);
    }

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should NOT fuse
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, MULFMAFusion_WithNegative) {
    // Test with negative constants
    // v_mul_f32 v0, -2.0, v1
    // v_fma_f32 v2, 3.0, v0, v3
    // => v_fma_f32 v2, -6.0, v1, v3

    createVMulF32WithConst(0, -2.0f, 1);

    auto* fma = createVFmaF32(2, 10, 0, 3);
    {
        auto regs = fma->getSrcRegs();
        regs[0] = StinkyRegister(3.0);
        fma->setSrcRegs(regs);
    }

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma_result = static_cast<StinkyInstruction*>(&inst);

    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[0].literalDouble, -6.0f);
}

// ============================================================================
// ADD+MUL Fusion Tests
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, ADDMULFusion_F32_Basic) {
    // Basic ADD+MUL fusion with constant folding
    // v_add_f32 v0, 2.0, v1        // v0 = 2.0 + v1
    // v_mul_f32 v2, 3.0, v0        // v2 = 3.0 * v0
    // => v_fma_f32 v2, 3.0, v1, 6.0  // v2 = 3.0 * v1 + 6.0
    // (const2 * const1 = 3.0 * 2.0 = 6.0)

    createVAddF32WithConst(0, 2.0f, 1);
    createVMulF32WithConst(2, 3.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // After fusion, should have only 1 instruction
    EXPECT_EQ(countInstructions(), 1);

    // The remaining instruction should be v_fma_f32 with constant 6.0
    auto& inst = *bb->begin();
    auto* fma_result = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(fma_result, nullptr);

    // Check that constants were folded: 3.0 * 2.0 = 6.0
    // srcRegs[0] should be 3.0 (mul_const)
    // srcRegs[2] should be 6.0 (mul_const * add_const)
    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[0].literalDouble, 3.0f);
    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[2].literalDouble, 6.0f);
}

TEST_F(PeepholeOptimizationPassTest, ADDMULFusion_F16_Basic) {
    // F16 variant of ADD+MUL fusion
    // v_add_f16 v0, 2.0, v1
    // v_mul_f16 v2, 3.0, v0
    // => v_fma_f16 v2, 3.0, v1, 6.0

    createVAddF16WithConst(0, 2.0f, 1);
    createVMulF16WithConst(2, 3.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma_result = static_cast<StinkyInstruction*>(&inst);

    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[0].literalDouble, 3.0f);
    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[2].literalDouble, 6.0f);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_ADDMUL_MultipleUses) {
    // ADD result has multiple uses - should NOT fuse
    // v_add_f32 v0, 2.0, v1
    // v_mul_f32 v2, 3.0, v0
    // v_add_f32 v3, v0, v4        // v0 used again!

    createVAddF32WithConst(0, 2.0f, 1);
    createVMulF32WithConst(2, 3.0f, 0);
    // Extra use of v0
    createVAddF32(3, 0, 4);

    ASSERT_EQ(countInstructions(), 3);

    runPass();

    // Should NOT fuse - still 3 instructions
    EXPECT_EQ(countInstructions(), 3);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_ADDMUL_MissingConstant_Add) {
    // ADD doesn't have constant - should NOT fuse
    // v_add_f32 v0, v1, v2        // No constant!
    // v_mul_f32 v3, 3.0, v0

    createVAddF32(0, 1, 2);  // v0 = v1 + v2 (no constant)
    createVMulF32WithConst(3, 3.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should NOT fuse
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, NoFusion_ADDMUL_MissingConstant_Mul) {
    // MUL doesn't have constant - should NOT fuse
    // v_add_f32 v0, 2.0, v1
    // v_mul_f32 v2, v3, v0        // No constant!

    createVAddF32WithConst(0, 2.0f, 1);
    createVMulF32(2, 3, 0);  // v2 = v3 * v0 (no constant)

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // Should NOT fuse
    EXPECT_EQ(countInstructions(), 2);
}

TEST_F(PeepholeOptimizationPassTest, ADDMULFusion_WithNegative) {
    // Test with negative constants
    // v_add_f32 v0, -2.0, v1
    // v_mul_f32 v2, 3.0, v0
    // => v_fma_f32 v2, 3.0, v1, -6.0
    // (3.0 * -2.0 = -6.0)

    createVAddF32WithConst(0, -2.0f, 1);
    createVMulF32WithConst(2, 3.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma_result = static_cast<StinkyInstruction*>(&inst);

    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[0].literalDouble, 3.0f);
    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[2].literalDouble, -6.0f);
}

TEST_F(PeepholeOptimizationPassTest, ADDMULFusion_BothNegative) {
    // Test with both constants negative
    // v_add_f32 v0, -2.0, v1
    // v_mul_f32 v2, -3.0, v0
    // => v_fma_f32 v2, -3.0, v1, 6.0
    // (-3.0 * -2.0 = 6.0)

    createVAddF32WithConst(0, -2.0f, 1);
    createVMulF32WithConst(2, -3.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* fma_result = static_cast<StinkyInstruction*>(&inst);

    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[0].literalDouble, -3.0f);
    EXPECT_FLOAT_EQ(fma_result->getSrcRegs()[2].literalDouble, 6.0f);
}

// ============================================================================
// Move Propagation Tests
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, MovPropagation_F32_Add_Basic) {
    // v_mov_b32 v1, v0
    // v_add_f32 v2, 2.0, v1
    // => v_add_f32 v2, 2.0, v0

    createVMovB32(1, 0);
    createVAddF32WithConst(2, 2.0f, 1);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* add = static_cast<StinkyInstruction*>(&inst);
    EXPECT_EQ(add->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_add_f32));
    EXPECT_EQ(add->getDestRegs()[0].reg.idx, 2u);
    EXPECT_EQ(add->getSrcRegs()[1].reg.idx, 0u);  // Should use v0, not v1
}

TEST_F(PeepholeOptimizationPassTest, MovPropagation_F16_Add_Basic) {
    // v_mov_b32 v1, v0
    // v_add_f16 v2, 2.0, v1
    // => v_add_f16 v2, 2.0, v0

    createVMovB32(1, 0);
    createVAddF16WithConst(2, 2.0f, 1);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* add = static_cast<StinkyInstruction*>(&inst);
    EXPECT_EQ(add->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_add_f16));
    EXPECT_EQ(add->getDestRegs()[0].reg.idx, 2u);
    EXPECT_EQ(add->getSrcRegs()[1].reg.idx, 0u);
}

TEST_F(PeepholeOptimizationPassTest, MovPropagation_F32_Mul_Basic) {
    // v_mov_b32 v1, v0
    // v_mul_f32 v2, 2.0, v1
    // => v_mul_f32 v2, 2.0, v0

    createVMovB32(1, 0);
    createVMulF32WithConst(2, 2.0f, 1);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* mul = static_cast<StinkyInstruction*>(&inst);
    EXPECT_EQ(mul->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_mul_f32));
    EXPECT_EQ(mul->getDestRegs()[0].reg.idx, 2u);
    EXPECT_EQ(mul->getSrcRegs()[1].reg.idx, 0u);
}

TEST_F(PeepholeOptimizationPassTest, MovPropagation_F16_Mul_Basic) {
    // v_mov_b32 v1, v0
    // v_mul_f16 v2, 2.0, v1
    // => v_mul_f16 v2, 2.0, v0

    createVMovB32(1, 0);
    createVMulF16WithConst(2, 2.0f, 1);

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* mul = static_cast<StinkyInstruction*>(&inst);
    EXPECT_EQ(mul->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_mul_f16));
    EXPECT_EQ(mul->getDestRegs()[0].reg.idx, 2u);
    EXPECT_EQ(mul->getSrcRegs()[1].reg.idx, 0u);
}

TEST_F(PeepholeOptimizationPassTest, NoMovPropagation_MultipleUses) {
    // v_mov_b32 v1, v0
    // v_add_f32 v2, 2.0, v1
    // v_mul_f32 v3, 3.0, v1  // v1 used twice!
    // => Should NOT propagate

    createVMovB32(1, 0);
    createVAddF32WithConst(2, 2.0f, 1);
    createVMulF32WithConst(3, 3.0f, 1);

    ASSERT_EQ(countInstructions(), 3);

    runPass();

    // Should still have 3 instructions
    EXPECT_EQ(countInstructions(), 3);
}

TEST_F(PeepholePassManagerTest, IntegrationWithPassManager) {
    // Test that the pass integrates correctly with PassManager
    createVFmaF32WithConst(0, 1, 2, 1.0f);
    createVAddF32WithConst(0, 1.0f, 0);

    ASSERT_EQ(countInstructions(), 2);

    addPass(createBuildUseDefChainPass(true));
    addPass(createPeepholeOptimizationPass());
    run(func);

    EXPECT_EQ(countInstructions(), 1);
}

// ============================================================================
// Test: HexLiteral x FloatLiteral Constant Folding
// ============================================================================

TEST_F(PeepholeOptimizationPassTest, HexLiteralTimesFloatLiteral_MulMulFusion) {
    // Test constant folding with hex literal (0x40ec7326) x float literal (2.0)
    //
    // Simulates the intrinsic flow:
    //   Intrinsic level:
    //     temp = v_mul_f32(src, 0x40ec7326)  // HexLiteral
    //     dest = v_mul_f32(temp, 2.0)        // FloatLiteral
    //
    //   After lowering to Assembly IR:
    //     temp = v_mul_f32(src, 7.3890562)   // Both become LiteralDouble
    //     dest = v_mul_f32(temp, 2.0)
    //
    //   After peephole optimization (MUL+MUL fusion):
    //     dest = v_mul_f32(src, 14.778112)   // ? Folded correctly
    //                                         // ? But hex format lost
    //
    // 0x40ec7326 = 7.3890562 (e^2 in IEEE 754)
    // 7.3890562 * 2.0 = 14.778112

    // Create the pattern using LiteralDouble (simulates post-lowering state)
    // In real flow, IntrinsicOperand::HexLiteral gets converted to StinkyRegister::LiteralDouble

    // temp = v_mul_f32(src, 7.3890562)  // Originally HexLiteral 0x40ec7326
    uint32_t hexBits = 0x40ec7326;
    float hexFloat = *reinterpret_cast<float*>(&hexBits);
    EXPECT_NEAR(hexFloat, 7.3890562f, 0.0001f) << "Verify hex literal value is e^2";

    createVMulF32WithConst(0, hexFloat, 1);  // v0 = 7.3890562 * v1

    // dest = v_mul_f32(temp, 2.0)  // FloatLiteral
    createVMulF32WithConst(2, 2.0f, 0);  // v2 = 2.0 * v0

    ASSERT_EQ(countInstructions(), 2);

    runPass();

    // After fusion: should fold constants
    EXPECT_EQ(countInstructions(), 1);

    auto& inst = *bb->begin();
    auto* mul = static_cast<StinkyInstruction*>(&inst);
    ASSERT_NE(mul, nullptr);

    EXPECT_EQ(mul->getUnifiedOpcode(), static_cast<uint16_t>(GFX::v_mul_f32));
    EXPECT_EQ(mul->getDestRegs()[0].reg.idx, 2u);  // Destination is v2
    EXPECT_EQ(mul->getSrcRegs()[1].reg.idx, 1u);   // Source is v1
    EXPECT_EQ(mul->getSrcRegs()[0].dataType, StinkyRegister::Type::LiteralDouble);

    // Verify constant folding: 7.3890562 * 2.0 = 14.778112
    float expectedResult = hexFloat * 2.0f;
    EXPECT_NEAR(expectedResult, 14.778112f, 0.0001f);
    EXPECT_NEAR(mul->getSrcRegs()[0].literalDouble, expectedResult, 0.0001f)
        << "Constant folding should compute: 7.3890562 * 2.0 = 14.778112";

    // Note: Hex format is lost - result is stored as decimal LiteralDouble
    // Original: 0x40ec7326 (hex) -> 7.3890562 (decimal) -> 14.778112 (decimal)
    // Not: 0x40ec7326 (hex) -> 0x416c73de (result in hex format)
}
