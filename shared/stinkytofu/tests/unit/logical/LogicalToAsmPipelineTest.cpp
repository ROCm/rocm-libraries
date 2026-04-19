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
