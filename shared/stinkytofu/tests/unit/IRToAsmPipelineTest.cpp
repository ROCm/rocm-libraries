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

#include "TestHelpers.hpp"
#include "ir/IRModule.hpp"
#include "ir/StinkyInstructions.hpp"
#include "ir/asm/StinkyAsmEmitter.hpp"
#include "ir/passes/PassManager.hpp"
#include "isa/ArchHelper.hpp"
#include "isa/gfx/GfxIsa.hpp"
#include "stinkypasses.hpp"
#include "stinkytofu.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

using namespace stinkytofu;

/**
 * Test the complete pipeline from high-level IR to assembly string:
 * 1. Create high-level IR instructions (IRInstruction with shared_ptr)
 * 2. Lower to assembly IR using IRInstPassManager
 * 3. Run assembly optimization passes using PassManager
 * 4. Emit assembly string
 */
class IRToAsmPipelineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Helper registers
        using namespace stinkytofu::test;
        v0 = vgpr(0);
        v1 = vgpr(1);
        v2 = vgpr(2);
        v3 = vgpr(3);
    }

    StinkyRegister v0, v1, v2, v3;
};

TEST_F(IRToAsmPipelineTest, SimpleVectorALU)
{
    // Step 1: Create high-level IR module with shared_ptr
    auto module = std::make_unique<IRModule>("test_vadd");

    module->add(
        std::make_shared<VAddF32>(v0, v1, v2, std::nullopt, std::nullopt, "add two floats"));
    module->add(
        std::make_shared<VMulF32>(v0, v0, v3, std::nullopt, std::nullopt, "multiply result"));

    EXPECT_EQ(module->size(), 2);

    // Step 2: Lower high-level IR to assembly IR using IRInstPassManager
    IRInstPassManager irPassManager(GfxArchID::Gfx942);
    irPassManager.addPass(createCompositeInstructionLoweringPass());
    irPassManager.addPass(createToStinkyAsmPass());

    auto asmIRList = irPassManager.run(module.get());
    ASSERT_NE(asmIRList, nullptr);
    EXPECT_EQ(asmIRList->size(), 2) << "Should have 2 assembly instructions";

    // TODO: Add assembly emission and optimization passes testing
    // This requires integrating with BasicBlock, PassManager, and StinkyAsmEmitter
}
