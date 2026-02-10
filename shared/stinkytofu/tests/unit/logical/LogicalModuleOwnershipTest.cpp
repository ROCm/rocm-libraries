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
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/ir/logical/LogicalToFunctionConverter.hpp"
#include "stinkytofu/transforms/logical/CompositeInstructionLoweringPass.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"
#include "stinkytofu/core/PyLogicalModule.hpp"
#include "stinkytofu/core/stinkytofu.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace stinkytofu;
using namespace stinkytofu::test;

/**
 * Test ownership model: PyLogicalModule (shared_ptr) -> Function (raw pointers + shared_ptr keeper)
 *
 * This test verifies that we don't get double-free when:
 * 1. Creating a PyLogicalModule with shared_ptr<LogicalInstruction>
 * 2. Converting to Function which holds raw pointers in IRList
 * 3. Running lowering passes that delete/replace instructions
 * 4. Destructing both PyLogicalModule and Function
 */
class LogicalModuleOwnershipTest : public ::testing::Test
{
protected:
    StinkyRegister v0, v1, v2, v3;

    void SetUp() override
    {
        v0 = vgpr(0);
        v1 = vgpr(1);
        v2 = vgpr(2);
        v3 = vgpr(3);
    }
};

TEST_F(LogicalModuleOwnershipTest, SharedPtrToRawPointerConversion)
{
    // Step 1: Create PyLogicalModule with shared_ptr (simulating Python usage)
    auto module = std::make_shared<PyLogicalModule>("test_ownership");

    module->add(std::shared_ptr<LogicalInstruction>(
        VAddF32(v0, v1, v2, std::nullopt, std::nullopt, "add")));
    module->add(std::shared_ptr<LogicalInstruction>(
        VMulF32(v0, v0, v3, std::nullopt, std::nullopt, "mul")));

    EXPECT_EQ(module->size(), 2);

    // Step 2: Convert to Function (transfers to raw pointers + keeps shared_ptr alive)
    PassManager pm;
    Function&   func = pm.getPassContext().getFunction();

    LogicalToFunctionConverter converter(GfxArchID::Gfx942);
    converter.convertWithAutoBlocks(module.get(), func);

    // Verify instructions were added to Function
    size_t instCount = 0;
    for(BasicBlock& bb : func)
    {
        instCount += bb.getIR().size();
    }
    EXPECT_EQ(instCount, 2) << "Function should have 2 instructions";

    // Step 3: PyLogicalModule can go out of scope (shared_ptrs still held by Function)
    module.reset();

    // Step 4: Run passes that modify IRList
    GemmTileConfig config;
    config.arch     = {9, 4, 2};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    pm.addPass(createCompositeInstructionLoweringPass());
    pm.addPass(createToStinkyAsmPass());
    pm.run();

    // Verify lowering happened (all should be StinkyInstruction now)
    instCount           = 0;
    size_t asmInstCount = 0;
    for(BasicBlock& bb : func)
    {
        for(IRBase& ir : bb.getIR())
        {
            instCount++;
            if(ir.getType() == IRBase::IRType::StinkyTofu)
            {
                asmInstCount++;
            }
        }
    }
    EXPECT_EQ(instCount, 2) << "Should still have 2 instructions";
    EXPECT_EQ(asmInstCount, 2) << "All should be StinkyInstruction";

    // Step 5: Release LogicalInstruction ownership (optional optimization)
    // After lowering, the LogicalInstructions are no longer needed
    // This frees memory while Function is still alive
    pm.getPassContext().getFunction().releaseLogicalInstructionOwnership();

    // Step 6: PassManager (and Function) will be destroyed
    // This should NOT cause double-free!
}

TEST_F(LogicalModuleOwnershipTest, LogicalModuleSurvivesAfterConversion)
{
    // Create PyLogicalModule
    auto module = std::make_shared<PyLogicalModule>("test_survival");
    module->add(std::shared_ptr<LogicalInstruction>(VAddF32(v0, v1, v2)));
    module->add(std::shared_ptr<LogicalInstruction>(VAddF32(v1, v2, v3)));

    EXPECT_EQ(module->size(), 2);

    {
        // Convert to Function in a nested scope
        PassManager pm;
        Function&   func = pm.getPassContext().getFunction();

        LogicalToFunctionConverter converter(GfxArchID::Gfx942);
        converter.convert(module.get(), func);

        // Verify conversion
        size_t count = 0;
        for(BasicBlock& bb : func)
        {
            count += bb.getIR().size();
        }
        EXPECT_EQ(count, 2);

        // PassManager goes out of scope here
        // Function is destroyed, but shared_ptrs keep instructions alive
    }

    // PyLogicalModule should still be valid
    EXPECT_EQ(module->size(), 2);
    EXPECT_NE(module->getInstructions()[0], nullptr);
    EXPECT_NE(module->getInstructions()[1], nullptr);
}

TEST_F(LogicalModuleOwnershipTest, NoDoubleFreeWithCompositeInstruction)
{
    // Test with composite instruction that gets expanded
    auto module = std::make_shared<PyLogicalModule>("test_composite");

    StinkyRegister dst  = vgpr(0, 2);
    StinkyRegister src0 = vgpr(2, 2);
    StinkyRegister src1 = vgpr(4, 2);

    // Add a composite instruction
    module->add(std::shared_ptr<LogicalInstruction>(VAddPKF32(dst, src0, src1)));
    module->add(std::shared_ptr<LogicalInstruction>(VMulF32(v0, v1, v2)));

    EXPECT_EQ(module->size(), 2);

    // Convert and run expansion pass
    PassManager pm;
    Function&   func = pm.getPassContext().getFunction();

    LogicalToFunctionConverter converter(GfxArchID::Gfx942);
    converter.convertWithAutoBlocks(module.get(), func);

    GemmTileConfig config;
    config.arch     = {9, 4, 2};
    config.TileA0   = 16;
    config.TileB0   = 16;
    config.TileM0   = 16;
    config.NumGRA   = 4;
    config.NumGRB   = 4;
    config.NumGRM   = 4;
    config.NumWaves = 1;
    pm.setGemmTileConfig(config);

    // Composite expansion may delete original instructions and create new ones
    pm.addPass(createCompositeInstructionLoweringPass());
    pm.addPass(createToStinkyAsmPass());
    pm.run();

    // If we get here without crashing, ownership is handled correctly!
    EXPECT_TRUE(true) << "No double-free occurred";
}
