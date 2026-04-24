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

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/ir/logical/LogicalToFunctionConverter.hpp"
#include "stinkytofu/transforms/logical/CompositeInstructionLoweringPass.hpp"
#include "stinkytofu/transforms/logical/ToStinkyAsmPass.hpp"

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
class LogicalModuleOwnershipTest : public ::testing::Test {
   protected:
    StinkyRegister v0, v1, v2, v3;

    void SetUp() override {
        v0 = vgpr(0);
        v1 = vgpr(1);
        v2 = vgpr(2);
        v3 = vgpr(3);
    }
};

TEST_F(LogicalModuleOwnershipTest, SharedPtrToRawPointerConversion) {
    // Step 1: Create PyLogicalModule with shared_ptr (simulating Python usage)
    auto module = std::make_shared<PyLogicalModule>("test_ownership");

    module->add(
        makeLogicalInstructionShared(VAddF32(v0, v1, v2, std::nullopt, std::nullopt, "add")));
    module->add(
        makeLogicalInstructionShared(VMulF32(v0, v0, v3, std::nullopt, std::nullopt, "mul")));

    EXPECT_EQ(module->size(), 2);

    // Step 2: Convert to PyLogicalFunction (external Function*; ~PyLogicalFunction detaches
    // ownedExternally)
    Function func("kernel");
    PyLogicalFunction pyFunc(&func);
    LogicalToFunctionConverter converter(GfxArchID::Gfx1250);
    converter.convertWithAutoBlocks(module.get(), pyFunc);

    // Verify instructions were added to Function.
    size_t instCount = 0;
    for (BasicBlock& bb : func) {
        instCount += bb.size();
    }
    EXPECT_EQ(instCount, 2) << "Function should have 2 instructions";

    // When pyFunc is destroyed it detaches ownedExternally IRs so the list does not delete them.
    // Note: Do not run lowering passes here; see LogicalModuleSurvivesAfterConversion for full
    // pipeline.
}

TEST_F(LogicalModuleOwnershipTest, LogicalModuleSurvivesAfterConversion) {
    // Create PyLogicalModule (Python-owned instructions)
    auto module = std::make_shared<PyLogicalModule>("test_survival");
    module->add(makeLogicalInstructionShared(VAddF32(v0, v1, v2)));
    module->add(makeLogicalInstructionShared(VAddF32(v1, v2, v3)));

    EXPECT_EQ(module->size(), 2);

    {
        Function func("kernel");
        PyLogicalFunction pyFunc(&func);
        PassManager pm;
        LogicalToFunctionConverter converter(GfxArchID::Gfx1250);
        converter.convert(module.get(), pyFunc);

        // Verify conversion
        size_t count = 0;
        for (BasicBlock& bb : func) {
            count += bb.size();
        }
        EXPECT_EQ(count, 2);

        // ~PyLogicalFunction detaches ownedExternally IRs. Caller owns func; shared_ptrs keep
        // instructions alive.
    }

    // PyLogicalModule should still be valid
    EXPECT_EQ(module->size(), 2);
    EXPECT_NE(module->getInstructions()[0], nullptr);
    EXPECT_NE(module->getInstructions()[1], nullptr);
}

TEST_F(LogicalModuleOwnershipTest, NoDoubleFreeWithCompositeInstruction) {
    // Test with composite instruction that gets expanded
    auto module = std::make_shared<PyLogicalModule>("test_composite");

    StinkyRegister dst = vgpr(0, 2);
    StinkyRegister src0 = vgpr(2, 2);
    StinkyRegister src1 = vgpr(4, 2);

    // Add a composite instruction (Python-owned)
    module->add(makeLogicalInstructionShared(VAddPKF32(dst, src0, src1)));
    module->add(makeLogicalInstructionShared(VMulF32(v0, v1, v2)));

    EXPECT_EQ(module->size(), 2);

    Function func("kernel");
    PyLogicalFunction pyFunc(&func);
    PassManager pm;
    LogicalToFunctionConverter converter(GfxArchID::Gfx1250);
    converter.convertWithAutoBlocks(module.get(), pyFunc);

    GemmTileConfig config;
    config.arch = {12, 5, 0};
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

    // ~PyLogicalFunction detaches ownedExternally IRs; no manual detach needed.
    // If we get here without crashing, ownership is handled correctly!
    EXPECT_TRUE(true) << "No double-free occurred";
}
