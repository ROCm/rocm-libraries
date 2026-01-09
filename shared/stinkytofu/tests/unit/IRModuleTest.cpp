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

#include "ir/IRModule.hpp"
#include "TestHelpers.hpp"
#include "ir/StinkyInstructions.hpp"
#include "ir/passes/CompositeInstructionLoweringPass.hpp"
#include "ir/passes/PassManager.hpp"
#include "ir/passes/ToStinkyAsmPass.hpp"
#include "isa/gfx/GfxIsa.hpp"
#include <gtest/gtest.h>

using namespace stinkytofu;
using namespace stinkytofu::test;

/**
 * @brief Test that IRModule is architecture-independent
 *
 * The IRModule should be created without specifying an architecture.
 * Architecture is only needed when lowering to assembly.
 */
TEST(IRModuleTest, ArchitectureIndependent)
{
    // Create an architecture-independent IR module
    auto module = std::make_shared<IRModule>("test_kernel");

    ASSERT_NE(module, nullptr);
    EXPECT_EQ(module->getName(), "test_kernel");
    EXPECT_EQ(module->size(), 0);
}

/**
 * @brief Test adding instructions to IRModule
 */
TEST(IRModuleTest, AddInstructions)
{
    auto module = std::make_shared<IRModule>("test_kernel");

    // Create some IR instructions manually
    StinkyRegister dst  = vgpr(0);
    StinkyRegister src0 = vgpr(1);
    StinkyRegister src1 = vgpr(2);

    auto inst1 = std::make_shared<VAddU32>(dst, src0, src1);
    auto inst2 = std::make_shared<VMulF32>(dst, src0, src1);

    module->add(inst1);
    module->add(inst2);

    EXPECT_EQ(module->size(), 2);
    EXPECT_EQ(module->getInstructions().size(), 2);
}

/**
 * @brief Test lowering to different architectures
 *
 * The same IRModule can be lowered to different target architectures.
 */
TEST(IRModuleTest, LowerToDifferentArchitectures)
{
    // Create architecture-independent IR module
    auto module = std::make_shared<IRModule>("test_kernel");

    StinkyRegister dst  = vgpr(0);
    StinkyRegister src0 = vgpr(1);
    StinkyRegister src1 = vgpr(2);

    module->add(std::make_shared<VAddU32>(dst, src0, src1));

    // Lower to gfx942 using IRInstPassManager with factory functions
    IRInstPassManager pm942(GfxArchID::Gfx942);
    pm942.addPass(createCompositeInstructionLoweringPass());
    pm942.addPass(createToStinkyAsmPass());
    auto asmList942 = pm942.run(module.get());

    ASSERT_NE(asmList942, nullptr);
    EXPECT_GE(asmList942->size(), 1);

    // Lower the same IR to gfx1250 using IRInstPassManager with factory functions
    IRInstPassManager pm1250(GfxArchID::Gfx1250);
    pm1250.addPass(createCompositeInstructionLoweringPass());
    pm1250.addPass(createToStinkyAsmPass());
    auto asmList1250 = pm1250.run(module.get());

    ASSERT_NE(asmList1250, nullptr);
    EXPECT_GE(asmList1250->size(), 1);
}

/**
 * @brief Test composite instruction lowering
 *
 * Composite instructions like VAddPKF32 may expand differently
 * depending on the target architecture.
 */
TEST(IRModuleTest, CompositeInstructionLowering)
{
    // Create IR module with composite instruction
    auto module = std::make_shared<IRModule>("test_composite");

    StinkyRegister dst  = vgpr(0, 2); // 2 registers for packed
    StinkyRegister src0 = vgpr(2, 2);
    StinkyRegister src1 = vgpr(4, 2);

    module->add(std::make_shared<VAddPKF32>(dst, src0, src1));

    // Lower to gfx942 using IRInstPassManager with factory functions (should use v_pk_add_f32)
    IRInstPassManager pm(GfxArchID::Gfx942);
    pm.addPass(createCompositeInstructionLoweringPass());
    pm.addPass(createToStinkyAsmPass());
    auto asmList942 = pm.run(module.get());
    ASSERT_NE(asmList942, nullptr);
    EXPECT_GE(asmList942->size(), 1);

    // The CompositeInstructionLoweringPass handles expansion transparently
    // based on architecture capabilities. On gfx942, VAddPKF32 should lower
    // to v_pk_add_f32 if supported, or expand to multiple instructions otherwise.
}
