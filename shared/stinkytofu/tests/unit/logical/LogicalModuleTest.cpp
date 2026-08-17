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

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

/**
 * @brief Test that PyLogicalModule is architecture-independent
 *
 * The PyLogicalModule should be created without specifying an architecture.
 * Architecture is only needed when lowering to assembly.
 */
TEST(IRModuleTest, ArchitectureIndependent) {
    // Create an architecture-independent IR module
    auto module = std::make_shared<PyLogicalModule>("test_kernel");

    ASSERT_NE(module, nullptr);
    EXPECT_EQ(module->getName(), "test_kernel");
    EXPECT_EQ(module->size(), 0);
}

/**
 * @brief Test adding instructions to PyLogicalModule
 */
TEST(IRModuleTest, AddInstructions) {
    auto module = std::make_shared<PyLogicalModule>("test_kernel");

    // Create some IR instructions manually
    StinkyRegister dst = vgpr(0);
    StinkyRegister src0 = vgpr(1);
    StinkyRegister src1 = vgpr(2);

    auto inst1 = makeLogicalInstructionShared(VAddU32(dst, src0, src1));
    auto inst2 = makeLogicalInstructionShared(VMulF32(dst, src0, src1));

    module->add(inst1);
    module->add(inst2);

    EXPECT_EQ(module->size(), 2);
    EXPECT_EQ(module->getInstructions().size(), 2);
}
