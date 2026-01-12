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

#include "ir/IROpcode.hpp"
#include "TestHelpers.hpp"
#include "ir/StinkyInstructions.hpp"
#include <gtest/gtest.h>

using namespace stinkytofu;
using namespace stinkytofu::test;

class IROpcodeTest : public ::testing::Test
{
};

// Test: IR instructions have correct opcodes
TEST_F(IROpcodeTest, InstructionOpcodes)
{
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    // Create some IR instructions and check their opcodes
    auto add = std::make_shared<VAddF32>(v0, v1, v2);
    EXPECT_EQ(add->getOpcode(), HLIR::VAddF32);

    auto mul = std::make_shared<VMulF32>(v0, v1, v2);
    EXPECT_EQ(mul->getOpcode(), HLIR::VMulF32);

    auto fma = std::make_shared<VFmaF32>(v0, v1, v2, v0);
    EXPECT_EQ(fma->getOpcode(), HLIR::VFmaF32);

    auto max = std::make_shared<VMaxF32>(v0, v1, v2);
    EXPECT_EQ(max->getOpcode(), HLIR::VMaxF32);

    auto mov = std::make_shared<VMovB32>(v0, v1);
    EXPECT_EQ(mov->getOpcode(), HLIR::VMovB32);
}

// Test: Opcode to name mapping
TEST_F(IROpcodeTest, OpcodeToName)
{
    EXPECT_STREQ(HLIR::getOpcodeName(HLIR::UNKNOWN), "UNKNOWN");
    EXPECT_STREQ(HLIR::getOpcodeName(HLIR::VAddF32), "VAddF32");
    EXPECT_STREQ(HLIR::getOpcodeName(HLIR::VMulF32), "VMulF32");
    EXPECT_STREQ(HLIR::getOpcodeName(HLIR::VFmaF32), "VFmaF32");
    EXPECT_STREQ(HLIR::getOpcodeName(HLIR::VMaxF32), "VMaxF32");
    EXPECT_STREQ(HLIR::getOpcodeName(HLIR::VMovB32), "VMovB32");
}

// Test: Opcode to mnemonic mapping
TEST_F(IROpcodeTest, OpcodeToMnemonic)
{
    EXPECT_STREQ(HLIR::getOpcodeMnemonic(HLIR::UNKNOWN), "unknown");
    EXPECT_STREQ(HLIR::getOpcodeMnemonic(HLIR::VAddF32), "v_add_f32");
    EXPECT_STREQ(HLIR::getOpcodeMnemonic(HLIR::VMulF32), "v_mul_f32");
    EXPECT_STREQ(HLIR::getOpcodeMnemonic(HLIR::VFmaF32), "v_fma_f32");
    EXPECT_STREQ(HLIR::getOpcodeMnemonic(HLIR::VMaxF32), "v_max_f32");
    EXPECT_STREQ(HLIR::getOpcodeMnemonic(HLIR::VMovB32), "v_mov_b32");
}

// Test: F16 variants
TEST_F(IROpcodeTest, F16Variants)
{
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    auto add = std::make_shared<VAddF16>(v0, v1, v2);
    EXPECT_EQ(add->getOpcode(), HLIR::VAddF16);
    EXPECT_STREQ(HLIR::getOpcodeName(HLIR::VAddF16), "VAddF16");
    EXPECT_STREQ(HLIR::getOpcodeMnemonic(HLIR::VAddF16), "v_add_f16");

    auto mul = std::make_shared<VMulF16>(v0, v1, v2);
    EXPECT_EQ(mul->getOpcode(), HLIR::VMulF16);
    EXPECT_STREQ(HLIR::getOpcodeName(HLIR::VMulF16), "VMulF16");
    EXPECT_STREQ(HLIR::getOpcodeMnemonic(HLIR::VMulF16), "v_mul_f16");
}

// Test: getLogicalName still works (backward compatibility)
TEST_F(IROpcodeTest, LogicalNameBackwardCompatibility)
{
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    auto add = std::make_shared<VAddF32>(v0, v1, v2);
    EXPECT_STREQ(add->getLogicalName(), "VAddF32");

    // Logical name should match the opcode name
    EXPECT_STREQ(add->getLogicalName(), HLIR::getOpcodeName(add->getOpcode()));
}
