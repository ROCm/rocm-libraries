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
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/ir/logical/LogicalOpcode.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

class IROpcodeTest : public ::testing::Test {};

// Test: IR instructions have correct opcodes
TEST_F(IROpcodeTest, InstructionOpcodes) {
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    // Create some IR instructions and check their opcodes
    auto add = makeLogicalInstructionShared(VAddF32(v0, v1, v2));
    EXPECT_EQ(add->getOpcode(), logical::VAddF32);

    auto mul = makeLogicalInstructionShared(VMulF32(v0, v1, v2));
    EXPECT_EQ(mul->getOpcode(), logical::VMulF32);

    auto fma = makeLogicalInstructionShared(VFmaF32(v0, v1, v2, v0));
    EXPECT_EQ(fma->getOpcode(), logical::VFmaF32);

    auto max = makeLogicalInstructionShared(VMaxF32(v0, v1, v2));
    EXPECT_EQ(max->getOpcode(), logical::VMaxF32);

    auto mov = makeLogicalInstructionShared(VMovB32(v0, v1));
    EXPECT_EQ(mov->getOpcode(), logical::VMovB32);
}

// Test: Opcode to name mapping
TEST_F(IROpcodeTest, OpcodeToName) {
    EXPECT_STREQ(logical::getOpcodeName(logical::UNKNOWN), "UNKNOWN");
    EXPECT_STREQ(logical::getOpcodeName(logical::VAddF32), "VAddF32");
    EXPECT_STREQ(logical::getOpcodeName(logical::VMulF32), "VMulF32");
    EXPECT_STREQ(logical::getOpcodeName(logical::VFmaF32), "VFmaF32");
    EXPECT_STREQ(logical::getOpcodeName(logical::VMaxF32), "VMaxF32");
    EXPECT_STREQ(logical::getOpcodeName(logical::VMovB32), "VMovB32");
}

// Test: Opcode to mnemonic mapping
TEST_F(IROpcodeTest, OpcodeToMnemonic) {
    EXPECT_STREQ(logical::getOpcodeMnemonic(logical::UNKNOWN), "unknown");
    EXPECT_STREQ(logical::getOpcodeMnemonic(logical::VAddF32), "v_add_f32");
    EXPECT_STREQ(logical::getOpcodeMnemonic(logical::VMulF32), "v_mul_f32");
    EXPECT_STREQ(logical::getOpcodeMnemonic(logical::VFmaF32), "v_fma_f32");
    EXPECT_STREQ(logical::getOpcodeMnemonic(logical::VMaxF32), "v_max_f32");
    EXPECT_STREQ(logical::getOpcodeMnemonic(logical::VMovB32), "v_mov_b32");
}

// Test: F16 variants
TEST_F(IROpcodeTest, F16Variants) {
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    auto add = makeLogicalInstructionShared(VAddF16(v0, v1, v2));
    EXPECT_EQ(add->getOpcode(), logical::VAddF16);
    EXPECT_STREQ(logical::getOpcodeName(logical::VAddF16), "VAddF16");
    EXPECT_STREQ(logical::getOpcodeMnemonic(logical::VAddF16), "v_add_f16");

    auto mul = makeLogicalInstructionShared(VMulF16(v0, v1, v2));
    EXPECT_EQ(mul->getOpcode(), logical::VMulF16);
    EXPECT_STREQ(logical::getOpcodeName(logical::VMulF16), "VMulF16");
    EXPECT_STREQ(logical::getOpcodeMnemonic(logical::VMulF16), "v_mul_f16");
}

// Test: getLogicalName still works (backward compatibility)
TEST_F(IROpcodeTest, LogicalNameBackwardCompatibility) {
    StinkyRegister v0 = vgpr(0);
    StinkyRegister v1 = vgpr(1);
    StinkyRegister v2 = vgpr(2);

    auto add = makeLogicalInstructionShared(VAddF32(v0, v1, v2));
    EXPECT_STREQ(add->getLogicalName(), "VAddF32");

    // Logical name should match the opcode name
    EXPECT_STREQ(add->getLogicalName(), logical::getOpcodeName(add->getOpcode()));
}

// Test: String to opcode parsing (CamelCase)
TEST_F(IROpcodeTest, ParseOpcodeCamelCase) {
    EXPECT_EQ(logical::parseOpcode("VAddF32"), logical::VAddF32);
    EXPECT_EQ(logical::parseOpcode("VMulF32"), logical::VMulF32);
    EXPECT_EQ(logical::parseOpcode("VMaxF32"), logical::VMaxF32);
    EXPECT_EQ(logical::parseOpcode("VMinF32"), logical::VMinF32);
    EXPECT_EQ(logical::parseOpcode("VCmpGTF32"), logical::VCmpGTF32);
    EXPECT_EQ(logical::parseOpcode("VCndMaskB32"), logical::VCndMaskB32);
    EXPECT_EQ(logical::parseOpcode("VMovB32"), logical::VMovB32);
}

// Test: String to opcode parsing (snake_case - using actual generated mnemonics)
TEST_F(IROpcodeTest, ParseOpcodeSnakeCase) {
    EXPECT_EQ(logical::parseOpcode("v_add_f32"), logical::VAddF32);
    EXPECT_EQ(logical::parseOpcode("v_mul_f32"), logical::VMulF32);
    EXPECT_EQ(logical::parseOpcode("v_max_f32"), logical::VMaxF32);
    EXPECT_EQ(logical::parseOpcode("v_min_f32"), logical::VMinF32);
    // Note: TableGen generates mnemonics with underscore between each capital letter
    EXPECT_EQ(logical::parseOpcode("v_cmp_g_t_f32"), logical::VCmpGTF32);
    EXPECT_EQ(logical::parseOpcode("v_cnd_mask_b32"), logical::VCndMaskB32);
    EXPECT_EQ(logical::parseOpcode("v_mov_b32"), logical::VMovB32);
}

// Test: Parse unknown/invalid opcodes
TEST_F(IROpcodeTest, ParseOpcodeInvalid) {
    EXPECT_EQ(logical::parseOpcode("invalid_instruction"), logical::UNKNOWN);
    EXPECT_EQ(logical::parseOpcode(""), logical::UNKNOWN);
    EXPECT_EQ(logical::parseOpcode(nullptr), logical::UNKNOWN);
    EXPECT_EQ(logical::parseOpcode("v_nonexistent_f32"), logical::UNKNOWN);
}

// Test: Round-trip conversion (opcode -> string -> opcode)
TEST_F(IROpcodeTest, OpcodeRoundTrip) {
    // Test with CamelCase
    EXPECT_EQ(logical::parseOpcode(logical::getOpcodeName(logical::VAddF32)), logical::VAddF32);
    EXPECT_EQ(logical::parseOpcode(logical::getOpcodeName(logical::VMaxF32)), logical::VMaxF32);

    // Test with snake_case
    EXPECT_EQ(logical::parseOpcode(logical::getOpcodeMnemonic(logical::VAddF32)), logical::VAddF32);
    EXPECT_EQ(logical::parseOpcode(logical::getOpcodeMnemonic(logical::VMaxF32)), logical::VMaxF32);
}
