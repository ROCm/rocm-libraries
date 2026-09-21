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

// Unit tests for virtual register support in StinkyRegister
// Tests template-based code generation with register offset remapping

#include <gtest/gtest.h>

#include <vector>

#include "stinkytofu/ir/asm/StinkyRegister.hpp"

using namespace stinkytofu;

// ============================================================================
// StinkyRegister Virtual Register Tests
// ============================================================================

class VirtualisterTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Nothing to set up for now
    }
};

TEST_F(VirtualisterTest, VirtualVGPRCreation) {
    // Test creating virtual VGPR
    auto vTemp0 = StinkyRegister::Virtual(0);

    EXPECT_TRUE(vTemp0.isVirtualReg());
    EXPECT_TRUE(vTemp0.isRegister());
    EXPECT_EQ(vTemp0.dataType, StinkyRegister::Type::Register);
    EXPECT_EQ(vTemp0.reg.type, RegType::V);
    EXPECT_EQ(vTemp0.reg.idx, 0u | StinkyRegister::kVirtualBit);
    EXPECT_EQ(vTemp0.reg.num, 1);
}

TEST_F(VirtualisterTest, VirtualVGPRMultipleRegisters) {
    // Test creating virtual VGPR with multiple consecutive registers
    auto vTemp = StinkyRegister::Virtual(5, 4);

    EXPECT_TRUE(vTemp.isVirtualReg());
    EXPECT_EQ(vTemp.reg.type, RegType::V);
    EXPECT_EQ(vTemp.reg.idx, 5u | StinkyRegister::kVirtualBit);
    EXPECT_EQ(vTemp.reg.num, 4);
}

TEST_F(VirtualisterTest, VirtualSGPRCreation) {
    // Test creating virtual SGPR
    auto sTemp = StinkyRegister::VirtualSGPR(2);

    EXPECT_TRUE(sTemp.isVirtualReg());
    EXPECT_TRUE(sTemp.isRegister());
    EXPECT_EQ(sTemp.dataType, StinkyRegister::Type::Register);
    EXPECT_EQ(sTemp.reg.type, RegType::S);
    EXPECT_EQ(sTemp.reg.idx, 2u | StinkyRegister::kVirtualBit);
    EXPECT_EQ(sTemp.reg.num, 1);
}

TEST_F(VirtualisterTest, PhysicalRegisterNotVirtual) {
    // Test that normal registers are not virtual
    StinkyRegister physReg(RegType::V, 10, 1);

    EXPECT_FALSE(physReg.isVirtualReg());
    EXPECT_TRUE(physReg.isRegister());
    EXPECT_EQ(physReg.reg.type, RegType::V);
    EXPECT_EQ(physReg.reg.idx, 10);
}

TEST_F(VirtualisterTest, LiteralNotVirtual) {
    // Test that literals are not virtual
    StinkyRegister lit(42);

    EXPECT_FALSE(lit.isVirtualReg());
    EXPECT_FALSE(lit.isRegister());
    EXPECT_EQ(lit.dataType, StinkyRegister::Type::LiteralInt);
}

TEST_F(VirtualisterTest, WithOffsetVGPR) {
    // Test applying offset to virtual VGPR
    auto vTemp0 = StinkyRegister::Virtual(0);
    auto v10 = vTemp0.resolveVirtualToPhysical(10);

    EXPECT_FALSE(v10.isVirtualReg());  // No longer virtual after remapping
    EXPECT_TRUE(v10.isRegister());
    EXPECT_EQ(v10.reg.type, RegType::V);
    EXPECT_EQ(v10.reg.idx, 10);
    EXPECT_EQ(v10.reg.num, 1);
}

TEST_F(VirtualisterTest, WithOffsetMultipleRegisters) {
    // Test offset with multiple consecutive registers
    auto vTemp = StinkyRegister::Virtual(2, 3);
    auto v20 = vTemp.resolveVirtualToPhysical(18);  // v[2:4] -> v[20:22]

    EXPECT_FALSE(v20.isVirtualReg());
    EXPECT_EQ(v20.reg.idx, 20);
    EXPECT_EQ(v20.reg.num, 3);
}

TEST_F(VirtualisterTest, WithOffsetSGPR) {
    // Test applying offset to virtual SGPR
    auto sTemp = StinkyRegister::VirtualSGPR(1);
    auto s5 = sTemp.resolveVirtualToPhysical(4);

    EXPECT_FALSE(s5.isVirtualReg());
    EXPECT_EQ(s5.reg.type, RegType::S);
    EXPECT_EQ(s5.reg.idx, 5);
}

TEST_F(VirtualisterTest, WithOffsetOnPhysicalRegisterNoEffect) {
    // Test that offset has no effect on physical registers
    StinkyRegister physReg(RegType::V, 10, 1);
    auto result = physReg.resolveVirtualToPhysical(100);

    EXPECT_EQ(result.reg.idx, 10);  // Should remain unchanged
    EXPECT_FALSE(result.isVirtualReg());
}

TEST_F(VirtualisterTest, WithOffsetOnLiteralNoEffect) {
    // Test that offset has no effect on literals
    StinkyRegister lit(42);
    auto result = lit.resolveVirtualToPhysical(100);

    EXPECT_EQ(result.dataType, StinkyRegister::Type::LiteralInt);
    EXPECT_EQ(result.getLiteralInt(), 42);
    EXPECT_FALSE(result.isVirtualReg());
}

TEST_F(VirtualisterTest, MultipleInstantiationsFromSameTemplate) {
    // Test that same virtual register can be remapped to different physical registers
    auto vTemp = StinkyRegister::Virtual(0);

    auto v10 = vTemp.resolveVirtualToPhysical(10);
    auto v20 = vTemp.resolveVirtualToPhysical(20);
    auto v30 = vTemp.resolveVirtualToPhysical(30);

    EXPECT_EQ(v10.reg.idx, 10);
    EXPECT_EQ(v20.reg.idx, 20);
    EXPECT_EQ(v30.reg.idx, 30);

    // Original should still be virtual
    EXPECT_TRUE(vTemp.isVirtualReg());
    EXPECT_FALSE(v10.isVirtualReg());
    EXPECT_FALSE(v20.isVirtualReg());
    EXPECT_FALSE(v30.isVirtualReg());
}

// ============================================================================
// StinkyInstruction remapRegisters Tests
// ============================================================================

TEST_F(VirtualisterTest, InstructionRemapDestinationVGPR) {
    // Test the concept: Create virtual registers that would be in an instruction
    auto vDst = StinkyRegister::Virtual(0);
    auto vSrc = StinkyRegister::Virtual(1);

    // Verify initial state
    EXPECT_TRUE(vDst.isVirtualReg());
    EXPECT_TRUE(vSrc.isVirtualReg());
    EXPECT_EQ(vDst.reg.idx, 0u | StinkyRegister::kVirtualBit);
    EXPECT_EQ(vSrc.reg.idx, 1u | StinkyRegister::kVirtualBit);

    // Simulate remapping
    auto mappedDst = vDst.resolveVirtualToPhysical(10);
    auto mappedSrc = vSrc.resolveVirtualToPhysical(10);

    EXPECT_FALSE(mappedDst.isVirtualReg());
    EXPECT_FALSE(mappedSrc.isVirtualReg());
    EXPECT_EQ(mappedDst.reg.idx, 10);
    EXPECT_EQ(mappedSrc.reg.idx, 11);
}

TEST_F(VirtualisterTest, InstructionRemapMixedVirtualAndPhysical) {
    // Test remapping when instruction has both virtual and physical registers
    auto vTemp = StinkyRegister::Virtual(0);
    auto vPhys = StinkyRegister(RegType::V, 100, 1);
    auto lit = StinkyRegister(42);

    // Remap all with offset 20
    auto mappedTemp = vTemp.resolveVirtualToPhysical(20);
    auto mappedPhys = vPhys.resolveVirtualToPhysical(20);  // Should not change
    auto mappedLit = lit.resolveVirtualToPhysical(20);     // Should not change

    // Virtual should be remapped
    EXPECT_EQ(mappedTemp.reg.idx, 20);
    EXPECT_FALSE(mappedTemp.isVirtualReg());

    // Physical should remain unchanged
    EXPECT_EQ(mappedPhys.reg.idx, 100);
    EXPECT_FALSE(mappedPhys.isVirtualReg());

    // Literal should remain unchanged
    EXPECT_EQ(mappedLit.dataType, StinkyRegister::Type::LiteralInt);
    EXPECT_EQ(mappedLit.getLiteralInt(), 42);
}

TEST_F(VirtualisterTest, TemplateInstantiationScenario) {
    // Simulate a real-world scenario: activation function template
    // Template: v_add_u32 v0, v1, v2 (virtual registers)

    std::vector<StinkyRegister> templateDest = {StinkyRegister::Virtual(0)};
    std::vector<StinkyRegister> templateSrc = {StinkyRegister::Virtual(1),
                                               StinkyRegister::Virtual(2)};

    // First instantiation: map to v[10:12]
    std::vector<StinkyRegister> inst1Dest, inst1Src;
    for (auto& reg : templateDest) inst1Dest.push_back(reg.resolveVirtualToPhysical(10));
    for (auto& reg : templateSrc) inst1Src.push_back(reg.resolveVirtualToPhysical(10));

    EXPECT_EQ(inst1Dest[0].reg.idx, 10);
    EXPECT_EQ(inst1Src[0].reg.idx, 11);
    EXPECT_EQ(inst1Src[1].reg.idx, 12);

    // Second instantiation: map to v[20:22]
    std::vector<StinkyRegister> inst2Dest, inst2Src;
    for (auto& reg : templateDest) inst2Dest.push_back(reg.resolveVirtualToPhysical(20));
    for (auto& reg : templateSrc) inst2Src.push_back(reg.resolveVirtualToPhysical(20));

    EXPECT_EQ(inst2Dest[0].reg.idx, 20);
    EXPECT_EQ(inst2Src[0].reg.idx, 21);
    EXPECT_EQ(inst2Src[1].reg.idx, 22);

    // Template should still be virtual and unchanged
    EXPECT_TRUE(templateDest[0].isVirtualReg());
    EXPECT_TRUE(templateSrc[0].isVirtualReg());
    EXPECT_TRUE(templateSrc[1].isVirtualReg());
    EXPECT_EQ(templateDest[0].reg.idx, 0u | StinkyRegister::kVirtualBit);
    EXPECT_EQ(templateSrc[0].reg.idx, 1u | StinkyRegister::kVirtualBit);
    EXPECT_EQ(templateSrc[1].reg.idx, 2u | StinkyRegister::kVirtualBit);
}

TEST_F(VirtualisterTest, SeparateVGPRAndSGPROffsets) {
    // Test that VGPR and SGPR can have different offsets
    auto vTemp = StinkyRegister::Virtual(0);
    auto sTemp = StinkyRegister::VirtualSGPR(0);

    auto v50 = vTemp.resolveVirtualToPhysical(50);
    auto s10 = sTemp.resolveVirtualToPhysical(10);

    EXPECT_EQ(v50.reg.type, RegType::V);
    EXPECT_EQ(v50.reg.idx, 50);

    EXPECT_EQ(s10.reg.type, RegType::S);
    EXPECT_EQ(s10.reg.idx, 10);
}

// ============================================================================
// Edge Cases and Validation
// ============================================================================

TEST_F(VirtualisterTest, ZeroOffset) {
    // Test that zero offset still marks register as non-virtual
    auto vTemp = StinkyRegister::Virtual(5);
    auto v5 = vTemp.resolveVirtualToPhysical(0);

    EXPECT_FALSE(v5.isVirtualReg());
    EXPECT_EQ(v5.reg.idx, 5);
}

TEST_F(VirtualisterTest, NegativeIndexInVirtual) {
    // Edge case: can we create virtual register with negative index?
    // (This might be implementation-defined behavior)
    auto vTemp = StinkyRegister::Virtual(-1);

    // The register system uses unsigned internally, so this will wrap
    // Just verify it's created and marked as virtual
    EXPECT_TRUE(vTemp.isVirtualReg());
}

TEST_F(VirtualisterTest, LargeOffset) {
    // Test with large offset values
    auto vTemp = StinkyRegister::Virtual(10);
    auto v1000 = vTemp.resolveVirtualToPhysical(990);

    EXPECT_EQ(v1000.reg.idx, 1000);
    EXPECT_FALSE(v1000.isVirtualReg());
}

TEST_F(VirtualisterTest, RegisterEquality) {
    // Test that remapped registers compare correctly
    auto vTemp = StinkyRegister::Virtual(0);
    auto v10a = vTemp.resolveVirtualToPhysical(10);
    auto v10b = vTemp.resolveVirtualToPhysical(10);
    auto v20 = vTemp.resolveVirtualToPhysical(20);

    // Same physical register
    EXPECT_TRUE(v10a == v10b);
    EXPECT_FALSE(v10a != v10b);

    // Different physical registers
    EXPECT_FALSE(v10a == v20);
    EXPECT_TRUE(v10a != v20);

    // Virtual vs physical
    EXPECT_FALSE(vTemp == v10a);
}
