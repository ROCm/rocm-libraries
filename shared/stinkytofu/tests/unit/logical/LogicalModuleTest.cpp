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

#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/transforms/logical/LowerLogicalModulePipeline.hpp"

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

TEST(IRModuleTest, LowersConditionalDirectivesInSourceOrder) {
    PyLogicalModule module("test_kernel");
    module.add(makeLogicalInstructionShared(SEndpgm()));
    module.addIfDirective("0");
    module.add(makeLogicalInstructionShared(SEndpgm()));
    module.addEndifDirective("overflowed resources");

    const auto& directives = module.getConditionalDirectives();
    ASSERT_EQ(directives.size(), 2);
    EXPECT_EQ(directives[0].position, 1);
    EXPECT_EQ(directives[0].kind, ConditionalDirectiveKind::IF);
    EXPECT_EQ(directives[1].position, 2);
    EXPECT_EQ(directives[1].kind, ConditionalDirectiveKind::ENDIF);

    auto asmModule = lowerLogicalModuleToAsm(module, {12, 5, 0});
    const std::string assembly = asmModule->emitAssembly();
    const size_t ifPos = assembly.find(".if 0\n");
    const std::string endifText =
        std::string(".endif") + std::string(44, ' ') + " // overflowed resources\n";
    const size_t endifPos = assembly.find(endifText);
    ASSERT_NE(ifPos, std::string::npos);
    ASSERT_NE(endifPos, std::string::npos);
    EXPECT_LT(ifPos, endifPos);
}

TEST(IRModuleTest, RecordsCallableMarkersInSourceOrder) {
    PyLogicalModule module("test_kernel");

    module.beginCallable("label_Activation_Relu_VW1");
    module.add(makeLogicalInstructionShared(VMovB32(vgpr(0), vgpr(1))));
    module.endCallable("label_Activation_Relu_VW1");

    const auto& markers = module.getCallableMarkers();
    ASSERT_EQ(markers.size(), 2);
    EXPECT_TRUE(markers[0].isBegin);
    EXPECT_EQ(markers[0].name, "label_Activation_Relu_VW1");
    EXPECT_EQ(markers[0].position, 0);
    EXPECT_FALSE(markers[1].isBegin);
    EXPECT_EQ(markers[1].name, "label_Activation_Relu_VW1");
    EXPECT_EQ(markers[1].position, 1);
}

TEST(IRModuleTest, LowersCallableIntoSeparateFunction) {
    PyLogicalModule module("test_kernel");
    module.add(makeLogicalInstructionShared(VMovB32(vgpr(0), vgpr(1))));
    module.beginCallable("label_Activation_Relu_VW1");
    module.addLabel("label_Activation_Relu_VW1");
    module.add(makeLogicalInstructionShared(VMovB32(vgpr(2), vgpr(3))));
    module.endCallable("label_Activation_Relu_VW1");
    module.addLabel("label_ASM_End");

    auto asmModule = lowerLogicalModuleToAsm(module, {12, 5, 0});

    ASSERT_EQ(asmModule->numFunctions(), 2);
    const std::string assembly = asmModule->emitAssembly();
    ASSERT_NE(assembly.find("label_ASM_End:"), std::string::npos);
    ASSERT_NE(assembly.find("label_Activation_Relu_VW1:"), std::string::npos);
    EXPECT_LT(assembly.find("label_ASM_End:"), assembly.find("label_Activation_Relu_VW1:"));
}

// Adaptor path: HWRegContainer.toString() becomes a LiteralString on logical IR.
// ToStinkyAsmPass must parse it via the per-arch DEF_HWREG table so emit matches
// the rocisa converter (hwreg(28,6,4) not hwreg(HW_REG_IB_STS2,6,4)).
TEST(IRModuleTest, LowersSymbolicHwregLiteralToNumericId) {
    PyLogicalModule module("test_kernel");
    module.add(makeLogicalInstructionShared(SGetRegB32(
        sgpr(60), StinkyRegister(std::string("hwreg(HW_REG_IB_STS2,6,4)")), "cluster_id")));
    module.add(makeLogicalInstructionShared(SSetRegIMM32B32(
        StinkyRegister(std::string("hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2)")), StinkyRegister(0))));

    auto asmModule = lowerLogicalModuleToAsm(module, {12, 5, 0});
    const std::string assembly = asmModule->emitAssembly();
    EXPECT_NE(assembly.find("hwreg(28,6,4)"), std::string::npos) << assembly;
    EXPECT_EQ(assembly.find("HW_REG_IB_STS2"), std::string::npos) << assembly;
    EXPECT_NE(assembly.find("hwreg(26,0,2)"), std::string::npos) << assembly;
    EXPECT_EQ(assembly.find("HW_REG_WAVE_SCHED_MODE"), std::string::npos) << assembly;

    bool sawGetregHwreg = false;
    for (Function* fn : asmModule->getFunctions()) {
        ASSERT_NE(fn, nullptr);
        for (BasicBlock& bb : *fn) {
            for (IRBase& ir : bb) {
                if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
                auto* inst = static_cast<StinkyInstruction*>(&ir);
                if (inst->getUnifiedOpcode() != GFX::s_getreg_b32) continue;
                const auto& srcs = inst->getSrcRegs();
                ASSERT_FALSE(srcs.empty());
                EXPECT_EQ(srcs[0].dataType, StinkyRegister::Type::HwReg);
                EXPECT_EQ(srcs[0].hwreg.id, 28);
                EXPECT_EQ(srcs[0].hwreg.offset, 6);
                EXPECT_EQ(srcs[0].hwreg.size, 4);
                sawGetregHwreg = true;
            }
        }
    }
    EXPECT_TRUE(sawGetregHwreg);
}

TEST(IRModuleTest, LeavesUnknownHwregLiteralUnchanged) {
    PyLogicalModule module("test_kernel");
    module.add(makeLogicalInstructionShared(
        SGetRegB32(sgpr(0), StinkyRegister(std::string("hwreg(HW_REG_DOES_NOT_EXIST,6,4)")))));

    auto asmModule = lowerLogicalModuleToAsm(module, {12, 5, 0});
    const std::string assembly = asmModule->emitAssembly();
    EXPECT_NE(assembly.find("hwreg(HW_REG_DOES_NOT_EXIST,6,4)"), std::string::npos) << assembly;
}
