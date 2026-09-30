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
#include "stinkytofu/analysis/logical/IRVerifierPass.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/logical/LogicalInstructions.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

void populateLogicalFunc(Function& func) {
    BasicBlock* bb = func.createBasicBlock("entry");
    StinkyRegister v0 = vgpr(0), v1 = vgpr(1), v2 = vgpr(2);
    bb->appendIR(static_cast<IRBase*>(VAddU32(v0, v1, v2)));
}

}  // namespace

TEST(LogicalIRVerifierTest, EmptyFunctionReturnsError) {
    Function func("empty");
    std::string err = validateLogicalIR(func);
    EXPECT_FALSE(err.empty());
}

TEST(LogicalIRVerifierTest, FunctionWithNoInstructionsReturnsError) {
    Function func("no_insts");
    func.createBasicBlock("entry");
    std::string err = validateLogicalIR(func);
    EXPECT_FALSE(err.empty());
}

TEST(LogicalIRVerifierTest, ValidLogicalIRReturnsEmpty) {
    Function func("f");
    populateLogicalFunc(func);
    std::string err = validateLogicalIR(func);
    EXPECT_TRUE(err.empty()) << "Unexpected error: " << err;
}

TEST(LogicalIRVerifierTest, MixedIRReturnsError) {
    Function func("mixed");
    setFunctionArch(func, GfxArchID::Gfx1250);
    BasicBlock* bb = func.createBasicBlock("entry");

    // Add a logical instruction followed by a StinkyInstruction
    StinkyRegister v0 = vgpr(0), v1 = vgpr(1), v2 = vgpr(2);
    bb->appendIR(static_cast<IRBase*>(VAddU32(v0, v1, v2)));

    AsmIRBuilder builder(*bb, GfxArchID::Gfx1250);
    builder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));

    std::string err = validateLogicalIR(func);
    EXPECT_FALSE(err.empty()) << "Expected error for mixed IR, got none";
    EXPECT_NE(err.find("StinkyTofu"), std::string::npos) << err;
}

TEST(LogicalIRVerifierTest, VerboseModeDoesNotCrash) {
    Function func("f");
    populateLogicalFunc(func);
    LogicalIRVerifierConfig cfg;
    cfg.verbose = true;
    std::string err = validateLogicalIR(func, cfg);
    EXPECT_TRUE(err.empty());
}

TEST(LogicalIRVerifierTest, PassRunsSuccessfullyOnValidIR) {
    Function func("f");
    populateLogicalFunc(func);
    LogicalIRVerifierConfig cfg;
    cfg.abortOnError = false;
    LogicalIRVerifierPass pass(cfg);
    PassContext ctx;
    AnalysisManager am;
    EXPECT_NO_FATAL_FAILURE(pass.run(func, ctx, am));
}
