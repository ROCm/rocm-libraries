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

#include <iostream>
#include <sstream>
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/support/StandardInstrumentations.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

// VerifyInstrumentation runs the StinkyTofu ASM IR verifier after each pass and
// reports failures (without aborting) tagged with the pass name. These tests
// drive afterPass() directly with a known-valid and a known-invalid function.
//
// The IR shape mirrors ReadWriteOperandTest: s_cmov_b32 has a read-write
// destination, so the verifier requires the dst register to appear in srcRegs.
// Omitting it produces a deterministic "Read-write" failure. Register-width
// checking is disabled so the only thing under test is the RW rule.
class VerifyInstrumentationTest : public ::testing::Test {
   protected:
    GfxArchID arch;

    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
    }

    const HwInstDesc* descFor(const std::string& mnemonic) {
        uint16_t isaOp = getMnemonicToIsaOpcode(mnemonic, arch);
        if (isaOp == GFX::INVALID) return nullptr;
        return getMCIDByIsaOp(isaOp, arch);
    }

    // Build a single s_cmov_b32 instruction. When `valid`, the RW destination
    // (sgpr1) also appears in srcRegs; otherwise it is missing and the verifier
    // flags it.
    void buildFunction(Function& func, bool valid) {
        setFunctionArch(func, arch);
        BasicBlock* bb = func.createBasicBlock("entry");
        const HwInstDesc* desc = descFor("s_cmov_b32");
        ASSERT_NE(desc, nullptr) << "s_cmov_b32 not found for arch";
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(sgpr(1));
        inst->addSrcReg(sgpr(2));
        if (valid) inst->addSrcReg(sgpr(1));
    }

    static AsmVerifierConfig rwOnlyConfig() {
        AsmVerifierConfig cfg;
        cfg.checkRegisterWidths = false;
        cfg.checkReadWriteOperands = true;
        return cfg;
    }

    // Invoke afterPass() while capturing std::cerr, returning what was printed.
    std::string captureAfterPass(VerifyInstrumentation& vi, const std::string& passName,
                                 Function& func) {
        PassContext ctx;
        std::ostringstream captured;
        std::streambuf* old = std::cerr.rdbuf(captured.rdbuf());
        vi.afterPass(passName, func, ctx);
        std::cerr.rdbuf(old);
        return captured.str();
    }
};

TEST_F(VerifyInstrumentationTest, ValidIRProducesNoOutput) {
    Function func("valid");
    buildFunction(func, /*valid=*/true);

    VerifyInstrumentation vi(rwOnlyConfig());
    std::string out = captureAfterPass(vi, "SomePass", func);
    EXPECT_TRUE(out.empty()) << "unexpected verifier output: " << out;
}

TEST_F(VerifyInstrumentationTest, InvalidIRReportsTaggedWithPassName) {
    Function func("invalid");
    buildFunction(func, /*valid=*/false);

    VerifyInstrumentation vi(rwOnlyConfig());
    std::string out = captureAfterPass(vi, "OffendingPass", func);

    EXPECT_NE(out.find("[verify-each]"), std::string::npos) << out;
    EXPECT_NE(out.find("OffendingPass"), std::string::npos) << out;
    EXPECT_NE(out.find("Read-write"), std::string::npos) << out;
}

TEST_F(VerifyInstrumentationTest, InvalidIRReportsButDoesNotAbort) {
    // The instrumentation forces abortOnError = false, so a failing verification
    // must return normally (report-only) and let the pipeline continue.
    Function func("invalid");
    buildFunction(func, /*valid=*/false);

    VerifyInstrumentation vi(rwOnlyConfig());
    EXPECT_NO_FATAL_FAILURE({
        std::string out = captureAfterPass(vi, "OffendingPass", func);
        EXPECT_FALSE(out.empty());
    });
}

}  // namespace
