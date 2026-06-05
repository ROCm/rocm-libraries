/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include <sstream>
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/transforms/asm/LoopRegionRemarkPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

static StinkyRegister litInt(int32_t v) {
    StinkyRegister r;
    r.dataType = StinkyRegister::Type::LiteralInt;
    r.literalInt = v;
    return r;
}

class LoopRegionRemarkPassTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    GemmTileConfig config;
    std::unique_ptr<Function> func;
    std::unique_ptr<Pass> pass;
    AnalysisManager am;

    void SetUp() override {
        config.arch[0] = 12;
        config.arch[1] = 5;
        config.arch[2] = 0;
        func = std::make_unique<Function>("loop_region_test");
        setFunctionArch(*func, arch);
        pass = createLoopRegionRemarkPass();
        registerAllAnalyses(am);
    }

    // Run pass with remarks enabled, capture stderr output
    std::string runPassWithRemarks() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        ctx.setRemarksEnabled(true);

        std::ostringstream captured;
        auto* origBuf = std::cerr.rdbuf(captured.rdbuf());
        pass->run(*func, ctx, am);
        std::cerr.rdbuf(origBuf);
        return captured.str();
    }

    // Run pass without remarks — should produce no output
    std::string runPassWithoutRemarks() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        ctx.setRemarksEnabled(false);

        std::ostringstream captured;
        auto* origBuf = std::cerr.rdbuf(captured.rdbuf());
        pass->run(*func, ctx, am);
        std::cerr.rdbuf(origBuf);
        return captured.str();
    }

    // Build a self-loop BB
    BasicBlock* buildLoopBB(const char* label = "loop_body") {
        BasicBlock* body = func->createBasicBlock(label);
        body->addSuccessor(body);
        return body;
    }
};

// No loops: pass should produce no output
TEST_F(LoopRegionRemarkPassTest, NoLoopsNoOutput) {
    func->createBasicBlock("entry");
    std::string output = runPassWithRemarks();
    EXPECT_TRUE(output.empty());
}

// Remarks disabled: pass should produce no output even with loops
TEST_F(LoopRegionRemarkPassTest, RemarksDisabledNoOutput) {
    BasicBlock* body = buildLoopBB("loop_body");
    createVAddInBlock(body, arch, 0, 1, 2);
    createVAddInBlock(body, arch, 3, 4, 5);

    std::string output = runPassWithoutRemarks();
    EXPECT_TRUE(output.empty());
}

// Simple loop with no side effects: should report 1 region
TEST_F(LoopRegionRemarkPassTest, SingleRegionLoop) {
    BasicBlock* body = buildLoopBB("loop_body");
    createVAddInBlock(body, arch, 0, 1, 2);
    createVAddInBlock(body, arch, 3, 4, 5);

    std::string output = runPassWithRemarks();
    EXPECT_NE(output.find("loop 'loop_body' summary"), std::string::npos);
    EXPECT_NE(output.find("1 regions"), std::string::npos);
    EXPECT_NE(output.find("0 s_nop"), std::string::npos);
}

// Loop with a wait instruction: should report 2 regions and a [wait] boundary
TEST_F(LoopRegionRemarkPassTest, WaitCreatesRegionBoundary) {
    BasicBlock* body = buildLoopBB("loop_body");

    createVAddInBlock(body, arch, 0, 1, 2);

    // Insert s_waitcnt — this is a non-movable side effect
    {
        AsmIRBuilder builder(*body, arch);
        StinkyInstruction* waitInst = builder.create(getMCIDByUOp(GFX::s_waitcnt, arch));
        waitInst->addSrcReg(litInt(0));
    }

    createVAddInBlock(body, arch, 3, 4, 5);

    std::string output = runPassWithRemarks();
    EXPECT_NE(output.find("2 regions"), std::string::npos);
    EXPECT_NE(output.find("[wait]"), std::string::npos);
}

// Loop with untokenized ds_load (no LDS pseudo-regs): should report as boundary
TEST_F(LoopRegionRemarkPassTest, UntokenizedDsLoadCreatesBoundary) {
    BasicBlock* body = buildLoopBB("loop_body");

    createVAddInBlock(body, arch, 0, 1, 2);

    // ds_load without LDS pseudo-register — non-movable
    createDsReadB128InBlock(body, arch, 10, 20);

    createVAddInBlock(body, arch, 3, 4, 5);

    std::string output = runPassWithRemarks();
    EXPECT_NE(output.find("2 regions"), std::string::npos);
    EXPECT_NE(output.find("[untokenized_mem]"), std::string::npos);
    EXPECT_NE(output.find("no MemTokenData"), std::string::npos);
}

// Loop with tokenized ds_load (has LDS pseudo-reg): should NOT be a boundary
TEST_F(LoopRegionRemarkPassTest, TokenizedDsLoadNotBoundary) {
    BasicBlock* body = buildLoopBB("loop_body");

    createVAddInBlock(body, arch, 0, 1, 2);

    // ds_load WITH LDS pseudo-register — movable, not a boundary
    {
        StinkyInstruction* inst = createDsReadB128InBlock(body, arch, 10, 20);
        inst->addSrcReg(StinkyRegister(RegType::LDS, 0, 1));
    }

    createVAddInBlock(body, arch, 3, 4, 5);

    std::string output = runPassWithRemarks();
    EXPECT_NE(output.find("1 regions"), std::string::npos);
}

// Loop with s_nop: should report wasted cycles
TEST_F(LoopRegionRemarkPassTest, SnopDetection) {
    BasicBlock* body = buildLoopBB("loop_body");

    createVAddInBlock(body, arch, 0, 1, 2);

    // Insert s_nop 3 (wastes 4 cycles)
    {
        AsmIRBuilder builder(*body, arch);
        StinkyInstruction* nopInst = builder.create(getMCIDByUOp(GFX::s_nop, arch));
        nopInst->addSrcReg(litInt(3));
    }

    createVAddInBlock(body, arch, 3, 4, 5);

    std::string output = runPassWithRemarks();
    EXPECT_NE(output.find("1 s_nop"), std::string::npos);
    EXPECT_NE(output.find("4 wasted cycles"), std::string::npos);
}

// Loop with multiple boundaries: should report correct count
TEST_F(LoopRegionRemarkPassTest, MultipleBoundaries) {
    BasicBlock* body = buildLoopBB("loop_body");

    createVAddInBlock(body, arch, 0, 1, 2);

    // wait
    {
        AsmIRBuilder builder(*body, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_waitcnt, arch));
        inst->addSrcReg(litInt(0));
    }

    createVAddInBlock(body, arch, 3, 4, 5);

    // untokenized ds_load
    createDsReadB128InBlock(body, arch, 10, 20);

    createVAddInBlock(body, arch, 6, 7, 8);

    std::string output = runPassWithRemarks();
    EXPECT_NE(output.find("3 regions"), std::string::npos);
    EXPECT_NE(output.find("2 boundaries"), std::string::npos);
}
