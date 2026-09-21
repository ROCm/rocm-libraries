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

// Tests for AccumulateInstructionSizePass as a pass (createAccumulateInstructionSizePass
// factories + Pass::run), separate from InstructionSizeCosting helper-function tests in
// AccumulateInstructionSizePassTest.cpp.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"

using namespace stinkytofu;
namespace fs = std::filesystem;

namespace {

constexpr std::array<int, 3> kArch{12, 5, 0};

std::unique_ptr<StinkyAsmModule> makeModule() {
    StinkyAsmModule::ModuleOptions opts{};
    opts.OptLevel = 0;
    return std::make_unique<StinkyAsmModule>("test", kArch, opts);
}

std::unique_ptr<Function> makeFunc() {
    auto func = std::make_unique<Function>("f");
    GfxArchID arch = getGfxArchID(kArch[0], kArch[1], kArch[2]);
    BasicBlock* bb = func->createBasicBlock("entry");
    AsmIRBuilder builder(*bb, arch);
    builder.create(getMCIDByUOp(GFX::s_nop, arch));
    builder.create(getMCIDByUOp(GFX::s_nop, arch));
    return func;
}

}  // namespace

class AccumulateInstructionSizePassRunTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = makeFunc();
        debugPath = (fs::temp_directory_path() / "stinkytofu_accum_debug.txt").string();
    }

    void TearDown() override {
        fs::remove(debugPath);
    }

    std::unique_ptr<Function> func;
    std::string debugPath;
    AnalysisManager am;
};

TEST_F(AccumulateInstructionSizePassRunTest, BasicFactoryRunsWithoutError) {
    auto pass = createAccumulateInstructionSizePass(std::string{});
    PassContext ctx;
    EXPECT_NO_FATAL_FAILURE(pass->run(*func, ctx, am));
}

TEST_F(AccumulateInstructionSizePassRunTest, NameAndPassIDNotNull) {
    auto pass = createAccumulateInstructionSizePass(std::string{});
    EXPECT_STREQ(pass->getName(), "AccumulateInstructionSizePass");
    EXPECT_NE(pass->getPassID(), nullptr);
}

TEST_F(AccumulateInstructionSizePassRunTest, DebugPathFactoryWritesFile) {
    auto pass = createAccumulateInstructionSizePass(debugPath);
    PassContext ctx;
    pass->run(*func, ctx, am);
    EXPECT_TRUE(fs::exists(debugPath));
    std::ifstream f(debugPath);
    std::string content((std::istreambuf_iterator<char>(f)), {});
    EXPECT_FALSE(content.empty());
}

TEST_F(AccumulateInstructionSizePassRunTest, DebugOutputContainsOpcode) {
    auto pass = createAccumulateInstructionSizePass(debugPath);
    PassContext ctx;
    pass->run(*func, ctx, am);
    std::ifstream f(debugPath);
    std::string content((std::istreambuf_iterator<char>(f)), {});
    EXPECT_NE(content.find("AccumulateInstructionSizePass"), std::string::npos) << content;
}

TEST_F(AccumulateInstructionSizePassRunTest, ModuleFactorySetsTotalBytes) {
    auto module = makeModule();
    auto pass = createAccumulateInstructionSizePass(*module);
    PassContext ctx;
    Function& func = module->getFunction();
    BasicBlock* bb = func.getEntryBlock();
    if (!bb) bb = func.createBasicBlock("entry");
    GfxArchID arch = getGfxArchID(kArch[0], kArch[1], kArch[2]);
    AsmIRBuilder builder(*bb, arch);
    builder.create(getMCIDByUOp(GFX::s_nop, arch));

    pass->run(func, ctx, am);
    EXPECT_GT(module->getTotalInstructionBytes(), 0);
}

TEST_F(AccumulateInstructionSizePassRunTest, EmptyFunctionRunsWithoutError) {
    Function empty("empty");
    empty.createBasicBlock("entry");
    auto pass = createAccumulateInstructionSizePass(std::string{});
    PassContext ctx;
    EXPECT_NO_FATAL_FAILURE(pass->run(empty, ctx, am));
}

TEST_F(AccumulateInstructionSizePassRunTest, MultipleBBsFunctionRunsWithoutError) {
    Function func("multi");
    GfxArchID arch = getGfxArchID(kArch[0], kArch[1], kArch[2]);
    BasicBlock* bb0 = func.createBasicBlock("entry");
    BasicBlock* bb1 = func.createBasicBlock("loop");
    AsmIRBuilder b0(*bb0, arch), b1(*bb1, arch);
    b0.create(getMCIDByUOp(GFX::s_nop, arch));
    b1.create(getMCIDByUOp(GFX::s_nop, arch));

    auto pass = createAccumulateInstructionSizePass(std::string{});
    PassContext ctx;
    EXPECT_NO_FATAL_FAILURE(pass->run(func, ctx, am));
}
