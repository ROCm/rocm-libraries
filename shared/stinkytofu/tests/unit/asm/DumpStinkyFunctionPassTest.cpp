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

#include <filesystem>
#include <fstream>
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/DumpStinkyFunctionPass.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;
namespace fs = std::filesystem;

class DumpStinkyFunctionPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("dump_test");
        setFunctionArch(*func, GfxArchID::Gfx1250);
        BasicBlock* bb = func->createBasicBlock("entry");
        AsmIRBuilder builder(*bb, GfxArchID::Gfx1250);
        builder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));

        stirPath = (fs::temp_directory_path() / "stinkytofu_dump_test.stir").string();
        asmPath = (fs::temp_directory_path() / "stinkytofu_dump_test.s").string();
    }

    void TearDown() override {
        fs::remove(stirPath);
        fs::remove(asmPath);
    }

    std::unique_ptr<Function> func;
    std::string stirPath;
    std::string asmPath;
};

TEST_F(DumpStinkyFunctionPassTest, EmptyConfigIsNoOp) {
    DumpStinkyFunctionPassConfig cfg;
    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyFunctionPass(cfg);
    EXPECT_NO_FATAL_FAILURE(pass->run(*func, ctx, am));
}

TEST_F(DumpStinkyFunctionPassTest, WritesStirFile) {
    DumpStinkyFunctionPassConfig cfg;
    cfg.stirPath = stirPath;

    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyFunctionPass(cfg);
    pass->run(*func, ctx, am);

    ASSERT_TRUE(fs::exists(stirPath));
    std::ifstream f(stirPath);
    std::string content((std::istreambuf_iterator<char>(f)), {});
    EXPECT_FALSE(content.empty());
}

TEST_F(DumpStinkyFunctionPassTest, EmitAsmWritesAssemblyFile) {
    DumpStinkyFunctionPassConfig cfg;
    cfg.emitAsm = true;
    cfg.asmPath = asmPath;

    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyFunctionPass(cfg);
    pass->run(*func, ctx, am);

    ASSERT_TRUE(fs::exists(asmPath));
    std::ifstream f(asmPath);
    std::string content((std::istreambuf_iterator<char>(f)), {});
    EXPECT_FALSE(content.empty());
}

TEST_F(DumpStinkyFunctionPassTest, GetConfigReturnsSetConfig) {
    DumpStinkyFunctionPassConfig cfg;
    cfg.stirPath = stirPath;

    DumpStinkyFunctionPass pass(cfg);
    EXPECT_EQ(pass.getConfig().stirPath, stirPath);
}

TEST_F(DumpStinkyFunctionPassTest, SetConfigUpdatesConfig) {
    DumpStinkyFunctionPass pass;
    DumpStinkyFunctionPassConfig cfg;
    cfg.stirPath = stirPath;
    pass.setConfig(cfg);
    EXPECT_EQ(pass.getConfig().stirPath, stirPath);
}

TEST_F(DumpStinkyFunctionPassTest, EmitAsmDerivesPathFromStirPath) {
    DumpStinkyFunctionPassConfig cfg;
    cfg.stirPath = stirPath;
    cfg.emitAsm = true;

    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyFunctionPass(cfg);
    pass->run(*func, ctx, am);

    // asmPath is stirPath with .stir replaced by .s
    std::string derivedAsm = stirPath.substr(0, stirPath.rfind('.')) + ".s";
    EXPECT_TRUE(fs::exists(derivedAsm));
    fs::remove(derivedAsm);
}
