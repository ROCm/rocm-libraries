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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/DumpStinkyModulePass.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;
namespace fs = std::filesystem;

class DumpStinkyModulePassTest : public ::testing::Test {
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

TEST_F(DumpStinkyModulePassTest, EmptyConfigIsNoOp) {
    DumpStinkyModulePassConfig cfg;
    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyModulePass(cfg);
    EXPECT_NO_FATAL_FAILURE(pass->run(*func, ctx, am));
}

TEST_F(DumpStinkyModulePassTest, WritesStirFile) {
    DumpStinkyModulePassConfig cfg;
    cfg.stirPath = stirPath;

    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyModulePass(cfg);
    pass->run(*func, ctx, am);

    ASSERT_TRUE(fs::exists(stirPath));
    std::ifstream f(stirPath);
    std::string content((std::istreambuf_iterator<char>(f)), {});
    EXPECT_FALSE(content.empty());
}

TEST_F(DumpStinkyModulePassTest, EmitAsmWritesAssemblyFile) {
    DumpStinkyModulePassConfig cfg;
    cfg.emitAsm = true;
    cfg.asmPath = asmPath;

    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyModulePass(cfg);
    pass->run(*func, ctx, am);

    ASSERT_TRUE(fs::exists(asmPath));
    std::ifstream f(asmPath);
    std::string content((std::istreambuf_iterator<char>(f)), {});
    EXPECT_FALSE(content.empty());
}

TEST_F(DumpStinkyModulePassTest, GetConfigReturnsSetConfig) {
    DumpStinkyModulePassConfig cfg;
    cfg.stirPath = stirPath;

    DumpStinkyModulePass pass(cfg);
    EXPECT_EQ(pass.getConfig().stirPath, stirPath);
}

TEST_F(DumpStinkyModulePassTest, SetConfigUpdatesConfig) {
    DumpStinkyModulePass pass;
    DumpStinkyModulePassConfig cfg;
    cfg.stirPath = stirPath;
    pass.setConfig(cfg);
    EXPECT_EQ(pass.getConfig().stirPath, stirPath);
}

TEST_F(DumpStinkyModulePassTest, EmitAsmDerivesPathFromStirPath) {
    DumpStinkyModulePassConfig cfg;
    cfg.stirPath = stirPath;
    cfg.emitAsm = true;

    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyModulePass(cfg);
    pass->run(*func, ctx, am);

    // asmPath is stirPath with .stir replaced by .s
    std::string derivedAsm = stirPath.substr(0, stirPath.rfind('.')) + ".s";
    EXPECT_TRUE(fs::exists(derivedAsm));
    fs::remove(derivedAsm);
}

TEST_F(DumpStinkyModulePassTest, WritesEntryAndCallees) {
    StinkyAsmModule::ModuleOptions options;
    StinkyAsmModule module("dump_module_test", {12, 5, 0}, options);

    Function& entry = module.getFunction();
    entry.setName("entry_func");
    setFunctionArch(entry, GfxArchID::Gfx1250);
    AsmIRBuilder entryBuilder(*entry.getEntryBlock(), GfxArchID::Gfx1250);
    entryBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));

    Function& callee = module.createFunction("callee_func");
    setFunctionArch(callee, GfxArchID::Gfx1250);
    AsmIRBuilder calleeBuilder(*callee.getEntryBlock(), GfxArchID::Gfx1250);
    calleeBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));

    DumpStinkyModulePassConfig cfg;
    cfg.stirPath = stirPath;
    cfg.emitAsm = true;
    cfg.asmPath = asmPath;

    PassContext ctx;
    AnalysisManager am;
    auto pass = createDumpStinkyModulePass(module, cfg);
    pass->run(entry, ctx, am);

    ASSERT_TRUE(fs::exists(stirPath));
    std::ifstream stirFile(stirPath);
    std::string stirContent((std::istreambuf_iterator<char>(stirFile)), {});
    EXPECT_EQ(stirContent.find("st.module @dump_module_test {\n"), 0u);
    EXPECT_NE(stirContent.find("  st.func @entry_func()"), std::string::npos);
    EXPECT_NE(stirContent.find("  st.func @callee_func()"), std::string::npos);
    EXPECT_NE(stirContent.rfind("}\n"), std::string::npos);

    ASSERT_TRUE(fs::exists(asmPath));
    std::ifstream asmFile(asmPath);
    std::string asmContent((std::istreambuf_iterator<char>(asmFile)), {});
    EXPECT_EQ(static_cast<size_t>(std::count(asmContent.begin(), asmContent.end(), '\n')), 2u);
}
