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
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <sstream>
#include <string>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;

namespace {

// DebugPrintInstrumentation dumps IR before/after passes. When constructed with
// a StinkyAsmModule, a dump of the module's entry Function is followed by every
// callee Function so whole-module debug output includes callee bodies (the
// PassManager only hands the observer the entry Function). Without a module it
// dumps only the Function it is given. These tests drive beforePass() directly
// and inspect the captured stream.
class DebugPrintInstrumentationTest : public ::testing::Test {
   protected:
    static constexpr std::array<int, 3> ARCH{12, 5, 0};

    StinkyAsmModule::ModuleOptions makeDefaultOptions() {
        StinkyAsmModule::ModuleOptions opts{};
        opts.OptLevel = 0;
        return opts;
    }

    // Append a v_mov_b32 writing vN to the given block.
    void addMov(BasicBlock& bb, int destReg) {
        AsmIRBuilder builder(bb, getGfxArchID(ARCH[0], ARCH[1], ARCH[2]));
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_mov_b32, archId()));
        inst->addDestReg(StinkyRegister("v", destReg, 1));
    }

    GfxArchID archId() {
        return getGfxArchID(ARCH[0], ARCH[1], ARCH[2]);
    }

    // Module with an entry Function plus one callable Function named "callee".
    std::unique_ptr<StinkyAsmModule> makeModuleWithCallee() {
        auto module = std::make_unique<StinkyAsmModule>("test", ARCH, makeDefaultOptions());
        addMov(*module->getFunction().getEntryBlock(), /*destReg=*/0);
        Function& callee = module->createFunction("callee", /*isCallable=*/true);
        addMov(*callee.getEntryBlock(), /*destReg=*/1);
        return module;
    }

    // Config that prints before every pass into a captured stream.
    std::unique_ptr<PassManagerDebugConfig> makeConfig(std::shared_ptr<std::ostream> sink) {
        auto cfg = std::make_unique<PassManagerDebugConfig>();
        cfg->setPrintBeforeAll(true);
        cfg->setDumpStreamBefore(std::move(sink));
        return cfg;
    }
};

TEST_F(DebugPrintInstrumentationTest, WithModuleDumpsEntryAndCallee) {
    auto module = makeModuleWithCallee();
    auto sink = std::make_shared<std::ostringstream>();

    DebugPrintInstrumentation inst(makeConfig(sink), module.get());
    PassContext ctx;
    inst.beforePass("SomePass", module->getFunction(), ctx);

    const std::string out = sink->str();
    EXPECT_NE(out.find("callee Function: callee"), std::string::npos)
        << "callee dump header missing:\n"
        << out;
}

TEST_F(DebugPrintInstrumentationTest, WithoutModuleDumpsOnlyEntry) {
    auto module = makeModuleWithCallee();
    auto sink = std::make_shared<std::ostringstream>();

    // No module supplied: only the given Function is dumped, no callee header.
    DebugPrintInstrumentation inst(makeConfig(sink), /*module=*/nullptr);
    PassContext ctx;
    inst.beforePass("SomePass", module->getFunction(), ctx);

    const std::string out = sink->str();
    EXPECT_EQ(out.find("callee Function:"), std::string::npos)
        << "callee should not be dumped without a module:\n"
        << out;
}

}  // namespace
