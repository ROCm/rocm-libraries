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

#include <array>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/RederiveExpertScopePass.hpp"

using namespace stinkytofu;

namespace {

constexpr std::array<int, 3> kArch{12, 5, 0};

std::unique_ptr<StinkyAsmModule> makeModule() {
    StinkyAsmModule::ModuleOptions opts{};
    opts.OptLevel = 0;
    return std::make_unique<StinkyAsmModule>("test", kArch, opts);
}

}  // namespace

// When the module has no matching group the pass must return PreservedAnalyses::all()
// without touching IR (early-exit path, line 63 of RederiveExpertScopePass.cpp).
TEST(RederiveExpertScopePassTest, NoGroupEarlyExit) {
    auto module = makeModule();
    // Do NOT add "scope" group -- hasGroup("scope") will be false.
    Function& func = module->getFunction();
    BasicBlock* bb = func.getEntryBlock();
    if (!bb) bb = func.createBasicBlock("entry");

    AsmIRBuilder builder(*bb, getGfxArchID(kArch[0], kArch[1], kArch[2]));
    builder.create(getMCIDByUOp(GFX::s_nop, getGfxArchID(kArch[0], kArch[1], kArch[2])));

    PassContext ctx;
    AnalysisManager am;
    auto pass = createRederiveExpertScopePass(*module, "scope", "start_label", "end_group");
    EXPECT_NO_FATAL_FAILURE(pass->run(func, ctx, am));
}

// When the group exists but no start label matches, the pass clears the range
// (sets both iterators to default) rather than leaving stale pointers.
TEST(RederiveExpertScopePassTest, GroupExistsButNoStartLabel) {
    auto module = makeModule();
    module->addGroup("scope");

    Function& func = module->getFunction();
    BasicBlock* bb = func.getEntryBlock();
    if (!bb) bb = func.createBasicBlock("entry");

    // One s_nop -- no label instruction, so startLabel_ won't be found.
    AsmIRBuilder builder(*bb, getGfxArchID(kArch[0], kArch[1], kArch[2]));
    builder.create(getMCIDByUOp(GFX::s_nop, getGfxArchID(kArch[0], kArch[1], kArch[2])));

    PassContext ctx;
    AnalysisManager am;
    auto pass = createRederiveExpertScopePass(*module, "scope", "nonexistent_label", "end_group");
    EXPECT_NO_FATAL_FAILURE(pass->run(func, ctx, am));
}

TEST(RederiveExpertScopePassTest, PassNameIsCorrect) {
    auto module = makeModule();
    auto pass = createRederiveExpertScopePass(*module, "scope", "start", "end");
    EXPECT_STREQ(pass->getName(), "RederiveExpertScopePass");
}
