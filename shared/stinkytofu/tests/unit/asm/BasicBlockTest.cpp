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

#include <sstream>

#include "TestHelpers.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

class BasicBlockTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("test_func");
        setFunctionArch(*func, GfxArchID::Gfx1250);
    }

    std::unique_ptr<Function> func;
};

TEST_F(BasicBlockTest, GetParentFuncReturnsOwningFunction) {
    BasicBlock* bb = func->createBasicBlock("entry");
    EXPECT_EQ(bb->getParentFunc(), func.get());
}

TEST_F(BasicBlockTest, RemoveDetachesFromFunction) {
    BasicBlock* bb = func->createBasicBlock("entry");
    ASSERT_EQ(func->size(), 1u);
    bb->remove();
    EXPECT_EQ(func->size(), 0u);
    delete bb;
}

TEST_F(BasicBlockTest, EraseRemovesAndDeletesBlock) {
    func->createBasicBlock("bb0");
    BasicBlock* bb1 = func->createBasicBlock("bb1");
    ASSERT_EQ(func->size(), 2u);
    bb1->erase();
    EXPECT_EQ(func->size(), 1u);
}

TEST_F(BasicBlockTest, DumpLabeledBlockDoesNotCrash) {
    BasicBlock* bb = func->createBasicBlock("my_label");
    AsmIRBuilder builder(*bb, GfxArchID::Gfx1250);
    builder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));

    std::streambuf* old = std::cerr.rdbuf(nullptr);
    bb->dump();
    std::cerr.rdbuf(old);
}

TEST_F(BasicBlockTest, DumpUnlabeledBlockDoesNotCrash) {
    BasicBlock* bb = func->createBasicBlock("");
    std::streambuf* old = std::cerr.rdbuf(nullptr);
    bb->dump();
    std::cerr.rdbuf(old);
}

TEST_F(BasicBlockTest, DumpBlockWithSuccessorDoesNotCrash) {
    BasicBlock* bb0 = func->createBasicBlock("bb0");
    BasicBlock* bb1 = func->createBasicBlock("bb1");
    bb0->addSuccessor(bb1);

    std::streambuf* old = std::cerr.rdbuf(nullptr);
    bb0->dump();
    std::cerr.rdbuf(old);
}
