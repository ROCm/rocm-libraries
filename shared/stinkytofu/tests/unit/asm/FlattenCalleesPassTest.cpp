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

// FlattenCalleesPass re-merges callees into the entry at their CALLEE_BODY
// markers. The marker comes from the rocisa converter and can't be written in
// .stir, so these tests build the module (entry + callees + markers) in C++.
//
// TEMPORARY: FlattenCalleesPass is a stopgap (see its header). Once a proper
// module-pass infrastructure lands and SwPrefetchInsertionPass handles multiple
// functions directly, the pass — and this test — should be removed/replaced.

#include <gtest/gtest.h>

#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/FlattenCalleesPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;
constexpr std::array<int, 3> kArchTriple = {12, 5, 0};

// CALLEE_BODY pseudo naming the callee = the splice marker.
StinkyInstruction* addSpliceMarker(BasicBlock* bb, const std::string& callee) {
    AsmIRBuilder builder(*bb, kArch);
    return builder.createCalleeBody(callee);
}

// Ordered list of v_add dest-register indices in a block (skips FENCE etc.).
std::vector<unsigned> vaddDestSeq(const BasicBlock& bb) {
    std::vector<unsigned> out;
    for (const IRBase& ir : bb) {
        const auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst && inst->getUnifiedOpcode() == GFX::v_add_f32)
            out.push_back(inst->getDestReg(0).reg.idx);
    }
    return out;
}

int countMarkers(const BasicBlock& bb) {
    int n = 0;
    for (const IRBase& ir : bb) {
        const auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst && isCalleeBody(*inst)) ++n;
    }
    return n;
}

class FlattenCalleesPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        module = std::make_unique<StinkyAsmModule>("flatten_test", kArchTriple,
                                                   StinkyAsmModule::ModuleOptions{});
        entry = &module->getFunction();
        entry->setName("entry_func");
        setFunctionArch(*entry, kArch);
        entryBlock = entry->getEntryBlock();
    }

    void runPass() {
        auto pass = createFlattenCalleesPass(module->getFunctions());
        PassContext ctx;
        AnalysisManager am;
        pass->run(*entry, ctx, am);
    }

    std::unique_ptr<StinkyAsmModule> module;
    Function* entry = nullptr;
    BasicBlock* entryBlock = nullptr;
};

TEST_F(FlattenCalleesPassTest, InlinesCalleeAtMarkerPosition) {
    // entry: v0, [marker -> callee], v1
    createVAddInBlock(entryBlock, kArch, /*dest=*/0, 1, 2);
    addSpliceMarker(entryBlock, "callee_fn");
    createVAddInBlock(entryBlock, kArch, /*dest=*/1, 1, 2);

    // callee: v10, v11
    Function& callee = module->createFunction("callee_fn");
    setFunctionArch(callee, kArch);
    BasicBlock* calleeBlock = callee.getEntryBlock();
    createVAddInBlock(calleeBlock, kArch, /*dest=*/10, 1, 2);
    createVAddInBlock(calleeBlock, kArch, /*dest=*/11, 1, 2);

    runPass();

    // Callee body now sits exactly where the marker was: v0, v10, v11, v1.
    EXPECT_EQ(vaddDestSeq(*entryBlock), (std::vector<unsigned>{0, 10, 11, 1}));
    // Marker consumed; callee emptied.
    EXPECT_EQ(countMarkers(*entryBlock), 0);
    EXPECT_TRUE(calleeBlock->empty());
}

TEST_F(FlattenCalleesPassTest, InlinesMultipleCalleesInMarkerOrder) {
    createVAddInBlock(entryBlock, kArch, /*dest=*/0, 1, 2);
    addSpliceMarker(entryBlock, "callee_a");
    addSpliceMarker(entryBlock, "callee_b");
    createVAddInBlock(entryBlock, kArch, /*dest=*/9, 1, 2);

    Function& a = module->createFunction("callee_a");
    setFunctionArch(a, kArch);
    createVAddInBlock(a.getEntryBlock(), kArch, /*dest=*/20, 1, 2);

    Function& b = module->createFunction("callee_b");
    setFunctionArch(b, kArch);
    createVAddInBlock(b.getEntryBlock(), kArch, /*dest=*/30, 1, 2);

    runPass();

    EXPECT_EQ(vaddDestSeq(*entryBlock), (std::vector<unsigned>{0, 20, 30, 9}));
    EXPECT_EQ(countMarkers(*entryBlock), 0);
}

TEST_F(FlattenCalleesPassTest, ResolvesNestedCalleeMarkersOnLaterIteration) {
    // entry: v0, [marker -> outer], v1
    createVAddInBlock(entryBlock, kArch, /*dest=*/0, 1, 2);
    addSpliceMarker(entryBlock, "outer_callee");
    createVAddInBlock(entryBlock, kArch, /*dest=*/1, 1, 2);

    // outer: v10, [marker -> inner], v11
    Function& outer = module->createFunction("outer_callee");
    setFunctionArch(outer, kArch);
    BasicBlock* outerBlock = outer.getEntryBlock();
    createVAddInBlock(outerBlock, kArch, /*dest=*/10, 1, 2);
    addSpliceMarker(outerBlock, "inner_callee");
    createVAddInBlock(outerBlock, kArch, /*dest=*/11, 1, 2);

    // inner: v20
    Function& inner = module->createFunction("inner_callee");
    setFunctionArch(inner, kArch);
    createVAddInBlock(inner.getEntryBlock(), kArch, /*dest=*/20, 1, 2);

    runPass();

    EXPECT_EQ(vaddDestSeq(*entryBlock), (std::vector<unsigned>{0, 10, 20, 11, 1}));
    EXPECT_EQ(countMarkers(*entryBlock), 0);
    EXPECT_TRUE(outerBlock->empty());
    EXPECT_TRUE(inner.getEntryBlock()->empty());
}

TEST_F(FlattenCalleesPassTest, InlinesMultiBlockCalleeInFunctionOrder) {
    createVAddInBlock(entryBlock, kArch, /*dest=*/0, 1, 2);
    addSpliceMarker(entryBlock, "multi_block_callee");
    createVAddInBlock(entryBlock, kArch, /*dest=*/1, 1, 2);

    Function& callee = module->createFunction("multi_block_callee");
    setFunctionArch(callee, kArch);
    BasicBlock* calleeEntry = callee.getEntryBlock();
    createVAddInBlock(calleeEntry, kArch, /*dest=*/10, 1, 2);

    BasicBlock* calleeTail = callee.createBasicBlock("tail");
    createVAddInBlock(calleeTail, kArch, /*dest=*/11, 1, 2);

    runPass();

    EXPECT_EQ(vaddDestSeq(*entryBlock), (std::vector<unsigned>{0, 10, 11, 1}));
    EXPECT_EQ(countMarkers(*entryBlock), 0);
    EXPECT_TRUE(calleeEntry->empty());
    EXPECT_TRUE(calleeTail->empty());
}

TEST_F(FlattenCalleesPassTest, NoMarkersIsNoOp) {
    createVAddInBlock(entryBlock, kArch, /*dest=*/0, 1, 2);
    createVAddInBlock(entryBlock, kArch, /*dest=*/1, 1, 2);

    runPass();

    EXPECT_EQ(vaddDestSeq(*entryBlock), (std::vector<unsigned>{0, 1}));
}

TEST_F(FlattenCalleesPassTest, UnknownCalleeNameJustDropsMarker) {
    createVAddInBlock(entryBlock, kArch, /*dest=*/0, 1, 2);
    addSpliceMarker(entryBlock, "does_not_exist");
    createVAddInBlock(entryBlock, kArch, /*dest=*/1, 1, 2);

    runPass();

    // Nothing to inline; marker removed, surrounding instructions intact.
    EXPECT_EQ(vaddDestSeq(*entryBlock), (std::vector<unsigned>{0, 1}));
    EXPECT_EQ(countMarkers(*entryBlock), 0);
}

}  // namespace
