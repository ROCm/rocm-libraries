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

#include <memory>
#include <sstream>
#include <string>

#include "PhiTestFixtures.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"
#include "stinkytofu/transforms/asm/ssa/SSADestruction.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class SSADestructionTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
    }

    /// Creates the entry block, for tests that do not use a CFG fixture.
    BasicBlock* makeEntry() {
        setFunctionArch(*func, kArch);
        return func->createBasicBlock("entry");
    }

    std::string physicalIR() const {
        std::ostringstream out;
        AsmPrinter printer(out);
        printer.print(*func);
        return out.str();
    }

    void lift() {
        Expected<LiftAttachedSSAResult> lifted = liftAsmRegistersToAttachedSSA(*func);
        ASSERT_TRUE(lifted.hasValue()) << lifted.getError();
        ASSERT_TRUE(func->hasAttachedSSA());
    }

    std::unique_ptr<Function> func;
};

size_t blockArgumentCount(const Function& function) {
    size_t count = 0;
    for (const BasicBlock& bb : function) count += bb.ssaArguments().size();
    return count;
}

StinkySSAValue* firstPhiIncoming(Function& function) {
    for (BasicBlock& bb : function) {
        for (const SSABlockArgument& arg : bb.ssaArguments()) {
            if (arg.incoming.empty()) continue;
            if (arg.incoming.front().use == nullptr) continue;
            return arg.incoming.front().use->value();
        }
    }
    return nullptr;
}

}  // namespace

TEST_F(SSADestructionTest, LegacyColoringAssignsEveryValueItsOrigin) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();

    const AllocationResult legacy = createLegacyColoring(*func);

    EXPECT_EQ(legacy.valueCount(), func->ssaArena().valueCount());
    EXPECT_EQ(legacy.unassignedCount(), 0u);
    for (StinkySSAValue* value : func->ssaArena().values()) {
        ASSERT_NE(value, nullptr);
        ASSERT_TRUE(legacy.isAssigned(value->valueId()));
        ASSERT_TRUE(value->hasPhysicalBinding());
        const StinkySSAValue::PhysicalBinding& binding = value->physical();
        EXPECT_EQ(legacy.assignmentOf(value->valueId()),
                  (RegKey{binding.type, binding.idx, RegHalf::NONE}));
    }
}

TEST_F(SSADestructionTest, SuccessfulReplayRecordsRewrittenOperands) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();

    const SSADestructionResult result = destroyAttachedSSA(*func, createLegacyColoring(*func));
    ASSERT_TRUE(result.ok()) << result.toString();
    ASSERT_FALSE(result.rewritten.empty());
    EXPECT_EQ(result.rewritten.front().beforeIdx, result.rewritten.front().afterIdx);
}

TEST_F(SSADestructionTest, AllocationResultPrintsEveryValueInIdOrder) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();

    // v0 and v1 arrive as live-ins, the add defines v2: three values, all bound.
    EXPECT_EQ(createLegacyColoring(*func).toString(),
              "values=3 unassigned=0 lifted=v,s\n%1 v0\n%2 v1\n%3 v2\n");
}

TEST_F(SSADestructionTest, AllocationResultPrintsAnUnassignedValueAsADash) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();

    // A partial colouring has to be readable, because "which value is missing"
    // is the question a rejected destruction raises.
    AllocationResult partial(*func);
    partial.assign(2, RegKey{RegType::V, 7, RegHalf::NONE});

    EXPECT_EQ(partial.toString(), "values=3 unassigned=2 lifted=v,s\n%1 -\n%2 v7\n%3 -\n");
}

TEST_F(SSADestructionTest, LiftThenReplayIsAnIdentityTransform) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 4, 4, 5);
    createDSWriteInBlock(entry, kArch, 0, 4);
    const std::string before = physicalIR();

    lift();
    const SSADestructionResult result = destroyAttachedSSA(*func, createLegacyColoring(*func));

    ASSERT_TRUE(result.ok()) << result.toString();
    EXPECT_EQ(physicalIR(), before);
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(SSADestructionTest, ReplayIsAnIdentityTransformAcrossControlFlow) {
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);
    ASSERT_NE(cfg.entry, nullptr);
    const std::string before = physicalIR();

    lift();
    ASSERT_GT(blockArgumentCount(*func), 0u);
    const SSADestructionResult result = destroyAttachedSSA(*func, createLegacyColoring(*func));

    ASSERT_TRUE(result.ok()) << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(SSADestructionTest, ANonIdentityColoringActuallyRewritesTheOperands) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 8, 4, 5);
    lift();

    // Shift every value by a constant. Ranges stay consecutive and each PHI's
    // inputs still land on its result, so no copies are needed; only the
    // register numbers change.
    constexpr unsigned kShift = 100;
    AllocationResult shifted(*func);
    for (StinkySSAValue* value : func->ssaArena().values()) {
        ASSERT_NE(value, nullptr);
        ASSERT_TRUE(value->hasPhysicalBinding());
        const StinkySSAValue::PhysicalBinding& binding = value->physical();
        shifted.assign(value->valueId(), RegKey{binding.type, binding.idx + kShift, RegHalf::NONE});
    }

    const SSADestructionResult result = destroyAttachedSSA(*func, shifted);

    ASSERT_TRUE(result.ok()) << result.toString();
    const std::string after = physicalIR();
    EXPECT_TRUE(contains(after, "v[104:107] = \"st.ds_load_b128\"(v100)")) << after;
    EXPECT_TRUE(contains(after, "v108 = \"st.v_add_f32\"(v104, v105)")) << after;
}

TEST_F(SSADestructionTest, RejectsARangeSplitAcrossNonConsecutiveRegisters) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, 4, 0);
    lift();
    const std::string before = physicalIR();

    // Scatter the four DWORDs of the load, which no range operand can encode.
    AllocationResult scattered(*func);
    unsigned next = 20;
    for (StinkySSAValue* value : func->ssaArena().values()) {
        scattered.assign(value->valueId(), RegKey{RegType::V, next, RegHalf::NONE});
        next += 7;
    }

    const SSADestructionResult result = destroyAttachedSSA(*func, scattered);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "must be consecutive in operand order"))
        << result.toString();
    // Rejection is atomic: the function keeps its original registers.
    EXPECT_EQ(physicalIR(), before);
    EXPECT_TRUE(func->hasAttachedSSA());
}

TEST_F(SSADestructionTest, RejectsAnUnassignedValue) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();
    const std::string before = physicalIR();

    AllocationResult partial(*func);
    for (StinkySSAValue* value : func->ssaArena().values()) {
        if (value->valueId() == 1) continue;  // leave the first live-in uncoloured
        ASSERT_TRUE(value->hasPhysicalBinding());
        const StinkySSAValue::PhysicalBinding& binding = value->physical();
        partial.assign(value->valueId(), RegKey{binding.type, binding.idx, RegHalf::NONE});
    }

    const SSADestructionResult result = destroyAttachedSSA(*func, partial);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "%1 has no physical register")) << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(SSADestructionTest, RejectsAPhiThatWouldNeedACopy) {
    SelfLoopJoinCfg cfg = buildSelfLoopJoinCfg(*func, kArch);
    ASSERT_NE(cfg.entry, nullptr);
    lift();
    const std::string before = physicalIR();

    StinkySSAValue* moved = firstPhiIncoming(*func);
    ASSERT_NE(moved, nullptr);

    // Colour one PHI input somewhere other than the result: lowering that needs
    // a copy on the incoming edge, which is not implemented.
    AllocationResult colouring = createLegacyColoring(*func);
    colouring.assign(moved->valueId(), RegKey{RegType::V, 200, RegHalf::NONE});

    const SSADestructionResult result = destroyAttachedSSA(*func, colouring);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "needs a copy on the incoming edge"))
        << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(SSADestructionTest, RejectsAGraphThatNoLongerDescribesTheFunction) {
    BasicBlock* entry = makeEntry();
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();

    // Rewriting an operand behind attached SSA's back is the mistake the shape
    // fingerprint exists to catch.
    add->setSrcReg(0, StinkyRegister("v", 9, 1));
    const std::string before = physicalIR();

    const SSADestructionResult result = destroyAttachedSSA(*func, createLegacyColoring(*func));

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "the function changed after it was lifted"))
        << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(SSADestructionTest, RejectsAnAllocationComputedAgainstAnotherGraph) {
    BasicBlock* entry = makeEntry();
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();
    const AllocationResult stale = createLegacyColoring(*func);

    // Change the program and lift again. The attached SSA now matches the
    // function, so only the allocation is out of date.
    add->setSrcReg(0, StinkyRegister("v", 9, 1));
    lift();
    const std::string before = physicalIR();

    const SSADestructionResult result = destroyAttachedSSA(*func, stale);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "computed against a different graph"))
        << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(SSADestructionTest, AScopedLiftLeavesOutOfScopeOperandsUntouched) {
    BasicBlock* entry = makeEntry();
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    LiftAsmRegistersToSSAOptions options;
    options.classes = RegClassSet::only(RegType::S);
    Expected<LiftAttachedSSAResult> lifted = liftAsmRegistersToAttachedSSA(*func, options);
    ASSERT_TRUE(lifted.hasValue()) << lifted.getError();

    // Move every value this lift produced, which can only be the scalar operand.
    AllocationResult moved(*func);
    for (StinkySSAValue* value : func->ssaArena().values()) {
        ASSERT_TRUE(value->hasPhysicalBinding());
        const StinkySSAValue::PhysicalBinding& binding = value->physical();
        moved.assign(value->valueId(), RegKey{binding.type, binding.idx + 10, RegHalf::NONE});
    }

    const SSADestructionResult result = destroyAttachedSSA(*func, moved);

    ASSERT_TRUE(result.ok()) << result.toString();
    // The scalar moved; the vector destination is exactly as the producer wrote
    // it, because an out-of-scope operand contributes no slot to rewrite.
    EXPECT_TRUE(contains(physicalIR(), "v0 = \"st.v_mov_b32\"(s14)")) << physicalIR();
}

TEST_F(SSADestructionTest, RejectsAnAllocationFromAnotherLiftScope) {
    BasicBlock* entry = makeEntry();
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    LiftAsmRegistersToSSAOptions scalarOnly;
    scalarOnly.classes = RegClassSet::only(RegType::S);
    ASSERT_TRUE(liftAsmRegistersToAttachedSSA(*func, scalarOnly).hasValue());
    const AllocationResult scalarColouring = createLegacyColoring(*func);

    // Re-lift the same program for both classes. The physical program did not
    // change, so the shape fingerprint still matches and only the scope differs -
    // but the value IDs now mean something else.
    ASSERT_TRUE(liftAsmRegistersToAttachedSSA(*func).hasValue());
    const std::string before = physicalIR();

    const SSADestructionResult result = destroyAttachedSSA(*func, scalarColouring);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "computed against a lift of s but this SSA covers v,s"))
        << result.toString();
    EXPECT_EQ(physicalIR(), before);
}
