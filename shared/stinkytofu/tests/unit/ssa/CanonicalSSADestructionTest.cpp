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

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "PhiTestFixtures.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSAAllocation.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/ssa/CanonicalSSADestruction.hpp"
#include "stinkytofu/transforms/ssa/LiftAsmRegistersToSSAPass.hpp"
#include "stinkytofu/transforms/ssa/ReplayLegacyColoringPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class CanonicalSSADestructionTest : public ::testing::Test {
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

    /// Lifts into `ssa`, which the test then hands to the lowering core. The
    /// graph is an ordinary local now: nothing on the function holds it.
    void lift() {
        Expected<CanonicalSSA> lifted = liftAsmRegistersToSSA(*func);
        ASSERT_TRUE(lifted.hasValue()) << lifted.getError();
        ssa = std::move(*lifted);
    }

    std::unique_ptr<Function> func;
    CanonicalSSA ssa;
};

}  // namespace

TEST_F(CanonicalSSADestructionTest, LegacyColoringAssignsEveryValueItsOrigin) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();

    const AllocationResult legacy = createLegacyColoring(ssa);

    EXPECT_EQ(legacy.valueCount(), ssa.valueCount());
    EXPECT_EQ(legacy.unassignedCount(), 0u);
    for (const SSAValue& value : ssa.values()) {
        ASSERT_TRUE(legacy.isAssigned(value.id));
        EXPECT_EQ(legacy.assignmentOf(value.id), value.origin);
    }
}

TEST_F(CanonicalSSADestructionTest, LiftThenReplayIsAnIdentityTransform) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 4, 4, 5);
    createDSWriteInBlock(entry, kArch, 0, 4);
    const std::string before = physicalIR();

    lift();
    const SSADestructionResult result = replayLegacyColoring(*func, ssa);

    ASSERT_TRUE(result.ok()) << result.toString();
    EXPECT_EQ(physicalIR(), before);
    // Lowering rewrites operands but does not decide the graph's fate; the pass
    // owns that, which PassRoundTripsThroughThePassManager covers.
}

TEST_F(CanonicalSSADestructionTest, ReplayIsAnIdentityTransformAcrossControlFlow) {
    IteratedDFCfg cfg = buildIteratedDFCfg(*func, kArch);
    ASSERT_NE(cfg.entry, nullptr);
    const std::string before = physicalIR();

    lift();
    ASSERT_GT(ssa.phiCount(), 0u);
    const SSADestructionResult result = replayLegacyColoring(*func, ssa);

    ASSERT_TRUE(result.ok()) << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(CanonicalSSADestructionTest, ANonIdentityColoringActuallyRewritesTheOperands) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 8, 4, 5);
    lift();

    // Shift every value by a constant. Ranges stay consecutive and each PHI's
    // inputs still land on its result, so no copies are needed; only the
    // register numbers change.
    constexpr unsigned kShift = 100;
    AllocationResult shifted(ssa);
    for (const SSAValue& value : ssa.values())
        shifted.assign(value.id,
                       RegKey{value.origin.type, value.origin.idx + kShift, value.origin.half});

    const SSADestructionResult result = destroyCanonicalSSA(*func, ssa, shifted);

    ASSERT_TRUE(result.ok()) << result.toString();
    const std::string after = physicalIR();
    EXPECT_TRUE(contains(after, "v[104:107] = \"st.ds_load_b128\"(v100)")) << after;
    EXPECT_TRUE(contains(after, "v108 = \"st.v_add_f32\"(v104, v105)")) << after;
}

TEST_F(CanonicalSSADestructionTest, RejectsARangeSplitAcrossNonConsecutiveRegisters) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, 4, 0);
    lift();
    const std::string before = physicalIR();

    // Scatter the four DWORDs of the load, which no range operand can encode.
    AllocationResult scattered(ssa);
    unsigned next = 20;
    for (const SSAValue& value : ssa.values()) {
        scattered.assign(value.id, RegKey{RegType::V, next, RegHalf::NONE});
        next += 7;
    }

    const SSADestructionResult result = destroyCanonicalSSA(*func, ssa, scattered);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "must be consecutive in operand order"))
        << result.toString();
    // Rejection is atomic: the function keeps its original registers.
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(CanonicalSSADestructionTest, RejectsAnUnassignedValue) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();
    const std::string before = physicalIR();

    AllocationResult partial(ssa);
    for (const SSAValue& value : ssa.values()) {
        if (value.id == 1) continue;  // leave the first live-in uncoloured
        partial.assign(value.id, value.origin);
    }

    const SSADestructionResult result = destroyCanonicalSSA(*func, ssa, partial);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "%1 has no physical register")) << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(CanonicalSSADestructionTest, RejectsAPhiThatWouldNeedACopy) {
    SelfLoopJoinCfg cfg = buildSelfLoopJoinCfg(*func, kArch);
    ASSERT_NE(cfg.entry, nullptr);
    lift();
    const std::string before = physicalIR();

    ASSERT_GT(ssa.phiCount(), 0u);

    // Colour one PHI input somewhere other than the result: lowering that needs
    // a copy on the incoming edge, which is not implemented.
    AllocationResult colouring = createLegacyColoring(ssa);
    const SSAValueID moved = ssa.phis().front().incoming.front().value;
    colouring.assign(moved, RegKey{RegType::V, 200, RegHalf::NONE});

    const SSADestructionResult result = destroyCanonicalSSA(*func, ssa, colouring);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "needs a copy on the incoming edge"))
        << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(CanonicalSSADestructionTest, RejectsAGraphThatNoLongerDescribesTheFunction) {
    BasicBlock* entry = makeEntry();
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();

    // Rewriting an operand behind the graph's back is the mistake the shape
    // fingerprint exists to catch: every binding still looks self-consistent.
    add->setSrcReg(0, StinkyRegister("v", 9, 1));
    const std::string before = physicalIR();

    const SSADestructionResult result = replayLegacyColoring(*func, ssa);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "the function changed after it was lifted"))
        << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(CanonicalSSADestructionTest, RejectsAnAllocationComputedAgainstAnotherGraph) {
    BasicBlock* entry = makeEntry();
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();
    const AllocationResult stale = createLegacyColoring(ssa);

    // Change the program and lift again. The attached graph now matches the
    // function, so only the allocation is out of date.
    add->setSrcReg(0, StinkyRegister("v", 9, 1));
    lift();
    const std::string before = physicalIR();

    const SSADestructionResult result = destroyCanonicalSSA(*func, ssa, stale);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(contains(result.toString(), "computed against a different graph"))
        << result.toString();
    EXPECT_EQ(physicalIR(), before);
}

TEST_F(CanonicalSSADestructionTest, PassReportsAMissingGraph) {
    // Deciding whether a graph exists belongs to the pass: the lowering core is
    // handed one explicitly and has nothing to report about its absence.
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);

    PassContext passCtx;
    passCtx.setRemarksEnabled(true);
    AnalysisManager am;
    registerAllAnalyses(am);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createReplayLegacyColoringPass()->run(*func, passCtx, am);
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "missed: ReplayLegacyColoring")) << text;
    EXPECT_TRUE(contains(text, "no canonical SSA attached")) << text;
}

TEST_F(CanonicalSSADestructionTest, PassRoundTripsThroughThePassManager) {
    BasicBlock* entry = makeEntry();
    createDsReadB128InBlock(entry, kArch, 4, 0);
    createVAddInBlock(entry, kArch, 8, 4, 5);
    const std::string before = physicalIR();

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.addPass(createLiftAsmRegistersToSSAPass());
    pm.addPass(createReplayLegacyColoringPass());
    pm.run(*func);

    EXPECT_EQ(physicalIR(), before);
    // The graph described pre-rewrite operands, so lowering must not leave it
    // cached. Declining to preserve it is what evicts it.
    EXPECT_EQ(pm.getAnalysisManager().getCachedResult<CanonicalSSAAnalysis>(), nullptr);
}

TEST_F(CanonicalSSADestructionTest, PassIsANoOpWithoutAGraph) {
    BasicBlock* entry = makeEntry();
    createVAddInBlock(entry, kArch, 2, 0, 1);
    const std::string before = physicalIR();

    PassContext passCtx;
    AnalysisManager am;
    registerAllAnalyses(am);
    createReplayLegacyColoringPass()->run(*func, passCtx, am);

    EXPECT_EQ(physicalIR(), before);
}
