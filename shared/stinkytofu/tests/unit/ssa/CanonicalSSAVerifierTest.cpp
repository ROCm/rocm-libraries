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
#include <string>
#include <utility>

#include "CanonicalSSATestUtils.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/controlflow/Dominance.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

/// Returns true when any diagnostic contains \p needle.
bool hasError(const CanonicalSSAVerificationResult& result, const std::string& needle) {
    for (const std::string& error : result.errors) {
        if (error.find(needle) != std::string::npos) return true;
    }
    return false;
}

class CanonicalSSAVerifierTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
    }

    /// v2 = v_add_f32 v0, v1 with v0/v1 as live-ins.
    void buildStraightLine() {
        add = createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);
        liveIn0 = addLiveIn(builder, 0);
        liveIn1 = addLiveIn(builder, 1);
        defined = bindInstruction(builder, *add, {{liveIn0}, {liveIn1}});
    }

    CanonicalSSAVerificationResult verify() {
        ssa = builder.take();
        return verifyCanonicalSSA(*func, ssa);
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
    StinkyInstruction* add = nullptr;
    CanonicalSSABuilder builder;
    CanonicalSSA ssa;
    SSAValueID liveIn0 = kInvalidSSAValueID;
    SSAValueID liveIn1 = kInvalidSSAValueID;
    std::vector<SSAValueID> defined;
};

}  // namespace

TEST_F(CanonicalSSAVerifierTest, EmptyGraphOnEmptyFunctionIsValid) {
    EXPECT_TRUE(verify().ok());
}

TEST_F(CanonicalSSAVerifierTest, StraightLineGraphIsValid) {
    buildStraightLine();
    const CanonicalSSAVerificationResult result = verify();
    EXPECT_TRUE(result.ok()) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, MultiDwordRangeAndPartialDefinitionAreValid) {
    // v[4:7] = ds_load_b128 v0 defines four units; the store then reads two of them.
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    const SSAValueID addr = addLiveIn(builder, 0);
    const std::vector<SSAValueID> loaded = bindInstruction(builder, *load, {{addr}});
    ASSERT_EQ(loaded.size(), 4u);

    StinkyInstruction* store = createDSWriteInBlock(entry, kArch, /*addr=*/0, /*data=*/4);
    bindInstruction(builder, *store, {{addr}, {loaded[0], loaded[1]}});

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_TRUE(result.ok()) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RepeatedDefinitionsOfOnePhysicalRegisterAreValid) {
    StinkyInstruction* first = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* second = createVAddInBlock(entry, kArch, 2, 2, 1);

    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    const std::vector<SSAValueID> firstDef = bindInstruction(builder, *first, {{in0}, {in1}});
    const std::vector<SSAValueID> secondDef =
        bindInstruction(builder, *second, {{firstDef[0]}, {in1}});

    EXPECT_NE(firstDef[0], secondDef[0]);
    const CanonicalSSAVerificationResult result = verify();
    EXPECT_TRUE(result.ok()) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, SameValueUsedTwiceByOneInstructionIsValid) {
    StinkyInstruction* self = createVAddInBlock(entry, kArch, 2, 0, 0);
    const SSAValueID in0 = addLiveIn(builder, 0);
    bindInstruction(builder, *self, {{in0}, {in0}});

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_TRUE(result.ok()) << result.toString();
    EXPECT_EQ(ssa.value(in0).uses.size(), 2u);
}

TEST_F(CanonicalSSAVerifierTest, DiamondPhiIsValid) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkyInstruction* leftDef = createVAddInBlock(left, kArch, 5, 0, 1);
    StinkyInstruction* rightDef = createVAddInBlock(right, kArch, 5, 1, 0);
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    const SSAValueID leftValue = bindInstruction(builder, *leftDef, {{in0}, {in1}}).front();
    const SSAValueID rightValue = bindInstruction(builder, *rightDef, {{in1}, {in0}}).front();

    const SSAValueID merged = addPhi(builder, *join, vgprKey(5), {leftValue, rightValue});
    StinkyInstruction* consumer = createVAddInBlock(join, kArch, 6, 5, 5);
    bindInstruction(builder, *consumer, {{merged}, {merged}});

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_TRUE(result.ok()) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsUseBeforeDefinitionInSameBlock) {
    // Bind the consumer first so its source is defined by a later instruction.
    StinkyInstruction* consumer = createVAddInBlock(entry, kArch, 3, 2, 1);
    StinkyInstruction* producer = createVAddInBlock(entry, kArch, 2, 0, 1);

    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    const SSAValueID produced = bindInstruction(builder, *producer, {{in0}, {in1}}).front();
    bindInstruction(builder, *consumer, {{produced}, {in1}});

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "used earlier or at the same position")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsUseNotDominatedByItsDefinition) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    func->addEdge(entry, left);
    func->addEdge(entry, right);

    // Defined on one arm of a branch, used on the other.
    StinkyInstruction* def = createVAddInBlock(left, kArch, 5, 20, 21);
    StinkyInstruction* use = createVAddInBlock(right, kArch, 6, 5, 5);
    const SSAValueID in20 = addLiveIn(builder, 20);
    const SSAValueID in21 = addLiveIn(builder, 21);
    const SSAValueID defined = bindInstruction(builder, *def, {{in20}, {in21}}).front();
    bindInstruction(builder, *use, {{defined}, {defined}});

    ssa = builder.take();
    // Without dominance info only same-block ordering is checked.
    EXPECT_TRUE(verifyCanonicalSSA(*func, ssa).ok());

    const DominanceInfo dominance = computeDominanceInfo(*func);
    const CanonicalSSAVerificationResult result = verifyCanonicalSSA(*func, ssa, dominance);
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "^left, which does not dominate ^right")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsPhiInputThatDoesNotDominateItsEdge) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkyInstruction* leftDef = createVAddInBlock(left, kArch, 5, 20, 21);
    StinkyInstruction* rightDef = createVAddInBlock(right, kArch, 5, 22, 23);
    const SSAValueID in20 = addLiveIn(builder, 20);
    const SSAValueID in21 = addLiveIn(builder, 21);
    const SSAValueID in22 = addLiveIn(builder, 22);
    const SSAValueID in23 = addLiveIn(builder, 23);
    const SSAValueID leftValue = bindInstruction(builder, *leftDef, {{in20}, {in21}}).front();
    const SSAValueID rightValue = bindInstruction(builder, *rightDef, {{in22}, {in23}}).front();

    // Swap the arms: each edge carries the value defined on the other side.
    addPhi(builder, *join, vgprKey(5), {rightValue, leftValue});

    ssa = builder.take();
    const DominanceInfo dominance = computeDominanceInfo(*func);
    const CanonicalSSAVerificationResult result = verifyCanonicalSSA(*func, ssa, dominance);
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "does not dominate predecessor ^left")) << result.toString();
    EXPECT_TRUE(hasError(result, "does not dominate predecessor ^right")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, AcceptsPhiInputsThatDominateTheirEdges) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkyInstruction* leftDef = createVAddInBlock(left, kArch, 5, 20, 21);
    StinkyInstruction* rightDef = createVAddInBlock(right, kArch, 5, 22, 23);
    const SSAValueID in20 = addLiveIn(builder, 20);
    const SSAValueID in21 = addLiveIn(builder, 21);
    const SSAValueID in22 = addLiveIn(builder, 22);
    const SSAValueID in23 = addLiveIn(builder, 23);
    const SSAValueID leftValue = bindInstruction(builder, *leftDef, {{in20}, {in21}}).front();
    const SSAValueID rightValue = bindInstruction(builder, *rightDef, {{in22}, {in23}}).front();
    addPhi(builder, *join, vgprKey(5), {leftValue, rightValue});

    ssa = builder.take();
    const DominanceInfo dominance = computeDominanceInfo(*func);
    const CanonicalSSAVerificationResult result = verifyCanonicalSSA(*func, ssa, dominance);
    EXPECT_TRUE(result.ok()) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsTwoValuesDefiningOneSlot) {
    buildStraightLine();

    SSAValue duplicate;
    duplicate.kind = SSAValueKind::InstructionDef;
    duplicate.origin = vgprKey(2);
    duplicate.definingInstruction = add;
    duplicate.definingOperand = 0;
    duplicate.definingUnit = 0;
    builder.addValue(std::move(duplicate));

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "both define")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsOriginThatDisagreesWithTheOperand) {
    StinkyInstruction* instruction = createVAddInBlock(entry, kArch, 2, 0, 1);
    // Claim v9 as the origin of a value bound to the v0 source operand.
    const SSAValueID wrongOrigin = addLiveIn(builder, 9);
    const SSAValueID in1 = addLiveIn(builder, 1);
    bindInstruction(builder, *instruction, {{wrongOrigin}, {in1}});

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "but the operand unit is v0")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsNonAllocatableOrigin) {
    SSAValue special;
    special.kind = SSAValueKind::LiveIn;
    special.origin = RegKey{RegType::SCC, 0, RegHalf::NONE};
    builder.addValue(std::move(special));

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "non-allocatable origin")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsWrongOperandUnitCount) {
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    const SSAValueID addr = addLiveIn(builder, 0);

    // Bind only two of the four destination DWORDs.
    SSAInstructionInfo info;
    info.sources.resize(1);
    info.sources[0].units = {addr};
    builder.value(addr).uses.push_back(SSAUse{load, 0, 0, kInvalidSSAPhiID, nullptr});
    info.destinations.resize(1);
    for (uint32_t unit = 0; unit < 2; ++unit) {
        SSAValue value;
        value.kind = SSAValueKind::InstructionDef;
        value.origin = vgprKey(4 + unit);
        value.definingInstruction = load;
        value.definingUnit = unit;
        info.destinations[0].units.push_back(builder.addValue(std::move(value)));
    }
    builder.setInstructionInfo(*load, std::move(info));

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "expects 4 SSA unit(s) but binds 2")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsMissingUseRecord) {
    StinkyInstruction* instruction = createVAddInBlock(entry, kArch, 2, 0, 1);
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    bindInstruction(builder, *instruction, {{in0}, {in1}});

    // Drop the use record that mirrors src0.
    builder.value(in0).uses.clear();

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "records this slot 0 time(s)")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsDuplicatedUseRecord) {
    buildStraightLine();
    builder.value(liveIn0).uses.push_back(builder.value(liveIn0).uses.front());

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "records this slot 2 time(s)")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsInstructionWithAllocatableOperandsAndNoBindings) {
    createVAddInBlock(entry, kArch, 2, 0, 1);

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "no SSA operand bindings")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsPhiWithWrongPredecessorCount) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, join);
    func->addEdge(left, join);

    const SSAValueID in0 = addLiveIn(builder, 5);
    addPhi(builder, *join, vgprKey(5), {in0});  // one incoming, two predecessors

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "incoming value(s) for 2 predecessor(s)")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsPhiIncomingOrderThatIgnoresPredecessorOrder) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(left, join);
    func->addEdge(right, join);

    const SSAValueID leftValue = addLiveIn(builder, 5);
    const SSAValueID rightValue = addLiveIn(builder, 5);
    const SSAValueID merged = addPhi(builder, *join, vgprKey(5), {leftValue, rightValue});
    ASSERT_NE(merged, kInvalidSSAValueID);

    // Swap the edges so incoming order no longer matches getPredecessors().
    SSAPhi& phi = builder.phi(1);
    std::swap(phi.incoming[0].predecessor, phi.incoming[1].predecessor);
    builder.value(leftValue).uses[0].predecessor = phi.incoming[0].predecessor;
    builder.value(rightValue).uses[0].predecessor = phi.incoming[1].predecessor;

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "but predecessor 0 of ^join is")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsPhiIncomingWithMismatchedOrigin) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(left, join);
    func->addEdge(right, join);

    const SSAValueID leftValue = addLiveIn(builder, 5);
    const SSAValueID rightValue = addLiveIn(builder, 6);
    addPhi(builder, *join, vgprKey(5), {leftValue, rightValue});

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "with origin v6")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsPhiMissingFromTheBlockIndex) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(left, join);

    const SSAValueID incoming = addLiveIn(builder, 5);
    const SSAValueID result0 = addLiveIn(builder, 5);
    ASSERT_NE(result0, kInvalidSSAValueID);

    SSAValue phiResult;
    phiResult.kind = SSAValueKind::Phi;
    phiResult.origin = vgprKey(5);
    const SSAValueID resultID = builder.addValue(std::move(phiResult));

    SSAPhi phi;
    phi.block = join;
    phi.origin = vgprKey(5);
    phi.result = resultID;
    phi.incoming.push_back(SSAPhiIncoming{left, incoming});
    const SSAPhiID phiID = builder.addPhi(std::move(phi));
    builder.value(resultID).definingPhi = phiID;
    builder.value(incoming).uses.push_back(SSAUse{nullptr, 0, 0, phiID, left});
    // Deliberately skip addPhiToBlock().

    const CanonicalSSAVerificationResult verification = verify();
    EXPECT_FALSE(verification.ok());
    EXPECT_TRUE(hasError(verification, "appears 0 time(s) in the block phi index"))
        << verification.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsDenseIdViolation) {
    SSAValue value;
    value.kind = SSAValueKind::LiveIn;
    value.origin = vgprKey(0);
    const SSAValueID id = builder.addValue(std::move(value));
    builder.value(id).id = 7;

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "expected %1")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, AcceptsAnEmptyGraphForAFunctionWithNoInstructions) {
    // Nothing to bind and nothing to check, but the graph is still well formed.
    const CanonicalSSAVerificationResult result = verify();
    EXPECT_TRUE(result.ok()) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, AcceptsAStampedGraphThatStillMatchesTheFunction) {
    buildStraightLine();
    builder.setShape(computeFunctionShape(*func));

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_TRUE(result.ok()) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, RejectsAStampedGraphAfterTheFunctionChanges) {
    buildStraightLine();
    builder.setShape(computeFunctionShape(*func));

    // Every binding still agrees with itself; only the program moved on.
    add->setSrcReg(0, StinkyRegister("v", 9, 1));

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(hasError(result, "the function changed after it was lifted")) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, HandBuiltGraphsAreExemptFromTheShapeCheck) {
    // A graph nobody stamped carries no claim about which program it describes,
    // so the check has nothing to compare and must not invent a failure. This is
    // what lets these tests build graphs by hand at all.
    buildStraightLine();

    const CanonicalSSAVerificationResult result = verify();
    EXPECT_EQ(ssa.shape(), kUnstampedShape);
    EXPECT_TRUE(result.ok()) << result.toString();
}

TEST_F(CanonicalSSAVerifierTest, DiagnosticsAreDeterministic) {
    createVAddInBlock(entry, kArch, 2, 0, 1);
    createVAddInBlock(entry, kArch, 3, 2, 1);

    ssa = builder.take();
    const CanonicalSSAVerificationResult first = verifyCanonicalSSA(*func, ssa);
    const CanonicalSSAVerificationResult second = verifyCanonicalSSA(*func, ssa);

    EXPECT_FALSE(first.ok());
    EXPECT_EQ(first.errors, second.errors);
    EXPECT_EQ(first.toString(), second.toString());
}
