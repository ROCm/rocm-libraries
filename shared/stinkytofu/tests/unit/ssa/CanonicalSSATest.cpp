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

#include <type_traits>
#include <utility>

#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

using namespace stinkytofu;

static_assert(!std::is_copy_constructible_v<CanonicalSSA>);
static_assert(!std::is_copy_assignable_v<CanonicalSSA>);
static_assert(std::is_move_constructible_v<CanonicalSSA>);
static_assert(std::is_move_assignable_v<CanonicalSSA>);

namespace {

StinkyInstruction* createVAdd(BasicBlock& block) {
    AsmIRBuilder builder(block, GfxArchID::Gfx1250);
    return builder.create(getMCIDByUOp(GFX::v_add_f32, GfxArchID::Gfx1250));
}

}  // namespace

TEST(CanonicalSSATest, EmptyGraphHasNoValuesOrPhis) {
    CanonicalSSA ssa;

    EXPECT_TRUE(ssa.empty());
    EXPECT_EQ(ssa.valueCount(), 0u);
    EXPECT_EQ(ssa.phiCount(), 0u);
    EXPECT_FALSE(ssa.containsValue(kInvalidSSAValueID));
    EXPECT_FALSE(ssa.containsPhi(kInvalidSSAPhiID));
}

TEST(CanonicalSSATest, BuilderAssignsDenseValueIDsAndPreservesData) {
    CanonicalSSABuilder builder;

    SSAValue liveIn;
    liveIn.kind = SSAValueKind::LiveIn;
    liveIn.origin = {RegType::V, 8, RegHalf::NONE};
    SSAValueID liveInID = builder.addValue(std::move(liveIn));

    SSAValue definition;
    definition.kind = SSAValueKind::InstructionDef;
    definition.origin = {RegType::V, 8, RegHalf::NONE};
    SSAValueID definitionID = builder.addValue(std::move(definition));

    ASSERT_EQ(liveInID, 1u);
    ASSERT_EQ(definitionID, 2u);

    builder.value(liveInID).uses.push_back(SSAUse{nullptr, 1, 2, kInvalidSSAPhiID, nullptr});

    CanonicalSSA ssa = builder.take();

    EXPECT_FALSE(ssa.empty());
    EXPECT_EQ(ssa.valueCount(), 2u);
    EXPECT_TRUE(ssa.containsValue(liveInID));
    EXPECT_TRUE(ssa.containsValue(definitionID));
    EXPECT_EQ(ssa.value(liveInID).kind, SSAValueKind::LiveIn);
    EXPECT_EQ(ssa.value(definitionID).kind, SSAValueKind::InstructionDef);
    EXPECT_EQ(ssa.value(definitionID).origin.type, RegType::V);
    EXPECT_EQ(ssa.value(definitionID).origin.idx, 8u);
    EXPECT_EQ(ssa.value(definitionID).origin.half, RegHalf::NONE);
    ASSERT_EQ(ssa.value(liveInID).uses.size(), 1u);
    EXPECT_FALSE(ssa.value(liveInID).uses.front().isPhiUse());

    CanonicalSSA secondTake = builder.take();
    EXPECT_TRUE(secondTake.empty());
}

TEST(CanonicalSSATest, StoresInstructionOperandBindings) {
    Function function("instruction_bindings");
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* instruction = createVAdd(*block);

    CanonicalSSABuilder builder;
    SSAValue source;
    source.kind = SSAValueKind::LiveIn;
    source.origin = {RegType::V, 1, RegHalf::NONE};
    SSAValueID sourceID = builder.addValue(std::move(source));

    SSAValue destination;
    destination.kind = SSAValueKind::InstructionDef;
    destination.origin = {RegType::V, 2, RegHalf::NONE};
    destination.definingInstruction = instruction;
    SSAValueID destinationID = builder.addValue(std::move(destination));

    SSAInstructionInfo info;
    info.sources = {{{sourceID}}, {{sourceID}}};
    info.destinations = {{{destinationID}}};
    builder.setInstructionInfo(*instruction, std::move(info));

    CanonicalSSA ssa = builder.take();
    const SSAInstructionInfo* stored = ssa.findInstructionInfo(*instruction);

    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->sources.size(), 2u);
    ASSERT_EQ(stored->destinations.size(), 1u);
    EXPECT_EQ(stored->sources[0].units, (std::vector<SSAValueID>{sourceID}));
    EXPECT_EQ(stored->sources[1].units, (std::vector<SSAValueID>{sourceID}));
    EXPECT_EQ(stored->destinations[0].units, (std::vector<SSAValueID>{destinationID}));
}

TEST(CanonicalSSATest, StoresPhisByDenseIDAndOwningBlock) {
    Function function("phi_storage");
    BasicBlock* left = function.createBasicBlock("left");
    BasicBlock* right = function.createBasicBlock("right");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(left, join);
    function.addEdge(right, join);

    CanonicalSSABuilder builder;
    SSAValue leftValue;
    leftValue.kind = SSAValueKind::InstructionDef;
    leftValue.origin = {RegType::V, 5, RegHalf::NONE};
    SSAValueID leftID = builder.addValue(std::move(leftValue));

    SSAValue rightValue;
    rightValue.kind = SSAValueKind::InstructionDef;
    rightValue.origin = {RegType::V, 5, RegHalf::NONE};
    SSAValueID rightID = builder.addValue(std::move(rightValue));

    SSAValue phiResult;
    phiResult.kind = SSAValueKind::Phi;
    phiResult.origin = {RegType::V, 5, RegHalf::NONE};
    SSAValueID resultID = builder.addValue(std::move(phiResult));

    SSAPhi phi;
    phi.block = join;
    phi.origin = {RegType::V, 5, RegHalf::NONE};
    phi.result = resultID;
    phi.incoming = {{left, leftID}, {right, rightID}};
    SSAPhiID phiID = builder.addPhi(std::move(phi));
    builder.value(resultID).definingPhi = phiID;
    builder.addPhiToBlock(*join, phiID);

    CanonicalSSA ssa = builder.take();

    ASSERT_EQ(phiID, 1u);
    ASSERT_TRUE(ssa.containsPhi(phiID));
    EXPECT_EQ(ssa.phiCount(), 1u);
    EXPECT_EQ(ssa.phi(phiID).block, join);
    EXPECT_EQ(ssa.phi(phiID).result, resultID);
    ASSERT_EQ(ssa.phi(phiID).incoming.size(), 2u);
    EXPECT_EQ(ssa.phi(phiID).incoming[0].predecessor, left);
    EXPECT_EQ(ssa.phi(phiID).incoming[0].value, leftID);
    EXPECT_EQ(ssa.phi(phiID).incoming[1].predecessor, right);
    EXPECT_EQ(ssa.phi(phiID).incoming[1].value, rightID);
    EXPECT_EQ(ssa.value(resultID).definingPhi, phiID);
    EXPECT_EQ(ssa.phisForBlock(*join), (std::vector<SSAPhiID>{phiID}));
    EXPECT_TRUE(ssa.phisForBlock(*left).empty());
}

TEST(CanonicalSSATest, DistinguishesInstructionAndPhiUses) {
    Function function("uses");
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* instruction = createVAdd(*block);

    SSAUse instructionUse{instruction, 2, 1, kInvalidSSAPhiID, nullptr};
    SSAUse phiUse{nullptr, 0, 0, 3, block};

    EXPECT_FALSE(instructionUse.isPhiUse());
    EXPECT_EQ(instructionUse.instruction, instruction);
    EXPECT_EQ(instructionUse.operand, 2u);
    EXPECT_EQ(instructionUse.unit, 1u);

    EXPECT_TRUE(phiUse.isPhiUse());
    EXPECT_EQ(phiUse.phi, 3u);
    EXPECT_EQ(phiUse.predecessor, block);
}
