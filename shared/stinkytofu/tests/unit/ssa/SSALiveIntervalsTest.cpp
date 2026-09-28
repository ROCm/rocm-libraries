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
#include <vector>

#include "AttachedSSATestUtils.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervals.hpp"
#include "stinkytofu/analysis/asm/ssa/SSALiveIntervalsAnalysis.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

class SSALiveIntervalsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    void lift() {
        Expected<LiftAttachedSSAResult> result = liftAsmRegistersToAttachedSSA(*func);
        ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
    }

    SSALiveIntervals intervals() {
        return computeSSALiveIntervals(*func);
    }

    SSAValueID idOf(const StinkySSAValue* value) const {
        return value == nullptr ? kInvalidSSAValueID : value->valueId();
    }

    /// Every pair of values lifted from the same physical register must have
    /// disjoint ranges, otherwise createLegacyColoring() would be illegal.
    void expectLegacyColouringIsConflictFree(const SSALiveIntervals& live) {
        const SSAArena& arena = func->ssaArena();
        for (StinkySSAValue* a : arena.values()) {
            for (StinkySSAValue* b : arena.values()) {
                if (a == nullptr || b == nullptr || a->valueId() >= b->valueId()) continue;
                if (bindingKeyOf(a) != bindingKeyOf(b)) continue;
                EXPECT_FALSE(live.overlap(a->valueId(), b->valueId()))
                    << "%" << a->valueId() << " and %" << b->valueId() << " both bound to "
                    << regKeyToString(bindingKeyOf(a)) << " but overlap\n"
                    << live.toString();
            }
        }
    }

    std::unique_ptr<Function> func;
};

std::vector<LiveSegment> segmentsOf(const SSALiveIntervals& live, SSAValueID id) {
    const std::span<const LiveSegment> segments = live.rangeOf(id).segments();
    return {segments.begin(), segments.end()};
}

}  // namespace

TEST_F(SSALiveIntervalsTest, FunctionWithoutAttachedSSAYieldsNothing) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kArch, 2, 0, 1);

    const SSALiveIntervals live = intervals();

    EXPECT_TRUE(live.empty());
    EXPECT_EQ(live.valueCount(), 0u);
    EXPECT_TRUE(live.rangeOf(1).empty());
}

TEST_F(SSALiveIntervalsTest, SlotIndexesPairUseWithDefAndTileBlocks) {
    BasicBlock* entry = block("entry");
    BasicBlock* next = block("next");
    func->addEdge(entry, next);
    StinkyInstruction* first = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* second = createVAddInBlock(next, kArch, 3, 2, 1);

    const SSASlotIndexes slots = computeSSASlotIndexes(*func);

    // Operands are read one point before results are written.
    EXPECT_EQ(slots.defSlot(first), slots.useSlot(first) + 1);
    EXPECT_EQ(slots.defSlot(second), slots.useSlot(second) + 1);

    // Block arguments are defined just after the block's first point.
    EXPECT_EQ(slots.blockArgDef(entry), slots.blockStart(entry) + 1);

    // Blocks tile the index space in layout order, with no gap between them.
    EXPECT_EQ(slots.blockStart(entry), 0u);
    EXPECT_EQ(slots.blockEnd(entry), slots.blockStart(next));
    EXPECT_EQ(slots.blockEnd(next), slots.slotCount());
    EXPECT_EQ(slots.slotCount(), 8u);  // two blocks, one instruction each
}

TEST_F(SSALiveIntervalsTest, StraightLineRangesRunFromDefinitionToLastUse) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);
    lift();

    const SSALiveIntervals live = intervals();
    const SSASlotIndexes& slots = live.slots();
    const SSAValueID result = idOf(ssaDefinedValue(*add));
    const SSAValueID liveIn = idOf(ssaSourceValue(*add, 0));

    // v0 arrives as a block argument and dies at the add that reads it.
    const std::vector<LiveSegment> expectedLiveIn{
        {slots.blockArgDef(entry), slots.useSlot(add) + 1}};
    EXPECT_EQ(segmentsOf(live, liveIn), expectedLiveIn);

    // The result is a dead definition: one point wide.
    const std::vector<LiveSegment> expectedResult{{slots.defSlot(add), slots.defSlot(add) + 1}};
    EXPECT_EQ(segmentsOf(live, result), expectedResult);

    // Both live-ins are simultaneously live, the result is not.
    EXPECT_EQ(live.peakPressure(RegType::V), 2u);
    EXPECT_FALSE(live.overlap(liveIn, result));
}

TEST_F(SSALiveIntervalsTest, ReadModifyWriteValuesDoNotOverlap) {
    // v2 = v_add_f32 v2, v1 reads the old v2 and defines a new one. The two
    // values share a physical register in the input program, so their ranges
    // must not overlap or no allocator could reproduce that.
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/2, /*src1=*/1);
    lift();

    const SSALiveIntervals live = intervals();
    const SSAValueID oldValue = idOf(ssaSourceValue(*add, 0));
    const SSAValueID newValue = idOf(ssaDefinedValue(*add));

    ASSERT_NE(oldValue, newValue);
    EXPECT_EQ(bindingKeyOf(func->ssaArena().get(oldValue)),
              bindingKeyOf(func->ssaArena().get(newValue)));
    EXPECT_FALSE(live.overlap(oldValue, newValue));
    expectLegacyColouringIsConflictFree(live);
}

TEST_F(SSALiveIntervalsTest, DestinationMayReuseASourceThatDiesAtTheSameInstruction) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);
    lift();

    const SSALiveIntervals live = intervals();

    // v0's last read is this instruction, so the result may take its register.
    EXPECT_FALSE(live.overlap(idOf(ssaSourceValue(*add, 0)), idOf(ssaDefinedValue(*add))));
}

TEST_F(SSALiveIntervalsTest, MergeArgumentDoesNotOverlapItsIncomingValues) {
    BasicBlock* entry = block("entry");
    BasicBlock* left = block("left");
    BasicBlock* right = block("right");
    BasicBlock* join = block("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkyInstruction* leftDef = createVAddInBlock(left, kArch, 5, 20, 21);
    StinkyInstruction* rightDef = createVAddInBlock(right, kArch, 5, 22, 23);
    createVAddInBlock(join, kArch, 6, 5, 5);
    lift();

    const SSALiveIntervals live = intervals();
    const SSASlotIndexes& slots = live.slots();
    const SSABlockArgument* arg = vgprArgumentFor(*join, 5);
    ASSERT_NE(arg, nullptr);

    const SSAValueID merged = idOf(arg->value);
    const SSAValueID fromLeft = idOf(ssaDefinedValue(*leftDef));
    const SSAValueID fromRight = idOf(ssaDefinedValue(*rightDef));

    // An incoming value is consumed on its own edge, so it stays live to the
    // end of that predecessor and stops there.
    EXPECT_EQ(live.rangeOf(fromLeft).end(), slots.blockEnd(left));
    EXPECT_EQ(live.rangeOf(fromRight).end(), slots.blockEnd(right));

    // The merge is defined at the join, so it never coexists with its inputs
    // and all three can share one register, which is what legacy replay does.
    EXPECT_EQ(live.rangeOf(merged).start(), slots.blockArgDef(join));
    EXPECT_FALSE(live.overlap(merged, fromLeft));
    EXPECT_FALSE(live.overlap(merged, fromRight));
    EXPECT_FALSE(live.overlap(fromLeft, fromRight));
    expectLegacyColouringIsConflictFree(live);
}

TEST_F(SSALiveIntervalsTest, ValueDeadOnOneArmLeavesAHole) {
    // Layout puts the killing arm between the definition and the surviving arm,
    // so the range needs two segments. A single span would claim a register
    // across a block where the value is not live.
    BasicBlock* entry = block("entry");
    BasicBlock* kills = block("kills");
    BasicBlock* uses = block("uses");
    BasicBlock* join = block("join");
    func->addEdge(entry, kills);
    func->addEdge(entry, uses);
    func->addEdge(kills, join);
    func->addEdge(uses, join);

    StinkyInstruction* def = createVAddInBlock(entry, kArch, 5, 20, 21);
    createVAddInBlock(kills, kArch, 5, 22, 23);  // redefines v5, so the entry value dies
    createVAddInBlock(uses, kArch, 6, 5, 5);     // reads the entry value
    createVAddInBlock(join, kArch, 7, 5, 5);
    lift();

    const SSALiveIntervals live = intervals();
    const SSASlotIndexes& slots = live.slots();
    const SSAValueID entryValue = idOf(ssaDefinedValue(*def));

    ASSERT_EQ(segmentsOf(live, entryValue).size(), 2u) << live.toString();
    EXPECT_FALSE(live.rangeOf(entryValue).covers(slots.blockStart(kills)));
    EXPECT_TRUE(live.rangeOf(entryValue).covers(slots.blockStart(uses)));
    expectLegacyColouringIsConflictFree(live);
}

TEST_F(SSALiveIntervalsTest, LoopCarriedValueIsLiveToTheEndOfTheLatch) {
    BasicBlock* entry = block("entry");
    BasicBlock* header = block("header");
    BasicBlock* body = block("body");
    BasicBlock* exit = block("exit");
    func->addEdge(entry, header);
    func->addEdge(header, body);
    func->addEdge(body, header);
    func->addEdge(header, exit);

    createVAddInBlock(entry, kArch, 5, 20, 21);
    StinkyInstruction* carried =
        createVAddInBlock(body, kArch, /*dest=*/5, /*src0=*/5, /*src1=*/22);
    createVAddInBlock(exit, kArch, 6, 5, 5);
    lift();

    const SSALiveIntervals live = intervals();
    const SSASlotIndexes& slots = live.slots();
    const SSABlockArgument* arg = vgprArgumentFor(*header, 5);
    ASSERT_NE(arg, nullptr);

    const SSAValueID headerValue = idOf(arg->value);
    const SSAValueID latchValue = idOf(ssaDefinedValue(*carried));

    // The value flowing round the back edge lives to the end of the latch.
    EXPECT_EQ(live.rangeOf(latchValue).end(), slots.blockEnd(body));

    // The header argument is live from the top of the loop and is still live in
    // the exit, which the latch is not, so the two never coexist.
    EXPECT_EQ(live.rangeOf(headerValue).start(), slots.blockArgDef(header));
    EXPECT_TRUE(live.rangeOf(headerValue).covers(slots.blockStart(exit)));
    EXPECT_FALSE(live.overlap(headerValue, latchValue));
    expectLegacyColouringIsConflictFree(live);
}

TEST_F(SSALiveIntervalsTest, EachDwordOfARangeOperandGetsItsOwnInterval) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, /*destReg=*/10, /*addrReg=*/4);
    createVAddInBlock(entry, kArch, 20, 10, 13);
    lift();

    const SSALiveIntervals live = intervals();
    const std::vector<StinkySSAValue*> units = ssaDestUnits(*load, 0);

    ASSERT_EQ(units.size(), 4u);
    for (StinkySSAValue* unit : units) {
        EXPECT_FALSE(live.rangeOf(idOf(unit)).empty()) << "%" << idOf(unit) << " has no range";
    }
    // v4 address plus four loaded DWORDs are live together after the load.
    EXPECT_GE(live.peakPressure(RegType::V), 4u);
    expectLegacyColouringIsConflictFree(live);
}

TEST_F(SSALiveIntervalsTest, ScalarAndVectorPressureAreCountedSeparately) {
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));
    lift();

    const SSALiveIntervals live = intervals();

    EXPECT_EQ(live.peakPressure(RegType::S), 1u);
    EXPECT_EQ(live.peakPressure(RegType::V), 1u);
}

TEST_F(SSALiveIntervalsTest, AnalysisManagerCachesTheResult) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kArch, 2, 0, 1);
    lift();

    AnalysisManager am;
    registerAllAnalyses(am);

    const SSALiveIntervals& first = am.getResult<SSALiveIntervalsAnalysis>(*func);
    const SSALiveIntervals& second = am.getResult<SSALiveIntervalsAnalysis>(*func);

    EXPECT_EQ(&first, &second);
    EXPECT_FALSE(first.empty());
    EXPECT_EQ(first.valueCount(), func->ssaArena().valueCount());
}
