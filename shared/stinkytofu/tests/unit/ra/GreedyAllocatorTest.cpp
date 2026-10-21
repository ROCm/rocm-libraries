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

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "stinkytofu/transforms/asm/ssa/SSADestruction.hpp"
#include "transforms/asm/ra/allocators/GreedyAllocator.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

std::string blockSection(const std::string& ir, const std::string& label) {
    const std::string marker = "^" + label + ":";
    const size_t start = ir.find(marker);
    if (start == std::string::npos) return {};
    const size_t next = ir.find("\n^", start + marker.size());
    if (next == std::string::npos) return ir.substr(start);
    return ir.substr(start, next - start);
}

class GreedyAllocatorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    /// Colours \p setup with greedy and requires a legal result.
    AllocationResult colour(AllocationSetup& setup) {
        GreedyAllocator allocator;
        Expected<AllocationResult> result = allocator.allocate(setup.context());
        EXPECT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
        if (!result.hasValue()) return AllocationResult{};

        const AllocationVerificationResult checked =
            verifyAllocation(*func, *result, setup.context());
        EXPECT_TRUE(checked.ok()) << checked.toString() << "\n" << result->toString();
        return std::move(*result);
    }

    std::string colourError(AllocationSetup& setup) {
        GreedyAllocator allocator;
        Expected<AllocationResult> result = allocator.allocate(setup.context());
        EXPECT_TRUE(result.hasError());
        return result.hasError() ? result.getError() : std::string{};
    }

    SSAValueID idOf(const StinkySSAValue* value) const {
        return value == nullptr ? kInvalidSSAValueID : value->valueId();
    }

    std::unique_ptr<Function> func;
};

/// mem[v<addr>] = v[<data>:<data>+3], so a 4-DWORD range is read as one operand.
StinkyInstruction* createDsStoreB128(BasicBlock* bb, int addrReg, int dataReg) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* store = builder.create(getMCIDByUOp(GFX::ds_store_b128, kRaTestArch));
    store->addSrcReg(StinkyRegister("v", addrReg, 1));
    store->addSrcReg(StinkyRegister("v", dataReg, 4));
    return store;
}

/// Withhold every VGPR except [0, keep), to put the colourer under real pressure.
void keepOnlyFirstVgprs(AllocationSetup& setup, uint32_t keep) {
    const uint32_t count = setup.target().indexCount(RegType::V);
    ASSERT_GT(count, keep);
    setup.target().reserve(RegType::V, keep, count - keep);
}

}  // namespace

TEST_F(GreedyAllocatorTest, HintIsHonouredSoASimpleFunctionMatchesLegacy) {
    // With room to spare every block lands on the register the producer chose, so
    // greedy reproduces the legacy assignment exactly. That is what makes a
    // shadow comparison meaningful: a difference means pressure, not churn.
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    EXPECT_EQ(colour(setup).toString(), createLegacyColoring(*func).toString());
}

TEST_F(GreedyAllocatorTest, TupleRunStaysConsecutive) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kRaTestArch, 10, 4);
    createVAddInBlock(entry, kRaTestArch, 20, 10, 13);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const AllocationResult result = colour(setup);

    const std::vector<StinkySSAValue*> units = ssaDestUnits(*load, 0);
    ASSERT_EQ(units.size(), 4u);
    const RegKey first = result.assignmentOf(idOf(units.front()));
    for (size_t unit = 0; unit < units.size(); ++unit) {
        const RegKey physical = result.assignmentOf(idOf(units[unit]));
        EXPECT_EQ(physical.type, first.type);
        EXPECT_EQ(physical.idx, first.idx + unit) << result.toString();
    }
}

TEST_F(GreedyAllocatorTest, OverlappingTupleRunsShareOneRegister) {
    // The case that forces tuple runs to be solved together rather than one at a
    // time: the 4-DWORD load, the 2-DWORD read of its first half, and the later
    // 4-DWORD read all constrain the same values. The narrow write to v4 must
    // therefore land on whatever register the original v4 got.
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createDSWriteInBlock(entry, kRaTestArch, /*addrReg=*/0, /*dataReg=*/4);
    StinkyInstruction* redefine = createVAddInBlock(entry, kRaTestArch, 4, 0, 0);
    createDsStoreB128(entry, /*addrReg=*/0, /*dataReg=*/4);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const AllocationResult result = colour(setup);

    const std::vector<StinkySSAValue*> loaded = ssaDestUnits(*load, 0);
    ASSERT_EQ(loaded.size(), 4u);
    const SSAValueID oldV4 = idOf(loaded.front());
    const SSAValueID newV4 = idOf(ssaDefinedValue(*redefine));
    ASSERT_NE(oldV4, newV4);

    EXPECT_EQ(result.assignmentOf(oldV4), result.assignmentOf(newV4)) << result.toString();
    // And the wide range is still one consecutive run around it.
    for (size_t unit = 1; unit < loaded.size(); ++unit) {
        EXPECT_EQ(result.assignmentOf(idOf(loaded[unit])).idx,
                  result.assignmentOf(oldV4).idx + unit)
            << result.toString();
    }
}

TEST_F(GreedyAllocatorTest, MergeAndItsIncomingValuesShareOneRegister) {
    BasicBlock* entry = block("entry");
    BasicBlock* left = block("left");
    BasicBlock* right = block("right");
    BasicBlock* join = block("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);
    StinkyInstruction* fromLeft = createVAddInBlock(left, kRaTestArch, 5, 20, 21);
    StinkyInstruction* fromRight = createVAddInBlock(right, kRaTestArch, 5, 22, 23);
    createVAddInBlock(join, kRaTestArch, 6, 5, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const AllocationResult result = colour(setup);

    const SSABlockArgument* arg = vgprArgumentFor(*join, 5);
    ASSERT_NE(arg, nullptr);
    const RegKey merged = result.assignmentOf(idOf(arg->value));

    // One colour for the merge and both inputs, so destruction never needs a copy
    // on either edge. This is why capabilities().mayRecolourMerges stays false.
    EXPECT_EQ(result.assignmentOf(idOf(ssaDefinedValue(*fromLeft))), merged) << result.toString();
    EXPECT_EQ(result.assignmentOf(idOf(ssaDefinedValue(*fromRight))), merged) << result.toString();
}

TEST_F(GreedyAllocatorTest, AnAffinitySetRelocatesAsAUnit) {
    // Withholding just the register the producer used for the merge forces the
    // argument and both inputs somewhere else. They have to move together or
    // destruction would reject the result.
    //
    // Only v5 is withheld, not a whole range: the live-ins v20 through v23 are
    // pinned to their own registers and reserving those would refuse the function
    // before any of this is exercised.
    BasicBlock* entry = block("entry");
    BasicBlock* left = block("left");
    BasicBlock* right = block("right");
    BasicBlock* join = block("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);
    StinkyInstruction* fromLeft = createVAddInBlock(left, kRaTestArch, 5, 20, 21);
    StinkyInstruction* fromRight = createVAddInBlock(right, kRaTestArch, 5, 22, 23);
    createVAddInBlock(join, kRaTestArch, 6, 5, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    setup.target().reserve(RegType::V, 5, 1);
    const AllocationResult result = colour(setup);

    const SSABlockArgument* arg = vgprArgumentFor(*join, 5);
    ASSERT_NE(arg, nullptr);
    const RegKey merged = result.assignmentOf(idOf(arg->value));

    EXPECT_NE(merged.idx, 5u) << result.toString();
    EXPECT_EQ(result.assignmentOf(idOf(ssaDefinedValue(*fromLeft))), merged) << result.toString();
    EXPECT_EQ(result.assignmentOf(idOf(ssaDefinedValue(*fromRight))), merged) << result.toString();
}

TEST_F(GreedyAllocatorTest, FunctionLiveInsKeepTheirRegisters) {
    // A live-in arrives in a register the dispatch filled, so nothing in the
    // function defines it and moving it changes what the kernel reads. It is
    // pinned regardless of policy, which is why greedy-compact cannot trade it
    // away for a lower high-water mark.
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, /*dest=*/40, 20, 21);
    ASSERT_TRUE(liftForAllocation(*func));

    const SSAValueID liveIn = idOf(ssaSourceValue(*add, 0));
    const SSAValueID defined = idOf(ssaDefinedValue(*add));
    ASSERT_TRUE(AllocationSetup(*func).constraints().isPinned(liveIn));
    ASSERT_FALSE(AllocationSetup(*func).constraints().isPinned(defined));

    // Compacting would pack everything from v0 up if it could.
    AllocationSetup setup(*func);
    CompactingGreedyAllocator compact;
    Expected<AllocationResult> result = compact.allocate(setup.context());
    ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());

    EXPECT_EQ(result->assignmentOf(liveIn), (RegKey{RegType::V, 20, RegHalf::NONE}))
        << result->toString();
    // The value the function defines is free to move down.
    EXPECT_LT(result->assignmentOf(defined).idx, 40u) << result->toString();
}

TEST_F(GreedyAllocatorTest, RefusesWhenAPinnedLiveInCannotKeepItsRegister) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    setup.target().reserve(RegType::V, 0, 1);  // v0 arrives as a live-in

    const std::string error = colourError(setup);
    EXPECT_TRUE(contains(error, "is a function live-in")) << error;
    EXPECT_TRUE(contains(error, "v0 is not allocatable")) << error;
}

TEST_F(GreedyAllocatorTest, ScalarsKeepTheirRegistersUnlessAsked) {
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));
    ASSERT_TRUE(liftForAllocation(*func));

    const SSAValueID scalar = idOf(ssaSourceValue(*mov, 0));

    // Default is VGPRs alone, so a scalar keeps the register it arrived in.
    AllocationSetup vgprOnly(*func);
    EXPECT_EQ(colour(vgprOnly).assignmentOf(scalar), (RegKey{RegType::S, 4, RegHalf::NONE}));

    // Asked for scalars too, it is free to place them - here the hint still fits.
    AllocationSetup withSgpr(*func, RegClassSet::all());
    EXPECT_EQ(colour(withSgpr).assignmentOf(scalar).type, RegType::S);
}

TEST_F(GreedyAllocatorTest, ScalarsAreRelocatableOnceAsked) {
    // The scalar that moves has to be one the function defines: a scalar it only
    // reads is a live-in, and those are pinned whatever the scope says.
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* add = builder.create(getMCIDByUOp(GFX::s_add_u32, kRaTestArch));
    add->addDestReg(StinkyRegister("s", 20, 1));
    add->addSrcReg(StinkyRegister("s", 8, 1));
    add->addSrcReg(StinkyRegister("s", 9, 1));
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 20, 1));
    ASSERT_TRUE(liftForAllocation(*func));

    const SSAValueID scalar = idOf(ssaDefinedValue(*add));
    ASSERT_NE(scalar, kInvalidSSAValueID);

    AllocationSetup setup(*func, RegClassSet::only(RegType::S));
    setup.target().reserve(RegType::S, 20, 1);  // s20 is now off limits
    EXPECT_NE(colour(setup).assignmentOf(scalar).idx, 20u);

    // Left out of the set the same reservation is unsatisfiable, because the value
    // may not move off s20 and may not stay there either. The refusal has to name
    // which of the two reasons applies.
    AllocationSetup pinned(*func);
    pinned.target().reserve(RegType::S, 20, 1);
    const std::string error = colourError(pinned);
    EXPECT_TRUE(contains(error, "in a class this run is not colouring")) << error;
    EXPECT_TRUE(contains(error, "s20 is not allocatable")) << error;
}

TEST_F(GreedyAllocatorTest, FailsWhenNoRegisterIsLeft) {
    // Two values the function defines are live across each other, so they need two
    // registers. The live-ins keep v0 and v1, and nothing else is allocatable.
    BasicBlock* entry = block("entry");
    StinkyInstruction* first = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    StinkyInstruction* second = createVAddInBlock(entry, kRaTestArch, 3, 0, 1);
    createVAddInBlock(entry, kRaTestArch, 4, 2, 3);
    ASSERT_TRUE(liftForAllocation(*func));
    ASSERT_TRUE(func->hasAttachedSSA());
    ASSERT_NE(idOf(ssaDefinedValue(*first)), idOf(ssaDefinedValue(*second)));

    AllocationSetup setup(*func);
    ASSERT_NO_FATAL_FAILURE(keepOnlyFirstVgprs(setup, 2));

    const std::string error = colourError(setup);
    EXPECT_TRUE(contains(error, "no v register is free")) << error;
    EXPECT_TRUE(contains(error, "splitting and spilling are not implemented")) << error;
    // Failing is not mutating: the function still carries the SSA it was lifted
    // with, and no operand was touched.
    EXPECT_TRUE(func->hasAttachedSSA());
}

TEST_F(GreedyAllocatorTest, ColouringIsDeterministic) {
    BasicBlock* entry = block("entry");
    BasicBlock* header = block("header");
    BasicBlock* body = block("body");
    BasicBlock* exit = block("exit");
    func->addEdge(entry, header);
    func->addEdge(header, body);
    func->addEdge(body, header);
    func->addEdge(header, exit);
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 9, 20, 21);
    createVAddInBlock(body, kRaTestArch, 9, 9, 22);
    createVAddInBlock(exit, kRaTestArch, 10, 9, 4);
    ASSERT_TRUE(liftForAllocation(*func));

    // Reserved above every live-in, so pressure bites without stranding a pinned
    // value that has nowhere else to go.
    AllocationSetup first(*func);
    AllocationSetup second(*func);
    first.target().reserve(RegType::V, 32, 64);
    second.target().reserve(RegType::V, 32, 64);

    // Weights tie constantly on small functions, so the value-ID tie break is
    // what keeps two runs from disagreeing.
    EXPECT_EQ(colour(first).toString(), colour(second).toString());
}

TEST_F(GreedyAllocatorTest, RegionScopeKeepsTailBlocksByteIdentical) {
    BasicBlock* entry = block("entry");
    BasicBlock* a = block("A");
    BasicBlock* b = block("B");
    BasicBlock* c = block("C");
    BasicBlock* d = block("D");
    BasicBlock* e = block("E");
    func->addEdge(entry, a);
    func->addEdge(a, b);
    func->addEdge(b, c);
    func->addEdge(c, d);
    func->addEdge(d, e);

    StinkyInstruction* first = createVAddInBlock(entry, kRaTestArch, 50, 0, 1);
    createVAddInBlock(a, kRaTestArch, 51, 50, 2);
    createVAddInBlock(b, kRaTestArch, 52, 51, 3);
    StinkyInstruction* cutDef = createVAddInBlock(c, kRaTestArch, 53, 52, 4);
    createVAddInBlock(d, kRaTestArch, 60, 53, 5);
    createVAddInBlock(e, kRaTestArch, 61, 60, 6);
    ASSERT_TRUE(liftForAllocation(*func));

    const std::string before = physicalIR(*func);
    const std::string dBefore = blockSection(before, "D");
    const std::string eBefore = blockSection(before, "E");
    ASSERT_FALSE(dBefore.empty());
    ASSERT_FALSE(eBefore.empty());

    AllocationSetup base(*func);
    AllocationSetup::RegionOptions region{
        .cut = base.intervals().slots().blockEnd(c),
    };
    AllocationSetup setup(*func, RegClassSet::only(RegType::V), region);
    CompactingGreedyAllocator compact;
    Expected<AllocationResult> coloured = compact.allocate(setup.context());
    ASSERT_TRUE(coloured.hasValue()) << (coloured.hasValue() ? "" : coloured.getError());
    const AllocationVerificationResult checked =
        verifyAllocation(*func, *coloured, setup.context());
    ASSERT_TRUE(checked.ok()) << checked.toString();

    // Destruction clears attached SSA, so the IDs have to be read while it is
    // still there.
    const SSAValueID regionOnly = idOf(ssaDefinedValue(*first));
    const SSAValueID crossing = idOf(ssaDefinedValue(*cutDef));

    const SSADestructionResult destroyed = destroyAttachedSSA(*func, *coloured);
    ASSERT_TRUE(destroyed.ok()) << destroyed.toString();

    const std::string after = physicalIR(*func);
    EXPECT_EQ(blockSection(after, "D"), dBefore);
    EXPECT_EQ(blockSection(after, "E"), eBefore);

    EXPECT_LT(coloured->assignmentOf(regionOnly).idx, 50u) << coloured->toString();
    EXPECT_EQ(coloured->assignmentOf(crossing).idx, 53u) << coloured->toString();
}
