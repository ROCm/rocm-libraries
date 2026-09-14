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
#include <set>
#include <string>
#include <vector>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "stinkytofu/transforms/asm/ssa/SSADestruction.hpp"
#include "transforms/asm/ra/allocators/GreedyAllocator.hpp"
#include "transforms/asm/ra/allocators/GreedyPlacement.hpp"

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

    /// Colours \p setup with \p allocator and requires a legal result.
    AllocationResult colourWith(RegisterAllocator& allocator, AllocationSetup& setup) {
        Expected<AllocationResult> result = allocator.allocate(setup.context());
        EXPECT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
        if (!result.hasValue()) return AllocationResult{};

        const AllocationVerificationResult checked =
            verifyAllocation(*func, *result, setup.context());
        EXPECT_TRUE(checked.ok()) << checked.toString() << "\n" << result->toString();
        return std::move(*result);
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

/// v[<dst>:<dst>+1] = ds_load_b64(v<addr>), a two-DWORD range to compete with.
StinkyInstruction* createDsLoadB64(BasicBlock* bb, int dstReg, int addrReg) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* load = builder.create(getMCIDByUOp(GFX::ds_load_b64, kRaTestArch));
    load->addDestReg(StinkyRegister("v", dstReg, 2));
    load->addSrcReg(StinkyRegister("v", addrReg, 1));
    return load;
}

/// `v_wmma_scale16 dst, a, b, 0, scaleA, scaleB`: six register fields against
/// the four slots s_set_vgpr_msb carries, so the two scale fields select none
/// and reach the first bank only. The one shipped format with that shape.
StinkyInstruction* createWmmaScale16(BasicBlock* bb, int dst, int a, int b, int scaleA,
                                     int scaleB) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* wmma =
        builder.create(getMCIDByUOp(GFX::v_wmma_scale16_f32_16x16x128_f8f6f4, kRaTestArch));
    wmma->addDestReg(StinkyRegister("v", dst, 8));
    wmma->addSrcReg(StinkyRegister("v", a, 8));
    wmma->addSrcReg(StinkyRegister("v", b, 8));
    wmma->addSrcReg(StinkyRegister(0));
    wmma->addSrcReg(StinkyRegister("v", scaleA, 2));
    wmma->addSrcReg(StinkyRegister("v", scaleB, 2));
    return wmma;
}

/// `v_wmma_f32_16x16x32_bf16 dst, a, b, c` -- three 8-wide sources and an 8-wide
/// destination, and no field short of an MSB slot, so it is a wide block with no
/// capped sibling to compete with.
StinkyInstruction* createWmmaBf16(BasicBlock* bb, int dst, int a, int b, int c) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* wmma =
        builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kRaTestArch));
    wmma->addDestReg(StinkyRegister("v", dst, 8));
    wmma->addSrcReg(StinkyRegister("v", a, 8));
    wmma->addSrcReg(StinkyRegister("v", b, 8));
    wmma->addSrcReg(StinkyRegister("v", c, 8));
    return wmma;
}

/// Defines \p count consecutive VGPRs from \p base, so operands reading them
/// are values the function defines rather than live-ins nothing may move.
void defineVgprs(BasicBlock* bb, int base, int count) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    for (int i = 0; i < count; ++i) {
        StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
        mov->addDestReg(StinkyRegister("v", base + i, 1));
        mov->addSrcReg(StinkyRegister(0));
    }
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

TEST_F(GreedyAllocatorTest, ACappedBlockTakesItsBankBeforeOthersCanFillIt) {
    // The scale operands of v_wmma_scale16 reach the first bank only, because
    // their encoding fields select no s_set_vgpr_msb slot. Every block prefers a
    // low base, since pickBase scans upward and takes the first that fits, so a
    // capped block left in the weight-ordered queue competes for its one bank
    // against blocks that could have gone anywhere -- and arrives to find it
    // full. Placing capped blocks in a phase of their own is the whole fix.
    //
    // Bank 0 is cut down to v251-v255 here, which holds the two scale pairs and
    // nothing else. The loads are defined next to their uses and so outweigh the
    // scale operands, which are defined first and read last; in one queue they
    // would take v251-v254 and leave the scale operands nowhere to go.
    BasicBlock* entry = block("entry");
    defineVgprs(entry, /*base=*/130, /*count=*/2);
    defineVgprs(entry, /*base=*/140, /*count=*/2);
    StinkyInstruction* firstLoad = createDsLoadB64(entry, /*dstReg=*/200, /*addrReg=*/400);
    StinkyInstruction* secondLoad = createDsLoadB64(entry, /*dstReg=*/202, /*addrReg=*/400);
    createVAddInBlock(entry, kRaTestArch, /*dest=*/210, /*src0=*/200, /*src1=*/202);
    // The matrix operands and the address are live-ins, so they are pinned where
    // they arrive and have to sit outside the range reserved below.
    StinkyInstruction* wmma = createWmmaScale16(entry, /*dst=*/100, /*a=*/300, /*b=*/320,
                                                /*scaleA=*/130, /*scaleB=*/140);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V));
    // v251-v254 is all that is left of the reachable bank, which the two scale
    // pairs fill exactly. v255 goes too, or a load takes it and runs into the
    // next bank, which is a separate problem and not the one under test.
    setup.target().reserve(RegType::V, /*first=*/0, /*count=*/251);
    setup.target().reserve(RegType::V, /*first=*/255, /*count=*/1);

    // Both policies, because this is the evidence that folding the capped phase
    // into the freedom order preserved what the phase was there to protect.
    GreedyAllocator byWeight;
    FreedomOrderedGreedyAllocator byFreedom;
    for (RegisterAllocator* allocator : {static_cast<RegisterAllocator*>(&byWeight),
                                         static_cast<RegisterAllocator*>(&byFreedom)}) {
        const AllocationResult coloured = colourWith(*allocator, setup);

        for (size_t operand = 3; operand <= 4; ++operand) {
            for (const StinkySSAValue* unit : ssaSourceUnits(*wmma, operand)) {
                ASSERT_NE(unit, nullptr);
                EXPECT_LT(coloured.assignmentOf(unit->valueId()).idx, kVgprBankSize)
                    << allocator->name() << ", operand " << operand << '\n'
                    << coloured.toString();
            }
        }

        // And the loads, which outweigh them, were pushed past the bank. Without
        // this the test would pass on a colouring that simply had room for both.
        for (const StinkyInstruction* load : {firstLoad, secondLoad}) {
            const StinkySSAValue* first = ssaDefinedValue(*load, 0);
            ASSERT_NE(first, nullptr);
            EXPECT_GE(coloured.assignmentOf(first->valueId()).idx, kVgprBankSize)
                << allocator->name() << '\n'
                << coloured.toString();
        }
    }
}

// ---------------------------------------------------------------------------
// placementFreedom: how many bases a block could legally take
// ---------------------------------------------------------------------------

TEST_F(GreedyAllocatorTest, FreedomIsTheWholeFileForAnUnconstrainedRegister) {
    // Nothing narrows a single register with no rules in force, so every index
    // is a candidate. Read back through the refusal, which is where the count
    // is reported; withholding the file is just a way to provoke one.
    BasicBlock* entry = block("entry");
    defineVgprs(entry, /*base=*/40, /*count=*/1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V));
    setup.target().reserve(RegType::V, /*first=*/0, /*count=*/1024);

    EXPECT_TRUE(contains(colourError(setup), "1024 legal base(s)")) << colourError(setup);
}

TEST_F(GreedyAllocatorTest, AnEvenBaseRuleHalvesTheFreedomOfAWideRange) {
    // VectorTupleAlignment removes every odd base, and a 4-wide range cannot
    // start past v1020, which leaves 511 of the 1021 bases it would otherwise
    // have. The address register is a live-in, so its register is left out of
    // the reservation or it would fail first with a pin message instead.
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kRaTestArch, /*destReg=*/100,
                                                      /*addrReg=*/500);
    ASSERT_TRUE(liftForAllocation(*func));
    ASSERT_NE(ssaDefinedValue(*load, 0), nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    setup.target().reserve(RegType::V, /*first=*/0, /*count=*/500);
    setup.target().reserve(RegType::V, /*first=*/501, /*count=*/523);

    EXPECT_TRUE(contains(colourError(setup), "511 legal base(s)")) << colourError(setup);
}

TEST_F(GreedyAllocatorTest, AnIndexCeilingCutsFreedomToTheBankItCanReach) {
    // A scale operand reaches v0-v255 only, and on even bases that is 128 of the
    // 1022 a free 2-wide range would have. Its own phase reports the count too,
    // because that is the refusal a capped block reaches.
    BasicBlock* entry = block("entry");
    // Both scale operands are defined here, so neither arrives as a live-in
    // pinned inside the bank the reservation withholds. Only the matrix
    // operands are live-ins, and they sit above it.
    defineVgprs(entry, /*base=*/130, /*count=*/2);
    defineVgprs(entry, /*base=*/140, /*count=*/2);
    createWmmaScale16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*scaleA=*/130, /*scaleB=*/140);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    setup.target().reserve(RegType::V, /*first=*/0, /*count=*/kVgprBankSize);

    EXPECT_TRUE(contains(colourError(setup), "128 legal base(s)")) << colourError(setup);
}

TEST_F(GreedyAllocatorTest, FreedomOrderGivesAWideRangeTheLowBasesFirst) {
    // What ordering by constrainedness buys. The wide block has one use per
    // member over a long range, so it is the lightest thing here and weight
    // order reaches it last, by which point the singles have taken the bottom
    // of the file and it has to start above them. Freedom order sees that it
    // has the fewest places to go and takes it first.
    BasicBlock* entry = block("entry");
    StinkyInstruction* wmma = createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*c=*/340);
    // Short ranges and a use each, so these outweigh the wide block.
    for (int i = 0; i < 8; ++i) {
        defineVgprs(entry, /*base=*/200 + i, /*count=*/1);
        createVAddInBlock(entry, kRaTestArch, /*dest=*/400 + i, /*src0=*/200 + i, /*src1=*/200 + i);
    }
    createVAddInBlock(entry, kRaTestArch, /*dest=*/40, /*src0=*/100, /*src1=*/101);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* tile = ssaDefinedValue(*wmma, 0);
    ASSERT_NE(tile, nullptr);
    const SSAValueID tileBase = tile->valueId();

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());

    CompactingGreedyAllocator byWeight;
    const uint32_t weightBase = colourWith(byWeight, setup).assignmentOf(tileBase).idx;

    FreedomOrderedGreedyAllocator byFreedom;
    const uint32_t freedomBase = colourWith(byFreedom, setup).assignmentOf(tileBase).idx;

    EXPECT_EQ(freedomBase, 0u) << "the least free block should get the lowest legal base";
    EXPECT_GT(weightBase, freedomBase) << "weight order should have placed it after the singles";
}

TEST_F(GreedyAllocatorTest, FreedomOrderKeepsACappedBlockAheadOfEveryWiderOne) {
    // The property that lets the capped phase be folded away rather than kept
    // beside the new order, asserted on the comparator because the colouring
    // cannot show it: a scale operand dies at the instruction that defines the
    // tile, so the tile reuses its registers and the two indices say nothing
    // about which was placed first.
    //
    // A ceiling of 255 holds a 2-wide block to 128 even bases. The widest
    // uncapped block on this target still has over 500, so no width can push a
    // capped block behind an uncapped one.
    Block capped;
    capped.width = 2;
    capped.maxIndex = kVgprBankSize - 1;
    capped.placementFreedom = 128;
    capped.weight = 0.0;  // and it still wins, because freedom outranks weight
    capped.leader = 100;

    Block tile;
    tile.width = 8;
    tile.placementFreedom = 509;
    tile.weight = 1000.0;
    tile.leader = 1;

    const PlacementPolicy& policy = freedomPolicy();
    EXPECT_TRUE(policy.placesBefore(capped, tile));
    EXPECT_FALSE(policy.placesBefore(tile, capped));

    // And neither needs a phase ahead of the queue, which is what the capped
    // blocks used to get.
    EXPECT_FALSE(policy.placesEarly(capped));
    EXPECT_FALSE(policy.placesEarly(tile));

    // Weight still separates blocks the count cannot tell apart.
    Block hotSingle;
    hotSingle.placementFreedom = 1024;
    hotSingle.weight = 5.0;
    hotSingle.leader = 2;
    Block coldSingle;
    coldSingle.placementFreedom = 1024;
    coldSingle.weight = 1.0;
    coldSingle.leader = 3;
    EXPECT_TRUE(policy.placesBefore(hotSingle, coldSingle));
    EXPECT_FALSE(policy.placesBefore(coldSingle, hotSingle));
}

TEST_F(GreedyAllocatorTest, FreedomEvictionDisplacesAHotterBlockThatHasSomewhereElseToGo) {
    // The recovery rule agreeing with the ordering rule. A hot single sitting
    // inside an otherwise-free run is what keeps a wide range out of it, and
    // weight says the single stays because it is worth more. Freedom asks the
    // question the situation poses instead: which of the two has anywhere else
    // to be.
    Block tile;
    tile.width = 8;
    tile.placementFreedom = 509;
    tile.weight = 0.1;
    tile.leader = 1;

    Block hotSingle;
    hotSingle.placementFreedom = 1024;
    hotSingle.weight = 100.0;
    hotSingle.leader = 2;

    EXPECT_TRUE(freedomPolicy().mayEvict(tile, hotSingle));
    EXPECT_FALSE(weightPolicy().mayEvict(tile, hotSingle));

    // Not a licence to shuffle: the freer block still cannot displace the one
    // with fewer places to go, which is what keeps eviction chains finite.
    EXPECT_FALSE(freedomPolicy().mayEvict(hotSingle, tile));

    // Among equally free blocks it is the weight rule, unchanged.
    Block otherSingle;
    otherSingle.placementFreedom = 1024;
    otherSingle.weight = 1.0;
    otherSingle.leader = 3;
    EXPECT_TRUE(freedomPolicy().mayEvict(hotSingle, otherSingle));
    EXPECT_FALSE(freedomPolicy().mayEvict(otherSingle, hotSingle));

    // And the weight policy keeps its own protection for capped blocks, which
    // it still needs while it still has a phase for them.
    Block capped;
    capped.width = 2;
    capped.maxIndex = kVgprBankSize - 1;
    capped.placementFreedom = 128;
    capped.weight = 0.1;
    capped.leader = 4;
    EXPECT_FALSE(weightPolicy().mayEvict(hotSingle, capped));
}

TEST_F(GreedyAllocatorTest, ARefusalSaysWhetherTheBlockersCouldHaveMoved) {
    // The question recoloring turns on, answered in the refusal itself. A wide
    // block that cannot be placed is worth reshuffling for only if the blocks in
    // its way have somewhere else to go, so the refusal reports how many of them
    // are freer than it and how many are pinned where they are.
    //
    // Two free regions remain: v240-v247, which is the only run the tile could
    // use, and v300-v307, which holds a live-in kept alive to the end so the
    // tile cannot reuse it. The singles are placed first under weight order,
    // take v240, and leave the tile nowhere.
    BasicBlock* entry = block("entry");
    StinkyInstruction* wmma = createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/300, /*c=*/300);
    for (int i = 0; i < 4; ++i) {
        defineVgprs(entry, /*base=*/200 + i, /*count=*/1);
        createVAddInBlock(entry, kRaTestArch, /*dest=*/500 + i, /*src0=*/200 + i, /*src1=*/200 + i);
    }
    createVAddInBlock(entry, kRaTestArch, /*dest=*/40, /*src0=*/100, /*src1=*/101);
    // Keeps the live-in occupying v300-v307 past the tile's definition, so the
    // tile cannot take the registers it vacates.
    createVAddInBlock(entry, kRaTestArch, /*dest=*/41, /*src0=*/300, /*src1=*/301);
    ASSERT_TRUE(liftForAllocation(*func));
    ASSERT_NE(ssaDefinedValue(*wmma, 0), nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    setup.target().reserve(RegType::V, /*first=*/0, /*count=*/240);
    setup.target().reserve(RegType::V, /*first=*/248, /*count=*/52);
    setup.target().reserve(RegType::V, /*first=*/308, /*count=*/716);

    const std::string error = colourError(setup);
    EXPECT_TRUE(contains(error, "freer than this one")) << error;
    EXPECT_TRUE(contains(error, "occupied by")) << error;
    // And the shape histogram comes with it, so a reader can see how coarse the
    // ordering had to be.
    EXPECT_TRUE(contains(error, "v blocks by width")) << error;
    EXPECT_TRUE(contains(error, "by freedom")) << error;
}

// ---------------------------------------------------------------------------
// Soft pairings
// ---------------------------------------------------------------------------

namespace {

/// A one-row table wanting every destination on its first source's register.
/// Stands in for a real pairing rule without depending on a shipped one.
AllocationRules destPrefersFirstSource() {
    AllocationRule rule;
    rule.name = "DestPrefersFirstSource";
    rule.description = "a destination would rather reuse its first source's register";
    rule.status = RuleStatus::Active;
    rule.satisfiedBy = [](RegKey a, RegKey b) { return a == b; };
    rule.addPreferences = [](const StinkyInstruction&, const OperandValues& values,
                             std::vector<Preference>& out) {
        const std::span<const SSAValueID> dest = values(0, /*isDest=*/true);
        const std::span<const SSAValueID> src = values(0, /*isDest=*/false);
        if (dest.empty() || src.empty()) return;
        out.push_back({dest[0], src[0], 0, 1.0});
    };
    return AllocationRules({rule});
}

/// `v[dest:dest+width-1] = ds_load_bN v<addr>` -- a vector tuple of a chosen
/// width, so a pairing can relate members sitting at different offsets.
StinkyInstruction* dsLoad(BasicBlock* bb, int dest, uint16_t width, int addr) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* load = builder.create(
        getMCIDByUOp(width == 2 ? GFX::ds_load_b64 : GFX::ds_load_b128, kRaTestArch));
    load->addDestReg(StinkyRegister("v", dest, width));
    load->addSrcReg(StinkyRegister("v", addr, 1));
    return load;
}

/// Wants a destination on its address register.
///
/// Contrived on purpose. What it buys over destPrefersFirstSource is a pairing
/// between members at *different* offsets of their tuples, which is the only
/// shape where folding produces a block wider than either end -- and so the
/// only shape where folding can ask placement for something neither end did.
AllocationRules destPrefersItsAddress() {
    AllocationRule rule;
    rule.name = "DestPrefersItsAddress";
    rule.description = "a load destination would rather reuse its address register";
    rule.status = RuleStatus::Active;
    rule.satisfiedBy = [](RegKey a, RegKey b) { return a == b; };
    rule.addPreferences = [](const StinkyInstruction&, const OperandValues& values,
                             std::vector<Preference>& out) {
        const std::span<const SSAValueID> dest = values(0, /*isDest=*/true);
        const std::span<const SSAValueID> addr = values(0, /*isDest=*/false);
        if (dest.empty() || addr.empty()) return;
        out.push_back({dest[0], addr[0], 0, 1.0});
    };
    return AllocationRules({rule});
}

/// destPrefersItsAddress, plus a placement rule only a folded block can break.
///
/// Five registers wide is what the fold below produces and neither end is, so
/// confining that width to one base leaves the fold legal to make and
/// impossible to place -- which is the case the retry exists for.
AllocationRules addressPairingWithOneWideBase() {
    AllocationRule narrow;
    narrow.name = "WideVBlocksStartAtZero";
    narrow.description = "a V block five registers or wider must start at index 0";
    narrow.status = RuleStatus::Active;
    narrow.forbidsBase = [](RegType regClass, uint32_t base, uint32_t width) {
        return regClass == RegType::V && width >= 5 && base != 0;
    };
    return AllocationRules({narrow, destPrefersItsAddress().all().front()});
}

}  // namespace

TEST_F(GreedyAllocatorTest, FoldingProducesABlockWiderThanEitherEndOfThePair) {
    // The address is DWORD 1 of a two-wide tuple and the destination is DWORD 0
    // of a four-wide one, so putting them on one register leaves the tuple
    // below the destination and the rest of the destination above it: five
    // registers where neither end needed more than four.
    BasicBlock* entry = block("entry");
    StinkyInstruction* narrow = dsLoad(entry, /*dest=*/10, /*width=*/2, /*addr=*/50);
    StinkyInstruction* wide = dsLoad(entry, /*dest=*/12, /*width=*/4, /*addr=*/11);
    createVAddInBlock(entry, kRaTestArch, /*dest=*/61, /*src0=*/12, /*src1=*/13);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* dest = ssaDefinedValue(*wide, 0);
    const std::vector<StinkySSAValue*> address = ssaSourceUnits(*wide, 0);
    ASSERT_NE(dest, nullptr);
    ASSERT_EQ(address.size(), 1u);
    ASSERT_NE(address[0], nullptr);

    AllocationSetup paired(*func, RegClassSet::only(RegType::V), {}, destPrefersItsAddress());
    CompactingGreedyAllocator allocator;
    const AllocationResult coloured = colourWith(allocator, paired);

    EXPECT_EQ(coloured.assignmentOf(dest->valueId()).idx,
              coloured.assignmentOf(address[0]->valueId()).idx)
        << coloured.toString();

    // The width is the point, so count it rather than infer it. Six values,
    // two of them sharing one register, leaves five in a row -- which is what
    // the test below then finds nowhere to put.
    std::set<uint32_t> occupied;
    for (unsigned unit = 0; unit < 4; ++unit) {
        const StinkySSAValue* value = ssaDefinedValue(*wide, unit);
        ASSERT_NE(value, nullptr) << "wide unit " << unit;
        occupied.insert(coloured.assignmentOf(value->valueId()).idx);
    }
    for (unsigned unit = 0; unit < 2; ++unit) {
        const StinkySSAValue* value = ssaDefinedValue(*narrow, unit);
        ASSERT_NE(value, nullptr) << "narrow unit " << unit;
        occupied.insert(coloured.assignmentOf(value->valueId()).idx);
    }
    ASSERT_EQ(occupied.size(), 5u) << coloured.toString();
    EXPECT_EQ(*occupied.rbegin() - *occupied.begin(), 4u) << "expected one run of five";
}

TEST_F(GreedyAllocatorTest, AFoldThatCannotBePlacedGivesWayRatherThanRefusing) {
    // The same pair, with the one base its folded width may start at already
    // held by a live-in. Folding is legal to make and impossible to place.
    //
    // This is the invariant the whole framework rests on: a soft rule may not
    // decide whether a kernel colours. Placement refuses, the run is retried
    // with folding off, and the two ends go back to separate registers -- four
    // wide and two wide, which the rule above does not constrain.
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, /*dest=*/60, /*src0=*/4, /*src1=*/4);
    dsLoad(entry, /*dest=*/10, /*width=*/2, /*addr=*/50);
    StinkyInstruction* wide = dsLoad(entry, /*dest=*/12, /*width=*/4, /*addr=*/11);
    createVAddInBlock(entry, kRaTestArch, /*dest=*/61, /*src0=*/12, /*src1=*/13);
    // Read again, so v4 is live across everything and base 0 stays unusable for
    // any block reaching as far as index 4.
    createVAddInBlock(entry, kRaTestArch, /*dest=*/62, /*src0=*/4, /*src1=*/4);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* dest = ssaDefinedValue(*wide, 0);
    const std::vector<StinkySSAValue*> address = ssaSourceUnits(*wide, 0);
    ASSERT_NE(dest, nullptr);
    ASSERT_EQ(address.size(), 1u);
    ASSERT_NE(address[0], nullptr);

    AllocationSetup paired(*func, RegClassSet::only(RegType::V), {},
                           addressPairingWithOneWideBase());
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> coloured = allocator.allocate(paired.context());

    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_TRUE(verifyAllocation(*func, *coloured, paired.context()).ok());
    EXPECT_NE(coloured->assignmentOf(dest->valueId()).idx,
              coloured->assignmentOf(address[0]->valueId()).idx)
        << coloured->toString();
}

TEST_F(GreedyAllocatorTest, APairingPullsABlockOffTheRegisterFirstFitWouldTake) {
    // v300 is a live-in, so it is pinned up there and dies at the add. The
    // destination could take its register, but packing from the bottom has no
    // reason to: first-fit gives it v0. The pairing is what changes that, and
    // the contrast against the same function with no rule is what makes this
    // a test of the pairing rather than of the function being small.
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, /*dest=*/40, /*src0=*/300,
                                               /*src1=*/300);
    ASSERT_TRUE(liftForAllocation(*func));
    const StinkySSAValue* dest = ssaDefinedValue(*add);
    ASSERT_NE(dest, nullptr);

    CompactingGreedyAllocator allocator;

    AllocationSetup unpaired(*func, RegClassSet::only(RegType::V));
    EXPECT_EQ(colourWith(allocator, unpaired).assignmentOf(dest->valueId()).idx, 0u);

    AllocationSetup paired(*func, RegClassSet::only(RegType::V), {}, destPrefersFirstSource());
    EXPECT_EQ(colourWith(allocator, paired).assignmentOf(dest->valueId()).idx, 300u);
}

TEST_F(GreedyAllocatorTest, AnUnsatisfiablePairingIsSkippedRatherThanRefused) {
    // The safety property. Here v300 is read again after the add, so it is
    // still live where the destination would have to sit and the two cannot
    // share. The preference simply goes unmet: scoring only ever reorders
    // bases that placement had already accepted, so there is nothing it can
    // refuse.
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, /*dest=*/40, /*src0=*/300,
                                               /*src1=*/300);
    createVAddInBlock(entry, kRaTestArch, /*dest=*/41, /*src0=*/300, /*src1=*/40);
    ASSERT_TRUE(liftForAllocation(*func));
    const StinkySSAValue* dest = ssaDefinedValue(*add);
    ASSERT_NE(dest, nullptr);

    AllocationSetup paired(*func, RegClassSet::only(RegType::V), {}, destPrefersFirstSource());
    CompactingGreedyAllocator allocator;
    const AllocationResult coloured = colourWith(allocator, paired);

    EXPECT_NE(coloured.assignmentOf(dest->valueId()).idx, 300u) << coloured.toString();
}
