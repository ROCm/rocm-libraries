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

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class AllocationVerifierTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    AllocationResult legacyOf() {
        return createLegacyColoring(*func);
    }

    std::unique_ptr<Function> func;
};

}  // namespace

TEST_F(AllocationVerifierTest, LegacyColoringOfAStraightLinePasses) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked =
        verifyAllocation(*func, legacyOf(), setup.context());
    EXPECT_TRUE(checked.ok()) << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesAnUnassignedValue) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationResult partial(*func);
    for (StinkySSAValue* value : func->ssaArena().values()) {
        if (value->valueId() == 1) continue;
        ASSERT_TRUE(value->hasPhysicalBinding());
        const auto& binding = value->physical();
        partial.assign(value->valueId(), RegKey{binding.type, binding.idx, RegHalf::NONE});
    }

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked = verifyAllocation(*func, partial, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "has no physical register")) << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesOverlappingValuesOnTheSameUnit) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* src0 = ssaSourceValue(*add, 0);
    const StinkySSAValue* src1 = ssaSourceValue(*add, 1);
    ASSERT_NE(src0, nullptr);
    ASSERT_NE(src1, nullptr);

    AllocationResult colouring = legacyOf();
    colouring.assign(src1->valueId(), colouring.assignmentOf(src0->valueId()));

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked =
        verifyAllocation(*func, colouring, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "overlaps")) << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesANonConsecutiveTuple) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationResult scattered = legacyOf();
    const std::vector<StinkySSAValue*> units = ssaDestUnits(*load, 0);
    ASSERT_EQ(units.size(), 4u);
    unsigned next = 20;
    for (StinkySSAValue* unit : units) {
        scattered.assign(unit->valueId(), RegKey{RegType::V, next, RegHalf::NONE});
        next += 7;
    }

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked =
        verifyAllocation(*func, scattered, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "must be consecutive in operand order"))
        << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesASplitAffinitySet) {
    BasicBlock* entry = block("entry");
    BasicBlock* left = block("left");
    BasicBlock* right = block("right");
    BasicBlock* join = block("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);
    createVAddInBlock(left, kRaTestArch, 5, 20, 21);
    createVAddInBlock(right, kRaTestArch, 5, 22, 23);
    createVAddInBlock(join, kRaTestArch, 6, 5, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    const SSABlockArgument* arg = vgprArgumentFor(*join, 5);
    ASSERT_NE(arg, nullptr);
    ASSERT_FALSE(arg->incoming.empty());
    StinkySSAValue* incoming = arg->incoming.front().use->value();
    ASSERT_NE(incoming, nullptr);

    AllocationResult colouring = legacyOf();
    colouring.assign(incoming->valueId(), RegKey{RegType::V, 200, RegHalf::NONE});

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked =
        verifyAllocation(*func, colouring, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "must share one colour")) << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesAReservedUnit) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    setup.target().reserve(RegType::V, 0, 1);

    const AllocationVerificationResult checked =
        verifyAllocation(*func, legacyOf(), setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "reserved")) << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesAStaleShape) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));
    const AllocationResult stale = legacyOf();

    add->setSrcReg(0, StinkyRegister("v", 9, 1));
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked = verifyAllocation(*func, stale, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "computed against a different graph"))
        << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesARelocatedFunctionLiveIn) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 40, 20, 21);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* liveIn = ssaSourceValue(*add, 0);
    ASSERT_NE(liveIn, nullptr);

    AllocationResult moved = legacyOf();
    moved.assign(liveIn->valueId(), RegKey{RegType::V, 99, RegHalf::NONE});

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked = verifyAllocation(*func, moved, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "is a function live-in")) << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesANewcomerInAPinnedRegister) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 40, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* defined = ssaDefinedValue(*add);
    ASSERT_NE(defined, nullptr);

    // v41 was lifted from nothing, so there is no occupant check to catch this.
    AllocationResult moved = legacyOf();
    moved.assign(defined->valueId(), RegKey{RegType::V, 41, RegHalf::NONE});

    AllocationSetup::RegionOptions hold{.pinRegisters = {{RegType::V, 41, 41}}};
    AllocationSetup setup(*func, RegClassSet::only(RegType::V), hold);

    const AllocationVerificationResult checked = verifyAllocation(*func, moved, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "which this run is holding")) << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesARelocatedClassOutsideScope) {
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kRaTestArch);
    // The scalar has to be defined in the function. A live-in is pinned, and the
    // pin check owns that value, so the out-of-scope reason would never be the
    // one reported.
    StinkyInstruction* def = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
    def->addDestReg(StinkyRegister("s", 4, 1));
    def->addSrcReg(StinkyRegister("s", 5, 1));
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* scalar = ssaSourceValue(*mov, 0);
    ASSERT_NE(scalar, nullptr);

    AllocationResult moved = legacyOf();
    moved.assign(scalar->valueId(), RegKey{RegType::S, 99, RegHalf::NONE});

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked = verifyAllocation(*func, moved, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "in a class this run is not colouring"))
        << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesOverlapWithAFunctionLiveIn) {
    // A live-in that kept its hint still occupies the register. Skipping the
    // occupancy bind after the pin check would miss a mobile value placed on it.
    BasicBlock* entry = block("entry");
    StinkyInstruction* first = createVAddInBlock(entry, kRaTestArch, 40, 20, 21);
    createVAddInBlock(entry, kRaTestArch, 41, 20, 40);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* defined = ssaDefinedValue(*first);
    ASSERT_NE(defined, nullptr);

    AllocationResult colouring = legacyOf();
    colouring.assign(defined->valueId(), RegKey{RegType::V, 20, RegHalf::NONE});

    AllocationSetup setup(*func);
    const AllocationVerificationResult checked =
        verifyAllocation(*func, colouring, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "overlaps")) << checked.toString();
}

TEST_F(AllocationVerifierTest, CatchesARelocatedValueOutsideRegion) {
    BasicBlock* entry = block("entry");
    BasicBlock* tail = block("tail");
    func->addEdge(entry, tail);
    StinkyInstruction* def = createVAddInBlock(entry, kRaTestArch, 50, 0, 1);
    createVAddInBlock(tail, kRaTestArch, 60, 50, 2);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup base(*func);
    const SlotIndex cut = base.intervals().slots().blockEnd(entry);
    AllocationSetup::RegionOptions region{.cut = cut};
    AllocationSetup setup(*func, RegClassSet::only(RegType::V), region);

    AllocationResult moved = legacyOf();
    moved.assign(ssaDefinedValue(*def)->valueId(), RegKey{RegType::V, 10, RegHalf::NONE});

    const AllocationVerificationResult checked = verifyAllocation(*func, moved, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "outside the region this run is colouring"))
        << checked.toString();
}
