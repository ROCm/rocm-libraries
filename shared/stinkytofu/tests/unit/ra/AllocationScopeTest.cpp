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

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationScope.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class AllocationScopeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    SSAValueID idOf(const StinkySSAValue* value) const {
        return value == nullptr ? kInvalidSSAValueID : value->valueId();
    }

    std::unique_ptr<Function> func;
};

}  // namespace

TEST_F(AllocationScopeTest, WholeFunctionLeavesVgprsMobile) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 40, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const SSAValueID defined = idOf(ssaDefinedValue(*add));
    EXPECT_EQ(setup.scope().immobileReason(defined), nullptr);
}

TEST_F(AllocationScopeTest, ClassOutsideScopeIsImmobile) {
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const SSAValueID scalar = idOf(ssaSourceValue(*mov, 0));
    EXPECT_STREQ(setup.scope().immobileReason(scalar), "in a class this run is not colouring");
}

TEST_F(AllocationScopeTest, ContainedInRulePinsValuesCrossingTheCut) {
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

    const SSAValueID crossing = idOf(ssaDefinedValue(*def));
    EXPECT_STREQ(setup.scope().immobileReason(crossing),
                 "outside the region this run is colouring");
}

TEST_F(AllocationScopeTest, DefinedInRuleAllowsValuesUsedAfterTheCut) {
    BasicBlock* entry = block("entry");
    BasicBlock* tail = block("tail");
    func->addEdge(entry, tail);
    StinkyInstruction* def = createVAddInBlock(entry, kRaTestArch, 50, 0, 1);
    createVAddInBlock(tail, kRaTestArch, 60, 50, 2);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup base(*func);
    const SlotIndex cut = base.intervals().slots().blockEnd(entry);
    AllocationSetup::RegionOptions region{
        .cut = cut,
        .containment = AllocationScope::Containment::DefinedIn,
    };
    AllocationSetup setup(*func, RegClassSet::only(RegType::V), region);

    const SSAValueID crossing = idOf(ssaDefinedValue(*def));
    EXPECT_EQ(setup.scope().immobileReason(crossing), nullptr);
}

TEST_F(AllocationScopeTest, BackedgeExtendsRangePastTheCut) {
    BasicBlock* entry = block("entry");
    BasicBlock* body = block("body");
    BasicBlock* tail = block("tail");
    func->addEdge(entry, body);
    func->addEdge(body, tail);
    func->addEdge(tail, body);
    StinkyInstruction* def = createVAddInBlock(body, kRaTestArch, 50, 0, 1);
    createVAddInBlock(tail, kRaTestArch, 60, 50, 2);
    (void)def;
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup base(*func);
    const SlotIndex cut = base.intervals().slots().blockEnd(body);
    AllocationSetup::RegionOptions region{.cut = cut};
    AllocationSetup setup(*func, RegClassSet::only(RegType::V), region);

    const SSAValueID loopCarried = idOf(ssaDefinedValue(*def));
    EXPECT_STREQ(setup.scope().immobileReason(loopCarried),
                 "outside the region this run is colouring");
}

TEST_F(AllocationScopeTest, PinningARegisterKeepsItsOccupant) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* held = createVAddInBlock(entry, kRaTestArch, 40, 0, 1);
    StinkyInstruction* other = createVAddInBlock(entry, kRaTestArch, 41, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup::RegionOptions hold{.pinRegisters = {{RegType::V, 40, 40}}};
    AllocationSetup setup(*func, RegClassSet::only(RegType::V), hold);

    EXPECT_STREQ(setup.scope().immobileReason(idOf(ssaDefinedValue(*held))),
                 "in a register this run is holding");
    // Holding one register is not holding the run.
    EXPECT_EQ(setup.scope().immobileReason(idOf(ssaDefinedValue(*other))), nullptr);
}

TEST_F(AllocationScopeTest, PinnedRangesCoverExactlyWhatTheyName) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 40, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    // Two runs plus a single register as the degenerate {n, n}.
    AllocationSetup::RegionOptions hold{
        .pinRegisters = {{RegType::V, 0, 1}, {RegType::V, 8, 11}, {RegType::V, 20, 20}}};
    AllocationSetup setup(*func, RegClassSet::only(RegType::V), hold);
    const AllocationScope& scope = setup.scope();

    // Both ends of both runs: an off-by-one here silently holds or frees a
    // register.
    EXPECT_TRUE(scope.isPinnedRegister(RegType::V, 1));
    EXPECT_FALSE(scope.isPinnedRegister(RegType::V, 2));
    EXPECT_FALSE(scope.isPinnedRegister(RegType::V, 7));
    EXPECT_TRUE(scope.isPinnedRegister(RegType::V, 8));
    EXPECT_TRUE(scope.isPinnedRegister(RegType::V, 11));
    EXPECT_FALSE(scope.isPinnedRegister(RegType::V, 12));
    EXPECT_TRUE(scope.isPinnedRegister(RegType::V, 20));
    EXPECT_FALSE(scope.isPinnedRegister(RegType::V, 21));
    // Class is part of the key, so the same index elsewhere is untouched.
    EXPECT_FALSE(scope.isPinnedRegister(RegType::S, 8));
}

TEST_F(AllocationScopeTest, ValidateClassesRejectsUnliftedClasses) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);

    // A default lift covers every liftable class, so the lift has to be narrowed
    // for there to be a class left to reject.
    LiftAsmRegistersToSSAOptions options;
    options.classes = RegClassSet::only(RegType::V);
    ASSERT_TRUE(liftAsmRegistersToAttachedSSA(*func, options).hasValue());

    const std::optional<std::string> error =
        AllocationScope::validateClasses(*func, RegClassSet::only(RegType::S));
    ASSERT_TRUE(error.has_value());
    EXPECT_NE(error->find("asked to allocate s"), std::string::npos);
}
