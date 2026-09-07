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

#include <algorithm>
#include <memory>
#include <span>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class AllocationConstraintsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    /// s40 = s_mov_b32(s\p source), where \p source is read before anything
    /// writes it and so lifts to a live-in. Returns its value ID.
    SSAValueID scalarLiveIn(BasicBlock& entry, uint32_t source) {
        AsmIRBuilder builder(entry, kRaTestArch);
        StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
        mov->addDestReg(StinkyRegister("s", 40, 1));
        mov->addSrcReg(StinkyRegister("s", source, 1));
        if (!liftForAllocation(*func)) return kInvalidSSAValueID;
        const StinkySSAValue* value = ssaSourceValue(*mov, 0);
        return value == nullptr ? kInvalidSSAValueID : value->valueId();
    }

    std::unique_ptr<Function> func;
};

bool hasTuple(const AllocationConstraints& constraints, const std::vector<SSAValueID>& units) {
    for (const TupleRun& run : constraints.tupleRuns()) {
        if (run.units == units) return true;
    }
    return false;
}

bool hasAffinity(const AllocationConstraints& constraints, SSAValueID id) {
    for (const AffinitySet& set : constraints.affinitySets()) {
        if (std::find(set.members.begin(), set.members.end(), id) != set.members.end()) return true;
    }
    return false;
}

bool isUndefinedLiveIn(const AllocationConstraints& constraints, SSAValueID id) {
    const std::span<const SSAValueID> undefined = constraints.undefinedLiveIns();
    return std::find(undefined.begin(), undefined.end(), id) != undefined.end();
}

/// s0-s31, what .amdhsa_user_sgpr_count 29 plus three workgroup ids fills.
constexpr uint64_t kDispatchFills = 32;

}  // namespace

TEST_F(AllocationConstraintsTest, HintIsThePhysicalBinding) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const StinkySSAValue* result = ssaDefinedValue(*add);
    ASSERT_NE(result, nullptr);

    const std::optional<RegKey> hint = setup.constraints().hintFor(result->valueId());
    ASSERT_TRUE(hint.has_value());
    EXPECT_EQ(*hint, (RegKey{RegType::V, 2, RegHalf::NONE}));
    EXPECT_EQ(setup.constraints().classOf(result->valueId()), RegType::V);
    EXPECT_TRUE(setup.constraints().isAllocatable(result->valueId()));
    EXPECT_TRUE(setup.constraints().tupleRuns().empty());
    EXPECT_TRUE(setup.constraints().affinitySets().empty());
}

TEST_F(AllocationConstraintsTest, MultiDwordOperandIsATupleRun) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kRaTestArch, 10, 4);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    const std::vector<StinkySSAValue*> units = ssaDestUnits(*load, 0);
    ASSERT_EQ(units.size(), 4u);

    std::vector<SSAValueID> ids;
    ids.reserve(units.size());
    for (StinkySSAValue* unit : units) ids.push_back(unit->valueId());
    EXPECT_TRUE(hasTuple(setup.constraints(), ids)) << setup.constraints().toString();
}

TEST_F(AllocationConstraintsTest, MergeIsAnAffinitySet) {
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

    AllocationSetup setup(*func);
    const SSABlockArgument* arg = vgprArgumentFor(*join, 5);
    ASSERT_NE(arg, nullptr);
    ASSERT_NE(arg->value, nullptr);

    EXPECT_TRUE(hasAffinity(setup.constraints(), arg->value->valueId()))
        << setup.constraints().toString();
    EXPECT_FALSE(setup.constraints().affinitySets().empty());
    EXPECT_GE(setup.constraints().affinitySets().front().members.size(), 2u);
}

TEST_F(AllocationConstraintsTest, LiveInTheDispatchFilledIsPinned) {
    func->setMetaData(kSigDispatchFilledSgprsMetaKey, kDispatchFills);
    const SSAValueID liveIn = scalarLiveIn(*block("entry"), /*source=*/8);
    ASSERT_NE(liveIn, kInvalidSSAValueID);

    AllocationSetup setup(*func, RegClassSet::all());
    EXPECT_TRUE(setup.constraints().isPinned(liveIn)) << setup.constraints().toString();
    EXPECT_FALSE(isUndefinedLiveIn(setup.constraints(), liveIn));
}

TEST_F(AllocationConstraintsTest, LiveInAboveWhatTheDispatchFillsIsUndefinedAndFree) {
    // Nothing wrote s100, so it holds nothing and any register serves it equally.
    func->setMetaData(kSigDispatchFilledSgprsMetaKey, kDispatchFills);
    const SSAValueID liveIn = scalarLiveIn(*block("entry"), /*source=*/100);
    ASSERT_NE(liveIn, kInvalidSSAValueID);

    AllocationSetup setup(*func, RegClassSet::all());
    EXPECT_FALSE(setup.constraints().isPinned(liveIn)) << setup.constraints().toString();
    EXPECT_TRUE(isUndefinedLiveIn(setup.constraints(), liveIn)) << setup.constraints().toString();
}

TEST_F(AllocationConstraintsTest, WithoutTheBoundaryEveryLiveInStaysPinned) {
    // No metadata, as a .stir file or this suite leaves it. Unknown pins: not
    // knowing what the dispatch filled is no licence to move a register it did.
    const SSAValueID liveIn = scalarLiveIn(*block("entry"), /*source=*/100);
    ASSERT_NE(liveIn, kInvalidSSAValueID);

    AllocationSetup setup(*func, RegClassSet::all());
    EXPECT_TRUE(setup.constraints().isPinned(liveIn)) << setup.constraints().toString();
    EXPECT_TRUE(setup.constraints().undefinedLiveIns().empty());
}
