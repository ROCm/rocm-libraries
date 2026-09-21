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

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
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
