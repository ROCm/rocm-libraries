/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * ************************************************************************ */

#include <gtest/gtest.h>

#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "transforms/asm/dag/planning/DsReadOrderPlanner.hpp"

using namespace stinkytofu;
using namespace stinkytofu::dag;
using namespace stinkytofu::test;

namespace {

class DsReadOrderPlannerTest : public ::testing::Test {
   protected:
    Function func{"ds_read_order_planner_test"};
    BasicBlock* bb = func.createBasicBlock("entry");
    RegionDAG dag;

    DsReadOrderPlannerTest() {
        setFunctionArch(func, GfxArchID::Gfx1250);
    }

    // An empty token list means no PSEUDO token; inDag false leaves the load out of the DAG.
    StinkyInstruction* addLoad(const std::vector<int>& tokens, unsigned priority,
                               bool inDag = true) {
        AsmIRBuilder builder(*bb, GfxArchID::Gfx1250);
        StinkyInstruction* inst =
            builder.create(getMCIDByUOp(GFX::ds_load_b32, GfxArchID::Gfx1250));
        inst->addDestReg(StinkyRegister("v", 10 + static_cast<int>(bb->size()), 1));
        inst->addSrcReg(StinkyRegister("v", 0, 1));
        for (int token : tokens) inst->addSrcReg(StinkyRegister(RegType::LDS, token, 1));
        if (!inDag) return inst;
        const unsigned id = static_cast<unsigned>(dag.nodes.size());
        dag.nodes.emplace_back(inst, id);
        dag.nodes.back().dsReadPriority = priority;
        dag.instToId[inst] = id;
        return inst;
    }

    std::vector<HardSchedulingConstraint> plan() {
        return planDsReadOrder(bb->begin(), bb->end(), dag);
    }
};

TEST_F(DsReadOrderPlannerTest, ChainsSameTokenLoadsByPriority) {
    auto* a = addLoad({1}, /*priority=*/2);
    auto* b = addLoad({1}, /*priority=*/0);
    auto* c = addLoad({1}, /*priority=*/1);
    const auto edges = plan();
    ASSERT_EQ(edges.size(), 2u);
    EXPECT_EQ(edges[0], HardSchedulingConstraint(b, c));
    EXPECT_EQ(edges[1], HardSchedulingConstraint(c, a));
}

TEST_F(DsReadOrderPlannerTest, EqualPriorityKeepsDagIdOrder) {
    auto* a = addLoad({1}, 5);
    auto* b = addLoad({1}, 5);
    const auto edges = plan();
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(edges[0], HardSchedulingConstraint(a, b));
}

TEST_F(DsReadOrderPlannerTest, DifferentTokensAreNotChained) {
    addLoad({1}, 0);
    addLoad({2}, 1);
    EXPECT_TRUE(plan().empty());
}

TEST_F(DsReadOrderPlannerTest, LoadWithoutTokenIsUnordered) {
    addLoad({}, 0);
    addLoad({}, 1);
    addLoad({1}, 0);
    EXPECT_TRUE(plan().empty());
}

TEST_F(DsReadOrderPlannerTest, TokenListOrderAndDuplicatesDoNotChangeTheGroup) {
    auto* a = addLoad({1, 2}, 1);
    auto* b = addLoad({2, 1}, 0);
    auto* c = addLoad({1, 1, 2}, 2);
    const auto edges = plan();
    ASSERT_EQ(edges.size(), 2u);
    EXPECT_EQ(edges[0], HardSchedulingConstraint(b, a));
    EXPECT_EQ(edges[1], HardSchedulingConstraint(a, c));
}

TEST_F(DsReadOrderPlannerTest, LoadMissingFromDagSortsLast) {
    auto* missing = addLoad({1}, 0, /*inDag=*/false);
    auto* a = addLoad({1}, 3);
    const auto edges = plan();
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(edges[0], HardSchedulingConstraint(a, missing));
}

}  // namespace
