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

#include <climits>
#include <unordered_map>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "transforms/asm/dag/planning/FillerDeadlinePlanner.hpp"

using namespace stinkytofu;
using namespace stinkytofu::dag;
using namespace stinkytofu::test;

namespace {

class FillerDeadlinePlannerTest : public ::testing::Test {
   protected:
    Function func{"filler_deadline_planner_test"};
    BasicBlock* bb = func.createBasicBlock("entry");
    RegionDAG dag;
    std::unordered_map<const StinkyInstruction*, int> prefetchEarliest;
    std::unordered_map<const StinkyInstruction*, int> loadWindow;

    FillerDeadlinePlannerTest() {
        setFunctionArch(func, GfxArchID::Gfx1250);
    }

    unsigned add(GFX op) {
        AsmIRBuilder builder(*bb, GfxArchID::Gfx1250);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(op, GfxArchID::Gfx1250));
        const unsigned id = static_cast<unsigned>(dag.nodes.size());
        dag.nodes.emplace_back(inst, id);
        dag.graph.emplace_back();
        dag.instToId[inst] = id;
        return id;
    }

    void edge(unsigned from, unsigned to) {
        dag.graph[from].insert(to);
    }

    const StinkyInstruction* inst(unsigned id) const {
        return dag.nodes[id].inst;
    }

    std::vector<int> plan(int dsTotal = 0, int wmmaTotal = 0) {
        return planFillerDeadlines(dag, prefetchEarliest, loadWindow, dsTotal, wmmaTotal);
    }
};

TEST_F(FillerDeadlinePlannerTest, FillerFeedingPrefetchTakesItsWindow) {
    const unsigned filler = add(GFX::v_mov_b32);
    const unsigned prefetch = add(GFX::global_prefetch_b8);
    edge(filler, prefetch);
    prefetchEarliest[inst(prefetch)] = 5;
    const auto need = plan();
    EXPECT_EQ(need[filler], 5);
    EXPECT_EQ(need[prefetch], INT_MAX);
}

TEST_F(FillerDeadlinePlannerTest, ChainSubtractsOneWindowPerHop) {
    const unsigned first = add(GFX::s_mov_b32);
    const unsigned second = add(GFX::v_mov_b32);
    const unsigned load = add(GFX::tensor_load_to_lds);
    edge(first, second);
    edge(second, load);
    loadWindow[inst(load)] = 7;
    const auto need = plan();
    EXPECT_EQ(need[second], 7);
    EXPECT_EQ(need[first], 6);
}

TEST_F(FillerDeadlinePlannerTest, DsReadsArePlannedAtRegionPace) {
    const unsigned a = add(GFX::v_mov_b32);
    const unsigned b = add(GFX::v_mov_b32);
    const unsigned ds0 = add(GFX::ds_load_b32);
    const unsigned ds1 = add(GFX::ds_load_b32);
    edge(a, ds1);
    edge(b, ds0);
    const auto need = plan(/*dsTotal=*/2, /*wmmaTotal=*/8);
    EXPECT_EQ(need[a], 4);
    EXPECT_EQ(need[b], 0);
}

TEST_F(FillerDeadlinePlannerTest, UnplannedPrefetchDefaultsToWindowZero) {
    const unsigned filler = add(GFX::v_mov_b32);
    const unsigned prefetch = add(GFX::global_prefetch_b8);
    edge(filler, prefetch);
    EXPECT_EQ(plan()[filler], 0);
}

TEST_F(FillerDeadlinePlannerTest, EarliestConsumerWins) {
    const unsigned filler = add(GFX::v_mov_b32);
    const unsigned prefetch = add(GFX::global_prefetch_b8);
    const unsigned load = add(GFX::tensor_load_to_lds);
    edge(filler, prefetch);
    edge(filler, load);
    prefetchEarliest[inst(prefetch)] = 9;
    loadWindow[inst(load)] = 3;
    EXPECT_EQ(plan()[filler], 3);
}

TEST_F(FillerDeadlinePlannerTest, NoPlannedConsumerMeansNoDeadline) {
    const unsigned lone = add(GFX::v_mov_b32);
    const unsigned filler = add(GFX::v_mov_b32);
    const unsigned load = add(GFX::tensor_load_to_lds);
    edge(filler, load);
    const auto need = plan();
    EXPECT_EQ(need[lone], INT_MAX);
    EXPECT_EQ(need[filler], INT_MAX);
}

}  // namespace
