// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit tests for the Phase-1 structural Matcher: single-node binding, linear
// chain matching with a shared merge variable, injective node mapping, the
// all-or-nothing binding invariant on no-match, wrong-opcode / wrong-order /
// too-long negative cases, anchor selection, and the step-budget fail-closed.
// Graphs are built host-only with the test-SDK builders.

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;

// --- Single-node match ----------------------------------------------------

// createValidMatmulGraph wires a=1, b=2 -> c=3 on the one matmul node. The
// pattern's operand/result vars must bind to exactly those UIDs.
TEST(Matcher, SingleNodeBindsAllVarsToCorrectUids) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // Interning order: $a=0, $b=1, $c=2.
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_FALSE(r.budgetExceeded);
    EXPECT_EQ(r.uidOf(0), 1);  // $a -> A
    EXPECT_EQ(r.uidOf(1), 2);  // $b -> B
    EXPECT_EQ(r.uidOf(2), 3);  // $c -> C
    ASSERT_EQ(r.nodeMap.size(), 1u);
    EXPECT_EQ(r.nodeMap[0], 0u);
}

// --- Linear chain match ---------------------------------------------------

// createValidMatmulBiasActivGraph: node0 matmul a=1,b=2 -> c=3; node1 bias
// pointwise in_0=3,in_1=4 -> out_0=5. The chain pattern shares $h across the
// matmul result and the pointwise operand -- that shared var IS the edge, so $h
// must bind to the matmul output UID (3), and the two pattern nodes map to
// distinct graph nodes (matmul=0, the bias pointwise=1).
TEST(Matcher, LinearChainBindsMergeVarAndMapsNodesInjectively) {
    auto builder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // Interning: $a=0, $b=1, $h=2, $out=3.
    const uint32_t mm = pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    const uint32_t pw = pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$out"}});
    const gm::CompiledPattern pattern = pb.build();
    ASSERT_EQ(pattern.anchor(), pw);  // sink is the pointwise node

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(2), 3);  // $h == matmul output UID
    EXPECT_EQ(r.uidOf(0), 1);  // $a
    EXPECT_EQ(r.uidOf(1), 2);  // $b
    EXPECT_EQ(r.uidOf(3), 5);  // $out == bias output UID

    ASSERT_EQ(r.nodeMap.size(), 2u);
    EXPECT_EQ(r.nodeMap[mm], 0u);             // matmul pattern node -> graph node 0
    EXPECT_EQ(r.nodeMap[pw], 1u);             // pointwise pattern node -> graph node 1
    EXPECT_NE(r.nodeMap[mm], r.nodeMap[pw]);  // injective
}

// Selecting the source (matmul) as the anchor still finds the same match: the
// search walks the shared var the other direction.
TEST(Matcher, MatchesRegardlessOfAnchorEnd) {
    auto builder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    const uint32_t mm = pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    const uint32_t pw = pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$out"}});
    const gm::CompiledPattern pattern = pb.setAnchor(mm).build();
    ASSERT_EQ(pattern.anchor(), mm);

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(2), 3);  // $h still the matmul output
    EXPECT_EQ(r.nodeMap[mm], 0u);
    EXPECT_EQ(r.nodeMap[pw], 1u);
}

// --- No-match: all-or-nothing binding invariant --------------------------

// A conv_fwd pattern has no candidate node in a matmul-only graph, so the match
// fails and NO partial bindings leak out.
TEST(Matcher, NoMatchLeavesBindingsEmpty) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("conv_fwd", {{"x", "$x"}, {"w", "$w"}}, {{"y", "$y"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(r.matched);
    EXPECT_FALSE(r.budgetExceeded);
    EXPECT_TRUE(r.varUids.empty());
    EXPECT_TRUE(r.nodeMap.empty());
    EXPECT_EQ(r.uidOf(0), -1);  // unbound
}

// The chain in the wrong direction (pointwise produces $h, matmul consumes it)
// does not exist in the graph: the matmul's operand has no pointwise producer.
TEST(Matcher, WrongOrderChainDoesNotMatch) {
    auto builder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("pointwise", {{"in_0", "$p"}}, {{"out_0", "$h"}});
    pb.addNode("matmul", {{"a", "$h"}, {"b", "$b"}}, {{"c", "$c"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(r.matched);
    EXPECT_TRUE(r.nodeMap.empty());
}

// A two-node chain cannot match a single-node graph: the second opcode has no
// candidate at all.
TEST(Matcher, PatternLongerThanGraphFails) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$out"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(r.matched);
    EXPECT_FALSE(r.budgetExceeded);
    EXPECT_TRUE(r.nodeMap.empty());
}

// --- Step budget fail-closed ---------------------------------------------

// A pointwise pattern requiring the absent in_2 operand never unifies against
// either pointwise node (bias/activation). With the budget generous this is a
// clean no-match; a budget of 1 aborts before exhausting the two candidates and
// reports budgetExceeded -- distinguishing "gave up" from "no match".
TEST(Matcher, TinyStepBudgetTripsBudgetExceeded) {
    auto builder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // in_2 is a real pointwise role but absent from every node in this graph,
    // so unify fails on each of the two candidate pointwise nodes.
    pb.addNode("pointwise", {{"in_0", "$x"}, {"in_2", "$z"}}, {{"out_0", "$y"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult generous = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(generous.matched);
    EXPECT_FALSE(generous.budgetExceeded);  // exhausted the space cleanly

    const gm::MatchResult tight = gm::Matcher::match(pattern, view, gm::MatchOptions{1});
    EXPECT_FALSE(tight.matched);
    EXPECT_TRUE(tight.budgetExceeded);  // aborted on the budget
    EXPECT_TRUE(tight.nodeMap.empty());
}

}  // namespace
