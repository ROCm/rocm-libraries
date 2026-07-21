// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit tests for PatternBuilder / CompiledPattern (Phase 1): node index
// assignment, variable interning order, the validation throws (unknown opcode,
// unknown role, out-of-range anchor, empty pattern, disconnected pattern), and
// anchor resolution (unique sink by default, node 0 when ambiguous, explicit
// override). Role names are the lowercase fbs field bases per OpSchema.

#include <gtest/gtest.h>

#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <stdexcept>
#include <string_view>

namespace {

namespace gm = hipdnn::graph_matcher;

using EdgeSpec = gm::PatternBuilder::EdgeSpec;

// --- Node index assignment -----------------------------------------------

TEST(PatternBuilder, AddNodeReturnsIncreasingIndices) {
    gm::PatternBuilder pb;
    const uint32_t first = pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    const uint32_t second = pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$out"}});
    EXPECT_EQ(first, 0u);
    EXPECT_EQ(second, 1u);
}

// --- Validation throws on addNode ----------------------------------------

TEST(PatternBuilder, UnknownOpcodeThrows) {
    gm::PatternBuilder pb;
    EXPECT_THROW(pb.addNode("not_a_real_op", {}, {}), std::invalid_argument);
}

// The RFC names roles uppercase (A/B/C); the impl uses the lowercase fbs field
// bases. An uppercase role must therefore be rejected.
TEST(PatternBuilder, UnknownRoleThrows) {
    gm::PatternBuilder pb;
    EXPECT_THROW(pb.addNode("matmul", {{"A", "$a"}, {"b", "$b"}}, {{"c", "$c"}}),
                 std::invalid_argument);
}

TEST(PatternBuilder, UnknownResultRoleThrows) {
    gm::PatternBuilder pb;
    // "y" is conv_fwd's result role, not matmul's ("c").
    EXPECT_THROW(pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"y", "$c"}}),
                 std::invalid_argument);
}

// --- Variable interning order --------------------------------------------

TEST(PatternBuilder, VariablesInternInFirstUseOrder) {
    gm::PatternBuilder pb;
    // First-use order: $x, $y, $z, then $w (introduced by the pointwise result).
    pb.addNode("matmul", {{"a", "$x"}, {"b", "$y"}}, {{"c", "$z"}});
    pb.addNode("pointwise", {{"in_0", "$z"}}, {{"out_0", "$w"}});
    const gm::CompiledPattern pattern = pb.build();

    EXPECT_EQ(pattern.varCount(), 4u);
    EXPECT_EQ(pattern.varName(0), std::string_view{"$x"});
    EXPECT_EQ(pattern.varName(1), std::string_view{"$y"});
    EXPECT_EQ(pattern.varName(2), std::string_view{"$z"});
    EXPECT_EQ(pattern.varName(3), std::string_view{"$w"});
}

TEST(PatternBuilder, RepeatedVariableIsNotReinterned) {
    gm::PatternBuilder pb;
    // $h reused across both operand slots of one matmul: still one VarId.
    pb.addNode("matmul", {{"a", "$h"}, {"b", "$h"}}, {{"c", "$out"}});
    const gm::CompiledPattern pattern = pb.build();

    EXPECT_EQ(pattern.varCount(), 2u);
    EXPECT_EQ(pattern.varName(0), std::string_view{"$h"});
    EXPECT_EQ(pattern.varName(1), std::string_view{"$out"});
}

// --- Anchor resolution ----------------------------------------------------

// A matmul->pointwise chain: the pointwise result is unconsumed (the unique
// sink), so the default anchor is the pointwise node (index 1), NOT the matmul.
TEST(PatternBuilder, AnchorDefaultsToUniqueSink) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$out"}});
    const gm::CompiledPattern pattern = pb.build();
    EXPECT_EQ(pattern.anchor(), 1u);
}

// Fan-out: one matmul feeding two pointwise consumers gives two sinks, so the
// sink is ambiguous and the anchor falls back to node 0.
TEST(PatternBuilder, AnchorFallsBackToNodeZeroWhenSinkAmbiguous) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$o1"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$o2"}});
    const gm::CompiledPattern pattern = pb.build();
    EXPECT_EQ(pattern.anchor(), 0u);
}

TEST(PatternBuilder, SetAnchorOverridesDefault) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$out"}});
    // Without the override this would resolve to the sink (index 1).
    const gm::CompiledPattern pattern = pb.setAnchor(0).build();
    EXPECT_EQ(pattern.anchor(), 0u);
}

TEST(PatternBuilder, SetAnchorOutOfRangeThrows) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.setAnchor(5);
    EXPECT_THROW(pb.build(), std::invalid_argument);
}

// --- Validation throws on build ------------------------------------------

TEST(PatternBuilder, EmptyPatternThrows) {
    gm::PatternBuilder pb;
    EXPECT_THROW(pb.build(), std::invalid_argument);
}

// Two nodes sharing no variable cannot be reached from a single anchor via
// var-adjacency, so the matcher could never walk to the second one.
TEST(PatternBuilder, DisconnectedPatternThrows) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.addNode("reduction", {{"in", "$x"}}, {{"out", "$y"}});
    EXPECT_THROW(pb.build(), std::invalid_argument);
}

// A shared variable IS the edge, so a chain with a joining var is connected and
// builds cleanly.
TEST(PatternBuilder, ConnectedChainBuilds) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    pb.addNode("reduction", {{"in", "$h"}}, {{"out", "$y"}});
    const gm::CompiledPattern pattern = pb.build();
    EXPECT_EQ(pattern.nodeCount(), 2u);
}

}  // namespace
