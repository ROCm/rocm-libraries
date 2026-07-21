// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Phase-2 Matcher tests: true DAG matching (fan-out, diamond/merge,
// multi-output, self-consume, and a diamond-vs-chain negative) plus symbolic
// dimension unification (PatternBuilder::bindDim / SymId first-bind-then-verify
// across edges). Graphs whose shape the test-SDK builders do not provide are
// built inline with the flatbuffers Direct builders. Phases 0/1 (OpSchema,
// GraphView, single-node / linear-chain structural matching, builder
// validation) are covered elsewhere and deliberately not retested here.

#include <gtest/gtest.h>

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <stdexcept>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;
namespace data = hipdnn_flatbuffers_sdk::data_objects;

// --- Inline graph-building helpers ---------------------------------------

// A FLOAT tensor with row-major contiguous strides derived from `dims`. Every
// inline UID gets one of these so it appears in the graph's tensor map (a UID is
// an edge only if the map holds it) and carries real dims for symbol tests.
flatbuffers::Offset<data::TensorAttributes> tensorT(flatbuffers::FlatBufferBuilder& b, int64_t uid,
                                                    const char* name,
                                                    const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return data::CreateTensorAttributesDirect(b, uid, name, data::DataType::FLOAT, &strides, &dims);
}

// A pointwise (ADD) node: in_0 required, in_1 optional, out_0 required. Absent
// in_1 (nullopt) means the node has a single operand edge.
flatbuffers::Offset<data::Node> pwNode(flatbuffers::FlatBufferBuilder& b, int64_t in0, int64_t out,
                                       flatbuffers::Optional<int64_t> in1 = flatbuffers::nullopt) {
    auto attrs = data::CreatePointwiseAttributes(b, data::PointwiseMode::ADD,
                                                 flatbuffers::nullopt,  // relu_lower_clip
                                                 flatbuffers::nullopt,  // relu_upper_clip
                                                 flatbuffers::nullopt,  // relu_lower_clip_slope
                                                 flatbuffers::nullopt,  // axis_tensor_uid
                                                 in0, in1,
                                                 flatbuffers::nullopt,  // in_2_tensor_uid
                                                 out);
    return data::CreateNodeDirect(b, "pw", data::DataType::FLOAT,
                                  data::NodeAttributes::PointwiseAttributes, attrs.Union());
}

flatbuffers::Offset<data::Graph> finishGraph(
    flatbuffers::FlatBufferBuilder& b,
    std::vector<flatbuffers::Offset<data::TensorAttributes>>& tensors,
    std::vector<flatbuffers::Offset<data::Node>>& nodes) {
    return data::CreateGraphDirect(b, "test", data::DataType::FLOAT, data::DataType::FLOAT,
                                   data::DataType::FLOAT, &tensors, &nodes);
}

// Fan-out: one input tensor (uid 1) feeds two independent pointwise nodes,
// producing uid 2 (node 0) and uid 3 (node 1).
flatbuffers::FlatBufferBuilder makeFanOutGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "x", {4, 4}), tensorT(b, 2, "y0", {4, 4}), tensorT(b, 3, "y1", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2), pwNode(b, 1, 3)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Diamond: A(node0) x=1 -> h=2; B(node1) h=2 -> p=3; C(node2) h=2 -> q=4;
// D(node3) in_0=p=3, in_1=q=4 -> out=5.
flatbuffers::FlatBufferBuilder makeDiamondGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "ain", {4, 4}), tensorT(b, 2, "h", {4, 4}), tensorT(b, 3, "p", {4, 4}),
        tensorT(b, 4, "q", {4, 4}), tensorT(b, 5, "out", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2), pwNode(b, 2, 3),
                                                   pwNode(b, 2, 4), pwNode(b, 3, 5, /*in1=*/4)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Multi-output producer: block_scale_quantize (node0) x=1 -> y=2, scale=3; each
// result feeds a distinct pointwise consumer (y->node1 out=4, scale->node2
// out=5).
flatbuffers::FlatBufferBuilder makeMultiOutputGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "x", {4, 4}), tensorT(b, 2, "y", {4, 4}), tensorT(b, 3, "scale", {4, 4}),
        tensorT(b, 4, "o0", {4, 4}), tensorT(b, 5, "o1", {4, 4})};
    auto bsq =
        data::CreateBlockScaleQuantizeAttributes(b, /*x=*/1, /*y=*/2, /*scale=*/3,
                                                 /*block_size=*/32, /*axis=*/flatbuffers::nullopt,
                                                 /*transpose=*/false);
    std::vector<flatbuffers::Offset<data::Node>> n{
        data::CreateNodeDirect(b, "bsq", data::DataType::FLOAT,
                               data::NodeAttributes::BlockScaleQuantizeAttributes, bsq.Union()),
        pwNode(b, 2, 4), pwNode(b, 3, 5)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Self-consume: one pointwise node with in_0 == in_1 == uid 1 -> out uid 2.
flatbuffers::FlatBufferBuilder makeSelfConsumeGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{tensorT(b, 1, "h", {4, 4}),
                                                               tensorT(b, 2, "out", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2, /*in1=*/1)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Linear chain of two pointwise nodes: 1 -> 2 (node0) -> 3 (node1).
flatbuffers::FlatBufferBuilder makeLinearChainGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "x", {4, 4}), tensorT(b, 2, "m", {4, 4}), tensorT(b, 3, "y", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2), pwNode(b, 2, 3)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// matmul (node0) a=1[2,3], b=2[3,4] -> c=3[2,4]; pointwise (node1) in_0=3 ->
// out=4[2,4]. Dims chosen so cross-node symbol tests have known values.
flatbuffers::FlatBufferBuilder makeMatmulPwChainGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "a", {2, 3}), tensorT(b, 2, "b", {3, 4}), tensorT(b, 3, "c", {2, 4}),
        tensorT(b, 4, "o", {2, 4})};
    auto mm = data::CreateMatmulAttributes(b, 1, 2, 3);
    std::vector<flatbuffers::Offset<data::Node>> n{
        data::CreateNodeDirect(b, "matmul", data::DataType::FLOAT,
                               data::NodeAttributes::MatmulAttributes, mm.Union()),
        pwNode(b, 3, 4)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// =========================================================================
// DAG structure
// =========================================================================

// Fan-out: the shared input var $x binds the single shared UID (1) while the two
// pattern nodes map to the two distinct consumers with distinct output UIDs, and
// the node map stays injective.
TEST(MatcherDag, FanOutSharesInputVarAcrossTwoConsumers) {
    auto builder = makeFanOutGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // Interning: $x=0, $y0=1, $y1=2.
    const uint32_t p0 = pb.addNode("pointwise", {{"in_0", "$x"}}, {{"out_0", "$y0"}});
    const uint32_t p1 = pb.addNode("pointwise", {{"in_0", "$x"}}, {{"out_0", "$y1"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(0), 1);           // $x -> the one shared input UID
    EXPECT_NE(r.uidOf(1), r.uidOf(2));  // the two outputs are distinct tensors
    EXPECT_EQ(r.uidOf(1), 2);           // $y0
    EXPECT_EQ(r.uidOf(2), 3);           // $y1
    ASSERT_EQ(r.nodeMap.size(), 2u);
    EXPECT_NE(r.nodeMap[p0], r.nodeMap[p1]);  // injective: two graph nodes
    EXPECT_EQ(r.nodeMap[p0], 0u);
    EXPECT_EQ(r.nodeMap[p1], 1u);
}

// Diamond/merge: A produces $h consumed by both B and C; D merges $p and $q.
// Full match binds every edge to its built UID, D's two operands unify to the
// two distinct intermediate UIDs (3 and 4), and the node map is injective.
TEST(MatcherDag, DiamondMergeBindsBranchesAndMerge) {
    auto builder = makeDiamondGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // Interning: $ain=0, $h=1, $p=2, $q=3, $out=4.
    const uint32_t a = pb.addNode("pointwise", {{"in_0", "$ain"}}, {{"out_0", "$h"}});
    const uint32_t bB = pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$p"}});
    const uint32_t c = pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$q"}});
    const uint32_t d =
        pb.addNode("pointwise", {{"in_0", "$p"}, {"in_1", "$q"}}, {{"out_0", "$out"}});
    const gm::CompiledPattern pattern = pb.build();
    ASSERT_EQ(pattern.anchor(), d);  // D is the unique sink

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(0), 1);           // $ain
    EXPECT_EQ(r.uidOf(1), 2);           // $h  (the merge point feeding both branches)
    EXPECT_EQ(r.uidOf(2), 3);           // $p  (B's output = D's in_0)
    EXPECT_EQ(r.uidOf(3), 4);           // $q  (C's output = D's in_1)
    EXPECT_EQ(r.uidOf(4), 5);           // $out
    EXPECT_NE(r.uidOf(2), r.uidOf(3));  // D's two operands are distinct edges

    ASSERT_EQ(r.nodeMap.size(), 4u);
    EXPECT_EQ(r.nodeMap[a], 0u);
    EXPECT_EQ(r.nodeMap[bB], 1u);  // in_0/in_1 role order on D pins B and C
    EXPECT_EQ(r.nodeMap[c], 2u);
    EXPECT_EQ(r.nodeMap[d], 3u);
}

// Multi-output node: block_scale_quantize produces two results, each consumed by
// a separate pointwise node. The pattern binds both result vars and the two
// consumer nodes map injectively.
TEST(MatcherDag, MultiOutputBindsBothResultVars) {
    auto builder = makeMultiOutputGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // Interning: $x=0, $y=1, $s=2, $o0=3, $o1=4.
    const uint32_t q =
        pb.addNode("block_scale_quantize", {{"x", "$x"}}, {{"y", "$y"}, {"scale", "$s"}});
    const uint32_t c0 = pb.addNode("pointwise", {{"in_0", "$y"}}, {{"out_0", "$o0"}});
    const uint32_t c1 = pb.addNode("pointwise", {{"in_0", "$s"}}, {{"out_0", "$o1"}});
    const gm::CompiledPattern pattern = pb.build();
    ASSERT_EQ(pattern.anchor(), q);  // producer is the only non-sink; two sinks -> node 0

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(1), 2);  // $y   -- first result edge
    EXPECT_EQ(r.uidOf(2), 3);  // $s   -- second result edge, distinctly bound
    EXPECT_NE(r.uidOf(1), r.uidOf(2));
    EXPECT_EQ(r.uidOf(3), 4);  // $o0
    EXPECT_EQ(r.uidOf(4), 5);  // $o1
    ASSERT_EQ(r.nodeMap.size(), 3u);
    EXPECT_EQ(r.nodeMap[q], 0u);
    EXPECT_EQ(r.nodeMap[c0], 1u);
    EXPECT_EQ(r.nodeMap[c1], 2u);
}

// Self-consume, single var: pointwise(in_0=$h, in_1=$h) matches the node whose
// in_0 == in_1 == same UID; the shared var unifies to that one UID.
TEST(MatcherDag, SelfConsumeSameVarUnifiesToOneUid) {
    auto builder = makeSelfConsumeGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("pointwise", {{"in_0", "$h"}, {"in_1", "$h"}}, {{"out_0", "$out"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(0), 1);  // $h -- both operand slots resolve to the same UID
    EXPECT_EQ(r.uidOf(1), 2);  // $out
}

// Self-consume, two DISTINCT vars: pointwise(in_0=$h, in_1=$g). The matcher does
// NOT require pattern variables to be distinct, so both $h and $g bind the same
// graph UID and the node still matches. (Impl fact: unifyEdge binds each var
// independently; there is no all-different constraint.)
TEST(MatcherDag, DistinctVarsMayBindSameUid) {
    auto builder = makeSelfConsumeGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // Interning: $h=0, $g=1, $out=2.
    pb.addNode("pointwise", {{"in_0", "$h"}, {"in_1", "$g"}}, {{"out_0", "$out"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(0), 1);  // $h
    EXPECT_EQ(r.uidOf(1), 1);  // $g -- same UID as $h; distinct vars, same tensor
    EXPECT_EQ(r.uidOf(0), r.uidOf(1));
    EXPECT_EQ(r.uidOf(2), 2);  // $out
}

// Negative: a diamond pattern must NOT match a linear chain -- the chain has no
// second branch for D's second operand to bind.
TEST(MatcherDag, DiamondDoesNotMatchLinearChain) {
    auto builder = makeLinearChainGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("pointwise", {{"in_0", "$ain"}}, {{"out_0", "$h"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$p"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$q"}});
    pb.addNode("pointwise", {{"in_0", "$p"}, {"in_1", "$q"}}, {{"out_0", "$out"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(r.matched);
    EXPECT_TRUE(r.nodeMap.empty());
    EXPECT_TRUE(r.varUids.empty());
}

// =========================================================================
// Symbolic dimension unification
// =========================================================================

// Positive unify across two operand edges: a[4,8].axis1 and b[8,5].axis0 both
// name "k"; they agree on 8, so the match succeeds and the symbol binds 8.
TEST(MatcherSymbols, UnifiesSharedContractionDim) {
    auto builder = util::createValidMatmulGraph();  // a[4,8], b[8,5], c[4,5]
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.bindDim("$a", 1, "k");  // a.dims[1] == 8
    pb.bindDim("$b", 0, "k");  // b.dims[0] == 8
    const gm::CompiledPattern pattern = pb.build();
    ASSERT_EQ(pattern.symCount(), 1u);

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    ASSERT_EQ(r.symVals.size(), 1u);
    EXPECT_EQ(r.symOf(0), 8);  // "k"
}

// Conflict: a[4,8].axis0 (==4) and b[8,5].axis0 (==8) both name "k"; the values
// disagree, so the whole match fails and -- all-or-nothing -- symVals AND
// varUids are empty, symOf/uidOf return -1.
TEST(MatcherSymbols, ConflictFailsAllOrNothing) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.bindDim("$a", 0, "k");  // 4
    pb.bindDim("$b", 0, "k");  // 8 -- conflict
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(r.matched);
    EXPECT_FALSE(r.budgetExceeded);
    EXPECT_TRUE(r.symVals.empty());
    EXPECT_TRUE(r.varUids.empty());
    EXPECT_EQ(r.symOf(0), -1);
    EXPECT_EQ(r.uidOf(0), -1);
}

// Cross-node symbol over an edge: bind "m" on $a (matmul operand, node0) and on
// $o (pointwise result, node1). With matching dims (both == 2) it unifies and
// binds; with differing dims it conflicts and the match fails.
TEST(MatcherSymbols, UnifiesSymbolAcrossNodes) {
    auto builder = makeMatmulPwChainGraph();  // a[2,3] -> c[2,4] -> o[2,4]
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$o"}});
    pb.bindDim("$a", 0, "m");  // a.dims[0] == 2
    pb.bindDim("$o", 0, "m");  // o.dims[0] == 2  -> agree
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.symOf(0), 2);  // "m"
}

TEST(MatcherSymbols, CrossNodeSymbolConflictFails) {
    auto builder = makeMatmulPwChainGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$h"}});
    pb.addNode("pointwise", {{"in_0", "$h"}}, {{"out_0", "$o"}});
    pb.bindDim("$a", 1, "m");  // a.dims[1] == 3
    pb.bindDim("$o", 0, "m");  // o.dims[0] == 2  -> conflict
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(r.matched);
    EXPECT_TRUE(r.symVals.empty());
}

// Out-of-range axis fails closed: a[4,8] has rank 2, so asking for axis 99 makes
// the match fail rather than reading past the tensor.
TEST(MatcherSymbols, OutOfRangeAxisFailsClosed) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.bindDim("$a", 99, "k");  // no such axis
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(r.matched);
    EXPECT_FALSE(r.budgetExceeded);
    EXPECT_TRUE(r.varUids.empty());
}

// bindDim on a variable that never appeared on an edge throws.
TEST(MatcherSymbols, BindDimUnknownVarThrows) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    EXPECT_THROW(pb.bindDim("$z", 0, "k"), std::invalid_argument);
}

// Symbols are interned in first-use order: symCount reflects the number of
// distinct names and symName maps SymId -> name in that order.
TEST(MatcherSymbols, SymbolInterningOrder) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.bindDim("$a", 1, "k");  // first-use -> SymId 0
    pb.bindDim("$b", 0, "n");  // second distinct -> SymId 1
    pb.bindDim("$b", 1, "k");  // reuse "k" -> still SymId 0, no new symbol
    const gm::CompiledPattern pattern = pb.build();

    ASSERT_EQ(pattern.symCount(), 2u);
    EXPECT_EQ(pattern.symName(0), "k");
    EXPECT_EQ(pattern.symName(1), "n");
}

}  // namespace
