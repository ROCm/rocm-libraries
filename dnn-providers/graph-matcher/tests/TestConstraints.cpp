// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Phase-3 Matcher tests: the full constraint vocabulary layered on structural
// matching. Each constraint kind is exercised with a positive AND a negative
// case: the positive proves a satisfied constraint keeps the match, the negative
// proves a violated constraint makes the whole match fail (all-or-nothing
// backtracking, empty bindings). Phases 0/1/2 (OpSchema, GraphView, structural /
// DAG matching, symbol unification via bindDim) are covered elsewhere and are
// deliberately not retested here. Graphs the test-SDK builders do not provide are
// built inline with the flatbuffers Direct builders.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
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

// FLOAT/HALF are macros in some Windows headers -- never name locals that.
const int32_t kF32 = static_cast<int32_t>(data::DataType::FLOAT);
const int32_t kF16 = static_cast<int32_t>(data::DataType::HALF);
const int64_t kDiv = static_cast<int64_t>(data::PointwiseMode::DIV);
const int64_t kAdd = static_cast<int64_t>(data::PointwiseMode::ADD);
const int64_t kMul = static_cast<int64_t>(data::PointwiseMode::MUL);

// --- Inline graph-building helpers ---------------------------------------

// A tensor with row-major contiguous strides derived from `dims`. Every inline
// UID gets one so it appears in the graph's tensor map (a UID is an edge only if
// the map holds it) and carries real dims/strides for value constraints.
flatbuffers::Offset<data::TensorAttributes> tensorT(flatbuffers::FlatBufferBuilder& b, int64_t uid,
                                                    const char* name,
                                                    const std::vector<int64_t>& dims,
                                                    data::DataType dtype = data::DataType::FLOAT) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return data::CreateTensorAttributesDirect(b, uid, name, dtype, &strides, &dims);
}

// A pointwise (ADD) node: in_0 required, in_1 optional, out_0 required. Absent
// in_1 (nullopt) => a single operand edge. Opcode resolves to "pointwise" via
// the attributes type regardless of the node's name string.
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

// Fan-out: input tensor uid 1 feeds two pointwise nodes -> useCount(1)==2,
// consumerNodeCount(1)==2.
flatbuffers::FlatBufferBuilder makeFanOutGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "x", {4, 4}), tensorT(b, 2, "y0", {4, 4}), tensorT(b, 3, "y1", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2), pwNode(b, 1, 3)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Self-consume: one pointwise node with in_0 == in_1 == uid 1 -> out uid 2, so
// useCount(1)==2 but consumerNodeCount(1)==1 (the load-bearing distinction).
flatbuffers::FlatBufferBuilder makeSelfConsumeGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{tensorT(b, 1, "h", {4, 4}),
                                                               tensorT(b, 2, "out", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2, /*in1=*/1)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Linear chain 1 -> 2 (node0) -> 3 (node1): uid 2's only consumer is node1.
flatbuffers::FlatBufferBuilder makeLinearChainGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "x", {4, 4}), tensorT(b, 2, "m", {4, 4}), tensorT(b, 3, "y", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2), pwNode(b, 2, 3)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Shared intermediate: node0 (1->2); node1 (2->3) and node2 (2->4) both consume
// uid 2. A two-node chain pattern matches node0 + one consumer, leaving the
// other consumer unmatched -> a NoConsumerOutside constraint on uid 2 fails.
flatbuffers::FlatBufferBuilder makeSharedIntermediateGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "x", {4, 4}), tensorT(b, 2, "m", {4, 4}), tensorT(b, 3, "y0", {4, 4}),
        tensorT(b, 4, "y1", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2), pwNode(b, 2, 3),
                                                   pwNode(b, 2, 4)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Single pointwise node in_0=1 -> out=2, in_1 absent (nullopt): the optional
// operand slot is truly absent from the graph.
flatbuffers::FlatBufferBuilder makeSinglePwNoIn1Graph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{tensorT(b, 1, "x", {4, 4}),
                                                               tensorT(b, 2, "o", {4, 4})};
    std::vector<flatbuffers::Offset<data::Node>> n{pwNode(b, 1, 2)};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// matmul with mixed operand dtypes: a=1[2,3] FLOAT, b=2[3,4] HALF, c=3[2,4].
flatbuffers::FlatBufferBuilder makeMixedDtypeMatmulGraph() {
    flatbuffers::FlatBufferBuilder b;
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        tensorT(b, 1, "a", {2, 3}, data::DataType::FLOAT),
        tensorT(b, 2, "b", {3, 4}, data::DataType::HALF),
        tensorT(b, 3, "c", {2, 4}, data::DataType::FLOAT)};
    auto mm = data::CreateMatmulAttributes(b, 1, 2, 3);
    std::vector<flatbuffers::Offset<data::Node>> n{data::CreateNodeDirect(
        b, "matmul", data::DataType::FLOAT, data::NodeAttributes::MatmulAttributes, mm.Union())};
    b.Finish(finishGraph(b, t, n));
    return b;
}

// Build a pattern via `cfg` and run it against `view`.
gm::MatchResult runMatch(const gm::GraphView& view,
                         const std::function<void(gm::PatternBuilder&)>& cfg) {
    gm::PatternBuilder pb;
    cfg(pb);
    return gm::Matcher::match(pb.build(), view);
}

// A single-node matmul pattern binding $a/$b/$c, plus caller constraints.
auto matmulPattern(const std::function<void(gm::PatternBuilder&)>& constrain) {
    return [constrain](gm::PatternBuilder& pb) {
        pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
        constrain(pb);
    };
}

// =========================================================================
// Dtype
// =========================================================================

TEST(MatcherConstraints, DtypeInSetVsOutOfSet) {
    auto builder = util::createValidMatmulGraph();  // a is FLOAT
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_TRUE(runMatch(view, matmulPattern(
                                   [](gm::PatternBuilder& pb) { pb.constrainDtype("$a", {kF32}); }))
                    .matched);  // FLOAT is in {FLOAT}
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainDtype("$a", {kF16});
                          }))
                     .matched);  // FLOAT is not in {HALF}
}

TEST(MatcherConstraints, DtypeNegatedExcludesSet) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // FLOAT is NOT in {HALF} -> negated passes.
    EXPECT_TRUE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                             pb.constrainDtype("$a", {kF16}, /*negated=*/true);
                         }))
                    .matched);
    // FLOAT IS in {FLOAT} -> negated fails.
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainDtype("$a", {kF32}, /*negated=*/true);
                          }))
                     .matched);
}

// =========================================================================
// Rank
// =========================================================================

TEST(MatcherConstraints, RankMatchesExactAndRejectsOthers) {
    auto builder = util::createValidMatmulGraph();  // a is [4,8], rank 2
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_TRUE(
        runMatch(view, matmulPattern([](gm::PatternBuilder& pb) { pb.constrainRank("$a", 2); }))
            .matched);
    EXPECT_FALSE(
        runMatch(view, matmulPattern([](gm::PatternBuilder& pb) { pb.constrainRank("$a", 3); }))
            .matched);
}

// =========================================================================
// Shape
// =========================================================================

TEST(MatcherConstraints, ShapeLiteralMatchAndMismatch) {
    auto builder = util::createValidMatmulGraph();  // a is [4,8]
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_TRUE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                             pb.constrainShape("$a", {gm::DimSpec::lit(4), gm::DimSpec::lit(8)});
                         }))
                    .matched);
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainShape("$a", {gm::DimSpec::lit(4), gm::DimSpec::lit(9)});
                          }))
                     .matched);  // axis 1 is 8, not 9
}

TEST(MatcherConstraints, ShapeWrongRankFails) {
    auto builder = util::createValidMatmulGraph();  // a is [4,8], rank 2
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainShape(
                                  "$a", {gm::DimSpec::lit(4)});  // rank 1 spec vs rank 2 tensor
                          }))
                     .matched);
}

TEST(MatcherConstraints, ShapeWildcardIgnoresAxis) {
    auto builder = util::createValidMatmulGraph();  // a is [4,8]
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_TRUE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                             pb.constrainShape("$a", {gm::DimSpec::any(), gm::DimSpec::lit(8)});
                         }))
                    .matched);  // axis 0 wildcard, axis 1 == 8
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainShape("$a", {gm::DimSpec::any(), gm::DimSpec::lit(9)});
                          }))
                     .matched);  // wildcard cannot rescue the literal mismatch
}

// Symbol axes unify across tensors: a[4,8].axis1 and b[8,5].axis0 both name "k"
// and agree on 8, so the match succeeds and symOf resolves the bound value.
TEST(MatcherConstraints, ShapeSymbolUnifiesAcrossTensors) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.constrainShape("$a", {gm::DimSpec::any(), gm::DimSpec::of("k")});  // a.axis1 == 8
    pb.constrainShape("$b", {gm::DimSpec::of("k"), gm::DimSpec::any()});  // b.axis0 == 8
    const gm::CompiledPattern pattern = pb.build();
    ASSERT_EQ(pattern.symCount(), 1u);

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.symOf(0), 8);  // "k"
}

TEST(MatcherConstraints, ShapeSymbolConflictFails) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.constrainShape("$a", {gm::DimSpec::of("k"), gm::DimSpec::any()});  // a.axis0 == 4
    pb.constrainShape("$b",
                      {gm::DimSpec::of("k"), gm::DimSpec::any()});  // b.axis0 == 8 -> conflict
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    EXPECT_FALSE(r.matched);
    EXPECT_TRUE(r.varUids.empty());
    EXPECT_EQ(r.symOf(0), -1);
}

// =========================================================================
// Contiguous / packed-order layout
// =========================================================================

TEST(MatcherConstraints, ContiguousMatchesPackedRejectsStrided) {
    // Positive: matmul a[4,8] strides[8,1] is fully-packed row-major.
    auto mmBuilder = util::createValidMatmulGraph();
    fbu::GraphWrapper mmGraph(mmBuilder.GetBufferPointer(), mmBuilder.GetSize());
    gm::GraphView mmView(mmGraph);
    EXPECT_TRUE(runMatch(mmView, matmulPattern(
                                     [](gm::PatternBuilder& pb) { pb.constrainContiguous("$a"); }))
                    .matched);

    // Negative: pointwise graph tensors have dims[1,2,3,4] strides[5,6,7,8] --
    // not fully packed.
    auto pwBuilder = util::createPointwiseGraph();
    fbu::GraphWrapper pwGraph(pwBuilder.GetBufferPointer(), pwBuilder.GetSize());
    gm::GraphView pwView(pwGraph);
    EXPECT_FALSE(runMatch(pwView, [](gm::PatternBuilder& pb) {
                     pb.addNode("pointwise", {{"in_0", "$a"}}, {{"out_0", "$o"}});
                     pb.constrainContiguous("$a");
                 }).matched);
}

TEST(MatcherConstraints, PackedOrderLayoutMatchesRowMajorRejectsTransposed) {
    auto builder = util::createValidMatmulGraph();  // a[4,8] strides[8,1]
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // {0,1} major->minor == contiguous row-major -> matches.
    EXPECT_TRUE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                             pb.constrainLayout("$a", {0, 1});
                         }))
                    .matched);
    // {1,0} would require axis 0 to be the minor-most (stride 1), but its stride
    // is 8 -> fails.
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainLayout("$a", {1, 0});
                          }))
                     .matched);
    // An out-of-range axis in the order fails closed.
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainLayout("$a", {0, 2});
                          }))
                     .matched);
}

// =========================================================================
// Attributes
// =========================================================================

TEST(MatcherConstraints, AttrEqMatchesOperationValue) {
    auto builder = util::createPointwiseGraph();  // operation == DIV
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_TRUE(runMatch(view, [](gm::PatternBuilder& pb) {
                    pb.addNode("pointwise", {{"in_0", "$a"}}, {{"out_0", "$o"}});
                    pb.constrainAttr(0, "operation", gm::Cmp::Eq, {kDiv});
                }).matched);
    EXPECT_FALSE(runMatch(view, [](gm::PatternBuilder& pb) {
                     pb.addNode("pointwise", {{"in_0", "$a"}}, {{"out_0", "$o"}});
                     pb.constrainAttr(0, "operation", gm::Cmp::Eq, {kAdd});
                 }).matched);
}

TEST(MatcherConstraints, AttrOneOfMatchesMembershipOnly) {
    auto builder = util::createPointwiseGraph();  // operation == DIV
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_TRUE(runMatch(view, [](gm::PatternBuilder& pb) {
                    pb.addNode("pointwise", {{"in_0", "$a"}}, {{"out_0", "$o"}});
                    pb.constrainAttr(0, "operation", gm::Cmp::OneOf, {kDiv, kAdd});
                }).matched);
    EXPECT_FALSE(runMatch(view, [](gm::PatternBuilder& pb) {
                     pb.addNode("pointwise", {{"in_0", "$a"}}, {{"out_0", "$o"}});
                     pb.constrainAttr(0, "operation", gm::Cmp::OneOf, {kAdd, kMul});
                 }).matched);
}

TEST(MatcherConstraints, AttrNegatedFlipsBaseComparison) {
    auto builder = util::createPointwiseGraph();  // operation == DIV
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // operation != ADD -> negated Eq{ADD} passes.
    EXPECT_TRUE(runMatch(view, [](gm::PatternBuilder& pb) {
                    pb.addNode("pointwise", {{"in_0", "$a"}}, {{"out_0", "$o"}});
                    pb.constrainAttr(0, "operation", gm::Cmp::Eq, {kAdd}, /*negated=*/true);
                }).matched);
    // operation == DIV -> negated Eq{DIV} fails.
    EXPECT_FALSE(runMatch(view, [](gm::PatternBuilder& pb) {
                     pb.addNode("pointwise", {{"in_0", "$a"}}, {{"out_0", "$o"}});
                     pb.constrainAttr(0, "operation", gm::Cmp::Eq, {kDiv}, /*negated=*/true);
                 }).matched);
}

TEST(MatcherConstraints, AttrBoolOnSdpaCausalMask) {
    auto builder = util::createValidSdpaFwdGraph(
        {2, 8, 16, 64}, {8192, 1024, 64, 1}, {2, 8, 16, 64}, {8192, 1024, 64, 1}, {2, 8, 16, 64},
        {8192, 1024, 64, 1}, {2, 8, 16, 64}, {8192, 1024, 64, 1}, data::DataType::HALF,
        /*withAttnMask=*/false, /*withScale=*/false, /*withStats=*/false,
        /*alibiMask=*/false, /*paddingMask=*/false, /*causalMask=*/true);
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const auto sdpaPattern = [](gm::PatternBuilder& pb) {
        pb.addNode("sdpa_fwd", {{"q", "$q"}, {"k", "$k"}, {"v", "$v"}}, {{"o", "$o"}});
    };
    // causal_mask == true (1).
    EXPECT_TRUE(runMatch(view, [&](gm::PatternBuilder& pb) {
                    sdpaPattern(pb);
                    pb.constrainAttr(0, "causal_mask", gm::Cmp::Eq, {1});
                }).matched);
    EXPECT_FALSE(runMatch(view, [&](gm::PatternBuilder& pb) {
                     sdpaPattern(pb);
                     pb.constrainAttr(0, "causal_mask", gm::Cmp::Eq, {0});
                 }).matched);
}

TEST(MatcherConstraints, ConstrainAttrThrowsOnUnknownAttrAndBadNode) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    // matmul exposes no scalar attributes.
    EXPECT_THROW(pb.constrainAttr(0, "operation", gm::Cmp::Eq, {kAdd}), std::invalid_argument);

    gm::PatternBuilder pb2;
    pb2.addNode("pointwise", {{"in_0", "$a"}}, {{"out_0", "$o"}});
    // pointwise has "operation" but not "bogus".
    EXPECT_THROW(pb2.constrainAttr(0, "bogus", gm::Cmp::Eq, {kAdd}), std::invalid_argument);
    // Node index out of range (only node 0 exists).
    EXPECT_THROW(pb2.constrainAttr(5, "operation", gm::Cmp::Eq, {kAdd}), std::invalid_argument);
}

// =========================================================================
// Use-count / consumer-count
// =========================================================================

TEST(MatcherConstraints, UseCountCountsOperandSlots) {
    auto builder = makeFanOutGraph();  // uid 1 used by two nodes
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const auto fanOut = [](gm::PatternBuilder& pb) {
        pb.addNode("pointwise", {{"in_0", "$x"}}, {{"out_0", "$y0"}});
        pb.addNode("pointwise", {{"in_0", "$x"}}, {{"out_0", "$y1"}});
    };
    EXPECT_TRUE(runMatch(view, [&](gm::PatternBuilder& pb) {
                    fanOut(pb);
                    pb.constrainUseCount("$x", gm::Cmp::Eq, 2);
                }).matched);
    EXPECT_FALSE(runMatch(view, [&](gm::PatternBuilder& pb) {
                     fanOut(pb);
                     pb.constrainUseCount("$x", gm::Cmp::Eq, 1);
                 }).matched);
}

TEST(MatcherConstraints, ConsumerCountCountsDistinctNodes) {
    auto builder = makeFanOutGraph();  // uid 1 -> two distinct consumer nodes
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const auto fanOut = [](gm::PatternBuilder& pb) {
        pb.addNode("pointwise", {{"in_0", "$x"}}, {{"out_0", "$y0"}});
        pb.addNode("pointwise", {{"in_0", "$x"}}, {{"out_0", "$y1"}});
    };
    EXPECT_TRUE(runMatch(view, [&](gm::PatternBuilder& pb) {
                    fanOut(pb);
                    pb.constrainConsumerCount("$x", gm::Cmp::Eq, 2);
                }).matched);
    EXPECT_FALSE(runMatch(view, [&](gm::PatternBuilder& pb) {
                     fanOut(pb);
                     pb.constrainConsumerCount("$x", gm::Cmp::Eq, 1);
                 }).matched);
}

// The two counts differ for a self-consumed tensor: 2 operand slots, 1 node.
TEST(MatcherConstraints, UseCountAndConsumerCountDistinctForSelfConsume) {
    auto builder = makeSelfConsumeGraph();  // in_0 == in_1 == uid 1
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const auto selfConsume = [](gm::PatternBuilder& pb) {
        pb.addNode("pointwise", {{"in_0", "$h"}, {"in_1", "$h"}}, {{"out_0", "$o"}});
    };
    // useCount == 2 AND consumerCount == 1 both hold.
    EXPECT_TRUE(runMatch(view, [&](gm::PatternBuilder& pb) {
                    selfConsume(pb);
                    pb.constrainUseCount("$h", gm::Cmp::Eq, 2);
                    pb.constrainConsumerCount("$h", gm::Cmp::Eq, 1);
                }).matched);
    // consumerCount is NOT 2 (that is the use-count) -> fails.
    EXPECT_FALSE(runMatch(view, [&](gm::PatternBuilder& pb) {
                     selfConsume(pb);
                     pb.constrainConsumerCount("$h", gm::Cmp::Eq, 2);
                 }).matched);
}

// =========================================================================
// NoConsumerOutside
// =========================================================================

TEST(MatcherConstraints, NoConsumerOutsidePassesWhenAllConsumersMatched) {
    auto builder = makeLinearChainGraph();  // uid 2's only consumer is node1
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const gm::MatchResult r = runMatch(view, [](gm::PatternBuilder& pb) {
        pb.addNode("pointwise", {{"in_0", "$x"}}, {{"out_0", "$m"}});
        pb.addNode("pointwise", {{"in_0", "$m"}}, {{"out_0", "$y"}});
        pb.constrainNoConsumerOutside("$m");
    });
    EXPECT_TRUE(r.matched);
}

TEST(MatcherConstraints, NoConsumerOutsideFailsWithExternalConsumer) {
    auto builder = makeSharedIntermediateGraph();  // uid 2 has two consumers
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // The chain pattern matches node0 + exactly one consumer of uid 2; the other
    // consumer is always outside the match, so the constraint can never hold.
    const gm::MatchResult r = runMatch(view, [](gm::PatternBuilder& pb) {
        pb.addNode("pointwise", {{"in_0", "$x"}}, {{"out_0", "$m"}});
        pb.addNode("pointwise", {{"in_0", "$m"}}, {{"out_0", "$y"}});
        pb.constrainNoConsumerOutside("$m");
    });
    EXPECT_FALSE(r.matched);
}

// =========================================================================
// SameDtype / SameDim
// =========================================================================

TEST(MatcherConstraints, SameDtypeMatchesEqualDtypes) {
    auto builder = util::createValidMatmulGraph();  // a,b both FLOAT
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_TRUE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                             pb.constrainSameDtype("$a", "$b");
                         }))
                    .matched);
    // Same dtypes -> negated (require differ) fails.
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainSameDtype("$a", "$b", /*negated=*/true);
                          }))
                     .matched);
}

TEST(MatcherConstraints, SameDtypeDetectsDifferingDtypes) {
    auto builder = makeMixedDtypeMatmulGraph();  // a FLOAT, b HALF
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // Differing dtypes -> plain same-dtype fails.
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainSameDtype("$a", "$b");
                          }))
                     .matched);
    // ... and negated (require differ) passes.
    EXPECT_TRUE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                             pb.constrainSameDtype("$a", "$b", /*negated=*/true);
                         }))
                    .matched);
}

TEST(MatcherConstraints, SameDimComparesChosenAxes) {
    auto builder = util::createValidMatmulGraph();  // a[4,8], b[8,5]
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // a.axis1 (8) == b.axis0 (8) -> matches.
    EXPECT_TRUE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                             pb.constrainSameDim("$a", 1, "$b", 0);
                         }))
                    .matched);
    // a.axis0 (4) != b.axis0 (8) -> fails.
    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainSameDim("$a", 0, "$b", 0);
                          }))
                     .matched);
}

TEST(MatcherConstraints, SameDimOutOfRangeAxisFailsClosed) {
    auto builder = util::createValidMatmulGraph();  // a rank 2
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_FALSE(runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                              pb.constrainSameDim("$a", 99, "$b", 0);  // no such axis on a
                          }))
                     .matched);
}

// =========================================================================
// Optional operands
// =========================================================================

TEST(MatcherConstraints, OptionalOperandPresentBindsVar) {
    auto builder = util::createPointwiseGraph();  // in_1 == uid 2 present
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // Interning: $x=0, $b=1, $o=2.
    pb.addNode("pointwise", {{"in_0", "$x"}, {"in_1", "$b", /*optional=*/true}}, {{"out_0", "$o"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(0), 1);  // $x -> in_0
    EXPECT_EQ(r.uidOf(1), 2);  // $b -> in_1, present and bound
    EXPECT_EQ(r.uidOf(2), 4);  // $o -> out_0
    ASSERT_EQ(r.varBound.size(), 3u);
    EXPECT_TRUE(r.varBound[1]);
}

TEST(MatcherConstraints, OptionalOperandAbsentLeavesVarUnbound) {
    auto builder = makeSinglePwNoIn1Graph();  // in_1 absent
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    // Interning: $x=0, $b=1, $o=2.
    pb.addNode("pointwise", {{"in_0", "$x"}, {"in_1", "$b", /*optional=*/true}}, {{"out_0", "$o"}});
    const gm::CompiledPattern pattern = pb.build();

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(0), 1);   // $x bound
    EXPECT_EQ(r.uidOf(1), -1);  // $b absent optional -> unbound
    ASSERT_EQ(r.varBound.size(), 3u);
    EXPECT_FALSE(r.varBound[1]);
    EXPECT_EQ(r.uidOf(2), 2);  // $o bound
}

// A constraint on an unbound (absent-optional) var is skipped, not failed.
TEST(MatcherConstraints, ConstraintOnAbsentOptionalIsSkipped) {
    auto builder = makeSinglePwNoIn1Graph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // A dtype constraint that could never hold is harmless because $b is unbound.
    const gm::MatchResult r = runMatch(view, [](gm::PatternBuilder& pb) {
        pb.addNode("pointwise", {{"in_0", "$x"}, {"in_1", "$b", /*optional=*/true}},
                   {{"out_0", "$o"}});
        pb.constrainDtype("$b", {kF16});  // $b never binds -> skipped
    });
    EXPECT_TRUE(r.matched);
}

// =========================================================================
// All-or-nothing / builder validation
// =========================================================================

TEST(MatcherConstraints, ConstraintFailureClearsAllBindings) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const gm::MatchResult r = runMatch(view, matmulPattern([](gm::PatternBuilder& pb) {
                                           pb.constrainRank("$a", 99);  // impossible
                                       }));
    EXPECT_FALSE(r.matched);
    EXPECT_FALSE(r.budgetExceeded);
    EXPECT_TRUE(r.varUids.empty());
    EXPECT_TRUE(r.nodeMap.empty());
    EXPECT_TRUE(r.symVals.empty());
    EXPECT_EQ(r.uidOf(0), -1);
}

TEST(MatcherConstraints, ConstrainOnUnknownVarThrows) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});

    EXPECT_THROW(pb.constrainDtype("$z", {kF32}), std::invalid_argument);
    EXPECT_THROW(pb.constrainRank("$z", 2), std::invalid_argument);
    EXPECT_THROW(pb.constrainShape("$z", {gm::DimSpec::any()}), std::invalid_argument);
    EXPECT_THROW(pb.constrainContiguous("$z"), std::invalid_argument);
    EXPECT_THROW(pb.constrainLayout("$z", {0}), std::invalid_argument);
    EXPECT_THROW(pb.constrainUseCount("$z", gm::Cmp::Eq, 1), std::invalid_argument);
    EXPECT_THROW(pb.constrainConsumerCount("$z", gm::Cmp::Eq, 1), std::invalid_argument);
    EXPECT_THROW(pb.constrainNoConsumerOutside("$z"), std::invalid_argument);
    EXPECT_THROW(pb.constrainSameDtype("$a", "$z"), std::invalid_argument);
    EXPECT_THROW(pb.constrainSameDim("$z", 0, "$a", 0), std::invalid_argument);
}

}  // namespace
