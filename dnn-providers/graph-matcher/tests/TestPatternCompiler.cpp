// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Phase-4 PatternCompiler tests: the untrusted-JSON -> CompiledPattern compiler
// (PatternCompiler::fromJson). These exercise the JSON parsing / validation /
// bounds layer and prove a compiled pattern actually matches a real graph.
// The builder and matcher internals are covered by earlier phases and are not
// re-tested here directly -- everything goes through fromJson. fromJson never
// throws; it returns CompileResult{ok,pattern,error}, so rejections are asserted
// via ok==false and a non-empty error string (never EXPECT_THROW).

#include <gtest/gtest.h>

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_graph_matcher/PatternCompiler.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <string>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;
namespace data = hipdnn_flatbuffers_sdk::data_objects;

// --- Helpers -------------------------------------------------------------

// VarId whose author name is `name`, or UINT32_MAX if absent. Lets tests assert
// bindings by name instead of hard-coding the compiler's interning order.
gm::VarId varOf(const gm::CompiledPattern& p, std::string_view name) {
    for (gm::VarId v = 0; v < p.varCount(); ++v) {
        if (p.varName(v) == name) {
            return v;
        }
    }
    return UINT32_MAX;
}

gm::SymId symIdOf(const gm::CompiledPattern& p, std::string_view name) {
    for (gm::SymId s = 0; s < p.symCount(); ++s) {
        if (p.symName(s) == name) {
            return s;
        }
    }
    return UINT32_MAX;
}

// A single pointwise(ADD) node with in_0=uid1 -> out_0=uid2 and NO in_1. Used to
// prove the optional-operand JSON form ({"var":..,"optional":true}) both parses
// and still matches when the role is absent in the graph.
flatbuffers::FlatBufferBuilder makeSinglePointwiseGraph() {
    flatbuffers::FlatBufferBuilder b;
    const std::vector<int64_t> dims{4, 4};
    const std::vector<int64_t> strides{4, 1};
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        data::CreateTensorAttributesDirect(b, 1, "x", data::DataType::FLOAT, &strides, &dims),
        data::CreateTensorAttributesDirect(b, 2, "y", data::DataType::FLOAT, &strides, &dims)};
    auto attrs = data::CreatePointwiseAttributes(b, data::PointwiseMode::ADD, flatbuffers::nullopt,
                                                 flatbuffers::nullopt, flatbuffers::nullopt,
                                                 flatbuffers::nullopt, /*in_0=*/1,
                                                 flatbuffers::nullopt,  // in_1 absent
                                                 flatbuffers::nullopt, /*out_0=*/2);
    std::vector<flatbuffers::Offset<data::Node>> n{data::CreateNodeDirect(
        b, "pw", data::DataType::FLOAT, data::NodeAttributes::PointwiseAttributes, attrs.Union())};
    b.Finish(data::CreateGraphDirect(b, "test", data::DataType::FLOAT, data::DataType::FLOAT,
                                     data::DataType::FLOAT, &t, &n));
    return b;
}

// The matmul->bias->activation graph used by the positive matmul tests. Fixed
// facts asserted downstream (read from the builder): A=uid1[4,8], B=uid2[8,5],
// C_matmul=uid3, bias=uid4, C_bias=uid5, C=uid6; bias node = pointwise ADD(2),
// activation = pointwise RELU_FWD(34). So shared reduction dim k = 8, m = 4,
// n = 5, and C_matmul (uid3) is consumed exactly once (only by the bias node).
const char* const kMatmulBiasJson = R"({
  "schema": "hipdnn.criteria/v1",
  "nodes": [
    {"id":"mm","op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$h"}},
    {"id":"pw","op":"pointwise","operands":{"in_0":"$h"},"results":{"out_0":"$y"}}
  ],
  "constraints": [
    {"on":"$a","dtype":"FLOAT","shape":["m","k"]},
    {"on":"$b","shape":["k","n"]},
    {"on":"$h","use":"exactly_once"},
    {"kind":"same_dim","args":["$a",1,"$b",0]}
  ]
})";

// =========================================================================
// Positive: compile + match end to end
// =========================================================================

// The full matmul->pointwise chain compiles, has the expected shape, matches the
// real matmul-bias graph, and binds the shared symbol k to the concrete dim 8
// (with m=4, n=5) read off the graph tensors.
TEST(PatternCompiler, MatmulChainCompilesAndMatchesWithSymbolBindings) {
    const gm::CompileResult cr = gm::PatternCompiler::fromJson(kMatmulBiasJson);
    ASSERT_TRUE(cr.ok) << cr.error;

    const gm::CompiledPattern& pattern = cr.pattern;
    EXPECT_EQ(pattern.nodeCount(), 2u);
    EXPECT_EQ(pattern.varCount(), 4u);  // $a $b $h $y
    EXPECT_EQ(pattern.symCount(), 3u);  // m k n
    // Default anchor is the unique sink -- the pointwise node ($y is consumed by
    // nobody), which is the second node authored.
    EXPECT_EQ(pattern.anchor(), 1u);

    auto graphBuilder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    gm::GraphView view(graph);

    const gm::MatchResult r = gm::Matcher::match(pattern, view);
    ASSERT_TRUE(r.matched);

    EXPECT_EQ(r.uidOf(varOf(pattern, "$a")), 1);  // A
    EXPECT_EQ(r.uidOf(varOf(pattern, "$b")), 2);  // B
    EXPECT_EQ(r.uidOf(varOf(pattern, "$h")), 3);  // C_matmul (bias node in_0)
    EXPECT_EQ(r.uidOf(varOf(pattern, "$y")), 5);  // C_bias (bias node out_0)

    EXPECT_EQ(r.symOf(symIdOf(pattern, "k")), 8);  // A[.,8] == B[8,.]
    EXPECT_EQ(r.symOf(symIdOf(pattern, "m")), 4);  // A[4,.]
    EXPECT_EQ(r.symOf(symIdOf(pattern, "n")), 5);  // B[.,5]
}

// A node-id + attribute constraint compiles and evaluates against the graph:
// operation==ADD(2) matches the bias node; operation==RELU_FWD(34) does not
// (the only RELU node is the activation, whose in_0 is not a matmul result), so
// the same structural pattern with a different attr value fails to match. This
// proves the attr constraint actually gates the match rather than being inert.
TEST(PatternCompiler, NodeAttrConstraintGatesTheMatch) {
    const char* const matchJson = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[
        {"id":"mm","op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$h"}},
        {"id":"pw","op":"pointwise","operands":{"in_0":"$h"},"results":{"out_0":"$y"}}
      ],
      "constraints":[{"on":"pw","attr":{"operation":{"equals":2}}}]
    })";
    const char* const missJson = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[
        {"id":"mm","op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$h"}},
        {"id":"pw","op":"pointwise","operands":{"in_0":"$h"},"results":{"out_0":"$y"}}
      ],
      "constraints":[{"on":"pw","attr":{"operation":{"equals":34}}}]
    })";

    auto graphBuilder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    gm::GraphView view(graph);

    const gm::CompileResult ok = gm::PatternCompiler::fromJson(matchJson);
    ASSERT_TRUE(ok.ok) << ok.error;
    EXPECT_TRUE(gm::Matcher::match(ok.pattern, view).matched);  // ADD node

    const gm::CompileResult miss = gm::PatternCompiler::fromJson(missJson);
    ASSERT_TRUE(miss.ok) << miss.error;  // compiles fine; just won't match
    EXPECT_FALSE(gm::Matcher::match(miss.pattern, view).matched);  // no RELU off matmul
}

// A single sdpa_fwd node with a var layout constraint and a node-attr constraint
// compiles and matches the SDPA graph; the attr value gates the match the same
// way (causal_mask default is false==0, so ==0 matches and ==1 does not).
TEST(PatternCompiler, SdpaSingleNodeCompilesAndMatches) {
    const char* const okJson = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"id":"s","op":"sdpa_fwd",
                "operands":{"q":"$q","k":"$k","v":"$v"},"results":{"o":"$o"}}],
      "constraints":[
        {"on":"$q","layout":"contiguous"},
        {"on":"s","attr":{"causal_mask":{"equals":0}}}
      ]
    })";
    const char* const noMatchJson = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"id":"s","op":"sdpa_fwd",
                "operands":{"q":"$q","k":"$k","v":"$v"},"results":{"o":"$o"}}],
      "constraints":[{"on":"s","attr":{"causal_mask":{"equals":1}}}]
    })";

    const gm::CompileResult cr = gm::PatternCompiler::fromJson(okJson);
    ASSERT_TRUE(cr.ok) << cr.error;
    EXPECT_EQ(cr.pattern.nodeCount(), 1u);
    EXPECT_EQ(cr.pattern.varCount(), 4u);  // $q $k $v $o
    EXPECT_EQ(cr.pattern.symCount(), 0u);
    EXPECT_EQ(cr.pattern.anchor(), 0u);

    auto graphBuilder = util::createValidSdpaFwdGraph();
    fbu::GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    gm::GraphView view(graph);

    const gm::MatchResult r = gm::Matcher::match(cr.pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(varOf(cr.pattern, "$q")), 1);
    EXPECT_EQ(r.uidOf(varOf(cr.pattern, "$o")), 4);

    const gm::CompileResult nm = gm::PatternCompiler::fromJson(noMatchJson);
    ASSERT_TRUE(nm.ok) << nm.error;
    EXPECT_FALSE(gm::Matcher::match(nm.pattern, view).matched);  // causal_mask is false
}

// The optional-operand form {"var":..,"optional":true} compiles and lets a node
// with an extra operand role match a graph node that lacks it, leaving that var
// unbound; making the same operand required instead fails to match.
TEST(PatternCompiler, OptionalOperandFormCompilesAndMatchesWhenAbsent) {
    const char* const optJson = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"op":"pointwise",
                "operands":{"in_0":"$x","in_1":{"var":"$w","optional":true}},
                "results":{"out_0":"$y"}}]
    })";
    const char* const reqJson = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"op":"pointwise",
                "operands":{"in_0":"$x","in_1":"$w"},"results":{"out_0":"$y"}}]
    })";

    auto graphBuilder = makeSinglePointwiseGraph();
    fbu::GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    gm::GraphView view(graph);

    const gm::CompileResult opt = gm::PatternCompiler::fromJson(optJson);
    ASSERT_TRUE(opt.ok) << opt.error;
    const gm::MatchResult r = gm::Matcher::match(opt.pattern, view);
    ASSERT_TRUE(r.matched);
    EXPECT_EQ(r.uidOf(varOf(opt.pattern, "$x")), 1);
    EXPECT_EQ(r.uidOf(varOf(opt.pattern, "$w")), -1);  // absent optional stays unbound

    const gm::CompileResult req = gm::PatternCompiler::fromJson(reqJson);
    ASSERT_TRUE(req.ok) << req.error;
    EXPECT_FALSE(gm::Matcher::match(req.pattern, view).matched);  // in_1 missing in graph
}

// An explicit "anchor":true overrides the default unique-sink anchor.
TEST(PatternCompiler, ExplicitAnchorOverridesDefaultSink) {
    const char* const json = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[
        {"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$h"},"anchor":true},
        {"op":"pointwise","operands":{"in_0":"$h"},"results":{"out_0":"$y"}}
      ]
    })";
    const gm::CompileResult cr = gm::PatternCompiler::fromJson(json);
    ASSERT_TRUE(cr.ok) << cr.error;
    EXPECT_EQ(cr.pattern.anchor(), 0u);  // matmul, not the default sink (node 1)
}

// Compiling the same JSON twice yields patterns that match the same graph with
// identical bindings -- the compiler is deterministic.
TEST(PatternCompiler, CompileIsDeterministicAcrossRuns) {
    const gm::CompileResult a = gm::PatternCompiler::fromJson(kMatmulBiasJson);
    const gm::CompileResult b = gm::PatternCompiler::fromJson(kMatmulBiasJson);
    ASSERT_TRUE(a.ok) << a.error;
    ASSERT_TRUE(b.ok) << b.error;
    ASSERT_EQ(a.pattern.varCount(), b.pattern.varCount());
    ASSERT_EQ(a.pattern.symCount(), b.pattern.symCount());

    auto graphBuilder = util::createValidMatmulBiasActivGraph();
    fbu::GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    gm::GraphView view(graph);

    const gm::MatchResult ra = gm::Matcher::match(a.pattern, view);
    const gm::MatchResult rb = gm::Matcher::match(b.pattern, view);
    ASSERT_TRUE(ra.matched);
    ASSERT_TRUE(rb.matched);
    EXPECT_EQ(ra.varUids, rb.varUids);
    EXPECT_EQ(ra.symVals, rb.symVals);
    EXPECT_EQ(ra.nodeMap, rb.nodeMap);
}

// =========================================================================
// Negative: structural / schema / vocabulary rejections (default limits)
// =========================================================================

struct RejectCase {
    const char* name;
    const char* json;
};

class PatternCompilerReject : public ::testing::TestWithParam<RejectCase> {};

TEST_P(PatternCompilerReject, ReturnsErrorResult) {
    const gm::CompileResult cr = gm::PatternCompiler::fromJson(GetParam().json);
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

INSTANTIATE_TEST_SUITE_P(
    Cases, PatternCompilerReject,
    ::testing::Values(
        RejectCase{"MalformedJson", "{ this is not valid json "},
        RejectCase{"WrongSchema", R"({"schema":"hipdnn.criteria/v2","nodes":[{"op":"matmul"}]})"},
        RejectCase{"MissingSchema", R"({"nodes":[{"op":"matmul"}]})"},
        RejectCase{"EmptyNodes", R"({"schema":"hipdnn.criteria/v1","nodes":[]})"},
        RejectCase{"MissingNodes", R"({"schema":"hipdnn.criteria/v1"})"},
        RejectCase{"UnknownOp", R"({"schema":"hipdnn.criteria/v1","nodes":[{"op":"frobnicate"}]})"},
        RejectCase{"UnknownRole",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[{"op":"matmul","operands":{"z":"$x"}}]})"},
        RejectCase{"UnknownDtypeName",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},
                                 "results":{"c":"$c"}}],
                       "constraints":[{"on":"$a","dtype":"FLOOP"}]})"},
        RejectCase{"VarConstraintNoKeys",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},
                                 "results":{"c":"$c"}}],
                       "constraints":[{"on":"$a"}]})"},
        RejectCase{"AttrOnUnknownNodeId",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},
                                 "results":{"c":"$c"}}],
                       "constraints":[{"on":"nope","attr":{"operation":{"equals":1}}}]})"},
        RejectCase{"AttrOnOpWithoutIt",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[{"id":"mm","op":"matmul","operands":{"a":"$a","b":"$b"},
                                 "results":{"c":"$c"}}],
                       "constraints":[{"on":"mm","attr":{"operation":{"equals":1}}}]})"},
        RejectCase{"NativePredicateUnsupported",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},
                                 "results":{"c":"$c"}}],
                       "constraints":[{"kind":"native_predicate","args":["$a","$b"]}]})"},
        RejectCase{"UnknownCrossKind",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},
                                 "results":{"c":"$c"}}],
                       "constraints":[{"kind":"bogus","args":["$a","$b"]}]})"},
        RejectCase{"SameDimWrongArgCount",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},
                                 "results":{"c":"$c"}}],
                       "constraints":[{"kind":"same_dim","args":["$a",1,"$b"]}]})"},
        RejectCase{"DuplicateNodeId",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[
                         {"id":"n","op":"pointwise","operands":{"in_0":"$x"},
                          "results":{"out_0":"$y"}},
                         {"id":"n","op":"pointwise","operands":{"in_0":"$y"},
                          "results":{"out_0":"$z"}}]})"},
        RejectCase{"TwoAnchors",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[
                         {"op":"pointwise","operands":{"in_0":"$x"},
                          "results":{"out_0":"$y"},"anchor":true},
                         {"op":"pointwise","operands":{"in_0":"$y"},
                          "results":{"out_0":"$z"},"anchor":true}]})"},
        RejectCase{"DisconnectedPattern",
                   R"({"schema":"hipdnn.criteria/v1",
                       "nodes":[
                         {"op":"pointwise","operands":{"in_0":"$a"},"results":{"out_0":"$b"}},
                         {"op":"pointwise","operands":{"in_0":"$c"},"results":{"out_0":"$d"}}]})"}),
    [](const ::testing::TestParamInfo<RejectCase>& info) { return info.param.name; });

// =========================================================================
// Negative: every bound rejects an input that exceeds it (no crash / hang)
// =========================================================================

TEST(PatternCompilerBounds, MaxInputBytes) {
    gm::CompileLimits lim;
    lim.maxInputBytes = 10;  // far smaller than any real document
    const gm::CompileResult cr =
        gm::PatternCompiler::fromJson(kMatmulBiasJson, gm::OpSchemaRegistry::builtin(), lim);
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

TEST(PatternCompilerBounds, MaxDepth) {
    // Depth is scanned before parsing, so deep nesting is rejected up front even
    // though the string is not otherwise valid JSON.
    const std::string deep = std::string(200, '[') + std::string(200, ']');
    const gm::CompileResult cr = gm::PatternCompiler::fromJson(deep);  // default maxDepth 64
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

TEST(PatternCompilerBounds, MaxNodes) {
    gm::CompileLimits lim;
    lim.maxNodes = 1;
    const char* const json = R"({"schema":"hipdnn.criteria/v1",
      "nodes":[
        {"op":"pointwise","operands":{"in_0":"$x"},"results":{"out_0":"$y"}},
        {"op":"pointwise","operands":{"in_0":"$y"},"results":{"out_0":"$z"}}]})";
    const gm::CompileResult cr =
        gm::PatternCompiler::fromJson(json, gm::OpSchemaRegistry::builtin(), lim);
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

TEST(PatternCompilerBounds, MaxEdgesPerNode) {
    gm::CompileLimits lim;
    lim.maxEdgesPerNode = 1;
    const char* const json = R"({"schema":"hipdnn.criteria/v1",
      "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}]})";
    const gm::CompileResult cr =
        gm::PatternCompiler::fromJson(json, gm::OpSchemaRegistry::builtin(), lim);
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

TEST(PatternCompilerBounds, MaxConstraints) {
    gm::CompileLimits lim;
    lim.maxConstraints = 1;
    const char* const json = R"({"schema":"hipdnn.criteria/v1",
      "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
      "constraints":[{"on":"$a","rank":2},{"on":"$b","rank":2}]})";
    const gm::CompileResult cr =
        gm::PatternCompiler::fromJson(json, gm::OpSchemaRegistry::builtin(), lim);
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

TEST(PatternCompilerBounds, MaxShapeDims) {
    gm::CompileLimits lim;
    lim.maxShapeDims = 2;
    const char* const json = R"({"schema":"hipdnn.criteria/v1",
      "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
      "constraints":[{"on":"$a","shape":[1,2,3]}]})";
    const gm::CompileResult cr =
        gm::PatternCompiler::fromJson(json, gm::OpSchemaRegistry::builtin(), lim);
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

TEST(PatternCompilerBounds, MaxSetSize) {
    gm::CompileLimits lim;
    lim.maxSetSize = 1;
    const char* const json = R"({"schema":"hipdnn.criteria/v1",
      "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
      "constraints":[{"on":"$a","dtype":{"one_of":["FLOAT","HALF"]}}]})";
    const gm::CompileResult cr =
        gm::PatternCompiler::fromJson(json, gm::OpSchemaRegistry::builtin(), lim);
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

TEST(PatternCompilerBounds, MaxNameLen) {
    gm::CompileLimits lim;
    lim.maxNameLen = 20;  // schema string (18 chars) fits; the long var does not
    const std::string longVar = "$" + std::string(25, 'z');
    const std::string json = std::string(R"({"schema":"hipdnn.criteria/v1",)") +
                             R"("nodes":[{"op":"pointwise","operands":{"in_0":")" + longVar +
                             R"("},"results":{"out_0":"$y"}}]})";
    const gm::CompileResult cr =
        gm::PatternCompiler::fromJson(json, gm::OpSchemaRegistry::builtin(), lim);
    EXPECT_FALSE(cr.ok);
    EXPECT_FALSE(cr.error.empty());
}

}  // namespace
