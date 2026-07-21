// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Phase-5 native-predicate tests: the escape-hatch predicate surface layered on
// structural matching + constraints. Covers the PredicateRegistry (built-ins and
// consumer-registered author predicates), PatternBuilder::addPredicate
// (arity/kind/provenance validation + negation), the Matcher's match-time
// predicate evaluation (fail-closed on an unresolved name, skip on an unbound
// arg, negation), and the JSON "native_predicate" constraint via
// PatternCompiler::fromJson. Phases 0-4 (OpSchema, GraphView, structural/DAG
// matching, the constraint vocabulary, JSON basics) are covered elsewhere and are
// deliberately not re-tested here -- everything here defends the predicate layer.
//
// addPredicate throws std::invalid_argument on invalid predicates (EXPECT_THROW);
// Matcher::match and fromJson never throw (fromJson returns CompileResult{ok,..}).

#include <gtest/gtest.h>

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_graph_matcher/PatternCompiler.hpp>
#include <hipdnn_graph_matcher/Predicate.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;
namespace data = hipdnn_flatbuffers_sdk::data_objects;

// --- PredArgSpec factories ------------------------------------------------
using Spec = gm::PatternBuilder::PredArgSpec;

Spec V(std::string_view name) {
    return Spec{gm::PredicateArg::Source::Var, name, 0};
}
Spec S(std::string_view name) {
    return Spec{gm::PredicateArg::Source::Sym, name, 0};
}
Spec L(int64_t value) {
    return Spec{gm::PredicateArg::Source::Literal, {}, value};
}

// A registry-registered (non-builtin) author predicate that always passes. Used
// to prove the trust rule (barred under DropIn) and match-time registry
// resolution (fail-closed under the default builtin() registry).
bool pluginTrue(const std::vector<gm::BoundArg>& args) {
    return !args.empty() && args[0].tensor != nullptr;
}

// A copy of builtin() extended with the author predicate "test.plugin_true"
// (one Tensor arg). registerPredicate always marks it non-builtin.
gm::PredicateRegistry pluginRegistry() {
    gm::PredicateRegistry reg = gm::PredicateRegistry::builtin();
    reg.registerPredicate({"test.plugin_true", {gm::ArgKind::Tensor}, &pluginTrue, true});
    return reg;
}

// Row-major contiguous strides for `dims` (last dim = 1).
std::vector<int64_t> contigStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> s(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) {
        s[i - 1] = s[i] * dims[i];
    }
    return s;
}

// A single pointwise(ADD) node with in_0=uid1 -> out_0=uid2 and NO in_1, so the
// optional in_1 operand slot is genuinely absent from the graph (var $opt stays
// unbound). Proves a predicate over an unbound optional arg is skipped.
flatbuffers::FlatBufferBuilder makeSinglePwNoIn1Graph() {
    flatbuffers::FlatBufferBuilder b;
    const std::vector<int64_t> dims{4, 4};
    const std::vector<int64_t> strides{4, 1};
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        data::CreateTensorAttributesDirect(b, 1, "x", data::DataType::FLOAT, &strides, &dims),
        data::CreateTensorAttributesDirect(b, 2, "y", data::DataType::FLOAT, &strides, &dims)};
    auto attrs = data::CreatePointwiseAttributes(
        b, data::PointwiseMode::ADD, flatbuffers::nullopt, flatbuffers::nullopt,
        flatbuffers::nullopt, flatbuffers::nullopt,
        /*in_0=*/1, flatbuffers::nullopt, flatbuffers::nullopt, /*out_0=*/2);
    std::vector<flatbuffers::Offset<data::Node>> n{data::CreateNodeDirect(
        b, "pw", data::DataType::FLOAT, data::NodeAttributes::PointwiseAttributes, attrs.Union())};
    b.Finish(data::CreateGraphDirect(b, "test", data::DataType::FLOAT, data::DataType::FLOAT,
                                     data::DataType::FLOAT, &t, &n));
    return b;
}

// A matmul whose operands differ in dtype: a=uid1[2,3] FLOAT, b=uid2[3,4] HALF,
// c=uid3[2,4] FLOAT. same_dtype($a,$b) is false here.
flatbuffers::FlatBufferBuilder makeMixedDtypeMatmulGraph() {
    flatbuffers::FlatBufferBuilder b;
    const std::vector<int64_t> aDims{2, 3}, bDims{3, 4}, cDims{2, 4};
    const auto aS = contigStrides(aDims), bS = contigStrides(bDims), cS = contigStrides(cDims);
    std::vector<flatbuffers::Offset<data::TensorAttributes>> t{
        data::CreateTensorAttributesDirect(b, 1, "a", data::DataType::FLOAT, &aS, &aDims),
        data::CreateTensorAttributesDirect(b, 2, "b", data::DataType::HALF, &bS, &bDims),
        data::CreateTensorAttributesDirect(b, 3, "c", data::DataType::FLOAT, &cS, &cDims)};
    auto mm = data::CreateMatmulAttributes(b, 1, 2, 3);
    std::vector<flatbuffers::Offset<data::Node>> n{data::CreateNodeDirect(
        b, "matmul", data::DataType::FLOAT, data::NodeAttributes::MatmulAttributes, mm.Union())};
    b.Finish(data::CreateGraphDirect(b, "test", data::DataType::FLOAT, data::DataType::FLOAT,
                                     data::DataType::FLOAT, &t, &n));
    return b;
}

// =========================================================================
// PredicateRegistry
// =========================================================================

TEST(PredicateRegistry, BuiltinsPresentWithDeclaredArgKinds) {
    const auto& reg = gm::PredicateRegistry::builtin();

    const auto* sd = reg.find("hipdnn.same_dtype");
    ASSERT_NE(sd, nullptr);
    EXPECT_TRUE(sd->builtin);
    EXPECT_EQ(sd->argKinds, (std::vector<gm::ArgKind>{gm::ArgKind::Tensor, gm::ArgKind::Tensor}));

    const auto* shd = reg.find("hipdnn.same_head_dim");
    ASSERT_NE(shd, nullptr);
    EXPECT_TRUE(shd->builtin);
    EXPECT_EQ(shd->argKinds, (std::vector<gm::ArgKind>{gm::ArgKind::Tensor, gm::ArgKind::Tensor,
                                                       gm::ArgKind::Tensor}));

    const auto* div = reg.find("hipdnn.divisible_by");
    ASSERT_NE(div, nullptr);
    EXPECT_TRUE(div->builtin);
    EXPECT_EQ(div->argKinds, (std::vector<gm::ArgKind>{gm::ArgKind::Int, gm::ArgKind::Int}));
}

TEST(PredicateRegistry, UnknownNameResolvesToNull) {
    EXPECT_EQ(gm::PredicateRegistry::builtin().find("hipdnn.not_a_predicate"), nullptr);
}

TEST(PredicateRegistry, RegisterPredicateAddsNonBuiltinToCopyOnly) {
    gm::PredicateRegistry copy = pluginRegistry();

    const auto* e = copy.find("test.plugin_true");
    ASSERT_NE(e, nullptr);
    EXPECT_FALSE(e->builtin);  // registerPredicate forces builtin=false
    EXPECT_EQ(e->fn, &pluginTrue);

    // The process-wide builtin() set is unaffected by extending a copy.
    EXPECT_EQ(gm::PredicateRegistry::builtin().find("test.plugin_true"), nullptr);
}

// =========================================================================
// PatternBuilder::addPredicate -- matcher outcomes (positive/negative)
// =========================================================================

// same_dtype on the matmul (a,b both FLOAT): satisfied keeps the match, negated
// inverts it to a non-match.
TEST(Predicates, SameDtypeMatchesAndNegates) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pos;
    pos.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pos.addPredicate("hipdnn.same_dtype", {V("$a"), V("$b")});
    EXPECT_TRUE(gm::Matcher::match(pos.build(), view).matched);

    gm::PatternBuilder neg;
    neg.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    neg.addPredicate("hipdnn.same_dtype", {V("$a"), V("$b")}, /*negated=*/true);
    EXPECT_FALSE(gm::Matcher::match(neg.build(), view).matched);
}

// The mirror image: on a matmul whose operands differ in dtype (FLOAT vs HALF),
// same_dtype fails but its negation succeeds -- proving negation is wired to the
// real verdict, not a fixed answer.
TEST(Predicates, SameDtypeNegationTracksDifferingDtypes) {
    auto builder = makeMixedDtypeMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pos;
    pos.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pos.addPredicate("hipdnn.same_dtype", {V("$a"), V("$b")});
    EXPECT_FALSE(gm::Matcher::match(pos.build(), view).matched);  // FLOAT != HALF

    gm::PatternBuilder neg;
    neg.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    neg.addPredicate("hipdnn.same_dtype", {V("$a"), V("$b")}, /*negated=*/true);
    EXPECT_TRUE(gm::Matcher::match(neg.build(), view).matched);
}

// divisible_by(k, d) where k is a symbol bound (via constrainShape) to matmul a's
// last dim == 8: 8 % 4 == 0 matches, 8 % 3 != 0 fails. Pins the concrete k value.
TEST(Predicates, DivisibleByBoundSymbol) {
    auto builder = util::createValidMatmulGraph();  // a dims {4,8}
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pass;
    pass.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pass.constrainShape("$a", {gm::DimSpec::any(), gm::DimSpec::of("k")});  // k := 8
    pass.addPredicate("hipdnn.divisible_by", {S("k"), L(4)});
    EXPECT_TRUE(gm::Matcher::match(pass.build(), view).matched);  // 8 % 4 == 0

    gm::PatternBuilder fail;
    fail.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    fail.constrainShape("$a", {gm::DimSpec::any(), gm::DimSpec::of("k")});  // k := 8
    fail.addPredicate("hipdnn.divisible_by", {S("k"), L(3)});
    EXPECT_FALSE(gm::Matcher::match(fail.build(), view).matched);  // 8 % 3 != 0
}

// same_head_dim(q,k,v): SDPA q/k/v all share last dim 64 -> satisfied; override v
// to last dim 32 -> the three no longer share a last dim -> fails.
TEST(Predicates, SameHeadDimAcrossThreeTensors) {
    {
        auto builder = util::createValidSdpaFwdGraph();  // q/k/v last dim 64
        fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        gm::GraphView view(graph);

        gm::PatternBuilder pb;
        pb.addNode("sdpa_fwd", {{"q", "$q"}, {"k", "$k"}, {"v", "$v"}}, {{"o", "$o"}});
        pb.addPredicate("hipdnn.same_head_dim", {V("$q"), V("$k"), V("$v")});
        EXPECT_TRUE(gm::Matcher::match(pb.build(), view).matched);
    }
    {
        const std::vector<int64_t> vDims{2, 8, 16, 32};  // v head dim differs (32 != 64)
        auto builder =
            util::createValidSdpaFwdGraph({2, 8, 16, 64}, {8192, 1024, 64, 1}, {2, 8, 16, 64},
                                          {8192, 1024, 64, 1}, vDims, contigStrides(vDims));
        fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        gm::GraphView view(graph);

        gm::PatternBuilder pb;
        pb.addNode("sdpa_fwd", {{"q", "$q"}, {"k", "$k"}, {"v", "$v"}}, {{"o", "$o"}});
        pb.addPredicate("hipdnn.same_head_dim", {V("$q"), V("$k"), V("$v")});
        EXPECT_FALSE(gm::Matcher::match(pb.build(), view).matched);
    }
}

// =========================================================================
// PatternBuilder::addPredicate -- compile-time validation (throws)
// =========================================================================

TEST(Predicates, AddPredicateRejectsUnknownName) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    EXPECT_THROW(pb.addPredicate("hipdnn.not_a_predicate", {V("$a"), V("$b")}),
                 std::invalid_argument);
}

TEST(Predicates, AddPredicateRejectsWrongArity) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    // same_dtype declares 2 args; one is too few.
    EXPECT_THROW(pb.addPredicate("hipdnn.same_dtype", {V("$a")}), std::invalid_argument);
    // ...and three is too many.
    EXPECT_THROW(pb.addPredicate("hipdnn.same_dtype", {V("$a"), V("$b"), V("$c")}),
                 std::invalid_argument);
}

// Symbol/literal where a Tensor is expected: same_dtype(Tensor,Tensor) given a
// symbol arg resolves to Int -> kind mismatch.
TEST(Predicates, AddPredicateRejectsIntWhereTensorExpected) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.bindDim("$a", 1, "k");  // bind k so the failure is the kind check, not unknown-symbol
    EXPECT_THROW(pb.addPredicate("hipdnn.same_dtype", {S("k"), V("$b")}), std::invalid_argument);
}

// Variable where an Int is expected: divisible_by(Int,Int) given a var arg
// resolves to Tensor -> kind mismatch.
TEST(Predicates, AddPredicateRejectsTensorWhereIntExpected) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    EXPECT_THROW(pb.addPredicate("hipdnn.divisible_by", {V("$a"), L(4)}), std::invalid_argument);
}

TEST(Predicates, AddPredicateRejectsUnknownVar) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    // $missing never appears on an edge.
    EXPECT_THROW(pb.addPredicate("hipdnn.same_dtype", {V("$a"), V("$missing")}),
                 std::invalid_argument);
}

TEST(Predicates, AddPredicateRejectsUnknownSymbol) {
    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    // "m" is never bound via a shape/bindDim.
    EXPECT_THROW(pb.addPredicate("hipdnn.divisible_by", {S("m"), L(4)}), std::invalid_argument);
}

// =========================================================================
// Provenance trust rule (builder + fromJson)
// =========================================================================

// A registered author predicate is usable when the pattern is Builtin-provenance
// but rejected under DropIn provenance -- via the builder.
TEST(Predicates, ProvenanceGatesAuthorPredicateInBuilder) {
    const gm::PredicateRegistry reg = pluginRegistry();

    gm::PatternBuilder ok(gm::OpSchemaRegistry::builtin(), gm::Provenance::Builtin, reg);
    ok.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    EXPECT_NO_THROW(ok.addPredicate("test.plugin_true", {V("$a")}));

    gm::PatternBuilder blocked(gm::OpSchemaRegistry::builtin(), gm::Provenance::DropIn, reg);
    blocked.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    EXPECT_THROW(blocked.addPredicate("test.plugin_true", {V("$a")}), std::invalid_argument);
}

// Same trust rule through the JSON compiler: Builtin provenance compiles the
// author predicate, DropIn rejects it with a non-empty error (fromJson never
// throws).
TEST(Predicates, ProvenanceGatesAuthorPredicateInJson) {
    const gm::PredicateRegistry reg = pluginRegistry();
    const char* json = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"id":"mm","op":"matmul",
                "operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
      "constraints":[{"kind":"native_predicate","name":"test.plugin_true","args":["$a"]}]
    })";

    auto okRes = gm::PatternCompiler::fromJson(json, gm::OpSchemaRegistry::builtin(), {},
                                               gm::Provenance::Builtin, reg);
    EXPECT_TRUE(okRes.ok) << okRes.error;

    auto dropRes = gm::PatternCompiler::fromJson(json, gm::OpSchemaRegistry::builtin(), {},
                                                 gm::Provenance::DropIn, reg);
    EXPECT_FALSE(dropRes.ok);
    EXPECT_FALSE(dropRes.error.empty());
}

// =========================================================================
// Match-time registry resolution (fail-closed vs plugin registry)
// =========================================================================

// A pattern that compiled with an author predicate (under a registry that knows
// it) fails CLOSED when matched with the default builtin() registry, and matches
// when the plugin registry is passed to match(). This is the load-bearing
// registry-independence guarantee: the pattern carries only the name.
TEST(Predicates, MatchTimeRegistryResolutionFailsClosedWithoutPlugin) {
    const gm::PredicateRegistry reg = pluginRegistry();

    gm::PatternBuilder pb(gm::OpSchemaRegistry::builtin(), gm::Provenance::Builtin, reg);
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.addPredicate("test.plugin_true", {V("$a")});
    const gm::CompiledPattern pattern = pb.build();

    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    // Default registry (builtin() only) cannot resolve "test.plugin_true".
    EXPECT_FALSE(gm::Matcher::match(pattern, view).matched);
    // The same pattern matches once the plugin registry resolves the name.
    EXPECT_TRUE(gm::Matcher::match(pattern, view, {}, reg).matched);
}

// A predicate whose arg is an absent optional operand (var never bound) is
// skipped -- the match still succeeds. Teeth: were the arg evaluated, same_dtype
// would see a null tensor and the negated form below would still not save it;
// here the non-negated same_dtype must pass purely because the arg is unbound.
TEST(Predicates, PredicateOnUnboundOptionalArgIsSkipped) {
    auto builder = makeSinglePwNoIn1Graph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("pointwise", {{"in_0", "$x"}, {"in_1", "$opt", true}}, {{"out_0", "$y"}});
    // $opt is an absent optional operand -> unbound at match time -> predicate skipped.
    pb.addPredicate("hipdnn.same_dtype", {V("$x"), V("$opt")});
    EXPECT_TRUE(gm::Matcher::match(pb.build(), view).matched);
}

// =========================================================================
// JSON native_predicate (via fromJson)
// =========================================================================

TEST(Predicates, JsonNativePredicateCompilesAndMatches) {
    const char* json = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"id":"mm","op":"matmul",
                "operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
      "constraints":[{"kind":"native_predicate","name":"hipdnn.same_dtype","args":["$a","$b"]}]
    })";
    auto res = gm::PatternCompiler::fromJson(json);
    ASSERT_TRUE(res.ok) << res.error;

    auto builder = util::createValidMatmulGraph();  // a,b both FLOAT
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);
    EXPECT_TRUE(gm::Matcher::match(res.pattern, view).matched);
}

TEST(Predicates, JsonNativePredicateNegatedInverts) {
    const char* json = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"id":"mm","op":"matmul",
                "operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
      "constraints":[{"kind":"native_predicate","name":"hipdnn.same_dtype",
                      "args":["$a","$b"],"negated":true}]
    })";
    auto res = gm::PatternCompiler::fromJson(json);
    ASSERT_TRUE(res.ok) << res.error;

    auto builder = util::createValidMatmulGraph();  // a,b same dtype
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);
    EXPECT_FALSE(gm::Matcher::match(res.pattern, view).matched);  // negated same_dtype
}

TEST(Predicates, JsonSameHeadDimCompilesAndMatches) {
    const char* json = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"id":"s","op":"sdpa_fwd",
                "operands":{"q":"$q","k":"$k","v":"$v"},"results":{"o":"$o"}}],
      "constraints":[{"kind":"native_predicate","name":"hipdnn.same_head_dim",
                      "args":["$q","$k","$v"]}]
    })";
    auto res = gm::PatternCompiler::fromJson(json);
    ASSERT_TRUE(res.ok) << res.error;

    auto builder = util::createValidSdpaFwdGraph();  // q/k/v last dim 64
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);
    EXPECT_TRUE(gm::Matcher::match(res.pattern, view).matched);
}

TEST(Predicates, JsonRejectsUnknownPredicateName) {
    const char* json = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"id":"mm","op":"matmul",
                "operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
      "constraints":[{"kind":"native_predicate","name":"hipdnn.nope","args":["$a","$b"]}]
    })";
    auto res = gm::PatternCompiler::fromJson(json);
    EXPECT_FALSE(res.ok);
    EXPECT_FALSE(res.error.empty());
}

TEST(Predicates, JsonRejectsMalformedPredicateArg) {
    // An arg that is neither a $var/symbol string nor an integer (here a nested
    // array) is malformed -> compile fails closed.
    const char* json = R"({
      "schema":"hipdnn.criteria/v1",
      "nodes":[{"id":"mm","op":"matmul",
                "operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
      "constraints":[{"kind":"native_predicate","name":"hipdnn.same_dtype",
                      "args":["$a",["$b"]]}]
    })";
    auto res = gm::PatternCompiler::fromJson(json);
    EXPECT_FALSE(res.ok);
    EXPECT_FALSE(res.error.empty());
}

}  // namespace
