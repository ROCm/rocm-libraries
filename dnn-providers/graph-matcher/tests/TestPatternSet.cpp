// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Phase-7 PatternSet tests: deterministic arbitration + load-time duplicate
// detection (add/firstMatch/rankedMatches) plus the top-level "name"/"priority"
// parsing added to PatternCompiler::fromJson. The matcher, compiler, and codec
// internals are covered by earlier phases and are not re-tested here -- patterns
// are built through fromJson and matched against real FlatBuffer graphs so every
// assertion defends an observable PatternSet contract. add/firstMatch/
// rankedMatches and fromJson never throw, so failures are asserted via return
// values (ok/duplicate/index/-1), never EXPECT_THROW.
//
// Arbitration order (highest first): (1) priority; (2) specificity (constraint-
// like clauses, then nodes, then bound edges); (3) stable id (name hash, else
// criterion hash), then exact criterion bytes, then registration index.
// Declaration/registration order is NEVER the primary key -- proved by adding in
// both orders and asserting an identical ranked NAME sequence. firstMatch is the
// one registration-order path (cheap isApplicable), deliberately distinct from
// ranked arbitration.

#include <gtest/gtest.h>

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_graph_matcher/PatternCompiler.hpp>
#include <hipdnn_graph_matcher/PatternSet.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;

// --- Helpers -------------------------------------------------------------

gm::VarId varOf(const gm::CompiledPattern& p, std::string_view name) {
    for (gm::VarId v = 0; v < p.varCount(); ++v) {
        if (p.varName(v) == name) {
            return v;
        }
    }
    return UINT32_MAX;
}

// Compiles JSON and returns the pattern, aborting the test on a compile error so
// arbitration assertions never run against a default-constructed pattern.
gm::CompiledPattern compile(std::string_view json) {
    const gm::CompileResult cr = gm::PatternCompiler::fromJson(json);
    EXPECT_TRUE(cr.ok) << cr.error;
    return cr.pattern;
}

// The ranked entries' names, in best-first order. Names are the stable identity
// used by every determinism assertion (registration index is add-order-specific
// and therefore not comparable across add orders).
std::vector<std::string> rankedNames(const gm::PatternSet& set, const gm::GraphView& view) {
    std::vector<std::string> names;
    for (const gm::RankedMatch& m : set.rankedMatches(view)) {
        names.push_back(set.at(m.index).name);
    }
    return names;
}

// A bare matmul: a->b => c, no constraints. Matches createValidMatmulGraph. Zero
// constraint-like clauses, so it is the least specific of any matmul pattern.
const char* const kBroadMatmul = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}]
})";

// The same matmul plus a dtype constraint and a same_dim cross-constraint (A[.,1]
// == B[0,.] == 8). Two constraint-like clauses => strictly more specific than
// kBroadMatmul, yet it still matches createValidMatmulGraph.
const char* const kSpecificMatmul = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
  "constraints":[
    {"on":"$a","dtype":"FLOAT"},
    {"kind":"same_dim","args":["$a",1,"$b",0]}
  ]
})";

// A conv_fwd node: valid, compiles, but never matches a matmul-only graph.
const char* const kConvFwd = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"conv_fwd","operands":{"x":"$x","w":"$w"},"results":{"y":"$y"}}]
})";

// Two matmuls of identical specificity (one dtype constraint each) but distinct
// criterion bytes (constraint on $a vs $b). Both match createValidMatmulGraph;
// their order is settled purely by the stable-id tiebreak.
const char* const kMatmulDtypeA = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
  "constraints":[{"on":"$a","dtype":"FLOAT"}]
})";
const char* const kMatmulDtypeB = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
  "constraints":[{"on":"$b","dtype":"FLOAT"}]
})";

// =========================================================================
// fromJson: name / priority descriptor parsing
// =========================================================================

// Top-level "name" and "priority" land in CompileResult, distinct from the
// pattern's criterion. A negative priority round-trips as a signed int64.
TEST(PatternSetParse, NameAndPriorityPresent) {
    const char* const json = R"({
      "schema":"hipdnn.criteria/v1",
      "name":"fused_matmul",
      "priority":-7,
      "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}]
    })";
    const gm::CompileResult cr = gm::PatternCompiler::fromJson(json);
    ASSERT_TRUE(cr.ok) << cr.error;
    EXPECT_EQ(cr.name, "fused_matmul");
    EXPECT_EQ(cr.priority, -7);
}

// Absent name/priority default to "" and 0, not to garbage.
TEST(PatternSetParse, DefaultsWhenAbsent) {
    const gm::CompileResult cr = gm::PatternCompiler::fromJson(kBroadMatmul);
    ASSERT_TRUE(cr.ok) << cr.error;
    EXPECT_EQ(cr.name, "");
    EXPECT_EQ(cr.priority, 0);
}

// A non-integer priority is a hard compile error (fail closed), not a silent
// truncation to 0 or 2. Both a fractional number and a string are rejected.
TEST(PatternSetParse, NonIntegerPriorityIsError) {
    const char* const fractional = R"({
      "schema":"hipdnn.criteria/v1",
      "priority":2.5,
      "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}]
    })";
    const char* const stringy = R"({
      "schema":"hipdnn.criteria/v1",
      "priority":"high",
      "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}]
    })";

    const gm::CompileResult frac = gm::PatternCompiler::fromJson(fractional);
    EXPECT_FALSE(frac.ok);
    EXPECT_FALSE(frac.error.empty());

    const gm::CompileResult str = gm::PatternCompiler::fromJson(stringy);
    EXPECT_FALSE(str.ok);
    EXPECT_FALSE(str.error.empty());
}

// =========================================================================
// add: load-time duplicate detection
// =========================================================================

// Under the default Reject policy a repeated criterion is refused: ok=false,
// duplicate=true, a non-empty error, and the set does not grow.
TEST(PatternSetAdd, RejectsDuplicateCriterion) {
    gm::PatternSet set;
    const gm::AddResult first = set.add(compile(kBroadMatmul), "one");
    ASSERT_TRUE(first.ok);
    EXPECT_FALSE(first.duplicate);
    EXPECT_EQ(first.index, 0u);
    EXPECT_EQ(set.size(), 1u);

    const gm::AddResult dup = set.add(compile(kBroadMatmul), "two");
    EXPECT_FALSE(dup.ok);
    EXPECT_TRUE(dup.duplicate);
    EXPECT_FALSE(dup.error.empty());
    EXPECT_EQ(set.size(), 1u);  // unchanged
}

// Under Skip the duplicate is dropped silently: ok=true, duplicate=true, and the
// set still does not grow.
TEST(PatternSetAdd, SkipsDuplicateCriterion) {
    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kBroadMatmul)).ok);
    ASSERT_EQ(set.size(), 1u);

    const gm::AddResult dup = set.add(compile(kBroadMatmul), "", 0, gm::DuplicatePolicy::Skip);
    EXPECT_TRUE(dup.ok);
    EXPECT_TRUE(dup.duplicate);
    EXPECT_EQ(set.size(), 1u);  // still unchanged
}

// Dedup keys on serialized criterion bytes only; name/priority are excluded. Two
// adds of the same criterion under different names still collide.
TEST(PatternSetAdd, DifferentNameStillDuplicates) {
    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "alpha", 5).ok);

    const gm::AddResult dup = set.add(compile(kBroadMatmul), "beta", 99);
    EXPECT_FALSE(dup.ok);
    EXPECT_TRUE(dup.duplicate);
    EXPECT_EQ(set.size(), 1u);
}

// A genuinely different criterion is admitted: ok=true, duplicate=false, and the
// registration index advances with the size.
TEST(PatternSetAdd, DistinctCriterionIsAdded) {
    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "broad").ok);

    const gm::AddResult second = set.add(compile(kSpecificMatmul), "specific");
    EXPECT_TRUE(second.ok);
    EXPECT_FALSE(second.duplicate);
    EXPECT_EQ(second.index, 1u);
    EXPECT_EQ(set.size(), 2u);
}

// =========================================================================
// rankedMatches: arbitration
// =========================================================================

// When two patterns of equal priority both match, the strictly more specific one
// ranks first -- and that order is identical regardless of add order, proving
// declaration order is not consulted.
TEST(PatternSetRank, SpecificityWinsAndIsOrderIndependent) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet broadFirst;
    ASSERT_TRUE(broadFirst.add(compile(kBroadMatmul), "broad").ok);
    ASSERT_TRUE(broadFirst.add(compile(kSpecificMatmul), "specific").ok);

    gm::PatternSet specificFirst;
    ASSERT_TRUE(specificFirst.add(compile(kSpecificMatmul), "specific").ok);
    ASSERT_TRUE(specificFirst.add(compile(kBroadMatmul), "broad").ok);

    const std::vector<std::string> expected{"specific", "broad"};
    EXPECT_EQ(rankedNames(broadFirst, view), expected);
    EXPECT_EQ(rankedNames(specificFirst, view), expected);  // same order, reversed adds
}

// Explicit priority overrides specificity: a less-specific pattern with higher
// priority ranks ahead of a more-specific, lower-priority one -- again identical
// across add orders.
TEST(PatternSetRank, PriorityOverridesSpecificity) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet a;
    ASSERT_TRUE(a.add(compile(kBroadMatmul), "broad", 100).ok);
    ASSERT_TRUE(a.add(compile(kSpecificMatmul), "specific", 0).ok);

    gm::PatternSet b;
    ASSERT_TRUE(b.add(compile(kSpecificMatmul), "specific", 0).ok);
    ASSERT_TRUE(b.add(compile(kBroadMatmul), "broad", 100).ok);

    const std::vector<std::string> expected{"broad", "specific"};
    EXPECT_EQ(rankedNames(a, view), expected);
    EXPECT_EQ(rankedNames(b, view), expected);
}

// Two distinct patterns of equal priority AND equal specificity resolve via the
// stable-id tiebreak. The exact winner depends on the name hash, but the order
// MUST be deterministic and independent of add order.
TEST(PatternSetRank, StableIdTiebreakIsOrderIndependent) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet forward;
    ASSERT_TRUE(forward.add(compile(kMatmulDtypeA), "alpha").ok);
    ASSERT_TRUE(forward.add(compile(kMatmulDtypeB), "beta").ok);

    gm::PatternSet reversed;
    ASSERT_TRUE(reversed.add(compile(kMatmulDtypeB), "beta").ok);
    ASSERT_TRUE(reversed.add(compile(kMatmulDtypeA), "alpha").ok);

    const std::vector<std::string> f = rankedNames(forward, view);
    const std::vector<std::string> r = rankedNames(reversed, view);
    ASSERT_EQ(f.size(), 2u);
    EXPECT_EQ(f, r);  // identical order despite reversed registration
}

// rankedMatches lists only acceptors: a registered non-matching pattern is
// absent, and the surviving match carries matched=true with the correct
// bindings read off the graph.
TEST(PatternSetRank, ReturnsOnlyAcceptorsWithBindings) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kConvFwd), "conv").ok);        // will not match
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "matmul").ok);  // will match

    const std::vector<gm::RankedMatch> ranked = set.rankedMatches(view);
    ASSERT_EQ(ranked.size(), 1u);  // conv excluded
    EXPECT_EQ(set.at(ranked[0].index).name, "matmul");
    ASSERT_TRUE(ranked[0].result.matched);

    const gm::CompiledPattern& mm = set.at(ranked[0].index).pattern;
    EXPECT_EQ(ranked[0].result.uidOf(varOf(mm, "$a")), 1);
    EXPECT_EQ(ranked[0].result.uidOf(varOf(mm, "$b")), 2);
    EXPECT_EQ(ranked[0].result.uidOf(varOf(mm, "$c")), 3);
}

// =========================================================================
// firstMatch: registration-order applicability
// =========================================================================

// firstMatch returns the FIRST registered matcher in registration order, even
// when a later entry is strictly more specific. This is the deliberate contrast
// with rankedMatches, which would put the specific one first.
TEST(PatternSetFirst, UsesRegistrationOrderNotRanking) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "broad").ok);        // index 0
    ASSERT_TRUE(set.add(compile(kSpecificMatmul), "specific").ok);  // index 1

    EXPECT_EQ(set.firstMatch(view), 0);  // broad, first registered
    // ranking disagrees: the specific pattern (index 1) is best.
    const std::vector<gm::RankedMatch> ranked = set.rankedMatches(view);
    ASSERT_EQ(ranked.size(), 2u);
    EXPECT_EQ(ranked[0].index, 1u);
}

// firstMatch skips a leading non-matching pattern and returns the next acceptor.
TEST(PatternSetFirst, SkipsNonMatchingLeadingEntry) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kConvFwd), "conv").ok);        // index 0, no match
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "matmul").ok);  // index 1, matches

    EXPECT_EQ(set.firstMatch(view), 1);
}

// firstMatch on an empty set (and on a set where nothing matches) returns -1.
TEST(PatternSetFirst, ReturnsMinusOneWhenNoneMatch) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet empty;
    EXPECT_EQ(empty.firstMatch(view), -1);

    gm::PatternSet nonMatching;
    ASSERT_TRUE(nonMatching.add(compile(kConvFwd), "conv").ok);
    EXPECT_EQ(nonMatching.firstMatch(view), -1);
}

}  // namespace
