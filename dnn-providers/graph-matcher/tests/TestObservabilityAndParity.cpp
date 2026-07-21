// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Phase-8 (final) observability + parity tests. Three surfaces are exercised,
// all through public APIs against real FlatBuffer graphs:
//
//   (A) The match EXPLAINER: MatchOptions::explain gates MatchResult::diagnostic.
//       Off (default) the diagnostic is empty even on failure (zero hot-path
//       cost); on, a failing match records a terse, structured near-miss note.
//       Assertions look for the specific token the run() / describeConstraint /
//       checkPredicates paths emit (constraint#<i>+kind, predicate#<i>+name,
//       "no structural placement"), not mere non-emptiness -- so a regression in
//       WHICH reason is reported reddens.
//
//   (B) PatternSet METRICS + the log seam: add() counts registered/duplicates
//       and drives the LogFn; firstMatch/rankedMatches count matchAttempts,
//       matchSuccesses, and budgetAborts. Each counter is asserted against a
//       constructed scenario where a plausible miscount would show.
//
//   (C) The SDPA-fwd PARITY GATE: the committed criterion (embedded verbatim for
//       cwd-independence, plus one file-read test that skips if the file is not
//       reachable) is compiled and matched over a fixture matrix. Each fixture
//       asserts the criterion decision EQUALS a reference predicate computed from
//       the fixture's dtype/rank/mask/head parameters -- the graph-expressible
//       subset of SdpaFwdPlanBuilder::isApplicable. Encoding the reference makes
//       the parity explicit: a criterion regression on any expressible axis
//       diverges from the reference and reddens.
//
// Matcher/PatternSet/fromJson never throw; failures are asserted on result
// fields (matched / diagnostic / metrics), never via EXPECT_THROW.

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_graph_matcher/PatternCompiler.hpp>
#include <hipdnn_graph_matcher/PatternSet.hpp>
#include <hipdnn_graph_matcher/Predicate.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace gm = hipdnn::graph_matcher;
namespace fbu = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace util = hipdnn_test_sdk::utilities;
namespace data = hipdnn_flatbuffers_sdk::data_objects;

constexpr auto kNpos = std::string::npos;

// --- Helpers -------------------------------------------------------------

using Spec = gm::PatternBuilder::PredArgSpec;
Spec S(std::string_view name) {
    return Spec{gm::PredicateArg::Source::Sym, name, 0};
}
Spec L(int64_t value) {
    return Spec{gm::PredicateArg::Source::Literal, {}, value};
}

// Row-major contiguous strides for `dims` (last dim = 1).
std::vector<int64_t> contigStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> s(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;) {
        s[i - 1] = s[i] * dims[i];
    }
    return s;
}

// Compiles JSON, aborting the test on a compile error so no assertion runs
// against a default-constructed pattern.
gm::CompiledPattern compile(std::string_view json) {
    const gm::CompileResult cr = gm::PatternCompiler::fromJson(json);
    EXPECT_TRUE(cr.ok) << cr.error;
    return cr.pattern;
}

// A bare matmul pattern, matches createValidMatmulGraph.
const char* const kBroadMatmul = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}]
})";

// A distinct (dtype-constrained) matmul; also matches createValidMatmulGraph
// (FLOAT). Used only to have a second, non-duplicate registrant.
const char* const kSpecificMatmul = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
  "constraints":[{"on":"$a","dtype":"FLOAT"}]
})";

// A conv_fwd pattern: compiles, never matches a matmul-only graph.
const char* const kConvFwd = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"conv_fwd","operands":{"x":"$x","w":"$w"},"results":{"y":"$y"}}]
})";

// A matmul with a dtype constraint the FLOAT graph violates: the structure
// places fully, then constraint#0 (dtype) rejects it.
const char* const kMatmulRequiresBf16 = R"({
  "schema":"hipdnn.criteria/v1",
  "nodes":[{"op":"matmul","operands":{"a":"$a","b":"$b"},"results":{"c":"$c"}}],
  "constraints":[{"on":"$a","dtype":{"one_of":["BFLOAT16"]}}]
})";

// The committed SDPA-fwd criterion, copied verbatim from
// dnn-providers/graph-matcher/criteria/sdpa_fwd.json so the parity matrix is
// cwd-independent. A separate test reads the on-disk file to guard drift.
const char* const kSdpaFwdCriterion = R"({
  "schema": "hipdnn.criteria/v1",
  "name": "rocke_sdpa_fwd_mfma",
  "priority": 100,
  "nodes": [
    {
      "id": "sdpa",
      "op": "sdpa_fwd",
      "operands": { "q": "$q", "k": "$k", "v": "$v" },
      "results": { "o": "$o" }
    }
  ],
  "constraints": [
    { "on": "$q", "dtype": { "one_of": ["BFLOAT16", "FP8_E4M3"] }, "rank": 4 },
    { "on": "$k", "rank": 4 },
    { "on": "$v", "rank": 4 },
    { "on": "$o", "dtype": { "one_of": ["BFLOAT16"] }, "rank": 4 },
    { "on": "sdpa", "attr": { "alibi_mask": { "equals": 0 } } },
    { "on": "sdpa", "attr": { "padding_mask": { "equals": 0 } } },
    { "kind": "same_dtype", "args": ["$q", "$k"] },
    { "kind": "same_dtype", "args": ["$q", "$v"] },
    { "kind": "same_dim", "args": ["$k", 1, "$v", 1] }
  ]
})";

// =========================================================================
// (A) Match explainer -- MatchOptions::explain / MatchResult::diagnostic
// =========================================================================

// Default options (explain off): a failing match leaves diagnostic empty, so
// the hot path pays nothing. Uses the constraint-fail scenario -- structure
// places, dtype rejects -- to prove the emptiness is the explain gate, not a
// structural short-circuit.
TEST(Explainer, ExplainOffLeavesDiagnosticEmptyOnFailure) {
    auto builder = util::createValidMatmulGraph();  // FLOAT
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    const gm::MatchResult r = gm::Matcher::match(compile(kMatmulRequiresBf16), view);
    EXPECT_FALSE(r.matched);
    EXPECT_TRUE(r.diagnostic.empty());
}

// explain on + a fully-placed constraint failure: the diagnostic names the
// rejecting constraint by index AND kind. Same scenario as the off test, so the
// only difference is the explain flag.
TEST(Explainer, ExplainOnNamesConstraintIndexAndKind) {
    auto builder = util::createValidMatmulGraph();  // FLOAT, not BFLOAT16
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::MatchOptions opt;
    opt.explain = true;
    const gm::MatchResult r = gm::Matcher::match(compile(kMatmulRequiresBf16), view, opt);

    EXPECT_FALSE(r.matched);
    EXPECT_NE(r.diagnostic.find("constraint#0"), kNpos) << r.diagnostic;
    EXPECT_NE(r.diagnostic.find("dtype"), kNpos) << r.diagnostic;
}

// explain on + a native-predicate failure: the diagnostic names the predicate
// by index AND name. divisible_by(k=8, 3) on matmul a's last dim fails (8 % 3),
// and the structure/shape constraints all pass, so the predicate is the reason.
TEST(Explainer, ExplainOnNamesPredicateIndexAndName) {
    auto builder = util::createValidMatmulGraph();  // a dims {4,8}
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternBuilder pb;
    pb.addNode("matmul", {{"a", "$a"}, {"b", "$b"}}, {{"c", "$c"}});
    pb.constrainShape("$a", {gm::DimSpec::any(), gm::DimSpec::of("k")});  // k := 8
    pb.addPredicate("hipdnn.divisible_by", {S("k"), L(3)});               // 8 % 3 != 0

    gm::MatchOptions opt;
    opt.explain = true;
    const gm::MatchResult r = gm::Matcher::match(pb.build(), view, opt);

    EXPECT_FALSE(r.matched);
    EXPECT_NE(r.diagnostic.find("predicate#0"), kNpos) << r.diagnostic;
    EXPECT_NE(r.diagnostic.find("divisible_by"), kNpos) << r.diagnostic;
}

// explain on + a purely structural miss (conv_fwd pattern over a matmul graph):
// no full placement is ever reached, so the diagnostic is the structural note,
// distinct from the constraint#/predicate# forms above.
TEST(Explainer, ExplainOnReportsStructuralMiss) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::MatchOptions opt;
    opt.explain = true;
    const gm::MatchResult r = gm::Matcher::match(compile(kConvFwd), view, opt);

    EXPECT_FALSE(r.matched);
    EXPECT_NE(r.diagnostic.find("no structural placement"), kNpos) << r.diagnostic;
    // Teeth: a structural miss must NOT be reported as a constraint/predicate reason.
    EXPECT_EQ(r.diagnostic.find("constraint#"), kNpos) << r.diagnostic;
    EXPECT_EQ(r.diagnostic.find("predicate#"), kNpos) << r.diagnostic;
}

// A successful match records no diagnostic even with explain on -- the near-miss
// note is a failure-only surface.
TEST(Explainer, SuccessLeavesDiagnosticEmptyEvenWhenExplainOn) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::MatchOptions opt;
    opt.explain = true;
    const gm::MatchResult r = gm::Matcher::match(compile(kBroadMatmul), view, opt);

    ASSERT_TRUE(r.matched);
    EXPECT_TRUE(r.diagnostic.empty());
}

// =========================================================================
// (B) PatternSet metrics + log seam
// =========================================================================

// add() counts each admitted pattern in `registered` and each rejected-as-dup
// in `duplicates`. Two distinct criteria + one repeat => registered==2,
// duplicates==1, and size tracks registered (the dup did not grow the set).
TEST(Metrics, AddCountsRegisteredAndDuplicates) {
    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "broad").ok);
    ASSERT_TRUE(set.add(compile(kSpecificMatmul), "specific").ok);
    const gm::AddResult dup = set.add(compile(kBroadMatmul), "broad-again");  // duplicate criterion
    ASSERT_FALSE(dup.ok);
    ASSERT_TRUE(dup.duplicate);

    EXPECT_EQ(set.metrics().registered, 2u);
    EXPECT_EQ(set.metrics().duplicates, 1u);
    EXPECT_EQ(set.size(), 2u);
}

// The log sink fires on each add() with a message that classifies the event:
// a registration line contains "registered", a duplicate line contains
// "duplicate". Captured in order; both events must appear.
TEST(Metrics, LogSinkEmitsRegisteredAndDuplicateLines) {
    std::vector<std::string> lines;
    gm::PatternSet set;
    set.setLogSink([&lines](std::string_view msg) { lines.emplace_back(msg); });

    ASSERT_TRUE(set.add(compile(kBroadMatmul), "broad").ok);
    ASSERT_FALSE(set.add(compile(kBroadMatmul), "broad-again").ok);  // duplicate

    ASSERT_EQ(lines.size(), 2u);
    EXPECT_NE(lines[0].find("registered"), kNpos) << lines[0];
    EXPECT_NE(lines[1].find("duplicate"), kNpos) << lines[1];
}

// rankedMatches tries every registered pattern (matchAttempts == size) and
// counts only the acceptors (matchSuccesses == number that matched). One conv
// (no match) + one matmul (match) over a matmul graph => attempts 2, successes 1.
TEST(Metrics, RankedMatchesCountsAttemptsAndSuccesses) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kConvFwd), "conv").ok);        // never matches
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "matmul").ok);  // matches

    const std::vector<gm::RankedMatch> ranked = set.rankedMatches(view);
    ASSERT_EQ(ranked.size(), 1u);
    EXPECT_EQ(set.metrics().matchAttempts, set.size());  // == 2, every entry tried
    EXPECT_EQ(set.metrics().matchSuccesses, 1u);
    EXPECT_EQ(set.metrics().budgetAborts, 0u);
}

// firstMatch short-circuits on the first acceptor: with a matching pattern at
// index 0, exactly one attempt is charged even though more are registered --
// the deliberate cheap-path contrast with rankedMatches.
TEST(Metrics, FirstMatchStopsAttemptsAtFirstHit) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "broad").ok);        // index 0, matches
    ASSERT_TRUE(set.add(compile(kSpecificMatmul), "specific").ok);  // index 1, would also match
    ASSERT_TRUE(set.add(compile(kConvFwd), "conv").ok);             // index 2

    EXPECT_EQ(set.firstMatch(view), 0);
    EXPECT_EQ(set.metrics().matchAttempts, 1u);  // stopped at the first hit, not size()==3
    EXPECT_EQ(set.metrics().matchSuccesses, 1u);
}

// budgetAborts counts attempts that trip the step budget. A generous budget
// matches and charges no abort; a zero budget forces the search to fail closed,
// incrementing budgetAborts (not matchSuccesses) and returning -1.
TEST(Metrics, BudgetAbortsCountsBudgetExceededAttempts) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    gm::PatternSet set;
    ASSERT_TRUE(set.add(compile(kBroadMatmul), "broad").ok);

    // Generous default budget: matches, no abort.
    EXPECT_EQ(set.firstMatch(view), 0);
    EXPECT_EQ(set.metrics().budgetAborts, 0u);

    // Zero budget: the first unify step trips the cap, so the attempt aborts.
    gm::MatchOptions tiny;
    tiny.stepBudget = 0;
    EXPECT_EQ(set.firstMatch(view, tiny), -1);

    EXPECT_EQ(set.metrics().matchAttempts, 2u);
    EXPECT_EQ(set.metrics().matchSuccesses, 1u);  // only the generous call succeeded
    EXPECT_EQ(set.metrics().budgetAborts, 1u);    // only the zero-budget call aborted
}

// =========================================================================
// (C) SDPA-fwd parity gate -- criterion vs a transcribed reference
// =========================================================================

// One fixture in the parity matrix. All tensors are built at `dtype` (the
// graph builder wires q/k/v/o to the single type), q is rank 3 or 4, the mask
// bools flow straight through to the SDPA node attrs, and kHeads/vHeads set
// dim(1) of k/v.
struct SdpaFixture {
    const char* name;
    data::DataType dtype;
    int qRank;  // 3 or 4; k/v/o stay rank 4
    bool alibi;
    bool padding;
    bool causal;
    int64_t kHeads;
    int64_t vHeads;
};

// The graph-expressible subset of SdpaFwdPlanBuilder::isApplicable, computed
// purely from the fixture parameters. Device string, head-size kernel table,
// stride-overflow, dropout/stats/attn-mask tensors, and override-shapes are
// provider-side and OUT of scope. Because the builder sets o to the same dtype
// as q/k/v, "o must be BFLOAT16" collapses the accepted dtype to BFLOAT16.
bool sdpaReference(const SdpaFixture& f) {
    const bool qDtypeAllowed =
        (f.dtype == data::DataType::BFLOAT16 || f.dtype == data::DataType::FP8_E4M3);
    const bool oDtypeBf16 = (f.dtype == data::DataType::BFLOAT16);  // o == dtype in the builder
    const bool allRank4 = (f.qRank == 4);                           // k/v/o are always rank 4 here
    const bool masksOff = (!f.alibi && !f.padding);                 // causal is allowed
    const bool sameHeads = (f.kHeads == f.vHeads);
    // q==k==v dtype holds by construction (single-dtype builder).
    return qDtypeAllowed && oDtypeBf16 && allRank4 && masksOff && sameHeads;
}

// Builds an SDPA-fwd graph from the fixture with contiguous strides throughout.
flatbuffers::FlatBufferBuilder buildSdpaGraph(const SdpaFixture& f) {
    const std::vector<int64_t> qDims =
        (f.qRank == 3) ? std::vector<int64_t>{8, 16, 64} : std::vector<int64_t>{2, 8, 16, 64};
    const std::vector<int64_t> kDims{2, f.kHeads, 16, 64};
    const std::vector<int64_t> vDims{2, f.vHeads, 16, 64};
    const std::vector<int64_t> oDims{2, 8, 16, 64};

    return util::createValidSdpaFwdGraph(qDims, contigStrides(qDims), kDims, contigStrides(kDims),
                                         vDims, contigStrides(vDims), oDims, contigStrides(oDims),
                                         f.dtype,
                                         /*withAttnMask=*/false,
                                         /*withScale=*/false,
                                         /*withStats=*/false, f.alibi, f.padding, f.causal);
}

const SdpaFixture kSdpaFixtures[] = {
    // dtype axis: bf16 accepts; half rejects (o not bf16); fp8 rejects (o not bf16).
    {"bf16_all_rank4_no_mask", data::DataType::BFLOAT16, 4, false, false, false, 8, 8},
    {"half_rejected", data::DataType::HALF, 4, false, false, false, 8, 8},
    {"fp8_o_not_bf16_rejected", data::DataType::FP8_E4M3, 4, false, false, false, 8, 8},
    // rank axis: rank-3 q rejects.
    {"rank3_q_rejected", data::DataType::BFLOAT16, 3, false, false, false, 8, 8},
    // mask axis: alibi/padding reject; causal accepts.
    {"alibi_rejected", data::DataType::BFLOAT16, 4, true, false, false, 8, 8},
    {"padding_rejected", data::DataType::BFLOAT16, 4, false, true, false, 8, 8},
    {"causal_accepted", data::DataType::BFLOAT16, 4, false, false, true, 8, 8},
    // head axis: k/v head-count mismatch rejects.
    {"kv_head_mismatch_rejected", data::DataType::BFLOAT16, 4, false, false, false, 8, 4},
};

// The criterion's accept/reject decision EQUALS the reference for every fixture.
// Asserting equality (not a bare literal) means any criterion regression on an
// expressible axis diverges from the reference and reddens.
TEST(SdpaParity, CriterionMatchesReferenceAcrossFixtureMatrix) {
    const gm::CompiledPattern pattern = compile(kSdpaFwdCriterion);

    for (const SdpaFixture& f : kSdpaFixtures) {
        auto builder = buildSdpaGraph(f);
        fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        gm::GraphView view(graph);

        const bool matched = gm::Matcher::match(pattern, view).matched;
        const bool expected = sdpaReference(f);
        EXPECT_EQ(matched, expected) << "fixture: " << f.name;
    }
}

// A non-SDPA graph is a structural reject: there is no sdpa_fwd node for the
// single-node criterion to place. The reference agrees (no sdpa node => reject).
TEST(SdpaParity, NonSdpaGraphIsStructuralReject) {
    auto builder = util::createValidMatmulGraph();
    fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    gm::GraphView view(graph);

    EXPECT_FALSE(gm::Matcher::match(compile(kSdpaFwdCriterion), view).matched);
}

// The committed criteria file, compiled from disk, decides identically to the
// embedded copy on a representative accept (bf16) and reject (half). Skips
// gracefully when the file is not reachable from the test's cwd, so a non-root
// cwd never turns this into a false failure.
TEST(SdpaParity, CommittedCriterionFileAgreesWithEmbedded) {
    std::ifstream file("dnn-providers/graph-matcher/criteria/sdpa_fwd.json");
    if (!file) {
        GTEST_SKIP() << "criteria/sdpa_fwd.json not reachable from cwd; embedded matrix covers it";
    }
    const std::string json((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    const gm::CompileResult cr = gm::PatternCompiler::fromJson(json);
    ASSERT_TRUE(cr.ok) << cr.error;

    const SdpaFixture accept{"bf16", data::DataType::BFLOAT16, 4, false, false, false, 8, 8};
    const SdpaFixture reject{"half", data::DataType::HALF, 4, false, false, false, 8, 8};

    for (const SdpaFixture& f : {accept, reject}) {
        auto builder = buildSdpaGraph(f);
        fbu::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        gm::GraphView view(graph);
        EXPECT_EQ(gm::Matcher::match(cr.pattern, view).matched, sdpaReference(f))
            << "file fixture: " << f.name;
    }
}

}  // namespace
