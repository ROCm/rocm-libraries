// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/RegimeFocus.hpp>
#include <hipdnn_corpus_gen/RegimeLabel.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

/// @file TestRegimeLabel.cpp
/// @brief Which population a problem is reported under.
///
/// The table these reproduce is the retired Python assembler's, which computed the same label
/// from SDPA field names in C++'s absence. The point of the port is that the label now
/// comes from the declaration, so the tests read declarations -- the shipped one as an artifact,
/// and an inline one for the facets SDPA cannot yet express.

using namespace hipdnn_corpus_gen;

namespace
{

OperationMetadata shippedSdpa()
{
    const std::string path
        = std::string(HIPDNN_CORPUS_GEN_OPERATIONS_DIR) + "/sdpa_fwd.opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    return *parsed.metadata;
}

/// Multi-head unless a caller says otherwise, so the shipped declaration's third facet is a
/// constant across the phase and context cases below rather than a second thing moving in them.
ProblemPoint sdpaPoint(int64_t seqlenQ, int64_t seqlenK, int64_t headsKv = 32)
{
    return ProblemPoint{{"batch", int64_t{1}},
                        {"heads", int64_t{32}},
                        {"heads_kv", headsKv},
                        {"seqlen_q", seqlenQ},
                        {"seqlen_k", seqlenK},
                        {"head_dim", int64_t{128}},
                        {"is_causal", true},
                        {"dtype", std::string("bf16")}};
}

} // namespace

TEST(TestRegimeLabel, TheShippedDeclarationSeparatesDecodeFromPrefill)
{
    // These two are the populations a single aggregate regret number hides: decode is memory
    // bound on the KV cache, prefill is compute bound on the score matrix, and a model that is
    // excellent on one can be useless on the other.
    const auto metadata = shippedSdpa();
    ASSERT_FALSE(metadata.regimeLabel.empty()) << "sdpa_fwd declares no regime_label";

    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 4096)), "decode_long_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(2048, 2048)), "prefill_short_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(128, 8192)), "append_long_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(512, 512)), "prefill_short_mha");
}

TEST(TestRegimeLabel, TheShippedDeclarationSeparatesGroupedAttentionFromMultiHead)
{
    // How many query heads share one KV head decides how much of the cache a wavefront
    // re-reads, which is the difference between a decode that is bandwidth bound and one that
    // is not. Same phase, same context, three populations.
    const auto metadata = shippedSdpa();

    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 4096, 32)), "decode_long_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 4096, 8)), "decode_long_gqa");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 4096, 1)), "decode_long_mqa");
}

TEST(TestRegimeLabel, TheContextBoundaryIsInclusiveAtTheDeclaredValue)
{
    // 2048 is short and 2049 is long, per shapes.py's `seqlen_kv <= LONG_CONTEXT`. An off-by-one
    // here moves whole archetypes between buckets and changes every per-regime row.
    const auto metadata = shippedSdpa();
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(8, 2048)), "append_short_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(8, 2049)), "append_long_mha");
}

TEST(TestRegimeLabel, ClausesCascadeSoLaterOnesNeedNotRestateTheEarlier)
{
    // seqlen_q == 1 against seqlen_k == 1 satisfies BOTH the decode clause and the prefill one.
    // First match wins, so it is decode -- the reading that lets the facets be written as the
    // cascade they are rather than as four mutually exclusive predicates, each restating its
    // predecessors' negations.
    const auto metadata = shippedSdpa();
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 1)), "decode_short_mha");
}

TEST(TestRegimeLabel, AnUnmatchedPointTakesTheDeclaredOtherwiseLabel)
{
    // Not a silent "other": an unlabelled population is a row in the regret table that nobody
    // can act on, so the declaration has to name it.
    const auto load = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "0.1",
      "operation": "toy",
      "stratification_axis": "arithmetic_intensity",
      "graph_builder": {
        "function": "createValidLayernormFpropGraph",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": []
      },
      "parameters": {"groups": {"type": "int64"}},
      "regime_label": [
        {"name": "grouping",
         "labels": [{"label": "single", "when": {"==": ["$q.groups", 1]}}],
         "otherwise": "grouped"}
      ]
    })"));
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    EXPECT_EQ(regimeLabel(*load.metadata, ProblemPoint{{"groups", int64_t{1}}}), "single");
    EXPECT_EQ(regimeLabel(*load.metadata, ProblemPoint{{"groups", int64_t{8}}}), "grouped");
}

TEST(TestRegimeLabel, ThreeFacetsJoinInDeclarationOrder)
{
    // The full shapes.py label is phase_context_grouping. The order is the declaration's, not a
    // map's collation, because the label is what a reader greps for and what a manifest column
    // sorts by -- "prefill_short_mha" and "mha_short_prefill" are not interchangeable to either.
    const auto load = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "0.1",
      "operation": "toy_sdpa",
      "stratification_axis": "arithmetic_intensity",
      "graph_builder": {
        "function": "createValidLayernormFpropGraph",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": []
      },
      "parameters": {
        "seqlen_q": {"type": "int64"}, "seqlen_k": {"type": "int64"},
        "heads": {"type": "int64"}, "heads_kv": {"type": "int64"}
      },
      "regime_label": [
        {"name": "phase",
         "labels": [
           {"label": "decode", "when": {"==": ["$q.seqlen_q", 1]}},
           {"label": "cross", "when": {">": ["$q.seqlen_q", "$q.seqlen_k"]}},
           {"label": "prefill", "when": {"==": ["$q.seqlen_q", "$q.seqlen_k"]}}],
         "otherwise": "append"},
        {"name": "context",
         "labels": [{"label": "short", "when": {"<=": ["$q.seqlen_k", 2048]}}],
         "otherwise": "long"},
        {"name": "grouping",
         "labels": [
           {"label": "mha", "when": {"==": ["$q.heads", "$q.heads_kv"]}},
           {"label": "mqa", "when": {"==": ["$q.heads_kv", 1]}}],
         "otherwise": "gqa"}
      ]
    })"));
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    const auto label = [&load](int64_t sq, int64_t sk, int64_t hq, int64_t hkv) {
        return regimeLabel(
            *load.metadata,
            ProblemPoint{{"seqlen_q", sq}, {"seqlen_k", sk}, {"heads", hq}, {"heads_kv", hkv}});
    };

    EXPECT_EQ(label(2048, 2048, 32, 32), "prefill_short_mha");
    EXPECT_EQ(label(1, 8192, 64, 8), "decode_long_gqa");
    EXPECT_EQ(label(1, 128, 32, 1), "decode_short_mqa");
    EXPECT_EQ(label(256, 77, 16, 16), "cross_short_mha");
    EXPECT_EQ(label(128, 4096, 8, 2), "append_long_gqa");
}

TEST(TestRegimeLabel, AnAxisOverAnUndeclaredParameterIsALoadError)
{
    // The same §4.4 check constraints get. A facet referencing a parameter the operation does
    // not have labels every point "otherwise" -- one bucket holding the whole corpus, reported
    // as though it were a stratification.
    const auto load = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "0.1",
      "operation": "toy",
      "stratification_axis": "arithmetic_intensity",
      "graph_builder": {
        "function": "createValidLayernormFpropGraph",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": []
      },
      "parameters": {"groups": {"type": "int64"}},
      "regime_label": [
        {"name": "phase",
         "labels": [{"label": "decode", "when": {"==": ["$q.seqlen_q", 1]}}],
         "otherwise": "other"}
      ]
    })"));

    ASSERT_FALSE(load.errors.empty());
    EXPECT_NE(load.errors.front().find("seqlen_q"), std::string::npos) << load.errors.front();
}

TEST(TestRegimeLabel, AnOperationThatNamesNoPopulationsGetsAnEmptyLabel)
{
    // Coverage is "whatever has a declaration", so a declaration without this block must load
    // and produce a corpus -- with no regime column of its own rather than an invented one.
    const auto load = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "0.1",
      "operation": "toy",
      "stratification_axis": "arithmetic_intensity",
      "graph_builder": {
        "function": "createValidLayernormFpropGraph",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": []
      },
      "parameters": {"groups": {"type": "int64"}}
    })"));
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    EXPECT_TRUE(load.metadata->regimeLabel.empty());
    EXPECT_EQ(regimeLabel(*load.metadata, ProblemPoint{{"groups", int64_t{1}}}), "");
}

namespace
{

/// A small, fast search: every regime test drives the real walk over the shipped declaration,
/// so the budget is what keeps them quick.
ExplorationRequest focusRequest()
{
    ExplorationRequest request;
    request.budgetPerCombination = 1500;
    request.budgetGrowthLimit = 4;
    request.pointsPerCombination = 20;
    request.maxCombinations = 4;
    request.seed = 7;
    return request;
}

const ProblemOracle SERVES_EVERYTHING = [](const ProblemPoint&) { return true; };

} // namespace

TEST(TestRegimeFocus, AFocusPinsAndTiesTheEqualitiesItsLabelRequires)
{
    // decode is `seqlen_q == 1`, mha is `heads_kv == heads`; short is an inequality and is left
    // to the label check. Nothing else is fixed: a focus that pinned more than the label says
    // would generate a narrower population than the one it was asked for.
    const auto metadata = shippedSdpa();
    std::string error;

    const auto decode = compileRegimeFocus(metadata, "decode_short_mha", error);
    ASSERT_TRUE(decode.has_value()) << error;
    EXPECT_EQ(decode->pins, (std::map<std::string, int64_t>{{"seqlen_q", 1}}));
    EXPECT_EQ(decode->ties, (std::map<std::string, std::string>{{"heads_kv", "heads"}}));

    const auto prefill = compileRegimeFocus(metadata, "prefill_long_gqa", error);
    ASSERT_TRUE(prefill.has_value()) << error;
    EXPECT_TRUE(prefill->pins.empty());
    EXPECT_EQ(prefill->ties, (std::map<std::string, std::string>{{"seqlen_q", "seqlen_k"}}));

    const auto append = compileRegimeFocus(metadata, "append_short_mqa", error);
    ASSERT_TRUE(append.has_value()) << error;
    EXPECT_EQ(append->pins, (std::map<std::string, int64_t>{{"heads_kv", 1}}));
    EXPECT_TRUE(append->ties.empty()) << "append is an `otherwise` label and requires no equality";
}

TEST(TestRegimeFocus, ALabelNoDeclaredFacetSpellsIsRefused)
{
    // Searched for, it would spend the whole budget and come back saturated -- "the engine
    // serves none of these" -- about a population that does not exist.
    const auto metadata = shippedSdpa();
    std::string error;
    EXPECT_FALSE(compileRegimeFocus(metadata, "decode_medium_mha", error).has_value());
    EXPECT_NE(error.find("decode_medium_mha"), std::string::npos);
    EXPECT_FALSE(compileRegimeFocus(metadata, "decode_short", error).has_value());
}

TEST(TestRegimeFocus, AFocusedSearchProposesInsideTheRegime)
{
    // Multi-head decode: two equalities a walk over independent extents meets almost never.
    // With them pinned and tied, most of what the walk proposes is already in the regime, and
    // everything it returns is.
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "decode_short_mha", error);
    ASSERT_TRUE(focus.has_value()) << error;

    const auto found
        = exploreRegime(metadata, *focus, focusRequest(), 40, SERVES_EVERYTHING, {}, {});

    EXPECT_GE(found.problems.size(), 40u);
    EXPECT_GE(found.inRegime * 2, found.proposed)
        << found.inRegime << " of " << found.proposed << " proposals landed in the regime";
    for(const auto& point : found.problems)
    {
        EXPECT_EQ(regimeLabel(metadata, point), "decode_short_mha");
        EXPECT_TRUE(detail::satisfiesConstraints(metadata, point));
    }
}

TEST(TestRegimeFocus, HeldPointsAreNeverReturned)
{
    // A point already in the pools, or in a corpus it must not repeat, is not a new problem.
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "prefill_short_gqa", error);
    ASSERT_TRUE(focus.has_value()) << error;
    const ProblemOracle evenBatchIsHeld
        = [](const ProblemPoint& point) { return std::get<int64_t>(point.at("batch")) % 2 == 0; };

    const auto found = exploreRegime(
        metadata, *focus, focusRequest(), 20, SERVES_EVERYTHING, evenBatchIsHeld, {});

    ASSERT_FALSE(found.problems.empty());
    for(const auto& point : found.problems)
    {
        EXPECT_NE(std::get<int64_t>(point.at("batch")) % 2, 0);
        EXPECT_EQ(regimeLabel(metadata, point), "prefill_short_gqa");
    }
}

TEST(TestRegimeFocus, ARegimeTheEngineDoesNotServeIsReportedSaturated)
{
    // Long context is `seqlen_k > 2048`; an engine that serves nothing past 1024 has none. That
    // is a finding for the caller, and it must read as saturation, not as a budget limit.
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "decode_long_mha", error);
    ASSERT_TRUE(focus.has_value()) << error;
    const ProblemOracle shortOnly
        = [](const ProblemPoint& point) { return std::get<int64_t>(point.at("seqlen_k")) <= 1024; };

    const auto found = exploreRegime(metadata, *focus, focusRequest(), 10, shortOnly, {}, {});

    EXPECT_TRUE(found.problems.empty());
    EXPECT_TRUE(found.saturated);
    EXPECT_FALSE(found.searchCapped);
}
