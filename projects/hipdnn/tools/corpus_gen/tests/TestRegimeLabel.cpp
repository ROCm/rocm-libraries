// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/RegimeFocus.hpp>
#include <hipdnn_corpus_gen/RegimeLabel.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

/// @file TestRegimeLabel.cpp
/// @brief Which population a problem is reported under.
///
/// Labels come from the declaration: the shipped SDPA one, plus inline declarations for facets
/// SDPA does not express.

using namespace hipdnn_corpus_gen;

namespace
{

OperationMetadata shippedSdpa()
{
    return test::shippedOperation("sdpa_fwd");
}

/// The single parameter most inline declarations here need.
const char* const GROUPS = R"({"groups": {"type": "int64"}})";

/// Loads an operation `toy` over @p parameters (a JSON object), with the graph builder every
/// inline declaration here names. @p members adds top-level members, e.g. a `"regime_label"`.
MetadataLoad loadToy(const std::string& parameters, const std::string& members = {})
{
    auto declaration = nlohmann::json::parse(R"({
      "schema_version": "0.1",
      "operation": "toy",
      "stratification_axis": "arithmetic_intensity",
      "graph_builder": {
        "function": "createValidLayernormFpropGraph",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": []
      }
    })");
    declaration["parameters"] = nlohmann::json::parse(parameters);
    if(!members.empty())
    {
        // A named object: `items()` over a temporary dangles.
        const auto extra = nlohmann::json::parse("{" + members + "}");
        for(const auto& [key, value] : extra.items())
        {
            declaration[key] = value;
        }
    }
    return parseOperationMetadata(declaration);
}

/// The budget the shipped attention declarations' focused searches run under: 600 proposals for
/// each combination, doubled up to 4 times, and 20 first-pass points for each.
ExplorationRequest focusedSearchRequest(uint64_t seed)
{
    ExplorationRequest request;
    request.budgetPerCombination = 600;
    request.budgetGrowthLimit = 4;
    request.pointsPerCombination = 20;
    request.seed = seed;
    return request;
}

/// Multi-head by default, so the grouping facet stays fixed in the phase and context cases.
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
    // Decode is KV-cache bandwidth bound, prefill compute bound; they must not share a label.
    const auto metadata = shippedSdpa();
    ASSERT_FALSE(metadata.regimeLabel.empty()) << "sdpa_fwd declares no regime_label";

    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 4096)), "decode_long_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(2048, 2048)), "prefill_short_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(128, 8192)), "append_long_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(512, 512)), "prefill_short_mha");
}

TEST(TestRegimeLabel, TheShippedDeclarationSeparatesGroupedAttentionFromMultiHead)
{
    // Query heads per KV head decide how much cache is re-read.
    const auto metadata = shippedSdpa();

    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 4096, 32)), "decode_long_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 4096, 8)), "decode_long_gqa");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 4096, 1)), "decode_long_mqa");
}

TEST(TestRegimeLabel, TheContextBoundaryIsInclusiveAtTheDeclaredValue)
{
    // 2048 is short and 2049 is long, matching shapes.py's `seqlen_kv <= LONG_CONTEXT`.
    const auto metadata = shippedSdpa();
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(8, 2048)), "append_short_mha");
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(8, 2049)), "append_long_mha");
}

TEST(TestRegimeLabel, ClausesCascadeSoLaterOnesNeedNotRestateTheEarlier)
{
    // seqlen_q == 1 with seqlen_k == 1 matches both decode and prefill; first match wins, so
    // clauses need not restate earlier negations.
    const auto metadata = shippedSdpa();
    EXPECT_EQ(regimeLabel(metadata, sdpaPoint(1, 1)), "decode_short_mha");
}

TEST(TestRegimeLabel, AnUnmatchedPointTakesTheDeclaredOtherwiseLabel)
{
    // An unlabelled population is not actionable, so the declaration must name it.
    const auto load = loadToy(GROUPS, R"("regime_label": [
        {"name": "grouping",
         "labels": [{"label": "single", "when": {"==": ["$q.groups", 1]}}],
         "otherwise": "grouped"}
      ])");
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    EXPECT_EQ(regimeLabel(*load.metadata, ProblemPoint{{"groups", int64_t{1}}}), "single");
    EXPECT_EQ(regimeLabel(*load.metadata, ProblemPoint{{"groups", int64_t{8}}}), "grouped");
}

TEST(TestRegimeLabel, ThreeFacetsJoinInDeclarationOrder)
{
    // Declaration order, not map order: "prefill_short_mha" is what readers and the manifest
    // column expect.
    const auto load = loadToy(
        R"({
        "seqlen_q": {"type": "int64"}, "seqlen_k": {"type": "int64"},
        "heads": {"type": "int64"}, "heads_kv": {"type": "int64"}
      })",
        R"("regime_label": [
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
      ])");
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
    // §4.4 check, as for constraints: otherwise every point falls into "otherwise".
    const auto load = loadToy(GROUPS, R"("regime_label": [
        {"name": "phase",
         "labels": [{"label": "decode", "when": {"==": ["$q.seqlen_q", 1]}}],
         "otherwise": "other"}
      ])");

    ASSERT_FALSE(load.errors.empty());
    EXPECT_NE(load.errors.front().find("seqlen_q"), std::string::npos) << load.errors.front();
}

TEST(TestRegimeLabel, AnOperationThatNamesNoPopulationsGetsAnEmptyLabel)
{
    const auto load = loadToy(GROUPS);
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    EXPECT_TRUE(load.metadata->regimeLabel.empty());
    EXPECT_EQ(regimeLabel(*load.metadata, ProblemPoint{{"groups", int64_t{1}}}), "");
}

namespace
{

/// A small budget keeps these real-walk tests quick.
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
    // decode is `seqlen_q == 1`, mha is `heads_kv == heads`; short is an inequality left to the
    // label check. Pinning more would narrow the population.
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
    // Otherwise it would burn the budget and report a nonexistent population as saturated.
    const auto metadata = shippedSdpa();
    std::string error;
    EXPECT_FALSE(compileRegimeFocus(metadata, "decode_medium_mha", error).has_value());
    EXPECT_NE(error.find("decode_medium_mha"), std::string::npos);
    EXPECT_FALSE(compileRegimeFocus(metadata, "decode_short", error).has_value());
}

TEST(TestRegimeFocus, AFocusedSearchProposesInsideTheRegime)
{
    // Two equalities a free walk almost never meets; pinned and tied, results are all in-regime.
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
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "prefill_short_gqa", error);
    ASSERT_TRUE(focus.has_value()) << error;
    // Three batches in four are held, as a large --exclude-corpus holds most of a regime.
    const ProblemOracle mostBatchesAreHeld
        = [](const ProblemPoint& point) { return std::get<int64_t>(point.at("batch")) % 4 != 1; };
    // A small box, so the walk sees most of the region and held points crowd the cells.
    auto request = focusRequest();
    request.numericCeiling = 32;

    const auto found
        = exploreRegime(metadata, *focus, request, 20, SERVES_EVERYTHING, mostBatchesAreHeld, {});

    // Held points take cells, not budget: thousands of servable points remain, so the search
    // must deliver all 20 rather than stop capped or saturated.
    EXPECT_GE(found.problems.size(), 20u)
        << "saturated=" << found.saturated << " capped=" << found.searchCapped;
    EXPECT_FALSE(found.saturated);
    EXPECT_FALSE(found.searchCapped);
    for(const auto& point : found.problems)
    {
        EXPECT_EQ(std::get<int64_t>(point.at("batch")) % 4, 1);
        EXPECT_EQ(regimeLabel(metadata, point), "prefill_short_gqa");
    }
}

TEST(TestRegimeFocus, ARegimeTheEngineDoesNotServeIsNotReachedRatherThanSaturated)
{
    // Long context is `seqlen_k > 2048`; with none served the search found nothing, which shows
    // it did not reach the regime and nothing of whether a walk could: not saturation, and not
    // a budget limit either.
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "decode_long_mha", error);
    ASSERT_TRUE(focus.has_value()) << error;
    const ProblemOracle shortOnly
        = [](const ProblemPoint& point) { return std::get<int64_t>(point.at("seqlen_k")) <= 1024; };

    const auto found = exploreRegime(metadata, *focus, focusRequest(), 10, shortOnly, {}, {});

    EXPECT_TRUE(found.problems.empty());
    EXPECT_FALSE(found.saturated);
    EXPECT_FALSE(found.searchCapped);
}

TEST(TestRegimeFocus, ARegimeInACorpusWhoseEveryCombinationIsEmptyIsSaturated)
{
    // The ragged form is declared only with bshd, so every combination of these two keeps is
    // empty: nothing is left to search, which is a proof.
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "decode_long_mha", error);
    ASSERT_TRUE(focus.has_value()) << error;
    auto request = focusRequest();
    request.maxCombinations = 1024;
    request.keep = {{"form", "ragged"}, {"layout", "bhsd"}};

    const auto found = exploreRegime(metadata, *focus, request, 10, SERVES_EVERYTHING, {}, {});

    EXPECT_TRUE(found.problems.empty());
    EXPECT_TRUE(found.saturated);
}

TEST(TestRegimeFocus, ARegimeTheDeclaredConstraintsExcludeIsSaturatedWithoutAskingTheEngine)
{
    // Cross attention (more queries than keys) is admitted only under a non-causal mask, so
    // under a causal keep every proposal in the regime breaks a constraint.
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "cross_long_gqa", error);
    ASSERT_TRUE(focus.has_value()) << error;
    auto request = focusRequest();
    request.keep = {{"is_causal", "true"}};
    int64_t asked = 0;
    const ProblemOracle counting = [&asked](const ProblemPoint&) {
        ++asked;
        return true;
    };

    const auto found = exploreRegime(metadata, *focus, request, 3, counting, {}, {});

    EXPECT_TRUE(found.problems.empty());
    EXPECT_GT(found.inRegime, 0);
    EXPECT_EQ(found.admissible, 0);
    EXPECT_EQ(asked, 0);
    EXPECT_TRUE(found.saturated);
}

TEST(TestRegimeFocus, AnUnservedRegimeCostsOnlyTheProbe)
{
    // Each combination is probed before it gets the full budget and its doublings, as the
    // first pass does: an engine that serves none of the regime costs at most one probe each.
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "decode_long_mha", error);
    ASSERT_TRUE(focus.has_value()) << error;
    const ProblemOracle shortOnly
        = [](const ProblemPoint& point) { return std::get<int64_t>(point.at("seqlen_k")) <= 1024; };
    auto request = focusRequest();
    request.probeBudget = 200;

    const auto found = exploreRegime(metadata, *focus, request, 10, shortOnly, {}, {});

    EXPECT_TRUE(found.problems.empty());
    EXPECT_FALSE(found.saturated);
    // The sample is capped at maxCombinations, plus one extra combination per declared value it
    // missed, so the cap is what the enumeration actually yields.
    size_t total = 0;
    const auto combinations
        = detail::categoricalCombinations(metadata.parameters, request.maxCombinations, total);
    EXPECT_LE(found.proposed, static_cast<int64_t>(combinations.size()) * request.probeBudget);
}

TEST(TestRegimeFocus, EveryLabelTheFacetsCanSpellIsListedAndCompiles)
{
    // `--regime-floor`'s list: each facet's labels and `otherwise`, crossed; every label compiles
    // to a focus, even one the engine may not serve.
    const auto metadata = shippedSdpa();
    const auto labels = declaredRegimeLabels(metadata);
    size_t expected = 1;
    for(const auto& axis : metadata.regimeLabel)
    {
        std::set<std::string> options(axis.labels.begin(), axis.labels.end());
        options.insert(axis.otherwise);
        expected *= options.size();
    }
    EXPECT_EQ(labels.size(), expected);
    EXPECT_EQ(std::set<std::string>(labels.begin(), labels.end()).size(), labels.size());
    for(const auto& label : labels)
    {
        std::string error;
        EXPECT_TRUE(compileRegimeFocus(metadata, label, error).has_value()) << error;
    }
    EXPECT_NE(std::find(labels.begin(), labels.end(), "decode_short_mha"), labels.end());
    EXPECT_NE(std::find(labels.begin(), labels.end(), "append_long_gqa"), labels.end());
}

TEST(TestRegimeFocus, AFloorIsAskedOfEveryRegimeNoQuotaNames)
{
    const auto metadata = shippedSdpa();
    const auto floors = regimeFloors(metadata, {{"decode_short_mha", 9}});
    EXPECT_EQ(floors.count("decode_short_mha"), 0U) << "a named quota is not overridden";
    EXPECT_EQ(floors.size(), declaredRegimeLabels(metadata).size() - 1);
    EXPECT_EQ(floors.at("decode_long_gqa").pins, (std::map<std::string, int64_t>{{"seqlen_q", 1}}));
}
namespace
{

/// Each shipped attention declaration (forward and backward), whose weighted facets differ.
class TestShippedValueWeights : public ::testing::TestWithParam<std::string>
{
};

} // namespace

TEST_P(TestShippedValueWeights,
       AFocusedSearchRealisesTheDeclaredValueWeightsAndProbesOnlyTheLanesItNeeds)
{
    // layout 2:1:1 and scale 2:1:1 in both, window 3:1:1:1 in sdpa_fwd, which alone declares one;
    // the form is unweighted. The regime quota and floor path splits its need over categorical
    // lanes, so a lane-even split would make every value 1/3 (window none 1/4) however the main
    // search is weighted.
    const bool forward = GetParam() == "sdpa_fwd";
    const auto metadata = test::shippedOperation(GetParam());
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "decode_long_mha", error);
    ASSERT_TRUE(focus.has_value()) << error;
    const auto request = focusedSearchRequest(5);

    constexpr size_t ROWS = 2400;
    const auto found = exploreRegime(metadata, *focus, request, ROWS, SERVES_EVERYTHING, {}, {});

    ASSERT_GE(found.problems.size(), ROWS);
    // The scale is a free facet only under the forms that state one: scale_unset and
    // scale_tensor take inv_sqrt_d, a constraint holds it there. The declared 1/2 is realised
    // over every point, so over every point the weighted and the unweighted share both fall
    // within the tolerance of 0.5 (about 0.5 and 0.47 in sdpa_bwd). Among the points whose form states
    // the scale the weight is what remains once the forced share, the rows of those two forms,
    // is taken out, and the unweighted 1/3 is further from it than the tolerance.
    const auto* formParameter = metadata.find("form");
    ASSERT_NE(formParameter, nullptr);
    EXPECT_TRUE(formParameter->weights.empty()) << "the forms are even strata, not weighted";
    const auto forcedRows = static_cast<double>(
        std::count_if(found.problems.begin(), found.problems.end(), [](const ProblemPoint& point) {
            const auto& form = std::get<std::string>(point.at("form"));
            return form == "scale_unset" || form == "scale_tensor";
        }));
    const auto forced = forcedRows / static_cast<double>(found.problems.size());
    const auto statedInvSqrtD = (0.5 - forced) / (1.0 - forced);
    const auto shareOf
        = [&found](const std::string& parameter, const std::string& value, bool statesScale) {
              size_t held = 0;
              size_t counted = 0;
              for(const auto& point : found.problems)
              {
                  const auto& form = std::get<std::string>(point.at("form"));
                  if(statesScale && (form == "scale_unset" || form == "scale_tensor"))
                  {
                      continue;
                  }
                  ++counted;
                  held += std::get<std::string>(point.at(parameter)) == value ? 1 : 0;
              }
              return static_cast<double>(held) / static_cast<double>(counted);
          };
    EXPECT_NEAR(shareOf("layout", "bshd", false), 0.5, 0.05);
    EXPECT_NEAR(shareOf("layout", "bhsd", false), 0.25, 0.05);
    EXPECT_NEAR(shareOf("scale", "inv_sqrt_d", false), 0.5, 0.1);
    EXPECT_NEAR(shareOf("scale", "inv_sqrt_d", true), statedInvSqrtD, 0.03);
    // No form is starved: each holds at least 0.4 of an even share, though the three that fewer
    // combinations can hold (scale_unset, scale_tensor, ragged) hold less than the others.
    for(const auto& form : formParameter->values)
    {
        EXPECT_GE(shareOf("form", form, false),
                  0.4 / static_cast<double>(formParameter->values.size()))
            << form;
    }
    if(forward)
    {
        EXPECT_NEAR(shareOf("window", "none", false), 0.5, 0.1);
        EXPECT_NEAR(shareOf("window", "local_128", false), 1.0 / 6.0, 0.08);
    }

    // A floor of 60 needs at most 60 lanes of the declared lanes; running each of them once would
    // cost its whole budget.
    const auto few = exploreRegime(metadata, *focus, request, 60, SERVES_EVERYTHING, {}, {});
    ASSERT_GE(few.problems.size(), 60u);
    size_t total = 0;
    const auto lanes
        = detail::categoricalCombinations(metadata.parameters, request.maxCombinations, total)
              .size();
    EXPECT_LT(few.proposed, static_cast<int64_t>(lanes) * request.budgetPerCombination / 8)
        << "proposed " << few.proposed << " over " << lanes << " lanes";
}

INSTANTIATE_TEST_SUITE_P(SdpaDeclarations,
                         TestShippedValueWeights,
                         ::testing::Values("sdpa_fwd", "sdpa_bwd"));

TEST(TestRegimeFocus, AFocusedSearchUnderAKeepSpendsItsLanesOnTheCombinationsTheKeepAdmits)
{
    // The forward's 6912 combinations are past the bound of 1024, and a sample drawn before the
    // keep leaves a few of the 6 combinations the keep admits that can hold a problem, or none.
    const auto metadata = test::shippedOperation("sdpa_fwd");
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "decode_short_mha", error);
    ASSERT_TRUE(focus.has_value()) << error;
    auto request = focusedSearchRequest(3);
    request.keep = test::narrowSdpaFwdKeeps();

    const auto found = exploreRegime(
        metadata, *focus, request, 240, test::admitsOnlyWhatKeepKeeps(request.keep), {}, {});

    ASSERT_GE(found.problems.size(), 240U);
    std::set<std::string> combinations;
    for(const auto& point : found.problems)
    {
        EXPECT_EQ(regimeLabel(metadata, point), "decode_short_mha");
        combinations.insert(std::get<std::string>(point.at("alignment")) + "/"
                            + asText(point.at("is_causal")) + "/"
                            + asText(point.at("generate_stats")));
    }
    // Every combination of is_causal, alignment and generate_stats the declaration allows at
    // seqlen_q = 1: a causal bottom-right mask is a different problem from top-left, a
    // non-causal one has a single alignment.
    EXPECT_EQ(combinations.size(), 6U);
}

TEST(TestRegimeFocus, EveryRegimeDrawsItsOwnLanesInItsOwnOrder)
{
    // Lane order comes from tie keys; with only the seed in them every regime would try the same
    // lanes first and a sparse regime would be built from the same few each time.
    const auto metadata = shippedSdpa();
    const auto request = focusedSearchRequest(1);

    const auto firstLanes = [&](const std::string& regime) {
        std::string error;
        const auto focus = compileRegimeFocus(metadata, regime, error);
        EXPECT_TRUE(focus.has_value()) << error;
        std::vector<std::string> order;
        const ProblemOracle record = [&order](const ProblemPoint& point) {
            ProblemPoint lane;
            for(const auto* name : {"is_causal",
                                    "alignment",
                                    "dtype",
                                    "generate_stats",
                                    "layout",
                                    "scale",
                                    "window",
                                    "form"})
            {
                lane[name] = point.at(name);
            }
            const auto key = detail::describe(lane);
            if(std::find(order.begin(), order.end(), key) == order.end())
            {
                order.push_back(key);
            }
            return true;
        };
        exploreRegime(metadata, *focus, request, 40, record, {}, {});
        return order;
    };

    const auto a = firstLanes("decode_long_mha");
    const auto b = firstLanes("decode_long_mqa");
    ASSERT_GE(a.size(), 5U);
    ASSERT_GE(b.size(), 5U);
    EXPECT_NE(std::vector<std::string>(a.begin(), a.begin() + 5),
              std::vector<std::string>(b.begin(), b.begin() + 5));
}

TEST(TestRegimeFocus, ARegimeOnlyOneRareLaneServesIsFoundNotReportedSaturated)
{
    // saturated means every lane was tried and none could give more. The one lane the engine
    // serves here (kv_bhsd, jitter, local_4096) is among the smallest by weight, so it is the
    // last a weighted search reaches; stopping early would report an engine that serves nothing.
    const auto metadata = shippedSdpa();
    std::string error;
    const auto focus = compileRegimeFocus(metadata, "decode_long_mha", error);
    ASSERT_TRUE(focus.has_value()) << error;
    ExplorationRequest request;
    request.budgetPerCombination = 400;
    request.budgetGrowthLimit = 2;
    request.pointsPerCombination = 20;
    request.seed = 1;
    const ProblemOracle onlyTheRareLane = [](const ProblemPoint& point) {
        return std::get<std::string>(point.at("layout")) == "kv_bhsd"
               && std::get<std::string>(point.at("scale")) == "jitter"
               && std::get<std::string>(point.at("window")) == "local_4096";
    };

    const auto found = exploreRegime(metadata, *focus, request, 5, onlyTheRareLane, {}, {});

    EXPECT_GE(found.problems.size(), 5U);
    EXPECT_FALSE(found.saturated);
}

namespace
{

/// A form of a shipped attention declaration, over the point `sdpaPoint` builds.
ProblemPoint sdpaFormPoint(const std::string& form, int64_t seqlenQ = 2048)
{
    auto point = sdpaPoint(seqlenQ, 2048);
    point["form"] = form;
    return point;
}

/// The forms of @p metadata, as its own `form` parameter declares them.
std::vector<std::string> declaredForms(const OperationMetadata& metadata)
{
    const auto* form = metadata.find("form");
    return form == nullptr ? std::vector<std::string>{} : form->values;
}

/// Each shipped attention declaration (forward and backward), whose forms differ in number.
class TestShippedForms : public ::testing::TestWithParam<std::string>
{
protected:
    static OperationMetadata metadata()
    {
        return test::shippedOperation(GetParam());
    }
};

} // namespace

TEST_P(TestShippedForms, TheDeclarationNamesTheFormAndLeavesTheRegimeAlone)
{
    const auto metadata = TestShippedForms::metadata();
    ASSERT_EQ(metadata.formLabel.size(), 1U);
    EXPECT_EQ(metadata.formLabel.front().name, "form");
    const auto forms = declaredForms(metadata);
    ASSERT_FALSE(forms.empty());

    // Forms cross regimes: every form of one problem shape is the same regime, so no regime
    // name (or quota keyed on one) changes when the form does.
    const auto plain = regimeLabel(metadata, sdpaFormPoint("plain"));
    for(const auto& form : forms)
    {
        EXPECT_EQ(regimeLabel(metadata, sdpaFormPoint(form)), plain) << form;
        EXPECT_EQ(formQuotaKeys(metadata, sdpaFormPoint(form)),
                  std::vector<std::string>{"form:" + form});
    }
    EXPECT_EQ(plain, "prefill_short_mha");
    EXPECT_EQ(declaredRegimeLabels(metadata).size(), 4U * 2U * 3U);
}

TEST(TestFormLabel, AFormAxisOverAnUndeclaredParameterIsALoadError)
{
    const auto load = loadToy(GROUPS, R"("form_label": [
        {"name": "form",
         "labels": [{"label": "padded", "when": {"==": ["$q.form", "padded"]}}],
         "otherwise": "plain"}
      ])");

    ASSERT_FALSE(load.errors.empty());
    EXPECT_NE(load.errors.front().find("form_label"), std::string::npos) << load.errors.front();
    EXPECT_NE(load.errors.front().find("form"), std::string::npos) << load.errors.front();
}

TEST(TestFormLabel, ALabelWithAColonCouldReadAsAFormKeyAndIsRefused)
{
    const auto load = loadToy(GROUPS, R"("regime_label": [
        {"name": "phase",
         "labels": [{"label": "form:padded", "when": {"==": ["$q.groups", 1]}}],
         "otherwise": "other"}
      ])");

    ASSERT_FALSE(load.errors.empty());
    EXPECT_NE(load.errors.front().find("form:padded"), std::string::npos) << load.errors.front();
}

namespace
{

/// A `form` enum of three values labelled by @p labels, a clause per entry, `plain` otherwise.
MetadataLoad loadFormOver(const std::string& clauses)
{
    return loadToy(R"({"form": {"type": "enum", "values": ["plain", "padded", "bias"]}})",
                   R"("form_label": [{"name": "form", "labels": [)" + clauses
                       + R"(], "otherwise": "plain"}])");
}

std::string equalsClause(const std::string& value)
{
    return R"({"label": ")" + value + R"(", "when": {"==": ["$q.form", ")" + value + R"("]}})";
}

} // namespace

TEST(TestFormLabel, AnEnumAxisWithAClauseForEveryValueButOtherwiseLoads)
{
    const auto load = loadFormOver(equalsClause("padded") + "," + equalsClause("bias"));
    EXPECT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());
}

TEST(TestFormLabel, AnEnumValueTheAxisNeitherLabelsNorMakesOtherwiseIsALoadError)
{
    // A form added to the enum and not to the labels would read as `plain`, and a floor on it
    // would never be asked.
    const auto load = loadFormOver(equalsClause("padded"));

    ASSERT_FALSE(load.ok());
    EXPECT_NE(load.errors.front().find("'bias'"), std::string::npos) << load.errors.front();
}

TEST(TestFormLabel, AClauseComparingAValueTheEnumDoesNotDeclareIsALoadError)
{
    // A misspelt clause never matches, so the form it names would be labelled `plain`.
    const auto load = loadFormOver(equalsClause("padded") + "," + equalsClause("bias") + ","
                                   + equalsClause("paded"));

    ASSERT_FALSE(load.ok());
    EXPECT_NE(load.errors.front().find("'paded'"), std::string::npos) << load.errors.front();
}

TEST_P(TestShippedForms, EveryLabelOfTheAxisCompilesToAFocusKeyedAxisColonLabel)
{
    const auto metadata = TestShippedForms::metadata();
    const auto floors = formFloors(metadata);

    std::set<std::string> keys;
    for(const auto& floor : floors)
    {
        keys.insert(floor.first);
    }
    std::set<std::string> declared;
    for(const auto& form : declaredForms(metadata))
    {
        declared.insert("form:" + form);
    }
    ASSERT_FALSE(declared.empty());
    EXPECT_EQ(keys, declared);

    const auto& padded = floors.at("form:padded");
    EXPECT_EQ(padded.label, "padded");
    EXPECT_EQ(padded.formAxis, "form");
    EXPECT_EQ(padded.categoricalPins.at("form"), ParameterValue(std::string("padded")));
    EXPECT_TRUE(padded.pins.empty());

    // `otherwise` implies only negations, so it pins nothing and the label check decides.
    EXPECT_TRUE(floors.at("form:plain").categoricalPins.empty());
}

TEST_P(TestShippedForms, ALabelTheAxisDoesNotDeclareIsRefused)
{
    const auto metadata = TestShippedForms::metadata();
    std::string error;
    EXPECT_FALSE(compileFormFocus(metadata, "form", "paged", error).has_value());
    EXPECT_NE(error.find("paged"), std::string::npos) << error;
    error.clear();
    EXPECT_FALSE(compileFormFocus(metadata, "layout", "padded", error).has_value());
    EXPECT_NE(error.find("layout"), std::string::npos) << error;
}

TEST_P(TestShippedForms, AFocusedSearchReturnsOnlyPointsOfThatForm)
{
    const auto metadata = TestShippedForms::metadata();
    const auto floors = formFloors(metadata);
    for(const auto* form : {"padded", "plain"})
    {
        const auto found = exploreRegime(metadata,
                                         floors.at(std::string("form:") + form),
                                         focusRequest(),
                                         6,
                                         SERVES_EVERYTHING,
                                         {},
                                         {});
        ASSERT_FALSE(found.problems.empty()) << form;
        for(const auto& point : found.problems)
        {
            EXPECT_EQ(std::get<std::string>(point.at("form")), form);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SdpaDeclarations,
                         TestShippedForms,
                         ::testing::Values("sdpa_fwd", "sdpa_bwd"));
