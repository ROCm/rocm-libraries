// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestProblemSpace.cpp
 * @brief Covers the one exploration, against declarations the test writes.
 *
 * Exploration is generic: each case drives the same function with different metadata and a
 * known-region oracle instead of an engine.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <nlohmann/json.hpp>

#include <map>
#include <set>
#include <string>

namespace hipdnn_corpus_gen
{
namespace
{

OperationMetadata metadataFor(const std::string& json)
{
    auto load = parseOperationMetadata(nlohmann::json::parse(json));
    EXPECT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());
    return load.metadata.value_or(OperationMetadata{});
}

/// Two numeric extents and one dtype: the smallest declaration that has both kinds of axis.
OperationMetadata twoDimsAndADtype()
{
    return metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": {
        "M":     { "type": "int64" },
        "N":     { "type": "int64" },
        "dtype": { "type": "enum", "values": ["fp32", "fp16"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "createValidMatmulGraph", "source": "x.hpp",
                         "arguments": [] }
    })");
}

const ProblemOracle ACCEPT_EVERYTHING = [](const ProblemPoint&) { return true; };

} // namespace

TEST(TestProblemSpace, EnumeratesEveryDeclaredDtypeRatherThanPickingOne)
{
    // dtype is a declared parameter, so every value is explored.
    ExplorationRequest request;
    request.pointsPerCombination = 10;
    request.seed = 1;

    const auto corpus = exploreProblemSpace(twoDimsAndADtype(), request, ACCEPT_EVERYTHING);

    ASSERT_EQ(corpus.combinations.size(), 2U);
    std::set<std::string> dtypes;
    for(const auto& point : corpus.problems())
    {
        dtypes.insert(std::get<std::string>(point.at("dtype")));
    }
    EXPECT_EQ(dtypes, (std::set<std::string>{"fp16", "fp32"}));
}

TEST(TestProblemSpace, SearchesTheNumericAxesAndEnumeratesTheCategoricalOnes)
{
    ExplorationRequest request;
    request.pointsPerCombination = 20;
    request.seed = 2;

    const auto corpus = exploreProblemSpace(twoDimsAndADtype(), request, ACCEPT_EVERYTHING);

    EXPECT_EQ(corpus.numericParameters, (std::vector<std::string>{"M", "N"}));

    std::set<int64_t> distinctM;
    for(const auto& point : corpus.combinations.front().problems)
    {
        distinctM.insert(std::get<int64_t>(point.at("M")));
    }
    EXPECT_GT(distinctM.size(), 5U);
}

TEST(TestProblemSpace, GivesEveryCombinationItsOwnBudget)
{
    // A shared budget would be spent by whichever combination runs first.
    ExplorationRequest request;
    request.pointsPerCombination = 15;
    request.seed = 3;

    const auto corpus = exploreProblemSpace(twoDimsAndADtype(), request, ACCEPT_EVERYTHING);

    ASSERT_EQ(corpus.combinations.size(), 2U);
    for(const auto& combination : corpus.combinations)
    {
        EXPECT_EQ(combination.problems.size(), 15U) << detail::describe(combination.categorical);
    }
}

TEST(TestProblemSpace, AppliesTheOracleToWholeProblemPoints)
{
    // Engines may couple categorical and numeric axes, so the oracle sees the whole point.
    const ProblemOracle halfOnlySmall = [](const ProblemPoint& point) {
        const auto dtype = std::get<std::string>(point.at("dtype"));
        return dtype == "fp32" || std::get<int64_t>(point.at("M")) <= 64;
    };

    ExplorationRequest request;
    request.pointsPerCombination = 20;
    request.seed = 4;

    const auto corpus = exploreProblemSpace(twoDimsAndADtype(), request, halfOnlySmall);

    for(const auto& point : corpus.problems())
    {
        if(std::get<std::string>(point.at("dtype")) == "fp16")
        {
            EXPECT_LE(std::get<int64_t>(point.at("M")), 64);
        }
    }
}

TEST(TestProblemSpace, ProducesOneProblemPerCombinationWhenNothingIsNumeric)
{
    // An all-categorical operation still has problems.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "flags_only",
      "parameters": {
        "dtype":  { "type": "enum", "values": ["fp32", "fp16"] },
        "causal": { "type": "bool" }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] }
    })");

    const auto corpus = exploreProblemSpace(metadata, {}, ACCEPT_EVERYTHING);

    EXPECT_TRUE(corpus.numericParameters.empty());
    EXPECT_EQ(corpus.problems().size(), 4U) << "two dtypes times two boolean values";
}

TEST(TestProblemSpace, SaysSoWhenItDidNotExploreEveryCombination)
{
    // Partial coverage must be reported, not silent.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "many_flags",
      "parameters": {
        "a": { "type": "enum", "values": ["1", "2", "3", "4"] },
        "b": { "type": "enum", "values": ["1", "2", "3", "4"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] }
    })");

    ExplorationRequest request;
    request.maxCombinations = 5;

    const auto corpus = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING);

    EXPECT_LE(corpus.combinations.size(), 5U);
    ASSERT_FALSE(corpus.skippedCombinations.empty());
    EXPECT_NE(corpus.skippedCombinations.front().find("16"), std::string::npos);
}

TEST(TestProblemSpace, HonoursASemanticRangeButNotAnAbsentOne)
{
    // §4.3.2: a declared range binds; an absent one leaves only the ceiling.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "ranged",
      "parameters": {
        "bounded":   { "type": "int64", "range": [2, 8] },
        "unbounded": { "type": "int64" }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] }
    })");

    ExplorationRequest request;
    request.pointsPerCombination = 40;
    request.numericCeiling = 1024;
    request.seed = 7;

    const auto corpus = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING);

    int64_t widestUnbounded = 0;
    for(const auto& point : corpus.problems())
    {
        const auto bounded = std::get<int64_t>(point.at("bounded"));
        EXPECT_GE(bounded, 2);
        EXPECT_LE(bounded, 8);
        widestUnbounded = std::max(widestUnbounded, std::get<int64_t>(point.at("unbounded")));
    }
    EXPECT_GT(widestUnbounded, 8) << "an undeclared range must not be narrowed to a declared one";
}

TEST(TestProblemSpace, IsReproducibleFromItsSeed)
{
    ExplorationRequest request;
    request.pointsPerCombination = 12;
    request.seed = 9;

    const auto first = exploreProblemSpace(twoDimsAndADtype(), request, ACCEPT_EVERYTHING);
    const auto second = exploreProblemSpace(twoDimsAndADtype(), request, ACCEPT_EVERYTHING);

    request.seed = 10;
    const auto other = exploreProblemSpace(twoDimsAndADtype(), request, ACCEPT_EVERYTHING);

    EXPECT_EQ(first.problems(), second.problems());
    EXPECT_NE(first.problems(), other.problems());
}

TEST(TestProblemSpace, DeclaredConstraintsKeepInvalidPointsOutOfTheSearch)
{
    // Ranges bound a parameter against constants; relations such as filter <= input need
    // declared constraints.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy_conv",
      "parameters": { "H": { "type": "int64" }, "R": { "type": "int64" } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "constraints": [ { "<=": ["$q.R", "$q.H"] } ]
    })");

    EXPECT_TRUE(
        detail::satisfiesConstraints(metadata, ProblemPoint{{"H", int64_t{8}}, {"R", int64_t{3}}}));
    EXPECT_FALSE(
        detail::satisfiesConstraints(metadata, ProblemPoint{{"H", int64_t{3}}, {"R", int64_t{8}}}));

    ExplorationRequest request;
    request.pointsPerCombination = 25;
    request.seed = 21;

    const auto corpus = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING);
    ASSERT_FALSE(corpus.problems().empty()) << "a satisfiable constraint must not empty the space";
    for(const auto& point : corpus.problems())
    {
        EXPECT_LE(std::get<int64_t>(point.at("R")), std::get<int64_t>(point.at("H")));
    }
}

TEST(TestProblemSpace, AnUnevaluableConstraintRejectsRatherThanAdmits)
{
    // A valid compiled expression may still be undefined for a problem.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": { "H": { "type": "int64" } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "constraints": [ { "<=": [{"/": ["$q.H", 0]}, 1] } ]
    })");

    EXPECT_FALSE(detail::satisfiesConstraints(metadata, ProblemPoint{{"H", int64_t{8}}}));
}

namespace
{

/// An engine that serves one dtype only, so one combination must carry the whole corpus.
ProblemOracle servesOnlyFp16(int64_t mLimit, int64_t nLimit, std::map<std::string, int64_t>* asked)
{
    return [=](const ProblemPoint& point) {
        const auto dtype = std::get<std::string>(point.at("dtype"));
        if(asked != nullptr)
        {
            ++(*asked)[dtype];
        }
        return dtype == "fp16" && std::get<int64_t>(point.at("M")) <= mLimit
               && std::get<int64_t>(point.at("N")) <= nLimit;
    };
}

} // namespace

TEST(TestProblemSpace, GrowsAServedCombinationToTheCorpusTarget)
{
    ExplorationRequest request;
    request.pointsPerCombination = 10;
    request.numericCeiling = 256;
    request.corpusTarget = 80;
    request.seed = 11;

    const auto corpus
        = exploreProblemSpace(twoDimsAndADtype(), request, servesOnlyFp16(256, 256, nullptr));

    EXPECT_GE(corpus.problems().size(), 80U) << "stopped at the first-pass target";
    EXPECT_TRUE(corpus.shortfall.empty());
    for(const auto& point : corpus.problems())
    {
        EXPECT_EQ(std::get<std::string>(point.at("dtype")), "fp16");
    }
}

TEST(TestProblemSpace, ReturnsFewerOnlyWhenTheServedRegionIsSpent)
{
    // Nine served points exist; asking for fifty returns nine and reports saturation.
    ExplorationRequest request;
    request.pointsPerCombination = 4;
    request.numericCeiling = 64;
    request.corpusTarget = 50;
    request.seed = 12;

    const auto corpus
        = exploreProblemSpace(twoDimsAndADtype(), request, servesOnlyFp16(3, 3, nullptr));

    EXPECT_LE(corpus.problems().size(), 9U);
    ASSERT_FALSE(corpus.shortfall.empty());
    bool sawSaturated = false;
    for(const auto& combination : corpus.combinations)
    {
        if(!combination.problems.empty())
        {
            EXPECT_TRUE(combination.saturated) << detail::describe(combination.categorical);
            EXPECT_FALSE(combination.searchCapped);
            sawSaturated = true;
        }
    }
    EXPECT_TRUE(sawSaturated);
}

TEST(TestProblemSpace, DoesNotGrowACombinationTheEngineDeclines)
{
    // Growing a declined combination would only re-prove that it serves nothing.
    ExplorationRequest request;
    request.pointsPerCombination = 10;
    request.numericCeiling = 256;
    request.seed = 13;

    std::map<std::string, int64_t> firstPassOnly;
    exploreProblemSpace(twoDimsAndADtype(), request, servesOnlyFp16(256, 256, &firstPassOnly));

    request.corpusTarget = 80;
    std::map<std::string, int64_t> grown;
    exploreProblemSpace(twoDimsAndADtype(), request, servesOnlyFp16(256, 256, &grown));

    EXPECT_EQ(grown["fp32"], firstPassOnly["fp32"]);
    EXPECT_GT(grown["fp16"], firstPassOnly["fp16"]);
}

TEST(TestProblemSpace, ReportsABudgetLimitAsASearchLimitNotAnEngineLimit)
{
    // Budget-limited is not saturated: more points exist.
    ExplorationRequest request;
    request.pointsPerCombination = 5;
    request.budgetPerCombination = 60;
    request.budgetGrowthLimit = 1;
    request.numericCeiling = 4096;
    request.corpusTarget = 2000;
    request.seed = 14;

    const auto corpus
        = exploreProblemSpace(twoDimsAndADtype(), request, servesOnlyFp16(4096, 4096, nullptr));

    ASSERT_FALSE(corpus.shortfall.empty());
    bool sawCapped = false;
    for(const auto& combination : corpus.combinations)
    {
        if(!combination.problems.empty())
        {
            EXPECT_TRUE(combination.searchCapped);
            EXPECT_FALSE(combination.saturated);
            sawCapped = true;
        }
    }
    EXPECT_TRUE(sawCapped);
}

TEST(TestProblemSpace, AnEngineThatServesOnlyHeldPointsIsSearchedNotDeclined)
{
    // rocKE serves only pack geometries the caller already holds; the search must still run
    // and report nothing new rather than treat the engine as declining.
    ExplorationRequest request;
    request.pointsPerCombination = 4;
    request.numericCeiling = 64;
    request.corpusTarget = 50;
    request.seed = 15;

    const ProblemOracle heldAll = [](const ProblemPoint& point) {
        return std::get<int64_t>(point.at("M")) <= 3 && std::get<int64_t>(point.at("N")) <= 3;
    };
    const auto corpus
        = exploreProblemSpace(twoDimsAndADtype(), request, servesOnlyFp16(3, 3, nullptr), heldAll);

    EXPECT_TRUE(corpus.problems().empty()) << "returned a point the caller already held";
    ASSERT_FALSE(corpus.shortfall.empty());
    bool sawFp16 = false;
    for(const auto& combination : corpus.combinations)
    {
        if(std::get<std::string>(combination.categorical.at("dtype")) == "fp16")
        {
            EXPECT_TRUE(combination.saturated) << "a served combination was never grown";
            sawFp16 = true;
        }
    }
    EXPECT_TRUE(sawFp16);
}

TEST(TestProblemSpace, HeldPointsDoNotWallTheSearchOffFromTheRestOfTheRegion)
{
    // Held points must not be treated as refused, or they fence the walk in.
    ExplorationRequest request;
    request.pointsPerCombination = 10;
    request.numericCeiling = 256;
    request.corpusTarget = 60;
    request.seed = 16;

    const ProblemOracle heldHalf
        = [](const ProblemPoint& point) { return std::get<int64_t>(point.at("M")) % 2 == 0; };
    const auto corpus = exploreProblemSpace(
        twoDimsAndADtype(), request, servesOnlyFp16(256, 256, nullptr), heldHalf);

    EXPECT_GE(corpus.problems().size(), 60U);
    EXPECT_TRUE(corpus.shortfall.empty());
    for(const auto& point : corpus.problems())
    {
        EXPECT_NE(std::get<int64_t>(point.at("M")) % 2, 0) << "returned a held point";
    }
}

TEST(TestProblemSpace, ACorpusExcludingAnEarlierOneStillReachesItsCount)
{
    // `--exclude-corpus`: the earlier corpus took the small shapes, where log-uniform footholds
    // land. Those configurations are valid and held, as CorpusGen holds them, so the walk
    // starts on them and moves past them to shapes no corpus has yet.
    ExplorationRequest request;
    request.pointsPerCombination = 10;
    request.numericCeiling = 256;
    request.corpusTarget = 60;
    request.seed = 17;

    const ProblemOracle inEarlierCorpus = [](const ProblemPoint& point) {
        return std::get<int64_t>(point.at("M")) <= 64 && std::get<int64_t>(point.at("N")) <= 64;
    };
    const auto corpus = exploreProblemSpace(
        twoDimsAndADtype(), request, servesOnlyFp16(256, 256, nullptr), inEarlierCorpus);

    EXPECT_GE(corpus.problems().size(), 60U);
    EXPECT_TRUE(corpus.shortfall.empty());
    for(const auto& point : corpus.problems())
    {
        EXPECT_FALSE(inEarlierCorpus(point)) << "returned a point of the excluded corpus";
    }
}

TEST(TestProblemSpace, AllOfAdmitsOnlyWhatEveryEngineServes)
{
    // Coverage overlaps on one band of M; the cross-engine corpus is that band.
    int laterAsked = 0;
    const ProblemOracle small
        = [](const ProblemPoint& point) { return std::get<int64_t>(point.at("M")) <= 64; };
    const ProblemOracle large = [&laterAsked](const ProblemPoint& point) {
        ++laterAsked;
        return std::get<int64_t>(point.at("M")) >= 32;
    };
    const auto both = allOf({small, large});

    EXPECT_TRUE(both(ProblemPoint{{"M", int64_t{48}}}));
    EXPECT_FALSE(both(ProblemPoint{{"M", int64_t{16}}}));
    const auto askedBefore = laterAsked;
    EXPECT_FALSE(both(ProblemPoint{{"M", int64_t{128}}}));
    EXPECT_EQ(laterAsked, askedBefore) << "a problem the first engine declines is not offered on";
    EXPECT_TRUE(allOf({})(ProblemPoint{{"M", int64_t{1}}})) << "no engine named, no constraint";
}

TEST(TestProblemSpace, ReportsTheServedEdgesOfEveryCombinationNotOnlyOfWhatItSelected)
{
    // The edges come from every point the engine accepted, so they bound the selection; and
    // each is a served point, inside the region the oracle describes.
    const ProblemOracle bounded = [](const ProblemPoint& point) {
        return std::get<int64_t>(point.at("M")) <= 300 && std::get<int64_t>(point.at("N")) <= 50;
    };
    ExplorationRequest request;
    request.pointsPerCombination = 8;
    request.seed = 11;

    const auto corpus = exploreProblemSpace(twoDimsAndADtype(), request, bounded);

    for(const auto& combination : corpus.combinations)
    {
        ASSERT_EQ(combination.lowest.size(), 2U);
        ASSERT_EQ(combination.highest.size(), 2U);
        for(size_t d = 0; d < 2; ++d)
        {
            const auto& name = corpus.numericParameters[d];
            const auto low = std::get<int64_t>(combination.lowest[d].at(name));
            const auto high = std::get<int64_t>(combination.highest[d].at(name));
            EXPECT_TRUE(bounded(combination.lowest[d]));
            EXPECT_TRUE(bounded(combination.highest[d]));
            EXPECT_EQ(combination.lowest[d].at("dtype"), combination.categorical.at("dtype"));
            for(const auto& point : combination.problems)
            {
                EXPECT_LE(low, std::get<int64_t>(point.at(name)));
                EXPECT_GE(high, std::get<int64_t>(point.at(name)));
            }
        }
    }
    // Reproducible from the seed, ties included.
    const auto again = exploreProblemSpace(twoDimsAndADtype(), request, bounded);
    EXPECT_EQ(detail::describe(again.combinations.front().highest.front()),
              detail::describe(corpus.combinations.front().highest.front()));
}

TEST(TestProblemSpace, NothingServedHasNoEdges)
{
    ExplorationRequest request;
    request.pointsPerCombination = 4;
    request.budgetPerCombination = 200;
    request.probeBudget = 100;
    const auto corpus = exploreProblemSpace(
        twoDimsAndADtype(), request, [](const ProblemPoint&) { return false; });
    for(const auto& combination : corpus.combinations)
    {
        EXPECT_TRUE(combination.lowest.empty());
        EXPECT_TRUE(combination.highest.empty());
    }
}

} // namespace hipdnn_corpus_gen
