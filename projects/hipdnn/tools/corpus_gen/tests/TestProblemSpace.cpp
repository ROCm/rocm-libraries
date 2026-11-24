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

#include "OperationsDir.hpp"
#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

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

    ASSERT_FALSE(corpus.problems().empty());
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

    // The bound, plus one combination for each declared value the scattered sample missed.
    EXPECT_GE(corpus.combinations.size(), 5U);
    EXPECT_LE(corpus.combinations.size(), 5U + 4U + 4U);
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

TEST(TestProblemSpace, AnAllCategoricalDeclarationKeepsItsConstraintsAndExclusions)
{
    // With no numeric parameter there is no search, but a constraint still forbids a
    // combination and a held point is still not offered again.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy_unary",
      "parameters": {
        "mode":  { "type": "enum", "values": ["RELU_FWD", "TANH_FWD"] },
        "dtype": { "type": "enum", "values": ["fp32", "fp16"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "constraints": [ { "or": [ { "!=": ["$q.dtype", "fp16"] },
                                 { "==": ["$q.mode", "RELU_FWD"] } ] } ]
    })");
    const ProblemOracle heldFp32Relu = [](const ProblemPoint& point) {
        return std::get<std::string>(point.at("mode")) == "RELU_FWD"
               && std::get<std::string>(point.at("dtype")) == "fp32";
    };

    const auto corpus
        = exploreProblemSpace(metadata, ExplorationRequest{}, ACCEPT_EVERYTHING, heldFp32Relu);

    std::set<std::string> found;
    for(const auto& point : corpus.problems())
    {
        found.insert(std::get<std::string>(point.at("mode")) + "/"
                     + std::get<std::string>(point.at("dtype")));
    }
    EXPECT_EQ(found, (std::set<std::string>{"TANH_FWD/fp32", "RELU_FWD/fp16"}));
}

TEST(TestProblemSpace, ACappedCategoricalSpaceStillHoldsEveryDeclaredValue)
{
    // 4 x 4 x 4 combinations capped at 8: a prefix would keep two values of the first
    // parameter; every value of every parameter must still reach the corpus.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy_modes",
      "parameters": {
        "a": { "type": "enum", "values": ["a0", "a1", "a2", "a3"] },
        "b": { "type": "enum", "values": ["b0", "b1", "b2", "b3"] },
        "c": { "type": "enum", "values": ["c0", "c1", "c2", "c3"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] }
    })");
    ExplorationRequest request;
    request.maxCombinations = 8;

    const auto corpus = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING);

    std::map<std::string, std::set<std::string>> seen;
    for(const auto& point : corpus.problems())
    {
        for(const auto& [name, value] : point)
        {
            seen[name].insert(std::get<std::string>(value));
        }
    }
    for(const auto& name : {"a", "b", "c"})
    {
        EXPECT_EQ(seen[name].size(), 4U) << name;
    }
    EXPECT_FALSE(corpus.skippedCombinations.empty()) << "the cap is still reported";
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
    ASSERT_FALSE(corpus.combinations.empty());
    for(const auto& combination : corpus.combinations)
    {
        EXPECT_TRUE(combination.lowest.empty());
        EXPECT_TRUE(combination.highest.empty());
    }
}

namespace
{

/// Two combinations, each with an archetype list and neighbourhoods wide enough that the
/// anchored draws are not limited by the number of distinct points they can reach.
OperationMetadata anchoredToy()
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
                         "arguments": [] },
      "archetypes": [
        { "name": "layers",
          "values": { "M": [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536],
                      "N": [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536] } }
      ],
      "neighbourhood": {
        "M": { "kind": "scale", "factors": [0.5, 0.75, 1, 1.5, 2] },
        "N": { "kind": "scale", "factors": [0.5, 1, 2] }
      },
      "mixture": { "archetypes": 0.2, "neighbourhood": 0.6, "exploration": 0.2 }
    })");
}

std::vector<std::string> described(const ProblemCorpus& corpus)
{
    std::vector<std::string> out;
    for(const auto& point : corpus.problems())
    {
        out.push_back(detail::describe(point));
    }
    return out;
}

} // namespace

TEST(TestProblemSpace, TheAnchoredShareCollapsesAtALargeCountUnlessAnchorScalingHoldsIt)
{
    // The declared mixture is 80% anchored. The fixed quotas give each combination a few
    // anchored draws however large the corpus, so at a large count the corpus is exploration;
    // anchor scaling sizes the quotas to the target.
    ExplorationRequest request;
    request.pointsPerCombination = 10;
    request.corpusTarget = 300;
    request.seed = 6;

    const auto fixed = exploreProblemSpace(anchoredToy(), request, ACCEPT_EVERYTHING);
    request.anchorScaling = true;
    const auto scaled = exploreProblemSpace(anchoredToy(), request, ACCEPT_EVERYTHING);

    ASSERT_GE(fixed.problems().size(), 300U);
    ASSERT_GE(scaled.problems().size(), 300U);
    EXPECT_LT(fixed.anchoredShare(), 0.2);
    // Growth past the target is made of exploration, so the realised share sits below the
    // declared 0.8; the point is that it is not the few percent the fixed quotas leave.
    EXPECT_GT(scaled.anchoredShare(), 0.4);
}

TEST(TestProblemSpace, AnchorScalingNeverShrinksTheFixedQuotasAndNeedsATarget)
{
    // max(pointsPerCombination, target / combinations): a target the first pass already covers,
    // or none, leaves the corpus exactly as it was.
    ExplorationRequest request;
    request.pointsPerCombination = 10;
    request.seed = 7;

    for(const int64_t target : {int64_t{0}, int64_t{12}})
    {
        request.corpusTarget = target;
        request.anchorScaling = false;
        const auto fixed = exploreProblemSpace(anchoredToy(), request, ACCEPT_EVERYTHING);
        request.anchorScaling = true;
        const auto scaled = exploreProblemSpace(anchoredToy(), request, ACCEPT_EVERYTHING);
        EXPECT_EQ(described(scaled), described(fixed)) << "target " << target;
    }
}

namespace
{

/// Each combination's result of an anchor-scaled run toward 120 problems over a declaration
/// whose `layout` c is empty, with the values of `layout` weighted or not.
std::vector<CombinationResult> anchorScaledOverAnEmptyLayout(bool weighted)
{
    const auto metadata = metadataFor(std::string(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": {
        "M":      { "type": "int64" },
        "N":      { "type": "int64" },
        "layout": { "type": "enum", "values": ["a", "b", "c"])")
                                      + (weighted ? R"(,
                    "weights": {"a": 1, "b": 1, "c": 1})"
                                                  : "")
                                      + R"( }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "createValidMatmulGraph", "source": "x.hpp",
                         "arguments": [] },
      "constraints": [ { "!=": ["$q.layout", "c"] } ],
      "archetypes": [
        { "name": "layers",
          "values": { "M": [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536],
                      "N": [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536] } }
      ],
      "neighbourhood": {
        "M": { "kind": "scale", "factors": [0.5, 0.75, 1, 1.5, 2] },
        "N": { "kind": "scale", "factors": [0.5, 1, 2] }
      },
      "mixture": { "archetypes": 0.25, "neighbourhood": 0.5, "exploration": 0.25 }
    })");
    ExplorationRequest request;
    request.pointsPerCombination = 2;
    request.corpusTarget = 120;
    request.anchorScaling = true;
    request.seed = 8;

    return exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING).combinations;
}

void expectTheTargetSharedOverTheLiveCombinations(const std::vector<CombinationResult>& result)
{
    int64_t fromArchetypes = 0;
    for(const auto& combination : result)
    {
        if(!combination.problems.empty())
        {
            // The quarter of 120 / 2.
            EXPECT_EQ(combination.fromArchetypes, 15) << detail::describe(combination.categorical);
        }
        fromArchetypes += combination.fromArchetypes;
    }
    EXPECT_EQ(fromArchetypes, 30);
}

} // namespace

TEST(TestProblemSpace, AnchorScalingSharesTheTargetOverTheCombinationsThatCanHoldAProblem)
{
    // Layout c is empty by the declaration, so the corpus is split over layouts a and b: each
    // gets half of the target, not a third, and the quotas sum to the target.
    expectTheTargetSharedOverTheLiveCombinations(anchorScaledOverAnEmptyLayout(true));
}

TEST(TestProblemSpace, AnchorScalingSharesTheTargetOverTheLiveCombinationsOfAnUnweightedDeclaration)
{
    // The same split when no value is weighted: a combination the declaration empties is not
    // counted among those the target is divided over.
    expectTheTargetSharedOverTheLiveCombinations(anchorScaledOverAnEmptyLayout(false));
}

TEST(TestProblemSpace, EveryShippedSdpaCombinationIsExploredAndNoDeclaredValueIsDropped)
{
    // Past maxCombinations the product is sampled, which the manifest reports as skipped
    // combinations. Neither direction fits under it, with the optional forms crossed with every
    // facet, so what each must keep is every declared value of every categorical parameter, and
    // every value pair of the weighted facets and the form: a model trained on the corpus has
    // seen each one (RFC 0019 section 5.2), and `--form-floor` finds a form among them. The
    // sample is drawn from the combinations that can hold a problem.
    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto metadata = test::shippedOperation(operation);

        size_t total = 0;
        const auto all = detail::categoricalCombinations(metadata.parameters, 0, total);
        EXPECT_EQ(all.size(), total) << operation;
        // is_causal 2 x alignment 2 x dtype (the forward's 7, the backward's 3) x value_head_dim
        // 2 x layout 5 x scale 3 x window 4, then the forward's 8 forms and generate_stats 2,
        // the backward's 7 forms.
        EXPECT_EQ(total, operation == "sdpa_fwd" ? size_t{53760} : size_t{10080}) << operation;

        size_t sampledTotal = 0;
        const auto sampled = detail::categoricalCombinations(
            metadata.parameters,
            ExplorationRequest{}.maxCombinations,
            sampledTotal,
            [&metadata](const ProblemPoint& combination) {
                return !detail::excludedByCategoricals(metadata, combination);
            });
        EXPECT_EQ(sampledTotal, total) << operation;
        EXPECT_EQ(sampled.size(), ExplorationRequest{}.maxCombinations) << operation;
        for(const auto& combination : sampled)
        {
            EXPECT_FALSE(detail::excludedByCategoricals(metadata, combination))
                << detail::describe(combination);
        }
        // No triple of the weighted facets is left out.
        std::set<std::string> triples;
        for(const auto& combination : sampled)
        {
            triples.insert(std::get<std::string>(combination.at("layout")) + "/"
                           + std::get<std::string>(combination.at("scale")) + "/"
                           + std::get<std::string>(combination.at("window")));
        }
        EXPECT_EQ(triples.size(), 60U) << operation;
        // Every form holds at least half its share of the combinations that can hold a problem.
        // A stride through the product ties some digits together: sdpa_fwd's 23040 at a stride
        // of 14239 held 1 scale_unset combination where 56 were due, scale_unset's one scale
        // being a value of the digit the stride had tied to its own.
        std::map<std::string, double> liveForms;
        double liveTotal = 0.0;
        for(const auto& combination : all)
        {
            if(!detail::excludedByCategoricals(metadata, combination))
            {
                liveForms[std::get<std::string>(combination.at("form"))] += 1.0;
                liveTotal += 1.0;
            }
        }
        std::map<std::string, double> sampledForms;
        for(const auto& combination : sampled)
        {
            sampledForms[std::get<std::string>(combination.at("form"))] += 1.0;
        }
        for(const auto& [form, held] : liveForms)
        {
            const auto due = held / liveTotal * static_cast<double>(sampled.size());
            EXPECT_GE(sampledForms[form], due / 2.0)
                << operation << " " << form << ": " << sampledForms[form] << " of " << due
                << " due";
        }
        for(const auto& parameter : metadata.parameters)
        {
            for(const auto& value : parameter.enumerable())
            {
                EXPECT_TRUE(std::any_of(
                    sampled.begin(),
                    sampled.end(),
                    [&](const ProblemPoint& point) { return point.at(parameter.name) == value; }))
                    << operation << " dropped " << parameter.name;
            }
        }
    }
}

TEST(TestProblemSpace, AKeepOnACategoricalAxisNarrowsTheCombinationsBeforeTheBoundApplies)
{
    const auto metadata = test::shippedOperation("sdpa_fwd");
    const auto narrow = test::narrowSdpaFwdKeeps();

    size_t total = 0;
    auto combinations = detail::categoricalCombinations(
        metadata.parameters, ExplorationRequest{}.maxCombinations, total, {}, narrow);
    EXPECT_EQ(total, 8U);
    ASSERT_EQ(combinations.size(), 8U) << "the bound is 1024, so every admitted one is explored";
    for(const auto& combination : combinations)
    {
        EXPECT_TRUE(keeps(narrow, combination)) << detail::describe(combination);
    }

    // The same parameter repeated widens its axis, as it widens the filter on points.
    auto widened = narrow;
    widened.push_back({"form", "padded"});
    combinations = detail::categoricalCombinations(
        metadata.parameters, ExplorationRequest{}.maxCombinations, total, {}, widened);
    EXPECT_EQ(total, 16U);
    EXPECT_EQ(combinations.size(), 16U);

    // A clause on a numeric parameter leaves the categorical product alone; the oracle applies it.
    combinations
        = detail::categoricalCombinations(metadata.parameters, 0, total, {}, {{"head_dim", "64"}});
    EXPECT_EQ(total, 53760U);

    // A value the declaration does not hold admits nothing, and says so with a total of 0 rather
    // than sampling combinations that every point would then fail.
    combinations = detail::categoricalCombinations(
        metadata.parameters, ExplorationRequest{}.maxCombinations, total, {}, {{"form", "nope"}});
    EXPECT_EQ(total, 0U);
    EXPECT_TRUE(combinations.empty());
}

TEST(TestProblemSpace, EveryCombinationAShippedSdpaKeepAdmitsProducesProblemsPastTheBound)
{
    // The product is 53760, past the bound of 1024, and a sample of it drawn before the keep held
    // 16 problems from one combination of the 6 that can hold one.
    const auto metadata = test::shippedOperation("sdpa_fwd");
    ExplorationRequest request;
    request.keep = test::narrowSdpaFwdKeeps();
    request.pointsPerCombination = 8;
    request.budgetPerCombination = 2000;

    const auto corpus
        = exploreProblemSpace(metadata, request, test::admitsOnlyWhatKeepKeeps(request.keep));

    EXPECT_EQ(corpus.categoricalTotal, 8U);
    EXPECT_EQ(corpus.categoricalNotExplored, 0U);
    EXPECT_EQ(corpus.categoricalConstraintEmpty, 2U) << "counted though the product fits the bound";
    size_t live = 0;
    std::set<bool> causal;
    for(const auto& combination : corpus.combinations)
    {
        if(detail::excludedByCategoricals(metadata, combination.categorical))
        {
            EXPECT_TRUE(combination.problems.empty()) << detail::describe(combination.categorical);
            continue;
        }
        ++live;
        EXPECT_FALSE(combination.problems.empty()) << detail::describe(combination.categorical);
        causal.insert(std::get<bool>(combination.categorical.at("is_causal")));
    }
    EXPECT_EQ(live, 6U);
    EXPECT_EQ(causal, (std::set<bool>{false, true})) << "both masks are explored";

    // Narrowing by one more clause keeps exploring what is left rather than going empty.
    request.keep.push_back({"is_causal", "true"});
    const auto causalOnly
        = exploreProblemSpace(metadata, request, test::admitsOnlyWhatKeepKeeps(request.keep));
    EXPECT_EQ(causalOnly.categoricalTotal, 4U);
    EXPECT_GE(causalOnly.problems().size(), 4U * 4U);
    size_t withProblems = 0;
    for(const auto& combination : causalOnly.combinations)
    {
        withProblems += combination.problems.empty() ? 0 : 1;
    }
    EXPECT_EQ(withProblems, 4U);
    for(const auto& point : causalOnly.problems())
    {
        EXPECT_TRUE(std::get<bool>(point.at("is_causal")));
    }
}

TEST(TestProblemSpace, AKeepOnAnFp8DtypeReachesItsProblemsLikeAKeepOnALayout)
{
    // dtype is a categorical axis, so `--keep q.dtype=<v>` narrows the combinations before the
    // bound applies for each of the four fp8 encodings as for fp16, and every problem it finds
    // is of that encoding. Repeating the parameter widens the axis.
    const auto metadata = test::shippedOperation("sdpa_fwd");
    const auto* dtype = metadata.find("dtype");
    ASSERT_NE(dtype, nullptr);
    const std::string fp8 = "fp8_e4m3_fnuz";
    ASSERT_NE(std::find(dtype->values.begin(), dtype->values.end(), fp8), dtype->values.end());

    size_t everything = 0;
    detail::categoricalCombinations(metadata.parameters, 0, everything);
    size_t total = 0;
    detail::categoricalCombinations(metadata.parameters, 0, total, {}, {{"dtype", fp8}});
    EXPECT_EQ(total, everything / dtype->values.size()) << "one value of the declared dtypes";
    detail::categoricalCombinations(
        metadata.parameters, 0, total, {}, {{"dtype", fp8}, {"dtype", "fp8_e5m2"}});
    EXPECT_EQ(total, 2U * everything / dtype->values.size());

    // The narrow keeps of the other facets with the dtype swapped: 8 combinations, 6 of them
    // able to hold a problem, all explored.
    auto keep = test::narrowSdpaFwdKeeps();
    for(auto& clause : keep)
    {
        if(clause.parameter == "dtype")
        {
            clause.value = fp8;
        }
    }
    ExplorationRequest request;
    request.keep = keep;
    request.pointsPerCombination = 8;
    request.budgetPerCombination = 2000;
    const auto corpus
        = exploreProblemSpace(metadata, request, test::admitsOnlyWhatKeepKeeps(request.keep));

    EXPECT_EQ(corpus.categoricalTotal, 8U);
    EXPECT_EQ(corpus.categoricalNotExplored, 0U);
    ASSERT_GE(corpus.problems().size(), 6U * 4U);
    for(const auto& point : corpus.problems())
    {
        EXPECT_EQ(std::get<std::string>(point.at("dtype")), fp8) << detail::describe(point);
    }
}

TEST(TestProblemSpace, AKeepEveryDeclaredConstraintEmptiesExploresNothingRatherThanFailing)
{
    // The ragged form is declared only with the token-major bshd and qkv_packed, so no
    // combination of these two keeps can hold a problem; the bound is low enough that the sample
    // walks the product and finds none.
    const auto metadata = test::shippedOperation("sdpa_fwd");
    ExplorationRequest request;
    request.keep = {{"form", "ragged"}, {"layout", "bhsd"}};
    request.maxCombinations = 4;

    const auto corpus
        = exploreProblemSpace(metadata, request, test::admitsOnlyWhatKeepKeeps(request.keep));

    EXPECT_TRUE(corpus.problems().empty());
    EXPECT_EQ(corpus.categoricalTotal, 1344U);
    EXPECT_EQ(corpus.categoricalConstraintEmpty, 1344U);
    EXPECT_EQ(corpus.categoricalNotExplored, 0U) << "none was left out: none could hold a problem";
    ASSERT_FALSE(corpus.skippedCombinations.empty());
    EXPECT_NE(corpus.skippedCombinations.front().find("empty under the declared constraints"),
              std::string::npos)
        << corpus.skippedCombinations.front();
}

TEST(TestProblemSpace, TheCombinationsTheBoundLeavesOutAreCountedAgainstTheKeptProduct)
{
    // The product is 8 and 2 of them are empty by the declaration, so the 4 the bound allows are
    // taken from the 6 that can hold a problem and 2 of those are left out.
    const auto metadata = test::shippedOperation("sdpa_fwd");
    ExplorationRequest request;
    request.keep = test::narrowSdpaFwdKeeps();
    request.maxCombinations = 4;
    request.pointsPerCombination = 2;
    request.budgetPerCombination = 500;

    const auto corpus
        = exploreProblemSpace(metadata, request, test::admitsOnlyWhatKeepKeeps(request.keep));

    EXPECT_EQ(corpus.categoricalTotal, 8U) << "the keep's product, not the declared 53760";
    EXPECT_EQ(corpus.categoricalConstraintEmpty, 2U);
    EXPECT_EQ(corpus.categoricalNotExplored, 2U)
        << "the combinations the constraints empty were never candidates";
    ASSERT_FALSE(corpus.skippedCombinations.empty());
    EXPECT_NE(corpus.skippedCombinations.front().find("2 of 6"), std::string::npos)
        << corpus.skippedCombinations.front();
}

TEST(TestProblemSpace, TheConstraintEmptyCombinationsAreCountedOverTheWholeProductPastTheBound)
{
    // The sample stops once it holds 1024 combinations that can hold a problem; the count of
    // those that cannot covers the product the walk did not reach too.
    const auto metadata = test::shippedOperation("sdpa_fwd");
    const auto live = [&metadata](const ProblemPoint& combination) {
        return !detail::excludedByCategoricals(metadata, combination);
    };
    size_t total = 0;
    const auto everything = detail::categoricalCombinations(metadata.parameters, 0, total);
    const auto empty = static_cast<size_t>(
        std::count_if(everything.begin(), everything.end(), [&](const ProblemPoint& combination) {
            return !live(combination);
        }));
    ASSERT_GT(empty, 0U);

    size_t sampledTotal = 0;
    size_t counted = 0;
    const auto sampled = detail::categoricalCombinations(metadata.parameters,
                                                         ExplorationRequest{}.maxCombinations,
                                                         sampledTotal,
                                                         live,
                                                         {},
                                                         &counted);
    EXPECT_EQ(sampled.size(), ExplorationRequest{}.maxCombinations);
    EXPECT_EQ(counted, empty);

    size_t underTheBound = 0;
    detail::categoricalCombinations(
        metadata.parameters, 0, sampledTotal, live, test::narrowSdpaFwdKeeps(), &underTheBound);
    EXPECT_EQ(underTheBound, 2U);

    ExplorationRequest request;
    request.pointsPerCombination = 1;
    request.budgetPerCombination = 50;
    request.probeBudget = 25;
    const auto corpus
        = exploreProblemSpace(metadata, request, [](const ProblemPoint&) { return false; });
    EXPECT_EQ(corpus.categoricalTotal, total);
    EXPECT_EQ(corpus.categoricalConstraintEmpty, empty);
    EXPECT_EQ(corpus.categoricalNotExplored, total - empty - corpus.combinations.size())
        << "the combinations that can hold a problem and the bound left out";
}

TEST(TestProblemSpace, ACombinationTheConstraintsEmptyHasNoSizeWhetherOrNotAnythingIsWeighted)
{
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": {
        "M":      { "type": "int64" },
        "layout": { "type": "enum", "values": ["a", "b", "c"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "createValidMatmulGraph", "source": "x.hpp",
                         "arguments": [] },
      "constraints": [ { "!=": ["$q.layout", "c"] } ]
    })");
    size_t total = 0;
    const auto combinations = detail::categoricalCombinations(metadata.parameters, 0, total);
    ASSERT_EQ(combinations.size(), 3U);

    EXPECT_EQ(detail::combinationSizes(metadata, combinations),
              (std::vector<double>{1.0, 1.0, 0.0}));
}

/// The shipped attention declarations, whose constraints the forward and the backward share.
class TestShippedSdpaConstraints : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TestShippedSdpaConstraints, KeepOnlyProblemsThatAreDistinctAndHaveKeys)
{
    const auto metadata = test::shippedOperation(GetParam());

    struct Candidate
    {
        int64_t seqlenQ = 2048;
        int64_t seqlenK = 2048;
        int64_t heads = 8;
        int64_t headsKv = 8;
        bool causal = true;
        std::string window = "none";
        std::string layout = "bshd";
        std::string alignment = "top_left";
        int64_t batch = 1;
        int64_t headDim = 64;
        std::string valueHeadDim = "same";
    };
    const auto admits = [&](const Candidate& c) {
        ProblemPoint point{{"batch", c.batch},
                           {"heads", c.heads},
                           {"heads_kv", c.headsKv},
                           {"seqlen_q", c.seqlenQ},
                           {"seqlen_k", c.seqlenK},
                           {"head_dim", c.headDim},
                           {"value_head_dim", c.valueHeadDim},
                           {"is_causal", c.causal},
                           {"alignment", c.alignment},
                           {"dtype", std::string("fp16")},
                           {"layout", c.layout},
                           {"window", c.window}};
        return detail::satisfiesConstraints(metadata,
                                            test::withDefaultCategoricals(metadata, point));
    };

    // Cross attention, causal or not: anchored top-left every query row keeps a key (the rows
    // past seqlen_k keep all of them). Anchored bottom-right the first seqlen_q - seqlen_k rows
    // keep none, so the anchor stays pinned to top_left there.
    EXPECT_TRUE(admits({4096, 77, 8, 8, false}));
    EXPECT_TRUE(admits({4096, 77, 8, 8, true}));
    EXPECT_FALSE(admits({4096, 77, 8, 8, true, "none", "bshd", "bottom_right"}));
    // The anchor is a distinct problem where the mask is causal or windowed and
    // seqlen_q < seqlen_k, and pinned to top_left everywhere else.
    EXPECT_TRUE(admits({128, 4096, 8, 8, true, "none", "bshd", "bottom_right"}));
    EXPECT_FALSE(admits({128, 4096, 8, 8, false, "none", "bshd", "bottom_right"}));
    // Extents of 1 make kv_bhsd the same graph as bhsd (seqlen_q 1, one head) or as bshd
    // (seqlen_k 1, one kv head), so it is not listed there.
    EXPECT_FALSE(admits({1, 4096, 1, 1, true, "none", "kv_bhsd"}));
    EXPECT_TRUE(admits({1, 4096, 8, 1, true, "none", "kv_bhsd"}));
    EXPECT_TRUE(admits({2, 4096, 1, 1, true, "none", "kv_bhsd"}));
    EXPECT_FALSE(admits({1, 1, 8, 1, false, "none", "kv_bhsd"}));
    EXPECT_TRUE(admits({1, 1, 8, 1, false, "none", "bhsd"}));
    // sbhd is bshd with one batch, or with one row in both sequences.
    EXPECT_FALSE(admits({2048, 2048, 8, 8, true, "none", "sbhd"}));
    EXPECT_TRUE(admits({2048, 2048, 8, 8, true, "none", "sbhd", "top_left", 2}));
    EXPECT_TRUE(admits({1, 4096, 8, 8, true, "none", "sbhd", "top_left", 2}));
    EXPECT_FALSE(admits({1, 1, 8, 8, false, "none", "sbhd", "top_left", 2}));
    // A fused QKV projection is self-attention.
    EXPECT_TRUE(admits({2048, 2048, 8, 8, true, "none", "qkv_packed"}));
    EXPECT_FALSE(admits({128, 4096, 8, 8, true, "none", "qkv_packed"}));
    // An MLA value head is 64 narrower than Q and K, and kept at 64 or more.
    EXPECT_TRUE(admits({2048, 2048, 8, 8, true, "none", "bshd", "top_left", 1, 192, "mla"}));
    EXPECT_TRUE(admits({2048, 2048, 8, 8, true, "none", "bshd", "top_left", 1, 128, "mla"}));
    EXPECT_FALSE(admits({2048, 2048, 8, 8, true, "none", "bshd", "top_left", 1, 120, "mla"}));

    EXPECT_TRUE(admits({2048, 2048, 8, 8, true, "local_1024"}));
    // The frontend accepts a left bound under any mask.
    EXPECT_TRUE(admits({2048, 2048, 8, 8, false, "local_1024"}));
    // A window over more queries than keys: top-left, every row from seqlen_k + N - 1 on
    // keeps no key (90% of the rows at 2048 x 77, local_128), so the softmax is over nothing.
    EXPECT_FALSE(admits({2048, 77, 8, 8, false, "local_128"}));
    EXPECT_FALSE(admits({4096, 77, 8, 8, false, "local_4096"}));
    EXPECT_TRUE(admits({128, 4096, 8, 8, false, "local_128", "bshd", "bottom_right"}));
}

INSTANTIATE_TEST_SUITE_P(SdpaDeclarations,
                         TestShippedSdpaConstraints,
                         ::testing::Values("sdpa_fwd", "sdpa_bwd"));

TEST(TestProblemSpace, TheShippedSdpaFacetInteractionsRefuseWhatIsDegenerateAndAdmitTheRest)
{
    struct Candidate
    {
        std::string form = "plain";
        std::string layout = "bshd";
        std::string scale = "inv_sqrt_d";
        std::string window = "none";
        int64_t seqlenQ = 2048;
        int64_t seqlenK = 2048;
        bool causal = false;
    };
    const auto admitsFor = [](const std::string& operation, const Candidate& c) {
        const auto metadata = test::shippedOperation(operation);
        ProblemPoint point{{"batch", int64_t{2}},
                           {"heads", int64_t{8}},
                           {"heads_kv", int64_t{2}},
                           {"seqlen_q", c.seqlenQ},
                           {"seqlen_k", c.seqlenK},
                           {"head_dim", int64_t{64}},
                           {"is_causal", c.causal},
                           {"dtype", std::string("fp16")},
                           {"form", c.form},
                           {"layout", c.layout},
                           {"scale", c.scale},
                           {"window", c.window}};
        return detail::satisfiesConstraints(metadata,
                                            test::withDefaultCategoricals(metadata, point));
    };
    // The forms both directions build; the forward also builds sink.
    const auto formsOf = [](const std::string& operation) {
        std::vector<std::string> forms{"plain", "dropout", "padded", "ragged", "bias"};
        if(operation == "sdpa_fwd")
        {
            forms.emplace_back("sink");
        }
        return forms;
    };

    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        // A form that states no scale value (scale_unset), or passes it as a tensor
        // (scale_tensor), ignores the scale facet: three values would list one graph three times
        // (or three tensors that differ only in their default), so only inv_sqrt_d is declared.
        for(const std::string form : {"scale_unset", "scale_tensor"})
        {
            EXPECT_TRUE(admitsFor(operation, Candidate{form})) << operation << " " << form;
            EXPECT_FALSE(admitsFor(operation, Candidate{form, "bshd", "unit"}))
                << operation << " " << form;
            EXPECT_FALSE(admitsFor(operation, Candidate{form, "bshd", "jitter"}))
                << operation << " " << form;
        }
        // The forms that state the scale keep the whole facet, over every layout.
        for(const std::string form : {"plain", "dropout", "padded", "bias"})
        {
            for(const std::string scale : {"inv_sqrt_d", "unit", "jitter"})
            {
                for(const std::string layout : {"bshd", "bhsd", "kv_bhsd", "sbhd", "qkv_packed"})
                {
                    EXPECT_TRUE(admitsFor(operation, Candidate{form, layout, scale}))
                        << operation << " " << form << " " << layout << " " << scale;
                }
            }
        }

        // Ragged packs tokens through offset tensors whose multiplier is the token stride, so it
        // needs token-major strides: head-major or sequence-major Q, K or V would be addressed
        // as if one sequence's tokens were contiguous across heads.
        EXPECT_TRUE(admitsFor(operation, Candidate{"ragged", "bshd"})) << operation;
        EXPECT_TRUE(admitsFor(operation, Candidate{"ragged", "qkv_packed"})) << operation;
        EXPECT_FALSE(admitsFor(operation, Candidate{"ragged", "bhsd"})) << operation;
        EXPECT_FALSE(admitsFor(operation, Candidate{"ragged", "kv_bhsd"})) << operation;
        EXPECT_FALSE(admitsFor(operation, Candidate{"ragged", "sbhd"})) << operation;
        EXPECT_TRUE(admitsFor(operation, Candidate{"ragged", "bshd", "unit"})) << operation;
        EXPECT_TRUE(admitsFor(operation, Candidate{"ragged", "bshd", "jitter", "local_128"}))
            << operation;

        // Everything else crosses with a window and with the layouts, the frontend validators
        // and references stating nothing against it and an engine's matcher declining what it
        // does not serve (which the oracle filters): every form with every window.
        for(const auto& form : formsOf(operation))
        {
            for(const std::string window : {"local_128", "local_1024", "local_4096"})
            {
                for(const bool causal : {true, false})
                {
                    EXPECT_TRUE(admitsFor(operation, Candidate{form, "bshd", "inv_sqrt_d", window}))
                        << operation << " " << form << " " << window;
                    if(form != "ragged")
                    {
                        EXPECT_TRUE(admitsFor(
                            operation, Candidate{form, "bhsd", "unit", window, 2048, 2048, causal}))
                            << operation << " " << form << " " << window << " causal " << causal;
                    }
                }
            }
        }

        // Cross attention (more queries than keys), causal or not, is never windowed, whatever
        // the form: the forms that carry per-batch lengths (padded, ragged) keep the same bounds.
        for(const auto& form : formsOf(operation))
        {
            for(const bool causal : {true, false})
            {
                EXPECT_TRUE(admitsFor(
                    operation, Candidate{form, "bshd", "inv_sqrt_d", "none", 4096, 77, causal}))
                    << operation << " " << form << " causal " << causal;
                EXPECT_FALSE(
                    admitsFor(operation,
                              Candidate{form, "bshd", "inv_sqrt_d", "local_128", 4096, 77, causal}))
                    << operation << " " << form << " causal " << causal;
            }
        }
    }
    EXPECT_TRUE(admitsFor("sdpa_fwd",
                          Candidate{"sink", "bshd", "inv_sqrt_d", "local_128", 2048, 2048, true}));
}

namespace
{

/// Two enums, one weighted 2:1:1, one not, over two numeric extents.
OperationMetadata weightedToy(const std::string& weights)
{
    return metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": {
        "M":      { "type": "int64" },
        "N":      { "type": "int64" },
        "layout": { "type": "enum", "values": ["a", "b", "c"])"
                       + weights + R"( },
        "dtype":  { "type": "enum", "values": ["fp32", "fp16"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "createValidMatmulGraph", "source": "x.hpp",
                         "arguments": [] }
    })");
}

double shareOf(const ProblemCorpus& corpus, const std::string& parameter, const std::string& value)
{
    size_t held = 0;
    const auto problems = corpus.problems();
    for(const auto& point : problems)
    {
        held += std::get<std::string>(point.at(parameter)) == value ? 1 : 0;
    }
    return static_cast<double>(held) / static_cast<double>(problems.size());
}

} // namespace

TEST(TestProblemSpace, AWeightedValueRealisesItsShareOfTheFirstPassAndOfTheGrownCorpus)
{
    const auto metadata = weightedToy(R"(, "weights": {"a": 2})");
    for(const int64_t target : {int64_t{0}, int64_t{600}})
    {
        ExplorationRequest request;
        request.pointsPerCombination = 40;
        request.corpusTarget = target;
        request.seed = 12;

        const auto corpus = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING);

        // 2:1:1 is 0.5, 0.25, 0.25; the other enum stays even.
        EXPECT_NEAR(shareOf(corpus, "layout", "a"), 0.5, 0.03) << "target " << target;
        EXPECT_NEAR(shareOf(corpus, "layout", "b"), 0.25, 0.03) << "target " << target;
        EXPECT_NEAR(shareOf(corpus, "layout", "c"), 0.25, 0.03) << "target " << target;
        EXPECT_NEAR(shareOf(corpus, "dtype", "fp32"), 0.5, 0.03) << "target " << target;
        if(target > 0)
        {
            EXPECT_GE(corpus.problems().size(), static_cast<size_t>(target));
        }
    }
}

TEST(TestProblemSpace, EqualWeightsAreNoWeightsInTheSharesAndTheFirstPassIsTheUnweightedMean)
{
    // Weights are relative: declaring every value equal must not move any value's share. It does
    // move the size of the first pass: the weighted axis splits pointsPerCombination for each
    // dtype combination (2 x 25) among its three values, where the unweighted search gives each
    // of the six combinations 25.
    for(const int64_t target : {int64_t{0}, int64_t{250}})
    {
        ExplorationRequest request;
        request.pointsPerCombination = 25;
        request.corpusTarget = target;
        request.seed = 13;

        const auto absent = exploreProblemSpace(weightedToy(""), request, ACCEPT_EVERYTHING);
        const auto equal = exploreProblemSpace(
            weightedToy(R"(, "weights": {"a": 3, "b": 3, "c": 3})"), request, ACCEPT_EVERYTHING);

        EXPECT_NEAR(shareOf(equal, "layout", "a"), shareOf(absent, "layout", "a"), 0.03);
        EXPECT_NEAR(shareOf(equal, "layout", "b"), shareOf(absent, "layout", "b"), 0.03);
        EXPECT_NEAR(shareOf(equal, "dtype", "fp32"), shareOf(absent, "dtype", "fp32"), 0.03);
        if(target == 0)
        {
            EXPECT_EQ(absent.problems().size(), size_t{150});
            EXPECT_EQ(equal.problems().size(), size_t{50});
        }
        else
        {
            // Growing splits the need exactly (as apportion splits it) where the unweighted
            // search rounds each share up, so the points differ.
            EXPECT_GE(equal.problems().size(), static_cast<size_t>(target));
        }
    }
}

TEST(TestProblemSpace, ApportionSplitsBySystematicSamplingAndSumsExactly)
{
    // 5 exactly for the heavy part; the two equal ones split the other 5 as 3 and 2, and which
    // gets the odd one is the tie-break's, not a fixed position.
    const auto split = detail::apportion(10, {2.0, 1.0, 1.0});
    EXPECT_EQ(split[0], 5);
    EXPECT_EQ(split[1] + split[2], 5);
    // A ceiling per part would give four here (1 each) and a rounding per part two or none.
    for(const int64_t total : {int64_t{1}, int64_t{3}, int64_t{7}, int64_t{50}, int64_t{1001}})
    {
        const auto parts = detail::apportion(total, {0.25, 0.75, 1.5, 0.25, 1.25});
        int64_t sum = 0;
        for(const auto part : parts)
        {
            sum += part;
        }
        EXPECT_EQ(sum, total);
    }
    EXPECT_EQ(detail::apportion(0, {1.0, 1.0}), (std::vector<int64_t>{0, 0}));

    // Every ideal below one: 300 over 864 parts of two sizes. A largest-remainder split gives
    // nearly all 300 units to the heavy parts; the light ones must keep their share of them.
    std::vector<double> sizes(864, 1.0);
    std::vector<uint64_t> keys(864);
    for(size_t i = 0; i < sizes.size(); ++i)
    {
        sizes[i] = i % 3 == 0 ? 2.0 : 0.5;
        keys[i] = detail::tieKey(3, i);
    }
    const auto spread = detail::apportion(300, sizes, keys);
    int64_t heavy = 0;
    int64_t all = 0;
    for(size_t i = 0; i < spread.size(); ++i)
    {
        all += spread[i];
        heavy += i % 3 == 0 ? spread[i] : 0;
    }
    EXPECT_EQ(all, 300);
    // The 288 heavy parts weigh 576 of 576 + 288 = 864: two thirds.
    EXPECT_NEAR(static_cast<double>(heavy) / 300.0, 2.0 / 3.0, 0.07);
}

TEST(TestProblemSpace, AWeightedValueKeepsItsShareAtASmallPointsPerCombination)
{
    // 0.75 of two points rounds to two, and 1.5 of two to three: a per-combination rounding
    // (and a per-combination ceiling when growing) puts every small value above its weight.
    // The split of the whole first pass (2 points for each of the two dtypes) by apportion
    // lands exactly on 2:1:1, and no combination is owed a point to spoil it.
    ExplorationRequest request;
    request.pointsPerCombination = 2;
    request.seed = 14;
    const auto corpus
        = exploreProblemSpace(weightedToy(R"(, "weights": {"a": 2})"), request, ACCEPT_EVERYTHING);
    // Exact for the heavy value; the two equal light ones split what is left, so one of them
    // may hold a point more than the other.
    EXPECT_NEAR(shareOf(corpus, "layout", "a"), 0.5, 0.01);
    EXPECT_NEAR(shareOf(corpus, "layout", "b") + shareOf(corpus, "layout", "c"), 0.5, 0.01);

    request.corpusTarget = 400;
    const auto grown
        = exploreProblemSpace(weightedToy(R"(, "weights": {"a": 2})"), request, ACCEPT_EVERYTHING);
    EXPECT_NEAR(shareOf(grown, "layout", "a"), 0.5, 0.03);
    EXPECT_NEAR(shareOf(grown, "layout", "b") + shareOf(grown, "layout", "c"), 0.5, 0.03);
}

TEST(TestProblemSpace, SizesAreRakedOverTheCombinationsThatCanHoldAProblem)
{
    // (alignment bottom_right, non-causal, window none) is empty by the declaration's own
    // constraints, while its windowed twins are not, so sizes taken as a product of weights over
    // every combination leave window none under its declared 50%. The forms empty more: a scale
    // the form does not state (scale_unset, scale_tensor) other than inv_sqrt_d, and a ragged
    // form over a layout that is not token-major (bshd, qkv_packed). The raking has to hold all
    // five weighted facets (dtype, value_head_dim, layout, scale, window) at once over what is
    // left. The form is not weighted.
    const auto metadata = test::shippedOperation("sdpa_fwd");

    size_t total = 0;
    const auto combinations = detail::categoricalCombinations(metadata.parameters, 0, total);
    ASSERT_EQ(combinations.size(), 53760U);
    const auto sizes = detail::combinationSizes(metadata, combinations);

    double all = 0.0;
    std::map<std::string, double> window;
    std::map<std::string, double> layout;
    std::map<std::string, double> scale;
    std::map<std::string, double> valueHeadDim;
    std::map<std::string, double> dtype;
    std::map<std::string, double> form;
    size_t empty = 0;
    for(size_t i = 0; i < combinations.size(); ++i)
    {
        const auto& c = combinations[i];
        const auto text = [&c](const char* name) { return std::get<std::string>(c.at(name)); };
        const bool scaleIgnored = text("form") == "scale_unset" || text("form") == "scale_tensor";
        const bool emptyByDeclaration
            = (text("alignment") == "bottom_right" && !std::get<bool>(c.at("is_causal"))
               && text("window") == "none")
              || (scaleIgnored && text("scale") != "inv_sqrt_d")
              || (text("form") == "ragged" && text("layout") != "bshd"
                  && text("layout") != "qkv_packed");
        if(emptyByDeclaration)
        {
            EXPECT_EQ(sizes[i], 0.0) << detail::describe(c);
            ++empty;
        }
        else
        {
            EXPECT_GT(sizes[i], 0.0) << detail::describe(c);
        }
        all += sizes[i];
        window[text("window")] += sizes[i];
        layout[text("layout")] += sizes[i];
        scale[text("scale")] += sizes[i];
        valueHeadDim[text("value_head_dim")] += sizes[i];
        dtype[text("dtype")] += sizes[i];
        form[text("form")] += sizes[i];
    }
    // Per form, 6720 combinations: 420 are the bottom_right, non-causal, no-window sixteenth of
    // the product. Five forms leave it at that; scale_unset and scale_tensor also lose the 4480
    // whose scale is not inv_sqrt_d (280 of them already in the 420), and ragged the 4032 whose
    // layout is not token-major (252 already in the 420):
    // 5 * 420 + 2 * (4480 + 420 - 280) + (4032 + 420 - 252) = 15540.
    EXPECT_EQ(empty, 15540U);
    EXPECT_NEAR(window["none"] / all, 0.5, 1e-6);
    EXPECT_NEAR(window["local_128"] / all, 1.0 / 6.0, 1e-6);
    EXPECT_NEAR(layout["bshd"] / all, 2.0 / 6.0, 1e-6);
    EXPECT_NEAR(layout["kv_bhsd"] / all, 1.0 / 6.0, 1e-6);
    EXPECT_NEAR(layout["qkv_packed"] / all, 1.0 / 6.0, 1e-6);
    EXPECT_NEAR(scale["inv_sqrt_d"] / all, 0.5, 1e-6);
    EXPECT_NEAR(valueHeadDim["same"] / all, 0.75, 1e-6);
    // The weights of dtype, 8:8:8:1:1:2:1 over fp16, bf16, fp32 and the four fp8 encodings, hold
    // however the other facets are cut: the three wider types at 8/29 each, fp8 together at 5/29
    // and the one gfx942 serves, fp8_e4m3_fnuz, at twice any other encoding.
    EXPECT_NEAR(dtype["fp16"] / all, 8.0 / 29.0, 1e-6);
    EXPECT_NEAR(dtype["bf16"] / all, 8.0 / 29.0, 1e-6);
    EXPECT_NEAR(dtype["fp32"] / all, 8.0 / 29.0, 1e-6);
    EXPECT_NEAR(dtype["fp8_e4m3_fnuz"] / all, 2.0 / 29.0, 1e-6);
    EXPECT_NEAR(dtype["fp8_e4m3"] / all, 1.0 / 29.0, 1e-6);
    EXPECT_NEAR(dtype["fp8_e5m2"] / all, 1.0 / 29.0, 1e-6);
    EXPECT_NEAR(dtype["fp8_e5m2_fnuz"] / all, 1.0 / 29.0, 1e-6);
    // The form carries no weight: a form's share is its share of the combinations that can hold
    // a problem, less for the three that the declaration's constraints leave fewer of than the
    // other five (those hold 6300 of their 6720 combinations each; scale_unset and scale_tensor
    // 2100, ragged 2520), and no form is left out.
    EXPECT_TRUE(metadata.find("form")->weights.empty());
    for(const auto& [name, held] : form)
    {
        EXPECT_GT(held / all, 0.04) << name;
    }
    EXPECT_GT(form["sink"], form["scale_unset"]);
    EXPECT_GT(form["sink"], form["ragged"]);
    // The mean over combinations that can hold a problem is 1, as it is for unweighted ones.
    EXPECT_NEAR(all, static_cast<double>(combinations.size() - empty), 1e-6);
}

namespace
{

/// Three equally weighted layouts over two extents and nothing else, so the first pass is
/// pointsPerCombination points split over three combinations.
OperationMetadata threeEqualLayouts()
{
    return metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": {
        "M":      { "type": "int64" },
        "N":      { "type": "int64" },
        "layout": { "type": "enum", "values": ["a", "b", "c"],
                    "weights": {"a": 1, "b": 1, "c": 1} }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "createValidMatmulGraph", "source": "x.hpp",
                         "arguments": [] }
    })");
}

/// Layout c admits every extent up to @p wideLimit; a and b admit only M = N = 1, so each is
/// exhausted by its first point.
ProblemOracle oneWideLayout(int64_t wideLimit)
{
    return [wideLimit](const ProblemPoint& point) {
        const auto m = std::get<int64_t>(point.at("M"));
        const auto n = std::get<int64_t>(point.at("N"));
        return std::get<std::string>(point.at("layout")) == "c" ? m <= wideLimit && n <= wideLimit
                                                                : m <= 1 && n <= 1;
    };
}

} // namespace

TEST(TestProblemSpace, TheFirstPassDrawsEveryDeclaredValueAndKeepsItsBudgetWhereItCan)
{
    ExplorationRequest request;
    request.seed = 15;
    size_t total = 0;
    const auto covered = [](const OperationMetadata& metadata,
                            const std::vector<ProblemPoint>& combinations,
                            const std::vector<int64_t>& targets,
                            const std::string& parameter) {
        std::set<std::string> values;
        for(size_t i = 0; i < combinations.size(); ++i)
        {
            if(targets[i] > 0)
            {
                values.insert(std::get<std::string>(combinations[i].at(parameter)));
            }
        }
        EXPECT_NE(metadata.find(parameter), nullptr);
        return values;
    };
    const auto sum = [](const std::vector<int64_t>& targets) {
        return std::accumulate(targets.begin(), targets.end(), int64_t{0});
    };

    // Layout a is a thousand times as common as b and c: their combinations' shares round to
    // nothing, yet each is drawn, with the points taken from layout a, not added to the budget
    // (10 points for each of the two dtypes).
    request.pointsPerCombination = 10;
    const auto skewed = weightedToy(R"(, "weights": {"a": 1000, "b": 1, "c": 1})");
    const auto skewedCombinations = detail::categoricalCombinations(skewed.parameters, 0, total);
    const auto skewedTargets = detail::firstPassTargets(
        skewed, request, skewedCombinations, detail::combinationSizes(skewed, skewedCombinations));
    EXPECT_EQ(sum(skewedTargets), 20);
    EXPECT_EQ(covered(skewed, skewedCombinations, skewedTargets, "layout"),
              (std::set<std::string>{"a", "b", "c"}));
    EXPECT_EQ(covered(skewed, skewedCombinations, skewedTargets, "dtype"),
              (std::set<std::string>{"fp32", "fp16"}));

    // Three values and one point: none may give theirs up, so the first pass is the three owed.
    request.pointsPerCombination = 1;
    const auto three = threeEqualLayouts();
    const auto threeCombinations = detail::categoricalCombinations(three.parameters, 0, total);
    const auto threeTargets = detail::firstPassTargets(
        three, request, threeCombinations, detail::combinationSizes(three, threeCombinations));
    EXPECT_EQ(threeTargets, std::vector<int64_t>(3, 1));

    // Nothing weighted: pointsPerCombination for each combination, as ever.
    const auto plain = weightedToy("");
    const auto plainCombinations = detail::categoricalCombinations(plain.parameters, 0, total);
    request.pointsPerCombination = 9;
    EXPECT_EQ(
        detail::firstPassTargets(
            plain, request, plainCombinations, detail::combinationSizes(plain, plainCombinations)),
        std::vector<int64_t>(6, 9));
}

TEST(TestProblemSpace, ACombinationLeftWithoutAFirstPassPointStillGrowsToTheTarget)
{
    // Six combinations share 4 first-pass points (2 for each dtype), layout a taking nearly
    // all of them; b and c are drawn once each, so two combinations are left with none. They are
    // untried, not declined: growing toward a target well past the first pass reaches them, in
    // proportion to their weight.
    const auto metadata = weightedToy(R"(, "weights": {"a": 100, "b": 1, "c": 1})");
    for(uint64_t seed = 0; seed < 5; ++seed)
    {
        ExplorationRequest request;
        request.pointsPerCombination = 2;
        request.seed = seed;
        size_t total = 0;
        const auto combinations = detail::categoricalCombinations(metadata.parameters, 0, total);
        const auto targets = detail::firstPassTargets(
            metadata, request, combinations, detail::combinationSizes(metadata, combinations));
        ASSERT_NE(std::count(targets.begin(), targets.end(), 0), 0) << "seed " << seed;

        request.corpusTarget = 2000;
        const auto corpus = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING);

        EXPECT_GE(corpus.problems().size(), 2000U) << "seed " << seed;
        EXPECT_TRUE(corpus.shortfall.empty()) << "seed " << seed;
        for(const auto& combination : corpus.combinations)
        {
            EXPECT_FALSE(combination.problems.empty())
                << "seed " << seed << ": " << detail::describe(combination.categorical);
        }
    }
}

TEST(TestProblemSpace, WeightedGrowthKeepsGoingWhenTheRoundsShareFallsOnExhaustedCombinations)
{
    // A round's need split by weight gives a unit to only some combinations: a need of one goes
    // to one of the three. When that one is exhausted it adds nothing and is spent, but the
    // others were never searched, so growth has to go on and reach layout c's points rather
    // than stop short of the target.
    for(uint64_t seed = 0; seed < 20; ++seed)
    {
        ExplorationRequest request;
        request.pointsPerCombination = 8;
        request.numericCeiling = 256;
        request.seed = seed;
        const auto firstPass = exploreProblemSpace(threeEqualLayouts(), request, oneWideLayout(256))
                                   .problems()
                                   .size();

        request.corpusTarget = static_cast<int64_t>(firstPass) + 1;
        const auto grown = exploreProblemSpace(threeEqualLayouts(), request, oneWideLayout(256));

        EXPECT_GE(grown.problems().size(), static_cast<size_t>(request.corpusTarget))
            << "seed " << seed;
        EXPECT_TRUE(grown.shortfall.empty()) << "seed " << seed;
    }
}

TEST(TestProblemSpace, WeightedGrowthStopsWhenEveryCombinationIsExhausted)
{
    // Each layout holds one point, so no target above three can be met. Growth must end with the
    // shortfall reported, having searched every combination rather than re-splitting a need
    // nobody can take.
    ExplorationRequest request;
    request.pointsPerCombination = 8;
    request.numericCeiling = 256;
    request.corpusTarget = 1000;
    request.seed = 3;

    const auto corpus = exploreProblemSpace(threeEqualLayouts(), request, oneWideLayout(1));

    EXPECT_EQ(corpus.problems().size(), 3U);
    ASSERT_EQ(corpus.shortfall.size(), 4U);
    for(size_t i = 1; i < corpus.shortfall.size(); ++i)
    {
        EXPECT_NE(corpus.shortfall[i].find("saturated"), std::string::npos) << corpus.shortfall[i];
    }
}

/// The shipped attention declarations, whose weighted facets and forms cross to thousands of
/// combinations.
class TestShippedFirstPass : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TestShippedFirstPass, TheShippedSdpaDeclarationsKeepTheDefaultFirstPassNearItsUnweightedSize)
{
    // Weighted axes split a budget; they do not multiply it. The budget is pointsPerCombination
    // for each distinct live combination of the unweighted parameters (is_causal, alignment,
    // dtype, form, and for the forward generate_stats), not for each of the thousands of
    // weighted combinations: crossing three weighted axes (layout, scale, window) in would multiply
    // it by their 36 value combinations in the forward.
    const auto metadata = test::shippedOperation(GetParam());
    const ExplorationRequest request;
    size_t total = 0;
    const auto combinations = detail::categoricalCombinations(
        metadata.parameters,
        request.maxCombinations,
        total,
        [&metadata](const ProblemPoint& combination) {
            return !detail::excludedByCategoricals(metadata, combination);
        });
    const auto sizes = detail::combinationSizes(metadata, combinations);
    const auto targets = detail::firstPassTargets(metadata, request, combinations, sizes);
    ASSERT_EQ(targets.size(), combinations.size());

    // U, counted here from the declaration: live combinations, projected onto the enumerable
    // parameters that have no weights.
    std::set<std::string> unweighted;
    int64_t live = 0;
    for(size_t i = 0; i < combinations.size(); ++i)
    {
        if(sizes[i] <= 0.0)
        {
            EXPECT_EQ(targets[i], 0) << detail::describe(combinations[i]);
            continue;
        }
        ++live;
        ProblemPoint projected;
        for(const auto& parameter : metadata.parameters)
        {
            if(parameter.weights.empty() && !parameter.enumerable().empty())
            {
                projected[parameter.name] = combinations[i].at(parameter.name);
            }
        }
        unweighted.insert(detail::describe(projected));
    }
    const auto distinct = static_cast<int64_t>(unweighted.size());
    ASSERT_GT(live, 0);
    ASSERT_GT(distinct, 0);

    int64_t sum = 0;
    for(const auto target : targets)
    {
        sum += target;
    }
    EXPECT_EQ(sum, request.pointsPerCombination * distinct);
    EXPECT_LT(sum * 4, request.pointsPerCombination * live)
        << "the weighted axes split the budget; one budget for each combination would be "
        << request.pointsPerCombination * live;

    // Every declared value that some live combination holds is drawn, weighted or not, and each
    // weighted value holds its declared share of the first pass.
    for(const auto& parameter : metadata.parameters)
    {
        const auto values = parameter.enumerable();
        for(size_t v = 0; v < values.size(); ++v)
        {
            int64_t held = 0;
            bool someLiveCombinationHoldsIt = false;
            for(size_t i = 0; i < combinations.size(); ++i)
            {
                if(combinations[i].at(parameter.name) == values[v])
                {
                    held += targets[i];
                    someLiveCombinationHoldsIt = someLiveCombinationHoldsIt || sizes[i] > 0.0;
                }
            }
            if(someLiveCombinationHoldsIt)
            {
                EXPECT_GE(held, 1) << GetParam() << " " << parameter.name << " value " << v;
            }
            if(!parameter.weights.empty())
            {
                const auto& value = parameter.values[v];
                EXPECT_NEAR(static_cast<double>(held) / static_cast<double>(sum),
                            parameter.relativeWeight(value)
                                / static_cast<double>(parameter.values.size()),
                            0.03)
                    << GetParam() << " " << parameter.name << "=" << value;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SdpaDeclarations,
                         TestShippedFirstPass,
                         ::testing::Values("sdpa_fwd", "sdpa_bwd"));

} // namespace hipdnn_corpus_gen
