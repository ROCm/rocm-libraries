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
#include <fstream>
#include <map>
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

TEST(TestProblemSpace, EveryShippedSdpaCombinationIsExploredAndNoDeclaredValueIsDropped)
{
    // Past maxCombinations the product is sampled; a facet added to SDPA must not push it over,
    // or combinations (and so graphs) silently vanish from every corpus.
    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        std::ifstream file(hipdnn_corpus_gen::test::operationsDir() + "/" + operation
                           + ".opmeta.json");
        ASSERT_TRUE(file.good()) << operation;
        const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
        ASSERT_TRUE(parsed.ok()) << operation;

        size_t total = 0;
        const auto combinations = detail::categoricalCombinations(
            parsed.metadata->parameters, ExplorationRequest{}.maxCombinations, total);
        EXPECT_EQ(combinations.size(), total) << operation;
        // is_causal 2 x alignment 2 x dtype 3 x layout 3 x scale 3, and for the forward
        // generate_stats 2 x window 4.
        EXPECT_EQ(total, operation == "sdpa_fwd" ? size_t{864} : size_t{108}) << operation;
    }
}

TEST(TestProblemSpace, TheShippedSdpaConstraintsAdmitCrossAttentionOnlyWithoutACausalMask)
{
    std::ifstream file(hipdnn_corpus_gen::test::operationsDir() + "/sdpa_fwd.opmeta.json");
    ASSERT_TRUE(file.good());
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    ASSERT_TRUE(parsed.ok());

    const auto admits
        = [&](int64_t seqlenQ, int64_t seqlenK, bool causal, const std::string& window) {
              const ProblemPoint point{{"batch", int64_t{1}},
                                       {"heads", int64_t{8}},
                                       {"heads_kv", int64_t{8}},
                                       {"seqlen_q", seqlenQ},
                                       {"seqlen_k", seqlenK},
                                       {"head_dim", int64_t{64}},
                                       {"is_causal", causal},
                                       {"alignment", std::string("top_left")},
                                       {"dtype", std::string("fp16")},
                                       {"generate_stats", false},
                                       {"layout", std::string("bshd")},
                                       {"scale", std::string("inv_sqrt_d")},
                                       {"window", window}};
              return detail::satisfiesConstraints(*parsed.metadata, point);
          };

    EXPECT_TRUE(admits(4096, 77, false, "none")) << "cross attention is bidirectional";
    EXPECT_FALSE(admits(4096, 77, true, "none")) << "a causal mask needs seqlen_q <= seqlen_k";
    EXPECT_TRUE(admits(2048, 2048, true, "none"));
    EXPECT_TRUE(admits(2048, 2048, true, "local_1024"));
    EXPECT_TRUE(admits(2048, 2048, false, "local_1024"))
        << "the frontend accepts a left bound under any mask";
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

TEST(TestProblemSpace, EqualWeightsAreExactlyNoWeights)
{
    // Weights are relative: declaring every value equal must not move a single problem, and a
    // declaration without weights is the unweighted exploration.
    for(const int64_t target : {int64_t{0}, int64_t{250}})
    {
        ExplorationRequest request;
        request.pointsPerCombination = 25;
        request.corpusTarget = target;
        request.seed = 13;

        const auto absent = exploreProblemSpace(weightedToy(""), request, ACCEPT_EVERYTHING);
        const auto equal = exploreProblemSpace(
            weightedToy(R"(, "weights": {"a": 3, "b": 3, "c": 3})"), request, ACCEPT_EVERYTHING);

        EXPECT_EQ(described(equal), described(absent)) << "target " << target;
    }
}

} // namespace hipdnn_corpus_gen
