// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestProblemSpace.cpp
 * @brief Covers the one exploration, against declarations the test writes.
 *
 * The property under test is that there is nothing engine-specific and nothing
 * operation-specific in how the space is explored — only in what is declared. So every case
 * here drives the same function with different metadata, and the oracle is a region the test
 * already knows rather than an engine.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <nlohmann/json.hpp>

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
        "dtype": { "type": "enum", "values": ["float32", "float16"] }
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
    // The failure this replaces: an earlier generator hardcoded float32, so half the engine's
    // kernels were unreachable and nothing in the corpus said so. dtype is a parameter of the
    // problem, and the declaration is what says which values exist.
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
    EXPECT_EQ(dtypes, (std::set<std::string>{"float16", "float32"}));
}

TEST(TestProblemSpace, SearchesTheNumericAxesAndEnumeratesTheCategoricalOnes)
{
    ExplorationRequest request;
    request.pointsPerCombination = 20;
    request.seed = 2;

    const auto corpus = exploreProblemSpace(twoDimsAndADtype(), request, ACCEPT_EVERYTHING);

    EXPECT_EQ(corpus.numericParameters, (std::vector<std::string>{"M", "N"}));

    // Numeric axes vary within one categorical combination; that is the search working.
    std::set<int64_t> distinctM;
    for(const auto& point : corpus.combinations.front().problems)
    {
        distinctM.insert(std::get<int64_t>(point.at("M")));
    }
    EXPECT_GT(distinctM.size(), 5U);
}

TEST(TestProblemSpace, GivesEveryCombinationItsOwnBudget)
{
    // A shared budget is spent by whichever combination runs first, and the corpus then covers
    // one dtype thoroughly and the rest not at all -- while reporting a total that looks whole.
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
    // Coupling between a categorical and a numeric axis is ordinary -- a dtype an engine only
    // serves at small extents, say. The oracle therefore sees the whole point, not a shape.
    const ProblemOracle halfOnlySmall = [](const ProblemPoint& point) {
        const auto dtype = std::get<std::string>(point.at("dtype"));
        return dtype == "float32" || std::get<int64_t>(point.at("M")) <= 64;
    };

    ExplorationRequest request;
    request.pointsPerCombination = 20;
    request.seed = 4;

    const auto corpus = exploreProblemSpace(twoDimsAndADtype(), request, halfOnlySmall);

    for(const auto& point : corpus.problems())
    {
        if(std::get<std::string>(point.at("dtype")) == "float16")
        {
            EXPECT_LE(std::get<int64_t>(point.at("M")), 64);
        }
    }
}

TEST(TestProblemSpace, ProducesOneProblemPerCombinationWhenNothingIsNumeric)
{
    // An operation whose every parameter is categorical still has problems. Returning nothing
    // would drop it from the corpus while looking like an empty region.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "flags_only",
      "parameters": {
        "dtype":  { "type": "enum", "values": ["float32", "float16"] },
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
    // A corpus covering three of twelve dtype/layout combinations, silently, is
    // indistinguishable from one that covered the space.
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
    // §4.3.2: a range is a limit inherent to the operation. Where one is declared it binds;
    // where none is, the ceiling applies and no authored guess narrows the space.
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
    // §4.3.2 calls a range "a limit inherent to the operation, such as one dimension that
    // cannot exceed another", but a range bounds one parameter against constants. A filter
    // fitting inside its input is a relation, and without one the frontend rejected 2559 of
    // 2559 candidates before any engine saw them.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy_conv",
      "parameters": { "H": { "type": "int64" }, "R": { "type": "int64" } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "constraints": [ { "<=": ["$q.R", "$q.H"] } ]
    })");

    ASSERT_EQ(metadata.constraints.size(), 1U);

    EXPECT_TRUE(detail::satisfiesConstraints(
        metadata, ProblemPoint{{"H", int64_t{8}}, {"R", int64_t{3}}}));
    EXPECT_FALSE(detail::satisfiesConstraints(
        metadata, ProblemPoint{{"H", int64_t{3}}, {"R", int64_t{8}}}));

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
    // Treating a malformed relation as satisfied would restore exactly the behaviour the
    // constraint was added to prevent, and would do it silently.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": { "H": { "type": "int64" } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "constraints": [ { "no_such_operator": ["$q.H", 1] } ]
    })");

    EXPECT_FALSE(detail::satisfiesConstraints(metadata, ProblemPoint{{"H", int64_t{8}}}));
}

} // namespace hipdnn_corpus_gen
