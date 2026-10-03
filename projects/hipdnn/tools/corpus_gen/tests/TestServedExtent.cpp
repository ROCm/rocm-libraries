// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/ServedExtent.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

/// @file TestServedExtent.cpp
/// @brief The edges of what an engine serves, as a corpus reports and takes them.

using namespace hipdnn_corpus_gen;

namespace
{

OperationMetadata toy()
{
    auto load = parseOperationMetadata(nlohmann::json::parse(R"({
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
    })"));
    EXPECT_TRUE(load.ok());
    return load.metadata.value_or(OperationMetadata{});
}

ProblemPoint at(int64_t m, int64_t n, const std::string& dtype = "fp32")
{
    return ProblemPoint{{"M", m}, {"N", n}, {"dtype", dtype}};
}

} // namespace

TEST(TestServedExtent, EachNumericParameterGetsItsRangeAndAServedPointAtEachEnd)
{
    const auto extent = servedExtent(toy(), {at(8, 64), at(4096, 2), at(512, 512)});

    ASSERT_EQ(extent.ranges.size(), 2U) << "dtype is not a range";
    EXPECT_EQ(extent.ranges.at("M"), (std::pair<int64_t, int64_t>{8, 4096}));
    EXPECT_EQ(extent.ranges.at("N"), (std::pair<int64_t, int64_t>{2, 512}));
    ASSERT_EQ(extent.edges.size(), 4U);
    EXPECT_EQ(extent.edges[0].parameter, "M");
    EXPECT_EQ(extent.edges[0].end, "low");
    EXPECT_EQ(std::get<int64_t>(extent.edges[0].point.at("M")), 8);
    EXPECT_EQ(std::get<int64_t>(extent.edges[1].point.at("M")), 4096);
    EXPECT_EQ(std::get<int64_t>(extent.edges[3].point.at("N")), 512);
    EXPECT_EQ(extent.servedPoints, 3U);
}

TEST(TestServedExtent, TiesGoToTheSamePointWhateverTheOrder)
{
    // Regenerated from one seed, a corpus takes the same edges: the order points arrive in
    // must not decide which of two equally extreme ones is taken.
    const auto forward = servedExtent(toy(), {at(4096, 2, "fp32"), at(4096, 9, "fp16")});
    const auto backward = servedExtent(toy(), {at(4096, 9, "fp16"), at(4096, 2, "fp32")});
    EXPECT_EQ(detail::describe(forward.edges[1].point), detail::describe(backward.edges[1].point));
}

TEST(TestServedExtent, ACombinationsEdgesCountAsServedEvenWhenNotSelected)
{
    ProblemCorpus corpus;
    CombinationResult combination;
    combination.problems = {at(64, 64)};
    combination.lowest = {at(1, 64), at(64, 1)};
    combination.highest = {at(9000, 64), at(64, 70)};
    corpus.combinations.push_back(combination);

    const auto extent = servedExtent(toy(), servedPoints(corpus, {at(64, 64)}));
    EXPECT_EQ(extent.ranges.at("M"), (std::pair<int64_t, int64_t>{1, 9000}));
    EXPECT_EQ(extent.ranges.at("N"), (std::pair<int64_t, int64_t>{1, 70}));
}

TEST(TestServedExtent, NothingServedHasNoExtent)
{
    const auto extent = servedExtent(toy(), {});
    EXPECT_TRUE(extent.ranges.empty());
    EXPECT_TRUE(extent.edges.empty());
}
