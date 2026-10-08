// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCorpusOutput.cpp
 * @brief Covers the two forms a problem is written in, and that they agree.
 *
 * The CSV `q.*` columns and the `--query` argument are rendered by separate code paths.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/CorpusOutput.hpp>

#include <map>
#include <string>

namespace hipdnn_corpus_gen
{
namespace
{

ProblemPoint conv()
{
    return ProblemPoint{{"N", int64_t{8}},
                        {"C", int64_t{64}},
                        {"H", int64_t{56}},
                        {"causal", false},
                        {"dtype", std::string{"fp16"}}};
}

} // namespace

TEST(TestCorpusOutput, TheHeaderAndTheRowLineUp)
{
    // A column-order mismatch would silently transpose features.
    const auto header = asQueryColumns(conv(), true);
    const auto row = asQueryColumns(conv(), false);

    const auto count
        = [](const std::string& text) { return std::count(text.begin(), text.end(), ','); };
    EXPECT_EQ(count(header), count(row));
    EXPECT_EQ(header, "q.C,q.H,q.N,q.causal,q.dtype") << "columns are not in point order";
    EXPECT_EQ(row, "64,56,8,false,fp16");
}

TEST(TestCorpusOutput, ColumnsCarryTheQualifierTheTrainerHashes)
{
    // tools/uhd_gen/features.py requires q./kernel./device. qualified feature names.
    for(const auto& name : {"q.C", "q.H", "q.N", "q.dtype"})
    {
        EXPECT_NE(asQueryColumns(conv(), true).find(name), std::string::npos) << name;
    }
    EXPECT_EQ(asQueryColumns(conv(), true).find("q.q."), std::string::npos);
}

TEST(TestCorpusOutput, AQueryArgumentRoundTripsBackToTheSameProblem)
{
    // What the generator emits must be what the benchmark parses.
    const auto parsed = parseQueryArgument(asQueryArgument(conv()));

    std::map<std::string, std::string> recovered(parsed.begin(), parsed.end());
    ASSERT_EQ(recovered.size(), conv().size());
    EXPECT_EQ(recovered.at("N"), "8");
    EXPECT_EQ(recovered.at("C"), "64");
    EXPECT_EQ(recovered.at("H"), "56");
    EXPECT_EQ(recovered.at("dtype"), "fp16");
    EXPECT_EQ(recovered.at("causal"), "false");
}

TEST(TestCorpusOutput, EveryDeclaredParameterReachesBothForms)
{
    const auto point = conv();
    const auto header = asQueryColumns(point, true);
    const auto query = asQueryArgument(point);

    for(const auto& entry : point)
    {
        EXPECT_NE(header.find("q." + entry.first), std::string::npos)
            << entry.first << " is missing from the CSV header";
        EXPECT_NE(query.find(entry.first + "="), std::string::npos)
            << entry.first << " is missing from the query argument";
    }
}

TEST(TestCorpusOutput, BooleansAreSpelledRatherThanNumbered)
{
    // "0"/"1" would be inferred as a numeric feature.
    const ProblemPoint point{{"causal", true}, {"padded", false}};
    EXPECT_EQ(asQueryColumns(point, false), "true,false");
    EXPECT_EQ(asQueryArgument(point), "causal=true,padded=false");
}

TEST(TestCorpusOutput, MalformedQueriesAreRefusedRatherThanPartlyParsed)
{
    // A partly parsed query would be a mislabeled training row.
    EXPECT_TRUE(parseQueryArgument("N=8,,C=64").empty());
    EXPECT_TRUE(parseQueryArgument("N=8,C").empty());
    EXPECT_TRUE(parseQueryArgument("=8").empty());
    EXPECT_TRUE(parseQueryArgument("N=").empty());
    EXPECT_TRUE(parseQueryArgument("").empty());

    EXPECT_EQ(parseQueryArgument("N=8").size(), 1U) << "a single valid field must still parse";
}

} // namespace hipdnn_corpus_gen
