// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCorpusOutput.cpp
 * @brief Covers the two forms a problem is written in, and that they agree.
 *
 * A corpus is only useful if the `q.*` columns of a CSV row and the `--query` argument of the
 * command that produced it describe the same problem. They are rendered by separate code paths
 * that must agree forever, and nothing checked it -- the generator's output was validated only
 * by being looked at.
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
                        {"dtype", std::string{"float16"}}};
}

} // namespace

TEST(TestCorpusOutput, TheHeaderAndTheRowLineUp)
{
    // A header and row that disagree about column order transpose two features silently, and
    // the model trains on the wrong ones -- there is nothing in the file to notice it.
    const auto header = asQueryColumns(conv(), true);
    const auto row = asQueryColumns(conv(), false);

    const auto count = [](const std::string& text) {
        return std::count(text.begin(), text.end(), ',');
    };
    EXPECT_EQ(count(header), count(row));
    EXPECT_EQ(header, "q.C,q.H,q.N,q.causal,q.dtype") << "columns are not in point order";
    EXPECT_EQ(row, "64,56,8,false,float16");
}

TEST(TestCorpusOutput, ColumnsCarryTheQualifierTheTrainerHashes)
{
    // tools/uhd_gen/features.py requires every feature to be q./kernel./device. qualified and
    // hashes that signature. Emitting bare names would need a renaming step, which is how the
    // two sides drift while the hash still matches.
    for(const auto& name : {"q.C", "q.H", "q.N", "q.dtype"})
    {
        EXPECT_NE(asQueryColumns(conv(), true).find(name), std::string::npos) << name;
    }
    EXPECT_EQ(asQueryColumns(conv(), true).find("q.q."), std::string::npos);
}

TEST(TestCorpusOutput, AQueryArgumentRoundTripsBackToTheSameProblem)
{
    // The check the plan called for and never got: what the generator emits is what the
    // benchmark can read. Without it a quoting or separator change breaks every command in a
    // ten-thousand-line file at once, and only at harvest time.
    const auto parsed = parseQueryArgument(asQueryArgument(conv()));

    std::map<std::string, std::string> recovered(parsed.begin(), parsed.end());
    ASSERT_EQ(recovered.size(), conv().size());
    EXPECT_EQ(recovered.at("N"), "8");
    EXPECT_EQ(recovered.at("C"), "64");
    EXPECT_EQ(recovered.at("H"), "56");
    EXPECT_EQ(recovered.at("dtype"), "float16");
    EXPECT_EQ(recovered.at("causal"), "false");
}

TEST(TestCorpusOutput, EveryDeclaredParameterReachesBothForms)
{
    // A parameter dropped from one form but not the other is the failure that survives review:
    // the CSV looks complete and the command under-specifies the problem, or the reverse.
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
    // The column is categorical, and "0" would be read back as an integer feature by anything
    // inferring column types -- a boolean silently becoming numeric changes what is learned.
    const ProblemPoint point{{"causal", true}, {"padded", false}};
    EXPECT_EQ(asQueryColumns(point, false), "true,false");
    EXPECT_EQ(asQueryArgument(point), "causal=true,padded=false");
}

TEST(TestCorpusOutput, MalformedQueriesAreRefusedRatherThanPartlyParsed)
{
    // Half a problem read as a whole one is a mislabeled training row, which nothing downstream
    // can detect. Refusing outright is the only safe reading.
    EXPECT_TRUE(parseQueryArgument("N=8,,C=64").empty());
    EXPECT_TRUE(parseQueryArgument("N=8,C").empty());
    EXPECT_TRUE(parseQueryArgument("=8").empty());
    EXPECT_TRUE(parseQueryArgument("N=").empty());
    EXPECT_TRUE(parseQueryArgument("").empty());

    EXPECT_EQ(parseQueryArgument("N=8").size(), 1U) << "a single valid field must still parse";
}

} // namespace hipdnn_corpus_gen
