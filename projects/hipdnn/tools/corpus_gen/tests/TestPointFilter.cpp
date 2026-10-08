// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/PointFilter.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

/// @file TestPointFilter.cpp
/// @brief `--keep`, whose only job is to be exact about what it removed.
///
/// An undeclared name is refused, and a point lacking the filtered parameter fails.

using namespace hipdnn_corpus_gen;

namespace
{

const std::vector<std::string> KNOWN{"dtype", "head_dim", "is_causal"};

} // namespace

TEST(TestPointFilter, AClauseIsAcceptedWithOrWithoutTheColumnPrefix)
{
    std::vector<KeepClause> parsed;
    std::string error;

    ASSERT_TRUE(parseKeepClauses({"q.dtype=bf16", "head_dim=128"}, KNOWN, parsed, error)) << error;
    ASSERT_EQ(parsed.size(), 2u);
    EXPECT_EQ(parsed[0].parameter, "dtype");
    EXPECT_EQ(parsed[0].value, "bf16");
    EXPECT_EQ(parsed[1].parameter, "head_dim");
    EXPECT_EQ(parsed[1].value, "128");
}

TEST(TestPointFilter, AParameterNoDeclarationDeclaresIsRefusedRatherThanIgnored)
{
    std::vector<KeepClause> parsed;
    std::string error;

    // A typo would otherwise filter nothing while the manifest claims a filter.
    EXPECT_FALSE(parseKeepClauses({"q.headdim=128"}, KNOWN, parsed, error));
    EXPECT_NE(error.find("headdim"), std::string::npos) << error;
}

TEST(TestPointFilter, AClauseWithNoValueIsRefused)
{
    std::vector<KeepClause> parsed;
    std::string error;

    EXPECT_FALSE(parseKeepClauses({"q.dtype"}, KNOWN, parsed, error));
    EXPECT_FALSE(parseKeepClauses({"q.dtype="}, KNOWN, parsed, error));
    EXPECT_FALSE(parseKeepClauses({"=bf16"}, KNOWN, parsed, error));
}

TEST(TestPointFilter, EveryDeclaredTypeIsFilteredWithTheSameTextSpelling)
{
    const ProblemPoint point{
        {"dtype", std::string("bf16")}, {"head_dim", int64_t{128}}, {"is_causal", true}};

    EXPECT_TRUE(keeps({{"dtype", "bf16"}}, point));
    EXPECT_TRUE(keeps({{"head_dim", "128"}}, point));
    EXPECT_TRUE(keeps({{"is_causal", "true"}}, point));

    EXPECT_FALSE(keeps({{"dtype", "fp16"}}, point));
    EXPECT_FALSE(keeps({{"head_dim", "64"}}, point));
    EXPECT_FALSE(keeps({{"is_causal", "false"}}, point));
}

TEST(TestPointFilter, ClausesAreConjunctiveAndAnEmptyFilterKeepsEverything)
{
    const ProblemPoint point{{"dtype", std::string("bf16")}, {"head_dim", int64_t{128}}};

    EXPECT_TRUE(keeps({}, point));
    EXPECT_TRUE(keeps({{"dtype", "bf16"}, {"head_dim", "128"}}, point));
    EXPECT_FALSE(keeps({{"dtype", "bf16"}, {"head_dim", "64"}}, point));
}

TEST(TestPointFilter, RepeatingAParameterWidensItRatherThanEmptyingTheCorpus)
{
    // A conjunctive reading would make this the empty corpus.
    const std::vector<KeepClause> either{{"head_dim", "64"}, {"head_dim", "128"}};

    EXPECT_TRUE(keeps(either, ProblemPoint{{"head_dim", int64_t{64}}}));
    EXPECT_TRUE(keeps(either, ProblemPoint{{"head_dim", int64_t{128}}}));
    EXPECT_FALSE(keeps(either, ProblemPoint{{"head_dim", int64_t{192}}}));

    // Different parameters still conjoin, so widening one facet does not widen another.
    const std::vector<KeepClause> mixed{{"head_dim", "64"}, {"head_dim", "128"}, {"dtype", "bf16"}};
    EXPECT_TRUE(
        keeps(mixed, ProblemPoint{{"head_dim", int64_t{64}}, {"dtype", std::string("bf16")}}));
    EXPECT_FALSE(
        keeps(mixed, ProblemPoint{{"head_dim", int64_t{64}}, {"dtype", std::string("fp16")}}));
}

TEST(TestPointFilter, APointWithoutTheFilteredParameterFails)
{
    const ProblemPoint elsewhere{{"m", int64_t{4096}}, {"n", int64_t{4096}}};
    EXPECT_FALSE(keeps({{"dtype", "bf16"}}, elsewhere));
}
