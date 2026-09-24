// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/PointFilter.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

/// @file TestPointFilter.cpp
/// @brief `--keep`, whose only job is to be exact about what it removed.
///
/// A filter that silently keeps everything is worse than no filter: the run reports itself as
/// narrow, the corpus comes back broad, and the model trained on it carries a bias nobody looks
/// for. So the two properties worth pinning are that an undeclared name is refused rather than
/// ignored, and that a point without the filtered parameter fails rather than passes.

using namespace hipdnn_corpus_gen;

namespace
{

const std::vector<std::string> kKnown{"dtype", "head_dim", "is_causal"};

} // namespace

TEST(TestPointFilter, AClauseIsAcceptedWithOrWithoutTheColumnPrefix)
{
    std::vector<KeepClause> parsed;
    std::string error;

    ASSERT_TRUE(parseKeepClauses({"q.dtype=bf16", "head_dim=128"}, kKnown, parsed, error)) << error;
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

    // `headdim` for `head_dim` is the whole failure mode: ignored, it filters nothing and the
    // manifest still says the run was filtered.
    EXPECT_FALSE(parseKeepClauses({"q.headdim=128"}, kKnown, parsed, error));
    EXPECT_NE(error.find("headdim"), std::string::npos) << error;
}

TEST(TestPointFilter, AClauseWithNoValueIsRefused)
{
    std::vector<KeepClause> parsed;
    std::string error;

    EXPECT_FALSE(parseKeepClauses({"q.dtype"}, kKnown, parsed, error));
    EXPECT_FALSE(parseKeepClauses({"q.dtype="}, kKnown, parsed, error));
    EXPECT_FALSE(parseKeepClauses({"=bf16"}, kKnown, parsed, error));
}

TEST(TestPointFilter, EveryDeclaredTypeIsFilteredWithTheSameTextSpelling)
{
    const ProblemPoint point{{"dtype", std::string("bf16")},
                             {"head_dim", int64_t{128}},
                             {"is_causal", true}};

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
    // An engine's kernel table covers a set of head dims, not one. Under a conjunctive reading
    // `--keep q.head_dim=64 --keep q.head_dim=128` names the empty corpus, and a run asked for
    // two facets would come back with nothing and no error to explain it.
    const std::vector<KeepClause> either{{"head_dim", "64"}, {"head_dim", "128"}};

    EXPECT_TRUE(keeps(either, ProblemPoint{{"head_dim", int64_t{64}}}));
    EXPECT_TRUE(keeps(either, ProblemPoint{{"head_dim", int64_t{128}}}));
    EXPECT_FALSE(keeps(either, ProblemPoint{{"head_dim", int64_t{192}}}));

    // Different parameters still conjoin, so widening one facet does not widen another.
    const std::vector<KeepClause> mixed{
        {"head_dim", "64"}, {"head_dim", "128"}, {"dtype", "bf16"}};
    EXPECT_TRUE(keeps(mixed, ProblemPoint{{"head_dim", int64_t{64}},
                                          {"dtype", std::string("bf16")}}));
    EXPECT_FALSE(keeps(mixed, ProblemPoint{{"head_dim", int64_t{64}},
                                           {"dtype", std::string("fp16")}}));
}

TEST(TestPointFilter, APointWithoutTheFilteredParameterFails)
{
    // It belongs to an operation that has no such facet. Admitting it would put exactly the
    // problems the filter was written to exclude into a corpus reporting itself as filtered.
    const ProblemPoint elsewhere{{"m", int64_t{4096}}, {"n", int64_t{4096}}};
    EXPECT_FALSE(keeps({{"dtype", "bf16"}}, elsewhere));
}
