// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Registration drops bundles the run's --gtest_filter is about to drop, using its own
// matcher because GTest's is internal. These pin that matcher to GTest's grammar: a
// divergence would silently remove tests from a run, which is the failure this suite
// exists to catch.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/GTestFilter.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests::bundle;

TEST(TestGtestFilter, StarSelectsEverything)
{
    EXPECT_TRUE(gtestFilterSelects("*", "quick_Conv_Default.1_2_3"));
    EXPECT_TRUE(gtestFilterSelects("*", "A.B"));
}

TEST(TestGtestFilter, ExactNameSelectsOnlyThatCase)
{
    EXPECT_TRUE(gtestFilterSelects("Suite.Case", "Suite.Case"));
    EXPECT_FALSE(gtestFilterSelects("Suite.Case", "Suite.Case2"));
    EXPECT_FALSE(gtestFilterSelects("Suite.Case", "Suite.Cas"));
}

TEST(TestGtestFilter, QuestionMarkMatchesExactlyOneCharacter)
{
    EXPECT_TRUE(gtestFilterSelects("quick_?onv.A", "quick_Conv.A"));
    EXPECT_FALSE(gtestFilterSelects("quick_?onv.A", "quick_onv.A"));
    EXPECT_FALSE(gtestFilterSelects("quick_?onv.A", "quick_CConv.A"));
}

TEST(TestGtestFilter, StarMatchesAcrossTheSuiteSeparator)
{
    EXPECT_TRUE(gtestFilterSelects("*Pointwise*", "quick_Pointwise_Default.2_4"));
    EXPECT_TRUE(gtestFilterSelects("quick_*.2_4", "quick_Pointwise_Default.2_4"));
    EXPECT_FALSE(gtestFilterSelects("*Reduction*", "quick_Pointwise_Default.2_4"));
}

TEST(TestGtestFilter, StarBacktracksPastAnEarlyPartialMatch)
{
    EXPECT_TRUE(gtestFilterSelects("*ab*abc", "xxabababc"));
    EXPECT_FALSE(gtestFilterSelects("*ab*abd", "xxabababc"));
}

TEST(TestGtestFilter, ColonSeparatesAlternatives)
{
    EXPECT_TRUE(gtestFilterSelects("A.*:B.*", "B.x"));
    EXPECT_TRUE(gtestFilterSelects("A.*:B.*", "A.x"));
    EXPECT_FALSE(gtestFilterSelects("A.*:B.*", "C.x"));
}

TEST(TestGtestFilter, NegativeSectionRemovesMatches)
{
    EXPECT_TRUE(gtestFilterSelects("*-*DISABLED*", "A.ok"));
    EXPECT_FALSE(gtestFilterSelects("*-*DISABLED*", "A.DISABLED_x"));
    EXPECT_FALSE(gtestFilterSelects("A.*-A.b:A.c", "A.c"));
    EXPECT_TRUE(gtestFilterSelects("A.*-A.b:A.c", "A.d"));
}

TEST(TestGtestFilter, EmptyPositivePartBeforeDashMeansEverything)
{
    EXPECT_TRUE(gtestFilterSelects("-quick_*", "standard_X.y"));
    EXPECT_FALSE(gtestFilterSelects("-quick_*", "quick_X.y"));
}

TEST(TestGtestFilter, OnlyTheFirstDashSplitsPositiveFromNegative)
{
    // The second dash is an ordinary character of the negative pattern.
    EXPECT_FALSE(gtestFilterSelects("*-A.b-c", "A.b-c"));
    EXPECT_TRUE(gtestFilterSelects("*-A.b-c", "A.b"));
}

TEST(TestGtestFilter, TrailingDashHasAnEmptyNegativeSection)
{
    EXPECT_TRUE(gtestFilterSelects("*Pointwise*-", "quick_Pointwise.x"));
}

TEST(TestGtestFilter, EmptyFilterSelectsNothing)
{
    EXPECT_FALSE(gtestFilterSelects("", "A.b"));
}

TEST(TestGtestFilter, TrailingColonDoesNotWidenTheSelection)
{
    EXPECT_TRUE(gtestFilterSelects("quick_*:", "quick_A.b"));
    EXPECT_FALSE(gtestFilterSelects("quick_*:", "standard_A.b"));
}

namespace
{

DiscoveredBundle bundleNamed(const std::string& suite, const std::string& test)
{
    DiscoveredBundle bundle;
    bundle.suiteName = suite;
    bundle.testName = test;
    return bundle;
}

} // namespace

TEST(TestGtestFilter, SplitKeepsDiscoveryOrderInBothHalves)
{
    std::vector<DiscoveredBundle> discovered;
    discovered.push_back(bundleNamed("quick_Pointwise", "a"));
    discovered.push_back(bundleNamed("standard_Reduction", "b"));
    discovered.push_back(bundleNamed("quick_Pointwise", "c"));
    discovered.push_back(bundleNamed("standard_Reduction", "d"));

    const auto split = splitByGTestFilter(std::move(discovered), "quick_*");

    ASSERT_EQ(split.selected.size(), 2u);
    EXPECT_EQ(split.selected[0].testName, "a");
    EXPECT_EQ(split.selected[1].testName, "c");
    ASSERT_EQ(split.excluded.size(), 2u);
    EXPECT_EQ(split.excluded[0].testName, "b");
    EXPECT_EQ(split.excluded[1].testName, "d");
}

TEST(TestGtestFilter, SplitMatchesOnTheJoinedSuiteAndCaseName)
{
    std::vector<DiscoveredBundle> discovered;
    discovered.push_back(bundleNamed("quick_Pointwise", "case1"));
    discovered.push_back(bundleNamed("quick_Pointwise", "case2"));

    const auto split = splitByGTestFilter(std::move(discovered), "quick_Pointwise.case2");

    ASSERT_EQ(split.selected.size(), 1u);
    EXPECT_EQ(split.selected[0].testName, "case2");
    EXPECT_EQ(split.excluded.size(), 1u);
}

// NOLINTEND(readability-identifier-naming)
