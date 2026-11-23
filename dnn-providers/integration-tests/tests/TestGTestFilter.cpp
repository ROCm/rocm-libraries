// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Registration drops bundles the run's --gtest_filter is about to drop, using its own
// matcher because GTest's is internal. These pin that matcher to GTest's grammar: a
// divergence would silently remove tests from a run, which is the failure this suite
// exists to catch.

#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/BundleRegistration.hpp"
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

// Every string of length minLength..maxLength over `alphabet`, shortest first.
std::vector<std::string>
    allStrings(const std::string& alphabet, std::size_t minLength, std::size_t maxLength)
{
    std::vector<std::string> result;
    std::vector<std::string> sameLength{""};
    for(std::size_t length = 0; length <= maxLength; ++length)
    {
        if(length >= minLength)
        {
            result.insert(result.end(), sameLength.begin(), sameLength.end());
        }
        std::vector<std::string> longer;
        for(const auto& text : sameLength)
        {
            for(const char c : alphabet)
            {
                longer.push_back(text + c);
            }
        }
        sameLength = std::move(longer);
    }
    return result;
}

std::string joined(std::initializer_list<std::string_view> parts)
{
    std::string result;
    for(const auto part : parts)
    {
        result += part;
    }
    return result;
}

std::vector<std::string> enumeratedFilters()
{
    const auto shortPatterns = allStrings("a_.*?", 0, 2);
    const auto longPatterns = allStrings("a_.*?", 0, 3);
    const auto tinyPatterns = allStrings("a_.*?", 0, 1);

    std::vector<std::string> filters;
    for(const auto& pattern : longPatterns)
    {
        filters.push_back(pattern);
        filters.push_back("-" + pattern);
    }
    for(const auto& first : shortPatterns)
    {
        for(const auto& second : shortPatterns)
        {
            filters.push_back(joined({first, "-", second}));
            filters.push_back(joined({first, ":", second}));
        }
    }
    for(const auto& positive1 : tinyPatterns)
    {
        for(const auto& positive2 : tinyPatterns)
        {
            for(const auto& negative1 : tinyPatterns)
            {
                for(const auto& negative2 : tinyPatterns)
                {
                    filters.push_back(
                        joined({positive1, ":", positive2, "-", negative1, ":", negative2}));
                }
            }
        }
    }
    for(const auto& first : tinyPatterns)
    {
        for(const auto& second : tinyPatterns)
        {
            for(const auto& third : tinyPatterns)
            {
                filters.push_back(joined({first, "-", second, "-", third}));
            }
        }
    }
    return filters;
}

} // namespace

TEST(TestGtestFilter, MaySelectSuiteHoldsForEverySuiteWithASelectedCase)
{
    const auto suites = allStrings("a_", 0, 2);
    const auto testNames = allStrings("a_.", 0, 3);

    std::size_t skippedSuites = 0;
    std::size_t parsedSuitesWithNoSelection = 0;
    for(const auto& filter : enumeratedFilters())
    {
        for(const auto& suite : suites)
        {
            const bool maySelect = gtestFilterMaySelectSuite(filter, suite);
            bool anySelected = false;
            for(const auto& testName : testNames)
            {
                if(gtestFilterSelects(filter, joined({suite, ".", testName})))
                {
                    anySelected = true;
                    EXPECT_TRUE(maySelect) << "filter '" << filter << "' selects '" << suite << "."
                                           << testName << "' but skips the suite";
                }
            }
            skippedSuites += maySelect ? 0 : 1;
            parsedSuitesWithNoSelection += (maySelect && !anySelected) ? 1 : 0;
        }
    }
    EXPECT_GT(skippedSuites, 0u);
    EXPECT_GT(parsedSuitesWithNoSelection, 0u);
}

TEST(TestGtestFilter, MaySelectSuiteExpectedVerdicts)
{
    EXPECT_FALSE(gtestFilterMaySelectSuite("", "quick_Matmul_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite("", ""));
    EXPECT_TRUE(gtestFilterMaySelectSuite("*", "quick_Matmul_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite("-quick_*", "standard_Matmul_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite("-quick_*", "quick_Matmul_X"));

    EXPECT_FALSE(gtestFilterMaySelectSuite("quick_SdpaFwd_*.*", "quick_SdpaFwdPlan_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite("quick_SdpaFwd_*.*", "quick_SdpaFwd_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite("quick_*Sdpa*", "quick_Matmul_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite("quick_*Sdpa*", "standard_Matmul_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite("quick_?atmul_X.case", "quick_Matmul_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite("quick_?atmul_X.case", "quick_atmul_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite("Suite.Case", "Suite.Case"));

    const std::string excludeOneSuite = "*-*_SdpaFwd_bhsd_fp16_hd128_causal_gqa4_Small.*";
    EXPECT_FALSE(gtestFilterMaySelectSuite(excludeOneSuite,
                                           "quick_SdpaFwd_bhsd_fp16_hd128_causal_gqa4_Small"));
    EXPECT_TRUE(gtestFilterMaySelectSuite(excludeOneSuite,
                                          "quick_SdpaFwd_bhsd_fp16_hd128_causal_gqa4_Smaller"));
    EXPECT_TRUE(gtestFilterMaySelectSuite(excludeOneSuite,
                                          "quick_SdpaFwd_bhsd_fp16_hd128_causal_mha_Small"));

    EXPECT_TRUE(gtestFilterMaySelectSuite("*-*_SdpaBackward_*.prbu01_*", "quick_SdpaBackward_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite("*-A.*?", "A"));
    EXPECT_FALSE(gtestFilterMaySelectSuite("*-A.**", "A"));
    EXPECT_FALSE(gtestFilterMaySelectSuite("*-A*", "A"));
    EXPECT_TRUE(gtestFilterMaySelectSuite("*-A.b-c", "A"));
}

TEST(TestGtestFilter, MaySelectSuiteOnTheLaneFilters)
{
    const std::string hipMlops = "quick_*:standard_*:Smoke/*:Quick/*:Standard/*-*DISABLED*";
    const std::string asmSdpa
        = "quick_*Sdpa*:standard_*Sdpa*:Smoke/*Sdpa*-*DISABLED*:*_SdpaBackward_*.prbu01_*:"
          "*_SdpaBackward_*.prbu03_*:*_SdpaBackward_*.prbu13_*:*_SdpaBackward_*.prbt04_*:"
          "*_SdpaBackward_*.prbt04n_*:*_SdpaBackward_*.prbr05_*";
    const std::string flyDslSdpa = "quick_SdpaFwd_*.*:standard_SdpaFwd_*.*-*DISABLED*:"
                                   "*_SdpaFwd_bhsd_fp16_hd128_causal_gqa4_Small.*:"
                                   "*_SdpaFwd_bhsd_fp16_hd128_causal_mha_Small.*";

    EXPECT_TRUE(gtestFilterMaySelectSuite(hipMlops, "quick_SdpaFwdPlan_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite(hipMlops, "standard_RMSNorm_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite(hipMlops, "comprehensive_SdpaFwdPlan_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite(hipMlops, "full_Matmul_X"));

    EXPECT_TRUE(gtestFilterMaySelectSuite(asmSdpa, "quick_SdpaFwdPlan_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite(asmSdpa, "standard_RMSNorm_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite(asmSdpa, "standard_SdpaBackward_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite(asmSdpa, "comprehensive_SdpaFwdPlan_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite(asmSdpa, "full_Matmul_X"));

    EXPECT_FALSE(gtestFilterMaySelectSuite(flyDslSdpa, "quick_SdpaFwdPlan_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite(flyDslSdpa, "quick_SdpaFwd_bhsd_fp16_hd64_X"));
    EXPECT_TRUE(gtestFilterMaySelectSuite(flyDslSdpa, "standard_SdpaFwd_bhsd_fp16_hd64_X"));
    EXPECT_FALSE(
        gtestFilterMaySelectSuite(flyDslSdpa, "quick_SdpaFwd_bhsd_fp16_hd128_causal_gqa4_Small"));
    EXPECT_FALSE(gtestFilterMaySelectSuite(flyDslSdpa, "standard_RMSNorm_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite(flyDslSdpa, "comprehensive_SdpaFwdPlan_X"));
    EXPECT_FALSE(gtestFilterMaySelectSuite(flyDslSdpa, "full_Matmul_X"));
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

    const auto split = detail::splitByGTestFilter(std::move(discovered), "quick_*");

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

    const auto split = detail::splitByGTestFilter(std::move(discovered), "quick_Pointwise.case2");

    ASSERT_EQ(split.selected.size(), 1u);
    EXPECT_EQ(split.selected[0].testName, "case2");
    EXPECT_EQ(split.excluded.size(), 1u);
}

// NOLINTEND(readability-identifier-naming)
