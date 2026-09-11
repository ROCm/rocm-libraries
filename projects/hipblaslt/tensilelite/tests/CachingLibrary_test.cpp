// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionLibrary.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::testing;

namespace
{
    // A sub-library that hands back a brand-new, uniquely-named solution on
    // every findBestSolution() call and records how many times it was hit.
    // Because each call yields a distinct solution, we can tell from the
    // returned solution (and the call count) whether CachingLibrary served a
    // cached entry or delegated to the sub-library for a fresh lookup.
    class CountingSubLibrary : public ContractionLibrary
    {
    public:
        mutable std::atomic<int> callCount{0};

        std::shared_ptr<ContractionSolution> getSolutionByIndex(ContractionProblemGemm const&,
                                                                 Hardware const&,
                                                                 int) const override
        {
            return {};
        }

        std::shared_ptr<ContractionSolution> findBestSolution(ContractionProblemGemm const&,
                                                              Hardware const&,
                                                              double* fitness = nullptr) const override
        {
            int index = ++callCount;
            if(fitness)
                *fitness = static_cast<double>(index);
            return makeSolution("sol_" + std::to_string(index), index);
        }

        SolutionSet<ContractionSolution> findAllSolutions(ContractionProblemGemm const&,
                                                          Hardware const&,
                                                          SolutionLibrarySearchType
                                                          = SolutionLibrarySearchType::DEFAULT) const override
        {
            return {};
        }

        SolutionSet<ContractionSolution>
            findAllSolutionsGroupedGemm(std::vector<ContractionProblemGemm> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return {};
        }

        std::string type() const override
        {
            return "CountingSub";
        }

        std::string description() const override
        {
            return "CountingSub";
        }
    };

    ContractionProblemGemm problemWithSmCountTarget(int smCountTarget)
    {
        auto problem = dummyProblem();
        problem.setParams().setSmCountTarget(smCountTarget);
        return problem;
    }

    ContractionProblemGemm problemWithSchedulingMode(int schedulingMode)
    {
        auto problem = dummyProblem();
        problem.setParams().setStreamKTileSchedulingMode(schedulingMode);
        return problem;
    }
} // namespace

// Same problem type and size but different smCountTarget values must produce
// distinct cache entries: the second lookup cannot be served from the entry
// cached for the first, and both entries coexist afterwards.
TEST(CachingLibraryTest, DifferentSmCountTargetCreatesDistinctCacheEntries)
{
    auto sub     = std::make_shared<CountingSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    auto problem64  = problemWithSmCountTarget(64);
    auto problem128 = problemWithSmCountTarget(128);

    auto first  = caching->findBestSolution(problem64, device);
    auto second = caching->findBestSolution(problem128, device);

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    // Two separate lookups reached the sub-library -> two distinct entries.
    EXPECT_EQ(sub->callCount.load(), 2)
        << "smCountTarget must be part of the cache key; the second problem "
           "should not reuse the entry cached for the first";
    EXPECT_NE(first->solutionName, second->solutionName)
        << "Distinct smCountTarget values must not collapse to the same entry";

    // Each smCountTarget resolves to its own cached solution.
    auto cached64  = caching->findSolutionInCache(problem64, device);
    auto cached128 = caching->findSolutionInCache(problem128, device);

    ASSERT_NE(cached64, nullptr);
    ASSERT_NE(cached128, nullptr);
    EXPECT_EQ(cached64->solutionName, first->solutionName);
    EXPECT_EQ(cached128->solutionName, second->solutionName);
    EXPECT_NE(cached64->solutionName, cached128->solutionName);
}

// Re-running an identical problem (same size AND same smCountTarget) must hit
// the existing cache entry rather than create a new one.
TEST(CachingLibraryTest, SameSmCountTargetReusesCacheEntry)
{
    auto sub     = std::make_shared<CountingSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    auto problem = problemWithSmCountTarget(64);

    auto first  = caching->findBestSolution(problem, device);
    auto second = caching->findBestSolution(problem, device);

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    EXPECT_EQ(sub->callCount.load(), 1)
        << "Identical problem (same size and smCountTarget) must be served from cache";
    EXPECT_EQ(first->solutionName, second->solutionName);
}

// Same problem type and size but different streamKTileSchedulingMode values
// must produce distinct cache entries: the scheduling mode changes solution
// selection, so it cannot be collapsed onto a single cached entry.
TEST(CachingLibraryTest, DifferentSchedulingModeCreatesDistinctCacheEntries)
{
    auto sub     = std::make_shared<CountingSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    auto problemOff = problemWithSchedulingMode(0);
    auto problemOn  = problemWithSchedulingMode(1);

    auto first  = caching->findBestSolution(problemOff, device);
    auto second = caching->findBestSolution(problemOn, device);

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    // Two separate lookups reached the sub-library -> two distinct entries.
    EXPECT_EQ(sub->callCount.load(), 2)
        << "streamKTileSchedulingMode must be part of the cache key; the second "
           "problem should not reuse the entry cached for the first";
    EXPECT_NE(first->solutionName, second->solutionName)
        << "Distinct streamKTileSchedulingMode values must not collapse to the same entry";

    // Each scheduling mode resolves to its own cached solution.
    auto cachedOff = caching->findSolutionInCache(problemOff, device);
    auto cachedOn  = caching->findSolutionInCache(problemOn, device);

    ASSERT_NE(cachedOff, nullptr);
    ASSERT_NE(cachedOn, nullptr);
    EXPECT_EQ(cachedOff->solutionName, first->solutionName);
    EXPECT_EQ(cachedOn->solutionName, second->solutionName);
    EXPECT_NE(cachedOff->solutionName, cachedOn->solutionName);
}

// Re-running an identical problem (same size AND same scheduling mode) must
// hit the existing cache entry rather than create a new one.
TEST(CachingLibraryTest, SameSchedulingModeReusesCacheEntry)
{
    auto sub     = std::make_shared<CountingSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    auto problem = problemWithSchedulingMode(1);

    auto first  = caching->findBestSolution(problem, device);
    auto second = caching->findBestSolution(problem, device);

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    EXPECT_EQ(sub->callCount.load(), 1)
        << "Identical problem (same size and scheduling mode) must be served from cache";
    EXPECT_EQ(first->solutionName, second->solutionName);
}
