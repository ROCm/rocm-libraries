// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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

namespace
{
    // Models the production pattern: findTopSolutions() decides whether it
    // returned everything it has, and that answer must travel with the result.
    // This sub-library holds up to 10 solutions for smCountTarget==64 ("A") and
    // 1 for smCountTarget==128 ("B"). The only synchronisation added is a gate
    // that holds thread A inside findTopSolutions, after its answer is computed,
    // until thread B's whole CachingLibrary::findTopSolutions() has returned.
    // That is the window in which the old shared flag was overwritten. The wait
    // is bounded and only thread A ever waits, so the gate cannot deadlock.
    class FlagSubLibrary : public ContractionLibrary
    {
    public:
        mutable std::mutex              m;
        mutable std::condition_variable cv;
        mutable bool                    gateArmed = false;
        mutable bool                    aComputed = false;
        mutable bool                    bDone     = false;

        std::shared_ptr<ContractionSolution> getSolutionByIndex(ContractionProblemGemm const&,
                                                                 Hardware const&,
                                                                 int) const override
        {
            return {};
        }
        std::shared_ptr<ContractionSolution>
            findBestSolution(ContractionProblemGemm const&, Hardware const&, double*) const override
        {
            return {};
        }
        SolutionSet<ContractionSolution>
            findAllSolutions(ContractionProblemGemm const&,
                             Hardware const&,
                             SolutionLibrarySearchType = SolutionLibrarySearchType::DEFAULT) const override
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
            return "FlagSub";
        }
        std::string description() const override
        {
            return "FlagSub";
        }

        void markBDone() const
        {
            std::lock_guard<std::mutex> lk(m);
            bDone = true;
            cv.notify_all();
        }

        static bool isA(ContractionProblemGemm const& p)
        {
            return p.getParams().smCountTarget() == 64;
        }

        TopSolutionsResult<ContractionSolution>
            findTopSolutions(ContractionProblemGemm const& problem,
                             Hardware const&,
                             int numSolutions) const override
        {
            int available = isA(problem) ? 10 : 1;
            int n         = std::min(available, numSolutions);
            SolutionVector<ContractionSolution> out;
            for(int i = 0; i < n; i++)
                out.push_back(makeSolution("sol_" + std::to_string(i), i));

            // Correct, per-problem answer: "I returned everything I have".
            bool retAll = (n >= available);

            if(isA(problem))
            {
                std::unique_lock<std::mutex> lk(m);
                if(gateArmed)
                {
                    // A's answer is computed. Let B's whole call run before A
                    // returns, so any state shared between calls gets overwritten.
                    aComputed = true;
                    cv.notify_all();
                    cv.wait_for(lk, std::chrono::seconds(5), [&] { return bDone; });
                }
            }
            return {std::move(out), retAll};
        }
    };
} // namespace

// M7: the already-returned-all bit used to be one atomic shared by all
// threads. Thread A asks for 5 of the 10 solutions it has (correct bit:
// retAll == false). Thread B (different problem, only 1 solution) runs a whole
// findTopSolutions while A is still inside its call, so A used to read B's
// "true" and cache it for A's problem. A later request by A for 10 solutions
// then hit the early return in CachingLibrary::findTopSolutions and got 5. The
// bit now comes back with each call's result, so B cannot reach A's cache entry.
TEST(CachingLibraryTest, ConcurrentFindTopSolutionsDoesNotPoisonAlreadyRetAllCache)
{
    auto sub     = std::make_shared<FlagSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    auto problemA = problemWithSmCountTarget(64);
    auto problemB = problemWithSmCountTarget(128);

    sub->gateArmed = true;

    size_t aFirst = 0;
    std::thread tA([&] { aFirst = caching->findTopSolutions(problemA, device, 5).solutions.size(); });

    // Start B only after A has computed its (correct, false) answer.
    {
        std::unique_lock<std::mutex> lk(sub->m);
        ASSERT_TRUE(
            sub->cv.wait_for(lk, std::chrono::seconds(5), [&] { return sub->aComputed; }));
    }
    size_t bSize = caching->findTopSolutions(problemB, device, 5).solutions.size();
    sub->markBDone();
    tA.join();
    sub->gateArmed = false;

    EXPECT_EQ(aFirst, 5u);
    EXPECT_EQ(bSize, 1u);

    // Same problem A, now asking for everything it has.
    auto again = caching->findTopSolutions(problemA, device, 10);
    EXPECT_EQ(again.solutions.size(), 10u)
        << "thread A's cache entry was marked already-returned-all using thread B's "
           "already-returned-all bit; A got fewer solutions than it asked for";
    EXPECT_FALSE(again.alreadyRetAll);
}

namespace
{
    // Holds 10 solutions for every problem. findTopSolutions(n) returns the
    // first min(n, 10) of them and reports alreadyRetAll == (n >= 10), which is
    // the per-call answer production code computes.
    class TenSolutionSubLibrary : public ContractionLibrary
    {
    public:
        static constexpr int available = 10;

        std::shared_ptr<ContractionSolution> getSolutionByIndex(ContractionProblemGemm const&,
                                                                 Hardware const&,
                                                                 int) const override
        {
            return {};
        }
        std::shared_ptr<ContractionSolution>
            findBestSolution(ContractionProblemGemm const&, Hardware const&, double*) const override
        {
            return {};
        }
        SolutionSet<ContractionSolution>
            findAllSolutions(ContractionProblemGemm const&,
                             Hardware const&,
                             SolutionLibrarySearchType = SolutionLibrarySearchType::DEFAULT) const override
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
            return "TenSolutionSub";
        }
        std::string description() const override
        {
            return "TenSolutionSub";
        }

        TopSolutionsResult<ContractionSolution>
            findTopSolutions(ContractionProblemGemm const&,
                             Hardware const&,
                             int numSolutions) const override
        {
            const int                           n = std::min(available, numSolutions);
            SolutionVector<ContractionSolution> out;
            for(int i = 0; i < n; i++)
                out.push_back(makeSolution("sol_" + std::to_string(i), i));
            return {std::move(out), numSolutions >= available};
        }
    };
} // namespace

// The solutions and the already-returned-all bit for one problem were stored in
// two separate maps with no common lock. Same problem, two numSolutions values:
//   B (asks 20) stores its 10 solutions,
//   A (asks 3)  stores its 3 solutions,
//   A stores its bit (false),
//   B stores its bit (true).
// The entry then held 3 solutions marked already-returned-all, and a later
// request for 10 got 3 back. The pair is now one cache entry, so a reader sees
// either both values from one writer or neither.
TEST(CachingLibraryTest, ConcurrentFindTopSolutionsSameProblemKeepsSolutionsAndFlagTogether)
{
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    auto         problem = problemWithSmCountTarget(64);

    constexpr int iterations      = 4000;
    constexpr int threadsPerGroup = 2;
    constexpr int fewSolutions    = 3;
    constexpr int manySolutions   = 20;
    const size_t  all             = TenSolutionSubLibrary::available;

    std::atomic<int> splitPairs{0};
    std::atomic<int> finalMismatches{0};

    for(int it = 0; it < iterations; it++)
    {
        auto sub     = std::make_shared<TenSolutionSubLibrary>();
        auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

        std::atomic<bool>        go{false};
        std::vector<std::thread> threads;
        for(int t = 0; t < 2 * threadsPerGroup; t++)
        {
            const int numSolutions = (t % 2 == 0) ? fewSolutions : manySolutions;
            threads.emplace_back([&, numSolutions] {
                while(!go.load(std::memory_order_acquire))
                    ;
                auto r = caching->findTopSolutions(problem, device, numSolutions);
                // A result marked already-returned-all must hold everything.
                if(r.alreadyRetAll && r.solutions.size() != all)
                    splitPairs++;
            });
        }
        go.store(true, std::memory_order_release);
        for(auto& th : threads)
            th.join();

        auto again = caching->findTopSolutions(problem, device, static_cast<int>(all));
        if(again.solutions.size() != all)
            finalMismatches++;
    }

    EXPECT_EQ(splitPairs.load(), 0)
        << "a cached entry paired one request's solutions with another request's "
           "already-returned-all bit";
    EXPECT_EQ(finalMismatches.load(), 0)
        << "a later request for all solutions was answered with a truncated cached list";
}
