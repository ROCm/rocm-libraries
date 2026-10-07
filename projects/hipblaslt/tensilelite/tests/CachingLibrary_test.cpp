// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <memory>
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

    // findTopSolutions returns at most `available` solutions and counts its calls.
    class TopCountingSubLibrary : public CountingSubLibrary
    {
    public:
        int                       available = 4;
        mutable std::atomic<int>  topCalls{0};
        mutable std::atomic<bool> lastAll{false};

        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                             Hardware const&,
                                                             int numSolutions) const override
        {
            int                                 call = ++topCalls;
            SolutionVector<ContractionSolution> rv;
            for(int i = 0; i < std::min(numSolutions, available); ++i)
                rv.push_back(
                    makeSolution("top_" + std::to_string(call) + "_" + std::to_string(i), i));
            lastAll = static_cast<int>(rv.size()) < numSolutions;
            return rv;
        }

        bool lastFindTopAlreadyRetAll() const override
        {
            return lastAll;
        }
    };

    template <typename T>
    std::string keyOf(T const& value)
    {
        std::string key;
        AppendCacheKey(key, value);
        return key;
    }

    // Problems that each differ from dummyProblem() in fields that
    // Comparison<ContractionProblemGemm> compares.
    std::vector<ContractionProblemGemm> problemVariants()
    {
        std::vector<ContractionProblemGemm> v;
        auto                                add = [&](auto&& change) {
            auto problem = dummyProblem();
            change(problem);
            v.push_back(problem);
        };
        auto gemm = [](bool transA, bool transB, size_t m, size_t n, size_t k, size_t lda) {
            auto problem = ContractionProblemGemm::GEMM(
                transA, transB, m, n, k, lda, 1024, 1024, 1.0, false, 1);
            problem.setComputeInputTypeA(rocisa::DataType::Float);
            problem.setComputeInputTypeB(rocisa::DataType::Float);
            return problem;
        };
        v.push_back(gemm(false, false, 1024, 1024, 512, 1024));
        v.push_back(gemm(true, false, 1024, 1024, 1024, 1024));
        v.push_back(gemm(false, false, 1024, 1024, 1024, 2048));
        v.push_back(gemm(false, false, 512, 1024, 1024, 1024));
        add([](auto& p) { p.setHighPrecisionAccumulate(true); });
        add([](auto& p) { p.setKernelLanguage(KernelLanguage::Assembly); });
        add([](auto& p) { p.setDeterministicMode(true); });
        add([](auto& p) { p.setComputeInputTypeA(rocisa::DataType::Half); });
        add([](auto& p) { p.setComputeInputTypeB(rocisa::DataType::Half); });
        add([](auto& p) { p.setWorkspaceSize(1 << 20); });
        add([](auto& p) { p.setStridedBatched(!p.stridedBatched()); });
        add([](auto& p) { p.setGroupedGemm(true); });
        add([](auto& p) { p.setPerformanceMetric(PerformanceMetric::CUEfficiency); });
        add([](auto& p) { p.setActivationType(ActivationType::Relu); });
        add([](auto& p) { p.setActivationComputeType(rocisa::DataType::Half); });
        add([](auto& p) { p.setActivationNoGuard(true); });
        add([](auto& p) { p.setUseGradient(true); });
        add([](auto& p) { p.setUseBias(1); });
        add([](auto& p) { p.setUseE(true); });
        add([](auto& p) { p.setUseGateResidual(true); });
        add([](auto& p) { p.setUseScaleAB("Scalar"); });
        add([](auto& p) { p.setUseScaleAB("Vector"); });
        add([](auto& p) { p.setUseScaleAB("Scalar2"); });
        add([](auto& p) { p.setUseScaleCD(true); });
        add([](auto& p) { p.setUseScaleAlphaVec(1); });
        add([](auto& p) { p.setOutputAmaxD(true); });
        add([](auto& p) { p.setF32XdlMathOp(rocisa::DataType::XFloat32); });
        add([](auto& p) { p.setSwizzleTensorA(true); });
        add([](auto& p) { p.setSwizzleTensorB(true); });
        add([](auto& p) { p.setFusedGemmA2A(true); });
        add([](auto& p) { p.setFusedA2AExtent(7); });
        add([](auto& p) { p.setFusedA2AWorld(8); });
        add([](auto& p) { p.setParams().setSmCountTarget(64); });
        add([](auto& p) { p.setParams().setStreamKTileSchedulingMode(1); });
        add([](auto& p) { p.setParams().setUniformSummationOrder(true); });
        return v;
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

// Byte keys of problems are equal exactly when the problems compare equal.
TEST(CachingLibraryTest, ProblemCacheKeyEqualityMatchesComparison)
{
    auto problems = problemVariants();
    problems.push_back(dummyProblem());
    problems.push_back(dummyProblem());

    for(size_t i = 0; i + 2 < problems.size(); ++i)
        EXPECT_FALSE(problems[i] == problems.back())
            << "variant " << i << " changes no compared field";

    for(size_t i = 0; i < problems.size(); ++i)
        for(size_t j = 0; j < problems.size(); ++j)
            EXPECT_EQ(keyOf(problems[i]) == keyOf(problems[j]), problems[i] == problems[j])
                << "problems " << i << " and " << j;
}

// Byte keys of devices are equal exactly when AMDGPU::operator== holds,
// including a missing PCI chip id versus chip id 0.
TEST(CachingLibraryTest, DeviceCacheKeyEqualityMatchesComparison)
{
    std::vector<AMDGPU> devices = {makeDevice(_MI350_CHIP_ID, _SPX_CU, "a"),
                                   makeDevice(_MI350_CHIP_ID, _SPX_CU, "b"),
                                   makeDevice(_MI350_CHIP_ID, _CPX_CU, "a"),
                                   makeDevice(_MI355_CHIP_ID, _SPX_CU, "a"),
                                   makeDevice(0, _SPX_CU, "a"),
                                   AMDGPU(AMDGPU::Processor::gfx950, _SPX_CU, "a"),
                                   AMDGPU(AMDGPU::Processor::gfx942, _SPX_CU, "a")};
    for(size_t i = 0; i < devices.size(); ++i)
        for(size_t j = 0; j < devices.size(); ++j)
            EXPECT_EQ(keyOf(devices[i]) == keyOf(devices[j]), devices[i] == devices[j])
                << "devices " << i << " and " << j;
}

// A repeat request is served from the cache unless it asks for more solutions
// than were cached and the sub-library had not already returned all it has.
TEST(CachingLibraryTest, FindTopSolutionsServesRepeatsFromCache)
{
    auto sub     = std::make_shared<TopCountingSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    auto         problem = dummyProblem();

    auto first = caching->findTopSolutions(problem, device, 2);
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(caching->findTopSolutions(problem, device, 2), first);
    EXPECT_EQ(caching->findTopSolutions(problem, device, 1), first);
    EXPECT_EQ(sub->topCalls.load(), 1);
    EXPECT_FALSE(caching->lastFindTopAlreadyRetAll());

    EXPECT_EQ(caching->findTopSolutions(problem, device, 3).size(), 3u);
    EXPECT_EQ(sub->topCalls.load(), 2);

    auto all = caching->findTopSolutions(problem, device, 10);
    EXPECT_EQ(all.size(), 4u);
    EXPECT_EQ(sub->topCalls.load(), 3);
    EXPECT_TRUE(caching->lastFindTopAlreadyRetAll());
    EXPECT_EQ(caching->findTopSolutions(problem, device, 20), all);
    EXPECT_EQ(sub->topCalls.load(), 3);
    EXPECT_TRUE(caching->lastFindTopAlreadyRetAll());

    EXPECT_EQ(caching->findTopSolutions(problemWithSmCountTarget(64), device, 2).size(), 2u);
    EXPECT_EQ(
        caching->findTopSolutions(problem, makeDevice(_MI350_CHIP_ID, _CPX_CU, "cpx"), 2).size(),
        2u);
    EXPECT_EQ(sub->topCalls.load(), 5);
}

// Empty results are not cached.
TEST(CachingLibraryTest, FindTopSolutionsDoesNotCacheEmptyResults)
{
    auto sub       = std::make_shared<TopCountingSubLibrary>();
    sub->available = 0;
    auto caching   = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    auto         problem = dummyProblem();
    EXPECT_TRUE(caching->findTopSolutions(problem, device, 2).empty());
    EXPECT_TRUE(caching->findTopSolutions(problem, device, 2).empty());
    EXPECT_EQ(sub->topCalls.load(), 2);
}

namespace
{
    using Caching = CachingLibrary<ContractionProblemGemm>;

    ContractionProblemGemm problemAt(size_t i)
    {
        auto problem = dummyProblem();
        problem.setWorkspaceSize(i);
        return problem;
    }

    // findTopSolutions names its solution after the problem's workspace size.
    class KeyedTopSubLibrary : public TopCountingSubLibrary
    {
    public:
        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const& problem,
                                                             Hardware const&,
                                                             int) const override
        {
            ++topCalls;
            lastAll = true;
            return {makeSolution(std::to_string(problem.workspaceSize()), 0)};
        }
    };
} // namespace

// At TopSolutionsCapacity problems, a new problem replaces the least recently
// used one and is served from the cache afterwards.
TEST(CachingLibraryTest, FindTopSolutionsEvictsTheLeastRecentlyUsed)
{
    auto         sub     = std::make_shared<TopCountingSubLibrary>();
    auto         caching = std::make_shared<Caching>(sub);
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    auto const   cap     = Caching::TopSolutionsCapacity;
    auto         calls   = [&] { return static_cast<size_t>(sub->topCalls.load()); };

    for(size_t i = 0; i < cap; ++i)
        caching->findTopSolutions(problemAt(i), device, 1);
    ASSERT_EQ(calls(), cap);

    caching->findTopSolutions(problemAt(cap), device, 1);
    caching->findTopSolutions(problemAt(cap), device, 1);
    EXPECT_EQ(calls(), cap + 1);

    caching->findTopSolutions(problemAt(1), device, 1);
    EXPECT_EQ(calls(), cap + 1);
    caching->findTopSolutions(problemAt(0), device, 1);
    EXPECT_EQ(calls(), cap + 2);

    caching->findTopSolutions(problemAt(cap - 1), device, 1);
    caching->findTopSolutions(problemAt(cap), device, 1);
    caching->findTopSolutions(problemAt(3), device, 1);
    EXPECT_EQ(calls(), cap + 2);
    caching->findTopSolutions(problemAt(2), device, 1);
    EXPECT_EQ(calls(), cap + 3);
    caching->findTopSolutions(problemAt(4), device, 1);
    EXPECT_EQ(calls(), cap + 4);
    for(size_t i : {size_t(0), size_t(1), size_t(2), size_t(3), cap - 1, cap})
        caching->findTopSolutions(problemAt(i), device, 1);
    EXPECT_EQ(calls(), cap + 4);
}

// A hit, or a lookup that asks the sub-library for more solutions, makes the
// problem the most recently used.
TEST(CachingLibraryTest, FindTopSolutionsUseRefreshesRecency)
{
    auto         sub     = std::make_shared<TopCountingSubLibrary>();
    auto         caching = std::make_shared<Caching>(sub);
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    auto const   cap     = Caching::TopSolutionsCapacity;
    auto         calls   = [&] { return static_cast<size_t>(sub->topCalls.load()); };

    for(size_t i = 0; i < cap; ++i)
        caching->findTopSolutions(problemAt(i), device, 1);
    auto first = caching->findTopSolutions(problemAt(0), device, 1);
    caching->findTopSolutions(problemAt(1), device, 2);
    ASSERT_EQ(calls(), cap + 1);

    for(size_t i = cap; i < 2 * cap - 2; ++i)
        caching->findTopSolutions(problemAt(i), device, 1);
    ASSERT_EQ(calls(), 2 * cap - 1);

    EXPECT_EQ(caching->findTopSolutions(problemAt(0), device, 1), first);
    EXPECT_EQ(caching->findTopSolutions(problemAt(1), device, 2).size(), 2u);
    EXPECT_EQ(calls(), 2 * cap - 1);
    caching->findTopSolutions(problemAt(2), device, 1);
    EXPECT_EQ(calls(), 2 * cap);
}

// Threads cycling through more problems than the cache holds always get the
// solutions of the problem they asked for.
TEST(CachingLibraryTest, FindTopSolutionsLruIsThreadSafe)
{
    auto         sub     = std::make_shared<KeyedTopSubLibrary>();
    auto         caching = std::make_shared<Caching>(sub);
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    size_t const span    = Caching::TopSolutionsCapacity + Caching::TopSolutionsCapacity / 2;

    constexpr int            numThreads = 8;
    std::atomic<int>         wrong{0};
    std::vector<std::thread> workers;
    workers.reserve(numThreads);
    for(int t = 0; t < numThreads; ++t)
    {
        workers.emplace_back([&, t]() {
            for(size_t j = 0; j < 2 * span; ++j)
            {
                size_t i   = (j * 7 + static_cast<size_t>(t) * 977) % span;
                auto   top = caching->findTopSolutions(problemAt(i), device, 1);
                if(top.size() != 1 || top[0]->solutionName != std::to_string(i))
                    ++wrong;
            }
        });
    }
    for(auto& worker : workers)
        worker.join();

    EXPECT_EQ(wrong.load(), 0);
    for(size_t i = 0; i < 4; ++i)
        EXPECT_EQ(caching->findTopSolutions(problemAt(i), device, 1)[0]->solutionName,
                  std::to_string(i));
    int calls = sub->topCalls.load();
    for(size_t i = 0; i < 4; ++i)
        caching->findTopSolutions(problemAt(i), device, 1);
    EXPECT_EQ(sub->topCalls.load(), calls);
}
