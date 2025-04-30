/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/UserDrivenTuningParser.hpp>

#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

TEST(Cache, Simple)
{
    using namespace Tensile;

    CacheMap<int, int> cache(-1);

    EXPECT_EQ(-1, cache.find(5));

    cache.add(7, 5);
    EXPECT_EQ(7, cache.find(5));

    cache.add(9, 5);
    EXPECT_EQ(7, cache.find(5));
}

TEST(Cache, Threaded)
{
    using namespace Tensile;
    CacheMap<int, int> cache(-1);

#pragma omp parallel num_threads(32)
    {
        int seed = 0;
#ifdef _OPENMP
        seed = omp_get_thread_num();
#endif
        std::uniform_int_distribution<int> dist(0, 10);
        std::mt19937                       rng(seed);

        for(int i = 0; i < 10000; i++)
        {
            int key   = dist(rng);
            int value = key + 1;

            int lookup = cache.find(key);
            if(lookup != -1)
                EXPECT_EQ(lookup, value);

            cache.add(value, key);
            EXPECT_EQ(value, cache.find(key));
        }
    }
}

TEST(Hashing, TensorDescriptor)
{
    using namespace Tensile;

    TensorDescriptor a(DataType::Float, {15, 8, 20});
    TensorDescriptor b(DataType::Int32, {15, 8, 20});
    TensorDescriptor c(DataType::Float, {15, 8, 20}, {1, 15, 15 * 8});
    TensorDescriptor d(DataType::Float, {15, 8, 20}, {1, 17, 19 * 8});

    EXPECT_NE(std::hash<TensorDescriptor>()(a), std::hash<TensorDescriptor>()(b));
    EXPECT_EQ(std::hash<TensorDescriptor>()(a), std::hash<TensorDescriptor>()(c));

    EXPECT_NE(std::hash<TensorDescriptor>()(c), std::hash<TensorDescriptor>()(d));
}

TEST(Hashing, ContractionProblem)
{
    using namespace Tensile;

    // Test sizes
    ContractionProblem a = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);
    ContractionProblem b = ContractionProblem::GEMM(false, true, 5, 7, 9, 7, 5, 5, 3.0, false, 5);
    ContractionProblem c = ContractionProblem::GEMM(false, true, 5, 7, 9, 0, 5, 5, 3.0, false, 5);
    ContractionProblem d = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);

    EXPECT_NE(std::hash<ContractionProblem>()(a), std::hash<ContractionProblem>()(b));
    EXPECT_NE(std::hash<ContractionProblem>()(a), std::hash<ContractionProblem>()(c));

    EXPECT_EQ(std::hash<ContractionProblem>()(a), std::hash<ContractionProblem>()(d));

    // Test high precision accumulate flag
    ContractionProblem e = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);
    ContractionProblem f = e;

    e.setHighPrecisionAccumulate(false);
    f.setHighPrecisionAccumulate(true);
    EXPECT_NE(std::hash<ContractionProblem>()(e), std::hash<ContractionProblem>()(f));

    e.setHighPrecisionAccumulate(true);
    f.setHighPrecisionAccumulate(true);
    EXPECT_EQ(std::hash<ContractionProblem>()(e), std::hash<ContractionProblem>()(f));

    e.setHighPrecisionAccumulate(false);
    f.setHighPrecisionAccumulate(false);
    EXPECT_EQ(std::hash<ContractionProblem>()(e), std::hash<ContractionProblem>()(f));

    // Test kernel language flag
    ContractionProblem g = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);
    ContractionProblem h = g;
    g.setKernelLanguage(KernelLanguage::Any);
    h.setKernelLanguage(KernelLanguage::Source);
    EXPECT_NE(std::hash<ContractionProblem>()(g), std::hash<ContractionProblem>()(h));

    g.setKernelLanguage(KernelLanguage::Any);
    h.setKernelLanguage(KernelLanguage::Assembly);
    EXPECT_NE(std::hash<ContractionProblem>()(g), std::hash<ContractionProblem>()(h));

    g.setKernelLanguage(KernelLanguage::Source);
    h.setKernelLanguage(KernelLanguage::Assembly);
    EXPECT_NE(std::hash<ContractionProblem>()(g), std::hash<ContractionProblem>()(h));

    g.setKernelLanguage(KernelLanguage::Any);
    h.setKernelLanguage(KernelLanguage::Any);
    EXPECT_EQ(std::hash<ContractionProblem>()(g), std::hash<ContractionProblem>()(h));

    g.setKernelLanguage(KernelLanguage::Source);
    h.setKernelLanguage(KernelLanguage::Source);
    EXPECT_EQ(std::hash<ContractionProblem>()(g), std::hash<ContractionProblem>()(h));

    g.setKernelLanguage(KernelLanguage::Assembly);
    h.setKernelLanguage(KernelLanguage::Assembly);
    EXPECT_EQ(std::hash<ContractionProblem>()(g), std::hash<ContractionProblem>()(h));

    // Test deterministic mode flag
    ContractionProblem i = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);
    ContractionProblem j = i;

    i.setDeterministicMode(true);
    j.setDeterministicMode(false);
    EXPECT_NE(std::hash<ContractionProblem>()(i), std::hash<ContractionProblem>()(j));

    i.setDeterministicMode(true);
    j.setDeterministicMode(true);
    EXPECT_EQ(std::hash<ContractionProblem>()(i), std::hash<ContractionProblem>()(j));

    i.setDeterministicMode(false);
    j.setDeterministicMode(false);
    EXPECT_EQ(std::hash<ContractionProblem>()(i), std::hash<ContractionProblem>()(j));

    // Test arithmetic unit flag
    ContractionProblem k = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);
    ContractionProblem l = k;

    k.setArithmeticUnit(ArithmeticUnit::Any);
    l.setArithmeticUnit(ArithmeticUnit::MFMA);
    EXPECT_NE(std::hash<ContractionProblem>()(k), std::hash<ContractionProblem>()(l));

    k.setArithmeticUnit(ArithmeticUnit::Any);
    l.setArithmeticUnit(ArithmeticUnit::VALU);
    EXPECT_NE(std::hash<ContractionProblem>()(k), std::hash<ContractionProblem>()(l));

    k.setArithmeticUnit(ArithmeticUnit::VALU);
    l.setArithmeticUnit(ArithmeticUnit::MFMA);
    EXPECT_NE(std::hash<ContractionProblem>()(k), std::hash<ContractionProblem>()(l));

    k.setArithmeticUnit(ArithmeticUnit::Any);
    l.setArithmeticUnit(ArithmeticUnit::Any);
    EXPECT_EQ(std::hash<ContractionProblem>()(k), std::hash<ContractionProblem>()(l));

    k.setArithmeticUnit(ArithmeticUnit::VALU);
    l.setArithmeticUnit(ArithmeticUnit::VALU);
    EXPECT_EQ(std::hash<ContractionProblem>()(k), std::hash<ContractionProblem>()(l));

    k.setArithmeticUnit(ArithmeticUnit::MFMA);
    l.setArithmeticUnit(ArithmeticUnit::MFMA);
    EXPECT_EQ(std::hash<ContractionProblem>()(k), std::hash<ContractionProblem>()(l));

    // Test performance metric flag
    ContractionProblem m = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);
    ContractionProblem n = m;

    m.setPerformanceMetric(PerformanceMetric::DeviceEfficiency);
    n.setPerformanceMetric(PerformanceMetric::CUEfficiency);
    EXPECT_NE(std::hash<ContractionProblem>()(m), std::hash<ContractionProblem>()(n));

    m.setPerformanceMetric(PerformanceMetric::Auto);
    n.setPerformanceMetric(PerformanceMetric::CUEfficiency);
    EXPECT_NE(std::hash<ContractionProblem>()(m), std::hash<ContractionProblem>()(n));

    m.setPerformanceMetric(PerformanceMetric::Auto);
    n.setPerformanceMetric(PerformanceMetric::DeviceEfficiency);
    EXPECT_NE(std::hash<ContractionProblem>()(m), std::hash<ContractionProblem>()(n));

    m.setPerformanceMetric(PerformanceMetric::Auto);
    n.setPerformanceMetric(PerformanceMetric::Auto);
    EXPECT_EQ(std::hash<ContractionProblem>()(m), std::hash<ContractionProblem>()(n));

    m.setPerformanceMetric(PerformanceMetric::DeviceEfficiency);
    n.setPerformanceMetric(PerformanceMetric::DeviceEfficiency);
    EXPECT_EQ(std::hash<ContractionProblem>()(m), std::hash<ContractionProblem>()(n));

    m.setPerformanceMetric(PerformanceMetric::CUEfficiency);
    n.setPerformanceMetric(PerformanceMetric::CUEfficiency);
    EXPECT_EQ(std::hash<ContractionProblem>()(m), std::hash<ContractionProblem>()(n));
}

TEST(Hashing, AMDGPU)
{
    using namespace Tensile;

    std::vector<AMDGPU::Processor> processors{AMDGPU::Processor::gfx803,
                                              AMDGPU::Processor::gfx900,
                                              AMDGPU::Processor::gfx906,
                                              AMDGPU::Processor::gfx908,
                                              AMDGPU::Processor::gfx90a};

    std::vector<int> counts{16, 20, 40, 56, 60, 64};

    // There aren't that many possible combinations here so it's reasonable to
    // have no hash collisions.
    for(AMDGPU::Processor p1 : processors)
        for(AMDGPU::Processor p2 : processors)
            for(int c1 : counts)
                for(int c2 : counts)
                {
                    AMDGPU g1(p1, c1, 0, "g1");
                    AMDGPU g2(p2, c2, 0, "g2");

                    if(p1 != p2 || c1 != c2)
                    {
                        EXPECT_NE(std::hash<AMDGPU>()(g1), std::hash<AMDGPU>()(g2))
                            << g1 << "/" << g2;
                    }
                    else
                    {
                        EXPECT_EQ(std::hash<AMDGPU>()(g1), std::hash<AMDGPU>()(g2))
                            << g1 << "/" << g2;
                    }
                }
}

TEST(Hashing, Tuple)
{
    using namespace Tensile;

    using TwoInts = std::tuple<int, int>;

    TwoInts tup(4, 5);

    EXPECT_EQ(std::hash<TwoInts>()(tup), hash_combine(5, 4));
}

TEST(Hashing, Tuple2)
{
    using namespace Tensile;

    using TwoInts = std::tuple<ContractionProblem, AMDGPU>;

    TwoInts tup;

    size_t h = std::hash<TwoInts>()(tup);
    if(h) // Use the code to quiet the compiler.
        return;
}

TEST(CachingLibrary, Simple)
{
    using namespace Tensile;

    auto Solution0 = std::make_shared<ContractionSolution>();
    auto Solution1 = std::make_shared<ContractionSolution>();
    auto Solution2 = std::make_shared<ContractionSolution>();
    auto Solution3 = std::make_shared<ContractionSolution>();

    Solution0->index = 0;
    Solution1->index = 1;
    Solution2->index = 2;
    Solution3->index = 3;

    SolutionMap<ContractionSolution> map(
        {{0, Solution0}, {1, Solution1}, {2, Solution2}, {3, Solution3}});

    auto Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
    auto Library1 = std::make_shared<SingleContractionLibrary>(Solution1);
    auto Library2 = std::make_shared<SingleContractionLibrary>(Solution2);
    auto Library3 = std::make_shared<SingleContractionLibrary>(Solution3);

    AMDGPU gpu;

    auto Problem0 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
    auto Problem1 = ContractionProblem::GEMM(false, false, 6, 6, 6, 6, 6, 6, 1.2, false, 1);
    auto Problem2 = ContractionProblem::GEMM(true, false, 14, 4, 4, 4, 4, 4, 1.2, false, 1);
    auto Problem3 = ContractionProblem::GEMM(true, true, 24, 4, 4, 4, 4, 4, 1.2, false, 1);

    using Key = std::array<int64_t, 4>;
    using Table
        = Matching::DistanceMatchingTable<Key,
                                          ContractionProblem,
                                          std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                                          std::shared_ptr<ContractionSolution>,
                                          Matching::EuclideanDistance<Key>>;
    using Properties = std::vector<std::shared_ptr<Property<ContractionProblem>>>;

    Properties properties;

    {
        auto freeSizeA   = std::make_shared<Contraction::FreeSizeA>();
        freeSizeA->index = 0;
        properties.push_back(freeSizeA);
        auto freeSizeB   = std::make_shared<Contraction::FreeSizeB>();
        freeSizeB->index = 0;
        properties.push_back(freeSizeB);
        auto batchSize   = std::make_shared<Contraction::BatchSize>();
        batchSize->index = 0;
        properties.push_back(batchSize);
        auto boundSize   = std::make_shared<Contraction::BoundSize>();
        boundSize->index = 0;
        properties.push_back(boundSize);
    }

    std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

    using Entry
        = Matching::MatchingTableEntry<Key, std::shared_ptr<SolutionLibrary<ContractionProblem>>>;

    std::vector<Entry> table;

    {
        Entry map0{{4, 4, 1, 4}, Library0, 1.0};
        table.push_back(map0);
        Entry map1{{6, 6, 1, 6}, Library1, 1.0};
        table.push_back(map1);
        Entry map2{{14, 4, 1, 4}, Library2, 1.0};
        table.push_back(map2);
        Entry map3{{24, 4, 1, 4}, Library3, 1.0};
        table.push_back(map3);
    }

    matchingTable->table = table;

    auto subLib = std::make_shared<ProblemMatchingLibrary<ContractionProblem>>();

    subLib->table = matchingTable;

    CachingLibrary<ContractionProblem> lib(subLib);

    EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem0, gpu));
    auto theSolution0 = lib.findBestSolution(Problem0, gpu);
    EXPECT_EQ(theSolution0, Solution0);
    auto theSolution0_cached = lib.findSolutionInCache(Problem0, gpu);
    EXPECT_EQ(theSolution0, theSolution0_cached);

    EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem1, gpu));
    auto theSolution1 = lib.findBestSolution(Problem1, gpu);
    EXPECT_EQ(theSolution1, Solution1);
    auto theSolution1_cached = lib.findSolutionInCache(Problem1, gpu);
    EXPECT_EQ(theSolution1, theSolution1_cached);

    EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem2, gpu));
    auto theSolution2 = lib.findBestSolution(Problem2, gpu);
    EXPECT_EQ(theSolution2, Solution2);
    auto theSolution2_cached = lib.findSolutionInCache(Problem2, gpu);
    EXPECT_EQ(theSolution2, theSolution2_cached);

    EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem3, gpu));
    auto theSolution3 = lib.findBestSolution(Problem3, gpu);
    EXPECT_EQ(theSolution3, Solution3);
    auto theSolution3_cached = lib.findSolutionInCache(Problem3, gpu);
    EXPECT_EQ(theSolution3, theSolution3_cached);
}

TEST(CachingLibrary, FlagsDiff)
{
    // This test is to ensure that the CachingLibrary differentiates between
    // problem 'flags', such as high precision accumulate, kernel language,
    // deterministic mode and arithmetic unit support.
    using namespace Tensile;

    using Key = std::array<int64_t, 4>;
    using Table
        = Matching::DistanceMatchingTable<Key,
                                          ContractionProblem,
                                          std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                                          std::shared_ptr<ContractionSolution>,
                                          Matching::EuclideanDistance<Key>>;
    using Properties = std::vector<std::shared_ptr<Property<ContractionProblem>>>;

    using Entry
        = Matching::MatchingTableEntry<Key, std::shared_ptr<SolutionLibrary<ContractionProblem>>>;

    // Default GPU device
    AMDGPU gpu;

    // Set up default distance matching properties
    Properties properties;

    {
        auto freeSizeA   = std::make_shared<Contraction::FreeSizeA>();
        freeSizeA->index = 0;
        properties.push_back(freeSizeA);
        auto freeSizeB   = std::make_shared<Contraction::FreeSizeB>();
        freeSizeB->index = 0;
        properties.push_back(freeSizeB);
        auto batchSize   = std::make_shared<Contraction::BatchSize>();
        batchSize->index = 0;
        properties.push_back(batchSize);
        auto boundSize   = std::make_shared<Contraction::BoundSize>();
        boundSize->index = 0;
        properties.push_back(boundSize);
    }

    // Test high precision accumulate caching
    {
        auto Solution0 = std::make_shared<ContractionSolution>();
        auto Solution1 = std::make_shared<ContractionSolution>();

        Solution0->problemPredicate
            = std::make_shared<Predicates::Contraction::HighPrecisionAccumulateEqual>(true);
        Solution1->problemPredicate
            = std::make_shared<Predicates::Contraction::HighPrecisionAccumulateEqual>(false);

        Solution0->index = 0;
        Solution1->index = 1;

        SolutionMap<ContractionSolution> map(
            {{Solution0->index, Solution0}, {Solution1->index, Solution1}});

        auto Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
        auto Library1 = std::make_shared<SingleContractionLibrary>(Solution1);

        auto Problem0 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem1 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);

        Problem0.setHighPrecisionAccumulate(true);
        Problem1.setHighPrecisionAccumulate(false);

        std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

        std::vector<Entry> table;
        {
            Entry map0{{4, 4, 1, 4}, Library0, 1.0};
            Entry map1{{4, 4, 1, 4}, Library1, 1.0};
            table.push_back(map0);
            table.push_back(map1);
        }

        matchingTable->table = table;

        auto subLib = std::make_shared<ProblemMatchingLibrary<ContractionProblem>>();

        subLib->table = matchingTable;

        CachingLibrary<ContractionProblem> lib(subLib);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem0, gpu));
        auto theSolution0 = lib.findBestSolution(Problem0, gpu);
        EXPECT_EQ(theSolution0, Solution0);
        auto theSolution0_cached = lib.findSolutionInCache(Problem0, gpu);
        EXPECT_EQ(theSolution0, theSolution0_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem1, gpu));
        auto theSolution1 = lib.findBestSolution(Problem1, gpu);
        EXPECT_EQ(theSolution1, Solution1);
        auto theSolution1_cached = lib.findSolutionInCache(Problem1, gpu);
        EXPECT_EQ(theSolution1, theSolution1_cached);
    }

    // Test kernel language caching
    {
        auto Solution0 = std::make_shared<ContractionSolution>();
        auto Solution1 = std::make_shared<ContractionSolution>();
        auto Solution2 = std::make_shared<ContractionSolution>();

        Solution0->problemPredicate
            = std::make_shared<Predicates::Contraction::KernelLanguageCompatible>(
                KernelLanguage::Assembly);
        Solution1->problemPredicate
            = std::make_shared<Predicates::Contraction::KernelLanguageCompatible>(
                KernelLanguage::Source);
        Solution2->problemPredicate
            = std::make_shared<Predicates::Contraction::KernelLanguageCompatible>(
                KernelLanguage::Any);

        Solution0->index = 0;
        Solution1->index = 1;
        Solution2->index = 2;

        SolutionMap<ContractionSolution> map({{Solution0->index, Solution0},
                                              {Solution1->index, Solution1},
                                              {Solution2->index, Solution2}});

        auto Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
        auto Library1 = std::make_shared<SingleContractionLibrary>(Solution1);
        auto Library2 = std::make_shared<SingleContractionLibrary>(Solution2);

        auto Problem0 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem1 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem2 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);

        Problem0.setKernelLanguage(KernelLanguage::Assembly);
        Problem1.setKernelLanguage(KernelLanguage::Source);
        Problem2.setKernelLanguage(KernelLanguage::Any);

        std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

        std::vector<Entry> table;
        {
            Entry map0{{4, 4, 1, 4}, Library0, 1.0};
            Entry map1{{4, 4, 1, 4}, Library1, 1.0};
            Entry map2{{4, 4, 1, 4}, Library2, 2.0};
            table.push_back(map0);
            table.push_back(map1);
            table.push_back(map2);
        }

        matchingTable->table = table;

        auto subLib = std::make_shared<ProblemMatchingLibrary<ContractionProblem>>();

        subLib->table = matchingTable;

        CachingLibrary<ContractionProblem> lib(subLib);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem0, gpu));
        auto theSolution0 = lib.findBestSolution(Problem0, gpu);
        EXPECT_EQ(theSolution0, Solution0);
        auto theSolution0_cached = lib.findSolutionInCache(Problem0, gpu);
        EXPECT_EQ(theSolution0, theSolution0_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem1, gpu));
        auto theSolution1 = lib.findBestSolution(Problem1, gpu);
        EXPECT_EQ(theSolution1, Solution1);
        auto theSolution1_cached = lib.findSolutionInCache(Problem1, gpu);
        EXPECT_EQ(theSolution1, theSolution1_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem2, gpu));
        auto theSolution2 = lib.findBestSolution(Problem2, gpu);
        EXPECT_EQ(theSolution2, Solution2);
        auto theSolution2_cached = lib.findSolutionInCache(Problem2, gpu);
        EXPECT_EQ(theSolution2, theSolution2_cached);
    }

    // Test deterministic mode caching
    {
        auto Solution0 = std::make_shared<ContractionSolution>();
        auto Solution1 = std::make_shared<ContractionSolution>();

        Solution0->problemPredicate
            = std::make_shared<Predicates::Contraction::DeterministicModeEqual>(true);
        Solution1->problemPredicate
            = std::make_shared<Predicates::Contraction::DeterministicModeEqual>(false);

        Solution0->index = 0;
        Solution1->index = 1;

        SolutionMap<ContractionSolution> map(
            {{Solution0->index, Solution0}, {Solution1->index, Solution1}});

        auto Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
        auto Library1 = std::make_shared<SingleContractionLibrary>(Solution1);

        auto Problem0 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem1 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);

        Problem0.setDeterministicMode(true);
        Problem1.setDeterministicMode(false);

        std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

        std::vector<Entry> table;
        {
            Entry map0{{4, 4, 1, 4}, Library0, 1.0};
            Entry map1{{4, 4, 1, 4}, Library1, 1.0};
            table.push_back(map0);
            table.push_back(map1);
        }

        matchingTable->table = table;

        auto subLib = std::make_shared<ProblemMatchingLibrary<ContractionProblem>>();

        subLib->table = matchingTable;

        CachingLibrary<ContractionProblem> lib(subLib);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem0, gpu));
        auto theSolution0 = lib.findBestSolution(Problem0, gpu);
        EXPECT_EQ(theSolution0, Solution0);
        auto theSolution0_cached = lib.findSolutionInCache(Problem0, gpu);
        EXPECT_EQ(theSolution0, theSolution0_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem1, gpu));
        auto theSolution1 = lib.findBestSolution(Problem1, gpu);
        EXPECT_EQ(theSolution1, Solution1);
        auto theSolution1_cached = lib.findSolutionInCache(Problem1, gpu);
        EXPECT_EQ(theSolution1, theSolution1_cached);
    }

    // Test arithmetic unit caching
    {
        auto Solution0 = std::make_shared<ContractionSolution>();
        auto Solution1 = std::make_shared<ContractionSolution>();
        auto Solution2 = std::make_shared<ContractionSolution>();

        Solution0->problemPredicate
            = std::make_shared<Predicates::Contraction::ArithmeticUnitCompatible>(
                ArithmeticUnit::MFMA);
        Solution1->problemPredicate
            = std::make_shared<Predicates::Contraction::ArithmeticUnitCompatible>(
                ArithmeticUnit::VALU);
        Solution2->problemPredicate
            = std::make_shared<Predicates::Contraction::ArithmeticUnitCompatible>(
                ArithmeticUnit::Any);

        Solution0->index = 0;
        Solution1->index = 1;
        Solution2->index = 2;

        SolutionMap<ContractionSolution> map({{Solution0->index, Solution0},
                                              {Solution1->index, Solution1},
                                              {Solution2->index, Solution2}});

        auto Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
        auto Library1 = std::make_shared<SingleContractionLibrary>(Solution1);
        auto Library2 = std::make_shared<SingleContractionLibrary>(Solution2);

        auto Problem0 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem1 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem2 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);

        Problem0.setArithmeticUnit(ArithmeticUnit::MFMA);
        Problem1.setArithmeticUnit(ArithmeticUnit::VALU);
        Problem2.setArithmeticUnit(ArithmeticUnit::Any);

        std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

        std::vector<Entry> table;
        {
            Entry map0{{4, 4, 1, 4}, Library0, 1.0};
            Entry map1{{4, 4, 1, 4}, Library1, 1.0};
            Entry map2{{4, 4, 1, 4}, Library2, 2.0};
            table.push_back(map0);
            table.push_back(map1);
            table.push_back(map2);
        }

        matchingTable->table = table;

        auto subLib = std::make_shared<ProblemMatchingLibrary<ContractionProblem>>();

        subLib->table = matchingTable;

        CachingLibrary<ContractionProblem> lib(subLib);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem0, gpu));
        auto theSolution0 = lib.findBestSolution(Problem0, gpu);
        EXPECT_EQ(theSolution0, Solution0);
        auto theSolution0_cached = lib.findSolutionInCache(Problem0, gpu);
        EXPECT_EQ(theSolution0, theSolution0_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem1, gpu));
        auto theSolution1 = lib.findBestSolution(Problem1, gpu);
        EXPECT_EQ(theSolution1, Solution1);
        auto theSolution1_cached = lib.findSolutionInCache(Problem1, gpu);
        EXPECT_EQ(theSolution1, theSolution1_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem2, gpu));
        auto theSolution2 = lib.findBestSolution(Problem2, gpu);
        EXPECT_EQ(theSolution2, Solution2);
        auto theSolution2_cached = lib.findSolutionInCache(Problem2, gpu);
        EXPECT_EQ(theSolution2, theSolution2_cached);
    }

    // Random combinations, realistic
    {
        using And = Predicates::And<ContractionProblem>;
        using PredicateList
            = std::vector<std::shared_ptr<Predicates::Predicate<ContractionProblem>>>;

        auto pred_HPA_true
            = std::make_shared<Predicates::Contraction::HighPrecisionAccumulateEqual>(true);
        auto pred_HPA_false
            = std::make_shared<Predicates::Contraction::HighPrecisionAccumulateEqual>(false);

        auto pred_KL_asm = std::make_shared<Predicates::Contraction::KernelLanguageCompatible>(
            KernelLanguage::Assembly);
        auto pred_KL_src = std::make_shared<Predicates::Contraction::KernelLanguageCompatible>(
            KernelLanguage::Source);

        auto pred_DM_false
            = std::make_shared<Predicates::Contraction::DeterministicModeEqual>(false);

        auto pred_AUC_mfma = std::make_shared<Predicates::Contraction::ArithmeticUnitCompatible>(
            ArithmeticUnit::MFMA);
        auto pred_AUC_valu = std::make_shared<Predicates::Contraction::ArithmeticUnitCompatible>(
            ArithmeticUnit::VALU);

        auto Solution0 = std::make_shared<ContractionSolution>();
        auto Solution1 = std::make_shared<ContractionSolution>();
        auto Solution2 = std::make_shared<ContractionSolution>();
        auto Solution3 = std::make_shared<ContractionSolution>();
        auto Solution4 = std::make_shared<ContractionSolution>();
        auto Solution5 = std::make_shared<ContractionSolution>();
        auto Solution6 = std::make_shared<ContractionSolution>();
        auto Solution7 = std::make_shared<ContractionSolution>();

        Solution0->problemPredicate
            = std::make_shared<And>(PredicateList({pred_HPA_true, pred_KL_asm, pred_AUC_valu}));
        Solution1->problemPredicate
            = std::make_shared<And>(PredicateList({pred_HPA_true, pred_KL_asm, pred_AUC_mfma}));
        Solution2->problemPredicate
            = std::make_shared<And>(PredicateList({pred_HPA_true, pred_KL_src, pred_AUC_valu}));
        Solution3->problemPredicate
            = std::make_shared<And>(PredicateList({pred_HPA_true, pred_KL_src, pred_AUC_mfma}));
        Solution4->problemPredicate = std::make_shared<And>(
            PredicateList({pred_HPA_true, pred_KL_src, pred_AUC_valu, pred_DM_false}));
        Solution5->problemPredicate = std::make_shared<And>(
            PredicateList({pred_HPA_true, pred_KL_src, pred_AUC_mfma, pred_DM_false}));
        Solution6->problemPredicate
            = std::make_shared<And>(PredicateList({pred_HPA_true, pred_AUC_valu}));
        Solution7->problemPredicate
            = std::make_shared<And>(PredicateList({pred_HPA_false, pred_AUC_valu}));

        Solution0->index = 0;
        Solution1->index = 1;
        Solution2->index = 2;
        Solution3->index = 3;
        Solution4->index = 4;
        Solution5->index = 5;
        Solution6->index = 6;
        Solution7->index = 7;

        SolutionMap<ContractionSolution> map({{Solution0->index, Solution0},
                                              {Solution1->index, Solution1},
                                              {Solution2->index, Solution2},
                                              {Solution3->index, Solution3},
                                              {Solution4->index, Solution4},
                                              {Solution5->index, Solution5},
                                              {Solution6->index, Solution6},
                                              {Solution7->index, Solution7}});

        auto Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
        auto Library1 = std::make_shared<SingleContractionLibrary>(Solution1);
        auto Library2 = std::make_shared<SingleContractionLibrary>(Solution2);
        auto Library3 = std::make_shared<SingleContractionLibrary>(Solution3);
        auto Library4 = std::make_shared<SingleContractionLibrary>(Solution4);
        auto Library5 = std::make_shared<SingleContractionLibrary>(Solution5);
        auto Library6 = std::make_shared<SingleContractionLibrary>(Solution6);
        auto Library7 = std::make_shared<SingleContractionLibrary>(Solution7);

        auto Problem0 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem1 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem2 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem3 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem4 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem5 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem6 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
        auto Problem7 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);

        Problem0.setHighPrecisionAccumulate(true);
        Problem0.setKernelLanguage(KernelLanguage::Assembly);
        Problem0.setArithmeticUnit(ArithmeticUnit::VALU);

        Problem1.setHighPrecisionAccumulate(true);
        Problem1.setKernelLanguage(KernelLanguage::Assembly);
        Problem1.setArithmeticUnit(ArithmeticUnit::MFMA);

        Problem2.setHighPrecisionAccumulate(true);
        Problem2.setKernelLanguage(KernelLanguage::Source);
        Problem2.setArithmeticUnit(ArithmeticUnit::VALU);
        Problem2.setDeterministicMode(true);

        Problem3.setHighPrecisionAccumulate(true);
        Problem3.setKernelLanguage(KernelLanguage::Source);
        Problem3.setArithmeticUnit(ArithmeticUnit::MFMA);
        Problem3.setDeterministicMode(true);

        Problem4.setHighPrecisionAccumulate(true);
        Problem4.setKernelLanguage(KernelLanguage::Source);
        Problem4.setArithmeticUnit(ArithmeticUnit::VALU);

        Problem5.setHighPrecisionAccumulate(true);
        Problem5.setKernelLanguage(KernelLanguage::Source);
        Problem5.setArithmeticUnit(ArithmeticUnit::MFMA);

        Problem6.setHighPrecisionAccumulate(true);
        Problem6.setArithmeticUnit(ArithmeticUnit::VALU);

        Problem7.setHighPrecisionAccumulate(false);
        Problem7.setArithmeticUnit(ArithmeticUnit::VALU);

        std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

        std::vector<Entry> table;
        {
            Entry map0{{4, 4, 1, 4}, Library0, 6.0};
            Entry map1{{4, 4, 1, 4}, Library1, 1.0};
            Entry map2{{4, 4, 1, 4}, Library2, 2.0};
            Entry map3{{4, 4, 1, 4}, Library3, 3.0};
            Entry map4{{4, 4, 1, 4}, Library4, 4.0};
            Entry map5{{4, 4, 1, 4}, Library5, 5.0};
            Entry map6{{4, 4, 1, 4}, Library6, 1.0};
            Entry map7{{4, 4, 1, 4}, Library7, 2.0};
            table.push_back(map0);
            table.push_back(map1);
            table.push_back(map2);
            table.push_back(map3);
            table.push_back(map4);
            table.push_back(map5);
            table.push_back(map6);
            table.push_back(map7);
        }

        matchingTable->table = table;

        auto subLib = std::make_shared<ProblemMatchingLibrary<ContractionProblem>>();

        subLib->table = matchingTable;

        CachingLibrary<ContractionProblem> lib(subLib);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem0, gpu));
        auto theSolution0 = lib.findBestSolution(Problem0, gpu);
        EXPECT_EQ(theSolution0, Solution0);
        auto theSolution0_cached = lib.findSolutionInCache(Problem0, gpu);
        EXPECT_EQ(theSolution0, theSolution0_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem1, gpu));
        auto theSolution1 = lib.findBestSolution(Problem1, gpu);
        EXPECT_EQ(theSolution1, Solution1);
        auto theSolution1_cached = lib.findSolutionInCache(Problem1, gpu);
        EXPECT_EQ(theSolution1, theSolution1_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem2, gpu));
        auto theSolution2 = lib.findBestSolution(Problem2, gpu);
        EXPECT_EQ(theSolution2, Solution2);
        auto theSolution2_cached = lib.findSolutionInCache(Problem2, gpu);
        EXPECT_EQ(theSolution2, theSolution2_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem3, gpu));
        auto theSolution3 = lib.findBestSolution(Problem3, gpu);
        EXPECT_EQ(theSolution3, Solution3);
        auto theSolution3_cached = lib.findSolutionInCache(Problem3, gpu);
        EXPECT_EQ(theSolution3, theSolution3_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem4, gpu));
        auto theSolution4 = lib.findBestSolution(Problem4, gpu);
        EXPECT_EQ(theSolution4, Solution4);
        auto theSolution4_cached = lib.findSolutionInCache(Problem4, gpu);
        EXPECT_EQ(theSolution4, theSolution4_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem5, gpu));
        auto theSolution5 = lib.findBestSolution(Problem5, gpu);
        EXPECT_EQ(theSolution5, Solution5);
        auto theSolution5_cached = lib.findSolutionInCache(Problem5, gpu);
        EXPECT_EQ(theSolution5, theSolution5_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem6, gpu));
        auto theSolution6 = lib.findBestSolution(Problem6, gpu);
        EXPECT_EQ(theSolution6, Solution0); // Best match
        auto theSolution6_cached = lib.findSolutionInCache(Problem6, gpu);
        EXPECT_EQ(theSolution6, theSolution6_cached);

        EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem7, gpu));
        auto theSolution7 = lib.findBestSolution(Problem7, gpu);
        EXPECT_EQ(theSolution7, Solution7);
        auto theSolution7_cached = lib.findSolutionInCache(Problem7, gpu);
        EXPECT_EQ(theSolution7, theSolution7_cached);
    }
}

TEST(CachingLibrary, Insert)
{
    using namespace Tensile;

    auto SolutionDefault  = std::make_shared<ContractionSolution>();
    auto SolutionOverride = std::make_shared<ContractionSolution>();

    SolutionDefault->index  = 0;
    SolutionOverride->index = 1;

    SolutionMap<ContractionSolution> map({{0, SolutionDefault}, {1, SolutionOverride}});

    auto LibraryDefault  = std::make_shared<SingleContractionLibrary>(SolutionDefault);
    auto LibraryOverride = std::make_shared<SingleContractionLibrary>(SolutionOverride);

    AMDGPU gpu;

    auto Problem0 = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
    auto Problem1 = ContractionProblem::GEMM(false, false, 5, 5, 5, 5, 5, 5, 1.2, false, 1);
    auto Problem2 = ContractionProblem::GEMM(true, false, 6, 6, 6, 6, 6, 6, 1.2, false, 1);
    auto Problem3 = ContractionProblem::GEMM(true, true, 7, 7, 7, 7, 7, 7, 1.2, false, 1);

    using Key = std::array<int64_t, 4>;
    using Table
        = Matching::DistanceMatchingTable<Key,
                                          ContractionProblem,
                                          std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                                          std::shared_ptr<ContractionSolution>,
                                          Matching::EuclideanDistance<Key>>;
    using Properties = std::vector<std::shared_ptr<Property<ContractionProblem>>>;

    Properties properties;

    {
        auto freeSizeA   = std::make_shared<Contraction::FreeSizeA>();
        freeSizeA->index = 0;
        properties.push_back(freeSizeA);
        auto freeSizeB   = std::make_shared<Contraction::FreeSizeB>();
        freeSizeB->index = 0;
        properties.push_back(freeSizeB);
        auto batchSize   = std::make_shared<Contraction::BatchSize>();
        batchSize->index = 0;
        properties.push_back(batchSize);
        auto boundSize   = std::make_shared<Contraction::BoundSize>();
        boundSize->index = 0;
        properties.push_back(boundSize);
    }

    std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

    using Entry
        = Matching::MatchingTableEntry<Key, std::shared_ptr<SolutionLibrary<ContractionProblem>>>;

    std::vector<Entry> table;

    {
        Entry map0{{4, 4, 1, 4}, LibraryDefault, 1.0};
        table.push_back(map0);
        Entry map1{{1000, 1000, 1, 1000}, LibraryOverride, 1.0};
        table.push_back(map1);
    }

    matchingTable->table = table;

    auto subLib = std::make_shared<ProblemMatchingLibrary<ContractionProblem>>();

    subLib->table = matchingTable;

    CachingLibrary<ContractionProblem> lib(subLib);

    // Add before solution cached
    EXPECT_TRUE(lib.addToOverride(Problem1, gpu, SolutionOverride));

    EXPECT_EQ(lib.findBestSolution(Problem0, gpu), SolutionDefault);
    EXPECT_EQ(lib.findBestSolution(Problem1, gpu), SolutionOverride);
    EXPECT_EQ(lib.findBestSolution(Problem2, gpu), SolutionDefault);
    EXPECT_EQ(lib.findBestSolution(Problem3, gpu), SolutionDefault);

    // Add after solution cached
    EXPECT_TRUE(lib.addToOverride(Problem3, gpu, SolutionOverride));
    EXPECT_FALSE(lib.addToOverride(Problem3, gpu, nullptr));

    EXPECT_EQ(lib.findBestSolution(Problem0, gpu), SolutionDefault);
    EXPECT_EQ(lib.findBestSolution(Problem1, gpu), SolutionOverride);
    EXPECT_EQ(lib.findBestSolution(Problem2, gpu), SolutionDefault);
    EXPECT_EQ(lib.findBestSolution(Problem3, gpu), SolutionOverride);
}

TEST(CachingLibrary, Parsing)
{
    using namespace Tensile;

    // Non-strided
    std::vector<std::string> entries0{"T",
                                      "N",
                                      "2304",
                                      "256",
                                      "1",
                                      "1729",
                                      "1",
                                      "1",
                                      "1729",
                                      "1729",
                                      "2304",
                                      "f16_r",
                                      "f16_r",
                                      "f32_r",
                                      "752"};

    auto probSol = problemFromEntries<ContractionProblem>(entries0);
    EXPECT_TRUE(probSol.first.transA());
    EXPECT_FALSE(probSol.first.transB());

    EXPECT_EQ(probSol.first.m(), 2304);
    EXPECT_EQ(probSol.first.n(), 256);
    EXPECT_EQ(probSol.first.batchSize(), 1);
    EXPECT_EQ(probSol.first.k(), 1729);

    EXPECT_EQ(probSol.first.beta(), 1.0);

    EXPECT_EQ(probSol.first.inputType(), DataType::Half);
    EXPECT_EQ(probSol.first.outputType(), DataType::Half);
    EXPECT_TRUE(probSol.first.HPA());

    EXPECT_EQ(probSol.second, 752);

    // Strided
    std::vector<std::string> entries1{"N",
                                      "T",
                                      "104",
                                      "104",
                                      "1024",
                                      "64",
                                      "1",
                                      "0",
                                      "104",
                                      "64",
                                      "104",
                                      "6656",
                                      "6656",
                                      "10816",
                                      "f32_r",
                                      "f32_r",
                                      "f32_r",
                                      "3976"};

    probSol = problemFromEntries<ContractionProblem>(entries1);
    EXPECT_FALSE(probSol.first.transA());
    EXPECT_TRUE(probSol.first.transB());

    EXPECT_EQ(probSol.first.m(), 104);
    EXPECT_EQ(probSol.first.n(), 104);
    EXPECT_EQ(probSol.first.batchSize(), 1024);
    EXPECT_EQ(probSol.first.k(), 64);

    EXPECT_EQ(probSol.first.beta(), 0.0);

    EXPECT_EQ(probSol.first.inputType(), DataType::Float);
    EXPECT_EQ(probSol.first.outputType(), DataType::Float);
    EXPECT_FALSE(probSol.first.HPA());

    EXPECT_EQ(probSol.second, 3976);

    // Bad args
    std::vector<std::string> entries2{"N", "T", "104"}; // Too short
    probSol = problemFromEntries<ContractionProblem>(entries2);
    EXPECT_EQ(probSol.second, -1);

    std::vector<std::string> entries3{"T",
                                      "N",
                                      "2304",
                                      "256",
                                      "1",
                                      "1729",
                                      "1",
                                      "1",
                                      "1729",
                                      "1729",
                                      "2304",
                                      "f1_r", // bad datatype
                                      "f16_r",
                                      "f32_r",
                                      "752"};
    probSol = problemFromEntries<ContractionProblem>(entries3);
    EXPECT_EQ(probSol.second, -1);

    std::vector<std::string> entries4{"T",
                                      "N",
                                      "b", // bad size
                                      "256",
                                      "1",
                                      "1729",
                                      "1",
                                      "1",
                                      "1729",
                                      "1729",
                                      "2304",
                                      "f16_r",
                                      "f16_r",
                                      "f32_r",
                                      "752"};
    probSol = problemFromEntries<ContractionProblem>(entries4);
    EXPECT_EQ(probSol.second, -1);
}