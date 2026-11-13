// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iterator>
#include <map>
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

    // findTopSolutions returns at most `available` solutions and counts its calls.
    class TopCountingSubLibrary : public CountingSubLibrary
    {
    public:
        int                      available = 4;
        mutable std::atomic<int> topCalls{0};

        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                             Hardware const&,
                                                             int numSolutions) const override
        {
            int                                 call = ++topCalls;
            SolutionVector<ContractionSolution> rv;
            for(int i = 0; i < std::min(numSolutions, available); ++i)
                rv.push_back(
                    makeSolution("top_" + std::to_string(call) + "_" + std::to_string(i), i));
            return rv;
        }
    };

    // Gives `problem` a bias of `type`.
    void withBias(ContractionProblemGemm& problem, rocisa::DataType type)
    {
        problem.setUseBias(1);
        problem.setBias(type, 1024, 1024);
    }

    // dummyProblem() with `beta`: the same GEMM, with what dummyProblem() sets beyond the
    // factory copied from it.
    ContractionProblemGemm withBeta(double beta)
    {
        auto const reference = dummyProblem();
        auto       problem   = ContractionProblemGemm::GEMM(
            false, false, 1024, 1024, 1024, 1024, 1024, 1024, beta, false, 1);
        problem.setComputeInputTypeA(reference.computeInputTypeA());
        problem.setComputeInputTypeB(reference.computeInputTypeB());
        problem.setParams().setUniformSummationOrder(reference.getParams().uniformSummationOrder());
        return problem;
    }

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
        add([](auto& p) { p.setWorld(8); });
        add([](auto& p) { p.setParams().setSmCountTarget(64); });
        add([](auto& p) { p.setParams().setStreamKTileSchedulingMode(1); });
        add([](auto& p) { p.setParams().setUniformSummationOrder(true); });
        // The fields of the second comparison call, which predicates read.
        add([](auto& p) { withBias(p, rocisa::DataType::Half); });
        add([](auto& p) { withBias(p, rocisa::DataType::BFloat16); });
        add([](auto& p) { p.setParams().setActivationEnum(ActivationType::Relu); });
        add([](auto& p) {
            p.setOutputAmaxD(true);
            p.setAmaxD(rocisa::DataType::Float);
        });
        add([](auto& p) {
            p.setOutputAmaxD(true);
            p.setAmaxD(rocisa::DataType::Half);
        });
        v.push_back(withBeta(0.0));
        add([](auto& p) { p.setParams().setGSU(2); });
        add([](auto& p) { p.setParams().setFallbackStatus(true); });
        add([](auto& p) { p.setParams().setDeviceScalarAlpha(true); });
        // The w4a16 mode of A: scale block size, zero point and Int4 encoding.
        add([](auto& p) { p.setW4A16ScaleBlockSizeA(128, rocisa::DataType::Half, 1024, 8); });
        add([](auto& p) { p.setW4A16ScaleBlockSizeA(128, rocisa::DataType::Half, 1024, 8, true); });
        add([](auto& p) {
            p.setInt4EncodingA(ContractionProblemGemm::Int4Encoding::UnsignedBias8);
        });
        // Each of the following differs in one compared field alone from dummyProblem() or
        // from the variant named beside it.
        auto strided = [](size_t ldc, size_t ldd) {
            auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                                false,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                1024,
                                                                1024,
                                                                1024,
                                                                1,
                                                                1024,
                                                                -1,
                                                                1024,
                                                                -1,
                                                                ldc,
                                                                -1,
                                                                ldd,
                                                                -1,
                                                                1.0);
            problem.setComputeInputTypeA(rocisa::DataType::Float);
            problem.setComputeInputTypeB(rocisa::DataType::Float);
            return problem;
        };
        v.push_back(strided(2048, 1024)); // c
        v.push_back(strided(1024, 2048)); // d
        add([](auto& p) { // e, beside the setUseE(true) variant
            p.setUseE(true);
            p.setE(rocisa::DataType::Float, {1024, 1024, 1}, {1, 1024, 1024 * 1024});
        });
        add([](auto& p) { // biasSrc, beside withBias(p, rocisa::DataType::Half)
            p.setUseBias(1);
            p.setBias(rocisa::DataType::Half, 1024, 1024, false, ContractionProblemGemm::TENSOR::A);
        });
        add([](auto& p) { p.setMXScaleA(rocisa::DataType::E8, 0); }); // mxTypeA
        add([](auto& p) { p.setMXScaleA(rocisa::DataType::E8, 32); }); // mxBlockA, beside it
        add([](auto& p) { // mxBlockFreeA, beside the mxBlockA variant
            p.setMXScaleA(rocisa::DataType::E8, 32, {}, true, 2);
        });
        add([](auto& p) { p.setMXScaleB(rocisa::DataType::E8, 0); }); // mxTypeB
        add([](auto& p) { p.setMXScaleB(rocisa::DataType::E8, 32); }); // mxBlockB, beside it
        add([](auto& p) { // mxBlockFreeB, beside the mxBlockB variant
            p.setMXScaleB(rocisa::DataType::E8, 32, {}, true, 2);
        });
        return v;
    }

#if defined(TENSILELITE_CONTRACTION_PROBLEM_DETAIL_HPP)
    // The source of `name` in `text` from its first '{' to the matching '}', comments
    // removed.
    std::string definitionOf(std::string const& text, std::string const& name)
    {
        auto const start = text.find(name);
        if(start == std::string::npos)
            return {};
        auto open  = text.find('{', start);
        int  depth = 0;
        for(auto i = open; i < text.size(); ++i)
        {
            if(text[i] == '{')
                ++depth;
            else if(text[i] == '}' && --depth == 0)
            {
                std::string body = text.substr(open, i + 1 - open);
                std::string out;
                for(size_t j = 0; j < body.size(); ++j)
                {
                    if(body.compare(j, 2, "//") == 0)
                        j = body.find('\n', j);
                    else if(body.compare(j, 2, "/*") == 0)
                    {
                        j = body.find("*/", j);
                        if(j != std::string::npos)
                            ++j;
                    }
                    else
                        out += body[j];
                    if(j == std::string::npos)
                        break;
                }
                return out;
            }
        }
        return {};
    }

    // The arguments of every call of `function` in `code`, in order, without whitespace.
    std::vector<std::string> callArguments(std::string const& code, std::string const& function)
    {
        std::vector<std::string> args;
        for(auto at = code.find(function + "("); at != std::string::npos;
            at      = code.find(function + "(", at + 1))
        {
            std::string arg;
            int         depth = 1;
            for(auto i = at + function.size() + 1; i < code.size() && depth > 0; ++i)
            {
                char const c = code[i];
                depth += c == '(' ? 1 : c == ')' ? -1 : 0;
                if(depth == 0 || (depth == 1 && c == ','))
                {
                    args.push_back(arg);
                    arg.clear();
                }
                else if(!std::isspace(static_cast<unsigned char>(c)))
                    arg += c;
            }
        }
        return args;
    }

    // `args` without the prefix each must start with; empty when one does not.
    std::vector<std::string> withoutPrefix(std::vector<std::string> const& args,
                                           std::string const&              prefix)
    {
        std::vector<std::string> out;
        for(auto const& arg : args)
        {
            if(arg.compare(0, prefix.size(), prefix) != 0)
                return {};
            out.push_back(arg.substr(prefix.size()));
        }
        return out;
    }
#endif
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

TEST(CachingLibraryTest, DeviceScalarAlphaCreatesDistinctCacheEntries)
{
    auto sub = std::make_shared<CountingSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    auto vectorProblem = dummyProblem();
    vectorProblem.setUseScaleAlphaVec(1);
    auto scalarProblem = vectorProblem;
    scalarProblem.setParams().setDeviceScalarAlpha(true);

    auto vectorSolution = caching->findBestSolution(vectorProblem, device);
    auto scalarSolution = caching->findBestSolution(scalarProblem, device);
    ASSERT_NE(vectorSolution, nullptr);
    ASSERT_NE(scalarSolution, nullptr);
    EXPECT_EQ(sub->callCount.load(), 2);
    EXPECT_NE(vectorSolution, scalarSolution);
    EXPECT_EQ(caching->findBestSolution(vectorProblem, device), vectorSolution);
    EXPECT_EQ(caching->findBestSolution(scalarProblem, device), scalarSolution);
    EXPECT_EQ(sub->callCount.load(), 2);
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

#if defined(TENSILELITE_CONTRACTION_PROBLEM_DETAIL_HPP)
// CacheKey<ContractionProblemGemm>::append appends the fields that
// Comparison<ContractionProblemGemm>::compare compares, all of them and in its order, as the
// header spells them: a field the comparison gains fails here until the key appends it too.
TEST(CachingLibraryTest, ProblemCacheKeyAppendsTheComparedFields)
{
    std::ifstream in(TENSILELITE_CONTRACTION_PROBLEM_DETAIL_HPP);
    ASSERT_TRUE(in.good()) << TENSILELITE_CONTRACTION_PROBLEM_DETAIL_HPP;
    std::string const text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};

    auto const compareArgs = callArguments(
        definitionOf(text, "struct Comparison<ContractionProblemGemm>"), "LexicographicCompare");
    std::vector<std::string> lhs, rhs;
    for(size_t i = 0; i < compareArgs.size(); ++i)
        (i % 2 == 0 ? lhs : rhs).push_back(compareArgs[i]);
    auto const compared = withoutPrefix(lhs, "lhs.");
    ASSERT_FALSE(compared.empty()) << "no compared field was read";
    EXPECT_EQ(withoutPrefix(rhs, "rhs."), compared);

    auto const keyArgs = callArguments(
        definitionOf(text, "struct CacheKey<ContractionProblemGemm>"), "AppendCacheKeys");
    ASSERT_FALSE(keyArgs.empty());
    EXPECT_EQ(keyArgs.front(), "key");
    EXPECT_EQ(withoutPrefix({keyArgs.begin() + 1, keyArgs.end()}, "problem."), compared);
}
#endif

// The key reads beta only as whether it is 0, as the comparison does.
TEST(CachingLibraryTest, ProblemCacheKeyReadsWhetherBetaIsZero)
{
    EXPECT_TRUE(withBeta(0.5) == dummyProblem());
    EXPECT_EQ(keyOf(withBeta(0.5)), keyOf(dummyProblem()));
    EXPECT_EQ(keyOf(withBeta(2.0)), keyOf(dummyProblem()));
    EXPECT_FALSE(withBeta(0.0) == dummyProblem());
    EXPECT_NE(keyOf(withBeta(0.0)), keyOf(dummyProblem()));
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

    EXPECT_EQ(caching->findTopSolutions(problem, device, 3).size(), 3u);
    EXPECT_EQ(sub->topCalls.load(), 2);

    auto all = caching->findTopSolutions(problem, device, 10);
    EXPECT_EQ(all.size(), 4u);
    EXPECT_EQ(sub->topCalls.load(), 3);
    EXPECT_EQ(caching->findTopSolutions(problem, device, 20), all);
    EXPECT_EQ(sub->topCalls.load(), 3);

    EXPECT_EQ(caching->findTopSolutions(problemWithSmCountTarget(64), device, 2).size(), 2u);
    EXPECT_EQ(
        caching->findTopSolutions(problem, makeDevice(_MI350_CHIP_ID, _CPX_CU, "cpx"), 2).size(),
        2u);
    EXPECT_EQ(sub->topCalls.load(), 5);
}

// A problem that differs from a cached one only in a field that predicates
// read, such as the bias data type, is not served the cached solutions: a bf16
// bias after an f16 bias of the same GEMM reaches the sub-library again.
TEST(CachingLibraryTest, FindTopSolutionsKeysOnTheBiasDataType)
{
    auto sub     = std::make_shared<TopCountingSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    auto         half   = dummyProblem();
    auto         bf16   = dummyProblem();
    withBias(half, rocisa::DataType::Half);
    withBias(bf16, rocisa::DataType::BFloat16);

    auto const halfTop = caching->findTopSolutions(half, device, 2);
    auto const bf16Top = caching->findTopSolutions(bf16, device, 2);
    EXPECT_EQ(sub->topCalls.load(), 2);
    EXPECT_NE(halfTop, bf16Top);
    EXPECT_EQ(caching->findTopSolutions(half, device, 2), halfTop);
    EXPECT_EQ(caching->findTopSolutions(bf16, device, 2), bf16Top);
    EXPECT_EQ(sub->topCalls.load(), 2);
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

namespace
{
    // A problem whose cache key is the device's followed by its text, so that a
    // test chooses the length of the key.
    struct TextProblem
    {
        std::string text;

        bool operator==(TextProblem const& other) const
        {
            return text == other.text;
        }
    };
} // namespace

namespace std
{
    template <>
    struct hash<TextProblem>
    {
        size_t operator()(TextProblem const& problem) const noexcept
        {
            return std::hash<std::string>{}(problem.text);
        }
    };

    template <>
    struct hash<std::vector<TextProblem>>
    {
        size_t operator()(std::vector<TextProblem> const& problems) const noexcept
        {
            size_t h = 0;
            for(auto const& problem : problems)
                h = h * 31 + std::hash<std::string>{}(problem.text);
            return h;
        }
    };
} // namespace std

namespace TensileLite
{
    template <>
    struct CacheKey<TextProblem>
    {
        enum
        {
            implemented = true
        };

        static void append(std::string& key, TextProblem const& problem)
        {
            key.append(problem.text);
        }
    };
} // namespace TensileLite

namespace
{
    // findTopSolutions returns a new solution on every call and counts the calls.
    class TextSubLibrary : public SolutionLibrary<TextProblem, ContractionSolution>
    {
    public:
        mutable int topCalls = 0;

        std::shared_ptr<ContractionSolution>
            getSolutionByIndex(TextProblem const&, Hardware const&, int) const override
        {
            return {};
        }

        std::shared_ptr<ContractionSolution>
            findBestSolution(TextProblem const&, Hardware const&, double* = nullptr) const override
        {
            return {};
        }

        SolutionSet<ContractionSolution>
            findAllSolutions(TextProblem const&,
                             Hardware const&,
                             SolutionLibrarySearchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            return {};
        }

        SolutionSet<ContractionSolution>
            findAllSolutionsGroupedGemm(std::vector<TextProblem> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return {};
        }

        std::string type() const override
        {
            return "TextSub";
        }

        std::string description() const override
        {
            return "TextSub";
        }

        SolutionVector<ContractionSolution>
            findTopSolutions(TextProblem const&, Hardware const&, int) const override
        {
            ++topCalls;
            return {makeSolution("top_" + std::to_string(topCalls), topCalls)};
        }
    };

    using TextCaching = CachingLibrary<TextProblem, ContractionSolution>;

    // A problem whose whole cache key is `bytes` long, `first` and `last` its
    // first and last bytes after the device's.
    TextProblem textProblem(AMDGPU const& device, size_t bytes, char first, char last)
    {
        std::string text(bytes - keyOf(device).size(), 'k');
        text.front() = first;
        text.back()  = last;
        return {text};
    }
} // namespace

// Keys of any length are stored and served, those longer than the 512 bytes an
// entry holds inline as well as shorter ones, and keys that differ only in
// their first or last byte stay apart.
TEST(CachingLibraryTest, FindTopSolutionsServesKeysOfAnyLength)
{
    auto         sub     = std::make_shared<TextSubLibrary>();
    auto         caching = std::make_shared<TextCaching>(sub);
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    size_t const prefix  = keyOf(device).size();
    ASSERT_LT(prefix + 64, size_t(500));

    std::vector<TextProblem> problems;
    for(size_t bytes : {prefix + 2,
                        prefix + 64,
                        size_t(511),
                        size_t(512),
                        size_t(513),
                        size_t(4096),
                        size_t(1) << 16})
        for(char first : {'a', 'b'})
            for(char last : {'y', 'z'})
                problems.push_back(textProblem(device, bytes, first, last));

    std::vector<SolutionVector<ContractionSolution>> first;
    for(auto const& problem : problems)
        first.push_back(caching->findTopSolutions(problem, device, 1));
    ASSERT_EQ(sub->topCalls, static_cast<int>(problems.size()));
    for(size_t i = 0; i < problems.size(); ++i)
    {
        SCOPED_TRACE(problems[i].text.size());
        ASSERT_EQ(first[i].size(), 1u);
        EXPECT_EQ(caching->findTopSolutions(problems[i], device, 1), first[i]);
    }
    EXPECT_EQ(sub->topCalls, static_cast<int>(problems.size()));
}

// At capacity, a long key replaces the least recently used short one and a
// short key a long one, and every problem is served its own solutions.
TEST(CachingLibraryTest, FindTopSolutionsEvictsLongAndShortKeysAlike)
{
    auto         sub     = std::make_shared<TextSubLibrary>();
    auto         caching = std::make_shared<TextCaching>(sub);
    const AMDGPU device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    size_t const cap     = TextCaching::TopSolutionsCapacity;

    // Problem i has a short key when i is even and a long one when it is odd.
    auto problem = [](size_t i) {
        std::string text = std::to_string(i) + "|";
        if(i % 2 == 1)
            text.resize(1024, 'x');
        return TextProblem{text};
    };
    // What each problem was given when the sub-library last answered it.
    std::map<std::string, SolutionVector<ContractionSolution>> given;
    // Asks for problem i; true when the sub-library answered it (a miss).
    auto ask = [&](size_t i) {
        auto const p      = problem(i);
        int const  before = sub->topCalls;
        auto const top    = caching->findTopSolutions(p, device, 1);
        bool const miss   = sub->topCalls != before;
        if(miss)
            given[p.text] = top;
        else
            EXPECT_EQ(top, given[p.text]) << "problem " << i;
        return miss;
    };

    for(size_t i = 0; i < cap; ++i)
        ASSERT_TRUE(ask(i));
    // Problem 0 (short) leaves for a long key, then problem 1 (long) for a
    // short one.
    EXPECT_TRUE(ask(cap + 1));
    EXPECT_TRUE(ask(cap + 2));
    for(size_t i : {size_t(2), size_t(3), cap - 1, cap + 1, cap + 2})
        EXPECT_FALSE(ask(i)) << "problem " << i;
    // Problems 4 (short) and 5 (long) are now the least recently used.
    EXPECT_TRUE(ask(1));
    EXPECT_TRUE(ask(0));
    for(size_t i : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(6), size_t(7)})
        EXPECT_FALSE(ask(i)) << "problem " << i;
    EXPECT_TRUE(ask(4));
    EXPECT_TRUE(ask(5));
    EXPECT_EQ(static_cast<size_t>(sub->topCalls), cap + 6);
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

        SolutionVector<ContractionSolution>
            findTopSolutions(ContractionProblemGemm const& problem,
                             Hardware const&,
                             int numSolutions) const override
        {
            int available = isA(problem) ? 10 : 1;
            int n         = std::min(available, numSolutions);
            SolutionVector<ContractionSolution> out;
            for(int i = 0; i < n; i++)
                out.push_back(makeSolution("sol_" + std::to_string(i), i));

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
            return out;
        }
    };
} // namespace

// Two problems share one CachingLibrary. Thread B's findTopSolutions runs
// while thread A is still inside its own call. A's later request for every
// solution must still return all of them.
TEST(CachingLibraryTest, ConcurrentFindTopSolutionsDoesNotPoisonAlreadyRetAllCache)
{
    auto sub     = std::make_shared<FlagSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    auto problemA = problemWithSmCountTarget(64);
    auto problemB = problemWithSmCountTarget(128);

    sub->gateArmed = true;

    size_t aFirst = 0;
    std::thread tA([&] { aFirst = caching->findTopSolutions(problemA, device, 5).size(); });

    // Releases A and joins it on every exit path, so a failed ASSERT does not
    // leave a joinable thread behind.
    struct Join
    {
        FlagSubLibrary* sub;
        std::thread*    thread;
        ~Join()
        {
            sub->markBDone();
            if(thread->joinable())
                thread->join();
        }
    } join{sub.get(), &tA};

    // Start B only after A has computed its (correct, false) answer.
    {
        std::unique_lock<std::mutex> lk(sub->m);
        ASSERT_TRUE(
            sub->cv.wait_for(lk, std::chrono::seconds(5), [&] { return sub->aComputed; }));
    }
    size_t bSize = caching->findTopSolutions(problemB, device, 5).size();
    sub->markBDone();
    tA.join();
    sub->gateArmed = false;

    EXPECT_EQ(aFirst, 5u);
    EXPECT_EQ(bSize, 1u);

    // Same problem A, now asking for everything it has.
    auto again = caching->findTopSolutions(problemA, device, 10);
    EXPECT_EQ(again.size(), 10u)
        << "thread A's cache entry was marked already-returned-all using thread B's "
           "already-returned-all bit; A got fewer solutions than it asked for";
}

namespace
{
    // Holds 10 solutions for every problem. findTopSolutions(n) returns the
    // first min(n, 10) of them.
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

        mutable std::atomic<int> calls{0};

        SolutionVector<ContractionSolution>
            findTopSolutions(ContractionProblemGemm const&,
                             Hardware const&,
                             int numSolutions) const override
        {
            calls.fetch_add(1, std::memory_order_relaxed);
            const int                           n = std::min(available, numSolutions);
            SolutionVector<ContractionSolution> out;
            for(int i = 0; i < n; i++)
                out.push_back(makeSolution("sol_" + std::to_string(i), i));
            return out;
        }
    };
} // namespace

// Same problem, two request sizes. A result with fewer solutions than were
// asked for must hold every solution. A shorter list must not replace a longer
// cached list, so the later request for every solution is served from the cache.
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
                // Fewer than requested means the library had nothing more.
                if(r.size() < static_cast<size_t>(numSolutions) && r.size() != all)
                    splitPairs++;
            });
        }
        go.store(true, std::memory_order_release);
        for(auto& th : threads)
            th.join();

        const int callsAfterJoin = sub->calls.load();
        auto again = caching->findTopSolutions(problem, device, static_cast<int>(all));
        if(again.size() != all)
            finalMismatches++;
        if(sub->calls.load() != callsAfterJoin)
            finalMismatches++;
    }

    EXPECT_EQ(splitPairs.load(), 0)
        << "a cached entry paired one request's solutions with another request's "
           "already-returned-all bit";
    EXPECT_EQ(finalMismatches.load(), 0)
        << "a later request for all solutions was answered with a truncated cached list";
}

namespace
{
    // Holds N solutions. Asking for N and then N+1 returns the same N, but only
    // the second call shows that the library has nothing more.
    class ExactNThenTopUpSubLibrary : public ContractionLibrary
    {
    public:
        static constexpr int available = 5;

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
            return "ExactNThenTopUpSub";
        }
        std::string description() const override
        {
            return "ExactNThenTopUpSub";
        }

        mutable std::atomic<int> calls{0};

        SolutionVector<ContractionSolution>
            findTopSolutions(ContractionProblemGemm const&, Hardware const&, int numSolutions) const override
        {
            calls.fetch_add(1, std::memory_order_relaxed);
            const int                           n = std::min(available, numSolutions);
            SolutionVector<ContractionSolution> out;
            for(int i = 0; i < n; i++)
                out.push_back(makeSolution("sol_" + std::to_string(i), i));
            return out;
        }
    };
} // namespace

// Two same-size results where only the second was shorter than its request:
// the cache keeps the second, so repeating that request does not rescan.
TEST(CachingLibraryTest, TopUpUpgradesAlreadyRetAllForEqualSizedResult)
{
    const AMDGPU device = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");
    auto         problem = problemWithSmCountTarget(64);

    auto sub     = std::make_shared<ExactNThenTopUpSubLibrary>();
    auto caching = std::make_shared<CachingLibrary<ContractionProblemGemm>>(sub);

    auto exact = caching->findTopSolutions(problem, device, ExactNThenTopUpSubLibrary::available);
    EXPECT_EQ(exact.size(), static_cast<size_t>(ExactNThenTopUpSubLibrary::available));

    auto topUp = caching->findTopSolutions(problem, device, ExactNThenTopUpSubLibrary::available + 1);
    EXPECT_EQ(topUp.size(), static_cast<size_t>(ExactNThenTopUpSubLibrary::available));

    const int callsAfterTopUp = sub->calls.load();
    auto repeatTopUp
        = caching->findTopSolutions(problem, device, ExactNThenTopUpSubLibrary::available + 1);
    EXPECT_EQ(repeatTopUp.size(), static_cast<size_t>(ExactNThenTopUpSubLibrary::available));
    EXPECT_EQ(sub->calls.load(), callsAfterTopUp)
        << "repeating N+1 should reuse upgraded cache entry instead of rescanning sub-library";
}
