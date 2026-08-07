// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/UnifiedHeuristic.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/origami.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::testing;

namespace
{
    constexpr size_t kAnalyticalCuCount = 256;

    origami::hardware_t makeGfx950AnalyticalHardware()
    {
        using arch_t = origami::hardware_t::architecture_t;
        return origami::hardware_t(arch_t::gfx950,
                                   kAnalyticalCuCount,
                                   163840,
                                   262144,
                                   8,
                                   1.0,
                                   1.0,
                                   1.0,
                                   4000000,
                                   1.2,
                                   1,
                                   std::make_tuple(0.0, 0.008, 0.0));
    }

    hip::HipAMDGPU makeHipDeviceWithAnalytical(std::shared_ptr<origami::hardware_t> const& hw)
    {
        hip::HipAMDGPU device;
        device.processor          = AMDGPU::Processor::gfx950;
        device.computeUnitCount   = static_cast<int>(hw->N_CU);
        device.deviceName         = "test-gfx950-analytical";
        device.analyticalHardware = hw;
        return device;
    }

    ContractionProblemGemm makeGemmProblem(size_t m, size_t n, size_t k)
    {
        auto problem = ContractionProblemGemm::GEMM(false, false, m, n, k, m, n, m, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }

    // Build a solution whose size mapping produces a meaningful origami config.
    std::shared_ptr<ContractionSolution> makeTiledSolution(const std::string& name,
                                                           int                index,
                                                           size_t             mtM,
                                                           size_t             mtN,
                                                           size_t             depthU)
    {
        auto sol                       = makeSolution(name, index);
        sol->sizeMapping.macroTile.x   = mtM;
        sol->sizeMapping.macroTile.y   = mtN;
        sol->sizeMapping.depthU        = depthU;
        sol->sizeMapping.matrixInstruction = {32, 32, 8, 1};
        sol->sizeMapping.CUOccupancy   = 2;
        sol->sizeMapping.workGroupMapping = 8;
        return sol;
    }

    // Minimal library that returns a fixed union from findAllSolutions and a
    // distinct sentinel from findTopSolutions so we can detect the fallback path.
    class StubUnionLibrary : public SolutionLibrary<ContractionProblemGemm>
    {
    public:
        SolutionSet<ContractionSolution>    all;
        SolutionVector<ContractionSolution> topSentinel;
        mutable SolutionLibrarySearchType   lastSearchType = SolutionLibrarySearchType::COUNT;

        std::shared_ptr<ContractionSolution>
            getSolutionByIndex(ContractionProblemGemm const&, Hardware const&, int) const override
        {
            return {};
        }

        std::shared_ptr<ContractionSolution> findBestSolution(ContractionProblemGemm const&,
                                                              Hardware const&,
                                                              double* = nullptr) const override
        {
            return {};
        }

        SolutionSet<ContractionSolution>
            findAllSolutions(ContractionProblemGemm const&,
                             Hardware const&,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            lastSearchType = searchType;
            return all;
        }

        SolutionSet<ContractionSolution>
            findAllSolutionsGroupedGemm(std::vector<ContractionProblemGemm> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return {};
        }

        SolutionVector<ContractionSolution> findTopSolutions(ContractionProblemGemm const&,
                                                            Hardware const&,
                                                            int) const override
        {
            return topSentinel;
        }

        SolutionVector<ContractionSolution>
            findTopSolutionsGroupedGemm(std::vector<ContractionProblemGemm> const&,
                                        Hardware const&,
                                        int) const override
        {
            return {};
        }

        std::string type() const override
        {
            return "StubUnion";
        }

        std::string description() const override
        {
            return "StubUnion";
        }
    };

    std::vector<int> indicesOf(SolutionVector<ContractionSolution> const& sols)
    {
        std::vector<int> out;
        for(auto const& s : sols)
            out.push_back(s->index);
        return out;
    }

    // Independent origami ranking of a set of solutions (best first), used to
    // derive the expected order of the analytically-ranked remainder.
    std::vector<int> expectedRankedIndices(std::vector<std::shared_ptr<ContractionSolution>> const& sols,
                                           ContractionProblemGemm const& problem,
                                           hip::HipAMDGPU const&         device)
    {
        origami::hardware_t hw    = makeRankingHardware(device);
        origami::problem_t  prob  = makeOrigamiProblem(problem);

        std::vector<origami::config_t> configs;
        for(size_t i = 0; i < sols.size(); ++i)
            configs.emplace_back(makeOrigamiConfig(*sols[i], static_cast<int>(i)));

        std::vector<int> out;
        for(auto const& r : origami::rank_configs(prob, hw, configs))
        {
            if(r.config.index < sols.size())
                out.push_back(sols[r.config.index]->index);
        }
        return out;
    }
} // namespace

// Without analytical hardware the unified path must fall back to the library's
// own findTopSolutions (byte-for-byte the existing behavior).
TEST(UnifiedHeuristicTest, FallsBackWhenNoAnalyticalHardware)
{
    StubUnionLibrary lib;
    lib.topSentinel = {makeSolution("sentinel", 99)};
    lib.all         = {makeTiledSolution("a", 1, 128, 128, 64)};

    AMDGPU plainDevice = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350-no-analytical");
    auto   problem     = makeGemmProblem(4096, 4096, 1024);

    auto result = findTopSolutionsUnified(lib, problem, plainDevice, 4);

    EXPECT_EQ(indicesOf(result), (std::vector<int>{99}))
        << "no analytical hardware must defer to findTopSolutions";
}

// Exact-tuned (Equal-tagged) matches are pinned ahead of the analytically
// ranked remainder.
TEST(UnifiedHeuristicTest, PinsExactTunedMatchesFirst)
{
    auto analyticalHw = std::make_shared<origami::hardware_t>(makeGfx950AnalyticalHardware());
    auto device       = makeHipDeviceWithAnalytical(analyticalHw);
    auto problem      = makeGemmProblem(4096, 4096, 1024);

    auto exact = makeTiledSolution("exact", 7, 64, 64, 32);
    exact->tag = ContractionSolution::MatchingTag::Equal;
    auto big   = makeTiledSolution("big", 1, 256, 256, 64);
    auto small = makeTiledSolution("small", 2, 32, 32, 16);

    StubUnionLibrary lib;
    lib.all = {exact, big, small};

    auto result = findTopSolutionsUnified(lib, problem, device, 3);
    auto idx    = indicesOf(result);

    // The union must be gathered with GEMM_TYPE_ONLY; DEFAULT drops the entire
    // prediction (analytical) library from the candidate pool.
    EXPECT_EQ(lib.lastSearchType, SolutionLibrarySearchType::GEMM_TYPE_ONLY)
        << "unified path must enumerate candidates with GEMM_TYPE_ONLY";

    ASSERT_EQ(idx.size(), 3u);
    EXPECT_EQ(idx.front(), 7) << "exact-tuned match must be pinned on top";

    // Remainder must match an independent origami ranking of the non-exact set.
    auto expectedRest = expectedRankedIndices({big, small}, problem, device);
    std::vector<int> actualRest(idx.begin() + 1, idx.end());
    EXPECT_EQ(actualRest, expectedRest)
        << "remainder must follow the origami analytical ranking";
}

// The union is de-duplicated and truncated to the requested count.
TEST(UnifiedHeuristicTest, DedupsAndTruncatesToRequestedCount)
{
    auto analyticalHw = std::make_shared<origami::hardware_t>(makeGfx950AnalyticalHardware());
    auto device       = makeHipDeviceWithAnalytical(analyticalHw);
    auto problem      = makeGemmProblem(4096, 4096, 1024);

    StubUnionLibrary lib;
    lib.all = {makeTiledSolution("a", 1, 256, 256, 64),
               makeTiledSolution("b", 2, 128, 128, 64),
               makeTiledSolution("c", 3, 64, 64, 32),
               makeTiledSolution("d", 4, 32, 32, 16)};

    auto result = findTopSolutionsUnified(lib, problem, device, 2);
    auto idx    = indicesOf(result);

    ASSERT_EQ(idx.size(), 2u) << "must truncate to the requested count";

    std::set<int> unique(idx.begin(), idx.end());
    EXPECT_EQ(unique.size(), idx.size()) << "results must be de-duplicated";

    auto expected = expectedRankedIndices({lib.all.begin(), lib.all.end()}, problem, device);
    ASSERT_GE(expected.size(), 2u);
    EXPECT_EQ(idx, (std::vector<int>{expected[0], expected[1]}))
        << "top-2 must be the two best analytically-ranked candidates";
}

// Requesting more than the union size returns the whole (ranked) union.
TEST(UnifiedHeuristicTest, ReturnsWholeUnionWhenRequestExceedsSize)
{
    auto analyticalHw = std::make_shared<origami::hardware_t>(makeGfx950AnalyticalHardware());
    auto device       = makeHipDeviceWithAnalytical(analyticalHw);
    auto problem      = makeGemmProblem(4096, 4096, 1024);

    StubUnionLibrary lib;
    lib.all = {makeTiledSolution("a", 1, 256, 256, 64),
               makeTiledSolution("b", 2, 128, 128, 64),
               makeTiledSolution("c", 3, 64, 64, 32)};

    auto result = findTopSolutionsUnified(lib, problem, device, 10);
    auto idx    = indicesOf(result);

    EXPECT_EQ(idx.size(), 3u);
    std::set<int> unique(idx.begin(), idx.end());
    EXPECT_EQ(unique, (std::set<int>{1, 2, 3}));
}
