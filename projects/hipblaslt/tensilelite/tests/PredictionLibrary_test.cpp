// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/PredictionLibrary.hpp>
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
                                   262144,  // rf_capacity: 65536 regs * 4 bytes
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
        auto problem = ContractionProblemGemm::GEMM(
            false, false, m, n, k, m, n, m, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }

    origami::config_t makeOrigamiConfig(size_t mt_m,
                                        size_t mt_n,
                                        size_t mt_k,
                                        size_t index)
    {
        origami::config_t config;
        config.mt.m              = mt_m;
        config.mt.n              = mt_n;
        config.mt.k              = mt_k;
        config.mi.m              = 32;
        config.mi.n              = 32;
        config.mi.k              = 8;
        config.hand_optimized_main_loop = false;
        config.occupancy         = 1;
        config.workgroup_mapping = 6;
        config.index             = index;
        return config;
    }

    origami::problem_t toOrigamiProblem(ContractionProblemGemm const& problem)
    {
        size_t m = 1;
        size_t n = 1;
        size_t k = 1;
        size_t batch = 1;
        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            m *= problem.freeSizeA(i);
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            n *= problem.freeSizeB(i);
        for(size_t i = 0; i < problem.boundIndices().size(); ++i)
            k *= problem.boundSize(i);
        for(size_t i = 0; i < problem.batchIndices().size(); ++i)
            batch *= problem.batchSize(i);

        auto miDataType = datatypeToAnalyticalDatatype(problem.computeInputTypeA());
        if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32)
            miDataType = origami::data_type_t::XFloat32;

        return origami::problem_t{
            .size        = {m, n, k},
            .batch       = batch,
            .a_transpose = problem.transA() ? origami::transpose_t::T : origami::transpose_t::N,
            .b_transpose = problem.transB() ? origami::transpose_t::T : origami::transpose_t::N,
            .a_dtype     = datatypeToAnalyticalDatatype(problem.a().dataType()),
            .b_dtype     = datatypeToAnalyticalDatatype(problem.b().dataType()),
            .c_dtype     = datatypeToAnalyticalDatatype(problem.c().dataType()),
            .d_dtype     = datatypeToAnalyticalDatatype(problem.d().dataType()),
            .mi_dtype    = miDataType,
            .a_mx_block_size = 0,
            .b_mx_block_size = 0,
        };
    }

    void populatePredictionLibrary(ContractionProblemPredictionLibrary& lib)
    {
        lib.origami_config_list = {
            makeOrigamiConfig(128, 128, 64, 0),
            makeOrigamiConfig(64, 64, 64, 1),
        };
        lib.solution_list = {
            {0, makeSolution("cfg0", 0)},
            {1, makeSolution("cfg1", 1)},
        };
    }

    std::vector<int> rankedSolutionIndices(ContractionProblemPredictionLibrary const& lib,
                                           ContractionProblemGemm const&              problem,
                                           Hardware const&                            hardware,
                                           int                                        numSolutions)
    {
        std::vector<int> indices;
        for(auto const& solution : lib.findTopSolutions(problem, hardware, numSolutions))
            indices.push_back(solution->index);
        return indices;
    }

    std::vector<int> rankedConfigIndices(origami::problem_t const&                 problem,
                                         origami::hardware_t const&                hardware,
                                         std::vector<origami::config_t> const&     configs)
    {
        std::vector<int> indices;
        for(auto const& result : origami::rank_configs(problem, hardware, configs))
            indices.push_back(static_cast<int>(result.config.index));
        return indices;
    }
} // namespace

TEST(PredictionLibrarySkMaxCUsTest, SharedAnalyticalHardwareUnchangedWhenSkMaxCUsZero)
{
    auto analyticalHw = std::make_shared<origami::hardware_t>(makeGfx950AnalyticalHardware());
    auto device       = makeHipDeviceWithAnalytical(analyticalHw);
    device.skMaxCUs   = 0;

    const size_t originalNCU = analyticalHw->N_CU;
    ContractionProblemPredictionLibrary lib;
    populatePredictionLibrary(lib);
    auto         problem     = makeGemmProblem(4096, 4096, 1024);

    auto ranked = rankedSolutionIndices(lib, problem, device, 2);
    ASSERT_FALSE(ranked.empty());

    EXPECT_EQ(analyticalHw->N_CU, originalNCU)
        << "findTopSolutions must not mutate shared analytical hardware when skMaxCUs=0";
}

TEST(PredictionLibrarySkMaxCUsTest, SharedAnalyticalHardwareUnchangedWhenSkMaxCUsPositive)
{
    auto analyticalHw = std::make_shared<origami::hardware_t>(makeGfx950AnalyticalHardware());
    auto device       = makeHipDeviceWithAnalytical(analyticalHw);
    device.skMaxCUs   = 64;
    device.computeUnitCount = 128;

    const size_t originalNCU = analyticalHw->N_CU;
    ContractionProblemPredictionLibrary lib;
    populatePredictionLibrary(lib);
    auto         problem     = makeGemmProblem(4096, 4096, 1024);

    auto ranked = rankedSolutionIndices(lib, problem, device, 2);
    ASSERT_FALSE(ranked.empty());

    EXPECT_EQ(analyticalHw->N_CU, originalNCU)
        << "findTopSolutions must not mutate shared analytical hardware when skMaxCUs>0";
    EXPECT_EQ(originalNCU, kAnalyticalCuCount);
}

TEST(PredictionLibrarySkMaxCUsTest, FindTopSolutionsMatchesUncappedRankingWhenSkMaxCUsZero)
{
    auto analyticalHw = std::make_shared<origami::hardware_t>(makeGfx950AnalyticalHardware());
    auto device       = makeHipDeviceWithAnalytical(analyticalHw);
    device.skMaxCUs   = 0;

    ContractionProblemPredictionLibrary lib;
    populatePredictionLibrary(lib);
    auto problem = makeGemmProblem(4096, 4096, 1024);

    auto const origami_problem = toOrigamiProblem(problem);
    auto const expected
        = rankedConfigIndices(origami_problem, *analyticalHw, lib.origami_config_list);
    auto const actual = rankedSolutionIndices(lib, problem, device, static_cast<int>(expected.size()));

    ASSERT_FALSE(expected.empty());
    EXPECT_EQ(actual, expected)
        << "skMaxCUs=0 should rank with uncapped analytical hardware N_CU";
}

TEST(PredictionLibrarySkMaxCUsTest, FindTopSolutionsMatchesRankingWhenSkMaxCUsBinds)
{
    auto analyticalHw = std::make_shared<origami::hardware_t>(makeGfx950AnalyticalHardware());
    auto device       = makeHipDeviceWithAnalytical(analyticalHw);
    device.skMaxCUs   = 64;
    device.computeUnitCount = 128;

    ContractionProblemPredictionLibrary lib;
    populatePredictionLibrary(lib);
    auto problem = makeGemmProblem(4096, 4096, 1024);

    auto const origami_problem = toOrigamiProblem(problem);
    // Fixture: analytical N_CU=256, computeUnitCount=128, skMaxCUs=64 → expect 64.
    origami::hardware_t expected_hw = *analyticalHw;
    expected_hw.N_CU                = 64;

    auto const expected
        = rankedConfigIndices(origami_problem, expected_hw, lib.origami_config_list);
    auto const actual = rankedSolutionIndices(lib, problem, device, static_cast<int>(expected.size()));

    ASSERT_FALSE(expected.empty());
    EXPECT_EQ(analyticalHw->N_CU, kAnalyticalCuCount)
        << "shared analytical hardware must remain uncapped";
    EXPECT_EQ(actual, expected)
        << "findTopSolutions should rank as if N_CU were capped to skMaxCUs";
}

TEST(PredictionLibrarySkMaxCUsTest, FindTopSolutionsMatchesRankingWhenComputeUnitCountBinds)
{
    auto analyticalHw = std::make_shared<origami::hardware_t>(makeGfx950AnalyticalHardware());
    auto device       = makeHipDeviceWithAnalytical(analyticalHw);
    device.skMaxCUs   = 200;
    device.computeUnitCount = 128;

    ContractionProblemPredictionLibrary lib;
    populatePredictionLibrary(lib);
    auto problem = makeGemmProblem(4096, 4096, 1024);

    auto const origami_problem = toOrigamiProblem(problem);
    // Fixture: analytical N_CU=256, computeUnitCount=128, skMaxCUs=200 → expect 128.
    origami::hardware_t expected_hw = *analyticalHw;
    expected_hw.N_CU                = 128;

    auto const expected
        = rankedConfigIndices(origami_problem, expected_hw, lib.origami_config_list);
    auto const actual = rankedSolutionIndices(lib, problem, device, static_cast<int>(expected.size()));

    ASSERT_FALSE(expected.empty());
    EXPECT_EQ(actual, expected)
        << "findTopSolutions should rank as if N_CU were capped to computeUnitCount";
}
