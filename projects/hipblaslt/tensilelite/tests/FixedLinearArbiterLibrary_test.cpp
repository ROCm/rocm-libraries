// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/FixedLinearArbiterLibrary.hpp>

namespace TensileLite
{
    static std::shared_ptr<ContractionSolution> makeSolution(int index, std::string name)
    {
        auto solution = std::make_shared<ContractionSolution>();
        solution->index = index;
        solution->kernelName = std::move(name);
        return solution;
    }

    TEST(FixedLinearArbiter, FrozenFeatureContractParsesKernelName)
    {
        FixedLinearArbiterLibrary<ContractionProblemGemm, ContractionSolution> library;
        library.cuCount = 96.0;
        auto solution = std::make_shared<ContractionSolution>();
        solution->kernelName = "Cijk_MT64x32x16_MIWT2_1_WG32_4_1_PGR2_PLR1_WGM8_GRVWA8_GRVWB4_LPA8_LPB8_SU32_X";
        ContractionProblemGemm problem;
        auto features = library.featureVector(problem, solution->kernelName);
        EXPECT_DOUBLE_EQ(features[7], std::log1p(64.0));
        EXPECT_DOUBLE_EQ(features[8], std::log1p(32.0));
        EXPECT_DOUBLE_EQ(features[9], std::log1p(16.0));
        EXPECT_DOUBLE_EQ(features[17], std::log1p(8.0));
        EXPECT_DOUBLE_EQ(features[21], std::log1p(32.0));
    }

    TEST(FixedLinearArbiter, HigherScoreWinsAndTieChoosesG0)
    {
        auto g0 = makeSolution(1, "Cijk_MT16x16x16_MIWT1_1_WG16_2_1_PGR0_PLR0_WGM1_GRVWA1_GRVWB1_LPA0_LPB0_SU0_X");
        auto o3 = makeSolution(2, "Cijk_MT64x64x16_MIWT1_1_WG16_2_1_PGR0_PLR0_WGM1_GRVWA1_GRVWB1_LPA0_LPB0_SU0_X");
        FixedLinearArbiterLibrary<ContractionProblemGemm, ContractionSolution> library;
        library.g0Library = std::make_shared<SingleContractionLibrary>(g0);
        library.o3Library = std::make_shared<SingleContractionLibrary>(o3);
        library.weights.assign(22, 0.0);
        library.weights[7] = 1.0;
        AMDGPU hardware(AMDGPU::Processor::gfx1100, 96, "gfx1100");
        ContractionProblemGemm problem;
        auto selected = library.findBestSolution(problem, hardware);
        EXPECT_EQ(selected, o3);
        EXPECT_EQ(selected->tag, ContractionSolution::MatchingTag::FixedLinearArbiter);

        library.weights.assign(22, 0.0);
        selected = library.findBestSolution(problem, hardware);
        EXPECT_EQ(selected, g0);
    }
}
