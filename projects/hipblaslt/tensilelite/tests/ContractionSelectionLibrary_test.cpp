/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/ExactLogicLibrary.hpp>

using namespace TensileLite;

TEST(ContractionSelectionLibraryTest, Single)
{
    std::shared_ptr<Hardware> hardware = std::make_shared<AMDGPU>(
        AMDGPU::Processor::gfx900, 64, "AMD Radeon Vega Frontier Edition");

    SingleContractionLibrary lib;

    lib.solution = std::make_shared<ContractionSolution>();

    auto problem = std::make_shared<ContractionProblemGemm>();

    EXPECT_EQ(lib.findBestSolution(*problem, *hardware), lib.solution);
}

TEST(ContractionSelectionLibraryTest, GPUSelection)
{
    std::shared_ptr<Hardware> v10 = std::make_shared<AMDGPU>(
        AMDGPU::Processor::gfx900, 64, "AMD Radeon Vega Frontier Edition");
    std::shared_ptr<Hardware> v20
        = std::make_shared<AMDGPU>(AMDGPU::Processor::gfx906, 60, "AMD Radeon Vega 7");
    std::shared_ptr<Hardware> v20_64CU
        = std::make_shared<AMDGPU>(AMDGPU::Processor::gfx906, 64, "AMD Radeon Vega 7");

    // Create solutions
    auto v20Solution      = std::make_shared<ContractionSolution>();
    auto v20Solution_64CU = std::make_shared<ContractionSolution>();
    auto genericSolution  = std::make_shared<ContractionSolution>();

    // Create libraries
    std::shared_ptr<ContractionLibrary> v20Lib
        = std::make_shared<SingleContractionLibrary>(v20Solution);
    std::shared_ptr<ContractionLibrary> v20Lib_64CU
        = std::make_shared<SingleContractionLibrary>(v20Solution_64CU);
    auto genericLib = std::make_shared<SingleContractionLibrary>(genericSolution);

    // Create hardware predicate for a generic "V20"
    auto isV20 = std::make_shared<Predicates::GPU::ProcessorEqual>(AMDGPU::Processor::gfx906);
    std::shared_ptr<Predicates::Predicate<Hardware>> isAMDGPUV20
        = std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isV20);
    HardwarePredicate hardwareIsAMDGPUV20(isAMDGPUV20);

    // Create hardware predicate for a "V20" with 64 CU
    std::shared_ptr<Predicates::Predicate<AMDGPU>> isV20Proc
        = std::make_shared<Predicates::GPU::ProcessorEqual>(AMDGPU::Processor::gfx906);
    std::shared_ptr<Predicates::Predicate<AMDGPU>> is64CU
        = std::make_shared<Predicates::GPU::CUCountEqual>(64);
    std::shared_ptr<Predicates::Predicate<AMDGPU>> isAMDGPUV20_64CU
        = std::make_shared<Predicates::And<AMDGPU>>(
            std::initializer_list<std::shared_ptr<Predicates::Predicate<AMDGPU>>>{isV20Proc,
                                                                                  is64CU});
    HardwarePredicate hardwareIsAMDGPUV20_64CU(
        std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isAMDGPUV20_64CU));

    // Create hierarchy for hardware selection
    ContractionHardwareSelectionLibrary::Row v20Row(hardwareIsAMDGPUV20, v20Lib);
    ContractionHardwareSelectionLibrary::Row v20Row_64CU(hardwareIsAMDGPUV20_64CU, v20Lib_64CU);
    ContractionHardwareSelectionLibrary      lib({v20Row_64CU, v20Row});

    auto problem = std::make_shared<ContractionProblemGemm>();

    EXPECT_EQ(lib.findBestSolution(*problem, *v20), v20Solution);
    EXPECT_EQ(lib.findBestSolution(*problem, *v20_64CU), v20Solution_64CU);
    EXPECT_EQ(lib.findBestSolution(*problem, *v10), std::shared_ptr<ContractionSolution>());

    HardwarePredicate allHardware(std::make_shared<Predicates::True<Hardware>>());
    lib.rows.push_back(std::make_pair(allHardware, genericLib));

    EXPECT_EQ(lib.findBestSolution(*problem, *v20), v20Solution);
    EXPECT_EQ(lib.findBestSolution(*problem, *v20_64CU), v20Solution_64CU);
    EXPECT_EQ(lib.findBestSolution(*problem, *v10), genericSolution);
}

TEST(ContractionSelectionLibraryTest, RegionSelection)
{
    // Create solutions
    auto region1Solution = std::make_shared<ContractionSolution>();
    auto region2Solution = std::make_shared<ContractionSolution>();
    auto genericSolution = std::make_shared<ContractionSolution>();

    // Create libraries
    auto region1Lib = std::make_shared<SingleContractionLibrary>(region1Solution);
    auto region2Lib = std::make_shared<SingleContractionLibrary>(region2Solution);
    auto genericLib = std::make_shared<SingleContractionLibrary>(genericSolution);

    // Create region predicate for (6000 <= M < 8000), (0 <= N < 7000)
    using Predicate   = Predicates::Predicate<ContractionProblemGemm>;
    using SizeInRange = Predicates::Contraction::SizeInRange;
    using Range       = Predicates::Contraction::Range;
    using And         = Predicates::And<ContractionProblemGemm>;
    size_t max_size   = std::numeric_limits<size_t>::max();

    std::shared_ptr<Predicate> regionM  = std::make_shared<SizeInRange>(0, Range{6000, 8000});
    std::shared_ptr<Predicate> regionN1 = std::make_shared<SizeInRange>(1, Range{0, 7000});
    std::shared_ptr<Predicate> regionN2 = std::make_shared<SizeInRange>(1, Range{7000, max_size});

    // Create region predicate for (6000 <= M < 8000), (0 <= N < 7000)
    auto preds1    = {regionM, regionN1};
    auto isRegion1 = std::make_shared<And>(preds1);

    // Create region predicate for (6000 <= M < 8000), (7000 <= N < max)
    auto preds2    = {regionM, regionN2};
    auto isRegion2 = std::make_shared<And>(preds2);

    // Create fallthrough predicate (i.e. default)
    ContractionProblemPredicate allProbs(std::make_shared<Predicates::True<ContractionProblemGemm>>());

    // Create hierarchy for region selection
    ContractionProblemSelectionLibrary::Row Region1Row(isRegion1, region1Lib);
    ContractionProblemSelectionLibrary::Row Region2Row(isRegion2, region2Lib);
    ContractionProblemSelectionLibrary::Row GenericRow(allProbs, genericLib);
    ContractionProblemSelectionLibrary      lib({Region1Row, Region2Row, GenericRow});

    auto Region1Problem
        = ContractionProblemGemm::GEMM(false, false, 7000, 6500, 1000, 7000, 1000, 7000, 1.0, false, 1);
    auto Region2Problem
        = ContractionProblemGemm::GEMM(false, false, 7000, 7500, 1000, 7000, 1000, 7000, 1.0, false, 1);
    auto OutRegionProblem
        = ContractionProblemGemm::GEMM(false, false, 5000, 2000, 1000, 5000, 1000, 5000, 1.0, false, 1);

    AMDGPU gpu;
    EXPECT_EQ(lib.findBestSolution(Region1Problem, gpu), region1Solution);
    EXPECT_EQ(lib.findBestSolution(Region2Problem, gpu), region2Solution);
    EXPECT_EQ(lib.findBestSolution(OutRegionProblem, gpu), genericSolution);
}

TEST(ContractionSelectionLibraryTest, TransposeSelection)
{
    auto NNSolution = std::make_shared<ContractionSolution>();
    auto NTSolution = std::make_shared<ContractionSolution>();
    auto TNSolution = std::make_shared<ContractionSolution>();
    auto TTSolution = std::make_shared<ContractionSolution>();

    NNSolution->index = 0;
    NTSolution->index = 1;
    TNSolution->index = 2;
    TTSolution->index = 3;

    SolutionMap<ContractionSolution> map(
        {{0, NNSolution}, {1, NTSolution}, {2, TNSolution}, {3, TTSolution}});

    std::shared_ptr<ContractionLibrary> NNLibrary
        = std::make_shared<SingleContractionLibrary>(NNSolution);
    std::shared_ptr<ContractionLibrary> NTLibrary
        = std::make_shared<SingleContractionLibrary>(NTSolution);
    std::shared_ptr<ContractionLibrary> TNLibrary
        = std::make_shared<SingleContractionLibrary>(TNSolution);
    std::shared_ptr<ContractionLibrary> TTLibrary
        = std::make_shared<SingleContractionLibrary>(TTSolution);

    auto lib = std::make_shared<ContractionProblemMapLibrary>();

    lib->property = std::make_shared<Contraction::OperationIdentifier>();
    lib->map["Contraction_l_Ailk_Bljk_Cijk_Dijk"] = NNLibrary;
    lib->map["Contraction_l_Ailk_Bjlk_Cijk_Dijk"] = NTLibrary;
    lib->map["Contraction_l_Alik_Bljk_Cijk_Dijk"] = TNLibrary;
    lib->map["Contraction_l_Alik_Bjlk_Cijk_Dijk"] = TTLibrary;

    AMDGPU gpu;

    auto NNProblem = ContractionProblemGemm::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
    auto NTProblem = ContractionProblemGemm::GEMM(false, true, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
    auto TNProblem = ContractionProblemGemm::GEMM(true, false, 4, 4, 4, 4, 4, 4, 1.2, false, 1);
    auto TTProblem = ContractionProblemGemm::GEMM(true, true, 4, 4, 4, 4, 4, 4, 1.2, false, 1);

    // auto WeirdProblemC = ContractionProblemGemm::FromBLAS( true,  true, 4,4,4,
    // 4,4,4, false, false, 1); WeirdProblemC.c.transpose(0,1);

    // auto WeirdProblemD = ContractionProblemGemm::FromBLAS( true,  true, 4,4,4,
    // 4,4,4, false, false, 1); WeirdProblemD.d.transpose(0,1);

    EXPECT_EQ(lib->findBestSolution(NNProblem, gpu), NNSolution);
    EXPECT_EQ(lib->findBestSolution(NTProblem, gpu), NTSolution);
    EXPECT_EQ(lib->findBestSolution(TNProblem, gpu), TNSolution);
    EXPECT_EQ(lib->findBestSolution(TTProblem, gpu), TTSolution);

    // EXPECT_EQ(lib->findBestSolution(WeirdProblemC, gpu), nullptr);
    // EXPECT_EQ(lib->findBestSolution(WeirdProblemD, gpu), nullptr);

    MasterContractionLibrary mlib;
    mlib.solutions = map;
    mlib.library   = lib;
}
