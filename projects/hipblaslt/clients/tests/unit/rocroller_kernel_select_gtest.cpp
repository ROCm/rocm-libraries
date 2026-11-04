// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the eligibility checks the rocRoller-off path applies to a
// checked-in kernel that implements a ranked tile. The relative include avoids
// shadowing the clients' utility.hpp with the library's internal header of the
// same name.

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocroller_kernel_select.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace
{
    namespace Predicates = TensileLite::Predicates;

    using Problem  = TensileLite::ContractionProblemGemm;
    using Solution = TensileLite::ContractionSolution;

    const std::string rrName = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_WGT_16x16x128_UR_2";

    // Without split-K, so it needs no workspace.
    std::shared_ptr<Solution> kernel(const std::string& name = rrName)
    {
        auto solution                      = std::make_shared<Solution>();
        solution->solutionName             = name;
        solution->sizeMapping.macroTile    = TensileLite::dim3(16, 16, 1);
        solution->sizeMapping.depthU       = 128;
        solution->sizeMapping.globalSplitU = 1;
        return solution;
    }

    // Reads only the scale tensors whose MX block sizes it is given.
    std::shared_ptr<Solution>
        scaleKernel(int mxBlockA, int mxBlockB, const std::string& name = rrName)
    {
        using ProblemPredicate = std::shared_ptr<Predicates::Predicate<Problem>>;
        const ProblemPredicate blockA
            = std::make_shared<Predicates::Contraction::MXBlockA>(mxBlockA);
        const ProblemPredicate blockB
            = std::make_shared<Predicates::Contraction::MXBlockB>(mxBlockB);

        auto solution              = kernel(name);
        solution->problemPredicate = std::make_shared<Predicates::And<Problem>>(
            std::vector<ProblemPredicate>{blockA, blockB});
        return solution;
    }

    std::shared_ptr<Solution> kernelFor(TensileLite::AMDGPU::Processor processor)
    {
        auto solution = kernel();
        solution->hardwarePredicate
            = std::make_shared<Predicates::IsSubclass<TensileLite::Hardware, TensileLite::AMDGPU>>(
                std::make_shared<Predicates::GPU::ProcessorEqual>(processor));
        return solution;
    }

    TensileLite::AMDGPU gfx950()
    {
        return TensileLite::AMDGPU(TensileLite::AMDGPU::Processor::gfx950, 256, "gfx950");
    }

    Problem fp8Problem(bool scaleA, bool scaleB)
    {
        const size_t M = 256, N = 256, K = 256;
        auto         problem = Problem::GEMM_Strides(
            /*transA*/ true,
            /*transB*/ false,
            rocisa::DataType::Float8,
            rocisa::DataType::Float8,
            rocisa::DataType::Half,
            rocisa::DataType::Half,
            M,
            N,
            K,
            1,
            K,
            K * M,
            K,
            K * N,
            M,
            M * N,
            M,
            M * N,
            0.0);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        if(scaleA)
            problem.setMXScaleA(rocisa::DataType::E8, 32);
        if(scaleB)
            problem.setMXScaleB(rocisa::DataType::E8, 32);
        return problem;
    }
}

// The one-sided MX case: a kernel that reads both scale tensors can carry the
// same tile as one that reads only A's, and must be refused for it.
TEST(HostUnitRocRollerKernelSelect, smoke_OneSidedMXRejectsAKernelThatReadsBothScales)
{
    const auto bothScales = scaleKernel(32, 32);
    const auto onlyA      = scaleKernel(32, 0);
    const auto device     = gfx950();
    const auto oneSided   = fp8Problem(true, false);

    EXPECT_FALSE(rocblaslt::rocRollerKernelAccepts(*bothScales, oneSided, device));
    EXPECT_TRUE(rocblaslt::rocRollerKernelAccepts(*onlyA, oneSided, device));
    EXPECT_TRUE(rocblaslt::rocRollerKernelAccepts(*bothScales, fp8Problem(true, true), device));
}

TEST(HostUnitRocRollerKernelSelect, smoke_RejectsAKernelBuiltForAnotherDevice)
{
    const auto device  = gfx950();
    const auto problem = fp8Problem(true, true);

    EXPECT_FALSE(rocblaslt::rocRollerKernelAccepts(
        *kernelFor(TensileLite::AMDGPU::Processor::gfx942), problem, device));
    EXPECT_TRUE(rocblaslt::rocRollerKernelAccepts(
        *kernelFor(TensileLite::AMDGPU::Processor::gfx950), problem, device));
}

// The off path matches a tile by name and the checked-in path by fields, so
// neither check may read the name. A kernel without predicates accepts.
TEST(HostUnitRocRollerKernelSelect, smoke_ChecksIgnoreTheKernelName)
{
    const auto device   = gfx950();
    const auto oneSided = fp8Problem(true, false);

    for(const std::string name :
        {rrName,
         std::string("RR_GEMM_TN_FP8_FP8_Half_Half_Float_SK_WGT_32x32x128_UR_2"),
         std::string("Cijk_Alik_Bljk_F8HS_BH_MT16x16x128"),
         std::string()})
    {
        SCOPED_TRACE("name: \"" + name + "\"");

        auto unpredicated               = kernel(name);
        unpredicated->hardwarePredicate = nullptr;
        unpredicated->problemPredicate  = nullptr;
        EXPECT_TRUE(rocblaslt::rocRollerKernelAccepts(*unpredicated, oneSided, device));
        EXPECT_TRUE(rocblaslt::rocRollerKernelFitsWorkspace(*unpredicated, oneSided, device, 0));

        EXPECT_FALSE(
            rocblaslt::rocRollerKernelAccepts(*scaleKernel(32, 32, name), oneSided, device));
    }
}

// A budget of exactly the requirement is enough; one byte less is not.
TEST(HostUnitRocRollerKernelSelect, smoke_RefusesAKernelThatNeedsMoreWorkspaceThanTheBudget)
{
    const auto device  = gfx950();
    const auto problem = fp8Problem(true, true);

    // A split-K kernel needs a partial tile per workgroup and split.
    const auto splitK                         = kernel();
    splitK->sizeMapping.globalSplitU          = 2;
    splitK->sizeMapping.workspaceSizePerElemC = 4;

    const size_t needed = splitK->requiredWorkspaceSize(problem, device);
    ASSERT_GT(needed, 0u);
    EXPECT_TRUE(rocblaslt::rocRollerKernelFitsWorkspace(*splitK, problem, device, needed));
    EXPECT_FALSE(rocblaslt::rocRollerKernelFitsWorkspace(*splitK, problem, device, needed - 1));

    const auto noWorkspace = kernel();
    ASSERT_EQ(noWorkspace->requiredWorkspaceSize(problem, device), 0u);
    EXPECT_TRUE(rocblaslt::rocRollerKernelFitsWorkspace(*noWorkspace, problem, device, 0));
}
