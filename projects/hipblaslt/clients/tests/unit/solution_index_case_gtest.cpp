// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the client case built from a solution's own record, on
// synthetic solutions.

#include "solution_index_case.hpp"

#include <gtest/gtest.h>

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>

#include <memory>
#include <string>
#include <variant>

namespace
{
    using Problem        = TensileLite::ContractionProblemGemm;
    namespace Predicates = TensileLite::Predicates;

    // A TN HHS solution with a 64x128 macro tile, DepthU 32 and no split.
    void makeHalfSolution(TensileLite::ContractionSolution& solution)
    {
        auto& p               = solution.problemType;
        p.operationIdentifier = "Contraction_l_Alik_Bljk_Cijk_Dijk";
        p.transA              = true;
        p.aType = p.bType = p.cType = p.dType = rocisa::DataType::Half;
        p.computeInputTypeA = p.computeInputTypeB = rocisa::DataType::Half;
        solution.sizeMapping.macroTile.x          = 64;
        solution.sizeMapping.macroTile.y          = 128;
        solution.sizeMapping.macroTile.z          = 1;
        solution.sizeMapping.depthU               = 32;
        solution.sizeMapping.globalSplitU         = 1;
        solution.problemPredicate                 = std::make_shared<Predicates::True<Problem>>();
    }

    void requireSizes(TensileLite::ContractionSolution&                            solution,
                      std::vector<std::shared_ptr<Predicates::Predicate<Problem>>> checks)
    {
        solution.problemPredicate = std::make_shared<Predicates::And<Problem>>(checks);
    }

    Arguments made(TensileLite::ContractionSolution const& solution,
                   SolutionCaseOptions const&              options = {})
    {
        auto result = solutionIndexCase(7, solution, options);
        if(auto const* unsupported = std::get_if<UnsupportedSolution>(&result))
            ADD_FAILURE() << unsupported->reason;
        return std::holds_alternative<Arguments>(result) ? std::get<Arguments>(result)
                                                         : Arguments{};
    }

    std::string unsupported(TensileLite::ContractionSolution const& solution,
                            SolutionCaseOptions const&              options = {})
    {
        auto        result = solutionIndexCase(7, solution, options);
        auto const* reason = std::get_if<UnsupportedSolution>(&result);
        return reason ? reason->reason : "";
    }
}

// One macro tile of output and eight DepthU blocks of K, run by index with
// packed storage and the solution's own types.
TEST(HostUnitSolutionIndexCase, smoke_OrdinaryGemm)
{
    SolutionCaseOptions options;
    options.alpha = 2;
    options.beta  = 1;
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    auto const arg = made(solution, options);
    EXPECT_EQ(arg.M[0], 64);
    EXPECT_EQ(arg.N[0], 128);
    EXPECT_EQ(arg.K[0], 256);
    EXPECT_EQ(arg.transA, 'T');
    EXPECT_EQ(arg.transB, 'N');
    EXPECT_EQ(arg.a_type, HIP_R_16F);
    EXPECT_EQ(arg.d_type, HIP_R_16F);
    EXPECT_EQ(arg.compute_type, HIPBLAS_COMPUTE_32F);
    EXPECT_EQ(arg.compute_input_typeA, HIP_R_16F);
    EXPECT_EQ(arg.solution_index, 7);
    EXPECT_EQ(arg.algo_method, 2);
    EXPECT_EQ(arg.alpha, 2);
    EXPECT_EQ(arg.beta, 1);
    EXPECT_EQ(arg.lda[0], 256);
    EXPECT_EQ(arg.stride_a[0], 256 * 64);
    EXPECT_EQ(arg.lde[0], 64);
    EXPECT_EQ(arg.stride_e[0], 64 * 128);
}

// The size predicates move the shape: N is pinned to 1 (a GEMV kernel) and
// GlobalSplitUCheckMinK raises K for a fixed split of 4.
TEST(HostUnitSolutionIndexCase, smoke_SizePredicatesMoveTheShape)
{
    namespace C = Predicates::Contraction;
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    solution.sizeMapping.globalSplitU = 4;
    requireSizes(solution,
                 {std::make_shared<C::SizeEqual>(1, 1),
                  std::make_shared<C::GlobalSplitUCheckMinK>(std::array<int, 2>{512, 4})});
    auto const arg = made(solution);
    EXPECT_EQ(arg.M[0], 64);
    EXPECT_EQ(arg.N[0], 1);
    // Eight blocks per split would be 1024; the predicate needs 512 * 4.
    EXPECT_EQ(arg.K[0], 2048);

    // A size check under Or has no single bound to aim for.
    requireSizes(solution,
                 {std::make_shared<Predicates::Or<Problem>>(
                     std::vector<std::shared_ptr<Predicates::Predicate<Problem>>>{
                         std::make_shared<C::SizeEqual>(1, 1)})});
    EXPECT_EQ(unsupported(solution), "size predicates under Or or Not");
}

// MX scale modes follow the scale type and the in-device layout; the gfx950
// preswizzled layout needs A transposed and B normal.
TEST(HostUnitSolutionIndexCase, smoke_MXScaleLayout)
{
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    auto& p = solution.problemType;
    p.aType = p.bType = p.computeInputTypeA = p.computeInputTypeB = rocisa::DataType::Float8;
    p.mxBlockA = p.mxBlockB = 32;
    auto arg                = made(solution);
    EXPECT_EQ(arg.scaleA, hipblaslt_scaling_format::Block_32_UE8M0);
    EXPECT_EQ(arg.initialization, hipblaslt_initialization::hpl);

    p.mxScaleFormat = 1;
    arg             = made(solution);
    EXPECT_EQ(arg.scaleA, hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT);
    EXPECT_EQ(arg.scaleB, hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT);

    p.mxScaleFormat = 2;
    EXPECT_EQ(unsupported(solution), "MX scale layout the client cannot express");
    SolutionCaseOptions gfx1250;
    gfx1250.gfx1250 = true;
    EXPECT_EQ(made(solution, gfx1250).scaleA, hipblaslt_scaling_format::Block_32_UE8M0);
}

// A grouped solution gets two equal members; mixed A/B storage goes through
// the dimension-based API with the compute inputs folded into the compute type.
TEST(HostUnitSolutionIndexCase, smoke_GroupedMixedStorage)
{
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    auto& p        = solution.problemType;
    p.groupedGemm  = true;
    p.aType        = rocisa::DataType::Float8;
    auto const arg = made(solution);
    EXPECT_EQ(arg.grouped_gemm, 2);
    EXPECT_TRUE(arg.use_ext_setproblem);
    EXPECT_EQ(arg.compute_type, HIPBLAS_COMPUTE_32F_FAST_16F);
    EXPECT_EQ(arg.M[1], arg.M[0]);
    EXPECT_EQ(arg.K[1], arg.K[0]);
    EXPECT_EQ(arg.lda[1], arg.lda[0]);
}

// A solution that always writes E runs with ReLU and a bias, unless its own
// activation whitelist excludes ReLU.
TEST(HostUnitSolutionIndexCase, smoke_RequiredEpilogue)
{
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    auto& p                 = solution.problemType;
    p.useE                  = true;
    p.useBias               = 1;
    p.biasSrcWhiteList      = {3};
    p.biasDataTypeWhiteList = {rocisa::DataType::Half, rocisa::DataType::Float};
    p.eType                 = rocisa::DataType::Half;
    auto const arg          = made(solution);
    EXPECT_TRUE(arg.use_e);
    EXPECT_EQ(arg.activation_type, hipblaslt_activation_type::relu);
    EXPECT_EQ(arg.bias_type, HIP_R_32F);
    EXPECT_EQ(arg.bias_source, hipblaslt_bias_source::d);

    auto allowed   = std::make_shared<Predicates::Contraction::ActivationEnumWhiteList>();
    allowed->value = {TensileLite::ActivationType::Gelu};
    requireSizes(solution, {allowed});
    EXPECT_EQ(unsupported(solution), "required ReLU activation not allowed");
}

// A CU-count predicate is what makes a solution need a secondary device.
TEST(HostUnitSolutionIndexCase, smoke_RequiresCUCount)
{
    using AMDGPU = TensileLite::AMDGPU;
    using GPU    = std::shared_ptr<Predicates::Predicate<AMDGPU>>;
    AMDGPU const                     device(AMDGPU::Processor::gfx942, 304, "MI300X");
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    auto pin = [&](std::vector<GPU> checks) {
        solution.hardwarePredicate
            = std::make_shared<Predicates::IsSubclass<TensileLite::Hardware, AMDGPU>>(
                std::make_shared<Predicates::And<AMDGPU>>(checks));
    };
    GPU const processor = std::make_shared<Predicates::GPU::ProcessorEqual>(device.processor);
    pin({processor});
    EXPECT_FALSE(requiresCUCount(solution, device));
    pin({processor, std::make_shared<Predicates::GPU::CUCountEqual>(304)});
    EXPECT_TRUE(requiresCUCount(solution, device));
    pin({processor, std::make_shared<Predicates::GPU::CUCountEqual>(228)});
    EXPECT_FALSE(requiresCUCount(solution, device));
}
