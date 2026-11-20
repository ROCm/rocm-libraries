// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the client case built from a solution's own record, on
// synthetic solutions.

#include "solution_index_case.hpp"

#include <gtest/gtest.h>

#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/MXScaleFormat.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace
{
    using hipblaslt_client::SolutionCaseOptions;
    using hipblaslt_client::solutionIndexCase;
    using hipblaslt_client::UnsupportedSolution;

    using Problem        = TensileLite::ContractionProblemGemm;
    namespace Predicates = TensileLite::Predicates;

    // A TN HHS solution with a 64x128 macro tile, DepthU 32 and no split.
    void makeHalfSolution(TensileLite::ContractionSolution& solution)
    {
        solution.index        = 7;
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
        auto result = solutionIndexCase(solution, options);
        if(auto const* unsupported = std::get_if<UnsupportedSolution>(&result))
            ADD_FAILURE() << unsupported->reason;
        return std::holds_alternative<Arguments>(result) ? std::get<Arguments>(result)
                                                         : Arguments{};
    }

    std::string unsupported(TensileLite::ContractionSolution const& solution,
                            SolutionCaseOptions const&              options = {})
    {
        auto        result = solutionIndexCase(solution, options);
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
    EXPECT_TRUE(arg.norm_check);
    EXPECT_TRUE(arg.norm_check_assert);
    EXPECT_TRUE(arg.allclose_check);
    EXPECT_EQ(arg.lda[0], 256);
    EXPECT_EQ(arg.stride_a[0], 256 * 64);
    EXPECT_EQ(arg.lde[0], 64);
    EXPECT_EQ(arg.stride_e[0], 64 * 128);
}

TEST(HostUnitSolutionIndexCase, smoke_PackedLayoutPreservesPaddingAndExplicitStrides)
{
    for(char transpose : {'N', 'T', 'C'})
    {
        SCOPED_TRACE(transpose);
        Arguments arg{};
        arg.init();
        arg.transA = arg.transB = transpose;
        arg.grouped_gemm        = 2;
        arg.M[0] = arg.M[1] = 3;
        arg.N[0] = arg.N[1] = 5;
        arg.K[0] = arg.K[1] = 7;
        arg.ldd[0]          = 32;
        arg.stride_b[1]     = 1000;

        arg.raise_to_packed_layout(false);
        EXPECT_EQ(arg.lda[0], transpose == 'N' ? 3 : 7);
        EXPECT_EQ(arg.ldb[1], transpose == 'N' ? 7 : 5);
        EXPECT_EQ(arg.ldc[0], 3);
        EXPECT_EQ(arg.ldd[0], 32);
        EXPECT_EQ(arg.lde[1], 3);
        EXPECT_EQ(arg.stride_a[0], 0);
        EXPECT_EQ(arg.stride_b[1], 1000);

        arg.raise_to_packed_layout(true);
        EXPECT_EQ(arg.stride_a[0], 21);
        EXPECT_EQ(arg.stride_b[0], 35);
        EXPECT_EQ(arg.stride_b[1], 1000);
        EXPECT_EQ(arg.stride_c[0], 15);
        EXPECT_EQ(arg.stride_d[0], 160);
        EXPECT_EQ(arg.stride_e[1], 15);
    }
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
    using F = TensileLite::MXScaleFormat;
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    auto& p = solution.problemType;
    p.aType = p.bType = p.computeInputTypeA = p.computeInputTypeB = rocisa::DataType::Float8;
    p.mxBlockA = p.mxBlockB = 32;
    auto arg                = made(solution);
    EXPECT_EQ(arg.scaleA, hipblaslt_scaling_format::Block_32_UE8M0);
    EXPECT_EQ(arg.initialization, hipblaslt_initialization::hpl);

    p.mxScaleFormat = static_cast<int>(F::HostPreSwizzle);
    arg             = made(solution);
    EXPECT_EQ(arg.scaleA, hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT);
    EXPECT_EQ(arg.scaleB, hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT);

    // FP4 x FP4 accepts the preswizzled layout in every orientation.
    p.transA              = false;
    p.operationIdentifier = "Contraction_l_Ailk_Bljk_Cijk_Dijk";
    EXPECT_EQ(unsupported(solution), "MX scale layout the client cannot express");
    p.aType = p.bType = p.computeInputTypeA = p.computeInputTypeB = rocisa::DataType::Float4;
    EXPECT_EQ(made(solution).scaleA, hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT);
    p.transA              = true;
    p.operationIdentifier = "Contraction_l_Alik_Bljk_Cijk_Dijk";
    p.aType = p.bType = p.computeInputTypeA = p.computeInputTypeB = rocisa::DataType::Float8;

    p.mxScaleFormat = static_cast<int>(F::InMemorySwizzle);
    EXPECT_EQ(made(solution).scaleA, hipblaslt_scaling_format::Block_32_UE8M0);

    p.mxScaleFormat = static_cast<int>(F::HostPreSwizzle128x4);
    p.aType = p.bType = p.computeInputTypeA = p.computeInputTypeB = rocisa::DataType::Float6;
    p.swizzleTensorA = p.swizzleTensorB = true;
    arg                                 = made(solution);
    EXPECT_EQ(arg.scaleA, hipblaslt_scaling_format::Block_32_UE8M0_128_4_EXT);
    EXPECT_EQ(arg.scaleB, hipblaslt_scaling_format::Block_32_UE8M0_128_4_EXT);
    EXPECT_TRUE(arg.swizzle_a);
    EXPECT_TRUE(arg.swizzle_b);

    p.mxBlockFreeA = 2;
    EXPECT_EQ(unsupported(solution), "MX scaling across the free dimension");
    p.mxBlockFreeA = 1;

    // Grouped MX solutions also report scalar input scaling; the block scales stay.
    p.aType = p.bType = p.computeInputTypeA = p.computeInputTypeB = rocisa::DataType::Float8;
    p.swizzleTensorA = p.swizzleTensorB = false;
    p.mxScaleFormat                     = static_cast<int>(F::NoSwizzle);
    p.groupedGemm                       = true;
    p.useScaleAB                        = "Scalar";
    EXPECT_EQ(made(solution).scaleB, hipblaslt_scaling_format::Block_32_UE8M0);
}

TEST(HostUnitSolutionIndexCase, smoke_BatchedSizeConstraints)
{
    namespace C = Predicates::Contraction;
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    requireSizes(solution,
                 {std::make_shared<C::SizeInRange>(0, C::Range{70, 100}),
                  std::make_shared<C::Free0SizeMultiple>(0, 16),
                  std::make_shared<C::SizeMultiple>(0, 24),
                  std::make_shared<C::BatchSizeEqual>(0, 3),
                  std::make_shared<C::BoundSizeMultiple>(-1, 96)});
    auto const arg = made(solution);
    EXPECT_EQ(arg.M[0], 96);
    EXPECT_EQ(arg.N[0], 128);
    EXPECT_EQ(arg.K[0], 288);
    EXPECT_EQ(arg.batch_count, 3);
    EXPECT_EQ(arg.stride_a[0], 96 * 288);

    requireSizes(
        solution,
        {std::make_shared<C::BatchSizeMultiple>(0, 4), std::make_shared<C::SizeLessThan>(3, 200)});
    EXPECT_EQ(made(solution).batch_count, 4);
    EXPECT_EQ(made(solution).K[0], 192);
}

TEST(HostUnitSolutionIndexCase, smoke_InvalidSizesAreUnsupported)
{
    namespace C = Predicates::Contraction;
    TensileLite::ContractionSolution solution;
    makeHalfSolution(solution);
    auto const maxSize = std::numeric_limits<int64_t>::max();
    for(auto check : std::vector<std::shared_ptr<Predicates::Predicate<Problem>>>{
            std::make_shared<C::SizeEqual>(4, 1),
            std::make_shared<C::SizeLessThan>(0, 0),
            std::make_shared<C::SizeGreaterThan>(0, maxSize),
            std::make_shared<C::SizeMultiple>(3, 0),
            std::make_shared<C::SizeMultiple>(3, maxSize),
            std::make_shared<C::BoundSizeMultiple>(1, 32),
            std::make_shared<C::SizeEqual>(0, maxSize),
            std::make_shared<C::BatchSizeEqual>(0, maxSize)})
    {
        SCOPED_TRACE(check->type());
        requireSizes(solution, {check});
        EXPECT_FALSE(unsupported(solution).empty());
    }

    requireSizes(solution, {});
    SolutionCaseOptions options;
    for(int64_t blocks : {int64_t{0}, int64_t{-1}, maxSize})
    {
        options.reductionBlocks = blocks;
        EXPECT_FALSE(unsupported(solution, options).empty());
    }
    solution.sizeMapping.depthU = 0;
    EXPECT_FALSE(unsupported(solution).empty());
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

// A solution that always writes E runs with ReLU, and with a bias when it
// supports one, unless its own activation whitelist excludes ReLU. Without
// bias support a gradient solution differentiates E alone (DReLU).
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

    p.useBias        = 0;
    p.useGradient    = true;
    auto const dRelu = made(solution);
    EXPECT_TRUE(dRelu.gradient);
    EXPECT_TRUE(dRelu.use_e);
    EXPECT_FALSE(dRelu.bias_vector);

    auto allowed   = std::make_shared<Predicates::Contraction::ActivationEnumWhiteList>();
    allowed->value = {TensileLite::ActivationType::Gelu};
    requireSizes(solution, {allowed});
    EXPECT_EQ(unsupported(solution), "required ReLU activation not allowed");
}
