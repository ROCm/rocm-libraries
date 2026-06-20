// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

#include "DataInitializationTestUtils.hpp"
#include "InputLayoutPolicy.hpp"

using TensileLite::ContractionProblemGemm;
using TensileLite::Client::InputLayoutContext;
using TensileLite::Client::InputLayoutPolicy;
using TensileLite::Client::MxInitializationPlan;
using TensileLite::Client::MxInitializationSidePlan;
using TensileLite::Client::MxPreswizzleState;
using TensileLite::Client::SelectedSolutionLayout;
using TensileLite::Client::TensorSwizzlePlan;
using TensileLite::Client::TensorUploadLayout;

namespace
{
    constexpr size_t A = ContractionProblemGemm::TENSOR::A;
    constexpr size_t B = ContractionProblemGemm::TENSOR::B;
    constexpr size_t MXSA = ContractionProblemGemm::TENSOR::MXSA;
    constexpr size_t MXSB = ContractionProblemGemm::TENSOR::MXSB;

    ContractionProblemGemm makeProblem(rocisa::DataType aType,
                                       rocisa::DataType bType,
                                       bool             swizzleA,
                                       bool             swizzleB,
                                       int              mxBlockA,
                                       int              mxBlockB,
                                       size_t           m = 128,
                                       size_t           n = 128,
                                       size_t           k = 256,
                                       size_t           batch = 1,
                                       bool             transA = true,
                                       bool             transB = false)
    {
        TensileLite::testing::PlainProblemSpec spec;
        spec.m     = m;
        spec.n     = n;
        spec.k     = k;
        spec.batch = batch;
        spec.transA = transA;
        spec.transB = transB;
        spec.aType  = aType;
        spec.bType  = bType;
        spec.cType  = rocisa::DataType::BFloat16;
        spec.dType  = rocisa::DataType::BFloat16;

        auto problem = TensileLite::testing::makePlainProblem(spec);
        problem.setSwizzleTensorA(swizzleA);
        problem.setSwizzleTensorB(swizzleB);
        if(mxBlockA > 0)
            problem.setMXScaleA(rocisa::DataType::E8, mxBlockA);
        if(mxBlockB > 0)
            problem.setMXScaleB(rocisa::DataType::E8, mxBlockB);
        return problem;
    }

    InputLayoutContext makeContext(int userMxScaleFormat,
                                   bool isMxPreswizzleArch,
                                   bool hasSolution,
                                   int mxScaleFormat = -1,
                                   size_t matrixInstructionK = 0)
    {
        InputLayoutContext context;
        context.userMxScaleFormat  = userMxScaleFormat;
        context.isMxPreswizzleArch = isMxPreswizzleArch;
        context.solution.present   = hasSolution;
        context.solution.mxScaleFormat = mxScaleFormat;
        context.solution.matrixInstructionK = matrixInstructionK;
        return context;
    }
} // namespace

TEST(InputLayoutPolicy, PlainProblemHasNoSpecialInputLayout)
{
    auto const problem = makeProblem(rocisa::DataType::Float,
                                     rocisa::DataType::Float,
                                     /*swizzleA=*/false,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/0,
                                     /*mxBlockB=*/0);

    InputLayoutPolicy policy;

    EXPECT_FALSE(policy.hasSpecialInputLayout(problem));
    EXPECT_EQ(policy.plannedAllocatedElements(problem, A), problem.a().totalAllocatedElements());
    EXPECT_EQ(policy.plannedAllocatedElements(problem, B), problem.b().totalAllocatedElements());
}

TEST(InputLayoutPolicy, TensorSwizzlePlanMatchesExistingGeometry)
{
    InputLayoutPolicy policy;

    {
        auto const problem = makeProblem(rocisa::DataType::Float,
                                         rocisa::DataType::Half,
                                         /*swizzleA=*/true,
                                         /*swizzleB=*/false,
                                         /*mxBlockA=*/0,
                                         /*mxBlockB=*/0,
                                         /*m=*/17,
                                         /*n=*/19,
                                         /*k=*/23,
                                         /*batch=*/2);

        auto const plan = policy.planTensorSwizzle(problem, A);
        EXPECT_TRUE(plan.enabled);
        EXPECT_EQ(plan.miMN, 16u);
        EXPECT_EQ(plan.miK, 4u);
        EXPECT_EQ(plan.miKv, 1u);
        EXPECT_EQ(plan.packK, 4u);
        EXPECT_EQ(plan.bitWidth, 32u);
        EXPECT_EQ(plan.unrolledSize, problem.a().sizes()[0]);
        EXPECT_EQ(plan.tiledSize, problem.a().sizes()[1]);
        EXPECT_EQ(plan.paddedShape[0], 32u);
        EXPECT_EQ(plan.paddedShape[1], 32u);
        EXPECT_EQ(plan.allocatedElements, 32u * 32u * 2u);
        EXPECT_EQ(policy.plannedAllocatedElements(problem, A), plan.allocatedElements);
    }

    {
        auto const problem = makeProblem(rocisa::DataType::Half,
                                         rocisa::DataType::Half,
                                         /*swizzleA=*/false,
                                         /*swizzleB=*/true,
                                         /*mxBlockA=*/0,
                                         /*mxBlockB=*/0,
                                         /*m=*/19,
                                         /*n=*/21,
                                         /*k=*/23,
                                         /*batch=*/3,
                                         /*transA=*/true,
                                         /*transB=*/false);

        auto const plan = policy.planTensorSwizzle(problem, B);
        EXPECT_TRUE(plan.enabled);
        EXPECT_EQ(plan.miMN, 16u);
        EXPECT_EQ(plan.miK, 16u);
        EXPECT_EQ(plan.miKv, 4u);
        EXPECT_EQ(plan.packK, 2u);
        EXPECT_EQ(plan.bitWidth, 16u);
        EXPECT_EQ(plan.unrolledSize, problem.b().sizes()[0]);
        EXPECT_EQ(plan.tiledSize, problem.b().sizes()[1]);
        EXPECT_EQ(plan.paddedShape[0], 32u);
        EXPECT_EQ(plan.paddedShape[1], 32u);
        EXPECT_EQ(plan.allocatedElements, 32u * 32u * 3u);
        EXPECT_EQ(policy.plannedAllocatedElements(problem, B), plan.allocatedElements);
    }
}

TEST(InputLayoutPolicy, MxInitializationPlanNoSolutionDoesNotHostPreswizzle)
{
    auto const problem = makeProblem(rocisa::DataType::Float4,
                                     rocisa::DataType::Float4,
                                     /*swizzleA=*/false,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/32,
                                     /*mxBlockB=*/32);

    InputLayoutPolicy policy;
    auto const        context = makeContext(/*userMxScaleFormat=*/1,
                                     /*isMxPreswizzleArch=*/true,
                                     /*hasSolution=*/false);
    auto const        plan    = policy.planMxInitialization(problem, context);

    EXPECT_TRUE(plan.useGenerator);
    EXPECT_TRUE(plan.a.useGenerator);
    EXPECT_TRUE(plan.b.useGenerator);
    EXPECT_TRUE(plan.a.preSwizzle.empty());
    EXPECT_TRUE(plan.a.preTile.empty());
    EXPECT_TRUE(plan.b.preSwizzle.empty());
    EXPECT_TRUE(plan.b.preTile.empty());
    EXPECT_FALSE(plan.a.canHostPreswizzle);
    EXPECT_FALSE(plan.b.canHostPreswizzle);
}

TEST(InputLayoutPolicy, MxInitializationPlanSkipsBroadDefaultSetAndFallbacks)
{
    auto const problem = makeProblem(rocisa::DataType::Float4,
                                     rocisa::DataType::BFloat16,
                                     /*swizzleA=*/false,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/32,
                                     /*mxBlockB=*/32);

    InputLayoutPolicy policy;
    auto const        context = makeContext(/*userMxScaleFormat=*/1,
                                     /*isMxPreswizzleArch=*/false,
                                     /*hasSolution=*/false);
    auto const        plan    = policy.planMxInitialization(problem, context);

    EXPECT_TRUE(plan.useGenerator);
    EXPECT_TRUE(plan.a.useGenerator);
    EXPECT_FALSE(plan.b.useGenerator);
    EXPECT_TRUE(policy.shouldSkipDefaultInitTensor(A, plan));
    EXPECT_TRUE(policy.shouldSkipDefaultInitTensor(B, plan));
    EXPECT_TRUE(policy.shouldSkipDefaultInitTensor(MXSA, plan));
    EXPECT_TRUE(policy.shouldSkipDefaultInitTensor(MXSB, plan));
}

TEST(InputLayoutPolicy, MxInitializationPlanComputesHostPreswizzleWhenEligible)
{
    auto const problem = makeProblem(rocisa::DataType::Float4,
                                     rocisa::DataType::Float4,
                                     /*swizzleA=*/false,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/32,
                                     /*mxBlockB=*/32,
                                     /*m=*/128,
                                     /*n=*/128,
                                     /*k=*/256);

    InputLayoutPolicy policy;
    auto const        context = makeContext(/*userMxScaleFormat=*/1,
                                     /*isMxPreswizzleArch=*/true,
                                     /*hasSolution=*/true,
                                     /*mxScaleFormat=*/1,
                                     /*matrixInstructionK=*/128);
    auto const        plan    = policy.planMxInitialization(problem, context);

    EXPECT_TRUE(plan.useGenerator);
    EXPECT_EQ(plan.a.preSwizzle, (std::vector<size_t>{32u, 8u, 4u}));
    EXPECT_EQ(plan.a.preTile, (std::vector<size_t>{8u, 32u}));
    EXPECT_EQ(plan.b.preSwizzle, (std::vector<size_t>{32u, 8u, 4u}));
    EXPECT_EQ(plan.b.preTile, (std::vector<size_t>{8u, 32u}));
    EXPECT_TRUE(plan.a.canHostPreswizzle);
    EXPECT_TRUE(plan.b.canHostPreswizzle);
}

TEST(InputLayoutPolicy, MxInitializationPlanRejectsIneligibleHostPreswizzle)
{
    auto const problem = makeProblem(rocisa::DataType::Float4,
                                     rocisa::DataType::Float4,
                                     /*swizzleA=*/false,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/32,
                                     /*mxBlockB=*/32,
                                     /*m=*/128,
                                     /*n=*/128,
                                     /*k=*/256);

    InputLayoutPolicy policy;
    auto const        context = makeContext(/*userMxScaleFormat=*/1,
                                     /*isMxPreswizzleArch=*/true,
                                     /*hasSolution=*/true,
                                     /*mxScaleFormat=*/1,
                                     /*matrixInstructionK=*/100);
    auto const        plan    = policy.planMxInitialization(problem, context);

    EXPECT_TRUE(plan.useGenerator);
    EXPECT_TRUE(plan.a.preSwizzle.empty());
    EXPECT_TRUE(plan.a.preTile.empty());
    EXPECT_TRUE(plan.b.preSwizzle.empty());
    EXPECT_TRUE(plan.b.preTile.empty());
    EXPECT_FALSE(plan.a.canHostPreswizzle);
    EXPECT_FALSE(plan.b.canHostPreswizzle);
}

TEST(InputLayoutPolicy, MxNoSwizzleCopiesCanonical)
{
    auto const problem = makeProblem(rocisa::DataType::Float4,
                                     rocisa::DataType::Float4,
                                     /*swizzleA=*/false,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/32,
                                     /*mxBlockB=*/32);

    InputLayoutPolicy policy;
    auto const        context = makeContext(/*userMxScaleFormat=*/1,
                                     /*isMxPreswizzleArch=*/false,
                                     /*hasSolution=*/true,
                                     /*mxScaleFormat=*/0,
                                     /*matrixInstructionK=*/128);

    EXPECT_EQ(policy.planMxTensorUpload(problem, MXSA, context, MxPreswizzleState{}).action,
              TensorUploadLayout::MxCopyCanonical);
    EXPECT_EQ(policy.planMxTensorUpload(problem, MXSB, context, MxPreswizzleState{}).action,
              TensorUploadLayout::MxCopyCanonical);
}

TEST(InputLayoutPolicy, MxPreswizzleArchUsesGpuValidOnlyWhenPreswizzled)
{
    auto const problem = makeProblem(rocisa::DataType::Float4,
                                     rocisa::DataType::Float4,
                                     /*swizzleA=*/false,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/32,
                                     /*mxBlockB=*/32);

    InputLayoutPolicy policy;
    auto const        context = makeContext(/*userMxScaleFormat=*/1,
                                     /*isMxPreswizzleArch=*/true,
                                     /*hasSolution=*/true,
                                     /*mxScaleFormat=*/1,
                                     /*matrixInstructionK=*/128);

    EXPECT_EQ(policy.planMxTensorUpload(problem,
                                        MXSA,
                                        context,
                                        MxPreswizzleState{true, false})
                  .action,
              TensorUploadLayout::MxUsePreswizzledGpuValid);
    EXPECT_EQ(policy.planMxTensorUpload(problem,
                                        MXSA,
                                        context,
                                        MxPreswizzleState{false, false})
                  .action,
              TensorUploadLayout::MxCopyCanonical);
}

TEST(InputLayoutPolicy, MxNonPreswizzleArchUsesKSwizzle)
{
    auto const problem = makeProblem(rocisa::DataType::Float4,
                                     rocisa::DataType::Float4,
                                     /*swizzleA=*/false,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/32,
                                     /*mxBlockB=*/32);

    InputLayoutPolicy policy;
    auto const        context = makeContext(/*userMxScaleFormat=*/1,
                                     /*isMxPreswizzleArch=*/false,
                                     /*hasSolution=*/true,
                                     /*mxScaleFormat=*/1,
                                     /*matrixInstructionK=*/128);
    auto const        aPlan   = policy.planMxTensorUpload(problem,
                                                         MXSA,
                                                         context,
                                                         MxPreswizzleState{});
    auto const        bPlan   = policy.planMxTensorUpload(problem,
                                                         MXSB,
                                                         context,
                                                         MxPreswizzleState{});

    EXPECT_EQ(aPlan.action, TensorUploadLayout::MxKSwizzle);
    EXPECT_EQ(aPlan.dimK, 4u);
    EXPECT_EQ(aPlan.unrollMajor, problem.freeIndicesA()[0].i != 0);
    EXPECT_EQ(bPlan.action, TensorUploadLayout::MxKSwizzle);
    EXPECT_EQ(bPlan.dimK, 4u);
    EXPECT_EQ(bPlan.unrollMajor, problem.freeIndicesB()[0].i != 0);
}

TEST(InputLayoutPolicy, SwizzleGeometryFeedsAllConsumers)
{
    auto const problem = makeProblem(rocisa::DataType::Float,
                                     rocisa::DataType::Float,
                                     /*swizzleA=*/true,
                                     /*swizzleB=*/false,
                                     /*mxBlockA=*/0,
                                     /*mxBlockB=*/0,
                                     /*m=*/17,
                                     /*n=*/19,
                                     /*k=*/23,
                                     /*batch=*/2);

    InputLayoutPolicy policy;
    auto const        swizzlePlan = policy.planTensorSwizzle(problem, A);

    EXPECT_TRUE(swizzlePlan.enabled);
    EXPECT_EQ(swizzlePlan.bitWidth, 32u);
    EXPECT_EQ(swizzlePlan.miMN, 16u);
    EXPECT_EQ(swizzlePlan.miK, 4u);
    EXPECT_EQ(swizzlePlan.miKv, 1u);
    EXPECT_EQ(swizzlePlan.packK, 4u);
    EXPECT_EQ(swizzlePlan.paddedShape, (std::array<size_t, 2>{32u, 32u}));
    EXPECT_EQ(swizzlePlan.allocatedElements, 32u * 32u * 2u);
    EXPECT_EQ(policy.plannedAllocatedElements(problem, A), swizzlePlan.allocatedElements);
    EXPECT_EQ(policy.tensorUploadLayout(problem,
                                        A,
                                        makeContext(/*userMxScaleFormat=*/0,
                                                    /*isMxPreswizzleArch=*/false,
                                                    /*hasSolution=*/false),
                                        MxPreswizzleState{}),
              TensorUploadLayout::TensorSwizzle);
    EXPECT_TRUE(policy.hasSpecialInputLayout(problem));
}
