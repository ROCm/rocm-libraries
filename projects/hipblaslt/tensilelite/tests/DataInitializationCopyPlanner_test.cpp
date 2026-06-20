// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime.h>

#include "DataInitializationCopyPlanner.hpp"
#include "DataInitializationTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;
using namespace TensileLite::Client::detail;

namespace
{
    using TensileLite::testing::makeBatchedProblem;

    std::vector<TensorCopyView> makeViews(ContractionProblemGemm const& problem,
                                          std::vector<bool> const&      pristineMask = {})
    {
        std::vector<TensorCopyView> views(problem.tensors().size());
        for(size_t i = 0; i < views.size(); ++i)
        {
            views[i].hasPristine = pristineMask.empty() || i >= pristineMask.size()
                                   || pristineMask.at(i);
            views[i].maxElements = 1000 + i;
            views[i].groupedOffsets = {i, i + 1, i + 2};
        }
        return views;
    }

    ptrdiff_t expectedGuardBackPadding(ContractionProblemGemm const& problem, size_t index)
    {
        InputLayoutPolicy const policy;
        auto const              swizzlePlan = policy.planTensorSwizzle(problem, index);
        if(!swizzlePlan.enabled)
            return -1;

        return static_cast<ptrdiff_t>(swizzlePlan.allocatedElements
                                      - problem.tensors().at(index).totalAllocatedElements());
    }
} // namespace

TEST(DataInitializationCopyPlanner, InputCopyPlansAllPristineTensorsWithGpuRoles)
{
    auto problem = makeBatchedProblem(32, 24, 16, 4);
    auto views   = makeViews(problem);

    auto const plan = planTensorCopies(problem,
                                       views,
                                       TensorCopyIntent::InputCopy,
                                       TensorCopyBoundsMode::Disable,
                                       hipMemcpyDeviceToDevice);

    ASSERT_EQ(plan.size(), problem.tensors().size());
    for(size_t i = 0; i < plan.size(); ++i)
    {
        ASSERT_TRUE(plan[i].has_value());
        auto const& instruction = *plan[i];

        EXPECT_EQ(instruction.tensorIndex, i);
        EXPECT_EQ(instruction.operationKind, TensorCopyOperationKind::Plain);
        EXPECT_EQ(instruction.copyKind, hipMemcpyDeviceToDevice);
        EXPECT_EQ(instruction.dstRole, TensorBufferRole::GpuCurrent);
        EXPECT_EQ(instruction.srcRole, TensorBufferRole::GpuValid);
        EXPECT_FALSE(instruction.badRole.has_value());
        EXPECT_EQ(instruction.batchRole, TensorBatchRole::GpuCurrent);
        EXPECT_FALSE(instruction.gpuTargetSlot.has_value());
        EXPECT_EQ(instruction.maxElements, views[i].maxElements);
        EXPECT_EQ(instruction.groupedOffsets, views[i].groupedOffsets);
        EXPECT_EQ(instruction.customPadding, -1);
    }
}

TEST(DataInitializationCopyPlanner, PlanTensorCopiesRejectsMismatchedViewCount)
{
    auto problem = makeBatchedProblem(32, 24, 16, 4);
    auto views   = makeViews(problem);
    views.pop_back();

    EXPECT_THROW(
        (planTensorCopies(problem,
                          views,
                          TensorCopyIntent::InputCopy,
                          TensorCopyBoundsMode::Disable,
                          hipMemcpyDeviceToDevice)),
        std::runtime_error);
}

TEST(DataInitializationCopyPlanner, InputCopySkipsViewsWithoutPristineData)
{
    auto problem = makeBatchedProblem(32, 24, 16, 4);
    auto views   = makeViews(problem, {true, false, true, true});

    auto const plan = planTensorCopies(problem,
                                       views,
                                       TensorCopyIntent::InputCopy,
                                       TensorCopyBoundsMode::Disable,
                                       hipMemcpyHostToHost);

    ASSERT_EQ(plan.size(), problem.tensors().size());
    EXPECT_TRUE(plan[0].has_value());
    EXPECT_FALSE(plan[1].has_value());
    EXPECT_TRUE(plan[2].has_value());
    EXPECT_TRUE(plan[3].has_value());
}

TEST(DataInitializationCopyPlanner, InputCopyNaNUsesCpuBadForHostCopies)
{
    auto problem = makeBatchedProblem(32, 24, 16, 4);
    auto views   = makeViews(problem);

    auto const plan = planTensorCopies(problem,
                                       views,
                                       TensorCopyIntent::InputCopy,
                                       TensorCopyBoundsMode::NaN,
                                       hipMemcpyHostToHost);

    for(size_t i = 0; i < plan.size(); ++i)
    {
        ASSERT_TRUE(plan[i].has_value());
        auto const& instruction = *plan[i];

        EXPECT_EQ(instruction.operationKind, TensorCopyOperationKind::BadBounds);
        EXPECT_EQ(instruction.dstRole, TensorBufferRole::CpuCurrent);
        EXPECT_EQ(instruction.srcRole, TensorBufferRole::CpuValid);
        ASSERT_TRUE(instruction.badRole.has_value());
        EXPECT_EQ(*instruction.badRole, TensorBufferRole::CpuBad);
        EXPECT_EQ(instruction.batchRole, TensorBatchRole::CpuCurrent);
        EXPECT_FALSE(instruction.gpuTargetSlot.has_value());
    }
}

TEST(DataInitializationCopyPlanner, InputCopyNaNUsesGpuBadForDeviceCopies)
{
    auto problem = makeBatchedProblem(32, 24, 16, 4);
    auto views   = makeViews(problem);

    auto const plan = planTensorCopies(problem,
                                       views,
                                       TensorCopyIntent::InputCopy,
                                       TensorCopyBoundsMode::NaN,
                                       hipMemcpyDeviceToDevice);

    for(size_t i = 0; i < plan.size(); ++i)
    {
        ASSERT_TRUE(plan[i].has_value());
        auto const& instruction = *plan[i];

        EXPECT_EQ(instruction.operationKind, TensorCopyOperationKind::BadBounds);
        EXPECT_EQ(instruction.dstRole, TensorBufferRole::GpuCurrent);
        EXPECT_EQ(instruction.srcRole, TensorBufferRole::GpuValid);
        ASSERT_TRUE(instruction.badRole.has_value());
        EXPECT_EQ(*instruction.badRole, TensorBufferRole::GpuBad);
        EXPECT_EQ(instruction.batchRole, TensorBatchRole::GpuCurrent);
        EXPECT_FALSE(instruction.gpuTargetSlot.has_value());
    }
}

TEST(DataInitializationCopyPlanner, InputCopyWithGpuSlotUsesSlotRoles)
{
    auto problem = makeBatchedProblem(32, 24, 16, 4);
    auto views   = makeViews(problem);

    auto const plan = planTensorCopies(problem,
                                       views,
                                       TensorCopyIntent::InputCopy,
                                       TensorCopyBoundsMode::Disable,
                                       hipMemcpyHostToDevice,
                                       2);

    for(size_t i = 0; i < plan.size(); ++i)
    {
        ASSERT_TRUE(plan[i].has_value());
        auto const& instruction = *plan[i];

        EXPECT_EQ(instruction.dstRole, TensorBufferRole::GpuSlotData);
        EXPECT_EQ(instruction.srcRole, TensorBufferRole::CpuValid);
        EXPECT_EQ(instruction.batchRole, TensorBatchRole::GpuSlot);
        ASSERT_TRUE(instruction.gpuTargetSlot.has_value());
        EXPECT_EQ(*instruction.gpuTargetSlot, 2u);
        EXPECT_FALSE(instruction.badRole.has_value());
    }
}

TEST(DataInitializationCopyPlanner, InputGuardPageBackUsesGuardBackAndCustomPadding)
{
    auto problem = makeBatchedProblem(17, 19, 23, 4);
    problem.setSwizzleTensorA(true);
    problem.setSwizzleTensorB(true);
    auto views   = makeViews(problem);

    auto const plan = planTensorCopies(problem,
                                       views,
                                       TensorCopyIntent::InputCopy,
                                       TensorCopyBoundsMode::GuardPageBack,
                                       hipMemcpyDeviceToDevice);

    ASSERT_EQ(plan.size(), problem.tensors().size());
    for(size_t i = 0; i < plan.size(); ++i)
    {
        ASSERT_TRUE(plan[i].has_value());
        auto const& instruction = *plan[i];

        EXPECT_EQ(instruction.operationKind, TensorCopyOperationKind::GuardBack);
        EXPECT_EQ(instruction.copyKind, hipMemcpyDeviceToDevice);
        EXPECT_EQ(instruction.batchRole, TensorBatchRole::GpuCurrent);

        if(i == ContractionProblemGemm::TENSOR::A || i == ContractionProblemGemm::TENSOR::B)
            EXPECT_EQ(instruction.customPadding, expectedGuardBackPadding(problem, i));
        else
            EXPECT_EQ(instruction.customPadding, -1);
    }
}

TEST(DataInitializationCopyPlanner, OutputResetSkipsNonOutputsAndUsesPlainGuardBack)
{
    auto problem = makeBatchedProblem(32, 24, 16, 4);
    auto views   = makeViews(problem);

    auto const plan = planTensorCopies(problem,
                                       views,
                                       TensorCopyIntent::OutputReset,
                                       TensorCopyBoundsMode::GuardPageBack,
                                       hipMemcpyDeviceToDevice);

    ASSERT_EQ(plan.size(), problem.tensors().size());
    for(size_t i = 0; i < plan.size(); ++i)
    {
        auto const& desc = problem.tensors().at(i);
        if(!desc.isOutput())
        {
            EXPECT_FALSE(plan[i].has_value());
            continue;
        }

        ASSERT_TRUE(plan[i].has_value());
        auto const& instruction = *plan[i];
        EXPECT_EQ(instruction.operationKind, TensorCopyOperationKind::Plain);
        EXPECT_EQ(instruction.dstRole, TensorBufferRole::GpuCurrent);
        EXPECT_EQ(instruction.srcRole, TensorBufferRole::GpuValid);
        EXPECT_EQ(instruction.batchRole, TensorBatchRole::GpuCurrent);
        EXPECT_FALSE(instruction.badRole.has_value());
        EXPECT_FALSE(instruction.gpuTargetSlot.has_value());
    }
}

TEST(DataInitializationCopyPlanner, OutputResetWithGpuSlotUsesSlotRoles)
{
    auto problem = makeBatchedProblem(32, 24, 16, 4);
    auto views   = makeViews(problem);

    auto const plan = planTensorCopies(problem,
                                       views,
                                       TensorCopyIntent::OutputReset,
                                       TensorCopyBoundsMode::Disable,
                                       hipMemcpyDeviceToDevice,
                                       1);

    for(size_t i = 0; i < plan.size(); ++i)
    {
        auto const& desc = problem.tensors().at(i);
        if(!desc.isOutput())
        {
            EXPECT_FALSE(plan[i].has_value());
            continue;
        }

        ASSERT_TRUE(plan[i].has_value());
        auto const& instruction = *plan[i];
        EXPECT_EQ(instruction.dstRole, TensorBufferRole::GpuSlotData);
        EXPECT_EQ(instruction.srcRole, TensorBufferRole::GpuValid);
        EXPECT_EQ(instruction.batchRole, TensorBatchRole::GpuSlot);
        ASSERT_TRUE(instruction.gpuTargetSlot.has_value());
        EXPECT_EQ(*instruction.gpuTargetSlot, 1u);
        EXPECT_EQ(instruction.operationKind, TensorCopyOperationKind::Plain);
    }
}
