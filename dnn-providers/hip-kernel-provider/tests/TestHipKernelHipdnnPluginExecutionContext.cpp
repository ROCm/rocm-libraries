// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include "mocks/MockPlan.hpp"

#include "HipdnnHipKernelContext.hpp"
#include "HipdnnHipKernelHandle.hpp"

using namespace hip_kernel_plugin;

TEST(TestHipKernelHipdnnHipKernelContext, SetAndGetPlan)
{
    HipdnnHipKernelContext ctx;

    auto mockPlan = std::make_unique<hip_kernel_plugin::MockPlan>();
    auto* planPtr = mockPlan.get();
    ctx.setPlan(std::move(mockPlan));

    hipdnn_plugin_sdk::IPlan<HipdnnHipKernelHandle>& planRef = ctx.plan();

    EXPECT_EQ(&planRef, planPtr);
}

TEST(TestHipKernelHipdnnHipKernelContext, HasValidPlan)
{
    HipdnnHipKernelContext ctx;

    EXPECT_FALSE(ctx.hasValidPlan());

    auto mockPlan = std::make_unique<hip_kernel_plugin::MockPlan>();
    ctx.setPlan(std::move(mockPlan));

    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST(TestHipKernelHipdnnHipKernelContext, GetPlanThrowsIfNotSet)
{
    HipdnnHipKernelContext ctx;

    EXPECT_THROW(ctx.plan(), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipKernelHipdnnHipKernelContext, GetWorkspaceSize)
{
    HipdnnHipKernelContext ctx;

    auto mockPlan = std::make_unique<hip_kernel_plugin::MockPlan>();
    EXPECT_CALL(*mockPlan, getWorkspaceSize(::testing::_)).WillOnce(testing::Return(42));
    ctx.setPlan(std::move(mockPlan));

    HipdnnHipKernelHandle dummyHandle;
    EXPECT_EQ(ctx.plan().getWorkspaceSize(dummyHandle), 42);
}
