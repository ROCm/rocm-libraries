// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include "mocks/MockPlan.hpp"

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"

using namespace hip_kernel_plugin;

TEST(TestHipKernelHipdnnEnginePluginExecutionContext, SetAndGetPlan)
{
    HipdnnEnginePluginExecutionContext ctx;

    auto mockPlan = std::make_unique<hip_kernel_plugin::MockPlan>();
    auto* planPtr = mockPlan.get();
    ctx.setPlan(std::move(mockPlan));

    hip_kernel_plugin::IPlan& planRef = ctx.plan();

    EXPECT_EQ(&planRef, planPtr);
}

TEST(TestHipKernelHipdnnEnginePluginExecutionContext, HasValidPlan)
{
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_FALSE(ctx.hasValidPlan());

    auto mockPlan = std::make_unique<hip_kernel_plugin::MockPlan>();
    ctx.setPlan(std::move(mockPlan));

    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST(TestHipKernelHipdnnEnginePluginExecutionContext, GetPlanThrowsIfNotSet)
{
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(ctx.plan(), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipKernelHipdnnEnginePluginExecutionContext, GetWorkspaceSize)
{
    HipdnnEnginePluginExecutionContext ctx;

    auto mockPlan = std::make_unique<hip_kernel_plugin::MockPlan>();
    EXPECT_CALL(*mockPlan, getWorkspaceSize(::testing::_)).WillOnce(testing::Return(42));
    ctx.setPlan(std::move(mockPlan));

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(ctx.plan().getWorkspaceSize(dummyHandle), 42);
}
