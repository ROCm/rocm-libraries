// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "mocks/MockPlan.hpp"

using namespace hipblaslt_plugin;

TEST(TestHipblasltHipdnnEnginePluginExecutionContext, SetAndGetPlan)
{
    HipdnnEnginePluginExecutionContext ctx;

    auto mockPlan = std::make_unique<hipblaslt_plugin::MockPlan>();
    auto* planPtr = mockPlan.get();
    ctx.setPlan(std::move(mockPlan));

    hipblaslt_plugin::IPlan const& planRef = ctx.plan();

    EXPECT_EQ(&planRef, planPtr);
}

TEST(TestHipblasltHipdnnEnginePluginExecutionContext, HasValidPlan)
{
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_FALSE(ctx.hasValidPlan());

    auto mockPlan = std::make_unique<hipblaslt_plugin::MockPlan>();
    ctx.setPlan(std::move(mockPlan));

    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST(TestHipblasltHipdnnEnginePluginExecutionContext, GetPlanThrowsIfNotSet)
{
    HipdnnEnginePluginExecutionContext const ctx;

    EXPECT_THROW(ctx.plan(), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipblasltHipdnnEnginePluginExecutionContext, GetWorkspaceSize)
{
    HipdnnEnginePluginExecutionContext ctx;

    auto mockPlan = std::make_unique<hipblaslt_plugin::MockPlan>();
    EXPECT_CALL(*mockPlan, getWorkspaceSize(::testing::_)).WillOnce(testing::Return(42));
    ctx.setPlan(std::move(mockPlan));

    HipdnnEnginePluginHandle const dummyHandle;
    EXPECT_EQ(ctx.plan().getWorkspaceSize(dummyHandle), 42);
}
