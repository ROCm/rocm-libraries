// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IExecutionContext.hpp>
#include <hipdnn_test_sdk/utilities/MockPlan.hpp>

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_test_sdk::utilities;
using ::testing::NiceMock;
using ::testing::Return;

TEST(TestExecutionContextBase, InitiallyHasNoValidPlan)
{
    ExecutionContextBase context;
    EXPECT_FALSE(context.hasValidPlan());
}

TEST(TestExecutionContextBase, HasValidPlanAfterSetPlan)
{
    ExecutionContextBase context;
    context.setPlan(std::make_unique<NiceMock<MockPlan>>());
    EXPECT_TRUE(context.hasValidPlan());
}

TEST(TestExecutionContextBase, GetPlanThrowsWhenNoPlanSet)
{
    ExecutionContextBase context;
    EXPECT_THROW(context.getPlan(), HipdnnPluginException);
}

TEST(TestExecutionContextBase, GetPlanReturnsSetPlan)
{
    ExecutionContextBase context;
    const size_t expectedWorkspaceSize = 2048;

    auto mockPlan = std::make_unique<NiceMock<MockPlan>>();
    ON_CALL(*mockPlan, getWorkspaceSize(testing::_)).WillByDefault(Return(expectedWorkspaceSize));
    context.setPlan(std::move(mockPlan));

    auto& plan = context.getPlan();
    EXPECT_EQ(plan.getWorkspaceSize(nullptr), expectedWorkspaceSize);
}

TEST(TestExecutionContextBase, SetPlanReplacesExistingPlan)
{
    ExecutionContextBase context;

    auto mockPlan1 = std::make_unique<NiceMock<MockPlan>>();
    ON_CALL(*mockPlan1, getWorkspaceSize(testing::_)).WillByDefault(Return(1024u));
    context.setPlan(std::move(mockPlan1));
    EXPECT_EQ(context.getPlan().getWorkspaceSize(nullptr), 1024u);

    auto mockPlan2 = std::make_unique<NiceMock<MockPlan>>();
    ON_CALL(*mockPlan2, getWorkspaceSize(testing::_)).WillByDefault(Return(4096u));
    context.setPlan(std::move(mockPlan2));
    EXPECT_EQ(context.getPlan().getWorkspaceSize(nullptr), 4096u);
}
