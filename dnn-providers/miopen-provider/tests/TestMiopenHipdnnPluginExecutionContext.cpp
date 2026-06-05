// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <memory>

#include "mocks/MockPlan.hpp"

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"

using namespace miopen_plugin;

TEST(TestMiopenHipdnnMiopenContext, SetAndGetPlan)
{
    HipdnnMiopenContext ctx;

    auto mockPlan = std::make_unique<miopen_plugin::MockPlan>();
    auto* planPtr = mockPlan.get();
    ctx.setPlan(std::move(mockPlan));

    const hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>& planRef = ctx.plan();

    EXPECT_EQ(&planRef, planPtr);
}

TEST(TestMiopenHipdnnMiopenContext, HasValidPlan)
{
    HipdnnMiopenContext ctx;

    EXPECT_FALSE(ctx.hasValidPlan());

    auto mockPlan = std::make_unique<miopen_plugin::MockPlan>();
    ctx.setPlan(std::move(mockPlan));

    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST(TestMiopenHipdnnMiopenContext, GetPlanThrowsIfNotSet)
{
    const HipdnnMiopenContext ctx;

    EXPECT_THROW(ctx.plan(), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestMiopenHipdnnMiopenContext, GetWorkspaceSize)
{
    SKIP_IF_NO_DEVICES();

    HipdnnMiopenContext ctx;

    auto mockPlan = std::make_unique<miopen_plugin::MockPlan>();
    EXPECT_CALL(*mockPlan, getWorkspaceSize(::testing::_)).WillOnce(testing::Return(42));
    ctx.setPlan(std::move(mockPlan));

    const HipdnnMiopenHandle dummyHandle;
    EXPECT_EQ(ctx.plan().getWorkspaceSize(dummyHandle), 42);
}
