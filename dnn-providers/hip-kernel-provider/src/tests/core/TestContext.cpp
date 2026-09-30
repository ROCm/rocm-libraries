// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "core/Context.hpp"

TEST(TestContext, ConstructsSuccessfully)
{
    const Context context;
}

TEST(TestContext, HasNoPlanByDefault)
{
    const Context context;

    EXPECT_FALSE(context.hasValidPlan());
}

TEST(TestContext, GetPlanThrowsWhenNoPlan)
{
    const Context context;

    EXPECT_THROW(context.plan(), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestContext, ExecutionSettingsAccessible)
{
    const Context context;

    const auto& settings = context.executionSettings();
    (void)settings;
}
