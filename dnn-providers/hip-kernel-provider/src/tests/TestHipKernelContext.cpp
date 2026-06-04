// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "HipKernelContext.hpp"

TEST(TestHipKernelContext, ConstructsSuccessfully)
{
    const HipKernelContext context;
}

TEST(TestHipKernelContext, HasNoPlanByDefault)
{
    const HipKernelContext context;

    EXPECT_FALSE(context.hasValidPlan());
}

TEST(TestHipKernelContext, GetPlanThrowsWhenNoPlan)
{
    const HipKernelContext context;

    EXPECT_THROW(context.plan(), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipKernelContext, ExecutionSettingsAccessible)
{
    const HipKernelContext context;

    const auto& settings = context.executionSettings();
    (void)settings;
}
