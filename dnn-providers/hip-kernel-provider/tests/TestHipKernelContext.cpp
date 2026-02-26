// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "HipKernelContext.hpp"

TEST(TestHipKernelContext, ConstructsSuccessfully)
{
    HipKernelContext context;
}

TEST(TestHipKernelContext, HasNoPlanByDefault)
{
    HipKernelContext context;

    EXPECT_FALSE(context.hasValidPlan());
}

TEST(TestHipKernelContext, GetPlanThrowsWhenNoPlan)
{
    HipKernelContext context;

    EXPECT_THROW(context.plan(), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipKernelContext, ExecutionSettingsAccessible)
{
    HipKernelContext context;

    const auto& settings = context.executionSettings();
    (void)settings;
}
