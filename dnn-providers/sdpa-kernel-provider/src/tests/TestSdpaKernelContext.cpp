// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "SdpaKernelContext.hpp"

TEST(TestSdpaKernelContext, ConstructsSuccessfully)
{
    SdpaKernelContext context;
}

TEST(TestSdpaKernelContext, HasNoPlanByDefault)
{
    SdpaKernelContext context;

    EXPECT_FALSE(context.hasValidPlan());
}

TEST(TestSdpaKernelContext, GetPlanThrowsWhenNoPlan)
{
    SdpaKernelContext context;

    EXPECT_THROW(context.plan(), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestSdpaKernelContext, ExecutionSettingsAccessible)
{
    SdpaKernelContext context;

    const auto& settings = context.executionSettings();
    (void)settings;
}
