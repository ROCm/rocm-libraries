// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_plugin_sdk/PluginApi.h>

TEST(TestHipKernelPluginPublic, HipdnnPluginSetLogLevelSuccess)
{
    auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}

TEST(TestHipKernelPluginPublic, HipdnnPluginSetLogLevelFiltersLowerSeverity)
{
    auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}
