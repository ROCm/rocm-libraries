// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <string_view>

#include "version.h"
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

TEST(TestPluginPublic, HipdnnPluginSetLogLevelSuccess)
{
    auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}

TEST(TestPluginPublic, HipdnnPluginSetLogLevelFiltersLowerSeverity)
{
    auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}

TEST(TestPluginPublic, HipdnnPluginGetVersionSuccess)
{
    const char* version;
    auto status = hipdnnPluginGetVersion(&version);

    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, HIP_KERNEL_PROVIDER_VERSION_STRING);
}

TEST(TestPluginPublic, HipdnnPluginGetVersionNullPtr)
{
    auto status = hipdnnPluginGetVersion(nullptr);
    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestPluginPublic, HipdnnPluginGetApiVersionSuccess)
{
    const char* version;
    auto status = hipdnnPluginGetApiVersion(&version);

    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(std::string_view{version}, hipdnn_plugin_sdk::K_MAX_SUPPORTED_API_VERSION);
}

TEST(TestPluginPublic, HipdnnPluginGetApiVersionNullPtr)
{
    auto status = hipdnnPluginGetApiVersion(nullptr);
    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}
