// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "MiopenPluginDefines.hpp"
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>

using namespace miopen_plugin;

TEST(TestMiopenPluginPublic, HipdnnPluginGetVersionSuccess)
{
    const char* version;
    const auto status = hipdnnPluginGetVersion(&version);

    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, HIPDNN_PLUGIN_VERSION);
}

TEST(TestMiopenPluginPublic, HipdnnPluginGetVersionNullPtr)
{
    const auto status = hipdnnPluginGetVersion(nullptr);
    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenPluginPublic, HipdnnPluginGetApiVersionSuccess)
{
    const char* version;
    const auto status = hipdnnPluginGetApiVersion(&version);

    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, HIPDNN_PLUGIN_API_VERSION);
}

TEST(TestMiopenPluginPublic, HipdnnPluginGetApiVersionNullPtr)
{
    const auto status = hipdnnPluginGetApiVersion(nullptr);
    EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestMiopenPluginPublic, HipdnnPluginSetLogLevelSuccess)
{
    const auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_INFO);
    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}

TEST(TestMiopenPluginPublic, HipdnnPluginSetLogLevelFiltersLowerSeverity)
{
    const auto status = hipdnnPluginSetLogLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}
