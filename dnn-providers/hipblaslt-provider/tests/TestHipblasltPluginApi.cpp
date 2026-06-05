// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_plugin_sdk/PluginApi.h"
#include "version.h"
#include <HipblasltPlugin.hpp>
#include <array>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <iostream>

namespace
{
void testLoggingCallback(hipdnnSeverity_t severity, const char* msg)
{
    (void)severity;
    // std::cout << msg << "\n"; // uncomment to see formatted log messages during tests.
    // It does not use the true callback yet since the plugin is not yet loaded.
    (void)msg;
}

class TestHipblasltPluginApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipdnnPluginSetLoggingCallbackImpl(testLoggingCallback),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
    }
};
} // namespace

TEST_F(TestHipblasltPluginApi, GetNameSuccess)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetNameImpl(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "hipblaslt_plugin");
}

TEST_F(TestHipblasltPluginApi, GetNameNullptr)
{
    EXPECT_EQ(hipdnnPluginGetNameImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestHipblasltPluginApi, GetVersionSuccess)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersionImpl(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, HIPBLASLT_PROVIDER_VERSION_STRING);
}

TEST_F(TestHipblasltPluginApi, GetVersionNullptr)
{
    EXPECT_EQ(hipdnnPluginGetVersionImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestHipblasltPluginApi, GetTypeSuccess)
{
    hipdnnPluginType_t type;
    EXPECT_EQ(hipdnnPluginGetTypeImpl(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_ENGINE);
}

TEST_F(TestHipblasltPluginApi, GetTypeNullptr)
{
    EXPECT_EQ(hipdnnPluginGetTypeImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestHipblasltPluginApi, GetLastErrorStringSuccess)
{
    const char* errorStr = nullptr;
    hipdnnPluginGetLastErrorStringImpl(&errorStr);
    ASSERT_NE(errorStr, nullptr);
    EXPECT_GE(strlen(errorStr), 0u);
}

TEST_F(TestHipblasltPluginApi, GetLastErrorStringNullptr)
{
    EXPECT_NO_THROW(hipdnnPluginGetLastErrorStringImpl(nullptr));
}

TEST_F(TestHipblasltPluginApi, SetLogLevelSuccess)
{
    EXPECT_EQ(hipdnnPluginSetLogLevelImpl(HIPDNN_SEV_INFO), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}

TEST_F(TestHipblasltPluginApi, SetLogLevelFiltersLowerSeverity)
{
    ASSERT_EQ(hipdnnPluginSetLogLevelImpl(HIPDNN_SEV_WARN), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
}
