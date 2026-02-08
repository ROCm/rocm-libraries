// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_plugin_sdk/PluginApi.h"
#include <HipKernelPlugin.hpp>
#include <array>
#include <gtest/gtest.h>
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

class TestHipKernelPluginApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipdnnPluginSetLoggingCallbackImpl(testLoggingCallback),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
    }
};
} // namespace

TEST_F(TestHipKernelPluginApi, GetNameSuccess)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetNameImpl(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "hip_kernel_plugin");
}

TEST_F(TestHipKernelPluginApi, GetNameNullptr)
{
    EXPECT_EQ(hipdnnPluginGetNameImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestHipKernelPluginApi, GetVersionSuccess)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersionImpl(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, "1.0.0");
}

TEST_F(TestHipKernelPluginApi, GetVersionNullptr)
{
    EXPECT_EQ(hipdnnPluginGetVersionImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestHipKernelPluginApi, GetTypeSuccess)
{
    hipdnnPluginType_t type;
    EXPECT_EQ(hipdnnPluginGetTypeImpl(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_ENGINE);
}

TEST_F(TestHipKernelPluginApi, GetTypeNullptr)
{
    EXPECT_EQ(hipdnnPluginGetTypeImpl(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestHipKernelPluginApi, GetLastErrorStringSuccess)
{
    const char* errorStr = nullptr;
    hipdnnPluginGetLastErrorStringImpl(&errorStr);
    ASSERT_NE(errorStr, nullptr);
    EXPECT_GE(strlen(errorStr), 0u);
}

TEST_F(TestHipKernelPluginApi, GetLastErrorStringNullptr)
{
    EXPECT_NO_THROW(hipdnnPluginGetLastErrorStringImpl(nullptr));
}
