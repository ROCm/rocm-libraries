// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include <hipdnn_sdk/plugin/PluginDataTypeHelpers.hpp>
#include <hipdnn_sdk/plugin/PluginLastErrorManager.hpp>

using namespace hipdnn_plugin;

// NOLINTNEXTLINE
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

TEST(TestPluginLastErrorManager, SetAndGetLastErrorString)
{
    const char* msg = "test error message";
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, msg);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), msg);
}

TEST(TestPluginLastErrorManager, SetLastErrorWithStdString)
{
    std::string msg = "std::string error";
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_BAD_PARAM, msg);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), msg.c_str());
}

TEST(TestPluginLastErrorManager, SetLastErrorSuccessDoesNotChangeError)
{
    const char* prevMsg = "previous error";
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, prevMsg);
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_SUCCESS, "should not overwrite");
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), prevMsg);
}
