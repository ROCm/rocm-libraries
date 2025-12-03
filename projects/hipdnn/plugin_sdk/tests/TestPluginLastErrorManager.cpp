// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

using namespace hipdnn_plugin;

// Define the thread_local static member
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

TEST(TestPluginLastErrorManager, SetLastErrorWithCharPointer)
{
    auto status
        = PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "test error");
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "test error");
}

TEST(TestPluginLastErrorManager, SetLastErrorWithString)
{
    std::string errorMsg = "string error";
    auto status
        = PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, errorMsg);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "string error");
}

TEST(TestPluginLastErrorManager, SetLastErrorDoesNotSetOnSuccess)
{
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "initial error");
    auto status
        = PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_SUCCESS, "success message");
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    // Error should still be the initial error, not changed
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "initial error");
}

TEST(TestPluginLastErrorManager, GetLastErrorReturnsLastSetError)
{
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "first error");
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_ALLOC_FAILED, "second error");
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "second error");
}
