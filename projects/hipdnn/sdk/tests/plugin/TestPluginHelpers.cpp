// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginHelpers.hpp>
#include <hipdnn_sdk/plugin/PluginLastErrorManager.hpp>

using namespace hipdnn_plugin;

TEST(TestPluginHelpers, TryCatchReturnsSuccessOnNoException)
{
    auto lambda = []() {};
    auto status = tryCatch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestPluginHelpers, TryCatchHandlesHipdnnException)
{
    auto lambda = []() {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "internal error");
    };
    auto status = tryCatch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "internal error");
}

TEST(TestPluginHelpers, TryCatchHandlesStdException)
{
    auto lambda = []() { throw std::runtime_error("std exception"); };
    auto status = tryCatch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "std exception");
}

TEST(TestPluginHelpers, TryCatchHandlesUnknownException)
{
    auto lambda = []() { throw 42; };
    auto status = tryCatch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "Unknown exception occured");
}
