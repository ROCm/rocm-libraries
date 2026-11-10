// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

using namespace hipdnn_plugin_sdk;

TEST(TestPluginHelpers, TryCatchReturnsSuccessOnNoException)
{
    const auto lambda = []() {};
    const auto status = tryCatch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestPluginHelpers, TryCatchHandlesHipdnnException)
{
    const auto lambda = []() {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "internal error");
    };
    const auto status = tryCatch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "internal error");
}

TEST(TestPluginHelpers, TryCatchHandlesStdException)
{
    const auto lambda = []() { throw std::runtime_error("std exception"); };
    const auto status = tryCatch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "std exception");
}

TEST(TestPluginHelpers, TryCatchHandlesUnknownException)
{
    const auto lambda = []() { throw 42; };
    const auto status = tryCatch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), "Unknown exception occured");
}
