// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/PluginException.hpp>

using namespace hipdnn_plugin;

TEST(TestPluginException, WhatReturnsMessage)
{
    HipdnnPluginException ex(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "bad param");
    EXPECT_STREQ(ex.what(), "bad param");
    EXPECT_EQ(ex.getMessage(), std::string("bad param"));
    EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestPluginException, ThrowIfNeMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_NE(1, 2, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "not equal"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfEqMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_EQ(1, 1, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "equal"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfTrueMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_TRUE(true, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "true"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfFalseMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_FALSE(false, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "false"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfNullMacroThrows)
{
    void* ptr = nullptr;
    EXPECT_THROW(PLUGIN_THROW_IF_NULL(ptr, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "null"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfLtMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_LT(1, 2, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "less than"),
                 HipdnnPluginException);
}
