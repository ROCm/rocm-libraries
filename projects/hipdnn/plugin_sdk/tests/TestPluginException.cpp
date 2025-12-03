// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hipdnn_plugin;

TEST(TestPluginException, ConstructorSetsStatusAndMessage)
{
    auto exception = HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "test message");
    EXPECT_EQ(exception.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(exception.what(), "test message");
    EXPECT_EQ(exception.getMessage(), "test message");
}

TEST(TestPluginException, ThrowIfNeThrowsWhenNotEqual)
{
    EXPECT_THROW(PLUGIN_THROW_IF_NE(1, 2, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "values not equal"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfNeDoesNotThrowWhenEqual)
{
    EXPECT_NO_THROW(PLUGIN_THROW_IF_NE(1, 1, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "values not equal"));
}

TEST(TestPluginException, ThrowIfEqThrowsWhenEqual)
{
    EXPECT_THROW(PLUGIN_THROW_IF_EQ(1, 1, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "values equal"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfEqDoesNotThrowWhenNotEqual)
{
    EXPECT_NO_THROW(PLUGIN_THROW_IF_EQ(1, 2, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "values equal"));
}

TEST(TestPluginException, ThrowIfTrueThrowsWhenTrue)
{
    EXPECT_THROW(PLUGIN_THROW_IF_TRUE(true, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "condition is true"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfTrueDoesNotThrowWhenFalse)
{
    EXPECT_NO_THROW(
        PLUGIN_THROW_IF_TRUE(false, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "condition is true"));
}

TEST(TestPluginException, ThrowIfFalseThrowsWhenFalse)
{
    EXPECT_THROW(PLUGIN_THROW_IF_FALSE(false, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "condition is false"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfFalseDoesNotThrowWhenTrue)
{
    EXPECT_NO_THROW(
        PLUGIN_THROW_IF_FALSE(true, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "condition is false"));
}

TEST(TestPluginException, ThrowIfNullThrowsWhenNull)
{
    int* ptr = nullptr;
    EXPECT_THROW(PLUGIN_THROW_IF_NULL(ptr, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "pointer is null"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfNullDoesNotThrowWhenNotNull)
{
    int value = 42;
    int* ptr = &value;
    EXPECT_NO_THROW(PLUGIN_THROW_IF_NULL(ptr, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "pointer is null"));
}

TEST(TestPluginException, ThrowIfLtThrowsWhenLessThan)
{
    EXPECT_THROW(PLUGIN_THROW_IF_LT(1, 2, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "value is less than"),
                 HipdnnPluginException);
}

TEST(TestPluginException, ThrowIfLtDoesNotThrowWhenGreaterOrEqual)
{
    EXPECT_NO_THROW(PLUGIN_THROW_IF_LT(2, 1, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "value is less than"));
    EXPECT_NO_THROW(PLUGIN_THROW_IF_LT(1, 1, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "value is less than"));
}
