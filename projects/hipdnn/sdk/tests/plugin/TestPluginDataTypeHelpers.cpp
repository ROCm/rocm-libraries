// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <sstream>

#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/PluginDataTypeHelpers.hpp>

TEST(TestPluginDataTypeHelpers, ToStringPluginStatus)
{
    EXPECT_STREQ(toString(HIPDNN_PLUGIN_STATUS_SUCCESS), "HIPDNN_PLUGIN_STATUS_SUCCESS");
    EXPECT_STREQ(toString(HIPDNN_PLUGIN_STATUS_BAD_PARAM), "HIPDNN_PLUGIN_STATUS_BAD_PARAM");
    EXPECT_STREQ(toString(HIPDNN_PLUGIN_STATUS_INVALID_VALUE),
                 "HIPDNN_PLUGIN_STATUS_INVALID_VALUE");
    EXPECT_STREQ(toString(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR),
                 "HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR");
    EXPECT_STREQ(toString(HIPDNN_PLUGIN_STATUS_ALLOC_FAILED), "HIPDNN_PLUGIN_STATUS_ALLOC_FAILED");

    EXPECT_STREQ(toString(static_cast<hipdnnPluginStatus_t>(999)), "HIPDNN_PLUGIN_STATUS_UNKNOWN");
}

TEST(TestPluginDataTypeHelpers, ToStringPluginType)
{
    EXPECT_STREQ(toString(HIPDNN_PLUGIN_TYPE_UNSPECIFIED), "HIPDNN_PLUGIN_TYPE_UNSPECIFIED");
    EXPECT_STREQ(toString(HIPDNN_PLUGIN_TYPE_ENGINE), "HIPDNN_PLUGIN_TYPE_ENGINE");
    EXPECT_STREQ(toString(static_cast<hipdnnPluginType_t>(999)), "HIPDNN_PLUGIN_TYPE_UNKNOWN");
}

TEST(TestPluginDataTypeHelpers, OstreamOperatorPluginStatus)
{
    std::ostringstream oss;
    oss << HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    EXPECT_EQ(oss.str(), "HIPDNN_PLUGIN_STATUS_BAD_PARAM");
}

TEST(TestPluginDataTypeHelpers, OstreamOperatorPluginType)
{
    std::ostringstream oss;
    oss << HIPDNN_PLUGIN_TYPE_ENGINE;
    EXPECT_EQ(oss.str(), "HIPDNN_PLUGIN_TYPE_ENGINE");
}
