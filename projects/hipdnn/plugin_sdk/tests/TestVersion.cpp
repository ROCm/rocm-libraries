// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/version.h>

#include <gtest/gtest.h>

using namespace hipdnn_data_sdk::utilities;

TEST(TestVersion, ParsedSuccessfully)
{
    EXPECT_NO_THROW(Version(std::string_view{HIPDNN_PLUGIN_SDK_VERSION_STRING}));
}

TEST(TestVersion, PositiveVersion)
{
    Version version;
    ASSERT_NO_THROW(version = Version(std::string_view{HIPDNN_PLUGIN_SDK_VERSION_STRING}));

    EXPECT_GE(version.major, 0);
    EXPECT_GE(version.minor, 0);
    EXPECT_GE(version.patch, 0);
}

TEST(TestVersion, OverrideExecuteMinApiVersionParses)
{
    EXPECT_NO_THROW(Version{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION});
    const Version v{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    EXPECT_EQ(v.major, 1);
    EXPECT_EQ(v.minor, 1);
    EXPECT_EQ(v.patch, 0);
}

TEST(TestVersion, BaselineVersionLessThanOverrideExecuteMinApiVersion)
{
    const Version baseline{hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE};
    const Version overrideMin{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    EXPECT_TRUE(baseline < overrideMin);
}

TEST(TestVersion, ZeroVersionLessThanBaseline)
{
    const Version zero{std::string_view{"0.0.0"}};
    const Version baseline{hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE};
    EXPECT_TRUE(zero < baseline);
}
