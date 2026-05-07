// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

#include <hipdnn_backend/version.h>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>

#include <gtest/gtest.h>

using namespace hipdnn_data_sdk::utilities;

TEST(TestVersion, ParsedSuccessfully)
{
    EXPECT_NO_THROW(Version(std::string_view{HIPDNN_BACKEND_VERSION_STRING}));
}

TEST(TestVersion, PositiveVersion)
{
    Version version;
    ASSERT_NO_THROW(version = Version(std::string_view{HIPDNN_BACKEND_VERSION_STRING}));

    EXPECT_GE(version.major, 0);
    EXPECT_GE(version.minor, 0);
    EXPECT_GE(version.patch, 0);
}

// Applicability filter prerequisites for the override-execute entry point
// (RFC 0008 §B.8). These tests anchor the constant
// `K_OVERRIDE_EXECUTE_MIN_API_VERSION` and the strict ordering relied on by
// `computeMinimumPluginApiVersion()` in `EnginePluginResourceManager.cpp`.

TEST(TestVersion, OverrideExecuteMinApiVersionParses)
{
    // Brace-init avoids the most-vexing-parse: `Version(<name>)` would be
    // read as a redeclaration of `<name>` with type `Version`.
    EXPECT_NO_THROW(Version{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION});
    const Version v{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    EXPECT_EQ(v.major, 1);
    EXPECT_EQ(v.minor, 1);
    EXPECT_EQ(v.patch, 0);
}

TEST(TestVersion, BaselineVersionLessThanOverrideExecuteMinApiVersion)
{
    // Override-omitting plugins (reporting "1.0.0") must compare strictly
    // less than `K_OVERRIDE_EXECUTE_MIN_API_VERSION` so the applicability
    // filter excludes them from override-flag graphs (Test #2 prerequisite).
    const Version baseline{std::string_view{"1.0.0"}};
    const Version overrideMin{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    EXPECT_TRUE(baseline < overrideMin);
}

TEST(TestVersion, ZeroVersionLessThanBaseline)
{
    // Pre-baseline plugins (reporting "0.0.0") still parse and compare
    // strictly less than the baseline — ensures the filter rejects them
    // from override-flag graphs as well (Test #4 supporting check).
    const Version zero{std::string_view{"0.0.0"}};
    const Version baseline{std::string_view{"1.0.0"}};
    EXPECT_TRUE(zero < baseline);
}
